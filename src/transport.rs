//! Transports, written once for every endpoint: whole JSON, SSE (`data:`
//! only or named `event:`), whole and chunked binary, and WebSocket. The
//! [`Answer`] timing and breaks (`delay_first`, `pace`, `cut_after`,
//! `error_after`, `retry_after`) apply here, to all of them alike.

use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::extract::ws::{Message, WebSocket};
use axum::http::StatusCode;
use axum::response::Response;
use bytes::Bytes;
use futures::StreamExt;
use serde_json::Value;

use crate::Answer;
use crate::core::{Ctx, Request, Script, Session, Turn};

/// A body an endpoint answers with, sent under the [`Answer`]'s timing.
#[derive(Debug)]
pub enum Out {
    /// A status and a JSON body. `cut_after` breaks it off halfway.
    Json(u16, Value),
    /// SSE events, each framed (see [`data`] and [`event`]), and the
    /// mid-stream error event `error_after` sends.
    Sse(Vec<String>, String),
    /// A whole binary body of a content type, such as audio.
    Bytes(String, Bytes),
    /// A binary (or any) body of a content type in chunks, each an event
    /// for `pace` and `cut_after`; `error_after` just ends it early.
    Chunks(String, Vec<Bytes>),
    /// The answer does not fit the endpoint: a `500` saying why.
    Misfit(String),
}

/// An SSE event of only a `data:` line.
pub fn data(v: &Value) -> String {
    format!("data: {v}\n\n")
}

/// A named SSE event.
pub fn event(name: &str, v: &Value) -> String {
    format!("event: {name}\ndata: {v}\n\n")
}

/// `frames` after `error_after` and `cut_after`.
fn breaks<T>(answer: &Answer, mut frames: Vec<T>, error: Option<T>) -> Vec<T> {
    if let Some(n) = answer.error_after {
        frames.truncate(n);
        frames.extend(error);
    }
    if let Some(n) = answer.cut_after {
        frames.truncate(n);
    }
    frames
}

/// Sends `out` with `headers` under `answer`'s timing and breaks.
pub(crate) async fn send(
    answer: &Answer,
    headers: &[(&'static str, String)],
    out: Out,
) -> Response {
    tokio::time::sleep(answer.delay_first).await;
    let cut = answer.cut_after.is_some();
    let half = |b: Bytes| if cut { b.slice(..b.len() / 2) } else { b };
    let (status, kind, frames) = match out {
        Out::Json(status, v) => (
            status,
            "application/json".into(),
            vec![half(v.to_string().into())],
        ),
        Out::Bytes(kind, b) => (200, kind, vec![half(b)]),
        Out::Sse(frames, error) => {
            let frames = breaks(answer, frames, Some(error));
            let kind = "text/event-stream; charset=utf-8".into();
            (200, kind, frames.into_iter().map(Bytes::from).collect())
        }
        Out::Chunks(kind, chunks) => (200, kind, breaks(answer, chunks, None)),
        Out::Misfit(why) => (500, "text/plain; charset=utf-8".into(), vec![why.into()]),
    };
    let pace = answer.pace;
    let body =
        futures::stream::iter(frames.into_iter().enumerate()).then(move |(i, f)| async move {
            if i > 0 {
                tokio::time::sleep(pace).await;
            }
            Ok::<_, std::io::Error>(f)
        });
    // Wait before breaking off: hyper drops bytes not yet flushed when
    // the body fails in the same poll.
    let broken = futures::stream::iter(cut.then_some(())).then(|()| async {
        tokio::time::sleep(Duration::from_millis(30)).await;
        Err(std::io::Error::other("cut by the script"))
    });
    let mut response = Response::new(Body::from_stream(body.chain(broken)));
    *response.status_mut() = StatusCode::from_u16(status).unwrap_or(StatusCode::OK);
    let h = response.headers_mut();
    let retry = answer.retry_after.map(|s| ("retry-after", s.to_string()));
    let all = [("content-type", kind)].into_iter();
    for (name, value) in all.chain(headers.iter().cloned()).chain(retry) {
        if let Ok(value) = value.parse() {
            h.insert(name, value);
        }
    }
    response
}

/// Runs a WebSocket `session`: its opening events, then one [`Turn`] per
/// client event, each recorded as a `"WS"` request.
pub(crate) async fn socket(
    mut ws: WebSocket,
    mut session: Box<dyn Session>,
    script: Arc<Script>,
    path: String,
    ctx: Ctx,
) {
    for e in session.open(&ctx) {
        if ws.send(Message::Text(e.to_string().into())).await.is_err() {
            return;
        }
    }
    while let Some(Ok(message)) = ws.recv().await {
        let text = match message {
            Message::Text(t) => t.to_string(),
            Message::Close(_) => return,
            _ => continue,
        };
        let body = serde_json::from_str(&text).unwrap_or(Value::String(text));
        let (method, path) = ("WS".to_owned(), path.clone());
        let req = Request {
            method,
            path,
            body: body.clone(),
            ..Request::default()
        };
        script.record(req);
        let turn = session.event(body, &mut || script.next());
        if !play(&mut ws, turn).await {
            // Dropped without a close frame: the connection just ends.
            return;
        }
    }
}

/// Sends a turn's events; false if it was cut.
async fn play(ws: &mut WebSocket, turn: Turn) -> bool {
    let answer = turn.answer.unwrap_or_default();
    tokio::time::sleep(answer.delay_first).await;
    for (i, e) in breaks(&answer, turn.events, Some(turn.error))
        .into_iter()
        .enumerate()
    {
        if i > 0 {
            tokio::time::sleep(answer.pace).await;
        }
        if ws.send(Message::Text(e.to_string().into())).await.is_err() {
            return false;
        }
    }
    answer.cut_after.is_none()
}
