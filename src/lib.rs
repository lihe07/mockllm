//! Mock LLM providers: real HTTP servers on `127.0.0.1:<random>` that
//! check requests by the provider's own rules, including per-model ones,
//! and answer in its exact framing. For testing an LLM client, SDK or
//! proxy from outside, over real HTTP.
//!
//! Each mock has three parts: `validate` (the provider's request rules,
//! failing in its own error shape and status), `respond` (whole or
//! streamed bodies in its framing) and the shared [`Answer`] script.
//! Per-model rules live in one table per provider (`openai_models.rs`,
//! `anthropic_models.rs`), each row citing the docs it comes from.
//! Nothing is read from fixture files; the provider logic is code.
//!
//! ```no_run
//! # async fn demo() {
//! use mockllm::{Answer, FakeOpenAi};
//! let openai = FakeOpenAi::start().await;
//! openai.answer(Answer::text("Hi there").cut_after(2));
//! // point the client under test at openai.base_url(), send, then:
//! let sent = openai.requests();
//! # }
//! ```

mod anthropic;
mod anthropic_models;
mod openai;
mod openai_models;

pub use anthropic::FakeAnthropic;
pub use openai::{FakeOpenAi, Quirks};

use std::collections::VecDeque;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use axum::body::Body;
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use bytes::Bytes;
use futures::StreamExt;
use serde_json::Value;
use tokio::task::JoinHandle;

/// What a fake answers to its next valid request. Requests that fail
/// validation do not use up an answer. With nothing queued, a fake
/// answers [`Answer::default`], a short text.
#[derive(Clone, Debug)]
pub struct Answer {
    pub(crate) reply: Reply,
    pub(crate) reasoning: Option<String>,
    pub(crate) cut_after: Option<usize>,
    pub(crate) error_after: Option<usize>,
    pub(crate) delay_first: Duration,
    pub(crate) pace: Duration,
    pub(crate) retry_after: Option<u64>,
}

#[derive(Clone, Debug)]
pub(crate) enum Reply {
    Text(String),
    Tool { name: String, args: Value },
    Error(u16),
}

impl Default for Answer {
    fn default() -> Self {
        Self::text("Hi there")
    }
}

impl Answer {
    fn of(reply: Reply) -> Self {
        Self {
            reply,
            reasoning: None,
            cut_after: None,
            error_after: None,
            delay_first: Duration::ZERO,
            pace: Duration::ZERO,
            retry_after: None,
        }
    }

    /// An assistant text; streamed a word per chunk.
    pub fn text(text: impl Into<String>) -> Self {
        Self::of(Reply::Text(text.into()))
    }

    /// One call of tool `name` with `args`; streamed arguments in pieces.
    pub fn tool_call(name: impl Into<String>, args: Value) -> Self {
        let name = name.into();
        Self::of(Reply::Tool { name, args })
    }

    /// An error of `status` in the provider's own shape, before any body.
    pub fn error(status: u16) -> Self {
        Self::of(Reply::Error(status))
    }

    /// Reasoning before the answer. OpenAI reasoning models count it in
    /// `reasoning_tokens` and show none of it; [`Quirks::reasoning_content`]
    /// shows it; Anthropic sends it as a `thinking` block when the model
    /// thinks on the request (its text empty when `display` is omitted).
    pub fn reasoning(mut self, text: impl Into<String>) -> Self {
        self.reasoning = Some(text.into());
        self
    }

    /// The stream breaks off (the connection closes without its end)
    /// after `n` SSE events. A whole body breaks off halfway.
    pub fn cut_after(mut self, n: usize) -> Self {
        self.cut_after = Some(n);
        self
    }

    /// After `n` SSE events the stream sends the provider's mid-stream
    /// error event and ends.
    pub fn error_after(mut self, n: usize) -> Self {
        self.error_after = Some(n);
        self
    }

    /// Waits `ms` before the response head: a slow first byte.
    pub fn delay_first(mut self, ms: u64) -> Self {
        self.delay_first = Duration::from_millis(ms);
        self
    }

    /// Waits `ms` between SSE events.
    pub fn pace(mut self, ms: u64) -> Self {
        self.pace = Duration::from_millis(ms);
        self
    }

    /// Adds `retry-after: <secs>` to the response.
    pub fn retry_after(mut self, secs: u64) -> Self {
        self.retry_after = Some(secs);
        self
    }
}

/// A request a fake received, valid or not.
#[derive(Clone, Debug)]
pub struct Request {
    pub method: String,
    pub path: String,
    pub headers: HeaderMap,
    /// The body as JSON; `Null` if it was not JSON.
    pub body: Value,
}

impl Request {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.get(name).and_then(|v| v.to_str().ok())
    }
}

#[derive(Debug, Default)]
pub(crate) struct Script {
    answers: Mutex<VecDeque<Answer>>,
    requests: Mutex<Vec<Request>>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl Script {
    pub(crate) fn next(&self) -> Answer {
        lock(&self.answers).pop_front().unwrap_or_default()
    }
}

/// The HTTP server under a fake; stops when dropped.
#[derive(Debug)]
pub(crate) struct Server {
    addr: SocketAddr,
    script: Arc<Script>,
    task: JoinHandle<()>,
}

impl Server {
    /// Serves every request with `handle`, after recording it.
    pub(crate) async fn start<F, Fut>(handle: F) -> Self
    where
        F: Fn(Request, Arc<Script>) -> Fut + Clone + Send + Sync + 'static,
        Fut: Future<Output = Response> + Send + 'static,
    {
        let script = Arc::new(Script::default());
        let shared = script.clone();
        let app = axum::Router::new().fallback(move |req: axum::extract::Request| {
            let (handle, script) = (handle.clone(), shared.clone());
            async move {
                let (parts, body) = req.into_parts();
                let bytes = axum::body::to_bytes(body, usize::MAX)
                    .await
                    .unwrap_or_default();
                let req = Request {
                    method: parts.method.to_string(),
                    path: parts.uri.path().to_owned(),
                    headers: parts.headers,
                    body: serde_json::from_slice(&bytes).unwrap_or(Value::Null),
                };
                lock(&script.requests).push(req.clone());
                handle(req, script).await
            }
        });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .unwrap_or_else(|e| panic!("bind a fake provider: {e}"));
        let addr = listener
            .local_addr()
            .unwrap_or_else(|e| panic!("fake provider address: {e}"));
        let task = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        Self { addr, script, task }
    }

    pub(crate) fn origin(&self) -> String {
        format!("http://{}", self.addr)
    }

    pub(crate) fn answer(&self, answer: Answer) {
        lock(&self.script.answers).push_back(answer);
    }

    pub(crate) fn requests(&self) -> Vec<Request> {
        lock(&self.script.requests).clone()
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.task.abort();
    }
}

/// A body to send under an [`Answer`]'s timing and breaks.
pub(crate) enum Out {
    Json(u16, Value),
    /// Whole SSE events, and the provider's mid-stream error event.
    Sse(Vec<String>, String),
}

/// `map[key]`, or `null` when missing (indexing a `Map` panics).
pub(crate) fn at<'a>(map: &'a serde_json::Map<String, Value>, key: &str) -> &'a Value {
    static NULL: Value = Value::Null;
    map.get(key).unwrap_or(&NULL)
}

/// A fresh number for ids.
pub(crate) fn next_id() -> u64 {
    static N: AtomicU64 = AtomicU64::new(1);
    N.fetch_add(1, Ordering::Relaxed)
}

pub(crate) fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// Splits `text` into word pieces, each but the first with its leading
/// space: "Hi there" is ["Hi", " there"].
pub(crate) fn words(text: &str) -> Vec<String> {
    let (mut out, mut cur) = (Vec::new(), String::new());
    for c in text.chars() {
        if c == ' ' && !cur.trim().is_empty() {
            out.push(std::mem::take(&mut cur));
        }
        cur.push(c);
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Splits `s` into pieces of up to `n` chars, as providers stream JSON.
pub(crate) fn pieces(s: &str, n: usize) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    chars.chunks(n).map(|c| c.iter().collect()).collect()
}

/// A rough token count: the words in every string under `v`, at least 1.
pub(crate) fn tokens(v: &Value) -> u64 {
    fn count(v: &Value) -> u64 {
        match v {
            Value::String(s) => s.split_whitespace().count() as u64,
            Value::Array(a) => a.iter().map(count).sum(),
            Value::Object(o) => o.values().map(count).sum(),
            _ => 0,
        }
    }
    count(v).max(1)
}

/// Sends `out` with `headers`, applying `answer`'s delay, pace, cut and
/// mid-stream error.
pub(crate) async fn send(
    answer: &Answer,
    headers: &[(&'static str, String)],
    out: Out,
) -> Response {
    tokio::time::sleep(answer.delay_first).await;
    let (status, kind, frames) = match out {
        Out::Json(status, v) => {
            let text = v.to_string();
            let frames = match answer.cut_after {
                Some(_) => vec![text[..text.len() / 2].to_owned()],
                None => vec![text],
            };
            (status, "application/json", frames)
        }
        Out::Sse(mut frames, error) => {
            if let Some(n) = answer.error_after {
                frames.truncate(n);
                frames.push(error);
            }
            if let Some(n) = answer.cut_after {
                frames.truncate(n);
            }
            (200, "text/event-stream; charset=utf-8", frames)
        }
    };
    let pace = answer.pace;
    let body =
        futures::stream::iter(frames.into_iter().enumerate()).then(move |(i, f)| async move {
            if i > 0 {
                tokio::time::sleep(pace).await;
            }
            Ok::<_, std::io::Error>(Bytes::from(f))
        });
    // Wait before breaking off: hyper drops bytes not yet flushed when
    // the body fails in the same poll.
    let cut = futures::stream::iter(answer.cut_after.map(|_| ())).then(|()| async {
        tokio::time::sleep(Duration::from_millis(30)).await;
        Err(std::io::Error::other("cut by the script"))
    });
    let mut response = Response::new(Body::from_stream(body.chain(cut)));
    *response.status_mut() = StatusCode::from_u16(status).unwrap_or(StatusCode::OK);
    let h = response.headers_mut();
    let retry = answer.retry_after.map(|s| ("retry-after", s.to_string()));
    let all = [("content-type", kind.to_owned())].into_iter();
    for (name, value) in all.chain(headers.iter().cloned()).chain(retry) {
        if let Ok(value) = value.parse() {
            h.insert(name, value);
        }
    }
    response
}
