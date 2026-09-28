#![allow(clippy::unwrap_used)]

use std::time::Duration;

use futures::{SinkExt, StreamExt};
use mockllm::{Answer, FakeOpenAi};
use serde_json::{Value, json};
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::{Error, Message};
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

type Ws = WebSocketStream<MaybeTlsStream<TcpStream>>;

async fn connect(openai: &FakeOpenAi, key: Option<&str>) -> Result<Ws, Error> {
    let url = openai.base_url().replace("http://", "ws://") + "/realtime?model=gpt-realtime";
    let mut req = url.into_client_request().unwrap();
    if let Some(key) = key {
        req.headers_mut()
            .insert("authorization", format!("Bearer {key}").parse().unwrap());
    }
    tokio_tungstenite::connect_async(req)
        .await
        .map(|(ws, _)| ws)
}

async fn send(ws: &mut Ws, v: Value) {
    ws.send(Message::Text(v.to_string().into())).await.unwrap();
}

/// The next event, or `None` when the connection ended.
async fn next(ws: &mut Ws) -> Result<Option<Value>, Error> {
    let message = tokio::time::timeout(Duration::from_secs(5), ws.next())
        .await
        .unwrap();
    match message {
        Some(Ok(Message::Text(t))) => Ok(Some(serde_json::from_str(&t).unwrap())),
        Some(Ok(Message::Close(_))) | None => Ok(None),
        Some(Ok(other)) => panic!("unexpected {other:?}"),
        Some(Err(e)) => Err(e),
    }
}

/// Events up to and including `response.done`.
async fn response(ws: &mut Ws) -> Vec<Value> {
    let mut events = Vec::new();
    while let Some(e) = next(ws).await.unwrap() {
        let done = e["type"] == "response.done";
        events.push(e);
        if done {
            break;
        }
    }
    events
}

fn types(events: &[Value]) -> Vec<&str> {
    events.iter().map(|e| e["type"].as_str().unwrap()).collect()
}

#[tokio::test]
async fn a_text_response_over_websocket() {
    let openai = FakeOpenAi::start().await;
    openai.answer(Answer::text("Hi there"));
    let mut ws = connect(&openai, Some("sk-test")).await.unwrap();
    let created = next(&mut ws).await.unwrap().unwrap();
    assert_eq!(created["type"], "session.created");
    assert_eq!(created["session"]["object"], "realtime.session");
    assert_eq!(created["session"]["model"], "gpt-realtime");
    assert!(created["event_id"].is_string());

    let update = json!({ "type": "session.update", "session": { "type": "realtime", "output_modalities": ["text"] } });
    send(&mut ws, update).await;
    let updated = next(&mut ws).await.unwrap().unwrap();
    assert_eq!(updated["type"], "session.updated");
    assert_eq!(updated["session"]["output_modalities"], json!(["text"]));

    send(&mut ws, json!({ "type": "response.create" })).await;
    let events = response(&mut ws).await;
    let expected = [
        "response.created",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.done",
    ];
    assert_eq!(types(&events), expected);
    let text: String = events.iter().filter_map(|e| e["delta"].as_str()).collect();
    assert_eq!(text, "Hi there");
    let done = &events.last().unwrap()["response"];
    assert_eq!(done["status"], "completed");
    assert_eq!(
        done["output"][0]["content"][0],
        json!({ "type": "output_text", "text": "Hi there" })
    );
    assert_eq!(done["usage"]["output_tokens"], 2);

    // The client's events are recorded.
    let seen = openai.requests();
    let sent: Vec<&Value> = seen
        .iter()
        .filter(|r| r.method == "WS")
        .map(|r| &r.body["type"])
        .collect();
    assert_eq!(sent, [&json!("session.update"), &json!("response.create")]);
}

#[tokio::test]
async fn audio_sessions_send_the_transcript() {
    let openai = FakeOpenAi::start().await;
    let mut ws = connect(&openai, Some("sk-test")).await.unwrap();
    next(&mut ws).await.unwrap();
    send(&mut ws, json!({ "type": "response.create" })).await;
    let events = response(&mut ws).await;
    assert!(types(&events).contains(&"response.output_audio_transcript.delta"));
    let done = &events.last().unwrap()["response"];
    assert_eq!(done["output"][0]["content"][0]["transcript"], "Hi there");
}

#[tokio::test]
async fn errors_auth_and_breaks() {
    let openai = FakeOpenAi::start().await;
    assert!(connect(&openai, None).await.is_err(), "no key, no upgrade");

    let mut ws = connect(&openai, Some("sk-test")).await.unwrap();
    next(&mut ws).await.unwrap();
    send(&mut ws, json!({ "session": {} })).await;
    let e = next(&mut ws).await.unwrap().unwrap();
    assert_eq!(
        (&e["type"], &e["error"]["code"]),
        (&json!("error"), &json!("invalid_event"))
    );

    openai.answer(Answer::text("one two three").error_after(2));
    send(&mut ws, json!({ "type": "response.create" })).await;
    let mut got = Vec::new();
    for _ in 0..3 {
        got.push(next(&mut ws).await.unwrap().unwrap());
    }
    assert_eq!(
        types(&got),
        ["response.created", "response.output_item.added", "error"]
    );
    assert_eq!(got[2]["error"]["type"], "server_error");

    // A cut closes the connection without a close frame.
    openai.answer(Answer::text("one two three").cut_after(1));
    send(&mut ws, json!({ "type": "response.create" })).await;
    assert_eq!(
        next(&mut ws).await.unwrap().unwrap()["type"],
        "response.created"
    );
    assert!(next(&mut ws).await.is_err());
}
