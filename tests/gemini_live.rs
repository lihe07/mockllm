#![allow(clippy::unwrap_used)]

use std::time::Duration;

use base64::Engine;
use futures::{SinkExt, StreamExt};
use mockllm::{Answer, FakeGemini};
use serde_json::{Value, json};
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::{Error, Message};
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

type Ws = WebSocketStream<MaybeTlsStream<TcpStream>>;

const PATH: &str = "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent";

async fn connect(gemini: &FakeGemini, key: bool) -> Result<Ws, Error> {
    let query = if key { "?key=AIza-test" } else { "" };
    let url = gemini.base_url().replace("http://", "ws://") + PATH + query;
    tokio_tungstenite::connect_async(url)
        .await
        .map(|(ws, _)| ws)
}

async fn send(ws: &mut Ws, v: Value) {
    ws.send(Message::Text(v.to_string().into())).await.unwrap();
}

enum Got {
    Event(Value),
    Closed(Option<(u16, String)>),
    Broken,
}

async fn next(ws: &mut Ws) -> Got {
    let message = tokio::time::timeout(Duration::from_secs(5), ws.next())
        .await
        .unwrap();
    match message {
        Some(Ok(Message::Text(t))) => Got::Event(serde_json::from_str(&t).unwrap()),
        Some(Ok(Message::Close(frame))) => {
            Got::Closed(frame.map(|f| (u16::from(f.code), f.reason.to_string())))
        }
        None => Got::Closed(None),
        Some(Ok(other)) => panic!("unexpected {other:?}"),
        Some(Err(_)) => Got::Broken,
    }
}

async fn event(ws: &mut Ws) -> Value {
    match next(ws).await {
        Got::Event(e) => e,
        _ => panic!("expected an event"),
    }
}

async fn closed(ws: &mut Ws) -> (u16, String) {
    match next(ws).await {
        Got::Closed(Some(frame)) => frame,
        _ => panic!("expected a close frame"),
    }
}

/// Messages up to and including `turnComplete`.
async fn turn(ws: &mut Ws) -> Vec<Value> {
    let mut out = Vec::new();
    loop {
        let e = event(ws).await;
        let done = e["serverContent"]["turnComplete"] == true;
        out.push(e);
        if done {
            return out;
        }
    }
}

async fn ready(gemini: &FakeGemini, setup: Value) -> Ws {
    let mut ws = connect(gemini, true).await.unwrap();
    send(&mut ws, json!({ "setup": setup })).await;
    assert_eq!(event(&mut ws).await, json!({ "setupComplete": {} }));
    ws
}

#[tokio::test]
async fn a_text_turn() {
    let gemini = FakeGemini::start().await;
    let setup = json!({ "model": "models/gemini-3.8-live", "generationConfig": { "responseModalities": ["TEXT"] } });
    let mut ws = ready(&gemini, setup).await;
    // Without `turnComplete`, only appended to the conversation.
    let turns = json!([{ "role": "user", "parts": [{ "text": "capital of France?" }] }]);
    send(&mut ws, json!({ "clientContent": { "turns": turns } })).await;
    gemini.answer(Answer::text("Paris it is"));
    send(
        &mut ws,
        json!({ "clientContent": { "turns": [], "turnComplete": true } }),
    )
    .await;
    let got = turn(&mut ws).await;
    assert_eq!(got.len(), 5);
    let text: String = got[..3]
        .iter()
        .map(|e| {
            e["serverContent"]["modelTurn"]["parts"][0]["text"]
                .as_str()
                .unwrap()
        })
        .collect();
    assert_eq!(text, "Paris it is");
    assert_eq!(
        got[3],
        json!({ "serverContent": { "generationComplete": true } })
    );
    let usage = &got[4]["usageMetadata"];
    assert_eq!(usage["responseTokenCount"], 3);
    assert_eq!(usage["responseTokensDetails"][0]["modality"], "TEXT");
    let prompt = usage["promptTokenCount"].as_u64().unwrap();
    assert_eq!(usage["totalTokenCount"], prompt + 3);

    // Realtime text asks for a response too; the messages are recorded.
    send(
        &mut ws,
        json!({ "realtimeInput": { "text": "and Germany?" } }),
    )
    .await;
    let got = turn(&mut ws).await;
    assert_eq!(
        got[0]["serverContent"]["modelTurn"]["parts"][0]["text"],
        "Hi"
    );
    let sent: Vec<Value> = gemini
        .requests()
        .iter()
        .filter(|r| r.method == "WS")
        .map(|r| r.body.clone())
        .collect();
    assert_eq!(sent.len(), 4);
    assert!(sent[0]["setup"].is_object());
}

#[tokio::test]
async fn audio_is_the_default_modality() {
    let gemini = FakeGemini::start().await;
    let setup = json!({ "model": "models/gemini-3.8-live", "outputAudioTranscription": {} });
    let mut ws = ready(&gemini, setup).await;
    send(&mut ws, json!({ "realtimeInput": { "text": "hi" } })).await;
    let got = turn(&mut ws).await;
    let first = &got[0]["serverContent"];
    let data = &first["modelTurn"]["parts"][0]["inlineData"];
    assert_eq!(data["mimeType"], "audio/pcm;rate=24000");
    assert_eq!(first["outputTranscription"]["text"], "Hi");
    assert_eq!(
        got.last().unwrap()["usageMetadata"]["responseTokensDetails"][0]["modality"],
        "AUDIO"
    );

    // Scripted audio bytes, in 4800-byte chunks.
    gemini.answer(Answer::audio(vec![7u8; 6000]));
    send(
        &mut ws,
        json!({ "realtimeInput": { "audioStreamEnd": true } }),
    )
    .await;
    let got = turn(&mut ws).await;
    let b64 = base64::engine::general_purpose::STANDARD;
    let chunk = |e: &Value| {
        b64.decode(
            e["serverContent"]["modelTurn"]["parts"][0]["inlineData"]["data"]
                .as_str()
                .unwrap(),
        )
        .unwrap()
    };
    assert_eq!((chunk(&got[0]).len(), chunk(&got[1]).len()), (4800, 1200));
}

#[tokio::test]
async fn tool_calls_and_raw_events() {
    let gemini = FakeGemini::start().await;
    let mut ws = ready(&gemini, json!({ "model": "models/gemini-3.8-live" })).await;
    gemini.answer(Answer::tool_call("get_weather", json!({ "city": "Paris" })));
    send(&mut ws, json!({ "realtimeInput": { "text": "weather?" } })).await;
    let call = event(&mut ws).await;
    let f = &call["toolCall"]["functionCalls"][0];
    assert_eq!(
        (&f["name"], &f["args"]),
        (&json!("get_weather"), &json!({ "city": "Paris" }))
    );
    assert!(f["id"].is_string());
    let response = json!({ "id": f["id"], "name": "get_weather", "response": { "t": 20 } });
    gemini.answer(Answer::events(vec![
        json!({ "goAway": { "timeLeft": "10s" } }),
    ]));
    send(
        &mut ws,
        json!({ "toolResponse": { "functionResponses": [response] } }),
    )
    .await;
    assert_eq!(
        event(&mut ws).await,
        json!({ "goAway": { "timeLeft": "10s" } })
    );
}

#[tokio::test]
async fn invalid_messages_close_the_connection() {
    let gemini = FakeGemini::start().await;
    assert!(connect(&gemini, false).await.is_err(), "no key, no upgrade");

    // `setup` must come first.
    let mut ws = connect(&gemini, true).await.unwrap();
    send(&mut ws, json!({ "realtimeInput": { "text": "hi" } })).await;
    assert_eq!(closed(&mut ws).await.0, 1007);
    // ... and has a model.
    let mut ws = connect(&gemini, true).await.unwrap();
    send(&mut ws, json!({ "setup": {} })).await;
    assert_eq!(closed(&mut ws).await.0, 1007);
    // Exactly one known field per message.
    let mut ws = ready(&gemini, json!({ "model": "models/m" })).await;
    send(&mut ws, json!({ "realtimeInput": {}, "clientContent": {} })).await;
    assert_eq!(closed(&mut ws).await.0, 1007);
    let mut ws = ready(&gemini, json!({ "model": "models/m" })).await;
    send(&mut ws, json!({ "setup": { "model": "models/m" } })).await;
    assert_eq!(closed(&mut ws).await.0, 1007);
    let mut ws = ready(&gemini, json!({ "model": "models/m" })).await;
    send(&mut ws, json!({ "response.create": {} })).await;
    let (code, reason) = closed(&mut ws).await;
    assert_eq!(code, 1007);
    assert!(reason.contains("Unknown name \"response.create\""));
}

#[tokio::test]
async fn scripts_close_break_and_pace() {
    let gemini = FakeGemini::start().await;
    let text = json!({ "realtimeInput": { "text": "hi" } });
    let setup =
        json!({ "model": "models/m", "generationConfig": { "responseModalities": ["TEXT"] } });

    let mut ws = ready(&gemini, setup.clone()).await;
    gemini.answer(Answer::error(429));
    send(&mut ws, text.clone()).await;
    assert_eq!(
        closed(&mut ws).await,
        (
            1008,
            "Resource has been exhausted (e.g. check quota).".into()
        )
    );

    let mut ws = ready(&gemini, setup.clone()).await;
    gemini.answer(Answer::text("one two three").error_after(2));
    send(&mut ws, text.clone()).await;
    event(&mut ws).await;
    event(&mut ws).await;
    assert_eq!(closed(&mut ws).await.0, 1011);

    // A cut ends the connection without a close frame.
    let mut ws = ready(&gemini, setup.clone()).await;
    gemini.answer(Answer::text("one two three").cut_after(1));
    send(&mut ws, text.clone()).await;
    event(&mut ws).await;
    assert!(matches!(
        next(&mut ws).await,
        Got::Broken | Got::Closed(None)
    ));

    let mut ws = ready(&gemini, setup).await;
    gemini.answer(Answer::text("a b").delay_first(150).pace(100));
    let start = std::time::Instant::now();
    send(&mut ws, text).await;
    turn(&mut ws).await;
    assert!(start.elapsed() >= Duration::from_millis(450));
}
