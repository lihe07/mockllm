#![allow(clippy::unwrap_used)]

mod support;

use std::time::{Duration, Instant};

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use futures::StreamExt;
use mockllm::{Answer, FakeOpenAi};
use serde_json::{Value, json};
use support::post;

const KEY: (&str, &str) = ("authorization", "Bearer sk-test");

/// A binary reply: status, content type, body, and whether it ended.
struct Audio {
    status: u16,
    kind: String,
    body: Vec<u8>,
    ended: bool,
}

async fn speak(openai: &FakeOpenAi, body: Value) -> Audio {
    let resp = reqwest::Client::new()
        .post(format!("{}/audio/speech", openai.base_url()))
        .header(KEY.0, KEY.1)
        .header("content-type", "application/json")
        .body(body.to_string())
        .send()
        .await
        .unwrap();
    let status = resp.status().as_u16();
    let kind = resp.headers()["content-type"].to_str().unwrap().to_owned();
    let (mut body, mut ended) = (Vec::new(), true);
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(b) => body.extend_from_slice(&b),
            Err(_) => {
                ended = false;
                break;
            }
        }
    }
    Audio {
        status,
        kind,
        body,
        ended,
    }
}

fn hi(model: &str) -> Value {
    json!({ "model": model, "input": "Hello there", "voice": "alloy" })
}

fn with(mut body: Value, extra: Value) -> Value {
    for (k, v) in extra.as_object().unwrap() {
        body[k] = v.clone();
    }
    body
}

#[tokio::test]
async fn unscripted_audio_per_format() {
    let openai = FakeOpenAi::start().await;
    let a = speak(&openai, hi("tts-1")).await;
    assert_eq!((a.status, a.kind.as_str()), (200, "audio/mpeg"));
    assert_eq!(&a.body[..2], &[0xFF, 0xFB]);
    let a = speak(
        &openai,
        with(hi("tts-1"), json!({ "response_format": "wav" })),
    )
    .await;
    assert_eq!(a.kind, "audio/wav");
    assert_eq!(&a.body[..4], b"RIFF");
    assert_eq!(&a.body[8..12], b"WAVE");
    let a = speak(
        &openai,
        with(hi("tts-1"), json!({ "response_format": "pcm" })),
    )
    .await;
    assert_eq!((a.kind.as_str(), a.body.len()), ("audio/pcm", 4800));
    let a = speak(
        &openai,
        with(hi("tts-1"), json!({ "response_format": "opus" })),
    )
    .await;
    assert_eq!(a.kind, "audio/opus");
}

#[tokio::test]
async fn scripted_audio_is_sent_chunked_and_can_be_cut() {
    let openai = FakeOpenAi::start().await;
    let bytes: Vec<u8> = (0..3000).map(|i| (i % 251) as u8).collect();
    openai.answer(Answer::audio(bytes.clone()));
    let a = speak(&openai, hi("gpt-4o-mini-tts")).await;
    assert_eq!(a.body, bytes);
    assert!(a.ended);
    // Chunks of 1024 bytes: cut after the first.
    openai.answer(Answer::audio(bytes.clone()).cut_after(1));
    let a = speak(&openai, hi("gpt-4o-mini-tts")).await;
    assert_eq!(a.body, bytes[..1024]);
    assert!(!a.ended);
    openai.answer(Answer::audio(bytes).delay_first(200).pace(100));
    let t = Instant::now();
    let a = speak(&openai, hi("gpt-4o-mini-tts")).await;
    assert!(t.elapsed() >= Duration::from_millis(400));
    assert_eq!(a.body.len(), 3000);
}

#[tokio::test]
async fn sse_stream_format_sends_base64_deltas_then_done() {
    let openai = FakeOpenAi::start().await;
    let bytes: Vec<u8> = (0..2500).map(|i| (i % 7) as u8).collect();
    openai.answer(Answer::audio(bytes.clone()));
    let url = format!("{}/audio/speech", openai.base_url());
    let body = with(hi("gpt-4o-mini-tts"), json!({ "stream_format": "sse" }));
    let r = post(&url, &[KEY], &body).await;
    assert!(
        r.header("content-type")
            .unwrap()
            .starts_with("text/event-stream")
    );
    let events = r.chunks();
    assert_eq!(events.len(), 4);
    let mut audio = Vec::new();
    for e in &events[..3] {
        assert_eq!(e["type"], "speech.audio.delta");
        audio.extend(STANDARD.decode(e["audio"].as_str().unwrap()).unwrap());
    }
    assert_eq!(audio, bytes);
    let done = &events[3];
    assert_eq!(done["type"], "speech.audio.done");
    assert_eq!(done["usage"]["total_tokens"], 5);
    // A mid-stream error in OpenAI's error shape.
    openai.answer(Answer::audio(bytes).error_after(1));
    let events = post(&url, &[KEY], &body).await.chunks();
    assert_eq!(events.len(), 2);
    assert_eq!(events[1]["error"]["type"], "server_error");
}

#[tokio::test]
async fn validation_follows_the_reference() {
    let openai = FakeOpenAi::start().await;
    let url = format!("{}/audio/speech", openai.base_url());
    let cases = [
        (
            json!({ "model": "tts-1", "voice": "alloy" }),
            "missing_required_parameter",
        ),
        (
            json!({ "model": "tts-1", "input": "x" }),
            "missing_required_parameter",
        ),
        (
            with(hi("tts-1"), json!({ "voice": "nobody" })),
            "invalid_value",
        ),
        // `marin` is not among tts-1's voices; fine elsewhere.
        (
            with(hi("tts-1"), json!({ "voice": "marin" })),
            "invalid_value",
        ),
        (
            with(hi("tts-1"), json!({ "stream_format": "sse" })),
            "unsupported_value",
        ),
        (
            with(hi("tts-1"), json!({ "speed": 5 })),
            "decimal_above_max_value",
        ),
        (
            with(hi("tts-1"), json!({ "speed": 0.1 })),
            "decimal_below_min_value",
        ),
        (
            with(hi("tts-1"), json!({ "response_format": "ogg" })),
            "invalid_value",
        ),
        (
            with(hi("tts-1"), json!({ "input": "x".repeat(4097) })),
            "string_above_max_length",
        ),
        (
            with(hi("tts-1"), json!({ "stream": true })),
            "unrecognized_request_argument",
        ),
    ];
    for (body, code) in cases {
        let r = post(&url, &[KEY], &body).await;
        assert_eq!((r.status, r.error().0.as_str()), (400, code), "{body}");
    }
    let custom = with(
        hi("gpt-4o-mini-tts"),
        json!({ "voice": { "id": "voice_1234" } }),
    );
    for body in [
        with(hi("gpt-4o-mini-tts"), json!({ "voice": "marin" })),
        custom,
    ] {
        assert_eq!(speak(&openai, body).await.status, 200);
    }
    let r = post(&url, &[], &hi("tts-1")).await;
    assert_eq!(r.status, 401);
}

#[tokio::test]
async fn scripted_errors_and_misfits() {
    let openai = FakeOpenAi::start().await;
    let url = format!("{}/audio/speech", openai.base_url());
    openai.answer(Answer::error(429).retry_after(3));
    let r = post(&url, &[KEY], &hi("tts-1")).await;
    assert_eq!(
        (r.status, r.error().0.as_str()),
        (429, "rate_limit_exceeded")
    );
    assert_eq!(r.header("retry-after"), Some("3"));
    openai.answer(Answer::text("not audio"));
    let r = post(&url, &[KEY], &hi("tts-1")).await;
    assert_eq!(r.status, 500);
    assert!(r.body.contains("speech cannot answer a text"));
}
