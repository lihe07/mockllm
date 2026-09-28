#![allow(clippy::unwrap_used)]

mod support;

use futures::StreamExt;
use mockllm::{Answer, FakeOpenAi};
use reqwest::multipart::{Form, Part};
use support::Reply;

/// POSTs a multipart form of `fields` and, unless `None`, an audio `file`
/// part of that name.
async fn transcribe(openai: &FakeOpenAi, fields: &[(&str, &str)], file: Option<&str>) -> Reply {
    let mut form = Form::new();
    for (k, v) in fields {
        form = form.text(k.to_string(), v.to_string());
    }
    if let Some(name) = file {
        form = form.part(
            "file",
            Part::bytes(vec![0u8; 2048]).file_name(name.to_owned()),
        );
    }
    let resp = reqwest::Client::new()
        .post(format!("{}/audio/transcriptions", openai.base_url()))
        .header("authorization", "Bearer sk-test")
        .multipart(form)
        .send()
        .await
        .unwrap();
    let (status, headers) = (resp.status().as_u16(), resp.headers().clone());
    let (mut body, mut ended) = (Vec::new(), Ok(()));
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(b) => body.extend_from_slice(&b),
            Err(e) => {
                ended = Err(e.to_string());
                break;
            }
        }
    }
    let body = String::from_utf8(body).unwrap();
    Reply {
        status,
        headers,
        body,
        ended,
    }
}

async fn ok(openai: &FakeOpenAi, fields: &[(&str, &str)]) -> Reply {
    let r = transcribe(openai, fields, Some("audio.wav")).await;
    assert_eq!(r.status, 200, "{}", r.body);
    r
}

#[tokio::test]
async fn json_answers_per_model() {
    let openai = FakeOpenAi::start().await;
    openai.answer(Answer::transcript("the quick brown fox"));
    let v = ok(&openai, &[("model", "whisper-1")]).await.json();
    assert_eq!(v["text"], "the quick brown fox");
    assert_eq!(v["usage"]["type"], "duration");
    let fields = [("model", "gpt-4o-transcribe"), ("include[]", "logprobs")];
    let v = ok(&openai, &fields).await.json();
    assert_eq!(v["text"], "Hi there");
    assert_eq!(v["usage"]["type"], "tokens");
    assert_eq!(v["usage"]["output_tokens"], 2);
    assert_eq!(v["logprobs"].as_array().unwrap().len(), 2);
    let v = ok(&openai, &[("model", "gpt-transcribe"), ("language", "fr")])
        .await
        .json();
    assert_eq!(v["languages"][0]["code"], "fr");
}

#[tokio::test]
async fn text_srt_vtt_and_verbose_json() {
    let openai = FakeOpenAi::start().await;
    let text = |f: &'static str| [("model", "whisper-1"), ("response_format", f)];
    let r = ok(&openai, &text("text")).await;
    assert!(r.header("content-type").unwrap().starts_with("text/plain"));
    assert_eq!(r.body, "Hi there\n");
    let r = ok(&openai, &text("srt")).await;
    assert_eq!(r.body, "1\n00:00:00,000 --> 00:00:01,000\nHi there\n\n");
    let r = ok(&openai, &text("vtt")).await;
    assert_eq!(
        r.body,
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHi there\n\n"
    );
    let v = ok(&openai, &text("verbose_json")).await.json();
    assert_eq!(
        (v["task"].as_str(), v["duration"].as_f64()),
        (Some("transcribe"), Some(1.0))
    );
    assert_eq!(v["segments"][0]["text"], "Hi there");
    assert!(v.get("words").is_none());
    // Both granularities, as repeated `[]` fields.
    let fields = [
        ("model", "whisper-1"),
        ("response_format", "verbose_json"),
        ("timestamp_granularities[]", "word"),
        ("timestamp_granularities[]", "segment"),
    ];
    let v = ok(&openai, &fields).await.json();
    assert_eq!(v["segments"].as_array().unwrap().len(), 1);
    assert_eq!(v["words"][1]["word"], "there");
    assert_eq!(v["words"][1]["start"], 0.5);
    let v = ok(&openai, &fields[..3]).await.json();
    assert!(v.get("segments").is_none());
}

#[tokio::test]
async fn stream_sends_transcript_events() {
    let openai = FakeOpenAi::start().await;
    openai.answer(Answer::transcript("one two three"));
    let fields = [("model", "gpt-4o-mini-transcribe"), ("stream", "true")];
    let r = ok(&openai, &fields).await;
    assert!(
        r.header("content-type")
            .unwrap()
            .starts_with("text/event-stream")
    );
    let events = r.chunks();
    let deltas: Vec<_> = events[..3]
        .iter()
        .map(|e| e["delta"].as_str().unwrap())
        .collect();
    assert_eq!(deltas, ["one", " two", " three"]);
    assert!(
        events[..3]
            .iter()
            .all(|e| e["type"] == "transcript.text.delta")
    );
    assert_eq!(events[3]["type"], "transcript.text.done");
    assert_eq!(events[3]["text"], "one two three");
    assert_eq!(events[3]["usage"]["type"], "tokens");
    // whisper-1 ignores `stream`.
    let v = ok(&openai, &[("model", "whisper-1"), ("stream", "true")])
        .await
        .json();
    assert_eq!(v["text"], "Hi there");
    // Diarized: deltas with `segment_id`, then the finished segment.
    let fields = [
        ("model", "gpt-4o-transcribe-diarize"),
        ("response_format", "diarized_json"),
        ("stream", "true"),
    ];
    let events = ok(&openai, &fields).await.chunks();
    assert_eq!(events[0]["segment_id"], "seg_001");
    let types: Vec<_> = events.iter().map(|e| e["type"].as_str().unwrap()).collect();
    assert_eq!(
        &types[2..],
        ["transcript.text.segment", "transcript.text.done"]
    );
    assert_eq!(events[2]["speaker"], "A");
    let v = ok(&openai, &fields[..2]).await.json();
    assert_eq!(v["segments"][0]["type"], "transcript.text.segment");
}

#[tokio::test]
async fn validation_follows_the_reference() {
    let openai = FakeOpenAi::start().await;
    let rejected = |r: Reply, code: &str| {
        assert_eq!(r.status, 400, "{}", r.body);
        assert_eq!(r.error().0, code, "{}", r.body);
    };
    let r = transcribe(&openai, &[("model", "whisper-1")], None).await;
    rejected(r, "missing_required_parameter");
    let r = transcribe(&openai, &[], Some("a.wav")).await;
    rejected(r, "missing_required_parameter");
    let r = transcribe(&openai, &[("model", "whisper-1")], Some("a.txt")).await;
    assert!(r.error().1.starts_with("Invalid file format"));
    let cases: &[(&[(&str, &str)], &str)] = &[
        (
            &[("model", "gpt-4o-transcribe"), ("response_format", "srt")],
            "unsupported_value",
        ),
        (
            &[("model", "whisper-1"), ("response_format", "xml")],
            "invalid_value",
        ),
        (
            &[("model", "whisper-1"), ("response_format", "diarized_json")],
            "unsupported_value",
        ),
        (
            &[
                ("model", "gpt-4o-transcribe"),
                ("timestamp_granularities[]", "word"),
            ],
            "unsupported_parameter",
        ),
        (
            &[("model", "gpt-4o-transcribe-diarize"), ("prompt", "hi")],
            "unsupported_parameter",
        ),
        (
            &[("model", "whisper-1"), ("include[]", "logprobs")],
            "unsupported_parameter",
        ),
        (
            &[("model", "whisper-1"), ("temperature", "2")],
            "decimal_above_max_value",
        ),
        (
            &[("model", "whisper-1"), ("color", "red")],
            "unrecognized_request_argument",
        ),
    ];
    for (fields, code) in cases {
        rejected(transcribe(&openai, fields, Some("a.mp3")).await, code);
    }
    // Timestamp granularities need verbose_json.
    let fields = [
        ("model", "whisper-1"),
        ("timestamp_granularities[]", "word"),
    ];
    let r = transcribe(&openai, &fields, Some("a.mp3")).await;
    assert_eq!(r.status, 400);
    assert!(r.error().1.contains("verbose_json"));
}

#[tokio::test]
async fn scripts_apply_to_the_stream() {
    let openai = FakeOpenAi::start().await;
    let fields = [("model", "gpt-4o-transcribe"), ("stream", "true")];
    openai.answer(Answer::transcript("a b c").error_after(1));
    let events = ok(&openai, &fields).await.chunks();
    assert_eq!(events.len(), 2);
    assert_eq!(events[1]["error"]["type"], "server_error");
    openai.answer(Answer::transcript("a b c").cut_after(2));
    let r = ok(&openai, &fields).await;
    assert_eq!(r.chunks().len(), 2);
    assert!(r.ended.is_err());
    openai.answer(Answer::error(429));
    let r = transcribe(&openai, &fields, Some("a.wav")).await;
    assert_eq!(
        (r.status, r.error().0.as_str()),
        (429, "rate_limit_exceeded")
    );
    openai.answer(Answer::image(vec![1, 2, 3]));
    let r = transcribe(&openai, &fields, Some("a.wav")).await;
    assert_eq!(r.status, 500);
}
