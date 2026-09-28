#![allow(clippy::unwrap_used)]

mod support;

use mockllm::{Answer, FakeGemini};
use serde_json::{Value, json};
use support::{Reply, post};

const KEY: (&str, &str) = ("x-goog-api-key", "AIza-test");

async fn generate(gemini: &FakeGemini, model: &str, body: Value) -> Reply {
    let url = format!(
        "{}/v1beta/models/{model}:generateContent",
        gemini.base_url()
    );
    post(&url, &[KEY], &body).await
}

async fn stream(gemini: &FakeGemini, model: &str, query: &str, body: Value) -> Reply {
    let url = format!(
        "{}/v1beta/models/{model}:streamGenerateContent{query}",
        gemini.base_url()
    );
    post(&url, &[KEY], &body).await
}

fn hi() -> Value {
    json!({ "contents": [{ "role": "user", "parts": [{ "text": "say hi" }] }] })
}

fn with(mut body: Value, extra: Value) -> Value {
    for (k, v) in extra.as_object().unwrap() {
        body[k] = v.clone();
    }
    body
}

/// Asserts a 400 `INVALID_ARGUMENT` in Google's shape whose message
/// contains `says`.
fn rejected(r: &Reply, says: &str) {
    assert_eq!(r.status, 400, "{}", r.body);
    let e = &r.json()["error"];
    assert_eq!(
        (&e["code"], &e["status"]),
        (&json!(400), &json!("INVALID_ARGUMENT"))
    );
    assert!(e["message"].as_str().unwrap().contains(says), "{}", r.body);
}

#[tokio::test]
async fn whole_text_answer_has_geminis_shape() {
    let gemini = FakeGemini::start().await;
    gemini.answer(Answer::text("Hi there"));
    let r = generate(&gemini, "gemini-2.5-flash", hi()).await;
    assert_eq!(r.status, 200, "{}", r.body);
    let v = r.json();
    let c = &v["candidates"][0];
    assert_eq!(
        c["content"],
        json!({ "parts": [{ "text": "Hi there" }], "role": "model" })
    );
    assert_eq!(
        (&c["finishReason"], &c["index"]),
        (&json!("STOP"), &json!(0))
    );
    let u = &v["usageMetadata"];
    assert_eq!(u["candidatesTokenCount"], 2);
    let total = u["promptTokenCount"].as_u64().unwrap() + 2;
    assert_eq!(u["totalTokenCount"], total);
    assert!(u.get("thoughtsTokenCount").is_none());
    assert_eq!(v["modelVersion"], "gemini-2.5-flash");
    assert!(v["responseId"].is_string());
    // The model comes from the path.
    assert_eq!(
        gemini.requests()[0].param.as_deref(),
        Some("gemini-2.5-flash")
    );
}

#[tokio::test]
async fn the_key_may_be_a_query_parameter_and_is_required() {
    let gemini = FakeGemini::start().await;
    let base = format!(
        "{}/v1beta/models/gemini-2.5-flash:generateContent",
        gemini.base_url()
    );
    let ok = post(&format!("{base}?key=AIza-test"), &[], &hi()).await;
    assert_eq!(ok.status, 200);
    let r = post(&base, &[], &hi()).await;
    assert_eq!(r.status, 403);
    assert_eq!(r.json()["error"]["status"], "PERMISSION_DENIED");
    let r = post(
        &format!("{}/v1beta/models/x:nope", gemini.base_url()),
        &[KEY],
        &hi(),
    )
    .await;
    assert_eq!(
        (r.status, &r.json()["error"]["status"]),
        (404, &json!("NOT_FOUND"))
    );
}

#[tokio::test]
async fn alt_sse_streams_whole_responses_with_crlf() {
    let gemini = FakeGemini::start().await;
    gemini.answer(Answer::text("Hi there you"));
    let r = stream(&gemini, "gemini-2.5-flash", "?alt=sse", hi()).await;
    assert!(
        r.header("content-type")
            .unwrap()
            .starts_with("text/event-stream")
    );
    assert!(r.body.starts_with("data: {") && r.body.ends_with("}\r\n\r\n"));
    let chunks = r.chunks();
    assert_eq!(chunks.len(), 3);
    let text: String = chunks
        .iter()
        .map(|c| {
            c["candidates"][0]["content"]["parts"][0]["text"]
                .as_str()
                .unwrap()
        })
        .collect();
    assert_eq!(text, "Hi there you");
    let (last, rest) = chunks.split_last().unwrap();
    assert!(rest.iter().all(|c| c.get("usageMetadata").is_none()));
    assert!(
        rest.iter()
            .all(|c| c["candidates"][0].get("finishReason").is_none())
    );
    assert_eq!(last["candidates"][0]["finishReason"], "STOP");
    assert_eq!(last["usageMetadata"]["candidatesTokenCount"], 3);
}

#[tokio::test]
async fn without_alt_sse_the_stream_is_one_json_array() {
    let gemini = FakeGemini::start().await;
    let r = stream(&gemini, "gemini-2.5-flash", "", hi()).await;
    assert!(
        r.header("content-type")
            .unwrap()
            .starts_with("application/json")
    );
    let all = r.json();
    assert_eq!(all.as_array().unwrap().len(), 2);
    assert_eq!(all[1]["candidates"][0]["finishReason"], "STOP");
}

#[tokio::test]
async fn function_calls_are_whole_parts() {
    let gemini = FakeGemini::start().await;
    let args = json!({ "city": "Paris" });
    gemini.answer(Answer::tool_call("get_weather", args.clone()));
    gemini.answer(Answer::tool_call("get_weather", args.clone()));
    let v = generate(&gemini, "gemini-2.5-pro", hi()).await.json();
    let call = json!({ "functionCall": { "name": "get_weather", "args": args } });
    assert_eq!(v["candidates"][0]["content"]["parts"], json!([call]));
    let r = stream(&gemini, "gemini-2.5-pro", "?alt=sse", hi()).await;
    assert_eq!(
        r.chunks()[0]["candidates"][0]["content"]["parts"],
        json!([call])
    );
}

#[tokio::test]
async fn validation_follows_geminis_rules() {
    let gemini = FakeGemini::start().await;
    let m = "gemini-2.5-flash";
    rejected(
        &generate(&gemini, m, with(hi(), json!({ "temperature": 1 }))).await,
        "Unknown name \"temperature\"",
    );
    rejected(
        &generate(&gemini, m, json!({})).await,
        "contents is not specified",
    );
    let bot = json!({ "contents": [{ "role": "assistant", "parts": [{ "text": "x" }] }] });
    rejected(&generate(&gemini, m, bot).await, "valid role");
    // No role is fine for a single turn.
    let bare = json!({ "contents": [{ "parts": [{ "text": "x" }] }] });
    assert_eq!(generate(&gemini, m, bare).await.status, 200);
}

#[tokio::test]
async fn max_output_tokens_is_per_model() {
    let gemini = FakeGemini::start().await;
    let config = |n: u64| {
        with(
            hi(),
            json!({ "generationConfig": { "maxOutputTokens": n } }),
        )
    };
    rejected(
        &generate(&gemini, "gemini-2.0-flash", config(8_193)).await,
        "from 1 (inclusive) to 8193 (exclusive)",
    );
    let ok = generate(&gemini, "gemini-2.5-flash", config(65_536)).await;
    assert_eq!(ok.status, 200);
}

#[tokio::test]
async fn thinking_level_is_for_gemini_3_and_its_levels() {
    let gemini = FakeGemini::start().await;
    let level = |l: &str| {
        with(
            hi(),
            json!({ "generationConfig": { "thinkingConfig": { "thinkingLevel": l } } }),
        )
    };
    rejected(
        &generate(&gemini, "gemini-2.5-pro", level("LOW")).await,
        "not supported for this model",
    );
    rejected(
        &generate(&gemini, "gemini-3-pro-preview", level("MINIMAL")).await,
        "Supported levels: low, high",
    );
    let ok = generate(&gemini, "gemini-3-flash-preview", level("MINIMAL")).await;
    assert_eq!(ok.status, 200);
}

#[tokio::test]
async fn thoughts_are_counted_and_shown_with_include_thoughts() {
    let gemini = FakeGemini::start().await;
    for _ in 0..3 {
        gemini.answer(Answer::text("Hi").reasoning("let me think"));
    }
    let v = generate(&gemini, "gemini-2.5-flash", hi()).await.json();
    assert_eq!(v["usageMetadata"]["thoughtsTokenCount"], 3);
    assert_eq!(
        v["candidates"][0]["content"]["parts"],
        json!([{ "text": "Hi" }])
    );
    let include = with(
        hi(),
        json!({ "generationConfig": { "thinkingConfig": { "includeThoughts": true } } }),
    );
    let v = generate(&gemini, "gemini-2.5-flash", include).await.json();
    let parts = &v["candidates"][0]["content"]["parts"];
    assert_eq!(parts[0], json!({ "text": "let me think", "thought": true }));
    assert_eq!(parts[1], json!({ "text": "Hi" }));
    // Off by default on Flash-Lite: nothing counted.
    let v = generate(&gemini, "gemini-2.5-flash-lite", hi())
        .await
        .json();
    assert!(v["usageMetadata"].get("thoughtsTokenCount").is_none());
}

#[tokio::test]
async fn scripted_error_and_mid_stream_error_in_googles_shape() {
    let gemini = FakeGemini::start().await;
    gemini.answer(Answer::error(429).retry_after(5));
    let r = generate(&gemini, "gemini-2.5-flash", hi()).await;
    assert_eq!((r.status, r.header("retry-after")), (429, Some("5")));
    let e = &r.json()["error"];
    assert_eq!(
        (&e["code"], &e["status"]),
        (&json!(429), &json!("RESOURCE_EXHAUSTED"))
    );

    gemini.answer(Answer::text("one two three").error_after(1));
    let r = stream(&gemini, "gemini-2.5-flash", "?alt=sse", hi()).await;
    assert!(r.ended.is_ok());
    let (first, error) = r.body.split_once("\r\n\r\n").unwrap();
    assert!(first.starts_with("data: "));
    let error: Value = serde_json::from_str(error).unwrap();
    assert_eq!(error["error"]["status"], "INTERNAL");
}

#[tokio::test]
async fn cut_after_breaks_the_stream() {
    let gemini = FakeGemini::start().await;
    gemini.answer(Answer::text("one two three").cut_after(1));
    let r = stream(&gemini, "gemini-2.5-flash", "?alt=sse", hi()).await;
    assert!(r.ended.is_err());
    assert_eq!(r.chunks().len(), 1);
}
