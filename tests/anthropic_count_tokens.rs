#![allow(clippy::unwrap_used)]

mod anthropic_support;
mod support;

use anthropic_support::{post, rejected};
use mockllm::{Answer, FakeAnthropic};
use serde_json::json;

const PATH: &str = "/v1/messages/count_tokens";

#[tokio::test]
async fn counts_messages_system_and_tools() {
    let anthropic = FakeAnthropic::start().await;
    let body = json!({ "model": "claude-opus-5", "messages": [{ "role": "user", "content": "one two three" }] });
    let r = post(&anthropic, PATH, &body).await;
    assert_eq!(r.status, 200);
    assert!(r.header("request-id").is_some());
    let v = r.json();
    assert_eq!(v.as_object().unwrap().len(), 1);
    let bare = v["input_tokens"].as_u64().unwrap();
    let mut more = body.clone();
    more["system"] = json!("be brief");
    more["tools"] = json!([{ "name": "get_weather", "input_schema": { "type": "object" } }]);
    let v = post(&anthropic, PATH, &more).await.json();
    assert!(v["input_tokens"].as_u64().unwrap() > bare);
}

#[tokio::test]
async fn validation_follows_the_reference() {
    let anthropic = FakeAnthropic::start().await;
    let msgs = json!([{ "role": "user", "content": "hi" }]);
    let r = post(&anthropic, PATH, &json!({ "messages": msgs })).await;
    rejected(&r, 400, "invalid_request_error", "model: Field required");
    let r = post(&anthropic, PATH, &json!({ "model": "claude-opus-5" })).await;
    rejected(&r, 400, "invalid_request_error", "messages: Field required");
    // `max_tokens` is not a parameter here.
    let body = json!({ "model": "claude-opus-5", "max_tokens": 10, "messages": msgs });
    let r = post(&anthropic, PATH, &body).await;
    rejected(&r, 400, "invalid_request_error", "max_tokens: Extra inputs");
    let body = json!({ "model": "m", "messages": [{ "role": "system", "content": "x" }] });
    let r = post(&anthropic, PATH, &body).await;
    rejected(
        &r,
        400,
        "invalid_request_error",
        "Unexpected role \"system\"",
    );
    let no_body = anthropic_support::call(&anthropic, "POST", PATH, &[], None, None);
    rejected(
        &no_body.await,
        400,
        "invalid_request_error",
        "parsing the body",
    );
}

#[tokio::test]
async fn answer_kinds_and_scripts() {
    let anthropic = FakeAnthropic::start().await;
    let body =
        json!({ "model": "claude-opus-5", "messages": [{ "role": "user", "content": "hi" }] });
    anthropic.answer(Answer::events(vec![json!({ "input_tokens": 2095 })]));
    assert_eq!(
        post(&anthropic, PATH, &body).await.json()["input_tokens"],
        2095
    );
    anthropic.answer(Answer::text("no"));
    assert_eq!(post(&anthropic, PATH, &body).await.status, 500);
    anthropic.answer(Answer::error(429).retry_after(3));
    let r = post(&anthropic, PATH, &body).await;
    rejected(&r, 429, "rate_limit_error", "rate limit");
    assert_eq!(r.header("retry-after"), Some("3"));
    anthropic.answer(Answer::default().cut_after(1));
    assert!(post(&anthropic, PATH, &body).await.ended.is_err());
    anthropic.answer(Answer::default().delay_first(150));
    let t = std::time::Instant::now();
    assert_eq!(post(&anthropic, PATH, &body).await.status, 200);
    assert!(t.elapsed().as_millis() >= 150);
}
