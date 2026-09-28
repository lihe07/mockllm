#![allow(clippy::unwrap_used)]

mod anthropic_support;
mod support;

use anthropic_support::{get, rejected};
use mockllm::{Answer, FakeAnthropic};
use serde_json::json;

#[tokio::test]
async fn lists_models_newest_first() {
    let anthropic = FakeAnthropic::start().await;
    let r = get(&anthropic, "/v1/models").await;
    assert_eq!(r.status, 200);
    let v = r.json();
    let data = v["data"].as_array().unwrap();
    assert_eq!(data[0]["id"], "claude-fable-5-1");
    assert_eq!(v["first_id"], data[0]["id"]);
    assert_eq!(v["last_id"], data.last().unwrap()["id"]);
    assert_eq!(v["has_more"], false);
    let m = &data[0];
    assert_eq!(
        (&m["type"], &m["display_name"]),
        (&json!("model"), &json!("Claude Fable 5.1"))
    );
    assert_eq!(
        (&m["max_input_tokens"], &m["max_tokens"]),
        (&json!(1_000_000), &json!(128_000))
    );
    assert!(m["created_at"].as_str().unwrap().ends_with('Z'));
    assert!(m.as_object().unwrap().contains_key("capabilities"));
}

#[tokio::test]
async fn pages_by_limit_and_cursors() {
    let anthropic = FakeAnthropic::start().await;
    let v = get(&anthropic, "/v1/models?limit=2").await.json();
    assert_eq!(v["data"].as_array().unwrap().len(), 2);
    assert_eq!(v["has_more"], true);
    let last = v["last_id"].as_str().unwrap();
    let next = get(&anthropic, &format!("/v1/models?limit=2&after_id={last}"))
        .await
        .json();
    assert_eq!(next["data"][0]["id"], "claude-sonnet-5-5");
    let first = next["first_id"].as_str().unwrap();
    let back = get(&anthropic, &format!("/v1/models?limit=5&before_id={first}"))
        .await
        .json();
    assert_eq!(back["data"], v["data"]);
    assert_eq!(back["has_more"], false);
}

#[tokio::test]
async fn retrieves_resolves_aliases_and_rejects() {
    let anthropic = FakeAnthropic::start().await;
    let v = get(&anthropic, "/v1/models/claude-opus-5").await.json();
    assert_eq!(v["created_at"], "2026-07-24T00:00:00Z");
    let v = get(&anthropic, "/v1/models/claude-haiku-4-5").await.json();
    assert_eq!(v["id"], "claude-haiku-4-5-20251001");
    assert_eq!(
        (&v["max_input_tokens"], &v["max_tokens"]),
        (&json!(200_000), &json!(64_000))
    );
    let r = get(&anthropic, "/v1/models/claude-nope").await;
    rejected(&r, 404, "not_found_error", "claude-nope");
    for bad in ["0", "1001", "x"] {
        let r = get(&anthropic, &format!("/v1/models?limit={bad}")).await;
        rejected(&r, 400, "invalid_request_error", "limit");
    }
}

#[tokio::test]
async fn answer_kinds_and_scripts() {
    let anthropic = FakeAnthropic::start().await;
    anthropic.answer(Answer::text("no"));
    let r = get(&anthropic, "/v1/models").await;
    assert_eq!(r.status, 500);
    assert!(r.body.contains("models cannot answer a text"));
    anthropic.answer(Answer::error(529));
    rejected(
        &get(&anthropic, "/v1/models").await,
        529,
        "overloaded_error",
        "Overloaded",
    );
    anthropic.answer(Answer::default().cut_after(1));
    assert!(get(&anthropic, "/v1/models").await.ended.is_err());
}
