#![allow(clippy::unwrap_used)]

mod support;

use base64::Engine;
use mockllm::{Answer, FakeOpenAi};
use serde_json::{Value, json};
use support::{Reply, post};

const KEY: (&str, &str) = ("authorization", "Bearer sk-test");

async fn generate(openai: &FakeOpenAi, body: Value) -> Reply {
    let url = format!("{}/images/generations", openai.base_url());
    post(&url, &[KEY], &body).await
}

#[tokio::test]
async fn dall_e_answers_urls_the_mock_serves() {
    let openai = FakeOpenAi::start().await;
    openai.answer(Answer::image(b"fake png".to_vec()));
    let v = generate(&openai, json!({ "prompt": "an otter", "n": 2 }))
        .await
        .json();
    assert!(v["created"].is_u64());
    let data = v["data"].as_array().unwrap();
    assert_eq!(data.len(), 2);
    let url = data[0]["url"].as_str().unwrap();
    let image = reqwest::get(url).await.unwrap();
    assert_eq!(image.headers()["content-type"], "image/png");
    assert_eq!(image.bytes().await.unwrap().as_ref(), b"fake png");
    assert!(v.get("usage").is_none());
}

#[tokio::test]
async fn b64_json_and_gpt_image_models() {
    let openai = FakeOpenAi::start().await;
    let body = json!({ "model": "dall-e-3", "prompt": "an otter", "response_format": "b64_json" });
    let v = generate(&openai, body).await.json();
    let b64 = v["data"][0]["b64_json"].as_str().unwrap();
    let png = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .unwrap();
    assert_eq!(&png[1..4], b"PNG");
    assert_eq!(v["data"][0]["revised_prompt"], "an otter");
    // GPT image models always answer base64, with usage.
    let v = generate(
        &openai,
        json!({ "model": "gpt-image-1", "prompt": "an otter" }),
    )
    .await
    .json();
    assert!(v["data"][0]["b64_json"].is_string());
    assert!(v["data"][0].get("url").is_none());
    assert!(v["usage"]["total_tokens"].is_u64());
}

#[tokio::test]
async fn validation_follows_the_reference() {
    let openai = FakeOpenAi::start().await;
    let r = generate(&openai, json!({ "model": "dall-e-2" })).await;
    assert_eq!(
        (r.status, r.error().0.as_str()),
        (400, "missing_required_parameter")
    );
    let r = generate(&openai, json!({ "prompt": "x", "n": 11 })).await;
    assert_eq!((r.status, r.error().0.as_str()), (400, "invalid_value"));
    let r = generate(
        &openai,
        json!({ "model": "dall-e-3", "prompt": "x", "n": 2 }),
    )
    .await;
    assert_eq!(r.status, 400);
    let r = generate(
        &openai,
        json!({ "model": "dall-e-3", "prompt": "x", "size": "256x256" }),
    )
    .await;
    assert!(r.error().1.contains("1792x1024"));
    let body = json!({ "model": "gpt-image-1", "prompt": "x", "response_format": "url" });
    let r = generate(&openai, body).await;
    assert_eq!((r.status, r.error().0.as_str()), (400, "unknown_parameter"));
}
