#![allow(clippy::unwrap_used)]

mod support;

use base64::Engine;
use mockllm::{Answer, FakeOpenAi};
use serde_json::{Value, json};
use support::{Reply, post};

const KEY: (&str, &str) = ("authorization", "Bearer sk-test");

async fn embed(openai: &FakeOpenAi, body: Value) -> Reply {
    post(&format!("{}/embeddings", openai.base_url()), &[KEY], &body).await
}

fn lengths(v: &Value) -> Vec<usize> {
    let data = v["data"].as_array().unwrap();
    data.iter()
        .map(|d| d["embedding"].as_array().unwrap().len())
        .collect()
}

#[tokio::test]
async fn embeddings_have_openais_shape_and_model_lengths() {
    let openai = FakeOpenAi::start().await;
    let v = embed(
        &openai,
        json!({ "model": "text-embedding-3-small", "input": "hello world" }),
    )
    .await
    .json();
    assert_eq!(
        (&v["object"], &v["model"]),
        (&json!("list"), &json!("text-embedding-3-small"))
    );
    assert_eq!(
        (&v["data"][0]["object"], &v["data"][0]["index"]),
        (&json!("embedding"), &json!(0))
    );
    assert_eq!(lengths(&v), [1536]);
    assert_eq!(v["usage"], json!({ "prompt_tokens": 2, "total_tokens": 2 }));
    let v = embed(
        &openai,
        json!({ "model": "text-embedding-3-large", "input": ["a", "b"] }),
    )
    .await
    .json();
    assert_eq!(lengths(&v), [3072, 3072]);
    assert_ne!(v["data"][0]["embedding"], v["data"][1]["embedding"]);
    let v = embed(
        &openai,
        json!({ "model": "text-embedding-3-small", "input": "x", "dimensions": 8 }),
    )
    .await
    .json();
    assert_eq!(lengths(&v), [8]);
}

#[tokio::test]
async fn scripted_vector_and_base64_encoding() {
    let openai = FakeOpenAi::start().await;
    openai.answer(Answer::embedding(vec![0.5, -1.0]));
    openai.answer(Answer::embedding(vec![0.5, -1.0]));
    let v = embed(
        &openai,
        json!({ "model": "text-embedding-3-small", "input": "x" }),
    )
    .await
    .json();
    assert_eq!(v["data"][0]["embedding"], json!([0.5, -1.0]));
    let body =
        json!({ "model": "text-embedding-3-small", "input": "x", "encoding_format": "base64" });
    let v = embed(&openai, body).await.json();
    let b64 = v["data"][0]["embedding"].as_str().unwrap();
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .unwrap();
    let floats: Vec<f32> = bytes
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(floats, [0.5, -1.0]);
}

#[tokio::test]
async fn validation_follows_the_reference() {
    let openai = FakeOpenAi::start().await;
    let r = embed(&openai, json!({ "model": "text-embedding-3-small" })).await;
    assert_eq!(
        (r.status, r.error().0.as_str()),
        (400, "missing_required_parameter")
    );
    let r = embed(
        &openai,
        json!({ "model": "text-embedding-3-small", "input": "" }),
    )
    .await;
    assert_eq!(r.status, 400);
    let too_many = vec!["x"; 2049];
    let r = embed(
        &openai,
        json!({ "model": "text-embedding-3-small", "input": too_many }),
    )
    .await;
    assert_eq!(r.status, 400);
    let r = embed(
        &openai,
        json!({ "model": "text-embedding-ada-002", "input": "x", "dimensions": 8 }),
    )
    .await;
    assert_eq!(r.status, 400);
    assert!(
        r.error()
            .1
            .contains("does not support specifying dimensions")
    );
    let r = embed(
        &openai,
        json!({ "model": "m", "input": "x", "encoding_format": "int8" }),
    )
    .await;
    assert_eq!((r.status, r.error().0.as_str()), (400, "invalid_value"));
    let r = embed(&openai, json!({ "model": "m", "input": "x", "n": 2 })).await;
    assert_eq!(r.error().0, "unrecognized_request_argument");
}

#[tokio::test]
async fn an_answer_that_does_not_fit_is_a_clear_500() {
    let openai = FakeOpenAi::start().await;
    openai.answer(Answer::text("not a vector"));
    let r = embed(
        &openai,
        json!({ "model": "text-embedding-3-small", "input": "x" }),
    )
    .await;
    assert_eq!(r.status, 500);
    assert_eq!(r.body, "mockllm: embeddings cannot answer a text");
}
