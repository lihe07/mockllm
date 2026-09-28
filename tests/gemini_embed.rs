#![allow(clippy::unwrap_used)]

mod support;

use mockllm::{Answer, FakeGemini};
use serde_json::{Value, json};
use support::{Reply, post};

const KEY: (&str, &str) = ("x-goog-api-key", "AIza-test");

async fn embed(gemini: &FakeGemini, model: &str, method: &str, body: Value) -> Reply {
    let url = format!("{}/v1beta/models/{model}:{method}", gemini.base_url());
    post(&url, &[KEY], &body).await
}

fn text(t: &str) -> Value {
    json!({ "parts": [{ "text": t }] })
}

fn rejected(r: &Reply, says: &str) {
    assert_eq!(r.status, 400, "{}", r.body);
    assert_eq!(r.json()["error"]["status"], "INVALID_ARGUMENT");
    let message = r.json()["error"]["message"].as_str().unwrap().to_owned();
    assert!(message.contains(says), "{message}");
}

#[tokio::test]
async fn embed_content_answers_one_embedding() {
    let gemini = FakeGemini::start().await;
    let body = json!({ "model": "models/gemini-embedding-001", "content": text("what is life") });
    let v = embed(&gemini, "gemini-embedding-001", "embedContent", body)
        .await
        .json();
    // Unscripted: a 3072-long unit vector.
    let values = v["embedding"]["values"].as_array().unwrap();
    assert_eq!(values.len(), 3072);
    assert_eq!(
        (values[0].as_f64(), values[1].as_f64()),
        (Some(1.0), Some(0.0))
    );
    assert_eq!(v["usageMetadata"]["promptTokenCount"], 3);

    // `outputDimensionality`, in the config or in snake_case at the top.
    let body =
        json!({ "content": text("x"), "embedContentConfig": { "outputDimensionality": 768 } });
    let v = embed(&gemini, "gemini-embedding-2", "embedContent", body)
        .await
        .json();
    assert_eq!(v["embedding"]["values"].as_array().unwrap().len(), 768);
    let body = json!({ "content": text("x"), "output_dimensionality": 128 });
    let v = embed(&gemini, "gemini-embedding-2", "embedContent", body)
        .await
        .json();
    assert_eq!(v["embedding"]["values"].as_array().unwrap().len(), 128);

    gemini.answer(Answer::embedding(vec![0.5, -0.25]));
    let body = json!({ "content": text("x") });
    let v = embed(&gemini, "gemini-embedding-001", "embedContent", body)
        .await
        .json();
    assert_eq!(v["embedding"]["values"], json!([0.5, -0.25]));
}

#[tokio::test]
async fn batch_answers_an_embedding_per_request() {
    let gemini = FakeGemini::start().await;
    let one = |t: &str| json!({ "model": "models/gemini-embedding-001", "content": text(t) });
    let body = json!({ "requests": [one("a"), one("b"), one("c")] });
    let v = embed(&gemini, "gemini-embedding-001", "batchEmbedContents", body)
        .await
        .json();
    let embeddings = v["embeddings"].as_array().unwrap();
    assert_eq!(embeddings.len(), 3);
    assert_eq!(embeddings[1]["values"][1], 1.0);
    assert_eq!(v["usageMetadata"]["promptTokenCount"], 3);

    // Each request's model must be the path's.
    let other = json!({ "model": "models/gemini-embedding-2", "content": text("b") });
    let body = json!({ "requests": [one("a"), other] });
    let r = embed(&gemini, "gemini-embedding-001", "batchEmbedContents", body).await;
    rejected(&r, "requests[1].model");
    let body = json!({ "requests": [] });
    let r = embed(&gemini, "gemini-embedding-001", "batchEmbedContents", body).await;
    rejected(&r, "requests is not specified");
}

#[tokio::test]
async fn validation_follows_the_reference() {
    let gemini = FakeGemini::start().await;
    let m = "gemini-embedding-001";
    let r = embed(&gemini, m, "embedContent", json!({ "model": "models/x" })).await;
    rejected(&r, "content is not specified");
    let r = embed(
        &gemini,
        m,
        "embedContent",
        json!({ "content": text("x"), "input": "x" }),
    )
    .await;
    rejected(&r, "Unknown name \"input\"");
    let body = json!({ "content": text("x"), "embedContentConfig": { "taskType": "SEARCH" } });
    rejected(&embed(&gemini, m, "embedContent", body).await, "task_type");
    let body = json!({ "content": text("x"), "taskType": "RETRIEVAL_DOCUMENT", "title": "t" });
    assert_eq!(embed(&gemini, m, "embedContent", body).await.status, 200);
    let body =
        json!({ "content": text("x"), "embedContentConfig": { "outputDimensionality": 4096 } });
    rejected(
        &embed(&gemini, m, "embedContent", body).await,
        "between 128 and 3072",
    );
    // A model without the rule takes any size.
    let body = json!({ "content": text("x"), "outputDimensionality": 4 });
    let v = embed(&gemini, "my-embedder", "embedContent", body)
        .await
        .json();
    assert_eq!(v["embedding"]["values"].as_array().unwrap().len(), 4);
    // Auth, as every Gemini endpoint.
    let url = format!("{}/v1beta/models/{m}:embedContent", gemini.base_url());
    let r = post(&url, &[], &json!({ "content": text("x") })).await;
    assert_eq!(r.status, 403);
}

#[tokio::test]
async fn scripts_errors_and_misfits() {
    let gemini = FakeGemini::start().await;
    let body = json!({ "content": text("x") });
    gemini.answer(Answer::error(429).retry_after(3));
    let r = embed(
        &gemini,
        "gemini-embedding-001",
        "embedContent",
        body.clone(),
    )
    .await;
    assert_eq!((r.status, r.header("retry-after")), (429, Some("3")));
    assert_eq!(r.json()["error"]["status"], "RESOURCE_EXHAUSTED");
    gemini.answer(Answer::text("no"));
    let r = embed(
        &gemini,
        "gemini-embedding-001",
        "embedContent",
        body.clone(),
    )
    .await;
    assert_eq!(r.status, 500);
    assert!(r.body.contains("cannot answer a text"));
    gemini.answer(Answer::embedding(vec![1.0]).cut_after(1));
    let r = embed(&gemini, "gemini-embedding-001", "embedContent", body).await;
    assert!(r.ended.is_err());
}
