#![allow(clippy::unwrap_used)]

mod support;

use mockllm::{Answer, FakeOpenAi};
use reqwest::multipart::{Form, Part};
use serde_json::{Value, json};
use support::{Reply, post};

const KEY: (&str, &str) = ("authorization", "Bearer sk-test");

async fn upload(openai: &FakeOpenAi, lines: &[Value], purpose: &str) -> String {
    let text: String = lines.iter().map(|l| format!("{l}\n")).collect();
    let form = Form::new()
        .text("purpose", purpose.to_owned())
        .part("file", Part::text(text).file_name("in.jsonl"));
    let url = format!("{}/files", openai.base_url());
    let resp = reqwest::Client::new()
        .post(url)
        .header(KEY.0, KEY.1)
        .multipart(form)
        .send()
        .await
        .unwrap();
    let f: Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
    f["id"].as_str().unwrap().to_owned()
}

async fn get(openai: &FakeOpenAi, path: &str) -> Reply {
    let resp = reqwest::Client::new()
        .get(format!("{}{path}", openai.base_url()))
        .header(KEY.0, KEY.1)
        .send()
        .await
        .unwrap();
    let (status, headers) = (resp.status().as_u16(), resp.headers().clone());
    let body = resp.text().await.unwrap();
    Reply {
        status,
        headers,
        body,
        ended: Ok(()),
    }
}

async fn create(openai: &FakeOpenAi, body: Value) -> Reply {
    post(&format!("{}/batches", openai.base_url()), &[KEY], &body).await
}

fn chat(id: &str, content: &str) -> Value {
    json!({ "custom_id": id, "method": "POST", "url": "/v1/chat/completions",
            "body": { "model": "gpt-4o", "messages": [{ "role": "user", "content": content }] } })
}

fn batch(file: &str) -> Value {
    json!({ "input_file_id": file, "endpoint": "/v1/chat/completions", "completion_window": "24h" })
}

/// The batch after one more retrieve.
async fn poll(openai: &FakeOpenAi, id: &str) -> Value {
    get(openai, &format!("/batches/{id}")).await.json()
}

#[tokio::test]
async fn a_batch_runs_to_its_output_file() {
    let openai = FakeOpenAi::start().await;
    // A request the chat endpoint rejects goes to the error file.
    let bad = json!({ "custom_id": "r3", "method": "POST", "url": "/v1/chat/completions",
                      "body": { "model": "gpt-4o", "messages": [] } });
    let file = upload(&openai, &[chat("r1", "hi"), chat("r2", "yo"), bad], "batch").await;
    openai.answer(Answer::text("Hello."));
    let mut body = batch(&file);
    body["metadata"] = json!({ "job": "nightly" });
    let b = create(&openai, body).await.json();
    assert_eq!(b["object"], "batch");
    assert_eq!(b["status"], "validating");
    assert_eq!(
        b["request_counts"],
        json!({ "total": 0, "completed": 0, "failed": 0 })
    );
    assert_eq!(b["metadata"], json!({ "job": "nightly" }));
    assert_eq!(
        b["expires_at"].as_u64().unwrap() - b["created_at"].as_u64().unwrap(),
        86_400
    );
    let id = b["id"].as_str().unwrap();

    let b = poll(&openai, id).await;
    assert_eq!(
        (b["status"].as_str(), &b["request_counts"]["total"]),
        (Some("in_progress"), &json!(3))
    );
    assert_eq!(b["model"], "gpt-4o");
    let b = poll(&openai, id).await;
    assert_eq!(b["status"], "finalizing");
    assert_eq!(
        b["request_counts"],
        json!({ "total": 3, "completed": 2, "failed": 1 })
    );
    let b = poll(&openai, id).await;
    assert_eq!(b["status"], "completed");
    assert!(b["completed_at"].is_u64());
    assert!(b["usage"]["total_tokens"].as_u64().unwrap() > 0);

    let out = get(
        &openai,
        &format!("/files/{}/content", b["output_file_id"].as_str().unwrap()),
    )
    .await;
    let lines: Vec<Value> = out
        .body
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[0]["custom_id"], "r1");
    assert_eq!(lines[0]["error"], Value::Null);
    let response = &lines[0]["response"];
    assert_eq!(response["status_code"], 200);
    assert!(response["request_id"].as_str().unwrap().starts_with("req_"));
    assert_eq!(response["body"]["object"], "chat.completion");
    assert_eq!(
        response["body"]["choices"][0]["message"]["content"],
        "Hello."
    );

    let errors = get(
        &openai,
        &format!("/files/{}/content", b["error_file_id"].as_str().unwrap()),
    )
    .await;
    let line: Value = serde_json::from_str(errors.body.trim()).unwrap();
    assert_eq!(line["custom_id"], "r3");
    assert_eq!(line["response"]["status_code"], 400);
    assert_eq!(line["response"]["body"]["error"]["code"], "empty_array");

    let output = get(
        &openai,
        &format!("/files/{}", b["output_file_id"].as_str().unwrap()),
    )
    .await
    .json();
    assert_eq!(output["purpose"], "batch_output");

    let list = get(&openai, "/batches?limit=1").await.json();
    assert_eq!(list["data"][0]["id"], json!(id));
    assert_eq!(list["has_more"], false);
}

#[tokio::test]
async fn bad_input_lines_fail_the_batch() {
    let openai = FakeOpenAi::start().await;
    let mut other = chat("r1", "hi");
    other["body"]["model"] = json!("gpt-4.1");
    let mut embed = chat("r3", "x");
    embed["url"] = json!("/v1/embeddings");
    let text = [chat("r1", "hi"), other, embed]
        .map(|l| l.to_string())
        .join("\n")
        + "\nnot json\n";
    let form = Form::new()
        .text("purpose", "batch")
        .part("file", Part::text(text).file_name("in.jsonl"));
    let resp = reqwest::Client::new()
        .post(format!("{}/files", openai.base_url()))
        .header(KEY.0, KEY.1)
        .multipart(form)
        .send()
        .await
        .unwrap();
    let file: Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
    let b = create(&openai, batch(file["id"].as_str().unwrap()))
        .await
        .json();
    let b = poll(&openai, b["id"].as_str().unwrap()).await;
    assert_eq!(b["status"], "failed");
    assert!(b["failed_at"].is_u64());
    let codes: Vec<(&str, u64)> = b["errors"]["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| (e["code"].as_str().unwrap(), e["line"].as_u64().unwrap()))
        .collect();
    assert_eq!(
        codes,
        [
            ("duplicate_custom_id", 2),
            ("model_mismatch", 2),
            ("url_mismatch", 3),
            ("invalid_json_line", 4)
        ]
    );
    assert_eq!(b["errors"]["object"], "list");
    // An empty file.
    let empty = upload(&openai, &[], "batch").await;
    let b = create(&openai, batch(&empty)).await.json();
    let b = poll(&openai, b["id"].as_str().unwrap()).await;
    assert_eq!(b["errors"]["data"][0]["code"], "empty_file");
}

#[tokio::test]
async fn cancel() {
    let openai = FakeOpenAi::start().await;
    let file = upload(&openai, &[chat("r1", "hi")], "batch").await;
    let b = create(&openai, batch(&file)).await.json();
    let id = b["id"].as_str().unwrap();
    let url = format!("{}/batches/{id}/cancel", openai.base_url());
    let b = post(&url, &[KEY], &json!(null)).await.json();
    assert_eq!(b["status"], "cancelling");
    assert!(b["cancelling_at"].is_u64());
    let b = poll(&openai, id).await;
    assert_eq!(b["status"], "cancelled");
    assert_eq!(b["output_file_id"], Value::Null);
    let r = post(&url, &[KEY], &json!(null)).await;
    assert_eq!(r.status, 409);
}

#[tokio::test]
async fn validation_follows_the_reference() {
    let openai = FakeOpenAi::start().await;
    let file = upload(&openai, &[chat("r1", "hi")], "batch").await;
    let other = upload(&openai, &[chat("r1", "hi")], "user_data").await;
    let r = create(
        &openai,
        json!({ "endpoint": "/v1/chat/completions", "completion_window": "24h" }),
    )
    .await;
    assert_eq!(
        (r.status, r.error().0.as_str()),
        (400, "missing_required_parameter")
    );
    let mut body = batch(&file);
    body["completion_window"] = json!("48h");
    assert_eq!(create(&openai, body).await.error().0, "invalid_value");
    let mut body = batch(&file);
    body["endpoint"] = json!("/v1/audio/speech");
    assert_eq!(create(&openai, body).await.status, 400);
    assert_eq!(create(&openai, batch("file-nope")).await.status, 404);
    assert_eq!(create(&openai, batch(&other)).await.status, 400);
    let mut body = batch(&file);
    body["metadata"] = json!({ "k": "v".repeat(513) });
    assert_eq!(
        create(&openai, body).await.json()["error"]["param"],
        "metadata"
    );
    let mut body = batch(&file);
    body["output_expires_after"] = json!({ "anchor": "created_at", "seconds": 60 });
    assert_eq!(create(&openai, body).await.status, 400);
    let mut body = batch(&file);
    body["extra"] = json!(1);
    assert_eq!(
        create(&openai, body).await.error().0,
        "unrecognized_request_argument"
    );
    assert_eq!(get(&openai, "/batches/batch_nope").await.status, 404);
    assert_eq!(get(&openai, "/batches?limit=101").await.status, 400);
}

#[tokio::test]
async fn scripts_apply() {
    let openai = FakeOpenAi::start().await;
    let file = upload(&openai, &[chat("r1", "hi")], "batch").await;
    openai.answer(Answer::error(500));
    let r = create(&openai, batch(&file)).await;
    assert_eq!(
        (r.status, r.json()["error"]["type"].as_str()),
        (500, Some("server_error"))
    );
    assert_eq!(get(&openai, "/batches").await.json()["data"], json!([]));
    // A failed poll does not move the batch on.
    let b = create(&openai, batch(&file)).await.json();
    let id = b["id"].as_str().unwrap();
    openai.answer(Answer::error(429));
    assert_eq!(get(&openai, &format!("/batches/{id}")).await.status, 429);
    assert_eq!(poll(&openai, id).await["status"], "in_progress");
    openai.answer(Answer::default().cut_after(0));
    let url = format!("{}/batches/{id}", openai.base_url());
    let resp = reqwest::Client::new()
        .get(url)
        .header(KEY.0, KEY.1)
        .send()
        .await
        .unwrap();
    assert!(resp.text().await.is_err());
}
