#![allow(clippy::unwrap_used)]

mod support;

use std::time::Instant;

use mockllm::{Answer, FakeOpenAi};
use reqwest::multipart::{Form, Part};
use serde_json::{Value, json};
use support::Reply;

const KEY: &str = "Bearer sk-test";

async fn send(req: reqwest::RequestBuilder) -> Reply {
    let resp = req.header("authorization", KEY).send().await.unwrap();
    let (status, headers) = (resp.status().as_u16(), resp.headers().clone());
    let (body, ended) = match resp.bytes().await {
        Ok(b) => (String::from_utf8_lossy(&b).into_owned(), Ok(())),
        Err(e) => (String::new(), Err(format!("{e:?}"))),
    };
    Reply {
        status,
        headers,
        body,
        ended,
    }
}

async fn upload(openai: &FakeOpenAi, name: &str, purpose: &str, bytes: &[u8]) -> Reply {
    let form = Form::new().text("purpose", purpose.to_owned()).part(
        "file",
        Part::bytes(bytes.to_vec()).file_name(name.to_owned()),
    );
    let url = format!("{}/files", openai.base_url());
    send(reqwest::Client::new().post(url).multipart(form)).await
}

async fn get(openai: &FakeOpenAi, path: &str) -> Reply {
    let url = format!("{}{path}", openai.base_url());
    send(reqwest::Client::new().get(url)).await
}

#[tokio::test]
async fn upload_list_retrieve_content_delete() {
    let openai = FakeOpenAi::start().await;
    let f = upload(&openai, "notes.txt", "user_data", b"hello")
        .await
        .json();
    assert_eq!(f["object"], "file");
    assert_eq!(f["bytes"], 5);
    assert_eq!(f["filename"], "notes.txt");
    assert_eq!(f["purpose"], "user_data");
    assert_eq!(f["expires_at"], Value::Null);
    let id = f["id"].as_str().unwrap().to_owned();

    // Batch files expire after 30 days by default.
    let b = upload(&openai, "in.jsonl", "batch", b"{}\n").await.json();
    let lifetime = b["expires_at"].as_u64().unwrap() - b["created_at"].as_u64().unwrap();
    assert_eq!(lifetime, 2_592_000);

    let list = get(&openai, "/files").await.json();
    assert_eq!(list["object"], "list");
    assert_eq!(list["data"].as_array().unwrap().len(), 2);
    assert_eq!(list["first_id"], b["id"], "newest first");
    assert_eq!(list["has_more"], false);
    let asc = get(&openai, "/files?order=asc&limit=1").await.json();
    assert_eq!(asc["data"][0]["id"], json!(id));
    assert_eq!(asc["has_more"], true);
    let batch = get(&openai, "/files?purpose=batch").await.json();
    assert_eq!(batch["data"].as_array().unwrap().len(), 1);

    assert_eq!(get(&openai, &format!("/files/{id}")).await.json(), f);
    let content = get(&openai, &format!("/files/{id}/content")).await;
    assert_eq!((content.status, content.body.as_str()), (200, "hello"));

    let url = format!("{}/files/{id}", openai.base_url());
    let deleted = send(reqwest::Client::new().delete(url)).await.json();
    assert_eq!(
        deleted,
        json!({ "id": id, "object": "file", "deleted": true })
    );
    let gone = get(&openai, &format!("/files/{id}")).await;
    assert_eq!(gone.status, 404);
    assert_eq!(gone.error().1, format!("No such File object: {id}"));
}

#[tokio::test]
async fn validation_follows_the_reference() {
    let openai = FakeOpenAi::start().await;
    let r = upload(&openai, "a.txt", "nonsense", b"x").await;
    assert_eq!((r.status, r.error().0.as_str()), (400, "invalid_value"));
    let r = upload(&openai, "a.txt", "batch", b"x").await;
    assert_eq!(r.status, 400);
    assert!(r.error().1.contains(".jsonl"));
    let r = upload(&openai, "a.csv", "fine-tune", b"x").await;
    assert_eq!(r.status, 400);

    let url = format!("{}/files", openai.base_url());
    let no_file = Form::new().text("purpose", "batch");
    let r = send(reqwest::Client::new().post(&url).multipart(no_file)).await;
    assert_eq!(
        (r.status, r.error().0.as_str()),
        (400, "missing_required_parameter")
    );
    let short = Form::new()
        .text("purpose", "user_data")
        .text("expires_after[anchor]", "created_at")
        .text("expires_after[seconds]", "60")
        .part("file", Part::bytes(b"x".to_vec()).file_name("a.txt"));
    let r = send(reqwest::Client::new().post(&url).multipart(short)).await;
    assert_eq!(r.json()["error"]["param"], "expires_after.seconds");
    let ok = Form::new()
        .text("purpose", "user_data")
        .text("expires_after[anchor]", "created_at")
        .text("expires_after[seconds]", "3600")
        .part("file", Part::bytes(b"x".to_vec()).file_name("a.txt"));
    let f = send(reqwest::Client::new().post(&url).multipart(ok))
        .await
        .json();
    assert_eq!(
        f["expires_at"].as_u64().unwrap() - f["created_at"].as_u64().unwrap(),
        3600
    );

    let r = get(&openai, "/files?limit=0").await;
    assert_eq!(r.status, 400);
    let r = get(&openai, "/files?order=up").await;
    assert_eq!(r.status, 400);
    let r = get(&openai, "/files/file-nope/content").await;
    assert_eq!(r.status, 404);
    // Auth, as on every endpoint.
    let r = reqwest::get(format!("{url}/file-nope")).await.unwrap();
    assert_eq!(r.status(), 401);
    // A failed request takes no answer.
    assert!(openai.requests().len() >= 9);
}

#[tokio::test]
async fn scripts_apply() {
    let openai = FakeOpenAi::start().await;
    openai.answer(Answer::error(429).retry_after(3));
    let r = upload(&openai, "a.txt", "user_data", b"x").await;
    assert_eq!((r.status, r.header("retry-after")), (429, Some("3")));
    assert_eq!(get(&openai, "/files").await.json()["data"], json!([]));

    let f = upload(&openai, "a.txt", "user_data", b"0123456789")
        .await
        .json();
    let path = format!("/files/{}/content", f["id"].as_str().unwrap());
    openai.answer(Answer::default().delay_first(200));
    let start = Instant::now();
    assert_eq!(get(&openai, &path).await.body, "0123456789");
    assert!(start.elapsed().as_millis() >= 200);
    // A cut whole body breaks off halfway.
    openai.answer(Answer::default().cut_after(0));
    assert!(get(&openai, &path).await.ended.is_err());
    // An answer files cannot frame.
    openai.answer(Answer::text("hi"));
    assert_eq!(get(&openai, "/files").await.status, 500);
}
