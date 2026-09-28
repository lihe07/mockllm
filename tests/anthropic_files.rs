#![allow(clippy::unwrap_used)]

mod anthropic_support;
mod support;

use anthropic_support::{call, get, rejected};
use mockllm::{Answer, FakeAnthropic};
use reqwest::multipart::{Form, Part};
use serde_json::{Value, json};
use support::Reply;

const BETA: (&str, &str) = ("anthropic-beta", "files-api-2025-04-14");

fn pdf(name: &str) -> Form {
    let part = Part::bytes(b"%PDF-1.4 fake".to_vec()).file_name(name.to_owned());
    Form::new().part("file", part.mime_str("application/pdf").unwrap())
}

async fn upload(anthropic: &FakeAnthropic, extra: &[(&str, &str)], form: Form) -> Reply {
    call(anthropic, "POST", "/v1/files", extra, None, Some(form)).await
}

async fn id_of(anthropic: &FakeAnthropic, form: Form) -> String {
    let v = upload(anthropic, &[], form).await.json();
    v["id"].as_str().unwrap().to_owned()
}

#[tokio::test]
async fn upload_retrieve_list_delete() {
    let anthropic = FakeAnthropic::start().await;
    let r = upload(&anthropic, &[], pdf("docs/report.pdf")).await;
    assert_eq!(r.status, 200);
    let v = r.json();
    let id = v["id"].as_str().unwrap().to_owned();
    assert!(id.starts_with("file_"));
    let expected = json!({
        "id": id, "type": "file", "filename": "report.pdf", "mime_type": "application/pdf",
        "size_bytes": 13, "created_at": v["created_at"], "downloadable": false, "expires_at": null,
    });
    assert_eq!(v, expected);
    assert!(v["created_at"].as_str().unwrap().ends_with('Z'));
    assert_eq!(
        get(&anthropic, &format!("/v1/files/{id}")).await.json(),
        expected
    );
    let list = get(&anthropic, "/v1/files").await.json();
    assert_eq!(list, json!({ "data": [expected], "next_page": null }));
    let del = call(
        &anthropic,
        "DELETE",
        &format!("/v1/files/{id}"),
        &[],
        None,
        None,
    )
    .await;
    assert_eq!(del.json(), json!({ "id": id, "type": "file_deleted" }));
    let r = get(&anthropic, &format!("/v1/files/{id}")).await;
    rejected(
        &r,
        404,
        "not_found_error",
        &format!("File `{id}` not found."),
    );
}

#[tokio::test]
async fn uploads_detect_type_name_and_expiry() {
    let anthropic = FakeAnthropic::start().await;
    // No part type: detected; no name: "unnamed" plus the extension.
    let form = Form::new().part(
        "file",
        Part::bytes(b"hello".to_vec()).file_name("notes.txt"),
    );
    let v = upload(&anthropic, &[], form).await.json();
    assert_eq!(v["mime_type"], "text/plain");
    let form = Form::new().part(
        "file",
        Part::bytes(b"x".to_vec())
            .file_name("")
            .mime_str("image/png")
            .unwrap(),
    );
    assert_eq!(
        upload(&anthropic, &[], form).await.json()["filename"],
        "unnamed.png"
    );
    let form = pdf("a.pdf").text("expires_in_seconds", "3600");
    let v = upload(&anthropic, &[], form).await.json();
    assert!(v["expires_at"].as_str().unwrap() > v["created_at"].as_str().unwrap());
}

#[tokio::test]
async fn upload_validation() {
    let anthropic = FakeAnthropic::start().await;
    let r = upload(&anthropic, &[], Form::new().text("purpose", "x")).await;
    rejected(&r, 400, "invalid_request_error", "file: Field required");
    for (secs, says) in [
        ("60", "greater than or equal to 3600"),
        ("7776001", "less than or equal to 7776000"),
    ] {
        let r = upload(
            &anthropic,
            &[],
            pdf("a.pdf").text("expires_in_seconds", secs),
        )
        .await;
        rejected(&r, 400, "invalid_request_error", says);
    }
    let r = upload(&anthropic, &[], pdf("what?.pdf")).await;
    rejected(&r, 400, "invalid_request_error", "Invalid filename");
    // The beta requires the part's Content-Type.
    let bare = Form::new().part("file", Part::bytes(b"x".to_vec()).file_name("a.txt"));
    let r = upload(&anthropic, &[BETA], bare).await;
    rejected(&r, 400, "invalid_request_error", "Content-Type");
    assert!(
        get(&anthropic, "/v1/files").await.json()["data"]
            .as_array()
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn ga_list_pages_and_ids() {
    let anthropic = FakeAnthropic::start().await;
    let mut ids = Vec::new();
    for i in 0..3 {
        ids.push(id_of(&anthropic, pdf(&format!("{i}.pdf"))).await);
    }
    let v = get(&anthropic, "/v1/files?limit=2").await.json();
    assert_eq!(v["data"][0]["id"], json!(ids[2]));
    let next = v["next_page"].as_str().unwrap();
    assert!(next.starts_with("page_"));
    let v = get(&anthropic, &format!("/v1/files?limit=2&page={next}"))
        .await
        .json();
    assert_eq!(
        (&v["data"][0]["id"], &v["next_page"]),
        (&json!(ids[0]), &Value::Null)
    );
    let v = get(
        &anthropic,
        &format!("/v1/files?ids%5B%5D={}&ids%5B%5D=file_gone", ids[1]),
    )
    .await
    .json();
    assert_eq!(v["data"].as_array().unwrap().len(), 1);
    assert_eq!(v["data"][0]["id"], json!(ids[1]));
    let r = get(&anthropic, &format!("/v1/files?after_id={}", ids[1])).await;
    rejected(&r, 400, "invalid_request_error", "after_id");
    let r = get(&anthropic, &format!("/v1/files?ids[]={}&limit=1", ids[1])).await;
    rejected(&r, 400, "invalid_request_error", "ids");
}

#[tokio::test]
async fn beta_header_keeps_beta_shapes() {
    let anthropic = FakeAnthropic::start().await;
    let a = upload(&anthropic, &[BETA], pdf("a.pdf")).await.json();
    assert!(a.get("expires_at").is_none());
    let b = id_of(&anthropic, pdf("b.pdf")).await;
    let list = call(&anthropic, "GET", "/v1/files?limit=1", &[BETA], None, None)
        .await
        .json();
    assert_eq!(
        (&list["data"][0]["id"], &list["has_more"]),
        (&json!(b), &json!(true))
    );
    assert_eq!(
        (&list["first_id"], &list["last_id"]),
        (&json!(b), &json!(b))
    );
    assert!(list.get("next_page").is_none());
    let path = format!("/v1/files?after_id={b}");
    let list = call(&anthropic, "GET", &path, &[BETA], None, None)
        .await
        .json();
    assert_eq!(list["data"][0]["id"], a["id"]);
    assert!(list["data"][0].get("expires_at").is_none());
}

#[tokio::test]
async fn uploaded_files_are_not_downloadable() {
    let anthropic = FakeAnthropic::start().await;
    let id = id_of(&anthropic, pdf("a.pdf")).await;
    let r = get(&anthropic, &format!("/v1/files/{id}/content")).await;
    rejected(&r, 400, "invalid_request_error", "not downloadable");
    let r = get(&anthropic, "/v1/files/file_nope/content").await;
    rejected(&r, 404, "not_found_error", "file_nope");
}

#[tokio::test]
async fn answer_kinds_and_scripts() {
    let anthropic = FakeAnthropic::start().await;
    anthropic.answer(Answer::text("no"));
    assert_eq!(upload(&anthropic, &[], pdf("a.pdf")).await.status, 500);
    anthropic.answer(Answer::error(500));
    rejected(
        &upload(&anthropic, &[], pdf("a.pdf")).await,
        500,
        "api_error",
        "Internal",
    );
    // Neither stored a file.
    assert_eq!(get(&anthropic, "/v1/files").await.json()["data"], json!([]));
    anthropic.answer(Answer::default().delay_first(120).cut_after(1));
    let t = std::time::Instant::now();
    let r = get(&anthropic, "/v1/files").await;
    assert!(t.elapsed().as_millis() >= 120);
    assert!(r.ended.is_err());
}
