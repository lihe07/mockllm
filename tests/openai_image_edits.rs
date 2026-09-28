#![allow(clippy::unwrap_used)]

mod support;

use base64::Engine;
use mockllm::{Answer, FakeOpenAi};
use reqwest::multipart::{Form, Part};
use serde_json::json;
use support::{Reply, post};

const KEY: (&str, &str) = ("authorization", "Bearer sk-test");

/// POSTs a multipart edit of `fields` and `files` (field, filename).
async fn edit(openai: &FakeOpenAi, fields: &[(&str, &str)], files: &[(&str, &str)]) -> Reply {
    let mut form = Form::new();
    for (k, v) in fields {
        form = form.text(k.to_string(), v.to_string());
    }
    for (field, name) in files {
        let part = Part::bytes(b"\x89PNG fake".to_vec()).file_name(name.to_string());
        form = form.part(field.to_string(), part);
    }
    let resp = reqwest::Client::new()
        .post(format!("{}/images/edits", openai.base_url()))
        .header(KEY.0, KEY.1)
        .multipart(form)
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

/// Form fields, or file parts (field, filename).
type Pairs<'a> = &'a [(&'a str, &'a str)];

const PNG: Pairs = &[("image", "in.png")];

#[tokio::test]
async fn gpt_image_edits_answer_base64_with_usage() {
    let openai = FakeOpenAi::start().await;
    openai.answer(Answer::image(b"edited".to_vec()));
    let r = edit(&openai, &[("prompt", "add a hat"), ("n", "2")], PNG).await;
    assert_eq!(r.status, 200, "{}", r.body);
    let v = r.json();
    let data = v["data"].as_array().unwrap();
    assert_eq!(data.len(), 2);
    let b64 = data[0]["b64_json"].as_str().unwrap();
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .unwrap();
    assert_eq!(bytes, b"edited");
    assert!(v["usage"]["total_tokens"].is_u64());
    // Up to 16 images, as `image[]`, and a PNG mask.
    let files = [
        ("image[]", "a.png"),
        ("image[]", "b.webp"),
        ("mask", "m.png"),
    ];
    let r = edit(&openai, &[("prompt", "merge")], &files).await;
    assert_eq!(r.status, 200, "{}", r.body);
    // The JSON form, with image references.
    let url = format!("{}/images/edits", openai.base_url());
    let body = json!({ "prompt": "x", "images": [{ "image_url": "https://example.com/a.png" }] });
    assert_eq!(post(&url, &[KEY], &body).await.status, 200);
}

#[tokio::test]
async fn dall_e_2_edits_answer_urls() {
    let openai = FakeOpenAi::start().await;
    let fields = [
        ("prompt", "add a hat"),
        ("model", "dall-e-2"),
        ("size", "256x256"),
    ];
    let v = edit(&openai, &fields, PNG).await.json();
    let url = v["data"][0]["url"].as_str().unwrap();
    let image = reqwest::get(url).await.unwrap();
    assert_eq!(image.headers()["content-type"], "image/png");
    assert!(v.get("usage").is_none());
}

#[tokio::test]
async fn stream_sends_named_partial_images_then_completed() {
    let openai = FakeOpenAi::start().await;
    let fields = [
        ("prompt", "add a hat"),
        ("stream", "true"),
        ("partial_images", "2"),
    ];
    let r = edit(&openai, &fields, PNG).await;
    assert!(
        r.header("content-type")
            .unwrap()
            .starts_with("text/event-stream")
    );
    let names = r.events();
    let want = [
        "image_edit.partial_image",
        "image_edit.partial_image",
        "image_edit.completed",
    ];
    assert_eq!(names, want);
    let events = r.chunks();
    assert_eq!(events[1]["partial_image_index"], 1);
    assert_eq!(events[1]["size"], "1024x1024");
    assert!(events[2]["usage"]["total_tokens"].is_u64());
    assert!(events[2]["b64_json"].is_string());
    openai.answer(Answer::image(b"x".to_vec()).error_after(1));
    let r = edit(&openai, &fields, PNG).await;
    assert_eq!(r.events(), ["image_edit.partial_image", "error"]);
}

#[tokio::test]
async fn validation_follows_the_reference() {
    let openai = FakeOpenAi::start().await;
    let dalle = ("model", "dall-e-2");
    let two = &[("image[]", "a.png"), ("image[]", "b.png")][..];
    let cases: &[(Pairs, Pairs, &str)] = &[
        (&[("prompt", "x")], &[], "missing_required_parameter"),
        (&[], PNG, "missing_required_parameter"),
        (&[("prompt", "x"), dalle], two, "array_above_max_length"),
        (
            &[("prompt", "x"), dalle],
            &[("image", "a.jpg")],
            "invalid_value",
        ),
        (&[("prompt", "x")], &[("image", "a.gif")], "invalid_value"),
        (
            &[("prompt", "x")],
            &[("image", "a.png"), ("mask", "m.jpg")],
            "invalid_value",
        ),
        (
            &[("prompt", "x"), ("model", "dall-e-3")],
            PNG,
            "invalid_value",
        ),
        (
            &[("prompt", "x"), ("response_format", "url")],
            PNG,
            "unknown_parameter",
        ),
        (
            &[("prompt", "x"), dalle, ("size", "1536x1024")],
            PNG,
            "invalid_value",
        ),
        (&[("prompt", "x"), ("n", "11")], PNG, "invalid_value"),
        (
            &[("prompt", "x"), dalle, ("stream", "true")],
            PNG,
            "unsupported_value",
        ),
        (
            &[("prompt", "x"), ("partial_images", "4")],
            PNG,
            "integer_above_max_value",
        ),
        (
            &[("prompt", "x"), ("style", "vivid")],
            PNG,
            "unrecognized_request_argument",
        ),
    ];
    for (fields, files, code) in cases {
        let r = edit(&openai, fields, files).await;
        assert_eq!(
            (r.status, r.error().0.as_str()),
            (400, *code),
            "{fields:?} {files:?}"
        );
    }
    let long = "x".repeat(1001);
    let r = edit(&openai, &[("prompt", &long), dalle], PNG).await;
    assert_eq!(r.error().0, "string_above_max_length");
    // JSON edits are for GPT image models only.
    let url = format!("{}/images/edits", openai.base_url());
    let body = json!({ "prompt": "x", "model": "dall-e-2", "images": [{ "file_id": "file-1" }] });
    assert_eq!(post(&url, &[KEY], &body).await.status, 400);
}
