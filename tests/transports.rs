//! The core's transports, through a provider written outside the crate:
//! multipart requests, whole and chunked binary, named SSE, and the
//! timing and breaks on each.

#![allow(clippy::unwrap_used)]

mod support;

use std::time::{Duration, Instant};

use bytes::Bytes;
use mockllm::{Answer, Ctx, Endpoint, Mock, Out, Provider, Rejection, Reply, Request, event};
use serde_json::{Value, json};

#[derive(Debug, Default)]
struct Test;

impl Provider for Test {
    const BASE_PATH: &'static str = "";

    fn endpoints() -> Vec<Box<dyn Endpoint<Self>>> {
        vec![Box::new(Upload), Box::new(Speech), Box::new(Named)]
    }

    fn authorize(&self, _: &Request) -> Result<(), Rejection> {
        Ok(())
    }

    fn error(&self, r: &Rejection, _: u64) -> Value {
        json!({ "error": r.message })
    }

    fn scripted(&self, status: u16) -> Rejection {
        Rejection::new(status, "scripted")
    }
}

/// Multipart in, JSON out: what was uploaded.
struct Upload;

impl Endpoint<Test> for Upload {
    fn route(&self) -> (&'static str, &'static str) {
        ("POST", "/upload")
    }

    fn validate(&self, _: &Test, req: &Request) -> Result<(), Rejection> {
        match req.body["model"].as_str() {
            Some(_) => Ok(()),
            None => Err(Rejection::new(400, "model is required")),
        }
    }

    fn respond(&self, _: &Test, req: &Request, _: &Answer, _: &Ctx) -> Out {
        let f = &req.files[0];
        let file = json!({ "field": f.field, "filename": f.filename, "type": f.content_type, "size": f.bytes.len() });
        Out::Json(200, json!({ "fields": req.body, "file": file }))
    }
}

/// Audio bytes, whole or chunked with `?chunked`.
struct Speech;

impl Endpoint<Test> for Speech {
    fn route(&self) -> (&'static str, &'static str) {
        ("POST", "/speech")
    }

    fn validate(&self, _: &Test, _: &Request) -> Result<(), Rejection> {
        Ok(())
    }

    fn respond(&self, _: &Test, req: &Request, answer: &Answer, _: &Ctx) -> Out {
        let Reply::Audio(bytes) = answer.reply() else {
            return Out::Misfit(format!("speech cannot answer {}", answer.reply().kind()));
        };
        if req.query_param("chunked").is_none() {
            return Out::Bytes("audio/mpeg".into(), bytes.clone());
        }
        let chunks = bytes.chunks(2).map(Bytes::copy_from_slice).collect();
        Out::Chunks("audio/mpeg".into(), chunks)
    }
}

/// Named SSE events, one per word.
struct Named;

impl Endpoint<Test> for Named {
    fn route(&self) -> (&'static str, &'static str) {
        ("POST", "/named")
    }

    fn validate(&self, _: &Test, _: &Request) -> Result<(), Rejection> {
        Ok(())
    }

    fn respond(&self, _: &Test, _: &Request, _: &Answer, _: &Ctx) -> Out {
        let frames = ["a", "b", "c"].map(|w| event("word", &json!({ "w": w })));
        Out::Sse(frames.to_vec(), event("oops", &json!({})))
    }
}

async fn post_raw(url: &str) -> reqwest::Response {
    reqwest::Client::new().post(url).send().await.unwrap()
}

#[tokio::test]
async fn multipart_bodies_are_split_into_fields_and_files() {
    let mock = Mock::serve(Test).await;
    let file = reqwest::multipart::Part::bytes(vec![1, 2, 3])
        .file_name("a.wav")
        .mime_str("audio/wav")
        .unwrap();
    let form = reqwest::multipart::Form::new()
        .text("model", "whisper-1")
        .part("file", file);
    let url = format!("{}/upload", mock.base_url());
    let r = reqwest::Client::new()
        .post(&url)
        .multipart(form)
        .send()
        .await
        .unwrap();
    let v: Value = serde_json::from_slice(&r.bytes().await.unwrap()).unwrap();
    assert_eq!(v["fields"], json!({ "model": "whisper-1" }));
    assert_eq!(
        v["file"],
        json!({ "field": "file", "filename": "a.wav", "type": "audio/wav", "size": 3 })
    );
    let form = reqwest::multipart::Form::new().text("other", "x");
    let r = reqwest::Client::new()
        .post(&url)
        .multipart(form)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 400);
    assert_eq!(mock.requests()[0].files.len(), 1);
}

#[tokio::test]
async fn binary_bodies_whole_and_chunked() {
    let mock = Mock::serve(Test).await;
    mock.answer(Answer::audio(b"ID3abcdef".to_vec()));
    let r = post_raw(&format!("{}/speech", mock.base_url())).await;
    assert_eq!(r.headers()["content-type"], "audio/mpeg");
    assert_eq!(r.bytes().await.unwrap().as_ref(), b"ID3abcdef");

    mock.answer(Answer::audio(b"ID3abcdef".to_vec()).pace(40));
    let t = Instant::now();
    let r = post_raw(&format!("{}/speech?chunked", mock.base_url())).await;
    assert_eq!(r.bytes().await.unwrap().as_ref(), b"ID3abcdef");
    assert!(
        t.elapsed() >= Duration::from_millis(160),
        "5 chunks, 4 gaps: {:?}",
        t.elapsed()
    );

    mock.answer(Answer::audio(b"ID3abcdef".to_vec()).cut_after(2));
    let r = post_raw(&format!("{}/speech?chunked", mock.base_url())).await;
    assert!(r.bytes().await.is_err(), "the body should break off");

    mock.answer(Answer::text("not audio"));
    let r = post_raw(&format!("{}/speech", mock.base_url())).await;
    assert_eq!(r.status(), 500);
    assert_eq!(r.text().await.unwrap(), "speech cannot answer a text");
}

#[tokio::test]
async fn named_sse_with_error_after() {
    let mock = Mock::serve(Test).await;
    let url = format!("{}/named", mock.base_url());
    let r = support::post(&url, &[], &json!({})).await;
    assert_eq!(r.events(), ["word", "word", "word"]);
    mock.answer(Answer::default().error_after(1));
    let r = support::post(&url, &[], &json!({})).await;
    assert_eq!(r.events(), ["word", "oops"]);
}
