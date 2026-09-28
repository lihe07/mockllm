//! Any-method requests to a mock Anthropic, with its auth headers.

#![allow(dead_code, clippy::unwrap_used)]

use futures::StreamExt;
use mockllm::FakeAnthropic;
use serde_json::Value;

use crate::support::Reply;

pub const HEADERS: &[(&str, &str)] = &[
    ("x-api-key", "sk-ant-test"),
    ("anthropic-version", "2023-06-01"),
];

/// Sends `method path` with the auth headers plus `extra`, and a JSON
/// `body` or a multipart `form`, reading the whole reply.
pub async fn call(
    anthropic: &FakeAnthropic,
    method: &str,
    path: &str,
    extra: &[(&str, &str)],
    body: Option<&Value>,
    form: Option<reqwest::multipart::Form>,
) -> Reply {
    let client = reqwest::Client::new();
    let url = format!("{}{path}", anthropic.base_url());
    let mut req = client.request(method.parse().unwrap(), url);
    for (k, v) in HEADERS.iter().chain(extra) {
        req = req.header(*k, *v);
    }
    if let Some(b) = body {
        req = req
            .header("content-type", "application/json")
            .body(b.to_string());
    }
    if let Some(f) = form {
        req = req.multipart(f);
    }
    let resp = req.send().await.unwrap();
    let (status, headers) = (resp.status().as_u16(), resp.headers().clone());
    let (mut out, mut ended) = (Vec::new(), Ok(()));
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(b) => out.extend_from_slice(&b),
            Err(e) => {
                ended = Err(format!("{e:?}"));
                break;
            }
        }
    }
    let body = String::from_utf8_lossy(&out).into_owned();
    Reply {
        status,
        headers,
        body,
        ended,
    }
}

pub async fn get(anthropic: &FakeAnthropic, path: &str) -> Reply {
    call(anthropic, "GET", path, &[], None, None).await
}

pub async fn post(anthropic: &FakeAnthropic, path: &str, body: &Value) -> Reply {
    call(anthropic, "POST", path, &[], Some(body), None).await
}

/// Asserts an error of `status` and `kind` in Anthropic's shape whose
/// message contains `says`.
pub fn rejected(r: &Reply, status: u16, kind: &str, says: &str) {
    assert_eq!(r.status, status, "{}", r.body);
    let v = r.json();
    assert_eq!(v["type"], "error");
    assert!(v["request_id"].as_str().unwrap().starts_with("req_"));
    let (k, message) = r.error();
    assert_eq!(k, kind, "{message}");
    assert!(message.contains(says), "{message}");
}
