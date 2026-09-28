//! Drives the mocks with reqwest, as any client would.

#![allow(dead_code, clippy::unwrap_used)]

use std::time::Duration;

use futures::StreamExt;
use serde_json::Value;

pub struct Reply {
    pub status: u16,
    pub headers: reqwest::header::HeaderMap,
    pub body: String,
    /// `Err` if the body broke off.
    pub ended: Result<(), String>,
}

impl Reply {
    pub fn json(&self) -> Value {
        serde_json::from_str(&self.body).unwrap_or_else(|e| panic!("not JSON ({e}): {}", self.body))
    }

    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.get(name).and_then(|v| v.to_str().ok())
    }

    /// The `data:` payloads of an SSE body, in order.
    pub fn data(&self) -> Vec<String> {
        let lines = self.body.lines();
        lines
            .filter_map(|l| l.strip_prefix("data: "))
            .map(str::to_owned)
            .collect()
    }

    /// The `data:` payloads as JSON, `[DONE]` left out.
    pub fn chunks(&self) -> Vec<Value> {
        let data = self.data().into_iter().filter(|d| d != "[DONE]");
        data.map(|d| serde_json::from_str(&d).unwrap()).collect()
    }

    /// The `event:` names of an SSE body, in order.
    pub fn events(&self) -> Vec<String> {
        let lines = self.body.lines();
        lines
            .filter_map(|l| l.strip_prefix("event: "))
            .map(str::to_owned)
            .collect()
    }

    /// The error of a JSON error body: OpenAI's `error.code` or
    /// Anthropic's `error.type`, then the message.
    pub fn error(&self) -> (String, String) {
        let v = self.json();
        let e = &v["error"];
        let kind = e["code"]
            .as_str()
            .or(e["type"].as_str())
            .unwrap_or_default();
        (
            kind.to_owned(),
            e["message"].as_str().unwrap_or_default().to_owned(),
        )
    }
}

/// POSTs `body` to `url` with `headers`, reading the whole reply.
pub async fn post(url: &str, headers: &[(&str, &str)], body: &Value) -> Reply {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap();
    let mut req = client.post(url).header("content-type", "application/json");
    for (k, v) in headers {
        req = req.header(*k, *v);
    }
    let resp = req.body(body.to_string()).send().await.unwrap();
    let (status, headers) = (resp.status().as_u16(), resp.headers().clone());
    let mut out = Vec::new();
    let mut stream = resp.bytes_stream();
    let mut ended = Ok(());
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(b) => out.extend_from_slice(&b),
            Err(e) => {
                ended = Err(format!("{e:?}"));
                break;
            }
        }
    }
    let body = String::from_utf8(out).unwrap();
    Reply {
        status,
        headers,
        body,
        ended,
    }
}
