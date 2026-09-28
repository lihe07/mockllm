//! OpenAI: what its endpoints share (Bearer auth, the error shape,
//! `x-request-id`, the quirks of OpenAI-compatible providers), and the
//! endpoints, one file each.

mod chat;
mod chat_models;
mod embeddings;
mod images;
mod realtime;
mod responses;

use serde_json::{Map, Value, json};

use crate::{Endpoint, Mock, Provider, Rejection, Request};

/// The OpenAI API, or an OpenAI-compatible one with [`Quirks`].
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenAi {
    quirks: Quirks,
}

/// A mock OpenAI.
pub type FakeOpenAi = Mock<OpenAi>;

impl Mock<OpenAi> {
    /// Starts an OpenAI-compatible provider with `quirks`.
    pub async fn with_quirks(quirks: Quirks) -> Self {
        Self::serve(OpenAi { quirks }).await
    }
}

impl Provider for OpenAi {
    const BASE_PATH: &'static str = "/v1";

    fn endpoints() -> Vec<Box<dyn Endpoint<Self>>> {
        vec![
            Box::new(chat::Chat),
            Box::new(embeddings::Embeddings),
            Box::new(images::Images),
            Box::new(realtime::Realtime),
            Box::new(responses::Responses),
        ]
    }

    fn authorize(&self, req: &Request) -> Result<(), Rejection> {
        let bearer = req.header("authorization");
        if bearer.is_some_and(|v| v.starts_with("Bearer ")) {
            return Ok(());
        }
        let message = "You didn't provide an API key. You need to provide your API key in an Authorization header using Bearer auth (i.e. Authorization: Bearer YOUR_KEY).";
        Err(Rejection::new(401, message))
    }

    fn error(&self, r: &Rejection, _id: u64) -> Value {
        error_body(r.status, &r.message, r.param.as_deref(), r.kind)
    }

    fn scripted(&self, status: u16) -> Rejection {
        let (message, code) = match status {
            401 => ("Incorrect API key provided.", Some("invalid_api_key")),
            404 => (
                "The model does not exist or you do not have access to it.",
                Some("model_not_found"),
            ),
            429 => (
                "Rate limit reached for requests.",
                Some("rate_limit_exceeded"),
            ),
            503 => (
                "The engine is currently overloaded, please try again later.",
                None,
            ),
            s if s >= 500 => (
                "The server had an error while processing your request. Sorry about that!",
                None,
            ),
            _ => ("Invalid request.", None),
        };
        Rejection {
            kind: code,
            ..Rejection::new(status, message)
        }
    }

    fn not_found(&self, req: &Request) -> Rejection {
        Rejection::new(404, format!("Invalid URL ({} {})", req.method, req.path))
    }

    fn headers(&self, id: u64) -> Vec<(&'static str, String)> {
        vec![("x-request-id", format!("req_fake{id}"))]
    }
}

/// OpenAI's error shape; `type` follows from the status.
pub(crate) fn error_body(
    status: u16,
    message: &str,
    param: Option<&str>,
    code: Option<&str>,
) -> Value {
    let kind = match status {
        429 => "requests",
        s if s >= 500 => "server_error",
        _ => "invalid_request_error",
    };
    json!({ "error": { "message": message, "type": kind, "param": param, "code": code } })
}

/// A 400 with `param` and `code`.
pub(crate) fn bad(
    message: impl Into<String>,
    param: Option<&str>,
    code: Option<&'static str>,
) -> Rejection {
    let r = Rejection::new(400, message);
    let r = Rejection { kind: code, ..r };
    match param {
        Some(p) => r.param(p),
        None => r,
    }
}

/// The body as an object of only `known` parameters, as every endpoint
/// checks first.
pub(crate) fn object<'a>(
    req: &'a Request,
    known: &[&str],
) -> Result<&'a Map<String, Value>, Rejection> {
    let Some(body) = req.object() else {
        return Err(bad(
            "We could not parse the JSON body of your request.",
            None,
            None,
        ));
    };
    if let Some(k) = body.keys().find(|k| !known.contains(&k.as_str())) {
        let message = format!("Unrecognized request argument supplied: {k}");
        return Err(bad(message, None, Some("unrecognized_request_argument")));
    }
    Ok(body)
}

/// How an OpenAI-compatible provider deviates from OpenAI.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Quirks {
    usage_every_chunk: bool,
    no_done: bool,
    reasoning_content: bool,
}

impl Quirks {
    pub fn new() -> Self {
        Self::default()
    }

    /// Every stream chunk carries the running `usage`, asked or not, as
    /// vLLM does with `continuous_usage_stats`:
    /// https://docs.vllm.ai/en/v0.8.5/serving/openai_compatible_server.html
    pub fn usage_every_chunk(mut self) -> Self {
        self.usage_every_chunk = true;
        self
    }

    /// The stream ends without `data: [DONE]`. Seen from OpenAI-compatible
    /// servers in production; no public doc is cited for it.
    pub fn no_done(mut self) -> Self {
        self.no_done = true;
        self
    }

    /// [`crate::Answer::reasoning`] comes as `reasoning_content`, "the
    /// reasoning contents of the assistant message, before the final
    /// answer", in stream deltas and in the whole message, and is counted
    /// in `reasoning_tokens`: https://api-docs.deepseek.com/api/create-chat-completion
    pub fn reasoning_content(mut self) -> Self {
        self.reasoning_content = true;
        self
    }
}
