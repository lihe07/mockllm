//! Google Gemini API: what its endpoints share (the API key, the error
//! shape), and the endpoints, one file each.

mod embed;
mod generate;
mod generate_models;
mod interactions;
mod live;

use serde_json::{Value, json};

use crate::{Endpoint, Mock, Provider, Rejection, Request};

/// The Gemini API (`generativelanguage.googleapis.com`).
#[derive(Clone, Copy, Debug, Default)]
pub struct Gemini;

/// A mock Gemini.
pub type FakeGemini = Mock<Gemini>;

impl Provider for Gemini {
    /// Without the version, like `https://generativelanguage.googleapis.com`;
    /// the SDKs add `/v1beta`.
    const BASE_PATH: &'static str = "";

    fn endpoints() -> Vec<Box<dyn Endpoint<Self>>> {
        vec![
            Box::new(generate::Generate { stream: false }),
            Box::new(generate::Generate { stream: true }),
            Box::new(embed::Embed { batch: false }),
            Box::new(embed::Embed { batch: true }),
            Box::new(live::Live),
            Box::new(interactions::Interactions),
        ]
    }

    /// The key in the `x-goog-api-key` header or the `key` query parameter:
    /// https://ai.google.dev/gemini-api/docs/api-key and the curl examples of
    /// https://ai.google.dev/api/generate-content. The wording of a missing
    /// key is the one commonly seen.
    fn authorize(&self, req: &Request) -> Result<(), Rejection> {
        if req.header("x-goog-api-key").is_some() || req.query_param("key").is_some() {
            return Ok(());
        }
        let message = "Method doesn't allow unregistered callers (callers without established identity). Please use API Key or other form of API consumer identity to call this API.";
        Err(Rejection::new(403, message))
    }

    /// Google's JSON error: `{"error": {"code", "message", "status"}}`,
    /// `status` the `google.rpc.Code` name of the HTTP status:
    /// https://cloud.google.com/apis/design/errors
    fn error(&self, r: &Rejection, _id: u64) -> Value {
        let status = r.kind.unwrap_or_else(|| code(r.status));
        json!({ "error": { "code": r.status, "message": r.message, "status": status } })
    }

    /// Wording commonly seen from the API.
    fn scripted(&self, status: u16) -> Rejection {
        let message = match status {
            429 => "Resource has been exhausted (e.g. check quota).",
            503 => "The model is overloaded. Please try again later.",
            504 => "Deadline expired before operation could complete.",
            403 => "The caller does not have permission",
            404 => "Requested entity was not found.",
            s if s >= 500 => {
                "An internal error has occurred. Please retry or report in https://developers.generativeai.google/guide/troubleshooting"
            }
            _ => "Request contains an invalid argument.",
        };
        Rejection::new(status, message)
    }
}

/// The `google.rpc.Code` of an HTTP status, by the "HTTP Mapping" of
/// https://github.com/googleapis/googleapis/blob/master/google/rpc/code.proto
fn code(status: u16) -> &'static str {
    match status {
        400 => "INVALID_ARGUMENT",
        401 => "UNAUTHENTICATED",
        403 => "PERMISSION_DENIED",
        404 => "NOT_FOUND",
        409 => "ABORTED",
        429 => "RESOURCE_EXHAUSTED",
        499 => "CANCELLED",
        501 => "UNIMPLEMENTED",
        503 => "UNAVAILABLE",
        504 => "DEADLINE_EXCEEDED",
        s if s >= 500 => "INTERNAL",
        _ => "FAILED_PRECONDITION",
    }
}

/// A 400 `INVALID_ARGUMENT`.
pub(crate) fn bad(message: impl Into<String>) -> Rejection {
    Rejection::new(400, message)
}
