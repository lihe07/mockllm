//! Anthropic: what its endpoints share (`x-api-key` and
//! `anthropic-version`, the error shape with `request_id`, `request-id`),
//! and the endpoints, one file each.

mod messages;
mod messages_models;

use serde_json::{Value, json};

use crate::{Endpoint, Mock, Provider, Rejection, Request};

/// The Anthropic API.
#[derive(Clone, Copy, Debug, Default)]
pub struct Anthropic;

/// A mock Anthropic.
pub type FakeAnthropic = Mock<Anthropic>;

impl Provider for Anthropic {
    /// Without the version, like `https://api.anthropic.com`.
    const BASE_PATH: &'static str = "";

    fn endpoints() -> Vec<Box<dyn Endpoint<Self>>> {
        vec![Box::new(messages::Messages)]
    }

    fn authorize(&self, req: &Request) -> Result<(), Rejection> {
        if req.header("x-api-key").is_none() {
            let message = "x-api-key header is required";
            return Err(Rejection::new(401, message).kind("authentication_error"));
        }
        if req.header("anthropic-version").is_none() {
            return Err(bad("anthropic-version: header is required"));
        }
        Ok(())
    }

    /// With `request_id`: https://platform.claude.com/docs/en/api/errors
    fn error(&self, r: &Rejection, id: u64) -> Value {
        let kind = r.kind.unwrap_or("invalid_request_error");
        let error = json!({ "type": kind, "message": r.message });
        json!({ "type": "error", "error": error, "request_id": format!("req_fake{id}") })
    }

    fn scripted(&self, status: u16) -> Rejection {
        let (kind, message) = match status {
            401 => ("authentication_error", "invalid x-api-key"),
            402 => (
                "billing_error",
                "Your credit balance is too low to access the Anthropic API.",
            ),
            403 => (
                "permission_error",
                "Your API key does not have permission to use the specified resource.",
            ),
            404 => (
                "not_found_error",
                "The requested resource could not be found.",
            ),
            413 => (
                "request_too_large",
                "Request exceeds the maximum allowed number of bytes.",
            ),
            429 => (
                "rate_limit_error",
                "Number of request tokens has exceeded your per-minute rate limit.",
            ),
            504 => ("timeout_error", "Request timed out."),
            529 => ("overloaded_error", "Overloaded"),
            s if s >= 500 => ("api_error", "Internal server error"),
            _ => ("invalid_request_error", "Invalid request"),
        };
        Rejection::new(status, message).kind(kind)
    }

    fn not_found(&self, _req: &Request) -> Rejection {
        Rejection::new(404, "Not Found").kind("not_found_error")
    }

    fn headers(&self, id: u64) -> Vec<(&'static str, String)> {
        vec![("request-id", format!("req_fake{id}"))]
    }
}

/// A 400 `invalid_request_error`.
pub(crate) fn bad(message: impl Into<String>) -> Rejection {
    Rejection::new(400, message).kind("invalid_request_error")
}
