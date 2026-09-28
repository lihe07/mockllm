//! Anthropic: what its endpoints share (`x-api-key` and
//! `anthropic-version`, the error shape with `request_id`, `request-id`),
//! and the endpoints, one file each.

mod batches;
mod count_tokens;
mod files;
mod messages;
mod messages_models;
mod models;

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
        let mut all: Vec<Box<dyn Endpoint<Self>>> = vec![Box::new(messages::Messages)];
        all.push(Box::new(count_tokens::CountTokens));
        all.extend(batches::endpoints());
        all.extend(models::endpoints());
        all.extend(files::endpoints());
        all
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

/// `limit` of a list request: "Defaults to `20`. Ranges from `1` to
/// `1000`." https://platform.claude.com/docs/en/api/models/list
/// (wording: the commonly seen validation wording).
pub(crate) fn limit(req: &Request) -> Result<usize, Rejection> {
    let Some(raw) = req.query_param("limit") else {
        return Ok(20);
    };
    match raw.parse::<i64>() {
        Ok(n) if n < 1 => Err(bad("limit: Input should be greater than or equal to 1")),
        Ok(n) if n > 1000 => Err(bad("limit: Input should be less than or equal to 1000")),
        Ok(n) => Ok(n as usize),
        Err(_) => Err(bad(
            "limit: Input should be a valid integer, unable to parse string as an integer",
        )),
    }
}

/// A page of `items` (newest first) by `before_id`, `after_id` and
/// `limit`: `{data, has_more, first_id, last_id}`.
/// https://platform.claude.com/docs/en/api/messages/batches/list
pub(crate) fn page(items: &[Value], req: &Request) -> Value {
    let limit = limit(req).unwrap_or(20);
    let at = |key| {
        let id = req.query_param(key)?;
        items.iter().position(|v| v["id"] == id)
    };
    let (data, has_more) = if let Some(i) = at("before_id") {
        let start = i.saturating_sub(limit);
        (&items[start..i], start > 0)
    } else {
        let start = at("after_id").map_or(0, |i| i + 1);
        let end = (start + limit).min(items.len());
        (&items[start..end], end < items.len())
    };
    let (first, last) = (
        data.first().map(|v| &v["id"]),
        data.last().map(|v| &v["id"]),
    );
    json!({ "data": data, "has_more": has_more, "first_id": first, "last_id": last })
}

/// Unix seconds as an RFC 3339 datetime, as every Anthropic timestamp.
pub(crate) fn rfc3339(secs: u64) -> String {
    // Days to a civil date: https://howardhinnant.github.io/date_algorithms.html
    let (days, rem) = ((secs / 86_400) as i64, secs % 86_400);
    let z = days + 719_468;
    let (era, doe) = (z.div_euclid(146_097), z.rem_euclid(146_097));
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    let (h, min, s) = (rem / 3600, rem % 3600 / 60, rem % 60);
    format!("{y:04}-{m:02}-{d:02}T{h:02}:{min:02}:{s:02}Z")
}

/// The one reply kind a bookkeeping endpoint (lists, metadata) frames:
/// unscripted. The rest are a misfit.
pub(crate) fn unscripted(endpoint: &str, answer: &crate::Answer) -> Option<crate::Out> {
    let reply = answer.reply();
    (*reply != crate::Reply::Unscripted).then(|| crate::core::misfit(endpoint, reply))
}
