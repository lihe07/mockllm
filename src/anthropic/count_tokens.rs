//! Anthropic token counting: `POST /v1/messages/count_tokens`.
//! https://platform.claude.com/docs/en/api/messages/count_tokens

use serde_json::{Value, json};

use super::messages::check_messages;
use super::{Anthropic, bad};
use crate::core::misfit;
use crate::util::tokens;
use crate::{Answer, Ctx, Endpoint, Out, Rejection, Reply, Request};

/// The body parameters of the reference; any other (`max_tokens`,
/// `stream`, ...) is "Extra inputs are not permitted".
#[rustfmt::skip]
const KNOWN: &[&str] = &[
    "messages", "model", "cache_control", "output_config", "system", "thinking",
    "tool_choice", "tools",
];

pub(crate) struct CountTokens;

impl Endpoint<Anthropic> for CountTokens {
    fn route(&self) -> (&'static str, &'static str) {
        ("POST", "/v1/messages/count_tokens")
    }

    fn validate(&self, _: &Anthropic, req: &Request) -> Result<(), Rejection> {
        let Some(body) = req.object() else {
            return Err(bad("There was an error parsing the body of your request."));
        };
        if let Some(k) = body.keys().find(|k| !KNOWN.contains(&k.as_str())) {
            return Err(bad(format!("{k}: Extra inputs are not permitted")));
        }
        // `messages` and `model` are the required parameters (reference).
        for field in ["messages", "model"] {
            if !body.contains_key(field) {
                return Err(bad(format!("{field}: Field required")));
            }
        }
        let Some(messages) = body["messages"].as_array() else {
            return Err(bad("messages: Input should be a valid list"));
        };
        // The same `MessageParam` rules as Messages (reference).
        check_messages(messages)
    }

    /// Unscripted: "the total number of tokens across the provided list of
    /// messages, system prompt, and tools" (reference), counted roughly.
    /// Raw events: the first is the whole body, to set an exact count.
    fn respond(&self, _: &Anthropic, req: &Request, answer: &Answer, _: &Ctx) -> Out {
        match answer.reply() {
            Reply::Unscripted => {
                let body = &req.body;
                let total: u64 = ["messages", "system", "tools"]
                    .iter()
                    .filter(|k| !body[**k].is_null())
                    .map(|k| tokens(&body[*k]))
                    .sum();
                Out::Json(200, json!({ "input_tokens": total }))
            }
            Reply::Events(e) => Out::Json(200, e.first().cloned().unwrap_or(Value::Null)),
            other => misfit("count_tokens", other),
        }
    }
}
