//! Anthropic models: `GET /v1/models` and `GET /v1/models/{model_id}`.
//! https://platform.claude.com/docs/en/api/models/list
//! https://platform.claude.com/docs/en/api/models/retrieve

use serde_json::{Value, json};

use super::messages_models::{BASE, MODELS as LIMITS};
use super::{Anthropic, limit, page, unscripted};
use crate::util::lookup;
use crate::{Answer, Ctx, Endpoint, Out, Rejection, Request};

/// "May be set to an epoch value if the release date is unknown" (list).
const EPOCH: &str = "1970-01-01T00:00:00Z";

/// The models listed, "more recently released models ... first" (list):
/// id, display name, release. Ids and names from the models overview
/// (https://platform.claude.com/docs/en/models/overview) and the `model`
/// enum of the Messages reference; Opus 5's date from the reference
/// example. Limited-access models (Mythos) are left out. Context window
/// and max output come from `messages_models.rs`, the table Messages
/// checks against.
#[rustfmt::skip]
const MODELS: &[(&str, &str, &str)] = &[
    ("claude-fable-5-1", "Claude Fable 5.1", EPOCH),
    ("claude-opus-5-5", "Claude Opus 5.5", EPOCH),
    ("claude-sonnet-5-5", "Claude Sonnet 5.5", EPOCH),
    ("claude-opus-5", "Claude Opus 5", "2026-07-24T00:00:00Z"),
    ("claude-sonnet-5", "Claude Sonnet 5", EPOCH),
    ("claude-fable-5", "Claude Fable 5", EPOCH),
    ("claude-opus-4-8", "Claude Opus 4.8", EPOCH),
    ("claude-opus-4-7", "Claude Opus 4.7", EPOCH),
    ("claude-opus-4-6", "Claude Opus 4.6", EPOCH),
    ("claude-sonnet-4-6", "Claude Sonnet 4.6", EPOCH),
    ("claude-opus-4-5-20251101", "Claude Opus 4.5", EPOCH),
    ("claude-haiku-4-5-20251001", "Claude Haiku 4.5", EPOCH),
    ("claude-sonnet-4-5-20250929", "Claude Sonnet 4.5", EPOCH),
];

/// "resolve a model alias to a model ID" (retrieve). Before the 4.6
/// generation "the alias is a convenience pointer that resolves to the
/// dated ID" (overview).
const ALIASES: &[(&str, &str)] = &[
    ("claude-opus-4-5", "claude-opus-4-5-20251101"),
    ("claude-haiku-4-5", "claude-haiku-4-5-20251001"),
    ("claude-sonnet-4-5", "claude-sonnet-4-5-20250929"),
];

pub(crate) fn endpoints() -> Vec<Box<dyn Endpoint<Anthropic>>> {
    vec![Box::new(Models(false)), Box::new(Models(true))]
}

/// The list, or (`true`) one model.
struct Models(bool);

fn resolve(id: &str) -> Option<&'static (&'static str, &'static str, &'static str)> {
    let id = ALIASES
        .iter()
        .find(|(a, _)| *a == id)
        .map_or(id, |(_, to)| to);
    MODELS.iter().find(|m| m.0 == id)
}

fn info(&(id, name, created): &(&str, &str, &str)) -> Value {
    let row = lookup(LIMITS, BASE, id);
    let known = |n: u64| (n != u64::MAX).then_some(n);
    // `capabilities` is "ModelCapabilities or null" (list); null here.
    json!({
        "type": "model", "id": id, "capabilities": null, "created_at": created,
        "display_name": name, "max_input_tokens": known(row.context),
        "max_tokens": known(row.max_output),
    })
}

impl Endpoint<Anthropic> for Models {
    fn route(&self) -> (&'static str, &'static str) {
        match self.0 {
            false => ("GET", "/v1/models"),
            true => ("GET", "/v1/models/{model_id}"),
        }
    }

    fn validate(&self, _: &Anthropic, req: &Request) -> Result<(), Rejection> {
        if !self.0 {
            return limit(req).map(|_| ());
        }
        let id = req.param.as_deref().unwrap_or_default();
        match resolve(id) {
            Some(_) => Ok(()),
            // The commonly seen wording.
            None => Err(Rejection::new(404, format!("model: {id}")).kind("not_found_error")),
        }
    }

    fn respond(&self, _: &Anthropic, req: &Request, answer: &Answer, _: &Ctx) -> Out {
        if let Some(misfit) = unscripted("models", answer) {
            return misfit;
        }
        let body = match req.param.as_deref().and_then(resolve) {
            Some(m) if self.0 => info(m),
            _ => page(&MODELS.iter().map(info).collect::<Vec<_>>(), req),
        };
        Out::Json(200, body)
    }
}
