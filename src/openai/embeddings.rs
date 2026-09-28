//! OpenAI Embeddings: `POST /v1/embeddings`.
//! https://developers.openai.com/api/reference/resources/embeddings/methods/create

use base64::Engine;
use serde_json::{Value, json};

use super::{OpenAi, bad, object};
use crate::core::misfit;
use crate::util::{Row, lookup, tokens};
use crate::{Answer, Ctx, Endpoint, Out, Rejection, Reply, Request};

/// The body parameters of the reference above; any other is rejected.
const KNOWN: &[&str] = &["input", "model", "dimensions", "encoding_format", "user"];

/// Per-model rules, one row per model.
#[derive(Clone, Copy, Debug)]
struct Model {
    prefix: &'static str,
    /// The vector length without `dimensions`.
    dims: usize,
    /// `dimensions` is accepted.
    shortens: bool,
}

impl Row for Model {
    fn prefix(&self) -> &'static str {
        self.prefix
    }
}

const BASE: Model = Model {
    prefix: "",
    dims: 1536,
    shortens: true,
};

#[rustfmt::skip]
const MODELS: &[Model] = &[
    // "By default, the length of the embedding vector is 1536 for
    // text-embedding-3-small or 3072 for text-embedding-3-large":
    // https://developers.openai.com/api/docs/guides/embeddings
    Model { prefix: "text-embedding-3-small", dims: 1536, ..BASE },
    Model { prefix: "text-embedding-3-large", dims: 3072, ..BASE },
    // `dimensions` is "only supported in text-embedding-3 and later
    // models" (reference above); 1536 is the length commonly seen.
    Model { prefix: "text-embedding-ada-002", dims: 1536, shortens: false },
];

pub(crate) struct Embeddings;

impl Endpoint<OpenAi> for Embeddings {
    fn route(&self) -> (&'static str, &'static str) {
        ("POST", "/v1/embeddings")
    }

    fn validate(&self, _: &OpenAi, req: &Request) -> Result<(), Rejection> {
        let body = object(req, KNOWN)?;
        let Some(model) = body.get("model").and_then(Value::as_str) else {
            return Err(bad("you must provide a model parameter", None, None));
        };
        // "cannot be an empty string, and any array must be 2048
        // dimensions or less" (reference above).
        match body.get("input") {
            None => {
                let message = "Missing required parameter: 'input'.";
                let code = Some("missing_required_parameter");
                return Err(bad(message, Some("input"), code));
            }
            Some(Value::String(s)) if s.is_empty() => {
                let message = "'$.input' is invalid. Please check the API reference: https://platform.openai.com/docs/api-reference.";
                return Err(bad(message, None, None));
            }
            Some(Value::Array(a)) if a.len() > 2048 => {
                let message = format!(
                    "Invalid 'input': array too long. Expected an array with maximum length 2048, but got an array with length {} instead.",
                    a.len()
                );
                return Err(bad(message, Some("input"), Some("array_above_max_length")));
            }
            _ => {}
        }
        let format = body.get("encoding_format").and_then(Value::as_str);
        if let Some(f) = format.filter(|f| !["float", "base64"].contains(f)) {
            let message =
                format!("Invalid value: '{f}'. Supported values are: 'float' and 'base64'.");
            return Err(bad(message, Some("encoding_format"), Some("invalid_value")));
        }
        let row = lookup(MODELS, BASE, model);
        if body.get("dimensions").is_some_and(|d| !d.is_null()) && !row.shortens {
            let message = "This model does not support specifying dimensions.";
            return Err(bad(message, Some("dimensions"), None));
        }
        Ok(())
    }

    fn respond(&self, _: &OpenAi, req: &Request, answer: &Answer, _: &Ctx) -> Out {
        let body = &req.body;
        let model = body["model"].as_str().unwrap_or_default();
        let dims = body["dimensions"]
            .as_u64()
            .map_or(lookup(MODELS, BASE, model).dims, |d| d as usize);
        let inputs = match &body["input"] {
            Value::Array(a) if a.iter().all(|x| !x.is_number()) => a.len(),
            _ => 1,
        };
        let vector = |i: usize| match &answer.reply {
            Reply::Embedding(v) => Some(v.clone()),
            // A unit vector per input, different for each.
            Reply::Unscripted => Some(
                (0..dims)
                    .map(|d| f32::from(u8::from(d == i % dims)))
                    .collect(),
            ),
            _ => None,
        };
        let base64 = body["encoding_format"] == "base64";
        let mut data = Vec::new();
        for i in 0..inputs {
            let Some(v) = vector(i) else {
                return misfit("embeddings", &answer.reply);
            };
            // base64: the floats as little-endian f32 bytes, as the SDKs decode.
            let embedding = if base64 {
                let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
                json!(base64::engine::general_purpose::STANDARD.encode(bytes))
            } else {
                json!(v)
            };
            data.push(json!({ "object": "embedding", "embedding": embedding, "index": i }));
        }
        let prompt = tokens(&body["input"]);
        let usage = json!({ "prompt_tokens": prompt, "total_tokens": prompt });
        Out::Json(
            200,
            json!({ "object": "list", "data": data, "model": model, "usage": usage }),
        )
    }
}
