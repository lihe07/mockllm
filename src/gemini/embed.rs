//! Gemini embeddings: `POST /v1beta/models/{model}:embedContent` and
//! `:batchEmbedContents`.
//! https://ai.google.dev/api/embeddings
//!
//! Error wording the docs do not give is the wording commonly seen, in
//! Google's field-path style; where marked, no wording is documented or
//! commonly seen and the message only names the rule.

use serde_json::{Map, Value, json};

use super::{Gemini, bad};
use crate::core::misfit;
use crate::util::{Row, lookup, tokens};
use crate::{Answer, Ctx, Endpoint, Out, Rejection, Reply, Request};

/// The `EmbedContentRequest` fields of the reference (`model` too, as in
/// its curl example); any other is rejected. Google's JSON also takes the
/// proto's snake_case names, as the embeddings guide's REST example does
/// (`output_dimensionality`).
const KNOWN: &[&str] = &[
    "model",
    "content",
    "taskType",
    "title",
    "outputDimensionality",
    "embedContentConfig",
];

/// `TaskType` of the reference.
#[rustfmt::skip]
const TASKS: &[&str] = &[
    "TASK_TYPE_UNSPECIFIED", "RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT", "SEMANTIC_SIMILARITY",
    "CLASSIFICATION", "CLUSTERING", "QUESTION_ANSWERING", "FACT_VERIFICATION",
    "CODE_RETRIEVAL_QUERY",
];

/// Per-model rules, one row per model.
#[derive(Clone, Copy, Debug)]
struct Model {
    prefix: &'static str,
    /// `outputDimensionality` accepted, `None` when not checked.
    range: Option<(u64, u64)>,
}

impl Row for Model {
    fn prefix(&self) -> &'static str {
        self.prefix
    }
}

const BASE: Model = Model {
    prefix: "",
    range: None,
};

#[rustfmt::skip]
const MODELS: &[Model] = &[
    // "Output dimension size: Flexible, supports: 128 - 3072", both models:
    // https://ai.google.dev/gemini-api/docs/embeddings#model-versions
    Model { prefix: "gemini-embedding-001", range: Some((128, 3072)) },
    Model { prefix: "gemini-embedding-2", range: Some((128, 3072)) },
];

/// "By default, both models output a 3072-dimensional embedding":
/// https://ai.google.dev/gemini-api/docs/embeddings#controlling-embedding-size
const DIMS: usize = 3072;

pub(crate) struct Embed {
    pub batch: bool,
}

/// `outputDimensionality` → `output_dimensionality`.
fn snake(camel: &str) -> String {
    let mut out = String::new();
    for c in camel.chars() {
        if c.is_ascii_uppercase() {
            out.push('_');
        }
        out.push(c.to_ascii_lowercase());
    }
    out
}

/// `req[name]`, by its JSON name or its proto (snake_case) name.
fn field<'a>(req: &'a Map<String, Value>, name: &str) -> Option<&'a Value> {
    req.get(name).or_else(|| req.get(&snake(name)))
}

/// A setting of `embedContentConfig`, else its deprecated top-level twin.
fn setting<'a>(req: &'a Map<String, Value>, name: &str) -> Option<&'a Value> {
    let config = field(req, "embedContentConfig").and_then(Value::as_object);
    config
        .and_then(|c| field(c, name))
        .or_else(|| field(req, name))
}

/// The rules of one `EmbedContentRequest`; `at` names it in errors.
fn check(req: &Value, at: &str, model: &str) -> Result<(), Rejection> {
    let Some(req) = req.as_object() else {
        return Err(bad("Invalid JSON payload received."));
    };
    let known = |k: &String| KNOWN.iter().any(|n| k == n || *k == snake(n));
    if let Some(k) = req.keys().find(|k| !known(k)) {
        let message =
            format!("Invalid JSON payload received. Unknown name \"{k}\": Cannot find field.");
        return Err(bad(message));
    }
    // `content`: "Required. The content to embed."
    if !field(req, "content").is_some_and(Value::is_object) {
        return Err(bad(format!("* {at}.content: content is not specified\n")));
    }
    // `taskType`: one of `TaskType`; the enum error is the one commonly seen.
    let task = setting(req, "taskType");
    if let Some(task) = task.filter(|t| !t.as_str().is_some_and(|t| TASKS.contains(&t))) {
        return Err(bad(format!(
            "Invalid value at 'task_type' (type.googleapis.com/google.ai.generativelanguage.v1beta.TaskType), {task}"
        )));
    }
    // Range per model (table above); wording not documented.
    let dims = setting(req, "outputDimensionality").and_then(Value::as_u64);
    if let (Some(n), Some((lo, hi))) = (dims, lookup(MODELS, BASE, model).range)
        && !(lo..=hi).contains(&n)
    {
        return Err(bad(format!(
            "* {at}.output_dimensionality: must be between {lo} and {hi}, got {n}\n"
        )));
    }
    Ok(())
}

impl Endpoint<Gemini> for Embed {
    fn route(&self) -> (&'static str, &'static str) {
        match self.batch {
            false => ("POST", "/v1beta/models/{model}:embedContent"),
            true => ("POST", "/v1beta/models/{model}:batchEmbedContents"),
        }
    }

    fn validate(&self, _: &Gemini, req: &Request) -> Result<(), Rejection> {
        let model = req.param.as_deref().unwrap_or_default();
        if !self.batch {
            return check(&req.body, "EmbedContentRequest", model);
        }
        let Some(body) = req.object() else {
            return Err(bad("Invalid JSON payload received."));
        };
        if let Some(k) = body.keys().find(|k| *k != "requests") {
            let message =
                format!("Invalid JSON payload received. Unknown name \"{k}\": Cannot find field.");
            return Err(bad(message));
        }
        // `requests[]`: "Required."
        let requests = body.get("requests").and_then(Value::as_array);
        let Some(requests) = requests.filter(|r| !r.is_empty()) else {
            return Err(bad(
                "* BatchEmbedContentsRequest.requests: requests is not specified\n",
            ));
        };
        for (i, r) in requests.iter().enumerate() {
            let at = format!("BatchEmbedContentsRequest.requests[{i}]");
            check(r, &at, model)?;
            // "The model in each of these requests must match the model
            // specified BatchEmbedContentsRequest.model." Wording not
            // documented.
            let own = r.get("model").and_then(Value::as_str).unwrap_or_default();
            if own != format!("models/{model}") {
                return Err(bad(format!(
                    "* {at}.model: model name must match the parent model name. Got '{own}', expected 'models/{model}'\n"
                )));
            }
        }
        Ok(())
    }

    fn respond(&self, _: &Gemini, req: &Request, answer: &Answer, _: &Ctx) -> Out {
        let requests = match self.batch {
            true => req.body["requests"].as_array().cloned().unwrap_or_default(),
            false => vec![req.body.clone()],
        };
        let mut embeddings = Vec::new();
        for (i, r) in requests.iter().enumerate() {
            let object = r.as_object().cloned().unwrap_or_default();
            let dims = setting(&object, "outputDimensionality")
                .and_then(Value::as_u64)
                .map_or(DIMS, |d| d as usize);
            let values: Vec<f32> = match &answer.reply {
                Reply::Embedding(v) => v.clone(),
                // A unit vector per request, different for each.
                Reply::Unscripted => (0..dims)
                    .map(|d| f32::from(u8::from(d == i % dims)))
                    .collect(),
                other => return misfit("embedContent", other),
            };
            // `ContentEmbedding`: `values`.
            embeddings.push(json!({ "values": values }));
        }
        // `EmbeddingUsageMetadata`: "Only the parts.text fields will be counted."
        let texts: Vec<Value> = requests.iter().map(|r| r["content"].clone()).collect();
        let prompt = tokens(&Value::Array(texts));
        let details = json!([{ "modality": "TEXT", "tokenCount": prompt }]);
        let usage = json!({ "promptTokenCount": prompt, "promptTokenDetails": details });
        let body = match self.batch {
            true => json!({ "embeddings": embeddings, "usageMetadata": usage }),
            false => json!({ "embedding": embeddings[0], "usageMetadata": usage }),
        };
        Out::Json(200, body)
    }
}
