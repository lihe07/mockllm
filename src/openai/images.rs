//! OpenAI Images, generations: `POST /v1/images/generations`.
//! https://developers.openai.com/api/reference/resources/images/methods/generate

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use bytes::Bytes;
use serde_json::{Value, json};

use super::{OpenAi, bad, object};
use crate::core::misfit;
use crate::util::{Row, lookup, tokens, unix_now};
use crate::{Answer, Ctx, Endpoint, Out, Rejection, Reply, Request};

/// The body parameters of the reference above; any other is rejected.
#[rustfmt::skip]
const KNOWN: &[&str] = &[
    "prompt", "background", "model", "moderation", "n", "output_compression",
    "output_format", "partial_images", "quality", "response_format", "size", "stream",
    "style", "user",
];

/// A 1x1 transparent PNG, the image when none is scripted.
const PNG: &str =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";

/// Per-model rules, one row per model family; all from the reference.
#[derive(Clone, Copy, Debug)]
struct Model {
    prefix: &'static str,
    /// The largest `n`.
    max_n: u64,
    /// The `size` values accepted; empty means not checked.
    sizes: &'static [&'static str],
    /// A GPT image model: "always return base64-encoded images", rejects
    /// `response_format`, and reports `usage`.
    gpt_image: bool,
    /// Has a `revised_prompt` ("For dall-e-3 only").
    revises: bool,
}

impl Row for Model {
    fn prefix(&self) -> &'static str {
        self.prefix
    }
}

const BASE: Model = Model {
    prefix: "",
    max_n: 10,
    sizes: &[],
    gpt_image: false,
    revises: false,
};

#[rustfmt::skip]
const MODELS: &[Model] = &[
    // "For dall-e-2, use one of 256x256, 512x512, or 1024x1024."
    Model { prefix: "dall-e-2", sizes: &["256x256", "512x512", "1024x1024"], ..BASE },
    // "For dall-e-3, only n=1 is supported"; "use one of 1024x1024,
    // 1792x1024, or 1024x1792".
    Model { prefix: "dall-e-3", max_n: 1, sizes: &["1024x1024", "1792x1024", "1024x1792"], revises: true, ..BASE },
    // `response_format` "isn't supported for the GPT image models, which
    // always return base64-encoded images".
    Model { prefix: "gpt-image", gpt_image: true, ..BASE },
];

/// "Defaults to dall-e-2" (reference above).
fn model(body: &Value) -> &str {
    body["model"].as_str().unwrap_or("dall-e-2")
}

pub(crate) struct Images;

impl Endpoint<OpenAi> for Images {
    fn route(&self) -> (&'static str, &'static str) {
        ("POST", "/v1/images/generations")
    }

    fn validate(&self, _: &OpenAi, req: &Request) -> Result<(), Rejection> {
        let body = object(req, KNOWN)?;
        if !body.get("prompt").is_some_and(Value::is_string) {
            let code = Some("missing_required_parameter");
            return Err(bad(
                "Missing required parameter: 'prompt'.",
                Some("prompt"),
                code,
            ));
        }
        let row = lookup(MODELS, BASE, model(&req.body));
        // "Must be between 1 and 10. For dall-e-3, only n=1 is supported."
        let n = body.get("n").and_then(Value::as_u64).unwrap_or(1);
        if n < 1 || n > row.max_n {
            let message = format!(
                "Invalid 'n': {n}. Expected a value between 1 and {}.",
                row.max_n
            );
            return Err(bad(message, Some("n"), Some("invalid_value")));
        }
        let size = body.get("size").and_then(Value::as_str);
        if let Some(s) = size.filter(|s| !row.sizes.is_empty() && !row.sizes.contains(s)) {
            let message = format!(
                "Invalid value: '{s}'. Supported values are: {}.",
                row.sizes.join(", ")
            );
            return Err(bad(message, Some("size"), Some("invalid_value")));
        }
        if row.gpt_image && body.get("response_format").is_some_and(|f| !f.is_null()) {
            let message = "Unknown parameter: 'response_format'.";
            return Err(bad(
                message,
                Some("response_format"),
                Some("unknown_parameter"),
            ));
        }
        Ok(())
    }

    fn respond(&self, _: &OpenAi, req: &Request, answer: &Answer, ctx: &Ctx) -> Out {
        let body = &req.body;
        if body["stream"] == true {
            return Out::Misfit("mockllm: image streaming is not modelled yet".into());
        }
        let image = match &answer.reply {
            Reply::Image(bytes) => bytes.clone(),
            Reply::Unscripted => Bytes::from(STANDARD.decode(PNG).unwrap_or_default()),
            other => return misfit("images", other),
        };
        let row = lookup(MODELS, BASE, model(body));
        let base64 = row.gpt_image || body["response_format"] == "b64_json";
        let n = body["n"].as_u64().unwrap_or(1);
        let data: Vec<Value> = (0..n)
            .map(|_| {
                let mut one = match base64 {
                    true => json!({ "b64_json": STANDARD.encode(&image) }),
                    false => json!({ "url": ctx.host("image/png", image.clone()) }),
                };
                if row.revises {
                    one["revised_prompt"] = body["prompt"].clone();
                }
                one
            })
            .collect();
        let mut out = json!({ "created": unix_now(), "data": data });
        if row.gpt_image {
            let input = tokens(&body["prompt"]);
            let details = json!({ "image_tokens": 0, "text_tokens": input });
            out["usage"] = json!({ "input_tokens": input, "input_tokens_details": details, "output_tokens": n, "total_tokens": input + n });
        }
        Out::Json(200, out)
    }
}
