//! OpenAI Images, generations: `POST /v1/images/generations`.
//! https://developers.openai.com/api/reference/resources/images/methods/generate

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use bytes::Bytes;
use serde_json::{Value, json};

use super::{OpenAi, bad, object};
use crate::core::misfit;
use crate::util::{Row, lookup, tokens, unix_now};
use crate::{Answer, Ctx, Endpoint, Out, Provider, Rejection, Reply, Request, event};

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

/// "Defaults to `dall-e-2` unless a parameter specific to the GPT image
/// models is used" (reference above), such as `stream`. Which GPT image
/// model is not said; the GPT image row applies.
fn model(body: &Value) -> &str {
    match body["model"].as_str() {
        Some(m) => m,
        None if streaming(body) => "gpt-image",
        None => "dall-e-2",
    }
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
        check_stream(&req.body, row.gpt_image)
    }

    fn respond(&self, openai: &OpenAi, req: &Request, answer: &Answer, ctx: &Ctx) -> Out {
        let row = lookup(MODELS, BASE, model(&req.body));
        let kind = Kind {
            name: "images",
            event: "image_generation",
            gpt_image: row.gpt_image,
            revises: row.revises,
        };
        frame(openai, &kind, &req.body, answer, ctx)
    }
}

/// How [`frame`] answers for one images endpoint and model.
pub(super) struct Kind {
    /// The endpoint, for misfit errors.
    pub name: &'static str,
    /// The stream event prefix: `image_generation`, `image_edit`.
    pub event: &'static str,
    pub gpt_image: bool,
    pub revises: bool,
}

/// `stream` "is only supported for the GPT image models", and
/// `partial_images` "must be between 0 and 3" (the reference of each
/// images endpoint). Wording as commonly seen.
pub(super) fn check_stream(body: &Value, gpt_image: bool) -> Result<(), Rejection> {
    if streaming(body) && !gpt_image {
        let message = "Unsupported value: 'stream' is not supported with this model.";
        return Err(bad(message, Some("stream"), Some("unsupported_value")));
    }
    let partial = &body["partial_images"];
    match (partial.is_null(), int(partial)) {
        (false, None) => {
            let message = format!(
                "Invalid type for 'partial_images': expected an integer, but got {partial} instead."
            );
            Err(bad(message, Some("partial_images"), Some("invalid_type")))
        }
        (_, Some(p)) if p > 3 => {
            let message = format!(
                "Invalid 'partial_images': integer above maximum value. Expected a value <= 3, but got {p} instead."
            );
            Err(bad(
                message,
                Some("partial_images"),
                Some("integer_above_max_value"),
            ))
        }
        _ => Ok(()),
    }
}

/// A form field's value or a JSON number, as a number.
pub(super) fn int(v: &Value) -> Option<u64> {
    v.as_u64().or_else(|| v.as_str()?.parse().ok())
}

/// `stream` set, as JSON `true` or the form text `"true"`.
pub(super) fn streaming(body: &Value) -> bool {
    body["stream"] == true || body["stream"] == "true"
}

/// The images response, or its stream: shared by generations and edits.
pub(super) fn frame(openai: &OpenAi, kind: &Kind, body: &Value, answer: &Answer, ctx: &Ctx) -> Out {
    // A mid-stream error as a named `error` event with OpenAI's error body;
    // the docs show none, this is the framing commonly seen.
    let error = event("error", &openai.error(&openai.scripted(500), ctx.id));
    let image = match &answer.reply {
        Reply::Image(bytes) => bytes.clone(),
        Reply::Unscripted => Bytes::from(STANDARD.decode(PNG).unwrap_or_default()),
        Reply::Events(events) if streaming(body) => {
            let named = |e: &Value| event(e["type"].as_str().unwrap_or("message"), e);
            return Out::Sse(events.iter().map(named).collect(), error);
        }
        other => return misfit(kind.name, other),
    };
    let n = int(&body["n"]).unwrap_or(1);
    let input = tokens(&body["prompt"]);
    let details = json!({ "image_tokens": 0, "text_tokens": input });
    let usage = json!({ "input_tokens": input, "input_tokens_details": details, "output_tokens": n, "total_tokens": input + n });
    if streaming(body) {
        return stream(kind, body, &image, n, usage, error);
    }
    let base64 = kind.gpt_image || body["response_format"] == "b64_json";
    let data: Vec<Value> = (0..n)
        .map(|_| {
            let mut one = match base64 {
                true => json!({ "b64_json": STANDARD.encode(&image) }),
                false => json!({ "url": ctx.host("image/png", image.clone()) }),
            };
            if kind.revises {
                one["revised_prompt"] = body["prompt"].clone();
            }
            one
        })
        .collect();
    let mut out = json!({ "created": unix_now(), "data": data });
    if kind.gpt_image {
        out["usage"] = usage;
    }
    Out::Json(200, out)
}

/// Named SSE events, per image: `partial_images` `<event>.partial_image`s
/// (0-based `partial_image_index`), then `<event>.completed` with usage.
/// https://developers.openai.com/api/reference/resources/images/generation-streaming-events
/// https://developers.openai.com/api/reference/resources/images/edit-streaming-events
/// The partials carry the scripted image too; the `event:` line naming each
/// by its `type` is the framing commonly seen.
fn stream(kind: &Kind, body: &Value, image: &Bytes, n: u64, usage: Value, error: String) -> Out {
    let or = |k: &str, d: &str| body[k].as_str().unwrap_or(d).to_owned();
    let settings = json!({
        "b64_json": STANDARD.encode(image),
        "created_at": unix_now(),
        "size": or("size", "1024x1024"),
        "quality": or("quality", "auto"),
        "background": or("background", "auto"),
        "output_format": or("output_format", "png"),
    });
    let with = |name: &str, extra: Value| {
        let mut e = json!({ "type": format!("{}.{name}", kind.event) });
        for (k, v) in settings
            .as_object()
            .into_iter()
            .flatten()
            .chain(extra.as_object().into_iter().flatten())
        {
            e[k] = v.clone();
        }
        event(e["type"].as_str().unwrap_or_default(), &e)
    };
    let partials = int(&body["partial_images"]).unwrap_or(0);
    let mut events = Vec::new();
    for _ in 0..n {
        for i in 0..partials {
            events.push(with("partial_image", json!({ "partial_image_index": i })));
        }
        events.push(with("completed", json!({ "usage": usage })));
    }
    Out::Sse(events, error)
}
