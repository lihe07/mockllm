//! OpenAI Images, edits: `POST /v1/images/edits`, multipart (or JSON
//! with image references, GPT image models only).
//! https://developers.openai.com/api/reference/resources/images/methods/edit
//! The JSON form is `EditImageBodyJsonParam` in the OpenAPI spec the
//! reference is built from: https://github.com/openai/openai-openapi

use serde_json::Value;

use super::images::{Kind, check_stream, frame, int};
use super::{OpenAi, bad, object};
use crate::{Answer, Ctx, Endpoint, Out, Rejection, Request};

/// The multipart fields of the reference above (`image` and `mask` are
/// file parts); any other is rejected.
#[rustfmt::skip]
const FORM: &[&str] = &[
    "image", "image[]", "mask", "prompt", "background", "model", "n", "size",
    "response_format", "output_format", "output_compression", "user", "input_fidelity",
    "stream", "partial_images", "quality",
];

/// The JSON body's parameters (`EditImageBodyJsonParam`).
#[rustfmt::skip]
const JSON: &[&str] = &[
    "images", "mask", "prompt", "background", "model", "moderation", "n", "size",
    "output_format", "output_compression", "user", "input_fidelity", "stream",
    "partial_images", "quality",
];

/// "Defaults to `gpt-image-1.5`" (reference above).
fn model(body: &Value) -> &str {
    body["model"].as_str().unwrap_or("gpt-image-1.5")
}

pub(crate) struct ImageEdits;

impl Endpoint<OpenAi> for ImageEdits {
    fn route(&self) -> (&'static str, &'static str) {
        ("POST", "/v1/images/edits")
    }

    fn validate(&self, _: &OpenAi, req: &Request) -> Result<(), Rejection> {
        let form = req
            .header("content-type")
            .is_some_and(|t| t.starts_with("multipart/form-data"));
        let body = object(req, if form { FORM } else { JSON })?;
        let missing = |p: &str| {
            let message = format!("Missing required parameter: '{p}'.");
            bad(message, Some(p), Some("missing_required_parameter"))
        };
        let Some(prompt) = body.get("prompt").and_then(Value::as_str) else {
            return Err(missing("prompt"));
        };
        let model = model(&req.body);
        let dall_e_2 = model.starts_with("dall-e-2");
        // "This endpoint supports GPT Image models and `dall-e-2`"; "JSON
        // edits support GPT image models only; DALL-E edits require
        // multipart" (spec). Wording as commonly seen.
        if model.starts_with("dall-e") && (!dall_e_2 || !form) {
            let message = format!(
                "Invalid value: '{model}'. This model is not supported for image edits in this request format."
            );
            return Err(bad(message, Some("model"), Some("invalid_value")));
        }
        let images: Vec<_> = req
            .files
            .iter()
            .filter(|f| f.field == "image" || f.field == "image[]")
            .collect();
        let count = match form {
            true => images.len(),
            false => body
                .get("images")
                .and_then(Value::as_array)
                .map_or(0, Vec::len),
        };
        if count == 0 {
            return Err(missing(if form { "image" } else { "images" }));
        }
        // "For `dall-e-2`, you can only provide one image"; GPT image
        // models: "You can provide up to 16 images". Wording as commonly
        // seen.
        let max = if dall_e_2 { 1 } else { 16 };
        if count > max {
            let message = format!(
                "Invalid 'image': array too long. Expected an array with maximum length {max}, but got an array with length {count} instead."
            );
            return Err(bad(message, Some("image"), Some("array_above_max_length")));
        }
        // dall-e-2: "a square `png`"; GPT image models: "a `png`, `webp`,
        // or `jpg` file"; the mask "Must be a valid PNG file". Wording as
        // commonly seen.
        let types: &[&str] = if dall_e_2 {
            &["png"]
        } else {
            &["png", "webp", "jpg", "jpeg"]
        };
        let masks = req
            .files
            .iter()
            .filter(|f| f.field == "mask")
            .map(|f| (f, &["png"][..]));
        for (file, allowed) in images.into_iter().map(|f| (f, types)).chain(masks) {
            let kind = file
                .content_type
                .as_deref()
                .and_then(|t| t.strip_prefix("image/"));
            let ext = file
                .filename
                .rsplit_once('.')
                .map(|(_, e)| e.to_lowercase());
            let kind = kind.map(str::to_owned).or(ext).unwrap_or_default();
            if !allowed.contains(&kind.as_str()) {
                let supported: Vec<String> =
                    allowed.iter().map(|t| format!("'image/{t}'")).collect();
                let message = format!(
                    "Invalid file '{}': unsupported mimetype ('image/{kind}'). Supported file formats are {}.",
                    file.field,
                    supported.join(", ")
                );
                return Err(bad(
                    message,
                    Some(file.field.as_str()),
                    Some("invalid_value"),
                ));
            }
        }
        // "The maximum length is 1000 characters for `dall-e-2`, and 32000
        // characters for the GPT image models."
        let (limit, len) = (if dall_e_2 { 1000 } else { 32000 }, prompt.chars().count());
        if len > limit {
            let message = format!(
                "Invalid 'prompt': string too long. Expected a string with maximum length {limit}, but got a string with length {len} instead."
            );
            return Err(bad(
                message,
                Some("prompt"),
                Some("string_above_max_length"),
            ));
        }
        // "Must be between 1 and 10."
        let n = &req.body["n"];
        if !n.is_null() && !int(n).is_some_and(|n| (1..=10).contains(&n)) {
            let shown = n.as_str().map_or(n.to_string(), str::to_owned);
            let message = format!("Invalid 'n': {shown}. Expected a value between 1 and 10.");
            return Err(bad(message, Some("n"), Some("invalid_value")));
        }
        // "For `dall-e-2`, use one of `256x256`, `512x512`, or `1024x1024`."
        let size = body.get("size").and_then(Value::as_str);
        let sizes = ["256x256", "512x512", "1024x1024"];
        if let Some(s) = size.filter(|s| dall_e_2 && !sizes.contains(s)) {
            let message = format!(
                "Invalid value: '{s}'. Supported values are: {}.",
                sizes.join(", ")
            );
            return Err(bad(message, Some("size"), Some("invalid_value")));
        }
        // `response_format` "is only supported for `dall-e-2`, as GPT image
        // models always return base64-encoded images".
        if !dall_e_2 && body.get("response_format").is_some_and(|f| !f.is_null()) {
            let message = "Unknown parameter: 'response_format'.";
            return Err(bad(
                message,
                Some("response_format"),
                Some("unknown_parameter"),
            ));
        }
        check_stream(&req.body, !dall_e_2)
    }

    fn respond(&self, openai: &OpenAi, req: &Request, answer: &Answer, ctx: &Ctx) -> Out {
        let gpt_image = !model(&req.body).starts_with("dall-e");
        let kind = Kind {
            name: "image edits",
            event: "image_edit",
            gpt_image,
            revises: false,
        };
        frame(openai, &kind, &req.body, answer, ctx)
    }
}
