//! Gemini `generateContent` and `streamGenerateContent`:
//! `POST /v1beta/models/{model}:generateContent`, and
//! `:streamGenerateContent`, as SSE with `?alt=sse`.
//! https://ai.google.dev/api/generate-content
//!
//! Error wording the docs do not give is the wording commonly seen.

use serde_json::{Value, json};

use super::generate_models::{BASE, MODELS};
use super::{Gemini, bad};
use crate::core::misfit;
use crate::util::{at, lookup, next_id, tokens, words};
use crate::{Answer, Ctx, Endpoint, Out, Provider, Rejection, Reply, Request};

/// The `GenerateContentRequest` fields of the reference; any other is
/// rejected.
#[rustfmt::skip]
const KNOWN: &[&str] = &[
    "contents", "tools", "toolConfig", "safetySettings", "labels", "systemInstruction",
    "generationConfig", "cachedContent", "serviceTier", "store",
];

pub(crate) struct Generate {
    pub stream: bool,
}

impl Endpoint<Gemini> for Generate {
    fn route(&self) -> (&'static str, &'static str) {
        match self.stream {
            false => ("POST", "/v1beta/models/{model}:generateContent"),
            true => ("POST", "/v1beta/models/{model}:streamGenerateContent"),
        }
    }

    fn validate(&self, _: &Gemini, req: &Request) -> Result<(), Rejection> {
        let Some(body) = req.object() else {
            return Err(bad("Invalid JSON payload received."));
        };
        if let Some(k) = body.keys().find(|k| !KNOWN.contains(&k.as_str())) {
            let message =
                format!("Invalid JSON payload received. Unknown name \"{k}\": Cannot find field.");
            return Err(bad(message));
        }
        // `contents[]` is "Required".
        let contents = body.get("contents").and_then(Value::as_array);
        let Some(contents) = contents.filter(|c| !c.is_empty()) else {
            return Err(bad(
                "* GenerateContentRequest.contents: contents is not specified\n",
            ));
        };
        // `role`: "Must be either 'user' or 'model'."
        let role = |c: &Value| {
            c.get("role")
                .and_then(Value::as_str)
                .unwrap_or("user")
                .to_owned()
        };
        if contents
            .iter()
            .map(role)
            .any(|r| r != "user" && r != "model")
        {
            return Err(bad("Please use a valid role: user, model."));
        }
        let model = lookup(MODELS, BASE, req.param.as_deref().unwrap_or_default());
        let config = at(body, "generationConfig");
        if let Some(n) = config["maxOutputTokens"]
            .as_u64()
            .filter(|n| *n > model.max_output)
        {
            let end = model.max_output + 1;
            let message = format!(
                "Unable to submit request because it has a maxOutputTokens value of {n} but the supported range is from 1 (inclusive) to {end} (exclusive). Update the value and try again."
            );
            return Err(bad(message));
        }
        let level = config["thinkingConfig"]["thinkingLevel"].as_str();
        if let (Some(level), Some(levels)) = (level, model.levels) {
            let lower = level.to_ascii_lowercase();
            if levels.is_empty() {
                return Err(bad("Thinking level is not supported for this model."));
            }
            if !levels.contains(&lower.as_str()) {
                let message = format!(
                    "Thinking level {level} is not supported for this model. Supported levels: {}.",
                    levels.join(", ")
                );
                return Err(bad(message));
            }
        }
        Ok(())
    }

    fn respond(&self, gemini: &Gemini, req: &Request, answer: &Answer, _: &Ctx) -> Out {
        let Some(body) = req.object() else {
            return Out::Misfit("mockllm: validated body is not an object".into());
        };
        let sse = req.query_param("alt") == Some("sse");
        let model = req.param.clone().unwrap_or_default();
        let row = lookup(MODELS, BASE, &model);
        // With `?alt=sse`, standard SSE `data:` events; Google ends each
        // with CRLF CRLF, one of the delimiters the SDK splits on:
        // https://github.com/googleapis/js-genai/blob/main/src/_api_client.ts
        let frame = |v: &Value| format!("data: {v}\r\n\r\n");
        // A mid-stream error is the bare error JSON, which the same SDK
        // parses from a chunk before looking for `data:`.
        let error = gemini.error(&gemini.scripted(500), 0).to_string();
        let parts: Vec<Value> = match (&answer.reply, answer.reply.text()) {
            (Reply::Events(events), _) if self.stream => {
                return Out::Sse(events.iter().map(frame).collect(), error);
            }
            (_, Some("")) => vec![json!({ "text": "" })],
            (_, Some(t)) => words(t).into_iter().map(|w| json!({ "text": w })).collect(),
            // "A predicted FunctionCall ... with the arguments and their values."
            (Reply::ToolCall { name, args }, _) => {
                vec![json!({ "functionCall": { "name": name, "args": args } })]
            }
            (other, _) => return misfit("generateContent", other),
        };
        let config = at(body, "generationConfig");
        let include = config["thinkingConfig"]["includeThoughts"] == true;
        let thought = answer.reasoning.as_deref().map(words).unwrap_or_default();
        let thoughts = if row.thinks { thought.len() as u64 } else { 0 };
        // `thought`: "Indicates if the part is thought from the model."
        let shown: Vec<Value> = match include && thoughts > 0 {
            true => thought
                .iter()
                .map(|w| json!({ "text": w, "thought": true }))
                .collect(),
            false => Vec::new(),
        };
        let prompt = tokens(&body["contents"]) + body.get("systemInstruction").map_or(0, tokens);
        let output = parts.len() as u64;
        let mut usage = json!({ "promptTokenCount": prompt, "candidatesTokenCount": output, "totalTokenCount": prompt + output + thoughts });
        if thoughts > 0 {
            usage["thoughtsTokenCount"] = thoughts.into();
        }
        let n = next_id();
        let response = |parts: Vec<Value>, last: bool| {
            let mut candidate =
                json!({ "content": { "parts": parts, "role": "model" }, "index": 0 });
            let mut v = json!({ "candidates": [], "modelVersion": model, "responseId": format!("fake{n}") });
            if last {
                candidate["finishReason"] = "STOP".into();
                v["usageMetadata"] = usage.clone();
            }
            v["candidates"] = json!([candidate]);
            v
        };
        if !self.stream {
            // Whole: one part per kind, the text joined.
            let text: String = parts.iter().filter_map(|p| p["text"].as_str()).collect();
            let visible = if text.is_empty() {
                parts.clone()
            } else {
                vec![json!({ "text": text })]
            };
            let thought: String = shown.iter().filter_map(|p| p["text"].as_str()).collect();
            let mut all: Vec<Value> = Vec::new();
            if !thought.is_empty() {
                all.push(json!({ "text": thought, "thought": true }));
            }
            all.extend(visible);
            return Out::Json(200, response(all, true));
        }
        // Streamed: a whole `GenerateContentResponse` per part; the last
        // has the `finishReason` and `usageMetadata`.
        let all: Vec<Value> = shown.into_iter().chain(parts).collect();
        let last = all.len().saturating_sub(1);
        let chunks: Vec<Value> = all
            .into_iter()
            .enumerate()
            .map(|(i, p)| response(vec![p], i == last))
            .collect();
        if sse {
            return Out::Sse(chunks.iter().map(frame).collect(), error);
        }
        // Without `alt=sse`, one JSON array, sent a response at a time.
        let mut pieces: Vec<bytes::Bytes> = Vec::new();
        for (i, c) in chunks.iter().enumerate() {
            let lead = if i == 0 { "[" } else { ",\r\n" };
            pieces.push(format!("{lead}{c}").into());
        }
        pieces.push("]".into());
        Out::Chunks("application/json; charset=UTF-8".into(), pieces)
    }
}
