//! Gemini Interactions API: `POST /v1beta/interactions` (and `/v1`), an
//! `Interaction` as JSON, or with `"stream": true` as named SSE events.
//! Reference: https://ai.google.dev/api/interactions-api (OpenAPI spec:
//! https://ai.google.dev/static/api/interactions.openapi.json)
//! Streaming: https://ai.google.dev/gemini-api/docs/interactions/streaming
//!
//! No error wording is documented; the messages here only name the rule.
//! Errors take the provider's shape (`{"error": {"code", "message",
//! "status"}}`); the OpenAPI spec gives `{"error": {"code", "message"}}`
//! with a string `code`, which is not confirmed for HTTP errors.

use base64::Engine;
use serde_json::{Map, Value, json};

use super::{Gemini, bad};
use crate::core::misfit;
use crate::transport::event;
use crate::util::{next_id, pieces, tokens, unix_now, words};
use crate::{Answer, Ctx, Endpoint, Out, Provider, Rejection, Reply, Request};

/// Fields of both `ModelInteraction` and `AgentInteraction`, input and
/// output (spec, `components.schemas`); any field in neither is rejected.
#[rustfmt::skip]
const COMMON: &[&str] = &[
    "input", "background", "environment", "labels", "previous_interaction_id",
    "response_format", "response_mime_type", "response_modalities", "safety_settings",
    "service_tier", "store", "stream", "system_instruction", "tools", "webhook_config",
    "id", "object", "status", "created", "updated", "steps", "usage", "environment_id", "errors",
];
const MODEL: &[&str] = &["model", "generation_config", "cached_content"];
const AGENT: &[&str] = &["agent", "agent_config"];

/// The mid-stream `error` event, the streaming guide's example.
const ERROR: &str = r#"{"error":{"message":"Deadline expired before operation could complete.","code":"gateway_timeout"},"event_type":"error"}"#;

pub(crate) struct Interactions;

/// `value` is absent or one of `allowed`.
fn one_of(value: &Value, name: &str, allowed: &[&str]) -> Result<(), Rejection> {
    match value.as_str() {
        _ if value.is_null() => Ok(()),
        Some(v) if allowed.contains(&v) => Ok(()),
        _ => Err(bad(format!(
            "Invalid value for {name}: {value}. Supported values: {}.",
            allowed.join(", ")
        ))),
    }
}

impl Endpoint<Gemini> for Interactions {
    fn route(&self) -> (&'static str, &'static str) {
        ("POST", "/{version}/interactions")
    }

    fn validate(&self, gemini: &Gemini, req: &Request) -> Result<(), Rejection> {
        // "Endpoints are under /v1beta/. The stable v1 version is also
        // available." (reference)
        if !matches!(req.param.as_deref(), Some("v1beta" | "v1")) {
            return Err(gemini.not_found(req));
        }
        let Some(body) = req.object() else {
            return Err(bad("Invalid JSON payload received."));
        };
        // The body is `oneOf` `AgentInteraction` (`agent` required) or
        // `ModelInteraction` (`model` required).
        let own = match (body.get("model"), body.get("agent")) {
            (Some(_), None) => MODEL,
            (None, Some(_)) => AGENT,
            (Some(_), Some(_)) => return Err(bad("Only one of model or agent may be set.")),
            (None, None) => return Err(bad("One of model or agent is required.")),
        };
        let known = |k: &String| COMMON.contains(&k.as_str()) || own.contains(&k.as_str());
        if let Some(k) = body.keys().find(|k| !known(k)) {
            return Err(bad(format!("Unknown field: {k}.")));
        }
        // `response_mime_type`: "This is required if response_format is set."
        if body.contains_key("response_format") && !body.contains_key("response_mime_type") {
            return Err(bad(
                "response_mime_type is required if response_format is set.",
            ));
        }
        // Enums of `ServiceTier`, `ThinkingLevel`, `ThinkingSummaries`.
        let tiers = ["flex", "standard", "priority", "deferred"];
        one_of(&req.body["service_tier"], "service_tier", &tiers)?;
        let config = &req.body["generation_config"];
        let levels = ["minimal", "low", "medium", "high"];
        one_of(&config["thinking_level"], "thinking_level", &levels)?;
        one_of(
            &config["thinking_summaries"],
            "thinking_summaries",
            &["auto", "none"],
        )?;
        Ok(())
    }

    fn respond(&self, _: &Gemini, req: &Request, answer: &Answer, _: &Ctx) -> Out {
        let body = req.object().cloned().unwrap_or_default();
        let stream = body.get("stream") == Some(&Value::Bool(true));
        // A named SSE event per `event_type`.
        let frame = |v: &Value| event(v["event_type"].as_str().unwrap_or("message"), v);
        if let (Reply::Events(events), true) = (&answer.reply, stream) {
            return Out::Sse(
                events.iter().map(frame).collect(),
                format!("event: error\ndata: {ERROR}\n\n"),
            );
        }
        let Some(steps) = steps(&body, answer) else {
            return misfit("interactions", &answer.reply);
        };
        // "function_call ... Sets interaction status to requires_action."
        let calls = steps.iter().any(|s| s.done["type"] == "function_call");
        let status = if calls {
            "requires_action"
        } else {
            "completed"
        };
        let input =
            body.get("input").map_or(0, tokens) + body.get("system_instruction").map_or(0, tokens);
        let output: usize = steps
            .iter()
            .filter(|s| s.done["type"] != "thought")
            .map(|s| s.deltas.len())
            .sum();
        let thought = answer.reasoning.as_deref().map_or(0, |r| words(r).len());
        let (output, thought) = (output as u64, thought as u64);
        // `Usage`, as in the reference's examples.
        let usage = json!({
            "input_tokens_by_modality": [{ "modality": "text", "tokens": input }],
            "total_cached_tokens": 0, "total_input_tokens": input, "total_output_tokens": output,
            "total_thought_tokens": thought, "total_tokens": input + output + thought, "total_tool_use_tokens": 0
        });
        let id = format!("v1_fake{}", next_id());
        let now = iso(unix_now());
        let (key, name) = match body.get("model") {
            Some(m) => ("model", m.clone()),
            None => ("agent", body.get("agent").cloned().unwrap_or_default()),
        };
        let mut interaction = Map::new();
        for (k, v) in [
            ("created", json!(now)),
            ("id", json!(id)),
            (key, name.clone()),
        ] {
            interaction.insert(k.into(), v);
        }
        interaction.insert("object".into(), json!("interaction"));
        interaction.insert("status".into(), json!(status));
        if !stream {
            let done: Vec<&Value> = steps.iter().map(|s| &s.done).collect();
            interaction.insert("steps".into(), json!(done));
            interaction.insert("updated".into(), json!(now));
            interaction.insert("usage".into(), usage);
            return Out::Json(200, interaction.into());
        }
        // The streaming guide's event flow: `interaction.created`, a
        // `step.start`, `step.delta`s, `step.stop` per step, then
        // `interaction.completed` (no steps) and `data: [DONE]`.
        let created = json!({ "interaction": { "id": id, "status": "in_progress", "object": "interaction", key: name }, "event_type": "interaction.created" });
        let update = json!({ "interaction_id": id, "status": "in_progress", "event_type": "interaction.status_update" });
        let mut events = vec![frame(&created), frame(&update)];
        for (index, s) in steps.iter().enumerate() {
            events.push(frame(
                &json!({ "index": index, "step": s.start, "event_type": "step.start" }),
            ));
            for d in &s.deltas {
                events.push(frame(
                    &json!({ "index": index, "delta": d, "event_type": "step.delta" }),
                ));
            }
            events.push(frame(&json!({ "index": index, "event_type": "step.stop" })));
        }
        let completed = json!({ "interaction": { "id": id, "status": status, "usage": usage, "created": now, "updated": now, "object": "interaction", key: name }, "event_type": "interaction.completed" });
        events.push(frame(&completed));
        events.push("event: done\ndata: [DONE]\n\n".into());
        Out::Sse(events, format!("event: error\ndata: {ERROR}\n\n"))
    }
}

/// One step: how its `step.start` opens, its deltas, and the whole step
/// of a non-streamed `Interaction`.
struct Step {
    start: Value,
    deltas: Vec<Value>,
    done: Value,
}

/// The steps of `answer`, `None` for a kind the endpoint cannot answer.
fn steps(body: &Map<String, Value>, answer: &Answer) -> Option<Vec<Step>> {
    let mut steps = Vec::new();
    if let Some(reasoning) = answer.reasoning.as_deref() {
        // `thought`: `thought_signature`, and `thought_summary` deltas
        // "only present when thinking_summaries is enabled".
        let summaries = body
            .get("generation_config")
            .and_then(|c| c.get("thinking_summaries"))
            == Some(&json!("auto"));
        let signature = format!("fake-signature-{}", next_id());
        let mut deltas: Vec<Value> = Vec::new();
        let mut done = json!({ "type": "thought", "signature": signature });
        if summaries {
            let text = |t: &str| json!({ "type": "text", "text": t });
            deltas.extend(
                words(reasoning)
                    .iter()
                    .map(|w| json!({ "type": "thought_summary", "content": text(w) })),
            );
            done["summary"] = json!([text(reasoning)]);
        }
        deltas.push(json!({ "type": "thought_signature", "signature": signature }));
        steps.push(Step {
            start: json!({ "type": "thought" }),
            deltas,
            done,
        });
    }
    let output = json!({ "type": "model_output" });
    let step = match (&answer.reply, answer.reply.text()) {
        (_, Some(text)) => Step {
            start: output,
            deltas: words(text)
                .iter()
                .map(|w| json!({ "type": "text", "text": w }))
                .collect(),
            done: json!({ "type": "model_output", "content": [{ "type": "text", "text": text }] }),
        },
        // `function_call`: "the function name, id and empty arguments {}",
        // then `arguments_delta` JSON pieces.
        (Reply::ToolCall { name, args }, _) => {
            let id = format!("fc_fake{}", next_id());
            let deltas = pieces(&args.to_string(), 16);
            Step {
                start: json!({ "type": "function_call", "id": id, "name": name, "arguments": {} }),
                deltas: deltas
                    .iter()
                    .map(|a| json!({ "type": "arguments_delta", "arguments": a }))
                    .collect(),
                done: json!({ "type": "function_call", "id": id, "name": name, "arguments": args }),
            }
        }
        // `image`: "Base64-encoded image data from a model_output step".
        (Reply::Image(bytes), _) => {
            let data = base64::engine::general_purpose::STANDARD.encode(bytes);
            let image = json!({ "type": "image", "mime_type": "image/png", "data": data });
            Step {
                start: output,
                deltas: vec![image.clone()],
                done: json!({ "type": "model_output", "content": [image] }),
            }
        }
        _ => return None,
    };
    steps.push(step);
    Some(steps)
}

/// `secs` since the epoch as `YYYY-MM-DDThh:mm:ssZ`, the reference's
/// timestamp format (days to date by H. Hinnant's `civil_from_days`).
fn iso(secs: u64) -> String {
    let (days, rest) = ((secs / 86_400) as i64 + 719_468, secs % 86_400);
    let era = days.div_euclid(146_097);
    let doe = days - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    let (h, m, s) = (rest / 3600, rest % 3600 / 60, rest % 60);
    format!("{year:04}-{month:02}-{day:02}T{h:02}:{m:02}:{s:02}Z")
}

#[cfg(test)]
mod tests {
    #[test]
    fn iso_dates() {
        assert_eq!(super::iso(0), "1970-01-01T00:00:00Z");
        assert_eq!(super::iso(1_764_159_915), "2025-11-26T12:25:15Z");
    }
}
