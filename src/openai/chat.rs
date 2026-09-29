//! OpenAI Chat Completions: `POST /v1/chat/completions`.
//! https://developers.openai.com/api/reference/resources/chat

use crate::transport::framed;
use serde_json::{Map, Value, json};

use super::chat_models::{Allowed, BASE, MODELS, Model};
use super::{OpenAi, Quirks, bad, object};
use crate::core::misfit;
use crate::util::{at, lookup, pieces, tokens, unix_now, words};
use crate::{Answer, Ctx, Endpoint, Out, Provider, Rejection, Reply, Request, data};

/// Top-level parameters the real API knows; any other is rejected.
/// https://developers.openai.com/api/reference/resources/chat (Body Parameters)
#[rustfmt::skip]
const KNOWN: &[&str] = &[
    "messages", "model", "audio", "frequency_penalty", "function_call", "functions",
    "logit_bias", "logprobs", "max_completion_tokens", "max_tokens", "metadata",
    "modalities", "moderation", "n", "parallel_tool_calls", "prediction",
    "presence_penalty", "prompt_cache_key", "prompt_cache_options",
    "prompt_cache_retention", "reasoning_effort", "response_format", "safety_identifier",
    "seed", "service_tier", "stop", "store", "stream", "stream_options", "temperature",
    "tool_choice", "tools", "top_logprobs", "top_p", "user", "verbosity",
    "web_search_options",
];

const ROLES: &[&str] = &[
    "system",
    "developer",
    "user",
    "assistant",
    "tool",
    "function",
];

pub(crate) struct Chat;

impl Endpoint<OpenAi> for Chat {
    fn route(&self) -> (&'static str, &'static str) {
        ("POST", "/v1/chat/completions")
    }

    fn validate(&self, _: &OpenAi, req: &Request) -> Result<(), Rejection> {
        validate(req)
    }

    fn respond(&self, openai: &OpenAi, req: &Request, answer: &Answer, ctx: &Ctx) -> Out {
        respond(openai, req, answer, ctx.id)
    }
}

fn validate(req: &Request) -> Result<(), Rejection> {
    let body = object(req, KNOWN)?;
    let Some(model) = body.get("model").and_then(Value::as_str) else {
        return Err(bad("you must provide a model parameter", None, None));
    };
    let messages = match body.get("messages") {
        None => {
            let message = "Missing required parameter: 'messages'.";
            return Err(bad(
                message,
                Some("messages"),
                Some("missing_required_parameter"),
            ));
        }
        Some(Value::Array(m)) if m.is_empty() => {
            let message = "Invalid 'messages': empty array. Expected an array with minimum length 1, but got an empty array instead.";
            return Err(bad(message, Some("messages"), Some("empty_array")));
        }
        Some(Value::Array(m)) => m,
        Some(_) => {
            let message = "Invalid type for 'messages': expected an array of objects.";
            return Err(bad(message, Some("messages"), Some("invalid_type")));
        }
    };
    check_messages(messages)?;
    let streaming = body.get("stream") == Some(&Value::Bool(true));
    if !streaming && body.get("stream_options").is_some_and(|o| !o.is_null()) {
        let message = "The 'stream_options' parameter is only allowed when 'stream' is enabled.";
        return Err(bad(message, Some("stream_options"), None));
    }
    check_model(body, &lookup(MODELS, BASE, model))
}

/// The rules of `model`'s row in `chat_models.rs`.
fn check_model(body: &Map<String, Value>, model: &Model) -> Result<(), Rejection> {
    let set = |p: &str| body.get(p).filter(|v| !v.is_null());
    if model.no_max_tokens && set("max_tokens").is_some() {
        let message = "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.";
        return Err(bad(
            message,
            Some("max_tokens"),
            Some("unsupported_parameter"),
        ));
    }
    for p in ["max_completion_tokens", "max_tokens"] {
        match set(p).and_then(Value::as_u64) {
            Some(n) if n > model.max_output => {
                let max = model.max_output;
                let message = format!(
                    "max_tokens is too large: {n}. This model supports at most {max} completion tokens, whereas you provided {n}."
                );
                return Err(bad(message, Some(p), Some("invalid_value")));
            }
            _ => {}
        }
    }
    let asked = set("reasoning_effort").and_then(Value::as_str);
    if let Some(e) = asked.filter(|e| !model.efforts.is_empty() && !model.efforts.contains(e)) {
        let supported = model
            .efforts
            .iter()
            .map(|e| format!("'{e}'"))
            .collect::<Vec<_>>();
        let message = format!(
            "Unsupported value: 'reasoning_effort' does not support '{e}' with this model. Supported values are: {}.",
            supported.join(", ")
        );
        return Err(bad(
            message,
            Some("reasoning_effort"),
            Some("unsupported_value"),
        ));
    }
    let effort_none = asked.or(model.default_effort) == Some("none");
    let allowed =
        |a: Allowed| a == Allowed::Always || (a == Allowed::OnlyEffortNone && effort_none);
    if !allowed(model.sampling) {
        let sampling = ["temperature", "top_p", "logprobs", "top_logprobs"];
        if let Some(p) = sampling.into_iter().find(|p| set(p).is_some()) {
            let message = format!("Unsupported parameter: '{p}' is not supported with this model.");
            return Err(bad(message, Some(p), Some("unsupported_parameter")));
        }
    }
    let tools = set("tools")
        .and_then(Value::as_array)
        .is_some_and(|t| !t.is_empty());
    if tools && !allowed(model.tools) {
        let message = "Unsupported parameter: 'tools' is not supported with this model.";
        return Err(bad(message, Some("tools"), Some("unsupported_parameter")));
    }
    Ok(())
}

/// Roles, and tool messages answering exactly the preceding tool calls.
fn check_messages(messages: &[Value]) -> Result<(), Rejection> {
    let mut open: Vec<String> = Vec::new();
    for (i, m) in messages.iter().enumerate() {
        let role = m["role"].as_str().unwrap_or_default();
        if !ROLES.contains(&role) {
            let message = format!(
                "Invalid value: '{role}'. Supported values are: 'system', 'assistant', 'user', 'function', 'tool', and 'developer'."
            );
            let param = format!("messages[{i}].role");
            return Err(bad(message, Some(&param), Some("invalid_value")));
        }
        if role == "tool" {
            let id = m["tool_call_id"].as_str().unwrap_or_default();
            if !open.iter().any(|o| o == id) {
                let message = "Invalid parameter: messages with role 'tool' must be a response to a preceeding message with 'tool_calls'.";
                return Err(bad(message, Some(&format!("messages.[{i}].role")), None));
            }
            open.retain(|o| o != id);
            continue;
        }
        if !open.is_empty() {
            let message = format!(
                "An assistant message with 'tool_calls' must be followed by tool messages responding to each 'tool_call_id'. The following tool_call_ids did not have response messages: {}",
                open.join(", ")
            );
            return Err(bad(message, Some("messages"), Some("invalid_parameter")));
        }
        let calls = m["tool_calls"]
            .as_array()
            .map(Vec::as_slice)
            .unwrap_or_default();
        open = calls
            .iter()
            .filter_map(|c| c["id"].as_str().map(str::to_owned))
            .collect();
    }
    if open.is_empty() {
        Ok(())
    } else {
        let message = format!(
            "An assistant message with 'tool_calls' must be followed by tool messages responding to each 'tool_call_id'. The following tool_call_ids did not have response messages: {}",
            open.join(", ")
        );
        Err(bad(message, Some("messages"), Some("invalid_parameter")))
    }
}

fn usage(prompt: u64, completion: u64, reasoning: u64) -> Value {
    json!({
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_tokens_details": { "cached_tokens": 0, "audio_tokens": 0 },
        "completion_tokens_details": { "reasoning_tokens": reasoning, "audio_tokens": 0, "accepted_prediction_tokens": 0, "rejected_prediction_tokens": 0 }
    })
}

fn respond(openai: &OpenAi, req: &Request, answer: &Answer, n: u64) -> Out {
    let (Some(body), quirks) = (req.object(), openai.quirks) else {
        return Out::Misfit("mockllm: validated body is not an object".into());
    };
    let streaming = body.get("stream") == Some(&Value::Bool(true));
    let error = data(&openai.error(&openai.scripted(500), n));
    let (text, tool) = match (&answer.reply, answer.reply.text()) {
        (_, Some(t)) => (words(t), None),
        (Reply::ToolCall { name, args }, _) => (Vec::new(), Some((name, args.to_string()))),
        (Reply::Events(events), _) if streaming => {
            return Out::Sse(events.iter().map(|e| framed(e, data)).collect(), error);
        }
        (other, _) => return misfit("chat completions", other),
    };
    let model = lookup(MODELS, BASE, body["model"].as_str().unwrap_or_default());
    // Reasoning is counted by reasoning models, and shown only by providers
    // with the `reasoning_content` quirk.
    let thought = answer.reasoning.as_deref().map(words).unwrap_or_default();
    let shown = if quirks.reasoning_content {
        thought.clone()
    } else {
        Vec::new()
    };
    let reasoning = if model.reasoning || quirks.reasoning_content {
        thought.len() as u64
    } else {
        0
    };
    let call_id = format!("call_fake{n}");
    let finish = if tool.is_some() { "tool_calls" } else { "stop" };
    let choices = body.get("n").and_then(Value::as_u64).unwrap_or(1);
    let prompt = tokens(&body["messages"]);
    let args = tool.as_ref().map(|(_, a)| pieces(a, 8)).unwrap_or_default();
    let visible = (text.len() + args.len()) as u64;
    let total = usage(prompt, (visible + reasoning) * choices, reasoning * choices);
    let mut head = Map::new();
    head.insert("id".into(), format!("chatcmpl-fake{n}").into());
    head.insert("object".into(), "chat.completion".into());
    head.insert("created".into(), unix_now().into());
    head.insert("model".into(), body["model"].clone());

    if !streaming {
        let content = tool.is_none().then(|| text.concat());
        let mut message =
            json!({ "role": "assistant", "content": content, "refusal": null, "annotations": [] });
        if let Some((name, args)) = &tool {
            let call = json!({ "id": call_id, "type": "function", "function": { "name": name, "arguments": args } });
            message["tool_calls"] = json!([call]);
        }
        if !shown.is_empty() {
            message["reasoning_content"] = shown.concat().into();
        }
        let all: Vec<Value> = (0..choices)
            .map(|i| json!({ "index": i, "message": message, "logprobs": null, "finish_reason": finish }))
            .collect();
        head.insert("choices".into(), all.into());
        head.insert("usage".into(), total);
        head.insert("service_tier".into(), "default".into());
        head.insert("system_fingerprint".into(), "fp_fake".into());
        return Out::Json(200, head.into());
    }

    // The deltas of one choice, in order; the last, empty, has the finish.
    let mut deltas = Vec::new();
    match &tool {
        Some((name, _)) => {
            let call = json!({ "index": 0, "id": call_id, "type": "function", "function": { "name": name, "arguments": "" } });
            deltas.push(json!({ "role": "assistant", "content": null, "tool_calls": [call], "refusal": null }));
        }
        None => deltas.push(json!({ "role": "assistant", "content": "", "refusal": null })),
    }
    deltas.extend(
        shown
            .into_iter()
            .map(|r| json!({ "content": null, "reasoning_content": r })),
    );
    deltas.extend(text.into_iter().map(|w| json!({ "content": w })));
    deltas.extend(
        args.into_iter()
            .map(|p| json!({ "tool_calls": [{ "index": 0, "function": { "arguments": p } }] })),
    );
    let last = deltas.len();
    deltas.push(json!({}));

    head.insert("object".into(), "chat.completion.chunk".into());
    head.insert("service_tier".into(), "default".into());
    head.insert("system_fingerprint".into(), "fp_fake".into());
    let options = at(body, "stream_options");
    let include_usage = options["include_usage"] == true;
    // "included by default and omitted when
    // `stream_options.include_obfuscation` is `false`":
    // https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events
    // OpenAI-compatible providers (any quirk) do not obfuscate.
    let obfuscate = options["include_obfuscation"] != false && quirks == Quirks::default();
    let mut frames = Vec::new();
    for (step, delta) in deltas.iter().enumerate() {
        for i in 0..choices {
            let fr = if step == last {
                json!(finish)
            } else {
                Value::Null
            };
            let mut chunk = head.clone();
            let choice =
                json!({ "index": i, "delta": delta, "logprobs": null, "finish_reason": fr });
            chunk.insert("choices".into(), json!([choice]));
            if quirks.usage_every_chunk {
                let so_far = step.min(last - 1) as u64;
                chunk.insert("usage".into(), usage(prompt, so_far, 0));
            } else if include_usage {
                chunk.insert("usage".into(), Value::Null);
            }
            if obfuscate {
                chunk.insert("obfuscation".into(), "x".repeat(step % 7 + 1).into());
            }
            frames.push(data(&chunk.into()));
        }
    }
    if include_usage {
        head.insert("choices".into(), json!([]));
        head.insert("usage".into(), total);
        frames.push(data(&head.into()));
    }
    if !quirks.no_done {
        frames.push("data: [DONE]\n\n".into());
    }
    Out::Sse(frames, error)
}
