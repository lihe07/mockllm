//! A mock Anthropic Messages: `POST /v1/messages`.

use std::sync::Arc;

use axum::response::Response;
use serde_json::{Map, Value, json};

use crate::anthropic_models::{Model, lookup};
use crate::{Answer, Out, Reply, Request, Script, Server, next_id, pieces, send, tokens, words};

/// Top-level fields the real API knows; any other is "Extra inputs are
/// not permitted". https://platform.claude.com/docs/en/api/messages/create
#[rustfmt::skip]
const KNOWN: &[&str] = &[
    "max_tokens", "messages", "model", "cache_control", "container", "diagnostics",
    "inference_geo", "metadata", "output_config", "service_tier", "stop_sequences",
    "stream", "system", "thinking", "tool_choice", "tools", "temperature", "top_k",
    "top_p",
];

/// A mock Anthropic Messages provider.
#[derive(Debug)]
pub struct FakeAnthropic {
    server: Server,
}

impl FakeAnthropic {
    pub async fn start() -> Self {
        let server = Server::start(handle).await;
        Self { server }
    }

    /// `http://127.0.0.1:<port>`, without the version, like
    /// `https://api.anthropic.com`.
    pub fn base_url(&self) -> String {
        self.server.origin()
    }

    /// Queues `answer` for the next valid request.
    pub fn answer(&self, answer: Answer) {
        self.server.answer(answer);
    }

    /// Every request received so far, valid or not.
    pub fn requests(&self) -> Vec<Request> {
        self.server.requests()
    }
}

/// A request the real API would reject: status, error type, message.
type Bad = (u16, &'static str, String);

fn bad(message: impl Into<String>) -> Bad {
    (400, "invalid_request_error", message.into())
}

/// The error shape, with `request_id`: https://platform.claude.com/docs/en/api/errors
fn error_body(kind: &str, message: &str, id: u64) -> Value {
    let error = json!({ "type": kind, "message": message });
    json!({ "type": "error", "error": error, "request_id": format!("req_fake{id}") })
}

async fn handle(req: Request, script: Arc<Script>) -> Response {
    let result = if req.method != "POST" || req.path != "/v1/messages" {
        Err((404, "not_found_error", "Not Found".to_owned()))
    } else {
        validate(&req)
    };
    match result {
        Ok((body, show)) => respond(body, show, script.next()).await,
        Err((status, kind, message)) => {
            let n = next_id();
            let headers = [("request-id", format!("req_fake{n}"))];
            let body = error_body(kind, &message, n);
            send(&Answer::default(), &headers, Out::Json(status, body)).await
        }
    }
}

/// The body, and whether a thinking block shows its text (`None` when
/// the model does not think on this request).
fn validate(req: &Request) -> Result<(&Map<String, Value>, Option<bool>), Bad> {
    if req.header("x-api-key").is_none() {
        let message = "x-api-key header is required".to_owned();
        return Err((401, "authentication_error", message));
    }
    if req.header("anthropic-version").is_none() {
        return Err(bad("anthropic-version: header is required"));
    }
    let Some(body) = req.body.as_object() else {
        return Err(bad("There was an error parsing the body of your request."));
    };
    if let Some(k) = body.keys().find(|k| !KNOWN.contains(&k.as_str())) {
        return Err(bad(format!("{k}: Extra inputs are not permitted")));
    }
    for field in ["model", "max_tokens", "messages"] {
        if !body.contains_key(field) {
            return Err(bad(format!("{field}: Field required")));
        }
    }
    if let Some(t) = body.get("temperature").and_then(Value::as_f64) {
        if t > 1.0 {
            return Err(bad("temperature: Input should be less than or equal to 1"));
        }
        if t < 0.0 {
            return Err(bad(
                "temperature: Input should be greater than or equal to 0",
            ));
        }
    }
    let Some(messages) = body["messages"].as_array() else {
        return Err(bad("messages: Input should be a valid list"));
    };
    // Consecutive same-role turns are combined, not rejected (see the
    // `messages` parameter in the reference above).
    check_messages(messages)?;
    let model = body["model"].as_str().unwrap_or_default();
    let beta = req.header("anthropic-beta").unwrap_or_default();
    let show = check_model(body, beta, model, &lookup(model))?;
    Ok((body, show))
}

fn check_messages(messages: &[Value]) -> Result<(), Bad> {
    let mut open: Vec<String> = Vec::new();
    for (i, m) in messages.iter().enumerate() {
        let role = m["role"].as_str().unwrap_or_default();
        match role {
            "system" => {
                return Err(bad(
                    "messages: Unexpected role \"system\". The Messages API accepts a top-level `system` parameter, not \"system\" as an input message role.",
                ));
            }
            "user" | "assistant" => {}
            _ => {
                return Err(bad(format!(
                    "messages.{i}.role: Input should be 'user' or 'assistant'"
                )));
            }
        }
        let content = &m["content"];
        let empty = content.as_str() == Some("") || content.as_array().is_some_and(Vec::is_empty);
        if empty && !(i + 1 == messages.len() && role == "assistant") {
            return Err(bad(format!(
                "messages.{i}: all messages must have non-empty content except for the optional final assistant message"
            )));
        }
        let answered = of(content, "tool_result", "tool_use_id");
        if let Some(missing) = open.iter().find(|id| !answered.contains(id)) {
            return Err(bad(format!(
                "messages.{i}: `tool_use` ids were found without `tool_result` blocks immediately after: {missing}. Each `tool_use` block must have a corresponding `tool_result` block in the next message."
            )));
        }
        open = of(content, "tool_use", "id");
    }
    Ok(())
}

/// The `key` of every block of type `kind` in `content`.
fn of(content: &Value, kind: &str, key: &str) -> Vec<String> {
    let blocks = content.as_array().map(Vec::as_slice).unwrap_or_default();
    let typed = blocks.iter().filter(|b| b["type"] == kind);
    typed
        .filter_map(|b| b[key].as_str().map(str::to_owned))
        .collect()
}

/// The rules of `model`'s row in [`crate::anthropic_models`], and the
/// thinking rules of every model. Wording marked (docs) is quoted from
/// the docs; the rest is the wording commonly seen from the API.
fn check_model(
    body: &Map<String, Value>,
    beta: &str,
    id: &str,
    model: &Model,
) -> Result<Option<bool>, Bad> {
    let set = |p: &str| body.get(p).filter(|v| !v.is_null());
    let max_tokens = body["max_tokens"].as_u64().unwrap_or_default();
    if max_tokens > model.max_output {
        let max = model.max_output;
        return Err(bad(format!(
            "max_tokens: {max_tokens} > {max}, which is the maximum allowed number of output tokens for {id}"
        )));
    }
    // "prompt is too long" (docs, context windows).
    let input = tokens(&body["messages"]) + set("system").map_or(0, tokens);
    if input > model.context {
        return Err(bad(format!(
            "prompt is too long: {input} tokens > {} maximum",
            model.context
        )));
    }
    let thinking = set("thinking");
    let kind = thinking.and_then(|t| t["type"].as_str());
    if let Some(k) = kind.filter(|k| !model.thinking.contains(k)) {
        let adaptive = "Use \"thinking.type.adaptive\" and \"output_config.effort\" to control thinking behavior.";
        // All (docs).
        return Err(bad(match k {
            "enabled" => format!("\"thinking.type.enabled\" is not supported for this model. {adaptive}"),
            "adaptive" => "adaptive thinking is not supported on this model".to_owned(),
            "disabled" if model.thinking.contains(&"between_tools") => "\"thinking.type.disabled\" is not supported for this model. Use \"thinking.type.between_tools\" for the lowest thinking setting, or \"thinking.type.adaptive\" and \"output_config.effort\" to control thinking behavior.".to_owned(),
            "disabled" => format!("\"thinking.type.disabled\" is not supported for this model. {adaptive}"),
            _ => format!("\"thinking.type.{k}\" is not supported for this model."),
        }));
    }
    let effort = crate::at(body, "output_config")["effort"]
        .as_str()
        .unwrap_or_default();
    if kind == Some("between_tools") && matches!(effort, "xhigh" | "max") {
        // (docs)
        return Err(bad(format!(
            "output_config.effort '{effort}' is not supported when thinking is disabled on this model. Use effort 'high' or below, or enable thinking."
        )));
    }
    let on =
        matches!(kind, Some("enabled" | "adaptive")) || (kind.is_none() && model.thinks_by_default);
    let forced = matches!(
        crate::at(body, "tool_choice")["type"].as_str(),
        Some("any" | "tool")
    );
    if forced && model.no_forced_tools {
        // (docs)
        return Err(bad(
            "tool_choice: type \"tool\" and \"any\" are not supported for this model.",
        ));
    }
    let messages = body["messages"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or_default();
    let prefill = messages.last().is_some_and(|m| m["role"] == "assistant");
    if prefill && model.no_prefill {
        // (docs)
        return Err(bad(
            "This model does not support assistant message prefill. The conversation must end with a user message.",
        ));
    }
    if kind == Some("enabled") {
        // Budget: "Minimum of 1,024 tokens" and "Less than `max_tokens`",
        // except with interleaved thinking; forced tool use is
        // incompatible with manual extended thinking; "the final assistant
        // turn of a thinking-enabled request must begin with a thinking block".
        // https://platform.claude.com/docs/en/build-with-claude/extended-thinking
        let budget = thinking
            .and_then(|t| t["budget_tokens"].as_u64())
            .unwrap_or_default();
        if budget < 1024 {
            return Err(bad(
                "thinking.enabled.budget_tokens: Input should be greater than or equal to 1024",
            ));
        }
        if budget >= max_tokens && !beta.contains("interleaved-thinking-2025-05-14") {
            return Err(bad(
                "`max_tokens` must be greater than `thinking.budget_tokens`.",
            ));
        }
        if forced {
            return Err(bad(
                "Thinking may not be enabled when tool_choice forces tool use.",
            ));
        }
        let looping = messages
            .last()
            .is_some_and(|m| !of(&m["content"], "tool_result", "tool_use_id").is_empty());
        if let Some(i) = messages.len().checked_sub(2).filter(|_| looping) {
            let first = &messages[i]["content"][0]["type"];
            if first != "thinking" && first != "redacted_thinking" {
                return Err(bad(format!(
                    "messages.{i}.content.0.type: Expected `thinking` or `redacted_thinking`, but found `{}`. When `thinking` is enabled, a final `assistant` message must start with a thinking block.",
                    first.as_str().unwrap_or("text")
                )));
            }
        }
    }
    if on && prefill {
        // "You can't prefill the assistant response while thinking is on" (L).
        return Err(bad(
            "Prefilling the assistant response is not supported when thinking is enabled.",
        ));
    }
    let (temperature, top_p, top_k) = (set("temperature"), set("top_p"), set("top_k"));
    let not = |v: Option<&Value>, default: f64| v.is_some_and(|v| v.as_f64() != Some(default));
    if model.fixed_sampling {
        // "non-default `temperature`, `top_p`, or `top_k` values return a
        // 400 error on every request" (L).
        let p = [
            ("temperature", not(temperature, 1.0)),
            ("top_p", not(top_p, 1.0)),
            ("top_k", top_k.is_some()),
        ];
        if let Some((p, _)) = p.into_iter().find(|(_, bad)| *bad) {
            return Err(bad(format!(
                "{p}: only the default value is supported for this model."
            )));
        }
    } else if on {
        // "`temperature` and `top_k` are incompatible with thinking, and
        // `top_p` is allowed at values between 0.95 and 1" (L).
        if not(temperature, 1.0) {
            return Err(bad(
                "`temperature` may only be set to 1 when thinking is enabled or in interleaved thinking mode.",
            ));
        }
        if top_k.is_some() {
            return Err(bad("`top_k` must be unset when thinking is enabled."));
        }
        if top_p.and_then(Value::as_f64).is_some_and(|p| p < 0.95) {
            return Err(bad(
                "`top_p` must be greater than or equal to 0.95 or unset when thinking is enabled.",
            ));
        }
    }
    if model.one_sampler && temperature.is_some() && top_p.is_some() {
        return Err(bad(
            "`temperature` and `top_p` cannot both be specified for this model. Please use only one.",
        ));
    }
    let display = thinking.and_then(|t| t["display"].as_str());
    let show = display.map_or(!model.display_omitted, |d| d == "summarized");
    Ok(on.then_some(show))
}

fn event(name: &str, data: Value) -> String {
    format!("event: {name}\ndata: {data}\n\n")
}

fn scripted_error(status: u16) -> (&'static str, &'static str) {
    match status {
        401 => ("authentication_error", "invalid x-api-key"),
        402 => (
            "billing_error",
            "Your credit balance is too low to access the Anthropic API.",
        ),
        403 => (
            "permission_error",
            "Your API key does not have permission to use the specified resource.",
        ),
        404 => (
            "not_found_error",
            "The requested resource could not be found.",
        ),
        413 => (
            "request_too_large",
            "Request exceeds the maximum allowed number of bytes.",
        ),
        429 => (
            "rate_limit_error",
            "Number of request tokens has exceeded your per-minute rate limit.",
        ),
        504 => ("timeout_error", "Request timed out."),
        529 => ("overloaded_error", "Overloaded"),
        s if s >= 500 => ("api_error", "Internal server error"),
        _ => ("invalid_request_error", "Invalid request"),
    }
}

/// One content block: whole, opened empty for a stream, and its deltas.
struct Block {
    whole: Value,
    start: Value,
    deltas: Vec<Value>,
}

async fn respond(body: &Map<String, Value>, show: Option<bool>, answer: Answer) -> Response {
    let n = next_id();
    let headers = [("request-id", format!("req_fake{n}"))];
    let mut blocks = Vec::new();
    let mut thought = 0;
    if let (Some(show), Some(reasoning)) = (show, &answer.reasoning) {
        let parts = if show {
            words(reasoning)
        } else {
            vec![String::new()]
        };
        thought = words(reasoning).len() as u64;
        let (text, signature) = (parts.concat(), format!("sig_fake{n}"));
        let mut deltas: Vec<Value> = parts
            .into_iter()
            .map(|p| json!({ "type": "thinking_delta", "thinking": p }))
            .collect();
        deltas.push(json!({ "type": "signature_delta", "signature": signature }));
        blocks.push(Block {
            whole: json!({ "type": "thinking", "thinking": text, "signature": signature }),
            start: json!({ "type": "thinking", "thinking": "", "signature": "" }),
            deltas,
        });
    }
    let stop = match &answer.reply {
        Reply::Error(status) => {
            let (kind, message) = scripted_error(*status);
            let body = error_body(kind, message, n);
            return send(&answer, &headers, Out::Json(*status, body)).await;
        }
        Reply::Text(t) => {
            let deltas = words(t)
                .into_iter()
                .map(|p| json!({ "type": "text_delta", "text": p }));
            blocks.push(Block {
                whole: json!({ "type": "text", "text": t }),
                start: json!({ "type": "text", "text": "" }),
                deltas: deltas.collect(),
            });
            "end_turn"
        }
        Reply::Tool { name, args } => {
            let id = format!("toolu_fake{n}");
            // The real stream opens a tool's input with an empty delta.
            let parts = std::iter::once(String::new()).chain(pieces(&args.to_string(), 8));
            blocks.push(Block {
                whole: json!({ "type": "tool_use", "id": id, "name": name, "input": args }),
                start: json!({ "type": "tool_use", "id": id, "name": name, "input": {} }),
                deltas: parts
                    .map(|p| json!({ "type": "input_json_delta", "partial_json": p }))
                    .collect(),
            });
            "tool_use"
        }
    };
    let visible = blocks.last().map_or(0, |b| b.deltas.len() as u64);
    let (input, output) = (tokens(&body["messages"]), visible + thought);
    let mut usage = json!({ "input_tokens": input, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "output_tokens": output });
    let mut end_usage = json!({ "output_tokens": output });
    if show.is_some() && thought > 0 {
        // "usage.output_tokens_details.thinking_tokens ... When streaming,
        // this breakdown appears only on the final `message_delta` event."
        // https://platform.claude.com/docs/en/build-with-claude/extended-thinking
        let details = json!({ "thinking_tokens": thought });
        usage["output_tokens_details"] = details.clone();
        end_usage["output_tokens_details"] = details;
    }
    let mut message = json!({
        "id": format!("msg_fake{n}"),
        "type": "message",
        "role": "assistant",
        "model": body["model"],
        "content": blocks.iter().map(|b| b.whole.clone()).collect::<Vec<_>>(),
        "stop_reason": stop,
        "stop_sequence": null,
        "usage": usage,
    });
    if body.get("stream") != Some(&Value::Bool(true)) {
        return send(&answer, &headers, Out::Json(200, message)).await;
    }

    message["content"] = json!([]);
    message["stop_reason"] = Value::Null;
    message["usage"]["output_tokens"] = 1.into();
    if let Some(u) = message["usage"].as_object_mut() {
        u.remove("output_tokens_details");
    }
    let mut frames = vec![event(
        "message_start",
        json!({ "type": "message_start", "message": message }),
    )];
    for (i, b) in blocks.into_iter().enumerate() {
        let start = json!({ "type": "content_block_start", "index": i, "content_block": b.start });
        frames.push(event("content_block_start", start));
        if i == 0 {
            frames.push(event("ping", json!({ "type": "ping" })));
        }
        for d in b.deltas {
            let v = json!({ "type": "content_block_delta", "index": i, "delta": d });
            frames.push(event("content_block_delta", v));
        }
        frames.push(event(
            "content_block_stop",
            json!({ "type": "content_block_stop", "index": i }),
        ));
    }
    let delta = json!({ "stop_reason": stop, "stop_sequence": null });
    frames.extend([
        event(
            "message_delta",
            json!({ "type": "message_delta", "delta": delta, "usage": end_usage }),
        ),
        event("message_stop", json!({ "type": "message_stop" })),
    ]);
    let (kind, message) = scripted_error(529);
    let error = json!({ "type": "error", "error": { "type": kind, "message": message } });
    send(&answer, &headers, Out::Sse(frames, event("error", error))).await
}
