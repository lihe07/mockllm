//! A mock OpenAI Chat Completions: `POST /v1/chat/completions`.

use std::sync::Arc;

use axum::response::Response;
use serde_json::{Map, Value, json};

use crate::openai_models::{Allowed, Model, lookup};
use crate::{Answer, Out, Reply, Request, Script, Server, next_id, pieces, send, tokens, words};

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

/// How an OpenAI-compatible provider deviates from OpenAI.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Quirks {
    usage_every_chunk: bool,
    no_done: bool,
    reasoning_content: bool,
}

impl Quirks {
    pub fn new() -> Self {
        Self::default()
    }

    /// Every stream chunk carries the running `usage`, asked or not, as
    /// vLLM does with `continuous_usage_stats`:
    /// https://docs.vllm.ai/en/v0.8.5/serving/openai_compatible_server.html
    pub fn usage_every_chunk(mut self) -> Self {
        self.usage_every_chunk = true;
        self
    }

    /// The stream ends without `data: [DONE]`. Seen from OpenAI-compatible
    /// servers in production; no public doc is cited for it.
    pub fn no_done(mut self) -> Self {
        self.no_done = true;
        self
    }

    /// [`Answer::reasoning`] comes as `reasoning_content`, "the reasoning
    /// contents of the assistant message, before the final answer", in
    /// stream deltas and in the whole message, and is counted in
    /// `reasoning_tokens`: https://api-docs.deepseek.com/api/create-chat-completion
    pub fn reasoning_content(mut self) -> Self {
        self.reasoning_content = true;
        self
    }
}

/// A fake OpenAI Chat Completions provider.
#[derive(Debug)]
pub struct FakeOpenAi {
    server: Server,
}

impl FakeOpenAi {
    /// Starts OpenAI itself.
    pub async fn start() -> Self {
        Self::with_quirks(Quirks::default()).await
    }

    /// Starts an OpenAI-compatible provider with `quirks`.
    pub async fn with_quirks(quirks: Quirks) -> Self {
        let server = Server::start(move |req, script| handle(req, script, quirks)).await;
        Self { server }
    }

    /// `http://127.0.0.1:<port>/v1`
    pub fn base_url(&self) -> String {
        format!("{}/v1", self.server.origin())
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

/// A request the real API would reject.
struct Bad {
    status: u16,
    message: String,
    param: Option<String>,
    code: Option<&'static str>,
}

fn bad(message: impl Into<String>, param: Option<&str>, code: Option<&'static str>) -> Bad {
    let (message, param) = (message.into(), param.map(str::to_owned));
    Bad {
        status: 400,
        message,
        param,
        code,
    }
}

fn error_body(status: u16, message: &str, param: Option<&str>, code: Option<&str>) -> Value {
    let kind = match status {
        429 => "requests",
        s if s >= 500 => "server_error",
        _ => "invalid_request_error",
    };
    json!({ "error": { "message": message, "type": kind, "param": param, "code": code } })
}

async fn handle(req: Request, script: Arc<Script>, quirks: Quirks) -> Response {
    let answer = Answer::default();
    if req.method != "POST" || req.path != "/v1/chat/completions" {
        let message = format!("Invalid URL ({} {})", req.method, req.path);
        return send(
            &answer,
            &[],
            Out::Json(404, error_body(404, &message, None, None)),
        )
        .await;
    }
    let body = match validate(&req) {
        Ok(body) => body,
        Err(b) => {
            let v = error_body(b.status, &b.message, b.param.as_deref(), b.code);
            return send(&answer, &[], Out::Json(b.status, v)).await;
        }
    };
    respond(body, script.next(), quirks).await
}

fn validate(req: &Request) -> Result<&Map<String, Value>, Bad> {
    if !req
        .header("authorization")
        .is_some_and(|v| v.starts_with("Bearer "))
    {
        let message = "You didn't provide an API key. You need to provide your API key in an Authorization header using Bearer auth (i.e. Authorization: Bearer YOUR_KEY).";
        return Err(Bad {
            status: 401,
            ..bad(message, None, None)
        });
    }
    let Some(body) = req.body.as_object() else {
        return Err(bad(
            "We could not parse the JSON body of your request.",
            None,
            None,
        ));
    };
    if let Some(k) = body.keys().find(|k| !KNOWN.contains(&k.as_str())) {
        let message = format!("Unrecognized request argument supplied: {k}");
        return Err(bad(message, None, Some("unrecognized_request_argument")));
    }
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
    check_model(body, &lookup(model))?;
    Ok(body)
}

/// The rules of `model`'s row in [`crate::openai_models`].
fn check_model(body: &Map<String, Value>, model: &Model) -> Result<(), Bad> {
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
fn check_messages(messages: &[Value]) -> Result<(), Bad> {
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

fn scripted_error(status: u16) -> Value {
    let (message, code) = match status {
        401 => ("Incorrect API key provided.", Some("invalid_api_key")),
        404 => (
            "The model does not exist or you do not have access to it.",
            Some("model_not_found"),
        ),
        429 => (
            "Rate limit reached for requests.",
            Some("rate_limit_exceeded"),
        ),
        503 => (
            "The engine is currently overloaded, please try again later.",
            None,
        ),
        s if s >= 500 => (
            "The server had an error while processing your request. Sorry about that!",
            None,
        ),
        _ => ("Invalid request.", None),
    };
    error_body(status, message, None, code)
}

async fn respond(body: &Map<String, Value>, answer: Answer, quirks: Quirks) -> Response {
    let n = next_id();
    let headers = [("x-request-id", format!("req_fake{n}"))];
    let (text, tool) = match &answer.reply {
        Reply::Error(status) => {
            let v = scripted_error(*status);
            return send(&answer, &headers, Out::Json(*status, v)).await;
        }
        Reply::Text(t) => (words(t), None),
        Reply::Tool { name, args } => (Vec::new(), Some((name, args.to_string()))),
    };
    let model = lookup(body["model"].as_str().unwrap_or_default());
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
    head.insert("created".into(), crate::unix_now().into());
    head.insert("model".into(), body["model"].clone());

    if body.get("stream") != Some(&Value::Bool(true)) {
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
        return send(&answer, &headers, Out::Json(200, head.into())).await;
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
    let options = crate::at(body, "stream_options");
    let include_usage = options["include_usage"] == true;
    // "included by default and omitted when
    // `stream_options.include_obfuscation` is `false`":
    // https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events
    // OpenAI-compatible providers (any quirk) do not obfuscate.
    let obfuscate = options["include_obfuscation"] != false && quirks == Quirks::default();
    let frame = |v: Value| format!("data: {v}\n\n");
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
            frames.push(frame(chunk.into()));
        }
    }
    if include_usage {
        head.insert("choices".into(), json!([]));
        head.insert("usage".into(), total);
        frames.push(frame(head.into()));
    }
    if !quirks.no_done {
        frames.push("data: [DONE]\n\n".into());
    }
    send(
        &answer,
        &headers,
        Out::Sse(frames, frame(scripted_error(500))),
    )
    .await
}
