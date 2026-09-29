//! OpenAI Responses: `POST /v1/responses`, whole JSON or named SSE events.
//! https://developers.openai.com/api/reference/resources/responses/methods/create
//! Events: https://developers.openai.com/api/reference/resources/responses/streaming-events
//!
//! Per-model rules come from the Chat Completions table (`chat_models.rs`):
//! max output, reasoning efforts and when sampling is allowed. Error
//! wording the docs do not give is the wording commonly seen from the API,
//! marked "(commonly seen)". "CREATE" in comments is the first URL above.

use crate::transport::framed;
use serde_json::{Map, Value, json};

use super::chat_models::{Allowed, BASE, MODELS, Model};
use super::{OpenAi, Quirks, bad, object};
use crate::core::misfit;
use crate::util::{at, lookup, pieces, tokens, unix_now, words};
use crate::{Answer, Ctx, Endpoint, Out, Provider, Rejection, Reply, Request, event};

/// Top-level parameters the real API knows; any other is rejected.
/// https://developers.openai.com/api/reference/resources/responses/methods/create (Body Parameters)
#[rustfmt::skip]
const KNOWN: &[&str] = &[
    "access_programs", "background", "context_management", "conversation", "include",
    "input", "instructions", "max_output_tokens", "max_tool_calls", "metadata", "model",
    "moderation", "parallel_tool_calls", "previous_response_id", "prompt",
    "prompt_cache_key", "prompt_cache_options", "prompt_cache_retention", "reasoning",
    "safety_identifier", "service_tier", "store", "stream", "stream_options",
    "temperature", "text", "tool_choice", "tools", "top_logprobs", "top_p", "truncation",
    "user",
];

/// The `include` values (`ResponseIncludable`), same source.
#[rustfmt::skip]
const INCLUDABLE: &[&str] = &[
    "file_search_call.results", "web_search_call.results", "web_search_call.action.sources",
    "message.input_image.image_url", "computer_call_output.output.image_url",
    "code_interpreter_call.outputs", "reasoning.encrypted_content",
    "message.output_text.logprobs",
];

/// Roles of an input message (`EasyInputMessage`), same source.
const ROLES: &[&str] = &["user", "assistant", "system", "developer"];

pub(crate) struct Responses;

impl Endpoint<OpenAi> for Responses {
    fn route(&self) -> (&'static str, &'static str) {
        ("POST", "/v1/responses")
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
    let set = |p: &str| body.get(p).filter(|v| !v.is_null());
    // `model` is optional in the reference only because a `prompt`
    // template can carry it (commonly seen wording).
    if set("model").is_none() && set("prompt").is_none() {
        let message = "Missing required parameter: 'model'.";
        return Err(bad(
            message,
            Some("model"),
            Some("missing_required_parameter"),
        ));
    }
    // previous_response_id "Cannot be used in conjunction with
    // `conversation`": CREATE (commonly seen wording).
    if set("previous_response_id").is_some() && set("conversation").is_some() {
        let message = "Mutually exclusive parameters: Ensure you are only providing one of: 'previous_response_id' or 'conversation'.";
        return Err(bad(message, None, Some("mutually_exclusive_parameters")));
    }
    // stream_options: "Only set this when you set `stream: true`": CREATE
    // (wording as chat.rs, commonly seen).
    if set("stream_options").is_some() && body.get("stream") != Some(&Value::Bool(true)) {
        let message = "The 'stream_options' parameter is only allowed when 'stream' is enabled.";
        return Err(bad(message, Some("stream_options"), None));
    }
    // temperature "between 0 and 2"; top_logprobs "between 0 and 20": CREATE.
    range(body, "temperature", 2.0, "decimal")?;
    range(body, "top_logprobs", 20.0, "integer")?;
    // `include` takes only the `ResponseIncludable` values: CREATE
    // (commonly seen wording).
    let include = set("include").and_then(Value::as_array);
    for (i, v) in include.into_iter().flatten().enumerate() {
        let v = v.as_str().unwrap_or_default();
        if !INCLUDABLE.contains(&v) {
            let message = format!(
                "Invalid value: '{v}'. Supported values are: {}.",
                quoted(INCLUDABLE)
            );
            return Err(bad(
                message,
                Some(&format!("include[{i}]")),
                Some("invalid_value"),
            ));
        }
    }
    let stateful = set("previous_response_id").is_some() || set("conversation").is_some();
    match set("input") {
        None | Some(Value::String(_)) => {}
        Some(Value::Array(items)) => check_input(items, stateful)?,
        Some(_) => {
            let message = "Invalid type for 'input': expected one of a string or array of input items, but got an object instead.";
            return Err(bad(message, Some("input"), Some("invalid_type")));
        }
    }
    check_tools(set("tools"))?;
    let model = set("model").and_then(Value::as_str).unwrap_or_default();
    check_model(body, &lookup(MODELS, BASE, model))
}

/// `p` within `0..=max` (commonly seen wording).
fn range(body: &Map<String, Value>, p: &str, max: f64, kind: &str) -> Result<(), Rejection> {
    let Some(v) = body.get(p).and_then(Value::as_f64) else {
        return Ok(());
    };
    let (what, bound) = if v < 0.0 {
        ("below_min", ">= 0".to_owned())
    } else if v > max {
        ("above_max", format!("<= {max}"))
    } else {
        return Ok(());
    };
    let words = if what == "below_min" {
        "below minimum"
    } else {
        "above maximum"
    };
    let message = format!(
        "Invalid '{p}': {kind} {words} value. Expected a value {bound}, but got {} instead.",
        body[p]
    );
    let code = match (kind, what) {
        ("decimal", "below_min") => "decimal_below_min_value",
        ("decimal", _) => "decimal_above_max_value",
        (_, "below_min") => "integer_below_min_value",
        _ => "integer_above_max_value",
    };
    Err(bad(message, Some(p), Some(code)))
}

fn quoted(values: &[&str]) -> String {
    let all: Vec<String> = values.iter().map(|v| format!("'{v}'")).collect();
    all.join(", ")
}

/// Message roles (`EasyInputMessage`), and function calls paired with their outputs by
/// `call_id` (the `FunctionCall` and `FunctionCallOutput` input items:
/// CREATE). The wording is commonly seen. With `previous_response_id` or
/// `conversation`, an output may answer a call from the stored context.
fn check_input(items: &[Value], stateful: bool) -> Result<(), Rejection> {
    let mut calls: Vec<&str> = Vec::new();
    let mut outputs: Vec<&str> = Vec::new();
    for (i, item) in items.iter().enumerate() {
        let kind = item["type"].as_str().unwrap_or("message");
        let call_id = item["call_id"].as_str().unwrap_or_default();
        match kind {
            "message" => {
                let role = item["role"].as_str().unwrap_or_default();
                if !ROLES.contains(&role) {
                    let message = format!(
                        "Invalid value: '{role}'. Supported values are: 'assistant', 'system', 'developer', and 'user'."
                    );
                    let param = format!("input[{i}].role");
                    return Err(bad(message, Some(&param), Some("invalid_value")));
                }
            }
            "function_call" => calls.push(call_id),
            "function_call_output" => {
                if !stateful && !calls.contains(&call_id) {
                    let message = format!(
                        "No tool call found for function call output with call_id {call_id}."
                    );
                    return Err(bad(message, Some("input"), None));
                }
                outputs.push(call_id);
            }
            _ => {}
        }
    }
    match calls.into_iter().find(|c| !outputs.contains(c)) {
        Some(c) => {
            let message = format!("No tool output found for function call {c}.");
            Err(bad(message, Some("input"), None))
        }
        None => Ok(()),
    }
}

/// A function tool has a top-level `name` (`FunctionTool`: CREATE), not
/// Chat Completions' nested `function` (commonly seen wording).
fn check_tools(tools: Option<&Value>) -> Result<(), Rejection> {
    let tools = tools.and_then(Value::as_array).map(Vec::as_slice);
    for (i, t) in tools.unwrap_or_default().iter().enumerate() {
        if t["type"] == "function" && !t["name"].is_string() {
            let message = format!("Missing required parameter: 'tools[{i}].name'.");
            let param = format!("tools[{i}].name");
            return Err(bad(
                message,
                Some(&param),
                Some("missing_required_parameter"),
            ));
        }
    }
    Ok(())
}

/// The rules of `model`'s row in `chat_models.rs`, as they apply here.
fn check_model(body: &Map<String, Value>, model: &Model) -> Result<(), Rejection> {
    let set = |p: &str| body.get(p).filter(|v| !v.is_null());
    // max_output_tokens counts "visible output tokens and reasoning
    // tokens": CREATE; the maximum is the model page's max output.
    let asked_max = set("max_output_tokens").and_then(Value::as_u64);
    if let Some(n) = asked_max.filter(|n| *n > model.max_output) {
        let max = model.max_output;
        let message = format!(
            "max_output_tokens is too large: {n}. This model supports at most {max} output tokens, whereas you provided {n}."
        );
        return Err(bad(
            message,
            Some("max_output_tokens"),
            Some("invalid_value"),
        ));
    }
    // reasoning.effort: "Not all reasoning models support every value":
    // CREATE; the values per model are the table's.
    let asked = body.get("reasoning").and_then(|r| r["effort"].as_str());
    if let Some(e) = asked.filter(|e| !model.efforts.is_empty() && !model.efforts.contains(e)) {
        let message = format!(
            "Unsupported value: 'reasoning.effort' does not support '{e}' with this model. Supported values are: {}.",
            quoted(model.efforts)
        );
        return Err(bad(
            message,
            Some("reasoning.effort"),
            Some("unsupported_value"),
        ));
    }
    // "When reasoning effort is not `none`, remove `temperature`, `top_p`,
    // and `top_logprobs` ... For Responses, remove
    // `message.output_text.logprobs` from `include`":
    // https://developers.openai.com/api/docs/guides/latest-model
    let effort_none = asked.or(model.default_effort) == Some("none");
    let allowed = model.sampling == Allowed::Always
        || (model.sampling == Allowed::OnlyEffortNone && effort_none);
    if !allowed {
        let sampling = ["temperature", "top_p", "top_logprobs"];
        if let Some(p) = sampling.into_iter().find(|p| set(p).is_some()) {
            let message = format!("Unsupported parameter: '{p}' is not supported with this model.");
            return Err(bad(message, Some(p), Some("unsupported_parameter")));
        }
        let include = set("include").and_then(Value::as_array);
        if include.is_some_and(|i| i.iter().any(|v| v == "message.output_text.logprobs")) {
            let message = "Unsupported parameter: 'include' value 'message.output_text.logprobs' is not supported with this model.";
            return Err(bad(message, Some("include"), Some("unsupported_parameter")));
        }
    }
    Ok(())
}

/// One output item: its whole form, and the events that stream it
/// between `response.output_item.added` and `.done`.
struct Item {
    added: Value,
    done: Value,
    /// Events without `sequence_number`, `output_index` added later.
    events: Vec<Value>,
}

fn respond(openai: &OpenAi, req: &Request, answer: &Answer, n: u64) -> Out {
    let Some(body) = req.object() else {
        return Out::Misfit("mockllm: validated body is not an object".into());
    };
    let streaming = body.get("stream") == Some(&Value::Bool(true));
    // The `error` event: `type`, `code`, `message`, `param`,
    // `sequence_number` (streaming events, `error`). It follows the
    // `error_after` events kept, numbered 0..n.
    let error = json!({
        "type": "error", "code": "server_error",
        "message": openai.scripted(500).message, "param": null,
        "sequence_number": answer.error_after.unwrap_or(0)
    });
    let error = event("error", &error);
    if let (Reply::Events(events), true) = (&answer.reply, streaming) {
        let named = |e: &Value| event(e["type"].as_str().unwrap_or_default(), e);
        return Out::Sse(events.iter().map(|e| framed(e, named)).collect(), error);
    }
    let model = lookup(MODELS, BASE, at(body, "model").as_str().unwrap_or_default());
    // Delta events carry `obfuscation` "by default", unless
    // `stream_options.include_obfuscation` is false: CREATE. Not from
    // OpenAI-compatible providers (any quirk), as in chat.rs.
    let obfuscate = at(body, "stream_options")["include_obfuscation"] != false
        && openai.quirks == Quirks::default();
    let mut deltas = 0;
    let mut delta = |mut e: Value| {
        if obfuscate {
            deltas += 1;
            e["obfuscation"] = "x".repeat(deltas % 7 + 1).into();
        }
        e
    };

    let mut items = Vec::new();
    // Reasoning models answer with a `reasoning` item first; its
    // `summary` is shown only when `reasoning.summary` is asked, and
    // `encrypted_content` "is populated by default": CREATE (Reasoning).
    let thought = answer.reasoning.as_deref().map(words).unwrap_or_default();
    let reasoning = if model.reasoning { thought.len() } else { 0 } as u64;
    if model.reasoning && !thought.is_empty() {
        let id = format!("rs_fake{n}");
        let text = thought.concat();
        let summarized = !at(body, "reasoning")["summary"].is_null();
        let summary: Vec<Value> = summarized
            .then(|| json!({ "type": "summary_text", "text": text }))
            .into_iter()
            .collect();
        let mut events = Vec::new();
        if summarized {
            let ids = json!({ "item_id": id, "summary_index": 0 });
            let with = |e: Value| merge(ids.clone(), e);
            events.push(with(json!({ "type": "response.reasoning_summary_part.added", "part": { "type": "summary_text", "text": "" } })));
            for w in &thought {
                let e = json!({ "type": "response.reasoning_summary_text.delta", "delta": w });
                events.push(delta(with(e)));
            }
            events.push(with(
                json!({ "type": "response.reasoning_summary_text.done", "text": text }),
            ));
            events.push(with(
                json!({ "type": "response.reasoning_summary_part.done", "part": summary[0] }),
            ));
        }
        items.push(Item {
            added: json!({ "id": id, "type": "reasoning", "summary": [] }),
            done: json!({ "id": id, "type": "reasoning", "summary": summary, "encrypted_content": format!("gAAAAfake{n}") }),
            events,
        });
    }
    let visible = match (&answer.reply, answer.reply.text()) {
        (_, Some(text)) => {
            let id = format!("msg_fake{n}");
            let part = |t: &str| json!({ "type": "output_text", "text": t, "annotations": [], "logprobs": [] });
            let item = |status: &str, content: Value| json!({ "id": id, "type": "message", "status": status, "role": "assistant", "content": content });
            let ids = json!({ "item_id": id, "content_index": 0 });
            let with = |e: Value| merge(ids.clone(), e);
            let pieces = words(text);
            let mut events = vec![with(
                json!({ "type": "response.content_part.added", "part": part("") }),
            )];
            for w in &pieces {
                let e = json!({ "type": "response.output_text.delta", "delta": w, "logprobs": [] });
                events.push(delta(with(e)));
            }
            events.push(with(
                json!({ "type": "response.output_text.done", "text": text, "logprobs": [] }),
            ));
            events.push(with(
                json!({ "type": "response.content_part.done", "part": part(text) }),
            ));
            items.push(Item {
                added: item("in_progress", json!([])),
                done: item("completed", json!([part(text)])),
                events,
            });
            pieces.len()
        }
        (Reply::ToolCall { name, args }, _) => {
            let (id, args) = (format!("fc_fake{n}"), args.to_string());
            let item = |status: &str, args: &str| json!({ "id": id, "type": "function_call", "status": status, "arguments": args, "call_id": format!("call_fake{n}"), "name": name });
            let parts = pieces(&args, 8);
            let mut events: Vec<Value> = parts
                .iter()
                .map(|p| delta(json!({ "type": "response.function_call_arguments.delta", "item_id": id, "delta": p })))
                .collect();
            events.push(json!({ "type": "response.function_call_arguments.done", "item_id": id, "arguments": args }));
            items.push(Item {
                added: item("in_progress", ""),
                done: item("completed", &args),
                events,
            });
            parts.len()
        }
        (other, _) => return misfit("responses", other),
    } as u64;

    let input = tokens(&json!([at(body, "instructions"), at(body, "input")]));
    let output = visible + reasoning;
    let usage = json!({
        "input_tokens": input,
        "input_tokens_details": { "cached_tokens": 0, "cache_write_tokens": 0 },
        "output_tokens": output,
        "output_tokens_details": { "reasoning_tokens": reasoning },
        "total_tokens": input + output
    });
    let mut response = head(body, &model, n);
    let done: Vec<Value> = items.iter().map(|i| i.done.clone()).collect();
    let complete = |mut r: Value| {
        r["status"] = "completed".into();
        r["completed_at"] = unix_now().into();
        r["output"] = done.clone().into();
        r["usage"] = usage.clone();
        r
    };
    if !streaming {
        return Out::Json(200, complete(response));
    }

    // Named events, `event: <type>`, each with a `sequence_number` from 0,
    // as in the reference's streaming example (CREATE, "Streaming").
    response["status"] = "in_progress".into();
    let mut all = vec![
        json!({ "type": "response.created", "response": response }),
        json!({ "type": "response.in_progress", "response": response }),
    ];
    for (i, item) in items.into_iter().enumerate() {
        let at = json!({ "output_index": i });
        all.push(merge(
            json!({ "type": "response.output_item.added", "item": item.added }),
            at.clone(),
        ));
        all.extend(item.events.into_iter().map(|e| merge(e, at.clone())));
        all.push(merge(
            json!({ "type": "response.output_item.done", "item": item.done }),
            at,
        ));
    }
    all.push(json!({ "type": "response.completed", "response": complete(response) }));
    let frames = all
        .into_iter()
        .enumerate()
        .map(|(i, mut e)| {
            e["sequence_number"] = i.into();
            event(e["type"].as_str().unwrap_or_default(), &e)
        })
        .collect();
    Out::Sse(frames, error)
}

/// The Response object before its output, echoing the request's settings
/// with the defaults of the reference's examples (CREATE, "Returns").
fn head(body: &Map<String, Value>, model: &Model, n: u64) -> Value {
    let or = |p: &str, default: Value| body.get(p).cloned().unwrap_or(default);
    let reasoning = at(body, "reasoning");
    let effort = match &reasoning["effort"] {
        Value::Null => model.default_effort.map_or(Value::Null, Value::from),
        e => e.clone(),
    };
    json!({
        "id": format!("resp_fake{n}"),
        "object": "response",
        "access_programs": null,
        "created_at": unix_now(),
        "status": "completed",
        "completed_at": null,
        "error": null,
        "incomplete_details": null,
        "instructions": or("instructions", Value::Null),
        "max_output_tokens": or("max_output_tokens", Value::Null),
        "model": or("model", Value::Null),
        "output": [],
        "parallel_tool_calls": or("parallel_tool_calls", true.into()),
        "previous_response_id": or("previous_response_id", Value::Null),
        "reasoning": { "effort": effort, "summary": reasoning["summary"] },
        // "Defaults to true when omitted": CREATE (store).
        "store": or("store", true.into()),
        "temperature": or("temperature", 1.into()),
        "text": or("text", json!({ "format": { "type": "text" } })),
        "tool_choice": or("tool_choice", "auto".into()),
        "tools": or("tools", json!([])),
        "top_p": or("top_p", 1.into()),
        // "`disabled` (default)": CREATE (truncation).
        "truncation": or("truncation", "disabled".into()),
        "usage": null,
        "user": or("user", Value::Null),
        "metadata": or("metadata", json!({})),
        "service_tier": "default"
    })
}

/// `base` with the fields of `more` added.
fn merge(mut base: Value, more: Value) -> Value {
    if let (Some(b), Value::Object(m)) = (base.as_object_mut(), more) {
        b.extend(m);
    }
    base
}
