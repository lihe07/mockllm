#![allow(clippy::unwrap_used)]

mod support;

use std::time::{Duration, Instant};

use mockllm::{Answer, FakeOpenAi, Quirks};
use serde_json::{Value, json};
use support::{Reply, post};

const KEY: (&str, &str) = ("authorization", "Bearer sk-test");

async fn chat(openai: &FakeOpenAi, body: Value) -> Reply {
    post(
        &format!("{}/chat/completions", openai.base_url()),
        &[KEY],
        &body,
    )
    .await
}

fn hi(model: &str) -> Value {
    json!({ "model": model, "messages": [{ "role": "user", "content": "say hi" }] })
}

fn with(mut body: Value, extra: Value) -> Value {
    for (k, v) in extra.as_object().unwrap() {
        body[k] = v.clone();
    }
    body
}

/// Asserts a 400 with OpenAI's error shape, `code` (or `type`) and a
/// message containing `says`.
fn rejected(r: &Reply, code: &str, says: &str) {
    assert_eq!(r.status, 400, "{}", r.body);
    let e = &r.json()["error"];
    for field in ["message", "type", "param", "code"] {
        assert!(e.get(field).is_some(), "error lacks {field}: {}", r.body);
    }
    let (kind, message) = r.error();
    assert_eq!(kind, code, "{}", r.body);
    assert!(message.contains(says), "{message}");
}

#[tokio::test]
async fn whole_text_answer_has_openais_shape() {
    let openai = FakeOpenAi::start().await;
    openai.answer(Answer::text("Hi there"));
    let r = chat(&openai, hi("gpt-4o")).await;
    assert_eq!(r.status, 200);
    assert!(r.header("x-request-id").is_some());
    let v = r.json();
    assert_eq!(v["object"], "chat.completion");
    assert!(v["id"].as_str().unwrap().starts_with("chatcmpl-"));
    assert!(v["created"].is_u64());
    assert_eq!(v["model"], "gpt-4o");
    let choice = &v["choices"][0];
    assert_eq!(choice["message"]["role"], "assistant");
    assert_eq!(choice["message"]["content"], "Hi there");
    assert_eq!(choice["finish_reason"], "stop");
    let u = &v["usage"];
    assert_eq!(u["completion_tokens"], 2);
    let total = u["prompt_tokens"].as_u64().unwrap() + 2;
    assert_eq!(u["total_tokens"], total);
}

#[tokio::test]
async fn unscripted_requests_get_a_short_text() {
    let openai = FakeOpenAi::start().await;
    let r = chat(&openai, hi("gpt-4o")).await;
    assert_eq!(r.json()["choices"][0]["message"]["content"], "Hi there");
}

#[tokio::test]
async fn stream_frames_like_openai() {
    let openai = FakeOpenAi::start().await;
    openai.answer(Answer::text("Hi there you"));
    let r = chat(&openai, with(hi("gpt-4o"), json!({ "stream": true }))).await;
    assert_eq!(r.status, 200);
    assert!(
        r.header("content-type")
            .unwrap()
            .starts_with("text/event-stream")
    );
    // Every event is one `data:` line and a blank line.
    assert!(
        r.body
            .split("\n\n")
            .filter(|e| !e.is_empty())
            .all(|e| e.starts_with("data: ") && !e.contains('\n'))
    );
    assert_eq!(r.data().last().unwrap(), "[DONE]");
    let chunks = r.chunks();
    assert!(
        chunks
            .iter()
            .all(|c| c["object"] == "chat.completion.chunk" && c.get("usage").is_none())
    );
    assert!(chunks.iter().all(|c| c["obfuscation"].is_string()));
    let deltas: Vec<&Value> = chunks.iter().map(|c| &c["choices"][0]["delta"]).collect();
    assert_eq!(
        deltas[0],
        &json!({ "role": "assistant", "content": "", "refusal": null })
    );
    let text: String = deltas
        .iter()
        .filter_map(|d| d["content"].as_str())
        .collect();
    assert_eq!(text, "Hi there you");
    // The last chunk has an empty delta and the finish reason.
    let last = &chunks.last().unwrap()["choices"][0];
    assert_eq!(
        (&last["delta"], &last["finish_reason"]),
        (&json!({}), &json!("stop"))
    );
    let finished = chunks
        .iter()
        .filter(|c| !c["choices"][0]["finish_reason"].is_null());
    assert_eq!(finished.count(), 1);
}

#[tokio::test]
async fn include_usage_adds_a_final_usage_chunk_and_nulls_elsewhere() {
    let openai = FakeOpenAi::start().await;
    let body = json!({ "stream": true, "stream_options": { "include_usage": true, "include_obfuscation": false } });
    let r = chat(&openai, with(hi("gpt-4o"), body)).await;
    let chunks = r.chunks();
    let (last, rest) = chunks.split_last().unwrap();
    assert!(
        rest.iter()
            .all(|c| c["usage"].is_null() && c.get("usage").is_some())
    );
    assert!(chunks.iter().all(|c| c.get("obfuscation").is_none()));
    assert_eq!(last["choices"], json!([]));
    assert_eq!(last["usage"]["completion_tokens"], 2);
    assert_eq!(r.data().last().unwrap(), "[DONE]");
}

#[tokio::test]
async fn tool_calls_stream_their_arguments_in_pieces() {
    let openai = FakeOpenAi::start().await;
    let args = json!({ "city": "Paris", "unit": "celsius" });
    openai.answer(Answer::tool_call("get_weather", args.clone()));
    openai.answer(Answer::tool_call("get_weather", args.clone()));
    let whole = chat(&openai, hi("gpt-4o")).await.json();
    let choice = &whole["choices"][0];
    assert_eq!(choice["finish_reason"], "tool_calls");
    assert!(choice["message"]["content"].is_null());
    let call = &choice["message"]["tool_calls"][0];
    assert_eq!(
        (&call["type"], &call["function"]["name"]),
        (&json!("function"), &json!("get_weather"))
    );
    let parsed: Value =
        serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(parsed, args);

    let r = chat(&openai, with(hi("gpt-4o"), json!({ "stream": true }))).await;
    let chunks = r.chunks();
    let first = &chunks[0]["choices"][0]["delta"]["tool_calls"][0];
    assert_eq!(
        first["function"],
        json!({ "name": "get_weather", "arguments": "" })
    );
    assert!(first["id"].as_str().unwrap().starts_with("call_"));
    let pieces: Vec<&str> = chunks[1..]
        .iter()
        .filter_map(|c| c["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"].as_str())
        .collect();
    assert!(pieces.len() > 2);
    let joined: Value = serde_json::from_str(&pieces.concat()).unwrap();
    assert_eq!(joined, args);
    assert_eq!(
        chunks.last().unwrap()["choices"][0]["finish_reason"],
        "tool_calls"
    );
}

#[tokio::test]
async fn n_choices_with_stream_is_fine() {
    let openai = FakeOpenAi::start().await;
    let r = chat(
        &openai,
        with(hi("gpt-4o"), json!({ "stream": true, "n": 2 })),
    )
    .await;
    assert_eq!(r.status, 200);
    let indexes: Vec<u64> = r
        .chunks()
        .iter()
        .map(|c| c["choices"][0]["index"].as_u64().unwrap())
        .collect();
    assert!(indexes.contains(&0) && indexes.contains(&1));
}

#[tokio::test]
async fn validation_follows_openais_rules() {
    let openai = FakeOpenAi::start().await;
    let url = format!("{}/chat/completions", openai.base_url());
    let no_key = post(&url, &[], &hi("gpt-4o")).await;
    assert_eq!(no_key.status, 401);
    assert!(no_key.error().1.contains("didn't provide an API key"));

    let r = chat(&openai, json!({ "messages": [] })).await;
    assert_eq!(
        (r.status, r.error().1.as_str()),
        (400, "you must provide a model parameter")
    );
    rejected(
        &chat(&openai, json!({ "model": "gpt-4o" })).await,
        "missing_required_parameter",
        "'messages'",
    );
    rejected(
        &chat(&openai, json!({ "model": "gpt-4o", "messages": [] })).await,
        "empty_array",
        "empty array",
    );
    let unknown = with(hi("gpt-4o"), json!({ "x_new_field": 1 }));
    rejected(
        &chat(&openai, unknown).await,
        "unrecognized_request_argument",
        "x_new_field",
    );
    let options = with(
        hi("gpt-4o"),
        json!({ "stream_options": { "include_usage": true } }),
    );
    let r = chat(&openai, options).await;
    assert_eq!(r.status, 400);
    assert!(
        r.error()
            .1
            .contains("only allowed when 'stream' is enabled")
    );
    let role = json!({ "model": "gpt-4o", "messages": [{ "role": "bot", "content": "x" }] });
    rejected(&chat(&openai, role).await, "invalid_value", "'bot'");
    // Nothing valid was sent, so no answer was used up.
    openai.answer(Answer::text("kept"));
    let r = chat(&openai, hi("gpt-4o")).await;
    assert_eq!(r.json()["choices"][0]["message"]["content"], "kept");
}

#[tokio::test]
async fn tool_messages_must_answer_the_preceding_tool_calls() {
    let openai = FakeOpenAi::start().await;
    let call = json!({ "id": "call_1", "type": "function", "function": { "name": "f", "arguments": "{}" } });
    let asked = json!({ "role": "assistant", "content": null, "tool_calls": [call] });
    let user = json!({ "role": "user", "content": "hi" });
    let stray = json!({ "model": "gpt-4o", "messages": [user, { "role": "tool", "tool_call_id": "call_1", "content": "x" }] });
    let r = chat(&openai, stray).await;
    assert_eq!(r.status, 400);
    assert!(
        r.error()
            .1
            .contains("must be a response to a preceeding message with 'tool_calls'")
    );
    let unanswered =
        json!({ "model": "gpt-4o", "messages": [user, asked, { "role": "user", "content": "?" }] });
    rejected(
        &chat(&openai, unanswered).await,
        "invalid_parameter",
        "call_1",
    );
    let answered = json!({ "model": "gpt-4o", "messages": [user, asked, { "role": "tool", "tool_call_id": "call_1", "content": "x" }] });
    assert_eq!(chat(&openai, answered).await.status, 200);
}

// Per-model rules: one test each, in the order of `openai_models.rs`.

#[tokio::test]
async fn o_series_rejects_max_tokens() {
    let openai = FakeOpenAi::start().await;
    let r = chat(&openai, with(hi("o3-mini"), json!({ "max_tokens": 100 }))).await;
    rejected(
        &r,
        "unsupported_parameter",
        "Use 'max_completion_tokens' instead",
    );
    let ok = chat(
        &openai,
        with(hi("o3-mini"), json!({ "max_completion_tokens": 100 })),
    )
    .await;
    assert_eq!(ok.status, 200);
    assert_eq!(
        chat(&openai, with(hi("gpt-4o"), json!({ "max_tokens": 100 })))
            .await
            .status,
        200
    );
}

#[tokio::test]
async fn max_output_is_per_model() {
    let openai = FakeOpenAi::start().await;
    let r = chat(
        &openai,
        with(hi("gpt-4o"), json!({ "max_completion_tokens": 16_385 })),
    )
    .await;
    rejected(&r, "invalid_value", "at most 16384");
    let r = chat(
        &openai,
        with(hi("gpt-4.1"), json!({ "max_completion_tokens": 32_768 })),
    )
    .await;
    assert_eq!(r.status, 200);
    let r = chat(
        &openai,
        with(hi("o4-mini"), json!({ "max_completion_tokens": 100_001 })),
    )
    .await;
    rejected(&r, "invalid_value", "at most 100000");
}

#[tokio::test]
async fn o1_mini_has_no_function_calling() {
    let openai = FakeOpenAi::start().await;
    let tools = json!({ "tools": [{ "type": "function", "function": { "name": "f" } }] });
    rejected(
        &chat(&openai, with(hi("o1-mini"), tools.clone())).await,
        "unsupported_parameter",
        "'tools'",
    );
    assert_eq!(chat(&openai, with(hi("o1"), tools)).await.status, 200);
}

#[tokio::test]
async fn gpt5_rejects_sampling_and_effort_none() {
    let openai = FakeOpenAi::start().await;
    let r = chat(
        &openai,
        with(hi("gpt-5-mini"), json!({ "temperature": 0.2 })),
    )
    .await;
    rejected(&r, "unsupported_parameter", "'temperature'");
    let r = chat(
        &openai,
        with(hi("gpt-5"), json!({ "reasoning_effort": "none" })),
    )
    .await;
    rejected(
        &r,
        "unsupported_value",
        "'minimal', 'low', 'medium', 'high'",
    );
    let ok = chat(
        &openai,
        with(hi("gpt-5"), json!({ "reasoning_effort": "minimal" })),
    )
    .await;
    assert_eq!(ok.status, 200);
    // gpt-5-chat-latest is not a reasoning model.
    let chat_latest = chat(
        &openai,
        with(hi("gpt-5-chat-latest"), json!({ "temperature": 0.2 })),
    )
    .await;
    assert_eq!(chat_latest.status, 200);
}

#[tokio::test]
async fn gpt5_1_accepts_effort_none_not_xhigh() {
    let openai = FakeOpenAi::start().await;
    assert_eq!(
        chat(
            &openai,
            with(hi("gpt-5.1"), json!({ "reasoning_effort": "none" }))
        )
        .await
        .status,
        200
    );
    let r = chat(
        &openai,
        with(hi("gpt-5.1"), json!({ "reasoning_effort": "xhigh" })),
    )
    .await;
    rejected(&r, "unsupported_value", "'xhigh'");
}

#[tokio::test]
async fn gpt5_4_allows_sampling_only_at_effort_none() {
    let openai = FakeOpenAi::start().await;
    // The default effort is none.
    assert_eq!(
        chat(&openai, with(hi("gpt-5.4"), json!({ "temperature": 0.2 })))
            .await
            .status,
        200
    );
    let r = chat(
        &openai,
        with(
            hi("gpt-5.2"),
            json!({ "top_p": 0.5, "reasoning_effort": "high" }),
        ),
    )
    .await;
    rejected(&r, "unsupported_parameter", "'top_p'");
    let r = chat(
        &openai,
        with(
            hi("gpt-5.4"),
            json!({ "logprobs": true, "reasoning_effort": "low" }),
        ),
    )
    .await;
    rejected(&r, "unsupported_parameter", "'logprobs'");
}

#[tokio::test]
async fn gpt5_5_and_5_6_efforts() {
    let openai = FakeOpenAi::start().await;
    let r = chat(
        &openai,
        with(hi("gpt-5.5"), json!({ "reasoning_effort": "max" })),
    )
    .await;
    rejected(&r, "unsupported_value", "'max'");
    let ok = chat(
        &openai,
        with(hi("gpt-5.6-terra"), json!({ "reasoning_effort": "max" })),
    )
    .await;
    assert_eq!(ok.status, 200);
}

#[tokio::test]
async fn gpt6_astra_rejects_effort_none_sampling_and_tools() {
    let openai = FakeOpenAi::start().await;
    let r = chat(
        &openai,
        with(hi("gpt-6-astra"), json!({ "reasoning_effort": "none" })),
    )
    .await;
    rejected(&r, "unsupported_value", "'none'");
    let r = chat(
        &openai,
        with(hi("gpt-6-astra"), json!({ "temperature": 0.5 })),
    )
    .await;
    rejected(&r, "unsupported_parameter", "'temperature'");
    let tools = json!({ "tools": [{ "type": "function", "function": { "name": "f" } }] });
    rejected(
        &chat(&openai, with(hi("gpt-6-astra"), tools)).await,
        "unsupported_parameter",
        "'tools'",
    );
}

#[tokio::test]
async fn gpt6_sol_allows_tools_only_at_effort_none() {
    let openai = FakeOpenAi::start().await;
    let tools = json!({ "tools": [{ "type": "function", "function": { "name": "f" } }] });
    rejected(
        &chat(&openai, with(hi("gpt-6-sol"), tools.clone())).await,
        "unsupported_parameter",
        "'tools'",
    );
    let none = with(
        with(hi("gpt-6-luna"), tools),
        json!({ "reasoning_effort": "none", "temperature": 0.3 }),
    );
    assert_eq!(chat(&openai, none).await.status, 200);
}

#[tokio::test]
async fn reasoning_models_count_hidden_reasoning_tokens() {
    let openai = FakeOpenAi::start().await;
    openai.answer(Answer::text("Hi").reasoning("one two three"));
    let v = chat(&openai, hi("o3")).await.json();
    assert_eq!(
        v["usage"]["completion_tokens_details"]["reasoning_tokens"],
        3
    );
    assert_eq!(v["usage"]["completion_tokens"], 4);
    assert!(
        v["choices"][0]["message"]
            .get("reasoning_content")
            .is_none()
    );
    openai.answer(Answer::text("Hi").reasoning("one two three"));
    let v = chat(&openai, hi("gpt-4o")).await.json();
    assert_eq!(
        v["usage"]["completion_tokens_details"]["reasoning_tokens"],
        0
    );
}

// Scripts.

#[tokio::test]
async fn scripted_error_in_openais_shape_with_retry_after() {
    let openai = FakeOpenAi::start().await;
    openai.answer(Answer::error(429).retry_after(7));
    let r = chat(&openai, hi("gpt-4o")).await;
    assert_eq!(r.status, 429);
    assert_eq!(r.header("retry-after"), Some("7"));
    let e = &r.json()["error"];
    assert_eq!(
        (&e["type"], &e["code"]),
        (&json!("requests"), &json!("rate_limit_exceeded"))
    );
    assert!(e["param"].is_null());
}

#[tokio::test]
async fn cut_after_breaks_the_stream() {
    let openai = FakeOpenAi::start().await;
    openai.answer(Answer::text("one two three four").cut_after(2));
    let r = chat(&openai, with(hi("gpt-4o"), json!({ "stream": true }))).await;
    assert!(r.ended.is_err(), "the body should break off");
    assert_eq!(r.chunks().len(), 2);
    assert!(!r.body.contains("[DONE]"));
}

#[tokio::test]
async fn cut_after_breaks_a_whole_body_halfway() {
    let openai = FakeOpenAi::start().await;
    openai.answer(Answer::text("Hi").cut_after(0));
    let r = chat(&openai, hi("gpt-4o")).await;
    assert!(r.ended.is_err());
    assert!(serde_json::from_str::<Value>(&r.body).is_err());
}

#[tokio::test]
async fn error_after_sends_an_error_chunk_mid_stream() {
    let openai = FakeOpenAi::start().await;
    openai.answer(Answer::text("one two three").error_after(2));
    let r = chat(&openai, with(hi("gpt-4o"), json!({ "stream": true }))).await;
    assert!(r.ended.is_ok());
    let chunks = r.chunks();
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[2]["error"]["type"], "server_error");
}

#[tokio::test]
async fn delay_first_slows_the_head_and_pace_spaces_chunks() {
    let openai = FakeOpenAi::start().await;
    openai.answer(Answer::text("a b").delay_first(300));
    let t = Instant::now();
    chat(&openai, hi("gpt-4o")).await;
    assert!(t.elapsed() >= Duration::from_millis(300));
    openai.answer(Answer::text("a b c").pace(50));
    let t = Instant::now();
    let r = chat(&openai, with(hi("gpt-4o"), json!({ "stream": true }))).await;
    // 5 chunks and [DONE]: 5 gaps.
    assert_eq!(r.data().len(), 6);
    assert!(t.elapsed() >= Duration::from_millis(250));
}

#[tokio::test]
async fn requests_records_what_was_sent() {
    let openai = FakeOpenAi::start().await;
    chat(&openai, hi("gpt-4o")).await;
    post(&format!("{}/nope", openai.base_url()), &[KEY], &json!({})).await;
    let seen = openai.requests();
    assert_eq!(seen.len(), 2);
    assert_eq!(
        (seen[0].method.as_str(), seen[0].path.as_str()),
        ("POST", "/v1/chat/completions")
    );
    assert_eq!(seen[0].header("authorization"), Some("Bearer sk-test"));
    assert_eq!(seen[0].body, hi("gpt-4o"));
    assert_eq!(seen[1].path, "/v1/nope");
}

// Quirks of OpenAI-compatible providers.

#[tokio::test]
async fn quirk_usage_every_chunk() {
    let openai = FakeOpenAi::with_quirks(Quirks::new().usage_every_chunk()).await;
    let r = chat(&openai, with(hi("m"), json!({ "stream": true }))).await;
    let chunks = r.chunks();
    assert!(chunks.iter().all(|c| c["usage"]["prompt_tokens"].is_u64()));
    assert!(chunks.iter().all(|c| c.get("obfuscation").is_none()));
    assert_eq!(chunks.last().unwrap()["usage"]["completion_tokens"], 2);
}

#[tokio::test]
async fn quirk_no_done() {
    let openai = FakeOpenAi::with_quirks(Quirks::new().no_done()).await;
    let r = chat(&openai, with(hi("m"), json!({ "stream": true }))).await;
    assert!(r.ended.is_ok());
    assert!(!r.body.contains("[DONE]"));
    assert_eq!(
        r.chunks().last().unwrap()["choices"][0]["finish_reason"],
        "stop"
    );
}

#[tokio::test]
async fn quirk_reasoning_content() {
    let openai = FakeOpenAi::with_quirks(Quirks::new().reasoning_content()).await;
    openai.answer(Answer::text("Hi").reasoning("let me think"));
    openai.answer(Answer::text("Hi").reasoning("let me think"));
    let v = chat(&openai, hi("deepseek-x")).await.json();
    assert_eq!(
        v["choices"][0]["message"]["reasoning_content"],
        "let me think"
    );
    assert_eq!(
        v["usage"]["completion_tokens_details"]["reasoning_tokens"],
        3
    );
    let r = chat(&openai, with(hi("deepseek-x"), json!({ "stream": true }))).await;
    let deltas: Vec<Value> = r
        .chunks()
        .iter()
        .map(|c| c["choices"][0]["delta"].clone())
        .collect();
    let thought: String = deltas
        .iter()
        .filter_map(|d| d["reasoning_content"].as_str())
        .collect();
    assert_eq!(thought, "let me think");
    let first_text = deltas.iter().position(|d| d["content"] == "Hi").unwrap();
    let last_thought = deltas
        .iter()
        .rposition(|d| d["reasoning_content"].is_string())
        .unwrap();
    assert!(last_thought < first_text);
}

#[tokio::test]
async fn body_is_the_whole_response_exactly() {
    let openai = FakeOpenAi::start().await;
    let body = json!({ "id": "chatcmpl-fixed", "object": "chat.completion", "created": 1, "model": "gpt-4o", "choices": [] });
    openai.answer(Answer::body(body.clone()));
    let r = chat(&openai, hi("gpt-4o")).await;
    assert_eq!(
        (r.status, r.body.as_str()),
        (200, body.to_string().as_str())
    );
    assert_eq!(r.header("content-type"), Some("application/json"));

    let raw = "{ \"id\": \"chatcmpl-fixed\",\n  \"created\": 1 }";
    openai.answer(Answer::raw(raw));
    let r = chat(&openai, hi("gpt-4o")).await;
    assert_eq!((r.status, r.body.as_str()), (200, raw));
    assert_eq!(r.header("content-type"), Some("application/json"));
}

#[tokio::test]
async fn raw_events_can_be_raw_lines() {
    let openai = FakeOpenAi::start().await;
    let events = vec![json!({ "custom": 1 }), json!("data: [DONE]")];
    openai.answer(Answer::events(events));
    let r = chat(&openai, with(hi("gpt-4o"), json!({ "stream": true }))).await;
    assert_eq!(r.body, "data: {\"custom\":1}\n\ndata: [DONE]\n\n");
}

#[tokio::test]
async fn answers_set_headers_and_error_messages() {
    let openai = FakeOpenAi::start().await;
    openai.answer(
        Answer::text("Hi")
            .header("x-custom", "1")
            .header("x-request-id", "req_mine"),
    );
    let r = chat(&openai, hi("gpt-4o")).await;
    assert_eq!(r.header("x-custom"), Some("1"));
    assert_eq!(r.header("x-request-id"), Some("req_mine"));

    openai.answer(
        Answer::error(429)
            .message("Too fast.")
            .header("x-ratelimit-remaining-requests", "0"),
    );
    let r = chat(&openai, hi("gpt-4o")).await;
    assert_eq!(r.status, 429);
    assert_eq!(r.header("x-ratelimit-remaining-requests"), Some("0"));
    let e = &r.json()["error"];
    assert_eq!(
        (&e["message"], &e["type"], &e["code"]),
        (
            &json!("Too fast."),
            &json!("requests"),
            &json!("rate_limit_exceeded")
        )
    );

    openai.answer(Answer::raw("plain").header("content-type", "text/plain"));
    let r = chat(&openai, hi("gpt-4o")).await;
    assert_eq!(
        (r.header("content-type"), r.body.as_str()),
        (Some("text/plain"), "plain")
    );
}
