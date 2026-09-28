#![allow(clippy::unwrap_used)]

mod support;

use mockllm::{Answer, FakeAnthropic};
use serde_json::{Value, json};
use support::{Reply, post};

const HEADERS: &[(&str, &str)] = &[
    ("x-api-key", "sk-ant-test"),
    ("anthropic-version", "2023-06-01"),
];

async fn messages(anthropic: &FakeAnthropic, body: Value) -> Reply {
    post(
        &format!("{}/v1/messages", anthropic.base_url()),
        HEADERS,
        &body,
    )
    .await
}

fn hi(model: &str) -> Value {
    json!({ "model": model, "max_tokens": 1024, "messages": [{ "role": "user", "content": "say hi" }] })
}

fn with(mut body: Value, extra: Value) -> Value {
    for (k, v) in extra.as_object().unwrap() {
        body[k] = v.clone();
    }
    body
}

/// Asserts a 400 `invalid_request_error` in Anthropic's shape whose
/// message contains `says`.
fn rejected(r: &Reply, says: &str) {
    assert_eq!(r.status, 400, "{}", r.body);
    let v = r.json();
    assert_eq!(v["type"], "error");
    assert!(v["request_id"].as_str().unwrap().starts_with("req_"));
    let (kind, message) = r.error();
    assert_eq!(kind, "invalid_request_error");
    assert!(message.contains(says), "{message}");
}

/// The data of every event named `name`.
fn data_of(r: &Reply, name: &str) -> Vec<Value> {
    let events = r
        .body
        .split("\n\n")
        .filter(|e| e.starts_with(&format!("event: {name}\n")));
    events
        .map(|e| serde_json::from_str(e.split_once("data: ").unwrap().1).unwrap())
        .collect()
}

#[tokio::test]
async fn whole_text_answer_has_anthropics_shape() {
    let anthropic = FakeAnthropic::start().await;
    anthropic.answer(Answer::text("Hi there"));
    let r = messages(&anthropic, hi("claude-sonnet-4-5")).await;
    assert_eq!(r.status, 200);
    assert!(r.header("request-id").is_some());
    let v = r.json();
    assert!(v["id"].as_str().unwrap().starts_with("msg_"));
    assert_eq!(
        (&v["type"], &v["role"], &v["model"]),
        (
            &json!("message"),
            &json!("assistant"),
            &json!("claude-sonnet-4-5")
        )
    );
    assert_eq!(
        v["content"],
        json!([{ "type": "text", "text": "Hi there" }])
    );
    assert_eq!(
        (&v["stop_reason"], &v["stop_sequence"]),
        (&json!("end_turn"), &Value::Null)
    );
    let u = &v["usage"];
    assert_eq!(u["output_tokens"], 2);
    assert!(u["input_tokens"].is_u64());
    assert_eq!(
        (
            &u["cache_creation_input_tokens"],
            &u["cache_read_input_tokens"]
        ),
        (&json!(0), &json!(0))
    );
}

#[tokio::test]
async fn stream_has_anthropics_named_events() {
    let anthropic = FakeAnthropic::start().await;
    anthropic.answer(Answer::text("Hi there"));
    let r = messages(
        &anthropic,
        with(hi("claude-sonnet-4-5"), json!({ "stream": true })),
    )
    .await;
    assert!(
        r.header("content-type")
            .unwrap()
            .starts_with("text/event-stream")
    );
    let expected = [
        "message_start",
        "content_block_start",
        "ping",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ];
    assert_eq!(r.events(), expected);
    // Each event's data names its own type.
    for (name, data) in r.events().iter().zip(r.chunks()) {
        assert_eq!(&data["type"], name);
    }
    let start = &data_of(&r, "message_start")[0]["message"];
    assert!(start["usage"]["input_tokens"].is_u64());
    assert_eq!(
        (&start["content"], &start["stop_reason"]),
        (&json!([]), &Value::Null)
    );
    let block = &data_of(&r, "content_block_start")[0];
    assert_eq!(
        block["content_block"],
        json!({ "type": "text", "text": "" })
    );
    let text: String = data_of(&r, "content_block_delta")
        .iter()
        .map(|d| {
            assert_eq!(d["delta"]["type"], "text_delta");
            d["delta"]["text"].as_str().unwrap().to_owned()
        })
        .collect();
    assert_eq!(text, "Hi there");
    let end = &data_of(&r, "message_delta")[0];
    assert_eq!(end["delta"]["stop_reason"], "end_turn");
    assert_eq!(end["usage"]["output_tokens"], 2);
}

#[tokio::test]
async fn tool_use_streams_input_json_deltas() {
    let anthropic = FakeAnthropic::start().await;
    let args = json!({ "city": "Paris", "unit": "celsius" });
    anthropic.answer(Answer::tool_call("get_weather", args.clone()));
    anthropic.answer(Answer::tool_call("get_weather", args.clone()));
    let v = messages(&anthropic, hi("claude-sonnet-4-5")).await.json();
    assert_eq!(v["stop_reason"], "tool_use");
    let block = &v["content"][0];
    assert_eq!(
        (&block["type"], &block["name"], &block["input"]),
        (&json!("tool_use"), &json!("get_weather"), &args)
    );
    assert!(block["id"].as_str().unwrap().starts_with("toolu_"));

    let r = messages(
        &anthropic,
        with(hi("claude-sonnet-4-5"), json!({ "stream": true })),
    )
    .await;
    let start = &data_of(&r, "content_block_start")[0]["content_block"];
    assert_eq!(start["input"], json!({}));
    let deltas = data_of(&r, "content_block_delta");
    assert_eq!(
        deltas[0]["delta"],
        json!({ "type": "input_json_delta", "partial_json": "" })
    );
    let json_text: String = deltas
        .iter()
        .map(|d| d["delta"]["partial_json"].as_str().unwrap().to_owned())
        .collect();
    assert_eq!(serde_json::from_str::<Value>(&json_text).unwrap(), args);
    assert_eq!(
        data_of(&r, "message_delta")[0]["delta"]["stop_reason"],
        "tool_use"
    );
}

#[tokio::test]
async fn validation_follows_anthropics_rules() {
    let anthropic = FakeAnthropic::start().await;
    let url = format!("{}/v1/messages", anthropic.base_url());
    let r = post(
        &url,
        &[("anthropic-version", "2023-06-01")],
        &hi("claude-x"),
    )
    .await;
    assert_eq!(
        (r.status, r.error().0.as_str()),
        (401, "authentication_error")
    );
    let r = post(&url, &[("x-api-key", "k")], &hi("claude-x")).await;
    rejected(&r, "anthropic-version: header is required");
    let no_max = json!({ "model": "claude-x", "messages": [{ "role": "user", "content": "hi" }] });
    rejected(
        &messages(&anthropic, no_max).await,
        "max_tokens: Field required",
    );
    rejected(
        &messages(&anthropic, with(hi("claude-x"), json!({ "n": 2 }))).await,
        "n: Extra inputs are not permitted",
    );
    rejected(
        &messages(
            &anthropic,
            with(hi("claude-x"), json!({ "temperature": 1.5 })),
        )
        .await,
        "less than or equal to 1",
    );
    let system = json!({ "role": "system", "content": "be brief" });
    let r = messages(
        &anthropic,
        with(
            hi("claude-x"),
            json!({ "messages": [system, { "role": "user", "content": "hi" }] }),
        ),
    )
    .await;
    rejected(&r, "top-level `system` parameter");
    let empty = json!({ "messages": [{ "role": "user", "content": "" }] });
    rejected(
        &messages(&anthropic, with(hi("claude-x"), empty)).await,
        "non-empty content",
    );
    // Consecutive same-role turns are combined, not rejected.
    let twice = json!({ "messages": [{ "role": "user", "content": "a" }, { "role": "user", "content": "b" }] });
    assert_eq!(
        messages(&anthropic, with(hi("claude-x"), twice))
            .await
            .status,
        200
    );
    let r = post(
        &format!("{}/v1/nope", anthropic.base_url()),
        HEADERS,
        &hi("claude-x"),
    )
    .await;
    assert_eq!((r.status, r.error().0.as_str()), (404, "not_found_error"));
}

#[tokio::test]
async fn tool_use_needs_tool_results_in_the_next_message() {
    let anthropic = FakeAnthropic::start().await;
    let used = json!({ "role": "assistant", "content": [{ "type": "tool_use", "id": "toolu_1", "name": "f", "input": {} }] });
    let user = json!({ "role": "user", "content": "hi" });
    let bare = with(
        hi("claude-x"),
        json!({ "messages": [user, used, { "role": "user", "content": "?" }] }),
    );
    rejected(&messages(&anthropic, bare).await, "toolu_1");
    let result = json!({ "role": "user", "content": [{ "type": "tool_result", "tool_use_id": "toolu_1", "content": "x" }] });
    let answered = with(hi("claude-x"), json!({ "messages": [user, used, result] }));
    assert_eq!(messages(&anthropic, answered).await.status, 200);
}

// Per-model rules: one test each, in the order of `anthropic_models.rs`.

#[tokio::test]
async fn always_thinking_models_reject_enabled_and_disabled() {
    let anthropic = FakeAnthropic::start().await;
    let enabled =
        json!({ "thinking": { "type": "enabled", "budget_tokens": 2000 }, "max_tokens": 4000 });
    rejected(
        &messages(&anthropic, with(hi("claude-opus-5-5"), enabled)).await,
        "\"thinking.type.enabled\" is not supported",
    );
    let disabled = json!({ "thinking": { "type": "disabled" } });
    let r = messages(&anthropic, with(hi("claude-fable-5-1"), disabled.clone())).await;
    rejected(
        &r,
        "\"thinking.type.disabled\" is not supported for this model. Use \"thinking.type.adaptive\"",
    );
    // Claude Opus 5 accepts "disabled".
    assert_eq!(
        messages(&anthropic, with(hi("claude-opus-5"), disabled))
            .await
            .status,
        200
    );
}

#[tokio::test]
async fn sonnet_5_5_takes_between_tools_in_place_of_disabled() {
    let anthropic = FakeAnthropic::start().await;
    let r = messages(
        &anthropic,
        with(
            hi("claude-sonnet-5-5"),
            json!({ "thinking": { "type": "disabled" } }),
        ),
    )
    .await;
    rejected(&r, "Use \"thinking.type.between_tools\"");
    let between = json!({ "thinking": { "type": "between_tools" } });
    assert_eq!(
        messages(&anthropic, with(hi("claude-sonnet-5-5"), between.clone()))
            .await
            .status,
        200
    );
    let xhigh = with(
        between.clone(),
        json!({ "output_config": { "effort": "xhigh" } }),
    );
    rejected(
        &messages(&anthropic, with(hi("claude-sonnet-5-5"), xhigh)).await,
        "output_config.effort 'xhigh' is not supported",
    );
    let r = messages(&anthropic, with(hi("claude-sonnet-5"), between)).await;
    rejected(
        &r,
        "\"thinking.type.between_tools\" is not supported for this model.",
    );
}

#[tokio::test]
async fn newest_models_reject_forced_tool_use() {
    let anthropic = FakeAnthropic::start().await;
    let forced = json!({ "tool_choice": { "type": "any" } });
    rejected(
        &messages(&anthropic, with(hi("claude-sonnet-5-5"), forced.clone())).await,
        "tool_choice: type \"tool\" and \"any\"",
    );
    assert_eq!(
        messages(&anthropic, with(hi("claude-opus-5"), forced))
            .await
            .status,
        200
    );
}

#[tokio::test]
async fn claude_4_6_and_later_reject_prefill() {
    let anthropic = FakeAnthropic::start().await;
    let prefill = json!({ "messages": [{ "role": "user", "content": "hi" }, { "role": "assistant", "content": "The answer is" }] });
    let r = messages(&anthropic, with(hi("claude-sonnet-4-6"), prefill.clone())).await;
    rejected(&r, "This model does not support assistant message prefill.");
    assert_eq!(
        messages(&anthropic, with(hi("claude-sonnet-4-5"), prefill))
            .await
            .status,
        200
    );
}

#[tokio::test]
async fn claude_4_7_and_later_fix_sampling() {
    let anthropic = FakeAnthropic::start().await;
    rejected(
        &messages(
            &anthropic,
            with(hi("claude-opus-4-7"), json!({ "temperature": 0.5 })),
        )
        .await,
        "temperature",
    );
    rejected(
        &messages(
            &anthropic,
            with(hi("claude-sonnet-5"), json!({ "top_k": 5 })),
        )
        .await,
        "top_k",
    );
    // The default is accepted.
    assert_eq!(
        messages(
            &anthropic,
            with(hi("claude-opus-4-8"), json!({ "temperature": 1.0 }))
        )
        .await
        .status,
        200
    );
    // Older models only while thinking.
    assert_eq!(
        messages(
            &anthropic,
            with(hi("claude-sonnet-4-6"), json!({ "temperature": 0.5 }))
        )
        .await
        .status,
        200
    );
}

#[tokio::test]
async fn extended_only_models_reject_adaptive() {
    let anthropic = FakeAnthropic::start().await;
    let adaptive = json!({ "thinking": { "type": "adaptive" } });
    rejected(
        &messages(&anthropic, with(hi("claude-haiku-4-5"), adaptive.clone())).await,
        "adaptive thinking is not supported on this model",
    );
    assert_eq!(
        messages(&anthropic, with(hi("claude-sonnet-4-6"), adaptive))
            .await
            .status,
        200
    );
}

#[tokio::test]
async fn haiku_4_5_and_opus_4_1_take_one_sampler() {
    let anthropic = FakeAnthropic::start().await;
    let both = json!({ "temperature": 0.3, "top_p": 1.0 });
    for model in ["claude-haiku-4-5-20251001", "claude-opus-4-1"] {
        rejected(
            &messages(&anthropic, with(hi(model), both.clone())).await,
            "cannot both be specified",
        );
    }
    assert_eq!(
        messages(&anthropic, with(hi("claude-opus-4"), both))
            .await
            .status,
        200
    );
}

#[tokio::test]
async fn max_tokens_is_capped_per_model() {
    let anthropic = FakeAnthropic::start().await;
    let r = messages(
        &anthropic,
        with(hi("claude-sonnet-4-5"), json!({ "max_tokens": 64_001 })),
    )
    .await;
    rejected(&r, "64001 > 64000");
    assert_eq!(
        messages(
            &anthropic,
            with(hi("claude-opus-5-5"), json!({ "max_tokens": 128_000 }))
        )
        .await
        .status,
        200
    );
}

#[tokio::test]
async fn context_window_is_per_model() {
    let anthropic = FakeAnthropic::start().await;
    let long = "word ".repeat(200_001);
    let body = with(
        hi("claude-sonnet-4-5"),
        json!({ "messages": [{ "role": "user", "content": long }] }),
    );
    rejected(
        &messages(&anthropic, body.clone()).await,
        "prompt is too long",
    );
    let body = with(body, json!({ "model": "claude-sonnet-4-6" }));
    assert_eq!(messages(&anthropic, body).await.status, 200);
}

// Thinking rules shared by every model that has the mode.

#[tokio::test]
async fn extended_thinking_budget_rules() {
    let anthropic = FakeAnthropic::start().await;
    let small = json!({ "thinking": { "type": "enabled", "budget_tokens": 1000 } });
    rejected(
        &messages(&anthropic, with(hi("claude-sonnet-4-5"), small)).await,
        "greater than or equal to 1024",
    );
    let over =
        json!({ "max_tokens": 2000, "thinking": { "type": "enabled", "budget_tokens": 2000 } });
    rejected(
        &messages(&anthropic, with(hi("claude-sonnet-4-5"), over.clone())).await,
        "must be greater than `thinking.budget_tokens`",
    );
    // Interleaved thinking lets the budget exceed max_tokens.
    let url = format!("{}/v1/messages", anthropic.base_url());
    let beta = [
        HEADERS,
        &[("anthropic-beta", "interleaved-thinking-2025-05-14")],
    ]
    .concat();
    assert_eq!(
        post(&url, &beta, &with(hi("claude-sonnet-4-5"), over))
            .await
            .status,
        200
    );
}

#[tokio::test]
async fn extended_thinking_rejects_forced_tools_and_sampling() {
    let anthropic = FakeAnthropic::start().await;
    let on =
        json!({ "max_tokens": 4000, "thinking": { "type": "enabled", "budget_tokens": 2000 } });
    let forced = with(
        on.clone(),
        json!({ "tool_choice": { "type": "tool", "name": "f" } }),
    );
    rejected(
        &messages(&anthropic, with(hi("claude-sonnet-4-5"), forced)).await,
        "tool_choice forces tool use",
    );
    let hot = with(on.clone(), json!({ "temperature": 0.7 }));
    rejected(
        &messages(&anthropic, with(hi("claude-sonnet-4-5"), hot)).await,
        "`temperature` may only be set to 1",
    );
    let top_k = with(on.clone(), json!({ "top_k": 5 }));
    rejected(
        &messages(&anthropic, with(hi("claude-sonnet-4-5"), top_k)).await,
        "`top_k` must be unset",
    );
    let top_p = with(on.clone(), json!({ "top_p": 0.9 }));
    rejected(
        &messages(&anthropic, with(hi("claude-sonnet-4-5"), top_p)).await,
        "0.95",
    );
    let top_p = with(on, json!({ "top_p": 0.96 }));
    assert_eq!(
        messages(&anthropic, with(hi("claude-sonnet-4-5"), top_p))
            .await
            .status,
        200
    );
}

#[tokio::test]
async fn extended_thinking_tool_loop_must_keep_the_thinking_block() {
    let anthropic = FakeAnthropic::start().await;
    let on =
        json!({ "max_tokens": 4000, "thinking": { "type": "enabled", "budget_tokens": 2000 } });
    let user = json!({ "role": "user", "content": "weather?" });
    let tool = json!({ "type": "tool_use", "id": "toolu_1", "name": "f", "input": {} });
    let result = json!({ "role": "user", "content": [{ "type": "tool_result", "tool_use_id": "toolu_1", "content": "sunny" }] });
    let stripped =
        json!({ "messages": [user, { "role": "assistant", "content": [tool] }, result] });
    rejected(
        &messages(
            &anthropic,
            with(with(hi("claude-sonnet-4-5"), on.clone()), stripped),
        )
        .await,
        "must start with a thinking block",
    );
    let thinking = json!({ "type": "thinking", "thinking": "hm", "signature": "sig" });
    let kept =
        json!({ "messages": [user, { "role": "assistant", "content": [thinking, tool] }, result] });
    assert_eq!(
        messages(&anthropic, with(with(hi("claude-sonnet-4-5"), on), kept))
            .await
            .status,
        200
    );
}

#[tokio::test]
async fn thinking_rejects_prefill() {
    let anthropic = FakeAnthropic::start().await;
    let prefill = json!({ "messages": [{ "role": "user", "content": "hi" }, { "role": "assistant", "content": "So" }] });
    let on = with(
        prefill.clone(),
        json!({ "max_tokens": 4000, "thinking": { "type": "enabled", "budget_tokens": 2000 } }),
    );
    rejected(
        &messages(&anthropic, with(hi("claude-sonnet-4-5"), on)).await,
        "thinking",
    );
}

#[tokio::test]
async fn thinking_blocks_come_first_with_a_signature() {
    let anthropic = FakeAnthropic::start().await;
    let on =
        json!({ "max_tokens": 4000, "thinking": { "type": "enabled", "budget_tokens": 2000 } });
    anthropic.answer(Answer::text("Hi").reasoning("let me think"));
    anthropic.answer(Answer::text("Hi").reasoning("let me think"));
    let v = messages(&anthropic, with(hi("claude-sonnet-4-5"), on.clone()))
        .await
        .json();
    let block = &v["content"][0];
    assert_eq!(
        (&block["type"], &block["thinking"]),
        (&json!("thinking"), &json!("let me think"))
    );
    assert!(block["signature"].as_str().unwrap().starts_with("sig_"));
    assert_eq!(v["content"][1]["text"], "Hi");
    assert_eq!(v["usage"]["output_tokens_details"]["thinking_tokens"], 3);
    assert_eq!(v["usage"]["output_tokens"], 4);

    let r = messages(
        &anthropic,
        with(hi("claude-sonnet-4-5"), with(on, json!({ "stream": true }))),
    )
    .await;
    let starts = data_of(&r, "content_block_start");
    assert_eq!(
        starts[0]["content_block"],
        json!({ "type": "thinking", "thinking": "", "signature": "" })
    );
    assert_eq!(starts[1]["index"], 1);
    let deltas: Vec<Value> = data_of(&r, "content_block_delta")
        .iter()
        .filter(|d| d["index"] == 0)
        .map(|d| d["delta"].clone())
        .collect();
    let (last, thinking) = deltas.split_last().unwrap();
    assert!(thinking.iter().all(|d| d["type"] == "thinking_delta"));
    assert_eq!(last["type"], "signature_delta");
    let end = &data_of(&r, "message_delta")[0];
    assert_eq!(end["usage"]["output_tokens_details"]["thinking_tokens"], 3);
    assert!(
        data_of(&r, "message_start")[0]["message"]["usage"]
            .get("output_tokens_details")
            .is_none()
    );
}

#[tokio::test]
async fn newest_models_think_by_default_with_display_omitted() {
    let anthropic = FakeAnthropic::start().await;
    anthropic.answer(Answer::text("Hi").reasoning("secret"));
    anthropic.answer(Answer::text("Hi").reasoning("secret"));
    anthropic.answer(Answer::text("Hi").reasoning("secret"));
    let v = messages(&anthropic, hi("claude-opus-5-5")).await.json();
    assert_eq!(
        (&v["content"][0]["type"], &v["content"][0]["thinking"]),
        (&json!("thinking"), &json!(""))
    );
    let summarized = json!({ "thinking": { "type": "adaptive", "display": "summarized" } });
    let v = messages(&anthropic, with(hi("claude-opus-5-5"), summarized))
        .await
        .json();
    assert_eq!(v["content"][0]["thinking"], "secret");
    // Off by default before: no thinking block.
    let v = messages(&anthropic, hi("claude-opus-4-8")).await.json();
    assert_eq!(v["content"][0]["type"], "text");
}

// Scripts.

#[tokio::test]
async fn overloaded_529_in_anthropics_shape() {
    let anthropic = FakeAnthropic::start().await;
    anthropic.answer(Answer::error(529).retry_after(3));
    let r = messages(&anthropic, hi("claude-x")).await;
    assert_eq!((r.status, r.header("retry-after")), (529, Some("3")));
    let v = r.json();
    assert_eq!(
        v["error"],
        json!({ "type": "overloaded_error", "message": "Overloaded" })
    );
    assert!(v["request_id"].is_string());
}

#[tokio::test]
async fn error_event_mid_stream() {
    let anthropic = FakeAnthropic::start().await;
    anthropic.answer(Answer::text("one two three").error_after(3));
    let r = messages(&anthropic, with(hi("claude-x"), json!({ "stream": true }))).await;
    assert!(r.ended.is_ok());
    assert_eq!(
        r.events(),
        ["message_start", "content_block_start", "ping", "error"]
    );
    assert_eq!(data_of(&r, "error")[0]["error"]["type"], "overloaded_error");
}

#[tokio::test]
async fn cut_after_breaks_the_stream() {
    let anthropic = FakeAnthropic::start().await;
    anthropic.answer(Answer::text("one two three").cut_after(4));
    let r = messages(&anthropic, with(hi("claude-x"), json!({ "stream": true }))).await;
    assert!(r.ended.is_err());
    assert_eq!(r.events().len(), 4);
}

#[tokio::test]
async fn requests_records_what_was_sent() {
    let anthropic = FakeAnthropic::start().await;
    messages(&anthropic, hi("claude-x")).await;
    let seen = anthropic.requests();
    assert_eq!(
        (seen[0].method.as_str(), seen[0].path.as_str()),
        ("POST", "/v1/messages")
    );
    assert_eq!(seen[0].header("anthropic-version"), Some("2023-06-01"));
    assert_eq!(seen[0].body, hi("claude-x"));
}
