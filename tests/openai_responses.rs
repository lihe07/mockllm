#![allow(clippy::unwrap_used)]

mod support;

use std::time::{Duration, Instant};

use mockllm::{Answer, FakeOpenAi};
use serde_json::{Value, json};
use support::{Reply, post};

const KEY: (&str, &str) = ("authorization", "Bearer sk-test");

async fn create(openai: &FakeOpenAi, body: Value) -> Reply {
    post(&format!("{}/responses", openai.base_url()), &[KEY], &body).await
}

fn hi(model: &str) -> Value {
    json!({ "model": model, "input": "say hi" })
}

fn with(mut body: Value, extra: Value) -> Value {
    for (k, v) in extra.as_object().unwrap() {
        body[k] = v.clone();
    }
    body
}

fn streamed(body: Value) -> Value {
    with(body, json!({ "stream": true }))
}

/// Asserts a 400 in OpenAI's error shape with `code` and a message
/// containing `says`.
fn rejected(r: &Reply, code: &str, says: &str) {
    assert_eq!(r.status, 400, "{}", r.body);
    let e = &r.json()["error"];
    for field in ["message", "type", "param", "code"] {
        assert!(e.get(field).is_some(), "error lacks {field}: {}", r.body);
    }
    assert_eq!(e["type"], "invalid_request_error");
    let (kind, message) = r.error();
    assert_eq!(kind, code, "{}", r.body);
    assert!(message.contains(says), "{message}");
}

/// Every SSE event is `event: <type>` then `data:` with that `type`,
/// numbered from 0.
fn events(r: &Reply) -> Vec<Value> {
    assert!(
        r.header("content-type")
            .unwrap()
            .starts_with("text/event-stream")
    );
    let chunks = r.chunks();
    assert_eq!(r.events().len(), chunks.len());
    for (i, (name, c)) in r.events().iter().zip(&chunks).enumerate() {
        assert_eq!(&c["type"], name);
        if c["type"] != "error" {
            assert_eq!(c["sequence_number"], i, "{c}");
        }
    }
    chunks
}

fn types(events: &[Value]) -> Vec<&str> {
    events.iter().map(|e| e["type"].as_str().unwrap()).collect()
}

#[tokio::test]
async fn whole_text_answer_has_the_response_shape() {
    let openai = FakeOpenAi::start().await;
    openai.answer(Answer::text("Hi there"));
    let r = create(&openai, hi("gpt-4o")).await;
    assert_eq!(r.status, 200);
    assert!(r.header("x-request-id").is_some());
    let v = r.json();
    assert_eq!(v["object"], "response");
    assert!(v["id"].as_str().unwrap().starts_with("resp_"));
    assert_eq!(
        (&v["status"], &v["model"]),
        (&json!("completed"), &json!("gpt-4o"))
    );
    assert!(v["created_at"].is_u64() && v["completed_at"].is_u64());
    assert_eq!(v["store"], true);
    assert_eq!(v["tool_choice"], "auto");
    assert_eq!(v["text"], json!({ "format": { "type": "text" } }));
    let item = &v["output"][0];
    assert_eq!(
        (&item["type"], &item["role"]),
        (&json!("message"), &json!("assistant"))
    );
    assert!(item["id"].as_str().unwrap().starts_with("msg_"));
    assert_eq!(
        item["content"],
        json!([{ "type": "output_text", "text": "Hi there", "annotations": [], "logprobs": [] }])
    );
    let u = &v["usage"];
    assert_eq!(u["output_tokens"], 2);
    assert_eq!(u["output_tokens_details"]["reasoning_tokens"], 0);
    assert_eq!(u["input_tokens_details"]["cached_tokens"], 0);
    let total = u["input_tokens"].as_u64().unwrap() + 2;
    assert_eq!(u["total_tokens"], total);
}

#[tokio::test]
async fn unscripted_requests_get_a_short_text() {
    let openai = FakeOpenAi::start().await;
    let v = create(&openai, hi("gpt-4o")).await.json();
    assert_eq!(v["output"][0]["content"][0]["text"], "Hi there");
}

#[tokio::test]
async fn stream_sends_named_events_in_order() {
    let openai = FakeOpenAi::start().await;
    openai.answer(Answer::text("Hi there you"));
    let r = create(&openai, streamed(hi("gpt-4o"))).await;
    assert_eq!(r.status, 200);
    assert!(!r.body.contains("[DONE]"));
    let all = events(&r);
    assert_eq!(
        types(&all),
        [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ]
    );
    let created = &all[0]["response"];
    assert_eq!(created["status"], "in_progress");
    assert!(created["usage"].is_null() && created["output"] == json!([]));
    let deltas: Vec<&Value> = all
        .iter()
        .filter(|e| e["type"] == "response.output_text.delta")
        .collect();
    let text: String = deltas
        .iter()
        .map(|d| d["delta"].as_str().unwrap())
        .collect();
    assert_eq!(text, "Hi there you");
    assert!(
        deltas
            .iter()
            .all(|d| d["obfuscation"].is_string() && d["item_id"] == all[2]["item"]["id"])
    );
    assert_eq!(all[7]["text"], "Hi there you");
    let done = &all[10]["response"];
    assert_eq!(done["status"], "completed");
    assert_eq!(done["output"][0]["content"][0]["text"], "Hi there you");
    assert_eq!(done["usage"]["output_tokens"], 3);
}

#[tokio::test]
async fn include_obfuscation_false_drops_it() {
    let openai = FakeOpenAi::start().await;
    let body = json!({ "stream": true, "stream_options": { "include_obfuscation": false } });
    let r = create(&openai, with(hi("gpt-4o"), body)).await;
    assert!(events(&r).iter().all(|e| e.get("obfuscation").is_none()));
}

#[tokio::test]
async fn tool_calls_are_function_call_items() {
    let openai = FakeOpenAi::start().await;
    let args = json!({ "city": "Paris", "unit": "celsius" });
    openai.answer(Answer::tool_call("get_weather", args.clone()));
    openai.answer(Answer::tool_call("get_weather", args.clone()));
    let v = create(&openai, hi("gpt-4o")).await.json();
    let call = &v["output"][0];
    assert_eq!(
        (&call["type"], &call["name"]),
        (&json!("function_call"), &json!("get_weather"))
    );
    assert!(call["id"].as_str().unwrap().starts_with("fc_"));
    assert!(call["call_id"].as_str().unwrap().starts_with("call_"));
    assert_eq!(call["status"], "completed");
    let parsed: Value = serde_json::from_str(call["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(parsed, args);

    let all = events(&create(&openai, streamed(hi("gpt-4o"))).await);
    let added = &all[2];
    assert_eq!(added["item"]["arguments"], "");
    assert_eq!(added["item"]["status"], "in_progress");
    let pieces: Vec<&str> = all
        .iter()
        .filter(|e| e["type"] == "response.function_call_arguments.delta")
        .map(|e| e["delta"].as_str().unwrap())
        .collect();
    assert!(pieces.len() > 2);
    let joined: Value = serde_json::from_str(&pieces.concat()).unwrap();
    assert_eq!(joined, args);
    let done = all
        .iter()
        .find(|e| e["type"] == "response.function_call_arguments.done")
        .unwrap();
    assert_eq!(done["arguments"], args.to_string());
    assert_eq!(done["item_id"], added["item"]["id"]);
}

#[tokio::test]
async fn reasoning_models_answer_a_reasoning_item_first() {
    let openai = FakeOpenAi::start().await;
    openai.answer(Answer::text("Hi").reasoning("one two three"));
    let v = create(&openai, hi("o3")).await.json();
    let rs = &v["output"][0];
    assert_eq!(
        (&rs["type"], &rs["summary"]),
        (&json!("reasoning"), &json!([]))
    );
    assert!(
        rs.get("encrypted_content").is_none(),
        "only when included: {rs}"
    );
    assert_eq!(v["output"][1]["type"], "message");
    assert_eq!(v["usage"]["output_tokens_details"]["reasoning_tokens"], 3);
    assert_eq!(v["usage"]["output_tokens"], 4);

    // With `reasoning.summary`, the summary streams.
    openai.answer(Answer::text("Hi").reasoning("one two three"));
    let body = with(
        streamed(hi("o3")),
        json!({ "reasoning": { "summary": "auto" } }),
    );
    let all = events(&create(&openai, body).await);
    let t = types(&all);
    assert_eq!(
        &t[2..5],
        [
            "response.output_item.added",
            "response.reasoning_summary_part.added",
            "response.reasoning_summary_text.delta"
        ]
    );
    let done = all
        .iter()
        .find(|e| e["type"] == "response.reasoning_summary_text.done")
        .unwrap();
    assert_eq!(
        (&done["text"], &done["summary_index"]),
        (&json!("one two three"), &json!(0))
    );
    let last = &all.last().unwrap()["response"];
    assert_eq!(last["output"][0]["summary"][0]["text"], "one two three");
    let message_added = all
        .iter()
        .filter(|e| e["type"] == "response.output_item.added")
        .nth(1)
        .unwrap();
    assert_eq!(message_added["output_index"], 1);

    // A model that does not reason shows and counts none.
    openai.answer(Answer::text("Hi").reasoning("one two three"));
    let v = create(&openai, hi("gpt-4o")).await.json();
    assert_eq!(v["output"].as_array().unwrap().len(), 1);
    assert_eq!(v["usage"]["output_tokens_details"]["reasoning_tokens"], 0);
}

#[tokio::test]
async fn encrypted_reasoning_comes_when_included() {
    let openai = FakeOpenAi::start().await;
    openai.answer(Answer::text("Hi").reasoning("one two"));
    let body = with(
        hi("o3"),
        json!({ "include": ["reasoning.encrypted_content"] }),
    );
    let v = create(&openai, body.clone()).await.json();
    assert!(v["output"][0]["encrypted_content"].is_string(), "{v}");

    openai.answer(Answer::text("Hi").reasoning("one two"));
    let all = events(&create(&openai, streamed(body)).await);
    let done = all
        .iter()
        .find(|e| e["type"] == "response.output_item.done")
        .unwrap();
    assert!(done["item"]["encrypted_content"].is_string(), "{done}");
}

/// A reasoning model's call, stateless: the call's `function_call` and
/// `reasoning` items, as the response gave them.
async fn reasoned_call(openai: &FakeOpenAi, store: bool) -> (Value, Value) {
    openai.answer(Answer::tool_call("f", json!({})).reasoning("think"));
    let body = with(
        hi("o3"),
        json!({ "store": store, "include": ["reasoning.encrypted_content"] }),
    );
    let v = create(openai, body).await.json();
    let output = v["output"].as_array().unwrap();
    assert_eq!(output[0]["type"], "reasoning", "{v}");
    assert_eq!(output[1]["type"], "function_call", "{v}");
    (output[0].clone(), output[1].clone())
}

#[tokio::test]
async fn a_reasoned_call_needs_its_reasoning_item_back() {
    let openai = FakeOpenAi::start().await;
    let (rs, fc) = reasoned_call(&openai, true).await;
    let user = json!({ "role": "user", "content": "say hi" });
    let output =
        json!({ "type": "function_call_output", "call_id": fc["call_id"], "output": "ok" });
    let next = |input: Value| json!({ "model": "o3", "input": input });

    let r = create(&openai, next(json!([user, fc, output]))).await;
    let says = format!(
        "Item '{}' of type 'function_call' was provided without its required 'reasoning' item: '{}'.",
        fc["id"].as_str().unwrap(),
        rs["id"].as_str().unwrap()
    );
    assert_eq!(r.status, 400, "{}", r.body);
    assert!(r.error().1.contains(&says), "{}", r.body);

    // With it before the call, or the call without its id, it is taken.
    let r = create(&openai, next(json!([user, rs, fc, output]))).await;
    assert_eq!(r.status, 200, "{}", r.body);
    let mut bare = fc.clone();
    bare.as_object_mut().unwrap().remove("id");
    let r = create(&openai, next(json!([user, bare, output]))).await;
    assert_eq!(r.status, 200, "{}", r.body);

    // A model that does not reason makes calls of no reasoning.
    openai.answer(Answer::tool_call("f", json!({})).reasoning("think"));
    let v = create(&openai, hi("gpt-4o")).await.json();
    let fc = v["output"][0].clone();
    let output =
        json!({ "type": "function_call_output", "call_id": fc["call_id"], "output": "ok" });
    let r = create(&openai, next(json!([user, fc, output]))).await;
    assert_eq!(r.status, 200, "{}", r.body);
}

#[tokio::test]
async fn unstored_reasoning_comes_back_encrypted() {
    let openai = FakeOpenAi::start().await;
    let (rs, fc) = reasoned_call(&openai, false).await;
    let user = json!({ "role": "user", "content": "say hi" });
    let output =
        json!({ "type": "function_call_output", "call_id": fc["call_id"], "output": "ok" });
    let next = |input: Value| json!({ "model": "o3", "store": false, "input": input });

    let r = create(&openai, next(json!([user, rs, fc, output]))).await;
    assert_eq!(r.status, 200, "{}", r.body);

    let mut plain = rs.clone();
    plain.as_object_mut().unwrap().remove("encrypted_content");
    let r = create(&openai, next(json!([user, plain, fc, output]))).await;
    assert_eq!(r.status, 404, "{}", r.body);
    assert_eq!(r.json()["error"]["param"], "input");
    assert!(
        r.error()
            .1
            .contains("Items are not persisted when `store` is set to false"),
        "{}",
        r.body
    );

    // A stored response's reasoning is found by its id alone.
    let (rs, fc) = reasoned_call(&openai, true).await;
    let mut plain = rs.clone();
    plain.as_object_mut().unwrap().remove("encrypted_content");
    let output =
        json!({ "type": "function_call_output", "call_id": fc["call_id"], "output": "ok" });
    let r = create(&openai, next(json!([user, plain, fc, output]))).await;
    assert_eq!(r.status, 200, "{}", r.body);
}

#[tokio::test]
async fn raw_events_are_named_by_their_type() {
    let openai = FakeOpenAi::start().await;
    openai.answer(Answer::events(vec![
        json!({ "type": "response.custom", "x": 1 }),
    ]));
    let r = create(&openai, streamed(hi("gpt-4o"))).await;
    assert_eq!(r.events(), ["response.custom"]);
    openai.answer(Answer::embedding(vec![0.1]));
    let r = create(&openai, hi("gpt-4o")).await;
    assert_eq!(r.status, 500);
    assert!(r.body.contains("responses cannot answer an embedding"));
}

#[tokio::test]
async fn validation_follows_openais_rules() {
    let openai = FakeOpenAi::start().await;
    let url = format!("{}/responses", openai.base_url());
    assert_eq!(post(&url, &[], &hi("gpt-4o")).await.status, 401);

    let cases = [
        (
            json!({ "input": "hi" }),
            "missing_required_parameter",
            "'model'",
        ),
        (
            with(hi("gpt-4o"), json!({ "messages": [] })),
            "unrecognized_request_argument",
            "messages",
        ),
        (
            with(
                hi("gpt-4o"),
                json!({ "previous_response_id": "resp_1", "conversation": "conv_1" }),
            ),
            "mutually_exclusive_parameters",
            "'previous_response_id' or 'conversation'",
        ),
        (
            with(hi("gpt-4o"), json!({ "temperature": 3 })),
            "decimal_above_max_value",
            "<= 2",
        ),
        (
            with(hi("gpt-4o"), json!({ "top_logprobs": 21 })),
            "integer_above_max_value",
            "<= 20",
        ),
        (
            with(hi("gpt-4o"), json!({ "include": ["nope"] })),
            "invalid_value",
            "'nope'",
        ),
        (
            with(hi("gpt-4o"), json!({ "input": { "role": "user" } })),
            "invalid_type",
            "'input'",
        ),
        (
            json!({ "model": "gpt-4o", "input": [{ "role": "bot", "content": "x" }] }),
            "invalid_value",
            "'bot'",
        ),
        (
            with(
                hi("gpt-4o"),
                json!({ "tools": [{ "type": "function", "function": { "name": "f" } }] }),
            ),
            "missing_required_parameter",
            "'tools[0].name'",
        ),
    ];
    for (body, code, says) in cases {
        rejected(&create(&openai, body).await, code, says);
    }
    let options = with(
        hi("gpt-4o"),
        json!({ "stream_options": { "include_obfuscation": false } }),
    );
    let r = create(&openai, options).await;
    assert_eq!(r.status, 400);
    assert!(
        r.error()
            .1
            .contains("only allowed when 'stream' is enabled")
    );
    // A prompt template carries the model.
    let prompt = json!({ "prompt": { "id": "pmpt_1" }, "input": "hi" });
    assert_eq!(create(&openai, prompt).await.status, 200);
    // Nothing valid was sent before, so no answer was used up.
    openai.answer(Answer::text("kept"));
    let v = create(&openai, hi("gpt-4o")).await.json();
    assert_eq!(v["output"][0]["content"][0]["text"], "kept");
}

#[tokio::test]
async fn function_calls_and_outputs_pair_by_call_id() {
    let openai = FakeOpenAi::start().await;
    let user = json!({ "role": "user", "content": "weather?" });
    let call =
        json!({ "type": "function_call", "call_id": "call_1", "name": "f", "arguments": "{}" });
    let output = json!({ "type": "function_call_output", "call_id": "call_1", "output": "sunny" });
    let body = |input: Value| json!({ "model": "gpt-4o", "input": input });
    let r = create(&openai, body(json!([user, output]))).await;
    assert_eq!(r.status, 400);
    assert!(
        r.error()
            .1
            .contains("No tool call found for function call output with call_id call_1")
    );
    let r = create(&openai, body(json!([user, call]))).await;
    assert!(
        r.error()
            .1
            .contains("No tool output found for function call call_1")
    );
    assert_eq!(
        create(&openai, body(json!([user, call, output])))
            .await
            .status,
        200
    );
    // The call may be in the stored previous response.
    let chained = with(
        body(json!([output])),
        json!({ "previous_response_id": "resp_1" }),
    );
    assert_eq!(create(&openai, chained).await.status, 200);
}

#[tokio::test]
async fn per_model_rules_from_the_chat_table() {
    let openai = FakeOpenAi::start().await;
    let r = create(
        &openai,
        with(hi("gpt-4o"), json!({ "max_output_tokens": 16_385 })),
    )
    .await;
    rejected(&r, "invalid_value", "at most 16384");
    let r = create(
        &openai,
        with(hi("gpt-5"), json!({ "reasoning": { "effort": "none" } })),
    )
    .await;
    rejected(&r, "unsupported_value", "'reasoning.effort'");
    let r = create(&openai, with(hi("gpt-5-mini"), json!({ "top_p": 0.5 }))).await;
    rejected(&r, "unsupported_parameter", "'top_p'");
    let astra = with(
        hi("gpt-6-astra"),
        json!({ "include": ["message.output_text.logprobs"] }),
    );
    rejected(
        &create(&openai, astra).await,
        "unsupported_parameter",
        "message.output_text.logprobs",
    );
    // Sampling at effort none; tools on GPT-6 Astra (they require Responses).
    let sol = with(
        hi("gpt-6-sol"),
        json!({ "reasoning": { "effort": "none" }, "temperature": 0.3 }),
    );
    assert_eq!(create(&openai, sol).await.status, 200);
    let tools = json!({ "tools": [{ "type": "function", "name": "f", "parameters": {} }] });
    let v = create(&openai, with(hi("gpt-6-astra"), tools)).await.json();
    assert_eq!(v["tools"][0]["name"], "f");
}

// Scripts.

#[tokio::test]
async fn scripted_error_in_openais_shape_with_retry_after() {
    let openai = FakeOpenAi::start().await;
    openai.answer(Answer::error(429).retry_after(7));
    let r = create(&openai, hi("gpt-4o")).await;
    assert_eq!(r.status, 429);
    assert_eq!(r.header("retry-after"), Some("7"));
    assert_eq!(r.json()["error"]["code"], "rate_limit_exceeded");
}

#[tokio::test]
async fn error_after_sends_an_error_event() {
    let openai = FakeOpenAi::start().await;
    openai.answer(Answer::text("one two three").error_after(3));
    let r = create(&openai, streamed(hi("gpt-4o"))).await;
    assert!(r.ended.is_ok());
    let all = events(&r);
    assert_eq!(all.len(), 4);
    let e = &all[3];
    assert_eq!(
        (&e["type"], &e["code"], &e["sequence_number"]),
        (&json!("error"), &json!("server_error"), &json!(3))
    );
    assert!(e["param"].is_null() && e["message"].is_string());
}

#[tokio::test]
async fn cut_after_breaks_the_stream_and_whole_bodies() {
    let openai = FakeOpenAi::start().await;
    openai.answer(Answer::text("one two three").cut_after(2));
    let r = create(&openai, streamed(hi("gpt-4o"))).await;
    assert!(r.ended.is_err());
    assert_eq!(r.events(), ["response.created", "response.in_progress"]);
    openai.answer(Answer::text("Hi").cut_after(0));
    let r = create(&openai, hi("gpt-4o")).await;
    assert!(r.ended.is_err());
    assert!(serde_json::from_str::<Value>(&r.body).is_err());
}

#[tokio::test]
async fn delay_first_and_pace() {
    let openai = FakeOpenAi::start().await;
    openai.answer(Answer::text("a").delay_first(300));
    let t = Instant::now();
    create(&openai, hi("gpt-4o")).await;
    assert!(t.elapsed() >= Duration::from_millis(300));
    openai.answer(Answer::text("a").pace(40));
    let t = Instant::now();
    let r = create(&openai, streamed(hi("gpt-4o"))).await;
    // 9 events: 8 gaps.
    assert_eq!(r.events().len(), 9);
    assert!(t.elapsed() >= Duration::from_millis(320));
}
