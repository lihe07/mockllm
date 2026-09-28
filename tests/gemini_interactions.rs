#![allow(clippy::unwrap_used)]

mod support;

use mockllm::{Answer, FakeGemini};
use serde_json::{Value, json};
use support::{Reply, post};

const KEY: (&str, &str) = ("x-goog-api-key", "AIza-test");

async fn create(gemini: &FakeGemini, body: Value) -> Reply {
    let url = format!("{}/v1beta/interactions", gemini.base_url());
    post(&url, &[KEY], &body).await
}

fn hi() -> Value {
    json!({ "model": "gemini-3.8-flash", "input": "say hi" })
}

fn with(mut body: Value, extra: Value) -> Value {
    for (k, v) in extra.as_object().unwrap() {
        body[k] = v.clone();
    }
    body
}

fn rejected(r: &Reply, says: &str) {
    assert_eq!(r.status, 400, "{}", r.body);
    let message = r.json()["error"]["message"].as_str().unwrap().to_owned();
    assert!(message.contains(says), "{message}");
}

#[tokio::test]
async fn a_whole_interaction() {
    let gemini = FakeGemini::start().await;
    gemini.answer(Answer::text("Hi there"));
    let v = create(&gemini, hi()).await.json();
    assert_eq!(
        (&v["object"], &v["status"]),
        (&json!("interaction"), &json!("completed"))
    );
    assert_eq!(v["model"], "gemini-3.8-flash");
    assert!(v["id"].as_str().unwrap().starts_with("v1_"));
    assert!(v["created"].as_str().unwrap().ends_with('Z'));
    assert_eq!(
        v["steps"],
        json!([{ "type": "model_output", "content": [{ "type": "text", "text": "Hi there" }] }])
    );
    let u = &v["usage"];
    assert_eq!(
        (&u["total_input_tokens"], &u["total_output_tokens"]),
        (&json!(2), &json!(2))
    );
    assert_eq!(u["total_tokens"], 4);

    // A function call needs action; the v1 path works too.
    gemini.answer(Answer::tool_call(
        "get_weather",
        json!({ "location": "Boston" }),
    ));
    let url = format!("{}/v1/interactions", gemini.base_url());
    let v = post(&url, &[KEY], &hi()).await.json();
    assert_eq!(v["status"], "requires_action");
    let step = &v["steps"][0];
    assert_eq!(
        (&step["type"], &step["name"]),
        (&json!("function_call"), &json!("get_weather"))
    );
    assert_eq!(step["arguments"], json!({ "location": "Boston" }));

    // Reasoning: a thought step, its summary with `thinking_summaries: auto`.
    gemini.answer(Answer::text("42").reasoning("think hard"));
    let body = with(
        hi(),
        json!({ "generation_config": { "thinking_summaries": "auto" } }),
    );
    let v = create(&gemini, body).await.json();
    assert_eq!(v["steps"][0]["type"], "thought");
    assert_eq!(v["steps"][0]["summary"][0]["text"], "think hard");
    assert_eq!(v["usage"]["total_thought_tokens"], 2);
}

#[tokio::test]
async fn streams_named_events_ending_in_done() {
    let gemini = FakeGemini::start().await;
    gemini.answer(Answer::text("Hi there you"));
    let r = create(&gemini, with(hi(), json!({ "stream": true }))).await;
    assert!(
        r.header("content-type")
            .unwrap()
            .starts_with("text/event-stream")
    );
    let names = r.events();
    let expected = [
        "interaction.created",
        "interaction.status_update",
        "step.start",
        "step.delta",
        "step.delta",
        "step.delta",
        "step.stop",
        "interaction.completed",
        "done",
    ];
    assert_eq!(names, expected);
    assert_eq!(r.data().last().unwrap(), "[DONE]");
    let events = r.chunks();
    // Every payload names itself in `event_type`.
    for (e, name) in events.iter().zip(&names) {
        assert_eq!(&e["event_type"], name);
    }
    assert_eq!(events[0]["interaction"]["status"], "in_progress");
    assert_eq!(events[2]["step"], json!({ "type": "model_output" }));
    let text: String = events[3..6]
        .iter()
        .map(|e| e["delta"]["text"].as_str().unwrap())
        .collect();
    assert_eq!(text, "Hi there you");
    let done = &events[7]["interaction"];
    assert_eq!(done["status"], "completed");
    assert!(done.get("steps").is_none());
    assert_eq!(done["usage"]["total_output_tokens"], 3);

    // Function call arguments stream as JSON pieces.
    gemini.answer(Answer::tool_call(
        "f",
        json!({ "location": "San Francisco, CA" }),
    ));
    let events = create(&gemini, with(hi(), json!({ "stream": true })))
        .await
        .chunks();
    assert_eq!(events[2]["step"]["arguments"], json!({}));
    let args: String = events
        .iter()
        .filter(|e| e["delta"]["type"] == "arguments_delta")
        .map(|e| e["delta"]["arguments"].as_str().unwrap())
        .collect();
    assert_eq!(args, json!({ "location": "San Francisco, CA" }).to_string());
}

#[tokio::test]
async fn validation_follows_the_reference() {
    let gemini = FakeGemini::start().await;
    rejected(
        &create(&gemini, json!({ "input": "x" })).await,
        "model or agent",
    );
    let both = json!({ "model": "gemini-3.8-flash", "agent": "deep-research-preview-04-2026" });
    rejected(&create(&gemini, both).await, "model or agent");
    rejected(
        &create(&gemini, with(hi(), json!({ "messages": [] }))).await,
        "messages",
    );
    // `agent_config` is an agent's, `generation_config` a model's.
    rejected(
        &create(&gemini, with(hi(), json!({ "agent_config": {} }))).await,
        "agent_config",
    );
    let agent =
        json!({ "agent": "deep-research-preview-04-2026", "input": "x", "agent_config": {} });
    assert_eq!(create(&gemini, agent).await.status, 200);
    let body = with(hi(), json!({ "response_format": { "type": "object" } }));
    rejected(&create(&gemini, body).await, "response_mime_type");
    let body = with(
        hi(),
        json!({ "generation_config": { "thinking_level": "max" } }),
    );
    rejected(&create(&gemini, body).await, "thinking_level");
    rejected(
        &create(&gemini, with(hi(), json!({ "service_tier": "gold" }))).await,
        "service_tier",
    );
    let url = format!("{}/v2/interactions", gemini.base_url());
    assert_eq!(post(&url, &[KEY], &hi()).await.status, 404);
    let url = format!("{}/v1beta/interactions", gemini.base_url());
    assert_eq!(post(&url, &[], &hi()).await.status, 403);
}

#[tokio::test]
async fn scripts_errors_and_breaks() {
    let gemini = FakeGemini::start().await;
    let stream = with(hi(), json!({ "stream": true }));
    gemini.answer(Answer::error(503));
    let r = create(&gemini, stream.clone()).await;
    assert_eq!(
        (r.status, &r.json()["error"]["status"]),
        (503, &json!("UNAVAILABLE"))
    );

    gemini.answer(Answer::text("a b c").error_after(3));
    let r = create(&gemini, stream.clone()).await;
    assert_eq!(r.events().last().unwrap(), "error");
    assert_eq!(
        r.chunks().last().unwrap()["error"]["code"],
        "gateway_timeout"
    );

    gemini.answer(Answer::text("a b c").cut_after(2));
    let r = create(&gemini, stream.clone()).await;
    assert!(r.ended.is_err());
    assert_eq!(r.events().len(), 2);

    gemini.answer(Answer::events(vec![
        json!({ "event_type": "step.stop", "index": 0 }),
    ]));
    let r = create(&gemini, stream).await;
    assert_eq!(
        r.body,
        "event: step.stop\ndata: {\"event_type\":\"step.stop\",\"index\":0}\n\n"
    );

    gemini.answer(Answer::embedding(vec![1.0]));
    assert_eq!(create(&gemini, hi()).await.status, 500);
}
