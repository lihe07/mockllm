#![allow(clippy::unwrap_used)]

mod anthropic_support;
mod support;

use anthropic_support::{call, get, post, rejected};
use mockllm::{Answer, FakeAnthropic};
use serde_json::{Value, json};

const PATH: &str = "/v1/messages/batches";

fn params(model: &str) -> Value {
    json!({ "model": model, "max_tokens": 1024, "messages": [{ "role": "user", "content": "hi" }] })
}

fn batch(requests: &[(&str, Value)]) -> Value {
    let r: Vec<Value> = requests
        .iter()
        .map(|(id, p)| json!({ "custom_id": id, "params": p }))
        .collect();
    json!({ "requests": r })
}

fn lines(body: &str) -> Vec<Value> {
    body.lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

#[tokio::test]
async fn create_poll_and_read_results() {
    let anthropic = FakeAnthropic::start().await;
    anthropic.answer(Answer::text("Hello again"));
    let body = batch(&[
        ("a", params("claude-sonnet-4-5")),
        ("b", params("claude-opus-4-8")),
    ]);
    let r = post(&anthropic, PATH, &body).await;
    assert_eq!(r.status, 200);
    let v = r.json();
    let id = v["id"].as_str().unwrap().to_owned();
    assert!(id.starts_with("msgbatch_"));
    assert_eq!(
        (&v["type"], &v["processing_status"]),
        (&json!("message_batch"), &json!("in_progress"))
    );
    let counts =
        json!({ "processing": 2, "succeeded": 0, "errored": 0, "canceled": 0, "expired": 0 });
    assert_eq!(v["request_counts"], counts);
    assert_eq!(
        (&v["ended_at"], &v["results_url"], &v["cancel_initiated_at"]),
        (&Value::Null, &Value::Null, &Value::Null)
    );
    assert!(v["expires_at"].as_str().unwrap().ends_with('Z'));
    // Results are not there until processing ends.
    let r = get(&anthropic, &format!("{PATH}/{id}/results")).await;
    rejected(&r, 400, "invalid_request_error", "not finished");
    // The first poll finds it ended.
    let v = get(&anthropic, &format!("{PATH}/{id}")).await.json();
    assert_eq!(v["processing_status"], "ended");
    assert_eq!(v["request_counts"]["succeeded"], 2);
    assert_eq!(v["request_counts"]["processing"], 0);
    assert!(v["ended_at"].is_string());
    let url = v["results_url"].as_str().unwrap();
    assert!(url.ends_with(&format!("/v1/messages/batches/{id}/results")));
    let r = get(&anthropic, &format!("{PATH}/{id}/results")).await;
    assert_eq!(r.status, 200);
    let out = lines(&r.body);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0]["custom_id"], "a");
    let m = &out[1]["result"]["message"];
    assert_eq!(out[1]["result"]["type"], "succeeded");
    assert_eq!(
        (&m["type"], &m["model"]),
        (&json!("message"), &json!("claude-opus-4-8"))
    );
    assert_eq!(
        m["content"],
        json!([{ "type": "text", "text": "Hello again" }])
    );
    // Deleted once ended; then gone.
    let r = call(
        &anthropic,
        "DELETE",
        &format!("{PATH}/{id}"),
        &[],
        None,
        None,
    )
    .await;
    assert_eq!(
        r.json(),
        json!({ "id": id, "type": "message_batch_deleted" })
    );
    rejected(
        &get(&anthropic, &format!("{PATH}/{id}")).await,
        404,
        "not_found_error",
        &id,
    );
}

#[tokio::test]
async fn params_are_checked_per_request_as_errored_results() {
    let anthropic = FakeAnthropic::start().await;
    let mut streaming = params("claude-sonnet-4-5");
    streaming["stream"] = json!(true);
    let mut empty = params("claude-sonnet-4-5");
    empty["max_tokens"] = json!(0);
    let body = batch(&[
        ("ok", params("claude-sonnet-4-5")),
        ("no-max", json!({ "model": "m", "messages": [] })),
        ("stream", streaming),
        ("zero", empty),
        ("tool", params("claude-sonnet-4-5")),
    ]);
    anthropic.answer(Answer::tool_call("get_weather", json!({ "city": "Paris" })));
    let id = post(&anthropic, PATH, &body).await.json()["id"]
        .as_str()
        .unwrap()
        .to_owned();
    let v = get(&anthropic, &format!("{PATH}/{id}")).await.json();
    assert_eq!(
        (
            &v["request_counts"]["succeeded"],
            &v["request_counts"]["errored"]
        ),
        (&json!(2), &json!(3))
    );
    let out = lines(&get(&anthropic, &format!("{PATH}/{id}/results")).await.body);
    let e = &out[1]["result"];
    assert_eq!(e["type"], "errored");
    assert_eq!(e["error"]["type"], "error");
    assert_eq!(e["error"]["error"]["type"], "invalid_request_error");
    assert!(
        e["error"]["error"]["message"]
            .as_str()
            .unwrap()
            .contains("max_tokens: Field required")
    );
    assert!(e["error"]["request_id"].is_string());
    assert!(
        out[2]["result"]["error"]["error"]["message"]
            .as_str()
            .unwrap()
            .contains("stream")
    );
    assert_eq!(
        out[4]["result"]["message"]["content"][0]["type"],
        "tool_use"
    );
    assert_eq!(out[4]["result"]["message"]["stop_reason"], "tool_use");
}

#[tokio::test]
async fn cancel_then_all_requests_are_canceled() {
    let anthropic = FakeAnthropic::start().await;
    let id = post(
        &anthropic,
        PATH,
        &batch(&[("a", params("m")), ("b", params("m"))]),
    )
    .await
    .json()["id"]
        .as_str()
        .unwrap()
        .to_owned();
    let r = call(
        &anthropic,
        "DELETE",
        &format!("{PATH}/{id}"),
        &[],
        None,
        None,
    )
    .await;
    rejected(&r, 400, "invalid_request_error", "Cancel it first");
    let v = post(&anthropic, &format!("{PATH}/{id}/cancel"), &json!(null))
        .await
        .json();
    assert_eq!(v["processing_status"], "canceling");
    assert!(v["cancel_initiated_at"].is_string());
    // Listing ends it too.
    let list = get(&anthropic, PATH).await.json();
    let v = &list["data"][0];
    assert_eq!(
        (&v["processing_status"], &v["request_counts"]["canceled"]),
        (&json!("ended"), &json!(2))
    );
    let out = lines(&get(&anthropic, &format!("{PATH}/{id}/results")).await.body);
    assert_eq!(out[0]["result"], json!({ "type": "canceled" }));
}

#[tokio::test]
async fn lists_newest_first_with_paging() {
    let anthropic = FakeAnthropic::start().await;
    let mut ids = Vec::new();
    for _ in 0..3 {
        let v = post(&anthropic, PATH, &batch(&[("a", params("m"))]))
            .await
            .json();
        ids.push(v["id"].as_str().unwrap().to_owned());
    }
    let v = get(&anthropic, &format!("{PATH}?limit=2")).await.json();
    assert_eq!(
        (&v["data"][0]["id"], &v["has_more"]),
        (&json!(ids[2]), &json!(true))
    );
    let next = get(
        &anthropic,
        &format!("{PATH}?after_id={}", v["last_id"].as_str().unwrap()),
    )
    .await
    .json();
    assert_eq!(
        (&next["data"][0]["id"], &next["has_more"]),
        (&json!(ids[0]), &json!(false))
    );
    rejected(
        &get(&anthropic, &format!("{PATH}?limit=0")).await,
        400,
        "invalid_request_error",
        "limit",
    );
}

#[tokio::test]
async fn create_validation() {
    let anthropic = FakeAnthropic::start().await;
    let cases = [
        (json!({}), "requests: Field required"),
        (json!({ "requests": [] }), "at least 1 item"),
        (json!({ "requests": [], "x": 1 }), "x: Extra inputs"),
        (
            json!({ "requests": [{ "custom_id": "a" }] }),
            "requests.0.params: Field required",
        ),
        (
            batch(&[("bad id!", params("m"))]),
            "requests.0.custom_id: String should match pattern",
        ),
        (
            batch(&[("a", params("m")), ("a", params("m"))]),
            "Duplicate custom_id",
        ),
    ];
    for (body, says) in cases {
        rejected(
            &post(&anthropic, PATH, &body).await,
            400,
            "invalid_request_error",
            says,
        );
    }
    let r = get(&anthropic, &format!("{PATH}/msgbatch_nope")).await;
    rejected(&r, 404, "not_found_error", "msgbatch_nope");
    // Rejections do not create batches.
    assert_eq!(get(&anthropic, PATH).await.json()["data"], json!([]));
}

#[tokio::test]
async fn answer_kinds_and_scripts() {
    let anthropic = FakeAnthropic::start().await;
    let body = batch(&[("a", params("m")), ("b", params("m")), ("c", params("m"))]);
    anthropic.answer(Answer::image(b"png".to_vec()));
    assert_eq!(post(&anthropic, PATH, &body).await.status, 500);
    anthropic.answer(Answer::error(429).retry_after(9));
    let r = post(&anthropic, PATH, &body).await;
    rejected(&r, 429, "rate_limit_error", "rate limit");
    assert_eq!(r.header("retry-after"), Some("9"));
    // Unscripted: "Hi there". Nothing was created by the two above.
    let id = post(&anthropic, PATH, &body).await.json()["id"]
        .as_str()
        .unwrap()
        .to_owned();
    assert_eq!(
        get(&anthropic, PATH).await.json()["data"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    let results = format!("{PATH}/{id}/results");
    let r = get(&anthropic, &results).await;
    assert_eq!(r.header("content-type"), Some("application/binary"));
    assert_eq!(
        lines(&r.body)[0]["result"]["message"]["content"][0]["text"],
        "Hi there"
    );
    // The JSONL streams a line per chunk: cut, paced.
    anthropic.answer(Answer::default().cut_after(2));
    let r = get(&anthropic, &results).await;
    assert!(r.ended.is_err());
    assert_eq!(r.body.lines().count(), 2);
    anthropic.answer(Answer::default().pace(60));
    let t = std::time::Instant::now();
    assert_eq!(lines(&get(&anthropic, &results).await.body).len(), 3);
    assert!(t.elapsed().as_millis() >= 120);
    // Raw events are the lines.
    anthropic.answer(Answer::events(vec![
        json!({ "custom_id": "a", "result": { "type": "expired" } }),
    ]));
    assert_eq!(
        lines(&get(&anthropic, &results).await.body)[0]["result"]["type"],
        "expired"
    );
}
