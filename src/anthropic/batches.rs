//! Anthropic Message Batches: `/v1/messages/batches` create and list,
//! `/{id}` retrieve and delete, `/{id}/cancel`, `/{id}/results` (JSONL).
//! https://platform.claude.com/docs/en/api/messages/batches/create and
//! the sibling pages (`list`, `retrieve`, `cancel`, `delete`, `results`);
//! the guide: https://platform.claude.com/docs/en/build-with-claude/batch-processing
//!
//! A batch is created `in_progress` and ends at the first retrieve or
//! list after that (a cancel first makes it `canceling`, and every
//! request `canceled`). Each request's `params` are checked by the
//! Messages rules when the batch is created, and a failing one is an
//! `errored` result, not a rejection: "Validation of the `params` object
//! for each message request is performed asynchronously" (guide). The
//! answer queued for the create is every succeeded request's message.

use std::sync::{Arc, Mutex, PoisonError};

use bytes::Bytes;
use serde_json::{Map, Value, json};

use super::{Anthropic, bad, limit, messages, page, rfc3339, unscripted};
use crate::core::misfit;
use crate::util::{next_id, unix_now};
use crate::{Answer, Ctx, Endpoint, Out, Provider, Rejection, Reply, Request};

#[derive(Clone, Copy, PartialEq)]
enum Op {
    Create,
    List,
    Retrieve,
    Delete,
    Cancel,
    Results,
}

struct Batch {
    id: String,
    created: u64,
    canceled: Option<u64>,
    ended: Option<u64>,
    results: Vec<Value>,
    url: String,
}

type Store = Arc<Mutex<Vec<Batch>>>;

struct Batches {
    op: Op,
    /// Newest first, shared by the six routes of one mock.
    store: Store,
}

pub(crate) fn endpoints() -> Vec<Box<dyn Endpoint<Anthropic>>> {
    let store = Store::default();
    let ops = [
        Op::Create,
        Op::List,
        Op::Retrieve,
        Op::Delete,
        Op::Cancel,
        Op::Results,
    ];
    let each = |op| {
        Box::new(Batches {
            op,
            store: store.clone(),
        }) as Box<dyn Endpoint<Anthropic>>
    };
    ops.into_iter().map(each).collect()
}

/// "Processing ends when every request ... has either succeeded, errored,
/// canceled, or expired" (list). A canceled batch's requests are all
/// canceled here.
fn end(b: &mut Batch) {
    if b.ended.is_none() {
        b.ended = Some(unix_now());
        if b.canceled.is_some() {
            for r in &mut b.results {
                r["result"] = json!({ "type": "canceled" });
            }
        }
    }
}

/// The `MessageBatch` object (list, cancel).
fn object(b: &Batch) -> Value {
    let count = |t: &str| {
        b.results
            .iter()
            .filter(|r| r["result"]["type"] == t)
            .count()
    };
    let ended = b.ended.is_some();
    // "This is zero until processing of the entire Message Batch has ended."
    let done = |t: &str| if ended { count(t) } else { 0 };
    let status = match (ended, b.canceled) {
        (true, _) => "ended",
        (false, Some(_)) => "canceling",
        (false, None) => "in_progress",
    };
    json!({
        "id": b.id, "type": "message_batch",
        "processing_status": status,
        "request_counts": {
            "processing": if ended { 0 } else { b.results.len() },
            "succeeded": done("succeeded"), "errored": done("errored"),
            "canceled": done("canceled"), "expired": done("expired"),
        },
        "ended_at": b.ended.map(rfc3339),
        "created_at": rfc3339(b.created),
        // "24 hours after creation".
        "expires_at": rfc3339(b.created + 86_400),
        "archived_at": null,
        "cancel_initiated_at": b.canceled.map(rfc3339),
        "results_url": ended.then_some(&b.url),
    })
}

/// The top-level shape, checked when the batch is created (create).
fn check_requests(anthropic: &Anthropic, req: &Request) -> Result<(), Rejection> {
    // "limited to either 100,000 Message requests or 256 MB" (guide);
    // "413 `request_too_large`" (guide, troubleshooting).
    if req.raw.len() > 256 * 1024 * 1024 {
        return Err(anthropic.scripted(413));
    }
    let Some(body) = req.object() else {
        return Err(bad("There was an error parsing the body of your request."));
    };
    if let Some(k) = body.keys().find(|k| *k != "requests") {
        return Err(bad(format!("{k}: Extra inputs are not permitted")));
    }
    let Some(requests) = body.get("requests") else {
        return Err(bad("requests: Field required"));
    };
    let Some(requests) = requests.as_array() else {
        return Err(bad("requests: Input should be a valid list"));
    };
    // "minItems: 1, maxItems: 100000" (create). Wording commonly seen.
    let n = requests.len();
    if n == 0 {
        return Err(bad(
            "requests: List should have at least 1 item after validation, not 0",
        ));
    }
    if n > 100_000 {
        return Err(bad(format!(
            "requests: List should have at most 100000 items after validation, not {n}"
        )));
    }
    let mut seen = Vec::new();
    for (i, r) in requests.iter().enumerate() {
        let Some(r) = r.as_object() else {
            return Err(bad(format!(
                "requests.{i}: Input should be a valid dictionary"
            )));
        };
        if let Some(k) = r
            .keys()
            .find(|k| !["custom_id", "params"].contains(&k.as_str()))
        {
            return Err(bad(format!(
                "requests.{i}.{k}: Extra inputs are not permitted"
            )));
        }
        for field in ["custom_id", "params"] {
            if !r.contains_key(field) {
                return Err(bad(format!("requests.{i}.{field}: Field required")));
            }
        }
        // "pattern: ^[a-zA-Z0-9_-]{1,64}$" (create).
        let id = r["custom_id"].as_str().unwrap_or_default();
        let ok = (1..=64).contains(&id.len())
            && id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
        if !ok {
            return Err(bad(format!(
                "requests.{i}.custom_id: String should match pattern '^[a-zA-Z0-9_-]{{1,64}}$'"
            )));
        }
        // "Must be unique for each request within the Message Batch"
        // (create). Wording not documented.
        if seen.contains(&id) {
            return Err(bad(format!(
                "requests.{i}.custom_id: Duplicate custom_id '{id}'. Each custom_id must be unique within the Message Batch."
            )));
        }
        seen.push(id);
        if !r["params"].is_object() {
            return Err(bad(format!(
                "requests.{i}.params: Input should be a valid dictionary"
            )));
        }
    }
    Ok(())
}

/// One request's result: its message, or `errored` for params the
/// Messages endpoint or the batch rules reject.
fn result(
    anthropic: &Anthropic,
    req: &Request,
    params: &Map<String, Value>,
    answer: &Answer,
) -> Result<Value, Out> {
    // "not supported in batch requests. Including any of these returns a
    // validation error": `stream: true`, `speed`, `max_tokens: 0` (guide).
    // Wording not documented.
    let batch_rule = if params.get("stream") == Some(&Value::Bool(true)) {
        Some("stream: Streaming is not supported in Message Batches.")
    } else if params.contains_key("speed") {
        Some("speed: Fast mode is not supported in Message Batches.")
    } else if params.get("max_tokens").and_then(Value::as_u64) == Some(0) {
        Some("max_tokens: Input should be greater than or equal to 1")
    } else {
        None
    };
    let one = Request {
        body: Value::Object(params.clone()),
        headers: req.headers.clone(),
        ..Request::default()
    };
    let checked = match batch_rule {
        Some(message) => Err(bad(message)),
        None => messages::validate(&one),
    };
    let id = next_id();
    match checked {
        Err(r) => Ok(json!({ "type": "errored", "error": anthropic.error(&r, id) })),
        Ok((body, show)) => match messages::respond(anthropic, body, show, answer, id) {
            Out::Json(_, message) => Ok(json!({ "type": "succeeded", "message": message })),
            other => Err(other),
        },
    }
}

impl Batches {
    fn store(&self) -> std::sync::MutexGuard<'_, Vec<Batch>> {
        self.store.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn create(&self, anthropic: &Anthropic, req: &Request, answer: &Answer) -> Out {
        if !matches!(
            answer.reply(),
            Reply::Unscripted | Reply::Text(_) | Reply::ToolCall { .. }
        ) {
            return misfit("message batches", answer.reply());
        }
        let n = next_id();
        let id = format!("msgbatch_fake{n}");
        let requests = req.body["requests"]
            .as_array()
            .map(Vec::as_slice)
            .unwrap_or_default();
        let mut results = Vec::new();
        for r in requests {
            let params = r["params"].as_object().cloned().unwrap_or_default();
            match result(anthropic, req, &params, answer) {
                Ok(v) => results.push(json!({ "custom_id": r["custom_id"], "result": v })),
                Err(out) => return out,
            }
        }
        let host = req.header("host").unwrap_or("127.0.0.1");
        let url = format!("http://{host}/v1/messages/batches/{id}/results");
        let batch = Batch {
            id,
            created: unix_now(),
            canceled: None,
            ended: None,
            results,
            url,
        };
        let out = object(&batch);
        self.store().insert(0, batch);
        Out::Json(200, out)
    }
}

impl Endpoint<Anthropic> for Batches {
    fn route(&self) -> (&'static str, &'static str) {
        match self.op {
            Op::Create => ("POST", "/v1/messages/batches"),
            Op::List => ("GET", "/v1/messages/batches"),
            Op::Retrieve => ("GET", "/v1/messages/batches/{message_batch_id}"),
            Op::Delete => ("DELETE", "/v1/messages/batches/{message_batch_id}"),
            Op::Cancel => ("POST", "/v1/messages/batches/{message_batch_id}/cancel"),
            Op::Results => ("GET", "/v1/messages/batches/{message_batch_id}/results"),
        }
    }

    fn validate(&self, anthropic: &Anthropic, req: &Request) -> Result<(), Rejection> {
        match self.op {
            Op::Create => return check_requests(anthropic, req),
            Op::List => return limit(req).map(|_| ()),
            _ => {}
        }
        let id = req.param.as_deref().unwrap_or_default();
        let store = self.store();
        let Some(b) = store.iter().find(|b| b.id == id) else {
            // Wording commonly seen.
            let message = format!("Message Batch `{id}` not found.");
            return Err(Rejection::new(404, message).kind("not_found_error"));
        };
        let ended = b.ended.is_some();
        match self.op {
            // "can only be deleted once they've finished processing. If
            // you'd like to delete an in-progress batch, you must first
            // cancel it" (delete). Wording not documented.
            Op::Delete if !ended => Err(bad(format!(
                "Message Batch `{id}` cannot be deleted while it is still processing. Cancel it first."
            ))),
            // "retrieve results when processing has ended for all requests"
            // (guide). Wording not documented.
            Op::Results if !ended => Err(bad(format!(
                "Message Batch `{id}` has not finished processing; its results are not available yet."
            ))),
            _ => Ok(()),
        }
    }

    fn respond(&self, anthropic: &Anthropic, req: &Request, answer: &Answer, _: &Ctx) -> Out {
        if self.op == Op::Create {
            return self.create(anthropic, req, answer);
        }
        if self.op == Op::Results {
            // "Streams the results ... as a `.jsonl` file" (results), a line
            // per chunk so `pace` and `cut_after` apply per result. Raw
            // events are sent as the lines. The content type is the one the
            // SDKs ask for (`Accept: application/binary`); not documented.
            let lines = match answer.reply() {
                Reply::Events(e) => e.clone(),
                Reply::Unscripted => {
                    let id = req.param.as_deref().unwrap_or_default();
                    let store = self.store();
                    let b = store.iter().find(|b| b.id == id);
                    b.map(|b| b.results.clone()).unwrap_or_default()
                }
                other => return misfit("message batch results", other),
            };
            let chunks = lines.iter().map(|l| Bytes::from(format!("{l}\n")));
            return Out::Chunks("application/binary".into(), chunks.collect());
        }
        if let Some(misfit) = unscripted("message batches", answer) {
            return misfit;
        }
        let mut store = self.store();
        if self.op == Op::List {
            store.iter_mut().for_each(end);
            let all: Vec<Value> = store.iter().map(object).collect();
            return Out::Json(200, page(&all, req));
        }
        let id = req.param.as_deref().unwrap_or_default();
        let Some(i) = store.iter().position(|b| b.id == id) else {
            return Out::Misfit(format!("mockllm: batch {id} vanished"));
        };
        let b = &mut store[i];
        match self.op {
            Op::Retrieve => end(b),
            // "Batches may be canceled any time before processing ends"
            // (cancel); an ended batch is returned as it is (not documented).
            Op::Cancel if b.ended.is_none() && b.canceled.is_none() => {
                b.canceled = Some(unix_now())
            }
            Op::Delete => {
                store.remove(i);
                return Out::Json(200, json!({ "id": id, "type": "message_batch_deleted" }));
            }
            _ => {}
        }
        Out::Json(200, object(b))
    }
}
