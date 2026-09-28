//! OpenAI Batch: `POST /v1/batches`, `GET /v1/batches`,
//! `GET /v1/batches/{batch_id}` and `POST /v1/batches/{batch_id}/cancel`.
//! https://developers.openai.com/api/reference/resources/batches
//! Guide: https://developers.openai.com/api/docs/guides/batch
//!
//! The input file comes from the mock's files. A batch moves one status
//! per retrieve, so a client's polling loop runs: `validating`, then
//! `failed` (input errors) or `in_progress`, `finalizing`, `completed`
//! with its output and error files; a cancel goes `cancelling`, then
//! `cancelled`. Each input line runs through the mock's own endpoint for
//! its `url`, rules included, answering the answer queued for the create.
//! Where the docs give no error wording, the wording is the one commonly
//! seen from the API, marked "(commonly seen)".

use std::collections::HashSet;
use std::sync::{Arc, Mutex, MutexGuard};

use serde_json::{Value, json};

use super::files::{EXPIRY, Store, THIRTY_DAYS, invalid, limit, missing, no_such_file, page};
use super::{OpenAi, bad, object};
use crate::util::{next_id, unix_now};
use crate::{Answer, Ctx, Endpoint, Out, Provider, Rejection, Request};

/// The body parameters of the create reference; any other is rejected.
/// https://developers.openai.com/api/reference/resources/batches/methods/create
#[rustfmt::skip]
const KNOWN: &[&str] = &[
    "completion_window", "endpoint", "input_file_id", "metadata", "output_expires_after",
];

/// "Currently /v1/responses, /v1/chat/completions, /v1/embeddings,
/// /v1/completions, /v1/moderations, /v1/images/generations,
/// /v1/images/edits, and /v1/videos are supported" (create reference).
#[rustfmt::skip]
const ENDPOINTS: &[&str] = &[
    "/v1/responses", "/v1/chat/completions", "/v1/embeddings", "/v1/completions",
    "/v1/moderations", "/v1/images/generations", "/v1/images/edits", "/v1/videos",
];

/// "A single batch may include up to 50,000 requests" (guide, rate limits).
const MAX_REQUESTS: usize = 50_000;

/// A batch: its `Batch` object, the answer its requests take, and its
/// results once run.
#[derive(Debug)]
struct Batch {
    object: Value,
    answer: Answer,
    /// The output and error file lines, once in progress.
    results: Option<(Vec<Value>, Vec<Value>)>,
    /// Seconds the output files live: `output_expires_after`, else 30 days
    /// ("The output file will automatically be deleted 30 days after").
    expires: u64,
}

pub(crate) struct Batches {
    files: Store,
    batches: Arc<Mutex<Vec<Batch>>>,
}

impl Batches {
    pub(crate) fn new(files: Store) -> Self {
        let batches = Arc::default();
        Self { files, batches }
    }

    fn lock(&self) -> MutexGuard<'_, Vec<Batch>> {
        self.batches.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The batch's status, if it exists.
    fn status(&self, id: &str) -> Option<String> {
        let batches = self.lock();
        let b = batches.iter().find(|b| b.object["id"] == id)?;
        b.object["status"].as_str().map(str::to_owned)
    }
}

/// "No batch found with id '{id}'." (commonly seen), a 404.
fn no_such_batch(id: &str) -> Rejection {
    Rejection::new(404, format!("No batch found with id '{id}'.")).param("batch_id")
}

enum Op<'a> {
    Create,
    List,
    Retrieve(&'a str),
    Cancel(&'a str),
}

fn op(req: &Request) -> Op<'_> {
    match (req.method.as_str(), req.param.as_deref()) {
        ("POST", Some(id)) => Op::Cancel(id),
        ("POST", None) => Op::Create,
        (_, Some(id)) => Op::Retrieve(id),
        _ => Op::List,
    }
}

impl Endpoint<OpenAi> for Batches {
    fn route(&self) -> (&'static str, &'static str) {
        ("POST", "/v1/batches")
    }

    fn routes(&self) -> Vec<(&'static str, &'static str)> {
        vec![
            ("POST", "/v1/batches"),
            ("GET", "/v1/batches"),
            ("GET", "/v1/batches/{batch_id}"),
            ("POST", "/v1/batches/{batch_id}/cancel"),
        ]
    }

    fn validate(&self, _: &OpenAi, req: &Request) -> Result<(), Rejection> {
        match op(req) {
            Op::Create => self.create(req),
            // "Limit can range between 1 and 100, and the default is 20"
            // (list reference).
            Op::List => limit(req, 100, 20).map(|_| ()),
            Op::Retrieve(id) => self.status(id).map(|_| ()).ok_or_else(|| no_such_batch(id)),
            // "Cancels an in-progress batch" (cancel reference): one not yet
            // done. The 409 and its wording are commonly seen.
            Op::Cancel(id) => match self.status(id) {
                None => Err(no_such_batch(id)),
                Some(s) if ["validating", "in_progress", "finalizing"].contains(&s.as_str()) => {
                    Ok(())
                }
                Some(s) => {
                    let message = format!("Cannot cancel a batch with status '{s}'.");
                    Err(Rejection::new(409, message))
                }
            },
        }
    }

    fn respond(&self, openai: &OpenAi, req: &Request, answer: &Answer, ctx: &Ctx) -> Out {
        let now = unix_now();
        match op(req) {
            Op::Create => {
                let body = &req.body;
                let expires = body["output_expires_after"]["seconds"].as_u64();
                // The create reference's example `Batch`; a batch "expires"
                // at the end of its 24h window.
                let object = json!({
                    "id": format!("batch_fake{}", ctx.id), "object": "batch",
                    "endpoint": body["endpoint"], "errors": null,
                    "input_file_id": body["input_file_id"],
                    "completion_window": body["completion_window"], "status": "validating",
                    "output_file_id": null, "error_file_id": null, "created_at": now,
                    "in_progress_at": null, "expires_at": now + 86_400, "finalizing_at": null,
                    "completed_at": null, "failed_at": null, "expired_at": null,
                    "cancelling_at": null, "cancelled_at": null,
                    "request_counts": { "total": 0, "completed": 0, "failed": 0 },
                    "metadata": body.get("metadata").cloned().unwrap_or(Value::Null)
                });
                self.lock().push(Batch {
                    object: object.clone(),
                    answer: answer.clone(),
                    results: None,
                    expires: expires.unwrap_or(THIRTY_DAYS),
                });
                Out::Json(200, object)
            }
            Op::List => {
                // Newest first (commonly seen).
                let all: Vec<Value> = self.lock().iter().rev().map(|b| b.object.clone()).collect();
                let limit = limit(req, 100, 20).unwrap_or(20);
                Out::Json(200, page(all, req.query_param("after"), limit))
            }
            Op::Retrieve(id) | Op::Cancel(id) => {
                let cancel = matches!(op(req), Op::Cancel(_));
                let mut batches = self.lock();
                let Some(b) = batches.iter_mut().find(|b| b.object["id"] == id) else {
                    return Out::Misfit("mockllm: validated batch is gone".into());
                };
                if cancel {
                    // "The batch will be in status cancelling for up to 10
                    // minutes, before changing to cancelled" (cancel reference).
                    b.object["status"] = "cancelling".into();
                    b.object["cancelling_at"] = now.into();
                } else {
                    self.advance(b, openai, ctx);
                }
                Out::Json(200, b.object.clone())
            }
        }
    }
}

impl Batches {
    /// The create reference's rules.
    fn create(&self, req: &Request) -> Result<(), Rejection> {
        let body = object(req, KNOWN)?;
        let get = |p: &str| body.get(p).filter(|v| !v.is_null());
        for p in ["input_file_id", "endpoint", "completion_window"] {
            if get(p).is_none() {
                return Err(missing(p));
            }
        }
        // "Currently only 24h is supported."
        let window = get("completion_window")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if window != "24h" {
            return Err(invalid("completion_window", window, &["24h"]));
        }
        let endpoint = get("endpoint").and_then(Value::as_str).unwrap_or_default();
        if !ENDPOINTS.contains(&endpoint) {
            return Err(invalid("endpoint", endpoint, ENDPOINTS));
        }
        // "must be uploaded with the purpose batch"; the wording is commonly
        // seen.
        let id = get("input_file_id")
            .and_then(Value::as_str)
            .unwrap_or_default();
        match self.files.get(id) {
            None => return Err(no_such_file(id, "input_file_id")),
            Some(f) if f.purpose != "batch" => {
                let message = format!(
                    "Invalid 'input_file_id': '{id}'. The file must be uploaded with purpose 'batch'."
                );
                return Err(bad(message, Some("input_file_id"), Some("invalid_value")));
            }
            Some(_) => {}
        }
        // "Set of 16 key-value pairs ... Keys are strings with a maximum
        // length of 64 characters. Values are strings with a maximum length
        // of 512 characters." The wording is commonly seen.
        if let Some(metadata) = get("metadata") {
            let ok = metadata.as_object().is_some_and(|m| {
                m.len() <= 16
                    && m.iter().all(|(k, v)| {
                        k.chars().count() <= 64
                            && v.as_str().is_some_and(|v| v.chars().count() <= 512)
                    })
            });
            if !ok {
                let message = "Invalid 'metadata': expected an object of at most 16 string values, keys of at most 64 characters and values of at most 512.";
                return Err(bad(message, Some("metadata"), Some("invalid_value")));
            }
        }
        // `output_expires_after`: anchor `created_at`, seconds 3600..=2592000.
        if let Some(after) = get("output_expires_after") {
            let anchor = after["anchor"].as_str().unwrap_or_default();
            if anchor != "created_at" {
                return Err(invalid(
                    "output_expires_after.anchor",
                    anchor,
                    &["created_at"],
                ));
            }
            if !after["seconds"]
                .as_u64()
                .is_some_and(|s| EXPIRY.contains(&s))
            {
                let message = format!(
                    "Invalid 'output_expires_after.seconds': {}. Expected a value between 3600 and 2592000.",
                    after["seconds"]
                );
                let p = Some("output_expires_after.seconds");
                return Err(bad(message, p, Some("invalid_value")));
            }
        }
        Ok(())
    }

    /// One step of the batch's life, as a retrieve sees it.
    fn advance(&self, b: &mut Batch, openai: &OpenAi, ctx: &Ctx) {
        let now = unix_now();
        let o = &mut b.object;
        match o["status"].as_str().unwrap_or_default() {
            "validating" => {
                let input = self
                    .files
                    .get(o["input_file_id"].as_str().unwrap_or_default());
                let input = input.map(|f| String::from_utf8_lossy(&f.bytes).into_owned());
                let lines: Vec<&str> = input.iter().flat_map(|s| s.lines()).collect();
                let endpoint = o["endpoint"].as_str().unwrap_or_default();
                match check(&lines, endpoint) {
                    Err(errors) => {
                        // "failed: the input file has failed the validation
                        // process" (guide), with `errors` a list of `BatchError`.
                        o["status"] = "failed".into();
                        o["failed_at"] = now.into();
                        o["errors"] = json!({ "object": "list", "data": errors });
                    }
                    Ok(requests) => {
                        o["status"] = "in_progress".into();
                        o["in_progress_at"] = now.into();
                        o["request_counts"]["total"] = requests.len().into();
                        o["model"] = requests[0]["body"]["model"].clone();
                        let results = run(openai, &requests, &b.answer, ctx);
                        b.results = Some(results);
                    }
                }
            }
            "in_progress" => {
                let (done, failed) = b
                    .results
                    .as_ref()
                    .map_or((0, 0), |(d, f)| (d.len(), f.len()));
                o["request_counts"]["completed"] = done.into();
                o["request_counts"]["failed"] = failed.into();
                o["status"] = "finalizing".into();
                o["finalizing_at"] = now.into();
            }
            "finalizing" => {
                o["status"] = "completed".into();
                o["completed_at"] = now.into();
                self.write(b);
            }
            // "cancelled, where it will have partial results (if any)
            // available in the output file" (cancel reference): those of a
            // batch cancelled while finalizing.
            "cancelling" => {
                let finalizing = !o["finalizing_at"].is_null();
                o["status"] = "cancelled".into();
                o["cancelled_at"] = now.into();
                if finalizing {
                    self.write(b);
                }
            }
            _ => {}
        }
    }

    /// Writes the output and error files, and the batch's usage.
    fn write(&self, b: &mut Batch) {
        let Some((done, failed)) = &b.results else {
            return;
        };
        let id = b.object["id"].as_str().unwrap_or_default().to_owned();
        let file = |lines: &[Value], kind: &str| {
            if lines.is_empty() {
                return Value::Null;
            }
            let text: String = lines.iter().map(|l| format!("{l}\n")).collect();
            // The purpose of the `FileObject` reference; the file name is
            // commonly seen.
            let name = format!("{id}_{kind}.jsonl");
            let purpose = "batch_output";
            self.files
                .add(&name, purpose, text.into(), Some(b.expires))
                .into()
        };
        b.object["output_file_id"] = file(done, "output");
        b.object["error_file_id"] = file(failed, "error");
        // `BatchUsage`: the tokens of the requests that ran, from the usage
        // each endpoint reports.
        let sum = |keys: [&str; 2]| -> u64 {
            let usage = done.iter().map(|l| &l["response"]["body"]["usage"]);
            usage
                .map(|u| keys.iter().find_map(|k| u[k].as_u64()).unwrap_or(0))
                .sum()
        };
        let input = sum(["prompt_tokens", "input_tokens"]);
        let output = sum(["completion_tokens", "output_tokens"]);
        b.object["usage"] = json!({
            "input_tokens": input, "input_tokens_details": { "cached_tokens": 0 },
            "output_tokens": output, "output_tokens_details": { "reasoning_tokens": 0 },
            "total_tokens": input + output
        });
    }
}

/// A `BatchError` of the input file.
fn error(code: &str, line: Option<usize>, message: &str) -> Value {
    json!({ "code": code, "line": line, "message": message, "param": null })
}

/// The input lines' rules, checked while `validating`: the requests, or
/// every error.
fn check(lines: &[&str], endpoint: &str) -> Result<Vec<Value>, Vec<Value>> {
    let lines: Vec<(usize, &str)> = lines
        .iter()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(i, l)| (i + 1, *l))
        .collect();
    // Commonly seen code and wording.
    if lines.is_empty() {
        return Err(vec![error(
            "empty_file",
            None,
            "The batch input file is empty.",
        )]);
    }
    if lines.len() > MAX_REQUESTS {
        let message = format!(
            "The batch input file contains {} requests, more than the maximum of {MAX_REQUESTS}.",
            lines.len()
        );
        return Err(vec![error("too_many_tasks", None, &message)]);
    }
    let (mut requests, mut errors, mut ids) = (Vec::new(), Vec::new(), HashSet::new());
    let mut model: Option<Value> = None;
    for (n, line) in lines {
        // The code and wording seen from the API:
        // https://community.openai.com/t/error-after-creating-batch-job-this-line-is-not-parseable-as-valid-json/728379
        let Ok(r @ Value::Object(_)) = serde_json::from_str::<Value>(line) else {
            errors.push(error(
                "invalid_json_line",
                Some(n),
                "This line is not parseable as valid JSON.",
            ));
            continue;
        };
        // "Each request must include a unique custom_id value" (guide).
        // Commonly seen code and wording.
        if !ids.insert(r["custom_id"].to_string()) {
            let message = format!(
                "The custom_id '{}' is duplicated.",
                r["custom_id"].as_str().unwrap_or_default()
            );
            errors.push(error("duplicate_custom_id", Some(n), &message));
        }
        // "The endpoint to be used for all requests in the batch" (create
        // reference). Commonly seen code and wording.
        if r["url"] != endpoint {
            let message = format!(
                "The URL provided for this request does not match the batch endpoint {endpoint}."
            );
            errors.push(error("url_mismatch", Some(n), &message));
        }
        // "each input file can only include requests to a single model"
        // (guide). Commonly seen code and wording.
        let m = &r["body"]["model"];
        match &model {
            None => model = Some(m.clone()),
            Some(first) if first != m => {
                let message = "The provided model does not match the model of the other requests in the batch.";
                errors.push(error("model_mismatch", Some(n), message));
            }
            Some(_) => {}
        }
        requests.push(r);
    }
    if errors.is_empty() {
        Ok(requests)
    } else {
        Err(errors)
    }
}

/// Runs each request through the mock's endpoint for its `url`, taking
/// `answer`: the output lines (status 200) and the error lines. The line
/// shapes are the guide's ("Retrieve the results"); a failed request's
/// line carries its status and the endpoint's error body (commonly seen).
fn run(
    openai: &OpenAi,
    requests: &[Value],
    answer: &Answer,
    ctx: &Ctx,
) -> (Vec<Value>, Vec<Value>) {
    let endpoints = OpenAi::endpoints();
    let (mut done, mut failed) = (Vec::new(), Vec::new());
    for r in requests {
        let id = next_id();
        let mut c = ctx.clone();
        c.id = id;
        let url = r["url"].as_str().unwrap_or_default();
        let req = Request {
            method: "POST".into(),
            path: url.into(),
            body: r["body"].clone(),
            ..Request::default()
        };
        let endpoint = endpoints
            .iter()
            .find(|e| e.routes().iter().any(|&(m, p)| m == "POST" && p == url));
        let out = match endpoint {
            None => Err(format!("mockllm: batch requests to {url} are not modelled")),
            Some(e) => match e.validate(openai, &req) {
                Err(rejection) => Ok((rejection.status, openai.error(&rejection, id))),
                Ok(()) => match e.respond(openai, &req, answer, &c) {
                    Out::Json(status, body) => Ok((status, body)),
                    Out::Misfit(why) => Err(why),
                    _ => Err(format!(
                        "mockllm: a batch request to {url} must answer JSON"
                    )),
                },
            },
        };
        let request_id = openai
            .headers(id)
            .into_iter()
            .find(|(h, _)| *h == "x-request-id");
        let mut line = json!({ "id": format!("batch_req_fake{id}"), "custom_id": r["custom_id"], "response": null, "error": null });
        match out {
            Ok((status, body)) => {
                line["response"] = json!({ "status_code": status, "request_id": request_id.map(|(_, v)| v), "body": body });
                if status == 200 {
                    done.push(line)
                } else {
                    failed.push(line)
                }
            }
            Err(why) => {
                line["error"] = json!({ "code": "mockllm", "message": why });
                failed.push(line);
            }
        }
    }
    (done, failed)
}
