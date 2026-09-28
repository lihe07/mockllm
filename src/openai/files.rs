//! OpenAI Files: `POST /v1/files` (multipart upload), `GET /v1/files`,
//! `GET /v1/files/{file_id}`, `GET /v1/files/{file_id}/content` and
//! `DELETE /v1/files/{file_id}`, the files kept in the mock (and shared
//! with batches, which read their input and write their output here).
//! https://developers.openai.com/api/reference/resources/files
//!
//! Where the docs give no error wording, the wording is the one commonly
//! seen from the API, marked "(commonly seen)".

use std::sync::{Arc, Mutex, MutexGuard};

use bytes::Bytes;
use serde_json::{Map, Value, json};

use super::{OpenAi, bad, object};
use crate::core::misfit;
use crate::util::{next_id, unix_now};
use crate::{Answer, Ctx, Endpoint, Out, Rejection, Reply, Request};

/// The form fields of the upload reference; any other is rejected.
/// https://developers.openai.com/api/reference/resources/files/methods/create
const KNOWN: &[&str] = &["purpose", "expires_after[anchor]", "expires_after[seconds]"];

/// "One of: assistants, batch, fine-tune, vision, user_data, evals" (the
/// upload reference); `*_output` and `*-results` are the API's own.
const PURPOSES: &[&str] = &[
    "assistants",
    "batch",
    "fine-tune",
    "vision",
    "user_data",
    "evals",
];

/// "Must be between 3600 (1 hour) and 2592000 (30 days)" (the upload
/// reference, `expires_after.seconds`; also batches' `output_expires_after`).
pub(crate) const EXPIRY: std::ops::RangeInclusive<u64> = 3600..=2_592_000;

/// "By default, files with purpose=batch expire after 30 days" (upload
/// reference, `expires_after`).
pub(crate) const THIRTY_DAYS: u64 = 2_592_000;

const MB: usize = 1024 * 1024;

/// A stored file.
#[derive(Clone, Debug)]
pub(crate) struct File {
    pub(crate) id: String,
    pub(crate) purpose: String,
    pub(crate) bytes: Bytes,
    filename: String,
    created_at: u64,
    expires_at: Option<u64>,
}

impl File {
    /// The `FileObject` of the reference; `status` is deprecated but still
    /// sent, `processed` as in its example.
    fn object(&self) -> Value {
        json!({
            "id": self.id, "object": "file", "bytes": self.bytes.len(),
            "created_at": self.created_at, "expires_at": self.expires_at,
            "filename": self.filename, "purpose": self.purpose, "status": "processed"
        })
    }
}

/// The mock's files, shared by the files and batches endpoints.
#[derive(Clone, Debug, Default)]
pub(crate) struct Store(Arc<Mutex<Vec<File>>>);

impl Store {
    fn lock(&self) -> MutexGuard<'_, Vec<File>> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub(crate) fn get(&self, id: &str) -> Option<File> {
        self.lock().iter().find(|f| f.id == id).cloned()
    }

    /// Stores a file, expiring `expires_after` seconds from now if set.
    pub(crate) fn add(
        &self,
        filename: &str,
        purpose: &str,
        bytes: Bytes,
        expires_after: Option<u64>,
    ) -> String {
        let created_at = unix_now();
        let file = File {
            id: format!("file-fake{}", next_id()),
            purpose: purpose.to_owned(),
            bytes,
            filename: filename.to_owned(),
            created_at,
            expires_at: expires_after.map(|s| created_at + s),
        };
        let id = file.id.clone();
        self.lock().push(file);
        id
    }
}

/// "No such File object: {id}" (commonly seen), a 404 naming `id`.
pub(crate) fn no_such_file(id: &str, param: &str) -> Rejection {
    Rejection::new(404, format!("No such File object: {id}")).param(param)
}

/// A missing required parameter, as the other OpenAI endpoints word it.
pub(crate) fn missing(p: &str) -> Rejection {
    let message = format!("Missing required parameter: '{p}'.");
    bad(message, Some(p), Some("missing_required_parameter"))
}

/// `value` not among `supported` (commonly seen wording).
pub(crate) fn invalid(p: &str, value: &str, supported: &[&str]) -> Rejection {
    let list: Vec<String> = supported.iter().map(|s| format!("'{s}'")).collect();
    let message = format!(
        "Invalid value: '{value}'. Supported values are: {}.",
        list.join(", ")
    );
    bad(message, Some(p), Some("invalid_value"))
}

/// A list page by the pagination rules shared by the list endpoints:
/// objects after the `after` id, at most `limit`, with `first_id`,
/// `last_id` and `has_more`.
pub(crate) fn page(all: Vec<Value>, after: Option<&str>, limit: usize) -> Value {
    let start = after
        .and_then(|a| all.iter().position(|o| o["id"] == a))
        .map_or(0, |i| i + 1);
    let rest = &all[start.min(all.len())..];
    let data = &rest[..limit.min(rest.len())];
    json!({
        "object": "list", "data": data,
        "first_id": data.first().map(|o| o["id"].clone()),
        "last_id": data.last().map(|o| o["id"].clone()),
        "has_more": rest.len() > data.len()
    })
}

/// `?limit=` in `1..=max`, else the default; an error naming the range.
pub(crate) fn limit(req: &Request, max: usize, default: usize) -> Result<usize, Rejection> {
    let Some(raw) = req.query_param("limit") else {
        return Ok(default);
    };
    match raw.parse::<usize>() {
        Ok(n) if (1..=max).contains(&n) => Ok(n),
        // Commonly seen wording.
        _ => {
            let message = format!(
                "Invalid 'limit': integer below minimum value or above maximum value. Expected a value between 1 and {max}."
            );
            Err(bad(message, Some("limit"), Some("invalid_value")))
        }
    }
}

pub(crate) struct Files(pub(crate) Store);

/// What a request to one of the routes asks.
enum Op<'a> {
    Upload,
    List,
    Retrieve(&'a str),
    Content(&'a str),
    Delete(&'a str),
}

fn op(req: &Request) -> Op<'_> {
    let id = req.param.as_deref().unwrap_or_default();
    match (req.method.as_str(), req.path.ends_with("/content")) {
        ("POST", _) => Op::Upload,
        ("DELETE", _) => Op::Delete(id),
        (_, true) => Op::Content(id),
        _ if req.param.is_some() => Op::Retrieve(id),
        _ => Op::List,
    }
}

impl Endpoint<OpenAi> for Files {
    fn route(&self) -> (&'static str, &'static str) {
        ("POST", "/v1/files")
    }

    fn routes(&self) -> Vec<(&'static str, &'static str)> {
        vec![
            ("POST", "/v1/files"),
            ("GET", "/v1/files"),
            ("GET", "/v1/files/{file_id}"),
            ("DELETE", "/v1/files/{file_id}"),
            ("GET", "/v1/files/{file_id}/content"),
        ]
    }

    fn validate(&self, _: &OpenAi, req: &Request) -> Result<(), Rejection> {
        match op(req) {
            Op::Upload => upload(req),
            Op::List => {
                // `limit` "can range between 1 and 10,000"; `order` "asc" or
                // "desc" (list reference).
                limit(req, 10_000, 10_000)?;
                match req.query_param("order") {
                    Some(o) if o != "asc" && o != "desc" => {
                        Err(invalid("order", o, &["asc", "desc"]))
                    }
                    _ => Ok(()),
                }
            }
            Op::Retrieve(id) | Op::Content(id) | Op::Delete(id) => match self.0.get(id) {
                Some(_) => Ok(()),
                None => Err(no_such_file(id, "id")),
            },
        }
    }

    fn respond(&self, _: &OpenAi, req: &Request, answer: &Answer, _: &Ctx) -> Out {
        if answer.reply != Reply::Unscripted {
            return misfit("files", &answer.reply);
        }
        let store = &self.0;
        match op(req) {
            Op::Upload => {
                let file = req.files.iter().find(|f| f.field == "file");
                let (Some(file), Some(purpose)) = (file, req.body["purpose"].as_str()) else {
                    return Out::Misfit("mockllm: validated upload has no file".into());
                };
                let seconds = req.body["expires_after[seconds]"].as_str();
                let seconds = seconds.and_then(|s| s.parse().ok());
                let expires = seconds.or((purpose == "batch").then_some(THIRTY_DAYS));
                let id = store.add(&file.filename, purpose, file.bytes.clone(), expires);
                Out::Json(200, store.get(&id).map(|f| f.object()).unwrap_or_default())
            }
            Op::List => {
                let purpose = req.query_param("purpose");
                let mut all: Vec<Value> = store
                    .lock()
                    .iter()
                    .filter(|f| purpose.is_none_or(|p| f.purpose == p))
                    .map(File::object)
                    .collect();
                // Newest first unless `order=asc` (commonly seen default).
                if req.query_param("order") != Some("asc") {
                    all.reverse();
                }
                let limit = limit(req, 10_000, 10_000).unwrap_or(10_000);
                Out::Json(200, page(all, req.query_param("after"), limit))
            }
            Op::Retrieve(id) => {
                Out::Json(200, store.get(id).map(|f| f.object()).unwrap_or_default())
            }
            // "Returns the contents of the specified file", the raw bytes.
            Op::Content(id) => {
                let bytes = store.get(id).map(|f| f.bytes).unwrap_or_default();
                Out::Bytes("application/octet-stream".into(), bytes)
            }
            // The `FileDeleted` object of the delete reference.
            Op::Delete(id) => {
                store.lock().retain(|f| f.id != id);
                Out::Json(200, json!({ "id": id, "object": "file", "deleted": true }))
            }
        }
    }
}

/// The upload reference's rules.
fn upload(req: &Request) -> Result<(), Rejection> {
    let body = match &req.body {
        Value::Null => &Map::new(),
        _ => object(req, KNOWN)?,
    };
    let Some(file) = req.files.iter().find(|f| f.field == "file") else {
        return Err(missing("file"));
    };
    let Some(purpose) = body.get("purpose").and_then(Value::as_str) else {
        return Err(missing("purpose"));
    };
    if !PURPOSES.contains(&purpose) {
        return Err(invalid("purpose", purpose, PURPOSES));
    }
    // `expires_after.anchor`: "Supported anchors: created_at".
    if let Some(a) = body.get("expires_after[anchor]").and_then(Value::as_str)
        && a != "created_at"
    {
        return Err(invalid("expires_after.anchor", a, &["created_at"]));
    }
    if let Some(s) = body.get("expires_after[seconds]").and_then(Value::as_str)
        && !s.parse().is_ok_and(|s| EXPIRY.contains(&s))
    {
        // Commonly seen wording.
        let message = format!(
            "Invalid 'expires_after.seconds': {s}. Expected a value between 3600 and 2592000."
        );
        return Err(bad(
            message,
            Some("expires_after.seconds"),
            Some("invalid_value"),
        ));
    }
    // "Individual files can be up to 512 MB" (upload reference); the
    // wording is commonly seen.
    if file.bytes.len() > 512 * MB {
        return Err(bad(
            "File is too large. Maximum size is 512 MB.",
            Some("file"),
            None,
        ));
    }
    // "The Batch API only supports .jsonl files up to 200 MB in size"; "The
    // Fine-tuning API only supports .jsonl files" (upload reference). The
    // wording is commonly seen.
    let jsonl = file.filename.ends_with(".jsonl");
    match purpose {
        "batch" if !jsonl => Err(bad(
            "Invalid file format for Batch API. Must be .jsonl",
            Some("file"),
            None,
        )),
        "batch" if file.bytes.len() > 200 * MB => Err(bad(
            "File is too large for Batch API. Maximum size is 200 MB.",
            Some("file"),
            None,
        )),
        "fine-tune" if !jsonl => Err(bad(
            "Invalid file format for Fine-Tuning API. Must be .jsonl",
            Some("file"),
            None,
        )),
        _ => Ok(()),
    }
}
