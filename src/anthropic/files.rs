//! Anthropic Files: `POST /v1/files` (multipart), `GET /v1/files`,
//! `GET` and `DELETE /v1/files/{file_id}`, `GET /v1/files/{file_id}/content`.
//! https://platform.claude.com/docs/en/api/files/upload and the sibling
//! pages (`list`, `delete`); the guide:
//! https://platform.claude.com/docs/en/build-with-claude/files
//!
//! Without a beta header the GA shapes; with `anthropic-beta:
//! files-api-2025-04-14` the beta ones: "requests that still send it keep
//! working and keep returning the beta response shapes" (guide, "Migrate
//! from `files-api-2025-04-14`").

use std::sync::{Arc, Mutex, PoisonError};

use serde_json::{Value, json};

use super::{Anthropic, bad, limit, page, rfc3339, unscripted};
use crate::util::{next_id, unix_now};
use crate::{Answer, Ctx, Endpoint, Out, Provider, Rejection, Request};

const BETA: &str = "files-api-2025-04-14";

#[derive(Clone, Copy, PartialEq)]
enum Op {
    Upload,
    List,
    Retrieve,
    Delete,
    Download,
}

struct File {
    id: String,
    filename: String,
    mime: String,
    size: usize,
    created: u64,
    expires: Option<u64>,
}

type Store = Arc<Mutex<Vec<File>>>;

struct Files {
    op: Op,
    /// Newest first ("Files are ordered newest first", guide).
    store: Store,
}

pub(crate) fn endpoints() -> Vec<Box<dyn Endpoint<Anthropic>>> {
    let store = Store::default();
    let ops = [Op::Upload, Op::List, Op::Retrieve, Op::Delete, Op::Download];
    let each = |op| {
        Box::new(Files {
            op,
            store: store.clone(),
        }) as Box<dyn Endpoint<Anthropic>>
    };
    ops.into_iter().map(each).collect()
}

fn beta(req: &Request) -> bool {
    req.header("anthropic-beta")
        .is_some_and(|b| b.contains(BETA))
}

/// `FileMetadata` (upload); `expires_at` is "Always present; `null` when
/// the file has no expiration" without the beta header, "Not returned"
/// with it (guide).
fn object(f: &File, beta: bool) -> Value {
    let mut v = json!({
        "id": f.id, "type": "file", "filename": f.filename, "mime_type": f.mime,
        "size_bytes": f.size, "created_at": rfc3339(f.created),
        // "`downloadable` is `false` for files you upload" (guide).
        "downloadable": false,
    });
    if !beta {
        v["expires_at"] = json!(f.expires.map(rfc3339));
    }
    v
}

/// The query's `ids[]` values (reqwest encodes the brackets).
fn ids(req: &Request) -> Vec<&str> {
    let pairs = req.query.split('&').filter_map(|p| p.split_once('='));
    let ids = pairs.filter(|(k, _)| matches!(*k, "ids[]" | "ids%5B%5D" | "ids"));
    ids.map(|(_, v)| v).collect()
}

/// The MIME type of an upload: the part's `Content-Type`, else detected
/// ("the type is detected when omitted", guide), here by extension.
fn mime(name: &str, given: Option<&str>) -> String {
    if let Some(m) = given {
        return m.to_owned();
    }
    let ext = name.rsplit_once('.').map(|(_, e)| e.to_ascii_lowercase());
    let m = match ext.as_deref() {
        Some("pdf") => "application/pdf",
        Some("txt" | "md" | "csv") => "text/plain",
        Some("png") => "image/png",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("webp") => "image/webp",
        _ => "application/octet-stream",
    };
    m.to_owned()
}

fn not_found(id: &str) -> Rejection {
    // Quoted from the guide's error example.
    Rejection::new(404, format!("File `{id}` not found.")).kind("not_found_error")
}

impl Files {
    fn store(&self) -> std::sync::MutexGuard<'_, Vec<File>> {
        self.store.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn check_upload(anthropic: &Anthropic, req: &Request) -> Result<(), Rejection> {
        // Wording commonly seen.
        let Some(file) = req.files.iter().find(|f| f.field == "file") else {
            return Err(bad("file: Field required"));
        };
        // "File too large (413): File exceeds the 500 MB limit" (guide).
        if file.bytes.len() > 500 * 1024 * 1024 {
            return Err(anthropic.scripted(413));
        }
        // "`Content-Type` on the uploaded file part: Required" with the
        // beta header (guide). Wording not documented.
        if beta(req) && file.content_type.is_none() {
            return Err(bad("file: the file part must have a Content-Type header"));
        }
        // "Must be between 3600 (one hour) and 7776000 (ninety days)"
        // (upload). Wording commonly seen.
        if let Some(raw) = req.body.get("expires_in_seconds").and_then(Value::as_str) {
            match raw.parse::<i64>() {
                Ok(s) if s < 3600 => {
                    return Err(bad(
                        "expires_in_seconds: Input should be greater than or equal to 3600",
                    ));
                }
                Ok(s) if s > 7_776_000 => {
                    return Err(bad(
                        "expires_in_seconds: Input should be less than or equal to 7776000",
                    ));
                }
                Ok(_) => {}
                Err(_) => return Err(bad("expires_in_seconds: Input should be a valid integer")),
            }
        }
        // "Invalid filename (400): ... length requirements (1-255
        // characters) or contains forbidden characters" (guide), after
        // keeping "only the final path component" (upload). Wording
        // not documented.
        let name = file.filename.rsplit(['/', '\\']).next().unwrap_or_default();
        let forbidden = |c: char| "<>:\"|?*".contains(c) || (c as u32) < 32;
        if name.chars().count() > 255 || name.chars().any(forbidden) {
            return Err(bad(
                "Invalid filename: it must be 1-255 characters without <, >, :, \", |, ?, *, \\, / or control characters.",
            ));
        }
        Ok(())
    }

    fn check_list(req: &Request) -> Result<(), Rejection> {
        limit(req)?;
        let has = |k| req.query_param(k).is_some();
        if beta(req) {
            return Ok(());
        }
        // "`before_id` and `after_id` return a 400 error" without the
        // beta header (guide). Wording not documented, here and below.
        if let Some(k) = ["before_id", "after_id"].into_iter().find(|k| has(k)) {
            return Err(bad(format!(
                "{k}: not supported; use `page` with the `next_page` cursor"
            )));
        }
        let n = ids(req).len();
        // "`ids[]` cannot be combined with `page` or `limit`"; "At most 100
        // entries" (guide, list).
        if n > 0 && (has("page") || has("limit")) {
            return Err(bad("ids: cannot be combined with `page` or `limit`"));
        }
        if n > 100 {
            return Err(bad(format!("ids: at most 100 ids, got {n}")));
        }
        Ok(())
    }

    fn upload(req: &Request) -> File {
        let part = req.files.iter().find(|f| f.field == "file");
        let part = part.cloned().unwrap_or_default();
        let name = part.filename.rsplit(['/', '\\']).next().unwrap_or_default();
        let mime = mime(name, part.content_type.as_deref());
        // "an absent or empty `filename` is replaced with `unnamed` plus
        // the extension for the file's stored `mime_type`" (upload).
        let filename = if name.is_empty() {
            let ext = match mime.as_str() {
                "application/pdf" => ".pdf",
                "text/plain" => ".txt",
                "image/png" => ".png",
                "image/jpeg" => ".jpg",
                "image/gif" => ".gif",
                "image/webp" => ".webp",
                _ => "",
            };
            format!("unnamed{ext}")
        } else {
            name.to_owned()
        };
        let created = unix_now();
        let secs = req.body.get("expires_in_seconds").and_then(Value::as_str);
        let expires = secs
            .and_then(|s| s.parse::<u64>().ok())
            .map(|s| created + s);
        let id = format!("file_fake{}", next_id());
        File {
            id,
            filename,
            mime,
            size: part.bytes.len(),
            created,
            expires,
        }
    }

    /// The GA list: `{data, next_page}`, cursors `page` (`page_<offset>`)
    /// or `ids[]` (list).
    fn list(files: &[File], req: &Request) -> Value {
        let wanted = ids(req);
        if !wanted.is_empty() {
            // "any ID that does not resolve ... is silently omitted" (guide).
            let data: Vec<Value> = files
                .iter()
                .filter(|f| wanted.contains(&f.id.as_str()))
                .map(|f| object(f, false))
                .collect();
            return json!({ "data": data, "next_page": null });
        }
        let limit = limit(req).unwrap_or(20);
        let start = req
            .query_param("page")
            .and_then(|p| p.strip_prefix("page_")?.parse::<usize>().ok())
            .unwrap_or(0)
            .min(files.len());
        let end = (start + limit).min(files.len());
        let data: Vec<Value> = files[start..end].iter().map(|f| object(f, false)).collect();
        let next = (end < files.len()).then(|| format!("page_{end}"));
        json!({ "data": data, "next_page": next })
    }
}

impl Endpoint<Anthropic> for Files {
    fn route(&self) -> (&'static str, &'static str) {
        match self.op {
            Op::Upload => ("POST", "/v1/files"),
            Op::List => ("GET", "/v1/files"),
            Op::Retrieve => ("GET", "/v1/files/{file_id}"),
            Op::Delete => ("DELETE", "/v1/files/{file_id}"),
            Op::Download => ("GET", "/v1/files/{file_id}/content"),
        }
    }

    fn validate(&self, anthropic: &Anthropic, req: &Request) -> Result<(), Rejection> {
        match self.op {
            Op::Upload => return Self::check_upload(anthropic, req),
            Op::List => return Self::check_list(req),
            _ => {}
        }
        let id = req.param.as_deref().unwrap_or_default();
        if !self.store().iter().any(|f| f.id == id) {
            return Err(not_found(id));
        }
        if self.op == Op::Download {
            // "Downloading a file you uploaded returns a 400 error" (guide);
            // every file here is uploaded. Wording not documented.
            return Err(bad(format!("File `{id}` is not downloadable.")));
        }
        Ok(())
    }

    fn respond(&self, _: &Anthropic, req: &Request, answer: &Answer, _: &Ctx) -> Out {
        if let Some(misfit) = unscripted("files", answer) {
            return misfit;
        }
        let beta = beta(req);
        let mut store = self.store();
        let id = req.param.as_deref().unwrap_or_default();
        let body = match self.op {
            Op::Upload => {
                let file = Self::upload(req);
                let v = object(&file, beta);
                store.insert(0, file);
                v
            }
            // The beta list: `{ data, has_more, first_id, last_id }` with
            // `before_id` and `after_id` (guide).
            Op::List if beta => page(
                &store.iter().map(|f| object(f, true)).collect::<Vec<_>>(),
                req,
            ),
            Op::List => Self::list(&store, req),
            Op::Retrieve => store
                .iter()
                .find(|f| f.id == id)
                .map_or(Value::Null, |f| object(f, beta)),
            // `DeletedFile` (delete).
            Op::Delete => {
                store.retain(|f| f.id != id);
                json!({ "id": id, "type": "file_deleted" })
            }
            Op::Download => Value::Null,
        };
        Out::Json(200, body)
    }
}
