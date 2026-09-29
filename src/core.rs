//! The shared core, written once for every provider: the [`Provider`],
//! [`Endpoint`] and [`Session`] traits, and the server under every
//! [`Mock`]: listener, recording, answer queue, routing, hosted files.
//! Transports (JSON, SSE, binary, WebSocket) and their timing are in
//! `transport.rs`.

use std::collections::VecDeque;
use std::convert::Infallible;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex, MutexGuard};

use axum::extract::FromRequestParts;
use axum::extract::ws::WebSocketUpgrade;
use axum::http::HeaderMap;
use axum::http::request::Parts;
use axum::response::Response;
use bytes::Bytes;
use serde_json::{Map, Value};
use tokio::task::JoinHandle;

use crate::transport::{self, Out};
use crate::util::next_id;
use crate::{Answer, Reply};

/// A provider: a named set of endpoints and the pieces they share.
pub trait Provider: Sized + Send + Sync + 'static {
    /// Appended to the origin by [`Mock::base_url`], as the provider's SDKs
    /// expect: `"/v1"` for OpenAI, `""` for Anthropic.
    const BASE_PATH: &'static str;

    /// Every endpoint the provider serves, one file each.
    fn endpoints() -> Vec<Box<dyn Endpoint<Self>>>;

    /// Checks every endpoint shares, such as the auth header, before the
    /// endpoint's own [`Endpoint::validate`].
    fn authorize(&self, req: &Request) -> Result<(), Rejection>;

    /// The provider's error body for `rejection`; `id` is the request id.
    fn error(&self, rejection: &Rejection, id: u64) -> Value;

    /// The rejection an [`Answer::error`] of `status` sends.
    fn scripted(&self, status: u16) -> Rejection;

    /// No endpoint has the request's method and path.
    fn not_found(&self, _req: &Request) -> Rejection {
        self.scripted(404)
    }

    /// Headers on every response, such as a request id.
    fn headers(&self, _id: u64) -> Vec<(&'static str, String)> {
        Vec::new()
    }
}

/// One endpoint: a route, its request rules, and how it frames an
/// [`Answer`]. HTTP endpoints implement [`respond`](Endpoint::respond),
/// WebSocket ones [`session`](Endpoint::session).
pub trait Endpoint<P>: Send + Sync + 'static {
    /// Method and path. One `{name}` placeholder matches a non-empty run
    /// without `/` or `:`, and is kept in [`Request::param`]:
    /// `("POST", "/v1beta/models/{model}:generateContent")`.
    fn route(&self) -> (&'static str, &'static str);

    /// Every method and path served, for an endpoint over a resource
    /// (`/v1/files`, `/v1/files/{id}`, ...); by default just [`route`](Endpoint::route).
    fn routes(&self) -> Vec<(&'static str, &'static str)> {
        vec![self.route()]
    }

    /// The endpoint's rules, per-model ones included, after
    /// [`Provider::authorize`].
    fn validate(&self, provider: &P, req: &Request) -> Result<(), Rejection>;

    /// The body for a valid request. An [`Answer::error`] never gets here;
    /// the core sends it in the provider's error shape.
    fn respond(&self, _provider: &P, _req: &Request, _answer: &Answer, _ctx: &Ctx) -> Out {
        Out::Misfit("mockllm: this endpoint is WebSocket-only".into())
    }

    /// A WebSocket endpoint's session for a valid upgrade request.
    fn session(&self, _provider: &P, _req: &Request) -> Option<Box<dyn Session>> {
        None
    }
}

/// One WebSocket connection's conversation, JSON events both ways.
pub trait Session: Send + 'static {
    /// Events sent as the connection opens.
    fn open(&mut self, _ctx: &Ctx) -> Vec<Value> {
        Vec::new()
    }

    /// One client event in. `next` takes the next queued [`Answer`], for
    /// events that ask for a response.
    fn event(&mut self, event: Value, next: &mut dyn FnMut() -> Answer) -> Turn;

    /// Asked after each turn is sent: a close code and reason ends the
    /// connection with that close frame, as providers that report errors
    /// by closing do.
    fn close(&mut self) -> Option<(u16, String)> {
        None
    }
}

/// What a [`Session`] sends back for one client event.
#[derive(Debug, Default)]
pub struct Turn {
    pub events: Vec<Value>,
    /// The answer used, whose timing and breaks apply; `None` sends at once.
    pub answer: Option<Answer>,
    /// The mid-stream error event `error_after` sends.
    pub error: Value,
}

impl Turn {
    /// `events`, sent at once.
    pub fn now(events: Vec<Value>) -> Self {
        Self {
            events,
            ..Self::default()
        }
    }
}

/// A request the real API would reject, rendered by [`Provider::error`].
#[derive(Clone, Debug, PartialEq)]
pub struct Rejection {
    pub status: u16,
    pub message: String,
    /// The provider's error kind: OpenAI's `code`, Anthropic's `type`,
    /// Gemini's `status`.
    pub kind: Option<&'static str>,
    /// The parameter at fault, where the provider names it.
    pub param: Option<String>,
}

impl Rejection {
    pub fn new(status: u16, message: impl Into<String>) -> Self {
        let message = message.into();
        Self {
            status,
            message,
            kind: None,
            param: None,
        }
    }

    pub fn kind(mut self, kind: &'static str) -> Self {
        self.kind = Some(kind);
        self
    }

    pub fn param(mut self, param: impl Into<String>) -> Self {
        self.param = Some(param.into());
        self
    }
}

/// A request a mock received, valid or not. A WebSocket's client events
/// are recorded too, each as a request of method `"WS"` whose body is the
/// event.
#[derive(Clone, Debug, Default)]
pub struct Request {
    pub method: String,
    pub path: String,
    /// The raw query string, `""` when there is none.
    pub query: String,
    pub headers: HeaderMap,
    /// The body as JSON; for `multipart/form-data`, an object of its text
    /// fields; `Null` otherwise.
    pub body: Value,
    /// The file parts of a `multipart/form-data` body.
    pub files: Vec<FormFile>,
    pub raw: Bytes,
    /// The `{name}` segment of the matched route, such as Gemini's model.
    pub param: Option<String>,
}

/// A file part of a `multipart/form-data` body.
#[derive(Clone, Debug, Default)]
pub struct FormFile {
    pub field: String,
    pub filename: String,
    pub content_type: Option<String>,
    pub bytes: Bytes,
}

impl Request {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.get(name).and_then(|v| v.to_str().ok())
    }

    /// The query parameter `name` (`""` when it has no value), not
    /// percent-decoded.
    pub fn query_param(&self, name: &str) -> Option<&str> {
        let mut pairs = self
            .query
            .split('&')
            .map(|p| p.split_once('=').unwrap_or((p, "")));
        pairs.find(|(k, _)| *k == name).map(|(_, v)| v)
    }

    /// The body as an object, if it is one.
    pub fn object(&self) -> Option<&Map<String, Value>> {
        self.body.as_object()
    }
}

/// What an endpoint may use while answering: the request id, and a place
/// to host files the client fetches back, such as image URLs.
#[derive(Clone, Debug)]
pub struct Ctx {
    pub id: u64,
    origin: String,
    files: Arc<Mutex<Vec<(String, Bytes)>>>,
}

const FILES: &str = "/mockllm/files/";

impl Ctx {
    /// Serves `bytes` as `content_type` from the mock; returns its URL.
    pub fn host(&self, content_type: &str, bytes: Bytes) -> String {
        let mut files = lock(&self.files);
        files.push((content_type.to_owned(), bytes));
        format!("{}{FILES}{}", self.origin, files.len() - 1)
    }

    fn hosted(&self, path: &str) -> Option<(String, Bytes)> {
        let i: usize = path.strip_prefix(FILES)?.parse().ok()?;
        lock(&self.files).get(i).cloned()
    }
}

/// The answer queue and the requests seen, shared with WebSocket tasks.
#[derive(Debug, Default)]
pub(crate) struct Script {
    answers: Mutex<VecDeque<Answer>>,
    requests: Mutex<Vec<Request>>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl Script {
    pub(crate) fn next(&self) -> Answer {
        lock(&self.answers).pop_front().unwrap_or_default()
    }

    pub(crate) fn record(&self, req: Request) {
        lock(&self.requests).push(req);
    }
}

struct State<P> {
    provider: P,
    endpoints: Vec<Box<dyn Endpoint<P>>>,
    script: Arc<Script>,
    ctx: Ctx,
}

/// A mock provider: a real server on `127.0.0.1:<random>`, stopped when
/// dropped. `FakeOpenAi`, `FakeAnthropic` and `FakeGemini` are aliases.
pub struct Mock<P> {
    script: Arc<Script>,
    origin: String,
    task: JoinHandle<()>,
    provider: PhantomData<fn() -> P>,
}

impl<P> std::fmt::Debug for Mock<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mock")
            .field("origin", &self.origin)
            .finish()
    }
}

impl<P: Provider + Default> Mock<P> {
    /// Starts the provider with its defaults.
    pub async fn start() -> Self {
        Self::serve(P::default()).await
    }
}

impl<P: Provider> Mock<P> {
    /// Starts `provider`.
    pub async fn serve(provider: P) -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .unwrap_or_else(|e| panic!("bind a mock provider: {e}"));
        let addr = listener
            .local_addr()
            .unwrap_or_else(|e| panic!("mock provider address: {e}"));
        let origin = format!("http://{addr}");
        let script = Arc::new(Script::default());
        let ctx = Ctx {
            id: 0,
            origin: origin.clone(),
            files: Arc::default(),
        };
        let state = Arc::new(State {
            endpoints: P::endpoints(),
            provider,
            script: script.clone(),
            ctx,
        });
        let app = axum::Router::new()
            .fallback(move |req: axum::extract::Request| handle(state.clone(), req));
        let task = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        let provider = PhantomData;
        Self {
            script,
            origin,
            task,
            provider,
        }
    }

    /// The URL to point the provider's SDK at.
    pub fn base_url(&self) -> String {
        format!("{}{}", self.origin, P::BASE_PATH)
    }

    /// Queues `answer` for the next valid request.
    pub fn answer(&self, answer: Answer) {
        lock(&self.script.answers).push_back(answer);
    }

    /// Every request received so far, valid or not.
    pub fn requests(&self) -> Vec<Request> {
        lock(&self.script.requests).clone()
    }
}

impl<P> Drop for Mock<P> {
    fn drop(&mut self) {
        self.task.abort();
    }
}

/// Whether `path` fits `pattern`, and its `{name}` segment if any.
fn fits(pattern: &str, path: &str) -> Option<Option<String>> {
    let Some((head, rest)) = pattern.split_once('{') else {
        return (pattern == path).then_some(None);
    };
    let tail = rest.split_once('}').map_or("", |(_, t)| t);
    let param = path.strip_prefix(head)?.strip_suffix(tail)?;
    let ok = !param.is_empty() && !param.contains(['/', ':']);
    ok.then(|| Some(param.to_owned()))
}

async fn handle<P: Provider>(state: Arc<State<P>>, req: axum::extract::Request) -> Response {
    let (mut parts, body) = req.into_parts();
    let upgrade = WebSocketUpgrade::from_request_parts(&mut parts, &()).await;
    let mut req = read(parts, body).await;
    let found = state.endpoints.iter().find_map(|e| {
        let param = e.routes().into_iter().find_map(|(method, pattern)| {
            fits(pattern, &req.path).filter(|_| method == req.method)
        })?;
        Some((e, param))
    });
    req.param = found.as_ref().and_then(|(_, param)| param.clone());
    state.script.record(req.clone());

    let (p, id) = (&state.provider, next_id());
    let ctx = Ctx {
        id,
        ..state.ctx.clone()
    };
    let headers = p.headers(id);
    let reject = |r: Rejection| Out::Json(r.status, p.error(&r, id));
    let default = Answer::default();
    if let Some((kind, bytes)) = ctx.hosted(&req.path).filter(|_| req.method == "GET") {
        return transport::send(&default, &[], Out::Bytes(kind, bytes)).await;
    }
    let Some((endpoint, _)) = found else {
        return transport::send(&default, &headers, reject(p.not_found(&req))).await;
    };
    if let Err(r) = p.authorize(&req).and_then(|()| endpoint.validate(p, &req)) {
        return transport::send(&default, &headers, reject(r)).await;
    }
    if let Some(session) = endpoint.session(p, &req) {
        let Ok(upgrade) = upgrade else {
            let r = Rejection::new(400, "mockllm: expected a WebSocket upgrade");
            return transport::send(&default, &headers, reject(r)).await;
        };
        let (script, path) = (state.script.clone(), req.path);
        return upgrade.on_upgrade(move |ws| transport::socket(ws, session, script, path, ctx));
    }
    let answer = state.script.next();
    let out = match (answer.status, &answer.reply) {
        (Some(status), _) => Out::Json(status, p.error(&scripted(p, &answer, status), id)),
        (None, Reply::Body(v)) => Out::Json(200, v.clone()),
        (None, Reply::Raw(b)) => Out::Bytes("application/json".into(), b.clone()),
        (None, _) => endpoint.respond(p, &req, &answer, &ctx),
    };
    transport::send(&answer, &headers, out).await
}

/// The rejection an [`Answer::error`] of `status` sends, with the
/// answer's own message if it has one.
pub(crate) fn scripted<P: Provider>(p: &P, answer: &Answer, status: u16) -> Rejection {
    let mut r = p.scripted(status);
    if let Some(m) = &answer.message {
        r.message.clone_from(m);
    }
    r
}

/// Reads a whole request; a `multipart/form-data` body is split into its
/// text fields (as `body`) and files.
async fn read(parts: Parts, body: axum::body::Body) -> Request {
    let raw = axum::body::to_bytes(body, usize::MAX)
        .await
        .unwrap_or_default();
    let mut req = Request {
        method: parts.method.to_string(),
        path: parts.uri.path().to_owned(),
        query: parts.uri.query().unwrap_or_default().to_owned(),
        headers: parts.headers,
        body: serde_json::from_slice(&raw).unwrap_or(Value::Null),
        files: Vec::new(),
        raw: raw.clone(),
        param: None,
    };
    let boundary = req.header("content-type").map(multer::parse_boundary);
    if let Some(Ok(boundary)) = boundary {
        let body = futures::stream::once(async move { Ok::<_, Infallible>(raw) });
        let mut form = multer::Multipart::new(body, boundary);
        let mut fields = Map::new();
        while let Ok(Some(part)) = form.next_field().await {
            let field = part.name().unwrap_or_default().to_owned();
            match part.file_name().map(str::to_owned) {
                Some(filename) => {
                    let content_type = part.content_type().map(ToString::to_string);
                    let bytes = part.bytes().await.unwrap_or_default();
                    let file = FormFile {
                        field,
                        filename,
                        content_type,
                        bytes,
                    };
                    req.files.push(file);
                }
                None => {
                    let text = part.text().await.unwrap_or_default();
                    fields.insert(field, text.into());
                }
            }
        }
        req.body = fields.into();
    }
    req
}

/// A 500 for an answer that does not fit the endpoint.
pub(crate) fn misfit(endpoint: &str, reply: &Reply) -> Out {
    Out::Misfit(format!(
        "mockllm: {endpoint} cannot answer {}",
        reply.kind()
    ))
}

#[cfg(test)]
mod tests {
    use super::fits;

    #[test]
    fn routes_match_one_placeholder() {
        let gemini = "/v1beta/models/{model}:generateContent";
        let path = "/v1beta/models/gemini-2.5-flash:generateContent";
        assert_eq!(fits(gemini, path), Some(Some("gemini-2.5-flash".into())));
        let stream = "/v1beta/models/gemini-2.5-flash:streamGenerateContent";
        assert_eq!(fits(gemini, stream), None);
        assert_eq!(fits("/v1/messages", "/v1/messages"), Some(None));
        assert_eq!(fits("/v1/messages", "/v1/messages/x"), None);
    }
}
