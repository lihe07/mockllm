//! The script: what a mock answers to its next valid request.

use std::time::Duration;

use bytes::Bytes;
use serde_json::Value;

/// The text an endpoint answers when nothing is queued.
pub(crate) const UNSCRIPTED: &str = "Hi there";

/// What a mock answers to its next valid request (or, on a WebSocket, to
/// the next client event that asks for a response). Requests that fail
/// validation do not use up an answer. With nothing queued, a mock
/// answers [`Answer::default`]: each endpoint's own default, such as the
/// text "Hi there" or a unit-vector embedding.
///
/// An answer an endpoint cannot frame (an image to a chat endpoint) is a
/// `500` whose body names both, so a test fails clearly.
#[derive(Clone, Debug)]
pub struct Answer {
    pub(crate) reply: Reply,
    pub(crate) reasoning: Option<String>,
    pub(crate) status: Option<u16>,
    pub(crate) cut_after: Option<usize>,
    pub(crate) error_after: Option<usize>,
    pub(crate) delay_first: Duration,
    pub(crate) pace: Duration,
    pub(crate) retry_after: Option<u64>,
}

/// What an [`Answer`] says. Each endpoint frames the kinds it can.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Reply {
    /// Nothing scripted: the endpoint's own default.
    Unscripted,
    Text(String),
    ToolCall {
        name: String,
        args: Value,
    },
    /// One vector, returned for every input.
    Embedding(Vec<f32>),
    /// Image bytes, returned for each image asked for.
    Image(Bytes),
    Audio(Bytes),
    Transcript(String),
    /// Raw events, sent as they are: SSE `data:` (or named by `type`) or
    /// WebSocket messages.
    Events(Vec<Value>),
}

impl Reply {
    /// The text a text endpoint says: a [`Reply::Text`], or its default
    /// when unscripted.
    pub fn text(&self) -> Option<&str> {
        match self {
            Reply::Unscripted => Some(UNSCRIPTED),
            Reply::Text(t) => Some(t),
            _ => None,
        }
    }

    /// The kind's name, for misfit errors.
    pub fn kind(&self) -> &'static str {
        match self {
            Reply::Unscripted => "an unscripted answer",
            Reply::Text(_) => "a text",
            Reply::ToolCall { .. } => "a tool call",
            Reply::Embedding(_) => "an embedding",
            Reply::Image(_) => "an image",
            Reply::Audio(_) => "audio",
            Reply::Transcript(_) => "a transcript",
            Reply::Events(_) => "raw events",
        }
    }
}

impl Default for Answer {
    fn default() -> Self {
        Self::of(Reply::Unscripted)
    }
}

impl Answer {
    fn of(reply: Reply) -> Self {
        Self {
            reply,
            reasoning: None,
            status: None,
            cut_after: None,
            error_after: None,
            delay_first: Duration::ZERO,
            pace: Duration::ZERO,
            retry_after: None,
        }
    }

    /// An assistant text; streamed a word per event.
    pub fn text(text: impl Into<String>) -> Self {
        Self::of(Reply::Text(text.into()))
    }

    /// One call of tool `name` with `args`; streamed arguments in pieces.
    pub fn tool_call(name: impl Into<String>, args: Value) -> Self {
        let name = name.into();
        Self::of(Reply::ToolCall { name, args })
    }

    /// `vector` as the embedding of every input, as floats or base64 per
    /// the request's `encoding_format`.
    pub fn embedding(vector: Vec<f32>) -> Self {
        Self::of(Reply::Embedding(vector))
    }

    /// `bytes` as each image asked for, base64 or a URL served by the mock,
    /// per the request's `response_format` and model.
    pub fn image(bytes: impl Into<Bytes>) -> Self {
        Self::of(Reply::Image(bytes.into()))
    }

    /// Audio bytes, for speech endpoints.
    pub fn audio(bytes: impl Into<Bytes>) -> Self {
        Self::of(Reply::Audio(bytes.into()))
    }

    /// A transcript, for transcription endpoints.
    pub fn transcript(text: impl Into<String>) -> Self {
        Self::of(Reply::Transcript(text.into()))
    }

    /// Raw events sent as they are, for what the endpoint does not model.
    pub fn events(events: Vec<Value>) -> Self {
        Self::of(Reply::Events(events))
    }

    /// An error of `status` in the provider's own shape, before any body.
    pub fn error(status: u16) -> Self {
        Self {
            status: Some(status),
            ..Self::default()
        }
    }

    /// Reasoning before the answer. OpenAI reasoning models count it in
    /// `reasoning_tokens` and show none of it; `Quirks::reasoning_content`
    /// shows it; Anthropic sends it as a `thinking` block when the model
    /// thinks on the request (its text empty when `display` is omitted);
    /// Gemini counts it in `thoughtsTokenCount` and shows it as a
    /// `thought` part with `includeThoughts`.
    pub fn reasoning(mut self, text: impl Into<String>) -> Self {
        self.reasoning = Some(text.into());
        self
    }

    /// The stream breaks off after `n` events: an SSE or chunked body
    /// ends without its end, a WebSocket closes without a close frame. A
    /// whole body breaks off halfway.
    pub fn cut_after(mut self, n: usize) -> Self {
        self.cut_after = Some(n);
        self
    }

    /// After `n` events the stream sends the endpoint's mid-stream error
    /// event and ends (a chunked binary body just ends).
    pub fn error_after(mut self, n: usize) -> Self {
        self.error_after = Some(n);
        self
    }

    /// Waits `ms` before the response head (or the first WebSocket event).
    pub fn delay_first(mut self, ms: u64) -> Self {
        self.delay_first = Duration::from_millis(ms);
        self
    }

    /// Waits `ms` between events or chunks.
    pub fn pace(mut self, ms: u64) -> Self {
        self.pace = Duration::from_millis(ms);
        self
    }

    /// Adds `retry-after: <secs>` to the response.
    pub fn retry_after(mut self, secs: u64) -> Self {
        self.retry_after = Some(secs);
        self
    }

    pub fn reply(&self) -> &Reply {
        &self.reply
    }

    pub fn reasoning_text(&self) -> Option<&str> {
        self.reasoning.as_deref()
    }

    /// The status of an [`Answer::error`].
    pub fn status(&self) -> Option<u16> {
        self.status
    }
}
