//! Mock LLM providers: real servers on `127.0.0.1:<random>` that check
//! requests by the provider's own rules, including per-model ones, and
//! answer in its exact framing, over HTTP (JSON, SSE, binary) and
//! WebSocket. For testing an LLM client, SDK or proxy from outside.
//!
//! The unit is an [`Endpoint`]: a route, its rules (`validate`) and how it
//! frames an [`Answer`] (`respond`, or a WebSocket [`Session`]). A
//! [`Provider`] is a named set of endpoints plus what they share: auth,
//! error shape, request ids. Everything else is written once in the core:
//! the listener, recording, the answer queue, routing, the transports and
//! their timing. Each endpoint is one file, with its per-model table
//! beside it, each row citing the docs it comes from. Nothing is read
//! from fixture files; the provider logic is code.
//!
//! ```no_run
//! # async fn demo() {
//! use mockllm::{Answer, FakeOpenAi};
//! let openai = FakeOpenAi::start().await;
//! openai.answer(Answer::text("Hi there").cut_after(2));
//! // point the client under test at openai.base_url(), send, then:
//! let sent = openai.requests();
//! # }
//! ```

mod answer;
mod core;
mod transport;
mod util;

mod anthropic;
mod gemini;
mod openai;

pub use crate::answer::{Answer, Reply};
pub use crate::core::{Ctx, Endpoint, FormFile, Mock, Provider, Rejection, Request, Session, Turn};
pub use crate::transport::{Out, data, event};

pub use anthropic::{Anthropic, FakeAnthropic};
pub use gemini::{FakeGemini, Gemini};
pub use openai::{FakeOpenAi, OpenAi, Quirks};
