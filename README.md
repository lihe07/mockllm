# mockllm

Mock LLM providers for tests: real servers on `127.0.0.1:<random>` that
behave like OpenAI, Anthropic and Gemini endpoints over HTTP and
WebSocket. Each endpoint checks requests by the provider's own rules,
including per-model ones, rejects them in the provider's own error shape
and status, and answers in its exact framing: JSON, SSE, binary or
WebSocket events. Test an LLM client, SDK or proxy from outside, over real
connections. No fixture files: the provider logic is code.

## API

```rust
use mockllm::{Answer, FakeAnthropic, FakeGemini, FakeOpenAi, Quirks};
use serde_json::json;

let openai = FakeOpenAi::start().await;   // real server, random port
openai.base_url();                        // "http://127.0.0.1:<port>/v1"

// Queue answers for the next valid requests. Unscripted, each endpoint
// answers its default: "Hi there", a unit vector, a 1x1 PNG.
openai.answer(Answer::text("Hi there"));
openai.answer(Answer::tool_call("get_weather", json!({ "city": "Paris" })));
openai.answer(Answer::text("Hi").reasoning("hidden thoughts"));
openai.answer(Answer::embedding(vec![0.1, 0.2]));    // /v1/embeddings
openai.answer(Answer::image(png_bytes));             // /v1/images/generations
openai.answer(Answer::events(vec![json!({ "type": "custom" })])); // raw SSE or WebSocket events
openai.answer(Answer::error(429).retry_after(7));    // provider's error shape

// Timing and breaks, on SSE, chunked bodies and WebSocket alike.
openai.answer(Answer::text("Hi there").cut_after(2));   // connection drops after 2 events
openai.answer(Answer::text("Hi there").error_after(2)); // provider's mid-stream error event
openai.answer(Answer::text("Hi").delay_first(500));     // slow first byte (ms)
openai.answer(Answer::text("a b c").pace(50));          // ms between events

// What the client sent, valid or not: method, path, query, headers, body
// JSON (multipart text fields), files. WebSocket client events too, as
// method "WS".
let sent = openai.requests();

let anthropic = FakeAnthropic::start().await;
anthropic.base_url();                     // "http://127.0.0.1:<port>", no version
let gemini = FakeGemini::start().await;
gemini.base_url();                        // "http://127.0.0.1:<port>", the SDK adds /v1beta

// An OpenAI-compatible provider that deviates from OpenAI.
let compat = FakeOpenAi::with_quirks(Quirks::new().no_done().reasoning_content()).await;
```

A request that fails validation does not use up a queued answer. An
answer the endpoint cannot frame (an image to chat completions) is a `500`
whose body says so. The server stops when the mock is dropped.

## Design

The unit is an **endpoint**: a route, its rules and how it frames an
`Answer`. A **provider** is a named set of endpoints plus what they share.
Everything else is the **core**, written once.

```
src/
  lib.rs            re-exports
  core.rs           Provider, Endpoint, Session traits; Mock<P>: listener,
                    recording, answer queue, routing, multipart parsing,
                    hosted files (image URLs)
  transport.rs      Out: JSON, SSE, bytes, chunks, misfit; WebSocket loop;
                    delay_first, pace, cut_after, error_after, retry-after
  answer.rs         Answer and Reply
  util.rs           words, tokens, ids, the per-model table lookup
  openai/
    mod.rs          OpenAi: Bearer auth, error shape, x-request-id, Quirks,
                    the endpoint list
    chat.rs         POST /v1/chat/completions   (+ chat_models.rs)
    embeddings.rs   POST /v1/embeddings          (table inline)
    images.rs       POST /v1/images/generations  (table inline)
    realtime.rs     GET  /v1/realtime, WebSocket
  anthropic/
    mod.rs          Anthropic: x-api-key, anthropic-version, error shape
    messages.rs     POST /v1/messages            (+ messages_models.rs)
  gemini/
    mod.rs          Gemini: API key, Google's error shape
    generate.rs     POST /v1beta/models/{model}:generateContent and
                    :streamGenerateContent       (+ generate_models.rs)
tests/<provider>_<endpoint>.rs, tests/transports.rs
```

```rust
pub trait Provider {
    const BASE_PATH: &'static str;                  // "/v1", ""
    fn endpoints() -> Vec<Box<dyn Endpoint<Self>>>;
    fn authorize(&self, req: &Request) -> Result<(), Rejection>;
    fn error(&self, rejection: &Rejection, id: u64) -> Value;   // the error shape
    fn scripted(&self, status: u16) -> Rejection;   // Answer::error(status)
    fn not_found(&self, req: &Request) -> Rejection;            // default: scripted(404)
    fn headers(&self, id: u64) -> Vec<(&'static str, String)>;  // default: none
}

pub trait Endpoint<P> {
    fn route(&self) -> (&'static str, &'static str); // ("POST", "/v1beta/models/{model}:generateContent")
    fn validate(&self, provider: &P, req: &Request) -> Result<(), Rejection>;
    fn respond(&self, provider: &P, req: &Request, answer: &Answer, ctx: &Ctx) -> Out; // HTTP
    fn session(&self, provider: &P, req: &Request) -> Option<Box<dyn Session>>;     // WebSocket
}

pub enum Out {
    Json(u16, Value),
    Sse(Vec<String>, String),   // framed events (data() or event()), mid-stream error event
    Bytes(String, Bytes),       // content type, whole body
    Chunks(String, Vec<Bytes>), // content type, chunks paced like events
    Misfit(String),             // the answer does not fit: a 500
}
```

A request is recorded, routed, checked by `authorize` then `validate`,
and only then takes an answer. An `Answer::error` is sent by the core in
the provider's shape; any other answer goes to `respond`, or, on a
WebSocket, to the `Session` when a client event asks for a response.

## What is modelled

Every endpoint: its auth, unknown top-level parameters rejected (from the
API reference's parameter list), required fields, the error shape and
status, `retry-after`.

- **OpenAI Chat Completions** (`openai/chat.rs`): `model`/`messages`
  required, roles, the tool-call/result pairing rule, `stream_options` only
  with `stream`, `n` choices (streamed too), the final empty-delta chunk
  with `finish_reason`, `include_usage` (usage `null` on every chunk, then
  a `choices: []` usage chunk), `obfuscation` unless
  `include_obfuscation: false`, `data: [DONE]`, `x-request-id`.
- **OpenAI Embeddings** (`openai/embeddings.rs`): `input` required, not
  empty, at most 2048 items; vector length per model, `dimensions` (not on
  `ada-002`), `encoding_format` float or base64 (little-endian f32), usage.
- **OpenAI Images, generations** (`openai/images.rs`): `prompt` required,
  `n` 1..=10 and 1 on `dall-e-3`, sizes per model, `url` (served by the
  mock) or `b64_json`, GPT image models always base64, no
  `response_format`, with `usage`; `revised_prompt` on `dall-e-3`. Image
  streaming is not modelled yet.
- **OpenAI Realtime** (`openai/realtime.rs`, WebSocket): `session.created`
  on connect, `session.update` to `session.updated`, `response.create` to
  `response.created`, `output_item.added`, `content_part.added`,
  `output_text.delta`s (output modality `text`; `output_audio_transcript`
  events and no audio bytes for `audio`, the default), the `.done` events
  and `response.done` with usage; an `error` event for a client event
  without `type`. Other client events are accepted without a reply.
- **Anthropic Messages** (`anthropic/messages.rs`): `anthropic-version`
  and `max_tokens` required, no `system` role, temperature in 0..=1,
  non-empty content, tool-use/result pairing, named SSE events with
  `ping`, `input_json_delta` (opening with an empty one), `thinking` blocks
  with `signature_delta`, `event: error` mid-stream, `request_id` in error
  bodies, 529 `overloaded_error`.
- **Gemini generateContent** (`gemini/generate.rs`): the model from the
  path, `x-goog-api-key` or `?key=`, `contents` required, roles `user` and
  `model`, `candidates` with `finishReason` and `usageMetadata`,
  `functionCall` parts, `thought` parts with `includeThoughts` and
  `thoughtsTokenCount`; `streamGenerateContent?alt=sse` as `data:` events
  ending in CRLF CRLF, a whole `GenerateContentResponse` each, the last
  with `finishReason` and usage; without `alt=sse`, one JSON array sent a
  response at a time; a mid-stream error as the bare error JSON; errors as
  `{"error": {"code", "message", "status"}}`.

### Per-model rules

One table per endpoint, beside it (`openai/chat_models.rs`,
`anthropic/messages_models.rs`, `gemini/generate_models.rs`) or inline for
small ones. Every table has the same shape: a `Model` row type with a
`prefix`, a `BASE` row of no rules, and `MODELS`; a model id takes the row
with the longest matching prefix (`util::lookup`), an unknown id gets
`BASE`. Every row cites its source in a comment. Only rules confirmed in
the docs are in; where the docs give no error wording, the wording is the
one commonly seen from the API.

| Model family | Rules | Source |
|---|---|---|
| `gpt-4o`, `gpt-4.1`, `gpt-5-chat` | max output 16,384 / 32,768 / 16,384 | [model pages](https://developers.openai.com/api/docs/models/gpt-4o) |
| `o1`, `o3`, `o4-mini` | `max_tokens` rejected (use `max_completion_tokens`); max output 100,000; reasoning tokens counted | [chat reference, `max_tokens`](https://developers.openai.com/api/reference/resources/chat), [o1](https://developers.openai.com/api/docs/models/o1) |
| `o1-mini` | as o-series, max output 65,536, no function calling | [o1-mini](https://developers.openai.com/api/docs/models/o1-mini) |
| `gpt-5` (`-mini`, `-nano`) | effort `minimal`..`high`; `temperature`/`top_p`/`logprobs` always rejected | [gpt-5](https://developers.openai.com/api/docs/models/gpt-5), [GPT-5.4 guide](https://developers.openai.com/api/docs/guides/latest-model/gpt-5.4) |
| `gpt-5.1` | effort `none` (default)..`high` | [gpt-5.1](https://developers.openai.com/api/docs/models/gpt-5.1) |
| `gpt-5.2`, `gpt-5.4` | effort `none` (default)..`xhigh`; sampling only at effort `none` | [GPT-5.4 guide](https://developers.openai.com/api/docs/guides/latest-model/gpt-5.4) |
| `gpt-5.5`, `gpt-5.6-terra` | effort `none`..`xhigh` / `max`, default `medium` | [gpt-5.5](https://developers.openai.com/api/docs/models/gpt-5.5), [gpt-5.6-terra](https://developers.openai.com/api/docs/models/gpt-5.6-terra) |
| `gpt-6-astra` | effort `none` rejected; so sampling rejected; no function calling in Chat Completions | [GPT-6 guide](https://developers.openai.com/api/docs/guides/latest-model) |
| `gpt-6-sol`, `gpt-6-luna` | sampling and function calling only at effort `none` | [GPT-6 guide](https://developers.openai.com/api/docs/guides/latest-model) |
| `text-embedding-3-small`, `-3-large` | 1536 / 3072 dimensions by default | [embeddings guide](https://developers.openai.com/api/docs/guides/embeddings) |
| `text-embedding-ada-002` | `dimensions` rejected | [embeddings reference](https://developers.openai.com/api/reference/resources/embeddings/methods/create) |
| `dall-e-2`, `dall-e-3`, `gpt-image-*` | sizes; `n` = 1 on `dall-e-3`; GPT image: base64 only, no `response_format` | [images reference](https://developers.openai.com/api/reference/resources/images/methods/generate) |
| `claude-fable-5-1`, `claude-mythos-5-1`, `claude-opus-5-5` | thinking always on (`enabled`, `disabled` rejected); forced tool use rejected | [thinking troubleshooting](https://platform.claude.com/docs/en/build-with-claude/thinking-troubleshooting), [errors](https://platform.claude.com/docs/en/api/errors) |
| `claude-fable-5`, `claude-mythos-5` | thinking always on | thinking troubleshooting |
| `claude-opus-5`, `claude-sonnet-5` | thinking on by default, `enabled` rejected | thinking troubleshooting |
| `claude-sonnet-5-5` | `between_tools` instead of `disabled` (not at effort `xhigh`/`max`); forced tool use rejected | thinking troubleshooting, errors |
| all Claude 5.x above | `display` defaults to omitted (empty thinking text) | [thinking](https://platform.claude.com/docs/en/build-with-claude/thinking) |
| Claude 4.7 and later | non-default `temperature`/`top_p`/`top_k` rejected; 1M context; 128k output | [thinking limits](https://platform.claude.com/docs/en/build-with-claude/thinking#limits-and-feature-compatibility), [context windows](https://platform.claude.com/docs/en/build-with-claude/context-windows), [models](https://platform.claude.com/docs/en/models/overview) |
| Claude 4.6 and later | assistant prefill rejected | [errors](https://platform.claude.com/docs/en/api/errors) |
| `claude-opus-4-6`, `claude-sonnet-4-6` | adaptive and extended thinking; sampling limits only while thinking | thinking troubleshooting, thinking limits |
| `claude-opus-4-5`, `claude-sonnet-4-5`, `claude-haiku-4-5` | extended only (`adaptive` rejected); 64k output; 200k context | thinking troubleshooting, [Sonnet 4.5](https://platform.claude.com/docs/en/models/sonnet-4-5/overview) |
| `claude-haiku-4-5`, `claude-opus-4-1` | `temperature` and `top_p` not both | [Haiku 4.5 migration](https://platform.claude.com/docs/en/models/haiku-4-5/migration-guide), [release notes 2025-08-05](https://platform.claude.com/docs/en/release-notes/overview) |
| Claude, extended thinking | `budget_tokens` >= 1024 and < `max_tokens` (unless the interleaved-thinking beta); no forced tool use; the assistant turn before tool results starts with a thinking block | [extended thinking](https://platform.claude.com/docs/en/build-with-claude/extended-thinking) |
| Claude, while thinking | no prefill; `temperature` only 1, no `top_k`, `top_p` >= 0.95 (older models) | thinking limits |
| `gemini-2.0-flash` | max output 8,192; no `thinkingLevel` | [model page](https://ai.google.dev/gemini-api/docs/models/gemini-2.0-flash), [`ThinkingConfig`](https://ai.google.dev/api/generate-content) |
| `gemini-2.5-pro`, `-flash`, `-flash-lite` | max output 65,536; no `thinkingLevel`; thinking on (off on Flash-Lite) | model pages, `ThinkingConfig`, [thinking](https://ai.google.dev/gemini-api/docs/thinking) |
| `gemini-3-*`, `gemini-3.x-*` | max output 65,536; thinking on; `thinkingLevel` values per model (`low`/`high` on 3 Pro, `low`..`high` on 3.1 Pro, 3.7 and 3.8 Flash, `minimal`..`high` otherwise) | model pages, thinking |

### Quirks of OpenAI-compatible providers

| Quirk | Behaviour | Source |
|---|---|---|
| `usage_every_chunk()` | every chunk carries running `usage` | [vLLM `continuous_usage_stats`](https://docs.vllm.ai/en/v0.8.5/serving/openai_compatible_server.html) |
| `no_done()` | the stream ends without `data: [DONE]` | seen in production; no public doc |
| `reasoning_content()` | `Answer::reasoning` streams as `delta.reasoning_content` before the content, and is a field of the whole message | [DeepSeek API](https://api-docs.deepseek.com/api/create-chat-completion) |

## How to add

- **An endpoint** to an existing provider: one file,
  `src/<provider>/<endpoint>.rs`, with a unit struct implementing
  `Endpoint<Provider>`: `route`, `validate` (its rules and per-model
  table, with the source URL on every rule), and `respond` returning an
  `Out` (or `session` for a WebSocket). Match on `answer.reply()` for the
  kinds it frames and return `misfit(..)` for the rest. Then one line in
  the provider's `endpoints()` in `mod.rs`, and
  `tests/<provider>_<endpoint>.rs`. Nothing else changes: auth, errors,
  recording, the queue and timing are already there.
- **A provider**: `src/<provider>/mod.rs` with a unit struct implementing
  `Provider` (base path, auth, error shape, scripted error table, request
  id header), `pub type Fake<Name> = Mock<Name>;`, its endpoints as
  above, and one `mod` and `pub use` line in `lib.rs`. Gemini
  (`src/gemini/`) is the smallest example.
- **A transport**: a variant of `Out` sent by `transport::send`, applying
  the answer's timing like the others (see `breaks`), and a test in
  `tests/transports.rs` through a small provider written in the test.
  WebSocket sessions go through `Session` and `transport::socket`.
- **A kind of answer**: a `Reply` variant and an `Answer` constructor in
  `answer.rs`, and a name in `Reply::kind`.
- **A per-model rule**: a field of the table's `Model` with its default in
  `BASE`, set on the rows it applies to with the source URL in a comment,
  checked in the endpoint's `validate`. A new model family is one row. A
  test that the rule rejects, and that a model without it accepts.
- **A quirk of an OpenAI-compatible provider**: a flag and a builder
  method on `Quirks` in `src/openai/mod.rs`, with the provider's doc in
  its doc comment, acted on in the endpoint, and one test.

## Roadmap

Each is its own endpoint file and test file; the only shared edit is one
line in the provider's `endpoints()`. The `Answer` kinds and transports
they need already exist.

| Endpoint | File | Needs | Docs |
|---|---|---|---|
| OpenAI Responses, `POST /v1/responses` | `openai/responses.rs` | JSON, named SSE | [reference](https://developers.openai.com/api/reference/resources/responses/methods/create), [streaming events](https://developers.openai.com/api/reference/resources/responses/streaming-events) |
| OpenAI speech, `POST /v1/audio/speech` | `openai/speech.rs` | `Answer::audio`, `Out::Bytes` / `Out::Chunks` | [reference](https://developers.openai.com/api/reference/resources/audio/subresources/speech/methods/create) |
| OpenAI transcriptions, `POST /v1/audio/transcriptions` | `openai/transcriptions.rs` | multipart, `Answer::transcript`, SSE | [reference](https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create) |
| OpenAI image edits, `POST /v1/images/edits` | `openai/image_edits.rs` | multipart, `Answer::image` | [reference](https://developers.openai.com/api/reference/resources/images/methods/edit) |
| OpenAI image streaming (`stream: true`) | `openai/images.rs` | named SSE | [image generation guide](https://developers.openai.com/api/docs/guides/image-generation) |
| OpenAI files, `/v1/files` | `openai/files.rs` | multipart, `Ctx::host` | [reference](https://developers.openai.com/api/reference/resources/files) |
| OpenAI batch, `/v1/batches` | `openai/batches.rs` | JSON | [reference](https://developers.openai.com/api/reference/resources/batches) |
| Realtime audio, tools, conversation items | `openai/realtime.rs` | WebSocket, `Answer::audio` | [server events](https://developers.openai.com/api/reference/resources/realtime/server-events) |
| Anthropic count tokens, `POST /v1/messages/count_tokens` | `anthropic/count_tokens.rs` | JSON | [reference](https://platform.claude.com/docs/en/api/messages/count_tokens) |
| Anthropic Message Batches, `/v1/messages/batches` | `anthropic/batches.rs` | JSON | [reference](https://platform.claude.com/docs/en/api/messages/batches) |
| Anthropic models, `GET /v1/models` | `anthropic/models.rs` | JSON | [reference](https://platform.claude.com/docs/en/api/models/list) |
| Anthropic files, `/v1/files` | `anthropic/files.rs` | multipart | [reference](https://platform.claude.com/docs/en/api/files-create) |
| Gemini embeddings, `:embedContent` | `gemini/embed.rs` | `Answer::embedding` | [reference](https://ai.google.dev/api/embeddings) |
| Gemini Live (WebSocket) | `gemini/live.rs` | `Session` | [guide](https://ai.google.dev/gemini-api/docs/live), [reference](https://ai.google.dev/api/live) |
| Gemini Imagen, `:predict` | `gemini/imagen.rs` | `Answer::image` | [guide](https://ai.google.dev/gemini-api/docs/imagen) |
| Gemini Interactions API | `gemini/interactions.rs` | JSON, named SSE | [guide](https://ai.google.dev/gemini-api/docs/interactions) |

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.
