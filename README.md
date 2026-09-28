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
openai.answer(Answer::image(png_bytes));             // /v1/images/generations, /edits
openai.answer(Answer::audio(mp3_bytes));             // /v1/audio/speech
openai.answer(Answer::transcript("Hello"));          // /v1/audio/transcriptions
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
    images.rs       POST /v1/images/generations  (table inline; JSON and
                    the named-SSE stream shared with image_edits.rs)
    image_edits.rs  POST /v1/images/edits        (multipart or JSON)
    speech.rs       POST /v1/audio/speech        (table inline)
    transcriptions.rs POST /v1/audio/transcriptions (multipart; table inline)
    realtime.rs     GET  /v1/realtime, WebSocket
    responses.rs    POST /v1/responses           (rules from chat_models.rs)
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
- **OpenAI Responses** (`openai/responses.rs`,
  [reference](https://developers.openai.com/api/reference/resources/responses/methods/create),
  [streaming events](https://developers.openai.com/api/reference/resources/responses/streaming-events)):
  `model` required unless a `prompt` template is given, not both
  `previous_response_id` and `conversation`, `stream_options` only with
  `stream`, `temperature` 0..=2, `top_logprobs` 0..=20, `include` values,
  `input` a string or items, message roles, `function_call` and
  `function_call_output` items paired by `call_id` (an output may answer a
  stored call with `previous_response_id` or `conversation`), function
  tools with a top-level `name`; per-model rules from the Chat Completions
  table (`max_output_tokens`, `reasoning.effort` values, sampling and
  `message.output_text.logprobs` only where the model allows, per the
  [GPT-6 guide](https://developers.openai.com/api/docs/guides/latest-model)).
  The Response object echoing the request's settings with their defaults;
  `message` items with `output_text`, `function_call` items (`fc_` id,
  `call_id`), a `reasoning` item first on reasoning models (`summary` only
  with `reasoning.summary`, `encrypted_content`), usage with
  `reasoning_tokens`. Streaming as named events (`event: <type>`) with
  `sequence_number` from 0: `response.created`, `response.in_progress`,
  `output_item.added`/`.done`, `content_part.*`, `output_text.delta`/`.done`,
  `function_call_arguments.delta`/`.done`,
  `reasoning_summary_part.*`/`reasoning_summary_text.*`,
  `response.completed`; `obfuscation` on deltas unless
  `include_obfuscation: false`; no `[DONE]`; a mid-stream `error` event
  (`code`, `message`, `param`, `sequence_number`).
- **OpenAI Embeddings** (`openai/embeddings.rs`): `input` required, not
  empty, at most 2048 items; vector length per model, `dimensions` (not on
  `ada-002`), `encoding_format` float or base64 (little-endian f32), usage.
- **OpenAI Images, generations** (`openai/images.rs`): `prompt` required,
  `n` 1..=10 and 1 on `dall-e-3`, sizes per model, `url` (served by the
  mock) or `b64_json`, GPT image models always base64, no
  `response_format`, with `usage`; `revised_prompt` on `dall-e-3`.
  `stream: true` (GPT image models only; without `model` it picks one):
  named SSE events, `partial_images` (0..=3)
  `image_generation.partial_image`s with `partial_image_index`, then
  `image_generation.completed` with `usage`; each carries `size`,
  `quality`, `background`, `output_format`, `created_at` ([streaming
  events](https://developers.openai.com/api/reference/resources/images/generation-streaming-events)).
- **OpenAI Images, edits** (`openai/image_edits.rs`): multipart with
  `image` / `image[]` and `mask` file parts, or JSON with `images`
  references (GPT image models only); `prompt` and an image required;
  default model `gpt-image-1.5`; `dall-e-2` one square PNG, prompt <= 1000
  chars, its sizes, `url` or `b64_json`; GPT image models up to 16
  PNG/WebP/JPEG images, prompt <= 32000, base64 only; mask PNG; `n`
  1..=10 (form text too); no `dall-e-3`; streaming as generations with
  `image_edit.*` events ([reference](https://developers.openai.com/api/reference/resources/images/methods/edit),
  [streaming events](https://developers.openai.com/api/reference/resources/images/edit-streaming-events)).
- **OpenAI speech** (`openai/speech.rs`): `model`, `input` (<= 4096
  chars), `voice` (a built-in name or `{ "id" }`) required; `speed`
  0.25..=4; `response_format` `mp3` (default) / `opus` / `aac` / `flac` /
  `wav` / `pcm` as `audio/mpeg`, `audio/opus`, `audio/aac`, `audio/flac`,
  `audio/wav`, `audio/pcm`, always chunked (1024-byte chunks, so `pace`
  and `cut_after` apply per chunk); `stream_format: "sse"` as `data:`
  events `speech.audio.delta` (base64 `audio`) then `speech.audio.done`
  with `usage`. Unscripted: 0.1 s of silence (a silent MP3 frame, a WAV,
  raw PCM; other formats get raw PCM, not decodable)
  ([reference](https://developers.openai.com/api/reference/resources/audio/subresources/speech/methods/create),
  events and content types from the [OpenAPI spec](https://github.com/openai/openai-openapi)).
- **OpenAI transcriptions** (`openai/transcriptions.rs`, multipart):
  `file` and `model` required, the file's extension among the listed
  audio formats, over 25 MB a 413; `response_format` `json` (text and
  usage, tokens or duration per model), `text` / `srt` / `vtt` as
  `text/plain` (one cue), `verbose_json` (`segments` by default, `words`
  with `timestamp_granularities[]=word`; repeated `[]` fields are read),
  `diarized_json`; `include[]=logprobs` (with `json` only); `temperature`
  0..=1; `stream: true` as `data:` events `transcript.text.delta` per
  word, `transcript.text.segment` when diarized (deltas with
  `segment_id`), then `transcript.text.done` with usage; `languages` from
  `gpt-transcribe` ([reference](https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create),
  [streaming events](https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/streaming-events),
  [guide](https://developers.openai.com/api/docs/guides/speech-to-text)).
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
| `dall-e-2`, `dall-e-3`, `gpt-image-*` | sizes; `n` = 1 on `dall-e-3`; GPT image: base64 only, no `response_format`; `stream` only on GPT image | [images reference](https://developers.openai.com/api/reference/resources/images/methods/generate) |
| `dall-e-2`, `gpt-image-*` (edits) | `dall-e-2`: one PNG, prompt 1000 chars, multipart only; GPT image: 16 images, prompt 32000; no `dall-e-3` | [edit reference](https://developers.openai.com/api/reference/resources/images/methods/edit) |
| `tts-1`, `tts-1-hd` | no `stream_format: "sse"`; voices `alloy`, `ash`, `coral`, `echo`, `fable`, `onyx`, `nova`, `sage`, `shimmer` only | [speech reference](https://developers.openai.com/api/reference/resources/audio/subresources/speech/methods/create), [TTS guide](https://developers.openai.com/api/docs/guides/text-to-speech#voice-options) |
| `whisper-1` | formats `json`/`text`/`srt`/`verbose_json`/`vtt`; `stream` ignored; timestamp granularities; no logprobs; billed by duration | [transcription reference](https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create), [guide](https://developers.openai.com/api/docs/guides/speech-to-text) |
| `gpt-4o-transcribe`, `gpt-4o-mini-transcribe` | `json` only; logprobs | transcription reference |
| `gpt-4o-transcribe-diarize` | `json`/`text`/`diarized_json`; no `prompt`, `include`, timestamp granularities | transcription reference |
| `gpt-transcribe` | returns `languages`; no logprobs, no timestamp granularities | transcription reference, guide |
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
