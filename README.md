# mockllm

Mock LLM providers for tests: real HTTP servers on `127.0.0.1:<random>`
that behave like OpenAI Chat Completions and Anthropic Messages. Each one
checks requests by the provider's own rules, including per-model ones,
rejects them in the provider's own error shape and status, and answers in
its exact JSON and SSE framing. Test an LLM client, SDK or proxy from
outside, over real HTTP. No fixture files: the provider logic is code.

## API

```rust
use mockllm::{Answer, FakeAnthropic, FakeOpenAi, Quirks};
use serde_json::json;

let openai = FakeOpenAi::start().await;   // real server, random port
openai.base_url();                        // "http://127.0.0.1:<port>/v1"

// Queue answers for the next valid requests. Unscripted: "Hi there".
openai.answer(Answer::text("Hi there"));
openai.answer(Answer::tool_call("get_weather", json!({ "city": "Paris" })));
openai.answer(Answer::text("Hi").reasoning("hidden thoughts"));
openai.answer(Answer::error(429).retry_after(7));     // provider's error shape

// Timing and breaks.
openai.answer(Answer::text("Hi there").cut_after(2));   // connection drops after 2 SSE events
openai.answer(Answer::text("Hi there").error_after(2)); // provider's mid-stream error event
openai.answer(Answer::text("Hi").delay_first(500));     // slow first byte (ms)
openai.answer(Answer::text("a b c").pace(50));          // ms between SSE events

// What the client sent, valid or not: method, path, headers, body JSON.
let sent = openai.requests();

let anthropic = FakeAnthropic::start().await;
anthropic.base_url();                     // "http://127.0.0.1:<port>", no version

// An OpenAI-compatible provider that deviates from OpenAI.
let compat = FakeOpenAi::with_quirks(Quirks::new().no_done().reasoning_content()).await;
```

A request that fails validation does not use up a queued answer. The
server stops when the mock is dropped.

## What is modelled

Both: the auth header, unknown top-level parameters rejected (from the
API reference's parameter list), required fields, the tool-call/result
pairing rule, whole and streamed text and tool-call answers, usage, the
error shapes, `retry-after`, request ids.

- **OpenAI** (`src/openai.rs`): `model`/`messages` required, roles,
  `stream_options` only with `stream`, `n` choices (streamed too), the
  final empty-delta chunk with `finish_reason`, `include_usage` (usage
  `null` on every chunk, then a `choices: []` usage chunk), `obfuscation`
  unless `include_obfuscation: false`, `data: [DONE]`.
- **Anthropic** (`src/anthropic.rs`): `anthropic-version` and `max_tokens`
  required, no `system` role, temperature in 0..=1, non-empty content,
  named SSE events with `ping`, `input_json_delta` (opening with an empty
  one), `thinking` blocks with `signature_delta`, `event: error` mid-stream,
  `request_id` in error bodies, 529 `overloaded_error`.

### Per-model rules

One table per provider, `src/openai_models.rs` and
`src/anthropic_models.rs`. A model id takes the row with the longest
matching prefix; an unknown id gets no per-model rules. Every row cites its
source in a comment. Only rules confirmed in the docs are in; where the docs
give no error wording, the wording is the one commonly seen from the API.

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

### Quirks of OpenAI-compatible providers

| Quirk | Behaviour | Source |
|---|---|---|
| `usage_every_chunk()` | every chunk carries running `usage` | [vLLM `continuous_usage_stats`](https://docs.vllm.ai/en/v0.8.5/serving/openai_compatible_server.html) |
| `no_done()` | the stream ends without `data: [DONE]` | seen in production; no public doc |
| `reasoning_content()` | `Answer::reasoning` streams as `delta.reasoning_content` before the content, and is a field of the whole message | [DeepSeek API](https://api-docs.deepseek.com/api/create-chat-completion) |

## Adding a rule or a quirk

- **A per-model rule**: add a field to `Model` in `src/<provider>_models.rs`
  with its default in `BASE`, set it on the rows it applies to with the
  source URL in a comment, and check it in `check_model` in
  `src/<provider>.rs`. A new model family is one row. Add a test in
  `tests/<provider>.rs` that the rule rejects, and that a model without it
  accepts.
- **A quirk of an OpenAI-compatible provider**: add a flag and a builder
  method to `Quirks` in `src/openai.rs`, with the provider's doc in its doc
  comment, act on it in `respond`, and add one test.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.
