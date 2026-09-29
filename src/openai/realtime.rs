//! OpenAI Realtime over WebSocket, a minimal session:
//! `GET /v1/realtime?model=...` upgraded, JSON events both ways.
//! https://developers.openai.com/api/docs/guides/realtime-websocket
//! Events: https://developers.openai.com/api/reference/resources/realtime/client-events
//! and https://developers.openai.com/api/reference/resources/realtime/server-events
//!
//! Modelled: `session.created` on connect; `session.update` answered with
//! `session.updated`; `response.create` answered with a response: a text
//! (the `output_text` events with output modality `text`, the
//! `output_audio_transcript` events and no audio bytes with `audio`, the
//! default), audio bytes (`output_audio` deltas) or a function call
//! (`function_call_arguments` deltas); `conversation.item.create`;
//! `input_audio_buffer.append`, `.commit` and `.clear`; `error` events for
//! a client event without a known `type` or that breaks a rule, naming the
//! client's `event_id`. Other client events of the reference are accepted
//! without a reply. Server VAD is not modelled: the buffer is committed only
//! by the client. Where the docs give no error wording, the wording is the
//! one commonly seen from the API, marked "(commonly seen)".

use serde_json::{Value, json};

use super::OpenAi;
use base64::Engine;
use base64::engine::general_purpose::STANDARD;

use crate::util::{next_id, pieces, tokens, unix_now, words};
use crate::{Answer, Ctx, Endpoint, Provider, Rejection, Reply, Request, Session, Turn};

pub(crate) struct Realtime;

impl Endpoint<OpenAi> for Realtime {
    fn route(&self) -> (&'static str, &'static str) {
        ("GET", "/v1/realtime")
    }

    fn validate(&self, _: &OpenAi, _: &Request) -> Result<(), Rejection> {
        Ok(())
    }

    fn session(&self, openai: &OpenAi, req: &Request) -> Option<Box<dyn Session>> {
        let model = req.query_param("model").unwrap_or("gpt-realtime");
        // The `session.created` example of the server events reference.
        let session = json!({
            "type": "realtime",
            "object": "realtime.session",
            "id": format!("sess_fake{}", next_id()),
            "model": model,
            "output_modalities": ["audio"],
            "instructions": "",
            "tools": [],
            "tool_choice": "auto",
            "max_output_tokens": "inf",
            "tracing": null,
            "prompt": null,
            "expires_at": unix_now() + 3600,
            "audio": {
                "input": { "format": { "type": "audio/pcm", "rate": 24000 }, "transcription": null, "noise_reduction": null, "turn_detection": { "type": "server_vad", "threshold": 0.5, "prefix_padding_ms": 300, "silence_duration_ms": 200, "idle_timeout_ms": null, "create_response": true, "interrupt_response": true } },
                "output": { "format": { "type": "audio/pcm", "rate": 24000 }, "voice": "marin", "speed": 1 }
            },
            "include": null
        });
        let message = openai.scripted(500).message;
        let error = event(error("server_error", None, &message, &Value::Null));
        Some(Box::new(Live {
            openai: *openai,
            session,
            error,
            context: Vec::new(),
            buffer: 0,
            last_item: None,
        }))
    }
}

struct Live {
    openai: OpenAi,
    session: Value,
    /// The error event `error_after` sends.
    error: Value,
    /// The text of the conversation so far, for usage.
    context: Vec<Value>,
    /// Bytes of input audio appended and not yet committed.
    buffer: usize,
    /// The id of the conversation's last item, each new item's
    /// `previous_item_id`.
    last_item: Option<String>,
}

/// The client events of the reference; any other `type` is an error.
/// https://developers.openai.com/api/reference/resources/realtime/client-events
#[rustfmt::skip]
const CLIENT_EVENTS: &[&str] = &[
    "session.update", "input_audio_buffer.append", "input_audio_buffer.commit",
    "input_audio_buffer.clear", "conversation.item.create", "conversation.item.retrieve",
    "conversation.item.truncate", "conversation.item.delete", "response.create",
    "response.cancel", "output_audio_buffer.clear",
];

/// The `ConversationItem` types `conversation.item.create` takes (client
/// events reference).
#[rustfmt::skip]
const ITEM_TYPES: &[&str] = &[
    "message", "function_call", "function_call_output", "mcp_approval_response",
    "mcp_list_tools", "mcp_call", "mcp_approval_request",
];

/// Bytes of audio per `response.output_audio.delta`: 100 ms of the
/// default 24 kHz 16-bit PCM.
const AUDIO_CHUNK: usize = 4800;

/// A server event: `fields` with a fresh `event_id`.
fn event(fields: Value) -> Value {
    merge(
        json!({ "event_id": format!("event_fake{}", next_id()) }),
        fields,
    )
}

/// `base` with the fields of `more` added.
fn merge(mut base: Value, more: Value) -> Value {
    if let (Some(b), Value::Object(m)) = (base.as_object_mut(), more) {
        b.extend(m);
    }
    base
}

/// An `error` event's fields: "The event_id of the client event that
/// caused the error, if applicable" (server events reference, `error`).
fn error(kind: &str, code: Option<&str>, message: &str, client: &Value) -> Value {
    let error = json!({ "type": kind, "code": code, "message": message, "param": null, "event_id": client });
    json!({ "type": "error", "error": error })
}

/// An `invalid_request_error` event for `client`.
fn invalid(client: &Value, code: &str, message: &str) -> Turn {
    let e = error(
        "invalid_request_error",
        Some(code),
        message,
        &client["event_id"],
    );
    Turn::now(vec![event(e)])
}

impl Session for Live {
    fn open(&mut self, _: &Ctx) -> Vec<Value> {
        vec![event(
            json!({ "type": "session.created", "session": self.session }),
        )]
    }

    fn event(&mut self, client: Value, next: &mut dyn FnMut() -> Answer) -> Turn {
        self.context.push(client.clone());
        match client["type"].as_str() {
            // The `error` example of the server events reference.
            None => invalid(&client, "invalid_event", "The 'type' field is missing."),
            // Commonly seen wording.
            Some(t) if !CLIENT_EVENTS.contains(&t) => {
                let supported: Vec<String> =
                    CLIENT_EVENTS.iter().map(|e| format!("'{e}'")).collect();
                let message = format!(
                    "Invalid value: '{t}'. Supported values are: {}.",
                    supported.join(", ")
                );
                invalid(&client, "invalid_value", &message)
            }
            Some("session.update") => {
                if let (Some(s), Some(update)) =
                    (self.session.as_object_mut(), client["session"].as_object())
                {
                    s.extend(update.clone());
                }
                Turn::now(vec![event(
                    json!({ "type": "session.updated", "session": self.session }),
                )])
            }
            // "the server will not send a confirmation response to this
            // event" (client events reference).
            Some("input_audio_buffer.append") => {
                let audio = client["audio"].as_str().unwrap_or_default();
                self.buffer += STANDARD.decode(audio).map_or(audio.len(), |a| a.len());
                Turn::now(Vec::new())
            }
            Some("input_audio_buffer.commit") => self.commit(&client),
            // "The server will respond with an `input_audio_buffer.cleared`
            // event."
            Some("input_audio_buffer.clear") => {
                self.buffer = 0;
                Turn::now(vec![event(json!({ "type": "input_audio_buffer.cleared" }))])
            }
            Some("conversation.item.create") => self.create_item(&client),
            Some("response.create") => self.respond(&client, next()),
            Some(_) => Turn::now(Vec::new()),
        }
    }
}

impl Live {
    /// "This event will produce an error if the input audio buffer is
    /// empty"; otherwise `input_audio_buffer.committed` and the user message
    /// item made from the buffer (client events reference, `commit`; server
    /// events reference, `conversation.item.added`).
    fn commit(&mut self, client: &Value) -> Turn {
        if self.buffer == 0 {
            // Commonly seen code and wording.
            let message = "Error committing input audio buffer: buffer too small. Expected at least 100ms of audio, but buffer only has 0.00ms of audio.";
            return invalid(client, "input_audio_buffer_commit_empty", message);
        }
        self.buffer = 0;
        let item_id = format!("item_fake{}", next_id());
        let committed = json!({ "type": "input_audio_buffer.committed", "previous_item_id": self.last_item, "item_id": item_id });
        // Audio data is left out of the item events ("except for audio data").
        let item = json!({ "id": item_id, "object": "realtime.item", "type": "message", "status": "completed", "role": "user", "content": [{ "type": "input_audio", "transcript": null }] });
        let mut events = vec![event(committed)];
        events.extend(self.added(item));
        Turn::now(events)
    }

    /// `conversation.item.added` and `conversation.item.done` for a new
    /// item, which becomes the last.
    fn added(&mut self, item: Value) -> Vec<Value> {
        let previous = self
            .last_item
            .replace(item["id"].as_str().unwrap_or_default().to_owned());
        ["conversation.item.added", "conversation.item.done"]
            .map(|t| event(json!({ "type": t, "previous_item_id": previous, "item": item })))
            .into()
    }

    /// "If successful, the server will emit a `conversation.item.added`
    /// event and, when the item is finalized, a `conversation.item.done`
    /// event. Otherwise, an `error` event will be sent." It "cannot
    /// populate assistant audio messages" (client events reference).
    fn create_item(&mut self, client: &Value) -> Turn {
        let Some(item) = client["item"].as_object() else {
            // Commonly seen wording.
            return invalid(
                client,
                "missing_required_parameter",
                "Missing required parameter: 'item'.",
            );
        };
        let kind = item.get("type").and_then(Value::as_str).unwrap_or_default();
        if !ITEM_TYPES.contains(&kind) {
            // Commonly seen wording.
            let message = format!(
                "Invalid value: '{kind}'. Supported values are: {}.",
                ITEM_TYPES.join(", ")
            );
            return invalid(client, "invalid_value", &message);
        }
        let content = item.get("content").and_then(Value::as_array);
        let audio = content.is_some_and(|c| {
            c.iter()
                .any(|p| p["type"] == "output_audio" || p["type"] == "audio")
        });
        if item.get("role").is_some_and(|r| r == "assistant") && audio {
            // Commonly seen wording.
            let message = "Assistant messages cannot contain audio content.";
            return invalid(client, "invalid_value", message);
        }
        let mut item = Value::Object(item.clone());
        let id = item["id"].as_str().map(str::to_owned);
        item["id"] = id
            .unwrap_or_else(|| format!("item_fake{}", next_id()))
            .into();
        item["object"] = "realtime.item".into();
        item["status"] = "completed".into();
        Turn::now(self.added(item))
    }

    fn respond(&mut self, client: &Value, answer: Answer) -> Turn {
        let events = match (answer.status, &answer.reply) {
            (Some(status), _) => {
                let r = crate::core::scripted(&self.openai, &answer, status);
                let server = status >= 500;
                let kind = if server {
                    "server_error"
                } else {
                    "invalid_request_error"
                };
                vec![event(error(kind, r.kind, &r.message, &client["event_id"]))]
            }
            (_, Reply::Events(events)) => events.clone(),
            (_, reply) => self.output(client, reply).unwrap_or_else(|| {
                let message = format!("mockllm: realtime cannot answer {}", reply.kind());
                vec![event(error("server_error", None, &message, &Value::Null))]
            }),
        };
        let error = self.error.clone();
        Turn {
            events,
            error,
            answer: Some(answer),
        }
    }

    /// The events of a response of one output item: a text (or, in audio,
    /// its transcript), audio bytes, or a function call. `None` for what a
    /// response cannot hold.
    fn output(&mut self, client: &Value, reply: &Reply) -> Option<Vec<Value>> {
        let modalities = &client["response"]["output_modalities"];
        let modalities = if modalities.is_array() {
            modalities
        } else {
            &self.session["output_modalities"]
        };
        let audio = !modalities
            .as_array()
            .is_some_and(|m| m.iter().any(|m| m == "text"));
        let n = next_id();
        let (response_id, item_id) = (format!("resp_fake{n}"), format!("item_fake{n}"));
        let ids = json!({ "response_id": response_id, "item_id": item_id, "output_index": 0 });
        let with = |fields: Value| event(merge(ids.clone(), fields));
        let message = |status: &str, content: Value| json!({ "id": item_id, "object": "realtime.item", "type": "message", "status": status, "role": "assistant", "content": content });
        // The item as added and as done, the events between, and the output
        // (text, audio) tokens.
        let (added, middle, finished, text_tokens, audio_tokens) = match (reply.text(), reply) {
            (Some(text), _) => {
                // Text parts and events, or an audio part's transcript.
                let (part, delta, done, content, key) = match audio {
                    false => (
                        "text",
                        "response.output_text.delta",
                        "response.output_text.done",
                        "output_text",
                        "text",
                    ),
                    true => (
                        "audio",
                        "response.output_audio_transcript.delta",
                        "response.output_audio_transcript.done",
                        "output_audio",
                        "transcript",
                    ),
                };
                let pieces = words(text);
                let part_event = |t: &str, v: &str| {
                    with(json!({ "type": t, "content_index": 0, "part": { "type": part, key: v } }))
                };
                let mut middle = vec![part_event("response.content_part.added", "")];
                middle.extend(
                    pieces
                        .iter()
                        .map(|p| with(json!({ "type": delta, "content_index": 0, "delta": p }))),
                );
                middle.push(with(json!({ "type": done, "content_index": 0, key: text })));
                middle.push(part_event("response.content_part.done", text));
                let finished = message("completed", json!([{ "type": content, key: text }]));
                (
                    message("in_progress", json!([])),
                    middle,
                    finished,
                    pieces.len(),
                    0,
                )
            }
            // "Base64-encoded audio data delta" per
            // `response.output_audio.delta`, then `response.output_audio.done`.
            (_, Reply::Audio(bytes)) if audio => {
                let part = |t: &str| {
                    with(
                        json!({ "type": t, "content_index": 0, "part": { "type": "audio", "transcript": "" } }),
                    )
                };
                let mut middle = vec![part("response.content_part.added")];
                let chunks: Vec<&[u8]> = bytes.chunks(AUDIO_CHUNK).collect();
                middle.extend(chunks.iter().map(|c| with(json!({ "type": "response.output_audio.delta", "content_index": 0, "delta": STANDARD.encode(c) }))));
                middle.push(with(
                    json!({ "type": "response.output_audio.done", "content_index": 0 }),
                ));
                middle.push(part("response.content_part.done"));
                let finished = message(
                    "completed",
                    json!([{ "type": "output_audio", "transcript": "" }]),
                );
                (
                    message("in_progress", json!([])),
                    middle,
                    finished,
                    0,
                    chunks.len(),
                )
            }
            // `response.function_call_arguments.delta` and `.done` with
            // `call_id` and `name`, on a `function_call` item.
            (_, Reply::ToolCall { name, args }) => {
                let (args, call_id) = (args.to_string(), format!("call_fake{n}"));
                let item = |status: &str, arguments: &str| json!({ "id": item_id, "object": "realtime.item", "type": "function_call", "status": status, "name": name, "call_id": call_id, "arguments": arguments });
                let deltas = pieces(&args, 8);
                let mut middle: Vec<Value> = deltas.iter().map(|d| with(json!({ "type": "response.function_call_arguments.delta", "call_id": call_id, "delta": d }))).collect();
                middle.push(with(json!({ "type": "response.function_call_arguments.done", "call_id": call_id, "name": name, "arguments": args })));
                (
                    item("in_progress", ""),
                    middle,
                    item("completed", &args),
                    deltas.len(),
                    0,
                )
            }
            _ => return None,
        };
        let mut response = json!({
            "object": "realtime.response", "id": response_id, "status": "in_progress",
            "status_details": null, "output": [], "conversation_id": format!("conv_fake{n}"),
            "output_modalities": modalities, "max_output_tokens": "inf", "usage": null, "metadata": null
        });
        let mut events = vec![
            event(json!({ "type": "response.created", "response": response })),
            event(
                json!({ "type": "response.output_item.added", "response_id": response_id, "output_index": 0, "item": added }),
            ),
        ];
        // "the `conversation.item.added` event will be sent when the model
        // starts generating a specific Item" (server events reference).
        let [item_added, item_done]: [Value; 2] = self.added(added).try_into().ok()?;
        events.push(item_added);
        events.extend(middle);
        events.push(event(json!({ "type": "response.output_item.done", "response_id": response_id, "output_index": 0, "item": finished })));
        events.push(merge(item_done, json!({ "item": finished })));
        let input = tokens(&Value::Array(self.context.clone()));
        let (text, audio) = (text_tokens as u64, audio_tokens as u64);
        let output = text + audio;
        let usage = json!({
            "total_tokens": input + output, "input_tokens": input, "output_tokens": output,
            "input_token_details": { "text_tokens": input, "audio_tokens": 0, "image_tokens": 0, "cached_tokens": 0, "cached_tokens_details": { "text_tokens": 0, "audio_tokens": 0, "image_tokens": 0 } },
            "output_token_details": { "text_tokens": text, "audio_tokens": audio }
        });
        response["status"] = "completed".into();
        response["output"] = json!([finished]);
        response["usage"] = usage;
        events.push(event(
            json!({ "type": "response.done", "response": response }),
        ));
        Some(events)
    }
}
