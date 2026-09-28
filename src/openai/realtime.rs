//! OpenAI Realtime over WebSocket, a minimal session:
//! `GET /v1/realtime?model=...` upgraded, JSON events both ways.
//! https://developers.openai.com/api/docs/guides/realtime-websocket
//! Events: https://developers.openai.com/api/reference/resources/realtime/client-events
//! and https://developers.openai.com/api/reference/resources/realtime/server-events
//!
//! Modelled: `session.created` on connect; `session.update` answered with
//! `session.updated`; `response.create` answered with a text response
//! (the `output_text` events with output modality `text`, the
//! `output_audio_transcript` events and no audio bytes with `audio`, the
//! default); an `error` event for a client event without `type`. Other
//! client events are accepted without a reply.

use serde_json::{Value, json};

use super::OpenAi;
use crate::util::{next_id, tokens, unix_now, words};
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
        let error = event(error("server_error", None, &openai.scripted(500).message));
        Some(Box::new(Live {
            openai: *openai,
            session,
            error,
            context: Vec::new(),
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
}

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

/// An `error` event's fields.
fn error(kind: &str, code: Option<&str>, message: &str) -> Value {
    let error =
        json!({ "type": kind, "code": code, "message": message, "param": null, "event_id": null });
    json!({ "type": "error", "error": error })
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
            None => {
                let e = error(
                    "invalid_request_error",
                    Some("invalid_event"),
                    "The 'type' field is missing.",
                );
                Turn::now(vec![event(e)])
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
            Some("response.create") => self.respond(&client, next()),
            Some(_) => Turn::now(Vec::new()),
        }
    }
}

impl Live {
    fn respond(&self, client: &Value, answer: Answer) -> Turn {
        let events = match (answer.status, answer.reply.text(), &answer.reply) {
            (Some(status), _, _) => {
                let r = self.openai.scripted(status);
                let server = status >= 500;
                let kind = if server {
                    "server_error"
                } else {
                    "invalid_request_error"
                };
                vec![event(error(kind, r.kind, &r.message))]
            }
            (_, Some(text), _) => self.text(client, text),
            (_, _, Reply::Events(events)) => events.clone(),
            (_, _, other) => {
                let message = format!("mockllm: realtime cannot answer {}", other.kind());
                vec![event(error("server_error", None, &message))]
            }
        };
        let error = self.error.clone();
        Turn {
            events,
            error,
            answer: Some(answer),
        }
    }

    /// The events of a text response.
    fn text(&self, client: &Value, text: &str) -> Vec<Value> {
        let modalities = &client["response"]["output_modalities"];
        let modalities = if modalities.is_array() {
            modalities
        } else {
            &self.session["output_modalities"]
        };
        let audio = !modalities
            .as_array()
            .is_some_and(|m| m.iter().any(|m| m == "text"));
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
        let n = next_id();
        let (response_id, item_id) = (format!("resp_fake{n}"), format!("item_fake{n}"));
        let ids = json!({ "response_id": response_id, "item_id": item_id, "output_index": 0, "content_index": 0 });
        let with = |fields: Value| event(merge(ids.clone(), fields));
        let mut response = json!({
            "object": "realtime.response", "id": response_id, "status": "in_progress",
            "status_details": null, "output": [], "conversation_id": format!("conv_fake{n}"),
            "output_modalities": modalities, "max_output_tokens": "inf", "usage": null, "metadata": null
        });
        let item = |status: &str, content: Value| json!({ "id": item_id, "object": "realtime.item", "type": "message", "status": status, "role": "assistant", "content": content });
        let pieces = words(text);
        let mut events = vec![
            event(json!({ "type": "response.created", "response": response })),
            event(
                json!({ "type": "response.output_item.added", "response_id": response_id, "output_index": 0, "item": item("in_progress", json!([])) }),
            ),
            with(
                json!({ "type": "response.content_part.added", "part": { "type": part, key: "" } }),
            ),
        ];
        events.extend(
            pieces
                .iter()
                .map(|p| with(json!({ "type": delta, "delta": p }))),
        );
        let finished = item("completed", json!([{ "type": content, key: text }]));
        events.extend([
            with(json!({ "type": done, key: text })),
            with(json!({ "type": "response.content_part.done", "part": { "type": part, key: text } })),
            event(json!({ "type": "response.output_item.done", "response_id": response_id, "output_index": 0, "item": finished })),
        ]);
        let input = tokens(&Value::Array(self.context.clone()));
        let output = pieces.len() as u64;
        let usage = json!({
            "total_tokens": input + output, "input_tokens": input, "output_tokens": output,
            "input_token_details": { "text_tokens": input, "audio_tokens": 0, "image_tokens": 0, "cached_tokens": 0, "cached_tokens_details": { "text_tokens": 0, "audio_tokens": 0, "image_tokens": 0 } },
            "output_token_details": { "text_tokens": output, "audio_tokens": 0 }
        });
        response["status"] = "completed".into();
        response["output"] = json!([finished]);
        response["usage"] = usage;
        events.push(event(
            json!({ "type": "response.done", "response": response }),
        ));
        events
    }
}
