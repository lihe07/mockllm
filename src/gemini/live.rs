//! Gemini Live over WebSocket: `GET /ws/google.ai.generativelanguage.
//! v1beta.GenerativeService.BidiGenerateContent?key=...` upgraded, JSON
//! messages both ways.
//! Reference: https://ai.google.dev/api/live
//! Guide: https://ai.google.dev/gemini-api/docs/live-api/get-started-websocket
//! and https://ai.google.dev/gemini-api/docs/live-api/capabilities
//!
//! Modelled: `setup` first, answered with `setupComplete`; a response,
//! taking the next answer, to `clientContent` with `turnComplete: true`,
//! to `realtimeInput` `text`, `activityEnd` or `audioStreamEnd`, and to
//! `toolResponse`; other messages are taken without a reply. A response
//! is `serverContent` messages: `modelTurn` parts, `generationComplete`,
//! then `turnComplete` with `usageMetadata`. Errors close the connection.
//! The docs give no close codes or reasons; those here (1007 for an
//! invalid message, 1008 and 1011 for scripted errors) are the ones
//! commonly seen.

use base64::Engine;
use serde_json::{Value, json};

use super::Gemini;
use crate::util::{next_id, tokens, words};
use crate::{Answer, Endpoint, Rejection, Reply, Request, Session, Turn};

const PATH: &str = "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent";

/// "The JSON object must have exactly one of the fields from the
/// following object set" (reference, "Send messages").
const MESSAGES: &[&str] = &["setup", "clientContent", "realtimeInput", "toolResponse"];

/// "Audio output always uses a sample rate of 24kHz" (capabilities,
/// "Audio formats").
const PCM: &str = "audio/pcm;rate=24000";

/// Commonly seen reason for an invalid message.
const INVALID: &str = "Request contains an invalid argument.";

pub(crate) struct Live;

impl Endpoint<Gemini> for Live {
    fn route(&self) -> (&'static str, &'static str) {
        ("GET", PATH)
    }

    fn validate(&self, _: &Gemini, _: &Request) -> Result<(), Rejection> {
        Ok(())
    }

    fn session(&self, gemini: &Gemini, _: &Request) -> Option<Box<dyn Session>> {
        Some(Box::new(Bidi {
            gemini: *gemini,
            setup: None,
            context: Vec::new(),
            close: None,
        }))
    }
}

struct Bidi {
    gemini: Gemini,
    /// The `setup` message, once sent.
    setup: Option<Value>,
    /// The client's messages so far, for usage.
    context: Vec<Value>,
    /// The close frame to end the connection with after this turn.
    close: Option<(u16, String)>,
}

impl Session for Bidi {
    fn event(&mut self, message: Value, next: &mut dyn FnMut() -> Answer) -> Turn {
        let Some(fields) = message.as_object() else {
            return self.closing(1007, INVALID);
        };
        if let Some(k) = fields.keys().find(|k| !MESSAGES.contains(&k.as_str())) {
            let reason =
                format!("Invalid JSON payload received. Unknown name \"{k}\": Cannot find field.");
            return self.closing(1007, &reason);
        }
        if fields.len() != 1 {
            return self.closing(1007, INVALID);
        }
        let (kind, body) = fields
            .iter()
            .next()
            .map_or(("", &Value::Null), |(k, v)| (k.as_str(), v));
        // `setup`: "to be sent in the first (and only in the first)" message.
        match (kind, &self.setup) {
            ("setup", None) => {
                // `model`: "Required."
                if body["model"].as_str().is_none_or(str::is_empty) {
                    return self.closing(1007, INVALID);
                }
                self.setup = Some(body.clone());
                // "Sent in response to a BidiGenerateContentSetup message".
                return Turn::now(vec![json!({ "setupComplete": {} })]);
            }
            ("setup", Some(_)) | (_, None) => return self.closing(1007, INVALID),
            _ => {}
        }
        self.context.push(message.clone());
        let respond = match kind {
            // `turnComplete`: "the server content generation should start
            // with the currently accumulated prompt".
            "clientContent" => body["turnComplete"] == true,
            "realtimeInput" => {
                body["text"].is_string()
                    || body.get("activityEnd").is_some()
                    || body["audioStreamEnd"] == true
            }
            // The model goes on once the function responses are in.
            _ => kind == "toolResponse",
        };
        match respond {
            true => self.respond(next()),
            false => Turn::now(Vec::new()),
        }
    }

    fn close(&mut self) -> Option<(u16, String)> {
        self.close.take()
    }
}

impl Bidi {
    /// Nothing more, then the connection closes with `code`. A close
    /// reason is at most 123 bytes (RFC 6455, 5.5).
    fn closing(&mut self, code: u16, reason: &str) -> Turn {
        let mut end = reason.len().min(123);
        while !reason.is_char_boundary(end) {
            end -= 1;
        }
        self.close = Some((code, reason[..end].to_owned()));
        Turn::now(Vec::new())
    }

    fn respond(&mut self, answer: Answer) -> Turn {
        if let Some(status) = answer.status {
            let code = if status >= 500 { 1011 } else { 1008 };
            let reason = crate::core::scripted(&self.gemini, &answer, status).message;
            let turn = self.closing(code, &reason);
            return Turn {
                answer: Some(answer),
                ..turn
            };
        }
        let setup = self.setup.clone().unwrap_or_default();
        // `responseModalities`: in `generationConfig` by the reference, at
        // the top of `setup` in the WebSocket guide's example. Audio is the
        // Live output modality ("Technical specifications").
        let modalities = [
            &setup["generationConfig"]["responseModalities"],
            &setup["responseModalities"],
        ];
        let text = modalities
            .iter()
            .filter_map(|m| m.as_array())
            .flatten()
            .any(|m| m.as_str().is_some_and(|m| m.eq_ignore_ascii_case("text")));
        let b64 = |b: &[u8]| base64::engine::general_purpose::STANDARD.encode(b);
        let audio = |b: &[u8]| json!({ "inlineData": { "mimeType": PCM, "data": b64(b) } });
        let content =
            |parts: Vec<Value>| json!({ "serverContent": { "modelTurn": { "parts": parts } } });
        let (mut events, output, modality) = match (&answer.reply, answer.reply.text()) {
            (_, Some(t)) if text => {
                let pieces = words(t);
                let events = pieces.iter().map(|w| content(vec![json!({ "text": w })]));
                (events.collect::<Vec<_>>(), pieces.len(), "TEXT")
            }
            (_, Some(t)) => {
                // Audio for the text: 10 ms of silence per word, and the
                // word as `outputTranscription` when it is set up.
                let transcribe = setup.get("outputAudioTranscription").is_some();
                let pieces = words(t);
                let events = pieces.iter().map(|w| {
                    let mut e = content(vec![audio(&[0; 480])]);
                    if transcribe {
                        e["serverContent"]["outputTranscription"] = json!({ "text": w });
                    }
                    e
                });
                (events.collect(), pieces.len(), "AUDIO")
            }
            (Reply::Audio(bytes), _) => {
                // 100 ms of 24 kHz 16-bit audio per message.
                let chunks = bytes.chunks(4800).map(|c| content(vec![audio(c)]));
                (chunks.collect(), bytes.len().div_ceil(4800).max(1), "AUDIO")
            }
            // `toolCall`: "Request for the client to execute the
            // functionCalls and return the responses with the matching ids."
            (Reply::ToolCall { name, args }, _) => {
                let call =
                    json!({ "id": format!("fc_fake{}", next_id()), "name": name, "args": args });
                let e = json!({ "toolCall": { "functionCalls": [call] } });
                return self.play(vec![e], answer);
            }
            (Reply::Events(events), _) => return self.play(events.clone(), answer),
            (other, _) => {
                let reason = format!("mockllm: live cannot answer {}", other.kind());
                let turn = self.closing(1011, &reason);
                return Turn {
                    answer: Some(answer),
                    ..turn
                };
            }
        };
        let prompt = tokens(&Value::Array(self.context.clone()));
        let output = output as u64;
        let usage = json!({
            "promptTokenCount": prompt, "responseTokenCount": output, "totalTokenCount": prompt + output,
            "promptTokensDetails": [{ "modality": "TEXT", "tokenCount": prompt }],
            "responseTokensDetails": [{ "modality": modality, "tokenCount": output }]
        });
        // "The last output transcription of this turn is sent before either
        // generationComplete or interrupted, which in turn are followed by
        // turnComplete" (`outputTranscription`). The usage rides on the
        // `turnComplete` message, as commonly seen.
        events.push(json!({ "serverContent": { "generationComplete": true } }));
        events.push(json!({ "serverContent": { "turnComplete": true }, "usageMetadata": usage }));
        self.play(events, answer)
    }

    /// `events` under `answer`; `error_after` closes the connection with
    /// 1011 after `n` events, the commonly seen internal error.
    fn play(&mut self, mut events: Vec<Value>, mut answer: Answer) -> Turn {
        if let Some(n) = answer.error_after.take() {
            events.truncate(n);
            self.closing(1011, "Internal error encountered.");
        }
        Turn {
            events,
            answer: Some(answer),
            ..Turn::default()
        }
    }
}
