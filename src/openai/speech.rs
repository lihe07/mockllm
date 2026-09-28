//! OpenAI speech: `POST /v1/audio/speech`.
//! https://developers.openai.com/api/reference/resources/audio/subresources/speech/methods/create
//! The response content types and the `sse` events are from the OpenAPI
//! spec the reference is built from:
//! https://github.com/openai/openai-openapi (`createSpeech`,
//! `SpeechAudioDeltaEvent`, `SpeechAudioDoneEvent`).

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use bytes::Bytes;
use serde_json::{Value, json};

use super::{OpenAi, bad, object};
use crate::core::misfit;
use crate::util::{Row, lookup, tokens};
use crate::{Answer, Ctx, Endpoint, Out, Provider, Rejection, Reply, Request, data};

/// The body parameters of the reference above; any other is rejected.
#[rustfmt::skip]
const KNOWN: &[&str] = &[
    "input", "model", "voice", "instructions", "response_format", "speed", "stream_format",
];

/// "Supported built-in voices are `alloy`, `ash`, `ballad`, `coral`,
/// `echo`, `fable`, `onyx`, `nova`, `sage`, `shimmer`, `verse`, `marin`,
/// and `cedar`" (reference above).
#[rustfmt::skip]
const VOICES: &[&str] = &[
    "alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer",
    "verse", "marin", "cedar",
];

/// Each format and its content type, as the spec lists them (the pairing
/// is the one commonly seen).
const FORMATS: &[(&str, &str)] = &[
    ("mp3", "audio/mpeg"),
    ("opus", "audio/opus"),
    ("aac", "audio/aac"),
    ("flac", "audio/flac"),
    ("wav", "audio/wav"),
    ("pcm", "audio/pcm"),
];

/// Bytes per streamed chunk (or `speech.audio.delta` event). The API
/// always answers with `Transfer-Encoding: chunked` (spec above).
const CHUNK: usize = 1024;

/// Per-model rules, one row per model family.
#[derive(Clone, Copy, Debug)]
struct Model {
    prefix: &'static str,
    /// The voices accepted.
    voices: &'static [&'static str],
    /// `stream_format: "sse"` is accepted.
    sse: bool,
}

impl Row for Model {
    fn prefix(&self) -> &'static str {
        self.prefix
    }
}

const BASE: Model = Model {
    prefix: "",
    voices: VOICES,
    sse: true,
};

#[rustfmt::skip]
const MODELS: &[Model] = &[
    // "`sse` is not supported for `tts-1` or `tts-1-hd`" (reference above);
    // "The `tts-1` and `tts-1-hd` models support a smaller set: `alloy`,
    // `ash`, `coral`, `echo`, `fable`, `onyx`, `nova`, `sage`, and
    // `shimmer`": https://developers.openai.com/api/docs/guides/text-to-speech#voice-options
    Model {
        prefix: "tts-1",
        voices: &["alloy", "ash", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer"],
        sse: false,
    },
];

pub(crate) struct Speech;

impl Endpoint<OpenAi> for Speech {
    fn route(&self) -> (&'static str, &'static str) {
        ("POST", "/v1/audio/speech")
    }

    fn validate(&self, _: &OpenAi, req: &Request) -> Result<(), Rejection> {
        let body = object(req, KNOWN)?;
        let Some(model) = body.get("model").and_then(Value::as_str) else {
            return Err(bad("you must provide a model parameter", None, None));
        };
        let missing = |p: &str| {
            let message = format!("Missing required parameter: '{p}'.");
            bad(message, Some(p), Some("missing_required_parameter"))
        };
        let Some(input) = body.get("input").and_then(Value::as_str) else {
            return Err(missing("input"));
        };
        let Some(voice) = body.get("voice").filter(|v| !v.is_null()) else {
            return Err(missing("voice"));
        };
        // "The maximum length is 4096 characters"; `instructions` has the
        // same `maxLength` in the spec. Wording as commonly seen.
        let instructions = body.get("instructions").and_then(Value::as_str);
        for (p, s) in [("input", Some(input)), ("instructions", instructions)] {
            let n = s.map_or(0, |s| s.chars().count());
            if n > 4096 {
                let message = format!(
                    "Invalid '{p}': string too long. Expected a string with maximum length 4096, but got a string with length {n} instead."
                );
                return Err(bad(message, Some(p), Some("string_above_max_length")));
            }
        }
        let row = lookup(MODELS, BASE, model);
        // A built-in voice name, or "a custom voice object with an `id`".
        let ok = match voice {
            Value::String(v) => row.voices.contains(&v.as_str()),
            Value::Object(o) => o.get("id").is_some_and(Value::is_string),
            _ => false,
        };
        if !ok {
            let supported = quoted(row.voices);
            let shown = voice
                .as_str()
                .map_or(voice.to_string(), |v| format!("'{v}'"));
            let message = format!("Invalid value: {shown}. Supported values are: {supported}.");
            return Err(bad(message, Some("voice"), Some("invalid_value")));
        }
        let format = body.get("response_format").and_then(Value::as_str);
        if let Some(f) = format.filter(|f| !FORMATS.iter().any(|(n, _)| n == f)) {
            let names: Vec<&str> = FORMATS.iter().map(|(n, _)| *n).collect();
            let message = format!(
                "Invalid value: '{f}'. Supported values are: {}.",
                quoted(&names)
            );
            return Err(bad(message, Some("response_format"), Some("invalid_value")));
        }
        // "Select a value from `0.25` to `4.0`." Wording as commonly seen.
        if let Some(s) = body.get("speed").and_then(Value::as_f64) {
            let (code, bound, which) = match s {
                s if s < 0.25 => ("decimal_below_min_value", ">= 0.25", "below minimum"),
                s if s > 4.0 => ("decimal_above_max_value", "<= 4", "above maximum"),
                _ => ("", "", ""),
            };
            if !code.is_empty() {
                let message = format!(
                    "Invalid 'speed': decimal {which} value. Expected a value {bound}, but got {s} instead."
                );
                return Err(bad(message, Some("speed"), Some(code)));
            }
        }
        match body.get("stream_format").and_then(Value::as_str) {
            Some("sse") if !row.sse => {
                // Wording as commonly seen for an unsupported value.
                let message = format!(
                    "Unsupported value: 'stream_format' does not support 'sse' with this model ({model})."
                );
                Err(bad(
                    message,
                    Some("stream_format"),
                    Some("unsupported_value"),
                ))
            }
            Some(f) if !["sse", "audio"].contains(&f) => {
                let message =
                    format!("Invalid value: '{f}'. Supported values are: 'sse' and 'audio'.");
                Err(bad(message, Some("stream_format"), Some("invalid_value")))
            }
            _ => Ok(()),
        }
    }

    fn respond(&self, openai: &OpenAi, req: &Request, answer: &Answer, ctx: &Ctx) -> Out {
        let body = &req.body;
        let sse = body["stream_format"] == "sse";
        // "The default response format is `mp3`":
        // https://developers.openai.com/api/docs/guides/text-to-speech#supported-output-formats
        let format = body["response_format"].as_str().unwrap_or("mp3");
        let error = data(&openai.error(&openai.scripted(500), ctx.id));
        let audio = match &answer.reply {
            Reply::Audio(bytes) => bytes.clone(),
            Reply::Unscripted => silence(format),
            Reply::Events(events) if sse => {
                return Out::Sse(events.iter().map(data).collect(), error);
            }
            other => return misfit("speech", other),
        };
        let chunks: Vec<Bytes> = (0..audio.len())
            .step_by(CHUNK)
            .map(|i| audio.slice(i..audio.len().min(i + CHUNK)))
            .collect();
        if !sse {
            let kind = FORMATS.iter().find(|(n, _)| *n == format);
            let kind = kind.map_or("application/octet-stream", |(_, k)| k);
            return Out::Chunks(kind.into(), chunks);
        }
        // `speech.audio.delta` with "A chunk of Base64-encoded audio data",
        // then `speech.audio.done` with `usage` (spec above). `data:`-only
        // framing, as commonly seen.
        let mut events: Vec<String> = chunks
            .iter()
            .map(|c| data(&json!({ "type": "speech.audio.delta", "audio": STANDARD.encode(c) })))
            .collect();
        let (input, output) = (tokens(&body["input"]), chunks.len() as u64);
        let usage = json!({ "input_tokens": input, "output_tokens": output, "total_tokens": input + output });
        events.push(data(
            &json!({ "type": "speech.audio.done", "usage": usage }),
        ));
        Out::Sse(events, error)
    }
}

/// `'a', 'b', and 'c'`.
fn quoted(names: &[&str]) -> String {
    let q: Vec<String> = names.iter().map(|n| format!("'{n}'")).collect();
    match q.split_last() {
        Some((last, rest)) if !rest.is_empty() => format!("{}, and {last}", rest.join(", ")),
        _ => q.join(""),
    }
}

/// The audio when none is scripted: 0.1 s of silence. PCM is "raw samples
/// in 24kHz (16-bit signed, low-endian), without the header" (TTS guide,
/// supported output formats); WAV is that with a header; MP3 is one silent
/// MPEG-1 Layer III frame. Other formats get the raw samples, which are
/// not a decodable file: script `Answer::audio` for real bytes.
fn silence(format: &str) -> Bytes {
    let pcm = vec![0u8; 4800];
    match format {
        "wav" => {
            let len = pcm.len() as u32;
            let mut wav = Vec::with_capacity(44 + pcm.len());
            wav.extend_from_slice(b"RIFF");
            wav.extend_from_slice(&(36 + len).to_le_bytes());
            wav.extend_from_slice(b"WAVEfmt ");
            // PCM, mono, 24 kHz, 48000 bytes/s, 2-byte blocks, 16 bits.
            for field in [16u32, 0x0001_0001, 24_000, 48_000, 0x0010_0002] {
                wav.extend_from_slice(&field.to_le_bytes());
            }
            wav.extend_from_slice(b"data");
            wav.extend_from_slice(&len.to_le_bytes());
            wav.extend_from_slice(&pcm);
            Bytes::from(wav)
        }
        // 128 kbps, 44.1 kHz: a 417-byte frame; zeroed side info is silence.
        "mp3" => {
            let mut frame = vec![0u8; 417];
            frame[..4].copy_from_slice(&[0xFF, 0xFB, 0x90, 0x00]);
            Bytes::from(frame)
        }
        _ => Bytes::from(pcm),
    }
}
