//! OpenAI transcriptions: `POST /v1/audio/transcriptions`, multipart.
//! https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create
//! Stream events:
//! https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/streaming-events
//! Guide: https://developers.openai.com/api/docs/guides/speech-to-text

use crate::transport::framed;
use serde_json::{Map, Value, json};

use super::{OpenAi, bad, object};
use crate::core::misfit;
use crate::util::{Row, lookup, words};
use crate::{Answer, Ctx, Endpoint, Out, Provider, Rejection, Reply, Request, data};

/// The form fields of the reference above, array ones also with the `[]`
/// the SDKs append; any other is rejected. `file` is a file part.
#[rustfmt::skip]
const KNOWN: &[&str] = &[
    "file", "model", "language", "languages", "languages[]", "keywords", "keywords[]",
    "prompt", "response_format", "temperature", "include", "include[]",
    "timestamp_granularities", "timestamp_granularities[]", "stream", "chunking_strategy",
    "known_speaker_names", "known_speaker_names[]", "known_speaker_references",
    "known_speaker_references[]",
];

const FORMATS: &[&str] = &[
    "json",
    "text",
    "srt",
    "verbose_json",
    "vtt",
    "diarized_json",
];

/// "in one of these formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or
/// webm" (reference above).
const AUDIO: &[&str] = &[
    "flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm",
];

/// "Files can be up to 25 MB" (guide); larger is a 413, "The audio request
/// exceeds the supported size limit" (the OpenAPI spec's responses).
const MAX_FILE: usize = 25 * 1024 * 1024;

/// Seconds of audio per transcribed word, for timestamps and durations.
const WORD_SECS: f64 = 0.5;

/// Per-model rules, one row per model.
#[derive(Clone, Copy, Debug)]
struct Model {
    prefix: &'static str,
    /// The `response_format`s accepted; empty means any.
    formats: &'static [&'static str],
    /// Streams with `stream: true` (otherwise it is ignored).
    streams: bool,
    /// Accepts `timestamp_granularities`.
    granularities: bool,
    /// Accepts `include[]=logprobs`.
    logprobs: bool,
    /// Accepts `prompt`.
    prompt: bool,
    /// Returns detected `languages`.
    languages: bool,
    /// Billed by duration, not tokens.
    duration: bool,
}

impl Row for Model {
    fn prefix(&self) -> &'static str {
        self.prefix
    }
}

const BASE: Model = Model {
    prefix: "",
    formats: &[],
    streams: true,
    granularities: true,
    logprobs: true,
    prompt: true,
    languages: false,
    duration: false,
};

#[rustfmt::skip]
const MODELS: &[Model] = &[
    // "Streaming is not supported for the `whisper-1` model and will be
    // ignored"; its formats are the translation endpoint's, which is
    // whisper-only (reference above); "The `timestamp_granularities[]`
    // parameter is only supported for `whisper-1`" (guide); billed by
    // duration (the verbose_json usage).
    Model {
        prefix: "whisper-1", formats: &["json", "text", "srt", "verbose_json", "vtt"],
        streams: false, logprobs: false, duration: true, ..BASE
    },
    // "For `gpt-4o-transcribe` and `gpt-4o-mini-transcribe`, the only
    // supported format is `json`" (reference, `response_format`).
    Model { prefix: "gpt-4o-transcribe", formats: &["json"], granularities: false, ..BASE },
    Model { prefix: "gpt-4o-mini-transcribe", formats: &["json"], granularities: false, ..BASE },
    // "For `gpt-4o-transcribe-diarize`, the supported formats are `json`,
    // `text`, and `diarized_json`"; `prompt`, `include` and timestamp
    // granularities are not supported (reference above).
    Model {
        prefix: "gpt-4o-transcribe-diarize", formats: &["json", "text", "diarized_json"],
        granularities: false, logprobs: false, prompt: false, ..BASE
    },
    // `languages` are "Returned by `gpt-transcribe`"; logprobs only with
    // the gpt-4o models (reference above).
    Model { prefix: "gpt-transcribe", granularities: false, logprobs: false, languages: true, ..BASE },
];

pub(crate) struct Transcriptions;

impl Endpoint<OpenAi> for Transcriptions {
    fn route(&self) -> (&'static str, &'static str) {
        ("POST", "/v1/audio/transcriptions")
    }

    fn validate(&self, _: &OpenAi, req: &Request) -> Result<(), Rejection> {
        let body = object(req, KNOWN)?;
        let missing = |p: &str| {
            let message = format!("Missing required parameter: '{p}'.");
            bad(message, Some(p), Some("missing_required_parameter"))
        };
        let Some(file) = req.files.iter().find(|f| f.field == "file") else {
            return Err(missing("file"));
        };
        let Some(model) = body.get("model").and_then(Value::as_str) else {
            return Err(missing("model"));
        };
        if file.bytes.len() > MAX_FILE {
            // Wording as commonly seen.
            let message = format!(
                "Maximum content size limit ({MAX_FILE}) exceeded ({} bytes read)",
                file.bytes.len()
            );
            return Err(Rejection::new(413, message));
        }
        let ext = file
            .filename
            .rsplit_once('.')
            .map(|(_, e)| e.to_lowercase());
        if !ext.is_some_and(|e| AUDIO.contains(&e.as_str())) {
            // Wording as commonly seen.
            let message = format!("Invalid file format. Supported formats: {AUDIO:?}");
            return Err(bad(message, Some("file"), None));
        }
        let row = lookup(MODELS, BASE, model);
        let format = field(body, "response_format").unwrap_or("json");
        if !FORMATS.contains(&format) {
            let message = format!(
                "Invalid value: '{format}'. Supported values are: 'json', 'text', 'srt', 'verbose_json', 'vtt', and 'diarized_json'."
            );
            return Err(bad(message, Some("response_format"), Some("invalid_value")));
        }
        if !row.formats.is_empty() && !row.formats.contains(&format) {
            // Wording as commonly seen.
            let message =
                format!("response_format '{format}' is not compatible with model '{model}'.");
            return Err(bad(
                message,
                Some("response_format"),
                Some("unsupported_value"),
            ));
        }
        let unsupported = |p: &str| {
            let message = format!("Unsupported parameter: '{p}' is not supported with this model.");
            bad(message, Some(p), Some("unsupported_parameter"))
        };
        let granular = !values(req, "timestamp_granularities").is_empty();
        if granular && !row.granularities {
            return Err(unsupported("timestamp_granularities"));
        }
        // "`response_format` must be set `verbose_json` to use timestamp
        // granularities" (reference above). Wording as commonly seen.
        if granular && format != "verbose_json" {
            let message =
                "timestamp_granularities is only supported with response_format 'verbose_json'.";
            return Err(bad(message, Some("timestamp_granularities"), None));
        }
        if !values(req, "include").is_empty() && !row.logprobs {
            return Err(unsupported("include"));
        }
        // "`logprobs` only works with response_format set to `json`".
        if logprobs(req) && format != "json" {
            let message = "logprobs is only supported with response_format 'json'.";
            return Err(bad(message, Some("include"), None));
        }
        if field(body, "prompt").is_some() && !row.prompt {
            return Err(unsupported("prompt"));
        }
        // "The sampling temperature, between 0 and 1."
        let t = field(body, "temperature").and_then(|t| t.parse::<f64>().ok());
        if let Some(t) = t.filter(|t| !(0.0..=1.0).contains(t)) {
            let (which, code, bound) = match t < 0.0 {
                true => ("below minimum", "decimal_below_min_value", ">= 0"),
                false => ("above maximum", "decimal_above_max_value", "<= 1"),
            };
            let message = format!(
                "Invalid 'temperature': decimal {which} value. Expected a value {bound}, but got {t} instead."
            );
            return Err(bad(message, Some("temperature"), Some(code)));
        }
        Ok(())
    }

    fn respond(&self, openai: &OpenAi, req: &Request, answer: &Answer, ctx: &Ctx) -> Out {
        let body = req.object().cloned().unwrap_or_default();
        let model = field(&body, "model").unwrap_or_default();
        let row = lookup(MODELS, BASE, model);
        let format = field(&body, "response_format").unwrap_or("json");
        let stream = field(&body, "stream") == Some("true") && row.streams;
        let error = data(&openai.error(&openai.scripted(500), ctx.id));
        let text = match &answer.reply {
            Reply::Transcript(t) | Reply::Text(t) => t.as_str(),
            Reply::Unscripted => crate::answer::UNSCRIPTED,
            Reply::Events(events) if stream => {
                return Out::Sse(events.iter().map(|e| framed(e, data)).collect(), error);
            }
            other => return misfit("transcriptions", other),
        };
        let pieces = words(text);
        let secs = (pieces.len().max(1) as f64) * WORD_SECS;
        let audio_bytes = req
            .files
            .iter()
            .find(|f| f.field == "file")
            .map_or(0, |f| f.bytes.len());
        let (input, output) = ((audio_bytes as u64 / 1024).max(1), pieces.len() as u64);
        let usage = match row.duration {
            true => json!({ "type": "duration", "seconds": secs.ceil() as u64 }),
            false => json!({
                "type": "tokens", "input_tokens": input,
                "input_token_details": { "text_tokens": 0, "audio_tokens": input },
                "output_tokens": output, "total_tokens": input + output
            }),
        };
        let prob = |p: &str| json!({ "token": p, "logprob": 0.0, "bytes": p.as_bytes() });
        let all_probs = || Value::from(pieces.iter().map(|p| prob(p)).collect::<Vec<_>>());
        let diarized = format == "diarized_json";
        let segment = json!({
            "type": "transcript.text.segment", "id": "seg_001",
            "start": 0.0, "end": secs, "text": text.trim(), "speaker": "A"
        });
        let language = field(&body, "language").unwrap_or("en");
        if stream {
            // `transcript.text.delta`s, a `transcript.text.segment` per
            // finished segment when diarized (with `segment_id` on the
            // deltas), then `transcript.text.done` with usage. `data:`-only
            // framing, as commonly seen.
            let mut events: Vec<String> = pieces
                .iter()
                .map(|p| {
                    let mut e = json!({ "type": "transcript.text.delta", "delta": p });
                    if logprobs(req) {
                        e["logprobs"] = json!([prob(p)]);
                    }
                    if diarized {
                        e["segment_id"] = json!("seg_001");
                    }
                    data(&e)
                })
                .collect();
            if diarized {
                events.push(data(&segment));
            }
            let mut done = json!({ "type": "transcript.text.done", "text": text, "usage": usage });
            if logprobs(req) {
                done["logprobs"] = all_probs();
            }
            if row.languages {
                done["languages"] = json!([{ "code": language }]);
            }
            events.push(data(&done));
            return Out::Sse(events, error);
        }
        let plain = |s: String| Out::Chunks("text/plain; charset=utf-8".into(), vec![s.into()]);
        match format {
            // Plain text, as commonly seen: the text and a newline.
            "text" => plain(format!("{text}\n")),
            // One cue for the whole transcript.
            "srt" => plain(format!(
                "1\n00:00:00,000 --> {}\n{text}\n\n",
                stamp(secs, ',')
            )),
            "vtt" => plain(format!(
                "WEBVTT\n\n00:00:00.000 --> {}\n{text}\n\n",
                stamp(secs, '.')
            )),
            "verbose_json" => {
                // `segments` unless only word timestamps are asked for; the
                // default granularity is `segment` (reference above).
                let asked = values(req, "timestamp_granularities");
                let segments = asked.is_empty() || asked.contains(&"segment");
                let mut v = json!({ "task": "transcribe", "language": language, "duration": secs, "text": text });
                if segments {
                    v["segments"] = json!([{
                        "id": 0, "seek": 0, "start": 0.0, "end": secs, "text": text,
                        "tokens": [], "temperature": 0.0, "avg_logprob": 0.0,
                        "compression_ratio": 1.0, "no_speech_prob": 0.0
                    }]);
                }
                if asked.contains(&"word") {
                    let at = |i: usize| i as f64 * WORD_SECS;
                    let w: Vec<Value> = pieces
                        .iter()
                        .enumerate()
                        .map(|(i, p)| json!({ "word": p.trim(), "start": at(i), "end": at(i + 1) }))
                        .collect();
                    v["words"] = w.into();
                }
                v["usage"] = json!({ "type": "duration", "seconds": secs.ceil() as u64 });
                Out::Json(200, v)
            }
            _ if diarized => Out::Json(
                200,
                json!({
                    "task": "transcribe", "duration": secs, "text": text,
                    "segments": [segment], "usage": usage
                }),
            ),
            _ => {
                let mut v = json!({ "text": text, "usage": usage });
                if logprobs(req) {
                    v["logprobs"] = all_probs();
                }
                if row.languages {
                    v["languages"] = json!([{ "code": language }]);
                }
                Out::Json(200, v)
            }
        }
    }
}

/// A text field, with or without the SDKs' `[]`.
fn field<'a>(body: &'a Map<String, Value>, name: &str) -> Option<&'a str> {
    let v = body.get(name).or_else(|| body.get(&format!("{name}[]")));
    v.and_then(Value::as_str)
}

/// Every value of a repeated form field (`name[]`, or `name`): the parsed
/// body keeps only the last, so the raw form is scanned.
fn values<'a>(req: &'a Request, name: &str) -> Vec<&'a str> {
    let raw = std::str::from_utf8(&req.raw).unwrap_or_default();
    let mut out = Vec::new();
    for n in [format!("{name}[]"), name.to_owned()] {
        let head = format!("name=\"{n}\"");
        for (i, _) in raw.match_indices(&head) {
            let rest = &raw[i + head.len()..];
            let value = rest
                .split_once("\r\n\r\n")
                .and_then(|(_, v)| v.split_once("\r\n"));
            out.extend(value.map(|(v, _)| v));
        }
    }
    out
}

fn logprobs(req: &Request) -> bool {
    values(req, "include").contains(&"logprobs")
}

/// `secs` as `HH:MM:SS,mmm` (SRT) or `HH:MM:SS.mmm` (WebVTT).
fn stamp(secs: f64, sep: char) -> String {
    let ms = (secs * 1000.0).round() as u64;
    let (h, m, s) = (ms / 3_600_000, ms / 60_000 % 60, ms / 1000 % 60);
    format!("{h:02}:{m:02}:{s:02}{sep}{:03}", ms % 1000)
}
