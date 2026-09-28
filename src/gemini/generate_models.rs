//! Per-model rules of Gemini `generateContent`, one row per model family.
//! A model id takes the row with the longest matching prefix; an id with
//! no row gets [`BASE`], no per-model rules. Each row cites its source.
//!
//! Sources:
//! - M: the model pages, "Output token limit", such as
//!   https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash
//! - T: https://ai.google.dev/gemini-api/docs/thinking ("Controlling
//!   thinking": default thinking and the levels supported, per model)
//! - R: https://ai.google.dev/api/generate-content (`ThinkingConfig`:
//!   `thinkingLevel` is "Recommended for Gemini 3 or later models. Use with
//!   earlier models results in an error.")

#[derive(Clone, Copy, Debug)]
pub(crate) struct Model {
    pub prefix: &'static str,
    /// The largest `generationConfig.maxOutputTokens`.
    pub max_output: u64,
    /// The `thinkingLevel` values accepted, `None` when not checked; empty
    /// rejects `thinkingLevel` (R).
    pub levels: Option<&'static [&'static str]>,
    /// Thinks by default: `Answer::reasoning` counts in `thoughtsTokenCount`.
    pub thinks: bool,
}

pub(crate) const BASE: Model = Model {
    prefix: "",
    max_output: u64::MAX,
    levels: None,
    thinks: false,
};

const LOW_HIGH: Option<&[&str]> = Some(&["low", "medium", "high"]);
const MINIMAL_HIGH: Option<&[&str]> = Some(&["minimal", "low", "medium", "high"]);

/// Gemini 3 and later: 65,536 output (M), thinking on (T).
const GEMINI3: Model = Model {
    max_output: 65_536,
    thinks: true,
    levels: MINIMAL_HIGH,
    ..BASE
};

/// Gemini 2.5: 65,536 output (M), no `thinkingLevel` (R).
const GEMINI25: Model = Model {
    max_output: 65_536,
    levels: Some(&[]),
    thinks: true,
    ..BASE
};

#[rustfmt::skip]
pub(crate) const MODELS: &[Model] = &[
    // 8,192 output (M: gemini-2.0-flash); before Gemini 3 (R).
    Model { prefix: "gemini-2.0-flash", max_output: 8_192, levels: Some(&[]), ..BASE },
    Model { prefix: "gemini-2.5-pro", ..GEMINI25 },
    Model { prefix: "gemini-2.5-flash", ..GEMINI25 },
    // Thinking "Off" by default (T).
    Model { prefix: "gemini-2.5-flash-lite", thinks: false, ..GEMINI25 },
    // Levels per model (T).
    Model { prefix: "gemini-3-pro-preview", levels: Some(&["low", "high"]), ..GEMINI3 },
    Model { prefix: "gemini-3-flash-preview", ..GEMINI3 },
    Model { prefix: "gemini-3.1-pro-preview", levels: LOW_HIGH, ..GEMINI3 },
    Model { prefix: "gemini-3.5-flash", ..GEMINI3 },
    Model { prefix: "gemini-3.6-flash", ..GEMINI3 },
    Model { prefix: "gemini-3.7-flash", levels: LOW_HIGH, ..GEMINI3 },
    Model { prefix: "gemini-3.8-flash", levels: LOW_HIGH, ..GEMINI3 },
];

impl crate::util::Row for Model {
    fn prefix(&self) -> &'static str {
        self.prefix
    }
}
