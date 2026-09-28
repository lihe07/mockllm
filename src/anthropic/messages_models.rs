//! Per-model rules of Anthropic Messages, one row per model family. A
//! model id takes the row with the longest matching prefix; an id with no
//! row gets [`BASE`], no per-model rules. Each row cites its source.
//!
//! Sources:
//! - T: https://platform.claude.com/docs/en/build-with-claude/thinking-troubleshooting
//!   (thinking types accepted, rejected and on by default, per model)
//! - O: https://platform.claude.com/docs/en/models/overview (max output)
//! - C: https://platform.claude.com/docs/en/build-with-claude/context-windows
//! - L: https://platform.claude.com/docs/en/build-with-claude/thinking#limits-and-feature-compatibility
//!   (fixed sampling; forced tool use; `display` defaults "omitted")
//! - E: https://platform.claude.com/docs/en/api/errors (prefill; forced tools)
//! - H: https://platform.claude.com/docs/en/models/haiku-4-5/migration-guide
//!   ("Use only `temperature` OR `top_p`, not both")
//! - R: https://platform.claude.com/docs/en/release-notes/overview
//!   (2025-08-05: "Opus 4.1 does not allow both `temperature` and `top_p`")

#[derive(Clone, Copy, Debug)]
pub(crate) struct Model {
    pub prefix: &'static str,
    pub max_output: u64,
    pub context: u64,
    /// The `thinking.type` values accepted; the rest are rejected.
    pub thinking: &'static [&'static str],
    /// Thinks when the request has no `thinking`.
    pub thinks_by_default: bool,
    /// `thinking.display` defaults to "omitted": empty thinking text.
    pub display_omitted: bool,
    /// Non-default `temperature`, `top_p` or `top_k` are rejected always.
    pub fixed_sampling: bool,
    /// `temperature` and `top_p` cannot both be set.
    pub one_sampler: bool,
    /// A last assistant message (prefill) is rejected.
    pub no_prefill: bool,
    /// `tool_choice` "any" and "tool" are rejected.
    pub no_forced_tools: bool,
}

const ALL: &[&str] = &["enabled", "adaptive", "disabled"];
const EXTENDED: &[&str] = &["enabled", "disabled"];
const ADAPTIVE: &[&str] = &["adaptive"];
const ADAPTIVE_OR_OFF: &[&str] = &["adaptive", "disabled"];

pub(crate) const BASE: Model = Model {
    prefix: "",
    max_output: u64::MAX,
    context: u64::MAX,
    thinking: ALL,
    thinks_by_default: false,
    display_omitted: false,
    fixed_sampling: false,
    one_sampler: false,
    no_prefill: false,
    no_forced_tools: false,
};

/// Claude 4.7 and later: 1M context, 128k output (O, C), fixed sampling
/// (L), no prefill (E: "Claude 4.6 and later models").
const LATEST: Model = Model {
    max_output: 128_000,
    context: 1_000_000,
    thinking: ADAPTIVE_OR_OFF,
    fixed_sampling: true,
    no_prefill: true,
    ..BASE
};

/// Always or by default thinking, with "omitted" display (T, L).
const THINKING: Model = Model {
    thinks_by_default: true,
    display_omitted: true,
    ..LATEST
};

/// Claude 4.5 and earlier: 200k context (C), extended thinking only (T).
const EARLIER: Model = Model {
    context: 200_000,
    thinking: EXTENDED,
    ..BASE
};

#[rustfmt::skip]
pub(crate) const MODELS: &[Model] = &[
    // Always on: "enabled" and "disabled" rejected (T); no forced tools (E).
    Model { prefix: "claude-fable-5-1", thinking: ADAPTIVE, no_forced_tools: true, ..THINKING },
    Model { prefix: "claude-mythos-5-1", thinking: ADAPTIVE, no_forced_tools: true, ..THINKING },
    Model { prefix: "claude-opus-5-5", thinking: ADAPTIVE, no_forced_tools: true, ..THINKING },
    Model { prefix: "claude-fable-5", thinking: ADAPTIVE, ..THINKING },
    Model { prefix: "claude-mythos-5", thinking: ADAPTIVE, ..THINKING },
    // On by default; "disabled" accepted (T; at effort high or below).
    Model { prefix: "claude-opus-5", ..THINKING },
    Model { prefix: "claude-sonnet-5", ..THINKING },
    // Sonnet 5.5 takes "between_tools" in place of "disabled" (T).
    Model { prefix: "claude-sonnet-5-5", thinking: &["adaptive", "between_tools"], no_forced_tools: true, ..THINKING },
    // Off by default, adaptive only (T).
    Model { prefix: "claude-opus-4-8", ..LATEST },
    Model { prefix: "claude-opus-4-7", ..LATEST },
    // Adaptive, and extended (deprecated) (T); sampling fixed only while
    // thinking (L: "On older models, the restriction applies only while
    // thinking is on").
    Model { prefix: "claude-opus-4-6", thinking: ALL, fixed_sampling: false, ..LATEST },
    Model { prefix: "claude-sonnet-4-6", thinking: ALL, fixed_sampling: false, ..LATEST },
    // Extended only, 64k output (T, O).
    Model { prefix: "claude-opus-4-5", max_output: 64_000, ..EARLIER },
    Model { prefix: "claude-sonnet-4-5", max_output: 64_000, ..EARLIER },
    Model { prefix: "claude-haiku-4-5", max_output: 64_000, one_sampler: true, ..EARLIER },
    // "Earlier Claude 4 models (Claude Opus 4.1, Claude Sonnet 4, and Claude
    // Opus 4) support extended thinking only" (T). Opus 4.1: one sampler (R).
    Model { prefix: "claude-opus-4-1", one_sampler: true, ..EARLIER },
    Model { prefix: "claude-opus-4", ..EARLIER },
    Model { prefix: "claude-sonnet-4", ..EARLIER },
];

impl crate::util::Row for Model {
    fn prefix(&self) -> &'static str {
        self.prefix
    }
}
