//! Per-model rules of OpenAI Chat Completions, one row per model family.
//! A model id takes the row with the longest matching prefix; an id with
//! no row gets [`BASE`], no per-model rules. Each row cites its source.
//!
//! Only rules confirmed in OpenAI's docs are here. Error wording that the
//! docs do not give is the wording commonly seen from the API.

/// When the sampling parameters (`temperature`, `top_p`, `logprobs`,
/// `top_logprobs`) are accepted. Also used for function calling (`tools`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Allowed {
    Always,
    Never,
    /// Only with `reasoning_effort: "none"` in effect.
    OnlyEffortNone,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Model {
    pub prefix: &'static str,
    /// The model page's "max output tokens".
    pub max_output: u64,
    /// A reasoning model: hidden reasoning is counted in
    /// `usage.completion_tokens_details.reasoning_tokens`.
    pub reasoning: bool,
    /// Rejects `max_tokens`; `max_completion_tokens` is required instead.
    pub no_max_tokens: bool,
    /// The `reasoning_effort` values accepted; empty means not checked.
    pub efforts: &'static [&'static str],
    /// The effort in effect when the request sets none.
    pub default_effort: Option<&'static str>,
    pub sampling: Allowed,
    pub tools: Allowed,
}

pub(crate) const BASE: Model = Model {
    prefix: "",
    max_output: u64::MAX,
    reasoning: false,
    no_max_tokens: false,
    efforts: &[],
    default_effort: None,
    sampling: Allowed::Always,
    tools: Allowed::Always,
};

const TO_HIGH: &[&str] = &["none", "low", "medium", "high"];
const TO_XHIGH: &[&str] = &["none", "low", "medium", "high", "xhigh"];
const TO_MAX: &[&str] = &["none", "minimal", "low", "medium", "high", "xhigh", "max"];

// o-series: `max_tokens` "is now deprecated in favor of
// `max_completion_tokens`, and is not compatible with o-series models".
// https://developers.openai.com/api/reference/resources/chat (max_tokens)
const O_SERIES: Model = Model {
    reasoning: true,
    no_max_tokens: true,
    max_output: 100_000,
    ..BASE
};

/// GPT-5 and later: reasoning models with 128,000 max output (model pages).
const GPT5: Model = Model {
    max_output: 128_000,
    reasoning: true,
    ..BASE
};

#[rustfmt::skip]
pub(crate) const MODELS: &[Model] = &[
    // https://developers.openai.com/api/docs/models/gpt-4o
    Model { prefix: "gpt-4o", max_output: 16_384, ..BASE },
    // https://developers.openai.com/api/docs/models/gpt-4.1
    Model { prefix: "gpt-4.1", max_output: 32_768, ..BASE },
    // https://developers.openai.com/api/docs/models/o1 (and o3, o3-mini, o4-mini)
    Model { prefix: "o1", ..O_SERIES },
    Model { prefix: "o3", ..O_SERIES },
    Model { prefix: "o4-mini", ..O_SERIES },
    // https://developers.openai.com/api/docs/models/o1-mini: 65,536 max
    // output; its supported features list no function_calling.
    Model { prefix: "o1-mini", max_output: 65_536, tools: Allowed::Never, ..O_SERIES },
    // https://developers.openai.com/api/docs/models/gpt-5: effort "minimal,
    // low, medium, and high". Sampling "will raise an error for ... older
    // GPT-5 models such as gpt-5, gpt-5-mini, or gpt-5-nano":
    // https://developers.openai.com/api/docs/guides/latest-model/gpt-5.4
    Model { prefix: "gpt-5", efforts: &["minimal", "low", "medium", "high"], default_effort: Some("medium"), sampling: Allowed::Never, ..GPT5 },
    // https://developers.openai.com/api/docs/models/gpt-5-chat-latest
    Model { prefix: "gpt-5-chat", max_output: 16_384, ..BASE },
    // https://developers.openai.com/api/docs/models/gpt-5.1: "none
    // (default), low, medium, and high".
    Model { prefix: "gpt-5.1", efforts: TO_HIGH, default_effort: Some("none"), ..GPT5 },
    // https://developers.openai.com/api/docs/models/gpt-5.2 and gpt-5.4:
    // "none (default), low, medium, high and xhigh". temperature, top_p and
    // logprobs are "only supported" with effort none:
    // https://developers.openai.com/api/docs/guides/latest-model/gpt-5.4
    Model { prefix: "gpt-5.2", efforts: TO_XHIGH, default_effort: Some("none"), sampling: Allowed::OnlyEffortNone, ..GPT5 },
    Model { prefix: "gpt-5.4", efforts: TO_XHIGH, default_effort: Some("none"), sampling: Allowed::OnlyEffortNone, ..GPT5 },
    // https://developers.openai.com/api/docs/models/gpt-5.5: "none, low,
    // medium (default), high and xhigh".
    Model { prefix: "gpt-5.5", efforts: TO_XHIGH, default_effort: Some("medium"), ..GPT5 },
    // https://developers.openai.com/api/docs/models/gpt-5.6-terra: "none,
    // low, medium (default), high, xhigh, and max".
    Model { prefix: "gpt-5.6-terra", efforts: &["none", "low", "medium", "high", "xhigh", "max"], default_effort: Some("medium"), ..GPT5 },
    // https://developers.openai.com/api/docs/guides/latest-model: GPT-6
    // Astra "does not support `none`"; with effort not none "remove
    // `temperature`, `top_p`, and `top_logprobs`. For Chat Completions, also
    // remove `logprobs`"; Astra "supports Chat Completions, but its tool
    // calling requires Responses"; Sol and Luna "support function calling
    // in Chat Completions only with `reasoning_effort: "none"`".
    Model { prefix: "gpt-6-astra", efforts: &["minimal", "low", "medium", "high", "xhigh", "max"], sampling: Allowed::OnlyEffortNone, tools: Allowed::Never, ..GPT5 },
    Model { prefix: "gpt-6-sol", efforts: TO_MAX, sampling: Allowed::OnlyEffortNone, tools: Allowed::OnlyEffortNone, ..GPT5 },
    Model { prefix: "gpt-6-luna", efforts: TO_MAX, sampling: Allowed::OnlyEffortNone, tools: Allowed::OnlyEffortNone, ..GPT5 },
];

/// The row for `model`: the longest matching prefix, else [`BASE`].
pub(crate) fn lookup(model: &str) -> Model {
    let rows = MODELS.iter().filter(|m| model.starts_with(m.prefix));
    rows.max_by_key(|m| m.prefix.len()).copied().unwrap_or(BASE)
}
