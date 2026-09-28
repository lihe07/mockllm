//! Small helpers shared by endpoints, and the per-model table lookup.

use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{Map, Value};

/// A row of a per-model table: the rules of every model id that starts
/// with its prefix.
pub(crate) trait Row: Copy {
    fn prefix(&self) -> &'static str;
}

/// The row for `model`: the longest matching prefix, else `base` (no
/// per-model rules). Every table in the crate is looked up this way.
pub(crate) fn lookup<M: Row>(rows: &[M], base: M, model: &str) -> M {
    let rows = rows.iter().filter(|m| model.starts_with(m.prefix()));
    rows.max_by_key(|m| m.prefix().len())
        .copied()
        .unwrap_or(base)
}

/// `map[key]`, or `null` when missing (indexing a `Map` panics).
pub(crate) fn at<'a>(map: &'a Map<String, Value>, key: &str) -> &'a Value {
    static NULL: Value = Value::Null;
    map.get(key).unwrap_or(&NULL)
}

/// A fresh number for ids.
pub(crate) fn next_id() -> u64 {
    static N: AtomicU64 = AtomicU64::new(1);
    N.fetch_add(1, Ordering::Relaxed)
}

pub(crate) fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// Splits `text` into word pieces, each but the first with its leading
/// space: "Hi there" is ["Hi", " there"].
pub(crate) fn words(text: &str) -> Vec<String> {
    let (mut out, mut cur) = (Vec::new(), String::new());
    for c in text.chars() {
        if c == ' ' && !cur.trim().is_empty() {
            out.push(std::mem::take(&mut cur));
        }
        cur.push(c);
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Splits `s` into pieces of up to `n` chars, as providers stream JSON.
pub(crate) fn pieces(s: &str, n: usize) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    chars.chunks(n).map(|c| c.iter().collect()).collect()
}

/// A rough token count: the words in every string under `v`, at least 1.
pub(crate) fn tokens(v: &Value) -> u64 {
    fn count(v: &Value) -> u64 {
        match v {
            Value::String(s) => s.split_whitespace().count() as u64,
            Value::Array(a) => a.iter().map(count).sum(),
            Value::Object(o) => o.values().map(count).sum(),
            _ => 0,
        }
    }
    count(v).max(1)
}
