# ADR 0005 — Typed values in registry `options`: `str` or `list[str]`

- **Status:** Accepted (2026-10)
- **Contracts:** none (see *Decision* 4)
- **Supersedes:** —
- **Issue / PRs:** #260; follows #258 (JSON argv inside the `command` string, never released)

## Context

Registry-dispatched backends (QMIO, the `"subprocess"` bridge, any third-party
backend registered with `register_backend`) receive their provider-specific
configuration through an `options` bag. Until now it was `dict[str, str]` end to
end: `HashMap<String, String>` in `BackendConfig::Registered` and in
`BackendBuildContext`, read with `BackendBuildContext::option(key) ->
Option<&str>`.

A string cannot carry an argv. The subprocess bridge's `command` was split on
whitespace (v0.7.2), so a worker or interpreter under a path with spaces could
not be launched. #258 worked around it by parsing a value that starts with `[` as
a JSON array of strings: a second, ad-hoc type system inside a string, with its
own error modes, that every third-party backend needing a list would have had to
reinvent. That form landed on `main` after v0.7.2 and has not been released.

`option()` also returned `None` for an absent key only, so the backends' rule
"absent key → default; present but malformed → error" held. Widening the value
type must keep it.

## Decision

1. **One value type, `polypus_backend::OptionValue`** — `Str(String)` or
   `List(Vec<String>)`, `Debug + Clone + PartialEq + Eq`, with
   `From<&str>`, `From<String>` and `From<Vec<String>>`. It lives in the pyo3-free
   `polypus-backend` and adds no dependency. `BackendBuildContext.options` and
   `BackendConfig::Registered.options` become `HashMap<String, OptionValue>`. Not
   `serde_json::Value` (an untyped tree every factory would have to validate, and
   a dependency of the public contract) and not a second map for lists (two
   places to look for one key, and a key could be in both).
2. **Two fallible, symmetric accessors replace `option()`.**
   `option_str(key) -> Result<Option<&str>, BackendError>` and
   `option_list(key) -> Result<Option<&[String]>, BackendError>`: `Ok(None)` for an
   absent key, `Ok(Some(_))` for a value of the expected shape, and
   `BackendError::Conversion` naming the key for the other shape. A lenient
   `option()` that returned `None` for a list would bring back the silent default
   the backends avoid. `option()` is removed rather than kept beside them.
3. **The JSON-in-a-string `command` is withdrawn now.** `command` is a `str`, split
   on whitespace without quotes or escapes (as in v0.7.2), or a `list[str]`, the
   exact argv. An empty list or an empty `argv[0]` is a `BackendError`. A string
   starting with `[` is split on whitespace like any other string, with no warning
   or error: there is no heuristic guessing it is a leftover JSON argv.
4. **No new contract.** `options` does not appear in `docs/CONTRACTS.md` (C-1 is
   the `polypus_python` seam, which this does not touch). The Python→Rust edge and
   the `BackendBuildContext` shape are documented in `docs/backends.md` and here.
5. **At the Python edge**, `run_quantum_circuit`, `train` and `qml.train` accept
   `options: dict[str, str | list[str]]`. The conversion lives in the `polypus`
   crate (a local `FromPyObject` newtype, since both the trait and `OptionValue`
   are foreign to it). It walks a snapshot of the dict's items and reads each
   list/tuple by index, so no user code (an overridden `__iter__`) runs while the
   dict is walked — a dict mutated mid-walk would otherwise panic inside PyO3.
6. **`OptionValue` is `#[non_exhaustive]`.** A further value shape (an integer, a
   nested map) can be added without another breaking change. The cost: code
   outside `polypus-backend` that `match`es on it needs a wildcard arm (the
   subprocess bridge's `command` reader has one). Factories are expected to use the
   accessors, not `match`.
7. **An empty list is accepted at the edge.** Whether it means anything is the
   backend's call: the subprocess bridge rejects an empty `command`.
8. **Accepted Python values are `str` and `list`/`tuple` of `str`.** Everything else
   — `int`, `bool`, `float`, `None`, `dict`, a list holding a non-`str` — is a
   `TypeError` naming the key (`options['k'] must be a str or a list of str, got
   int`; `options['k'][1] must be a str, got NoneType`). PyO3 already raised
   `TypeError` for a non-`str` value, so the exception type is unchanged; numbers
   are not coerced to strings (see *Alternatives considered* 5). Relaxing this
   later (coercing numbers) would break no caller, so it can be its own change.

Decisions 6, 7 and 8, and the absence of a heuristic in 3, were proposed by the
planner of #260 and confirmed by the maintainers on review (2026-10-09).

## Compatibility

Relative to v0.7.2 (the last release):

| Who | Breaks | Migration |
|-----|--------|-----------|
| Rust factory reading an option | `ctx.option(key)` no longer exists | `ctx.option_str(key)?` (now `Result<Option<&str>, _>`); `ctx.option_list(key)?` for a list |
| Rust code building a `BackendBuildContext` or `BackendConfig::Registered` | `options` is `HashMap<String, OptionValue>` | wrap values: `OptionValue::from("…")`, `OptionValue::from(vec![…])` |
| Python caller with `str` values | nothing | — |
| Python caller passing a non-`str` value | still a `TypeError`; the message now names the key | — |
| Subprocess `command` as a string | nothing (same whitespace split as v0.7.2) | use a list for any argument containing a space |

The JSON-array string from #258 is removed without a deprecation period because it
never reached a release. A `main` user who wrote `json.dumps([...])` must pass the
list itself.

## Consequences

- A backend that needs a list reads it with `option_list` and gets type checking
  for free; none needs its own string encoding.
- A list where a factory reads a string (QMIO's `endpoint`, the bridge's
  `recv_timeout_ms`, …) is a `BackendError::Conversion` naming the key, raised as
  `polypus.BackendError`, never a silent default.
- `polypus-backend` stays pyo3-free and gains no dependency.
- Every third-party factory using `option()` has to change one call per key. This
  is the price of D2, accepted over a tolerant accessor.
- `local`/`cunqa` keep rejecting a non-empty `options` (with `ValueError`), whatever
  the value types.

## Alternatives considered

1. **Keep `dict[str, str]` and the JSON-in-a-string argv (#258).** A private
   encoding per backend, invisible in the type, and a string that happens to start
   with `[` changes meaning.
2. **`serde_json::Value` as the value type.** Any shape, but an untyped tree every
   factory has to validate, and `serde_json` in the public contract's API.
3. **A second `list_options` map.** No enum, but two lookups for one key and an
   ambiguous key present in both.
4. **Keep `option()` beside the new accessors**, returning `None` for a list. No
   break for existing factories, but a list in a string key would silently become
   the default.
5. **Coerce Python numbers and booleans to strings at the edge.** Convenient
   (`{"recv_timeout_ms": 600000}`), but `True` → `"True"` and `1.0` → `"1.0"` are
   spellings each backend would then have to accept; kept out of scope.

## Reopening criteria

- A backend needs a value that is neither a string nor a list of strings (a
  number with its type preserved, a nested map): add a variant (non-breaking,
  thanks to `#[non_exhaustive]`) or reconsider alternative 2.
- `options` becomes part of a seam contract: move this into `docs/CONTRACTS.md`
  with its enforcing test.
