# `prompt_struct` schema (render-time provenance)

## Principle

**Record what was sampled and filled at `render_prompt` time.** No reverse lookup, no `questions[0]` default, no assumption that object slots are always `A,B,C`.

## Flow

1. `PromptTemplate.render_provenance()` samples `(question_index, answer_index)`, fills `[X]`/`[Y]`/… placeholders, returns `PromptRenderRecord`.
2. `render_prompt()` stores that record on thread-local and returns `question Answer: answer` (same string as `messages`).
3. `_record_turn()` passes `render` into `build_turn_record()` → `prompt_struct` + `question_text` / `answer_text`.

## `prompt_struct` fields

| Field | Meaning |
|--------|---------|
| `template_id` | Registry key, e.g. `counting.mcq` |
| `question_index` | Index into `PromptTemplate.questions` |
| `answer_index` | Index into `answers` / true/false pool |
| `question_pattern` | Selected question line with `[K]` → `{{K}}` |
| `answer_pattern` | Selected answer line with `{{K}}` |
| `question_bindings` | Values applied to question placeholders (`X`→tag, `Y`→options, …) |
| `answer_bindings` | Values applied to answer placeholders (MCQ letter in `X`, etc.) |
| `referent_slots` | Scene-object referents: `{{key}}` → `{obj_idx, tag}` for **object** placeholders only |

## `referent_slots` vs bindings

- **`question_bindings` / `answer_bindings`**: literal strings used in the prompt (including MCQ option blocks in `Y`).
- **`referent_slots`**: links template placeholder letters to preprocess object indices (for merge/dedup/fingerprint). Keys must match object placeholders in the sampled question (e.g. counting MCQ uses `X`, not `A`).

## Deprecated

- `slots` alone as ambiguous object map (kept readable in fingerprint/validator as fallback alias for `referent_slots`).
