# M8 L1 — Structured `prompt_template` design

## Goal

Redesign **structure only** (not QA copy): per `QuestionType`, split question into
`introduction` + `stem` + `question_instruction`, and answers into
`instruction_type` → `{instruction_snippets[], answer_templates[]}`.
All instruction pools are **task-registered**; framework provides no default pools.

## Question types (product)

| Type | Enum | Metadata short |
|------|------|----------------|
| 开放式问答 | `OPEN_ENDED` | `OE` |
| 选择题 | `MCQ` | `MCQ` |
| 判断题 | `JUDGMENT` | `Judgment` |

Legacy flaw: templates with `[O]` / explicit options must use `MCQ`, not `OPEN_ENDED`.
Fix at L2 task wiring + shim classification.

## L1 scope

- `QuestionType.JUDGMENT`
- `structured_prompt_template.py`: register/render API
- Extended `PromptRenderRecord` fields for provenance
- Unit tests; **no** task migration yet (L2)

## Out of scope (L1)

- Splitting grounding camera/JSON text
- Pre-built instruction_type presets
- Parity rerun / changing emitted QA strings
