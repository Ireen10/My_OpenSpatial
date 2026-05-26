# Change log — M8 L1 prompt structure

## Done

- `QuestionType.JUDGMENT` (判断题); metadata short `Judgment` via `_qtype_str`.
- `task/annotation/core/structured_prompt_template.py`: introduction / stem / question_instruction + answer pools keyed by `instruction_type` (task-registered only).
- `BaseAnnotationTask.render_structured_prompt`, `StructuredTemplateRegistry`, `from_legacy_prompt_template` for L2.
- `tests/test_prompt_template_structure.py`; `verify_milestone M8`.

## Not done (L2+)

- Wire annotation tasks to structured API; fix `[O]` mislabeled as OPEN_ENDED.
- Parity rerun vs frame_rot baseline.
- Task-level content / instruction pool registration (per-task kickoff).
