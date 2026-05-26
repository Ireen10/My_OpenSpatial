# Structured prompt placeholders (M8 L2)

Canonical syntax: **`[A]`–`[Z]`** uppercase single letters, filled at render time via `shared` (both sides), `q_args` (question only), or `a_args` (answer only). See `task/annotation/core/prompt_template.py` and `structured_prompt_template.py`.

## 1. Fill scopes

| Scope | Applies to | Typical use |
|--------|------------|-------------|
| `shared` | `question_text` + `answer_text` | Object names, colors, option blocks, values reused in Q and A |
| `q_args` | Question only | Stem-only slots (e.g. anchor in question, list of marks in MCQ stem) |
| `a_args` | Answer only | Answer-only slots (e.g. MCQ letter, point label, premise clause in answer) |

**Important:** The same letter may mean **different things** in question vs answer when question uses `q_args` and answer uses `a_args` (e.g. multiview N-ary distance: question `[X]` = anchor, answer `[X]` = winning object).

## 2. Global letter semantics

### 2.1 Scene object referents (bound to `referent_slots` / marks)

Letters used for **marked objects** in the image. Handler must align with `mark_spec` slot order.

| Slot | Meaning | Used in tasks |
|------|---------|----------------|
| **A** | Primary object / query object / first mark | distance, size, position (view-1 obj), depth (mark list), grounding (tag list), correspondence (query **color**), obj_cam |
| **B** | Second object / candidate color / anchor (position) | distance, size, position, correspondence (**candidate color**) |
| **C** | Third object / anchor (relative distance) / target in view 2 (position) | distance (anchor), position (object in image 2) |
| **D** | Fourth role or **non-object** field — **task-specific** (see §3) | distance (meters A→C), position (direction), depth (ordinal in choice MCQ) |

`sample_metadata._NON_OBJECT_PLACEHOLDERS` excludes **`Y`, `O`, `Z`, `T`, `D`** from object referent alignment (format / list / options / type label / direction text). **`E`, `L`, `P`** are not excluded but are **not** scene objects when used as MCQ letter / point id / premise clause.

### 2.2 Format & MCQ infrastructure (not scene referents)

| Slot | Meaning | Handler binding |
|------|---------|-----------------|
| **O** | MCQ **Options** block (full string, often `\nOptions: A. … B. …`) | `shared["O"]` |
| **T** | **Type** label (`points` / `boxes`) OR **enumerated list** of objects (`obj1, obj2, …`) OR full MCQ option line (`A. point-A`) — **task-specific** | depth, multiview size/distance, correspondence MCQ direct |
| **X** | **Primary answer token** — numeric distance, object name, MCQ option (`A. …`), JSON blob, ordering string, view id — **task-specific** | all tasks |
| **Y** | Secondary answer / options list / closer-farther / anchor in multiview distance **answer** | depth MCQ, obj_cam, multiview distance N-ary answer |
| **Z** | MCQ option list (depth choice MCQ) | depth |
| **E** | MCQ **option letter only** (`A`–`D`), must match `Options:` line | correspondence MCQ sentence, position MCQ sentence |
| **L** | Point **label** only (`point-A`, `point-1`) — not the option letter | correspondence OE/MCQ sentence |
| **P** | **Premise clause** copied from stem (position sentence answers) | multiview position |

### 2.3 Instruction modes (not placeholders)

Structured templates use separate `question_instruction` pools and `answer_profiles` (`direct` | `sentence` | `free` | `reasoning` | …). **OE** must not sample MCQ-only **`direct`** short-answer profiles when **`free`** must match full-sentence pools (correspondence, multiview position).

## 3. Per-task binding contracts

### 3.1 Correspondence (`multiview_correspondence`)

| Slot | Question | Answer |
|------|----------|--------|
| A | Query point **color** (e.g. `pink`) | same |
| B | Candidate **color** (e.g. `blue`) | same |
| O | `Options: A. point-1 …` (MCQ only) | — |
| L | — | `point-{id}` (`point-A` or `point-1`) |
| E | — | **Option letter `A`–`D` only** (never raw `1`–`4` when options are `A. point-1`) |
| T | — | Full option text `A. point-1` (MCQ direct) |

Templates: `.oe.{sentence|free}`, `.mcq.{direct|sentence|free}`.

### 3.2 Multiview position (`multiview_position`)

| Slot | Question | Answer |
|------|----------|--------|
| A | Object in image 1 | same |
| B | **Anchor** (in both views) | same |
| C | Object in image 2 | same |
| X | Direction of A relative to B in image 1 (`north`, `front`, …) | same |
| P | — | Premise clause (filled from `FRAME_PREMISE_POOLS`) |
| D | — | Direction of C from B (`east`, `front-left`, …) |
| E | — | MCQ option letter (`A`–`D`) |
| T | — | MCQ full option `B.front-left` or bare direction (OE direct) |
| O | MCQ options string | — |

OE modes: `sentence`, `free` only. MCQ: `direct`, `sentence`, `free`.

### 3.3 Distance — absolute (`distance.*` / `multiview_distance.absolute_*`)

| Slot | Question | Answer |
|------|----------|--------|
| A, B | Object descriptions | same |
| X | Numeric distance with unit (`12.34` in template + ` meters`) | same |

Modes: separate template ids `.direct` / `.sentence` (absolute; not random `instruction_type` on one id).

### 3.4 Distance — relative singleview (`distance.relative_{far|close}.*`)

| Slot | Question | Answer |
|------|----------|--------|
| A, B | Candidate objects | same |
| C | Anchor object | same |
| D | Distance A→C (formatted, e.g. `1.2 m`) | same |
| E | Distance B→C | same |
| X | Winning **object name** (OE) or **`A. Title`** MCQ option (MCQ) | same |
| O | MCQ options (MCQ only) | — |

### 3.5 Distance — multiview N-ary (`multiview_distance.{farthest|closest}.{direct|reasoning|free}`)

| Slot | Question | Answer |
|------|----------|--------|
| T | Comma-separated **candidate** object names | same (shared) |
| X | **Anchor** object (via `q_args`) | **Winner** object (via `a_args`) |
| Y | — | **Anchor** object (via `a_args`) |

Do not confuse question anchor `X` with answer winner `X`.

Instruction modes: `direct` / `reasoning` / `free` (not `sentence`; `free` uses the same answer pool as `reasoning`).

### 3.6 Distance — obj_cam (`multiview_distance.obj_cam_mcq`)

| Slot | Meaning |
|------|---------|
| A | Object description |
| Y | `closer` / `farther` |
| X | Answer view label / option |
| O | Options string |

### 3.7 Size (`size.*`, `multiview_size.*`)

| Slot | Meaning |
|------|---------|
| A, B | Two objects (pair judgment) |
| T | Candidate list (N-ary) |
| X | Winner name or absolute measure `value unit` |

Judgment uses `condition` → `true`/`false` profiles, not `[X]` in stem.

### 3.8 Depth (`depth.*`)

| Slot | Meaning |
|------|---------|
| T | Mark type label (`points` / `boxes`) |
| A | Comma-separated tags (OE) or mark list in stem |
| X | Ordering string, chosen tag, or MCQ answer token |
| Y | MCQ options block (ordering MCQ) or ordinal (choice) |
| Z | MCQ options (choice MCQ) |
| B | Ordinal word (`first`, …) in choice OE |
| O | `[O]` in MCQ stems → options via `Y`/`Z`, not always `shared["O"]` |

Uses `q_args` / `a_args` heavily; legacy `referent_mode`.

### 3.9 Position singleview (`position.*`)

| Slot | Meaning |
|------|---------|
| A, B | Objects (proximity) |
| O | Options |
| X | Full MCQ answer `A. The …` |

### 3.10 3D grounding (`grounding_3d.open_ended`)

| Slot | Meaning |
|------|---------|
| A | Comma-separated object tags |
| X | JSON answer payload |
| (+ camera keys in `camera_shared`) |

### 3.11 Caption (`caption.open_ended`)

Generative profile; placeholders in introduction/stem only.

## 4. Handler source map

| Task file | Prompt builder |
|-----------|----------------|
| `multiview_correspondence.py` | `point_correspondence_prompt_func` |
| `multiview_object_position.py` | `position_prompt_func` |
| `multiview_distance.py` | `pair_absolute_distance_prompt_func`, `multi_relative_distance_prompt_func` |
| `multiview_distance_obj_cam.py` | obj_cam prompt |
| `multiview_size.py` | `pair_relative_size_prompt_func`, `multi_relative_size_prompt_func` |
| `distance.py` | `absolute_distance_prompt_func`, `relative_distance_*` |
| `size.py` | size / height / absolute helpers |
| `depth_annotation.py` | `_build_ordering_prompt`, `_build_choice_prompt` |
| `position.py` | `height_comparison_prompt_func`, `proximity_prompt_func` |
| `3d_grounding.py` | `grounding_oe_prompt_func` |

## 5. Forward audit

Run binding unit checks + review parquet scan:

```bash
python verification/scripts/audit_structured_placeholder_bindings.py
python verification/scripts/audit_structured_placeholder_bindings.py --data-dir output/frame_rot/base_pipeline_demo_multiview_all_frame_rot_review
```

See script output for `FAIL` / `WARN` per rule.

## 6. Changelog (audit-driven fixes)

| Date | Issue | Fix |
|------|-------|-----|
| 2026-05 | OE correspondence sampled MCQ `direct` → bare `point-C` | OE modes `sentence`/`free` only; split `.oe.{mode}` templates |
| 2026-05 | OE position sampled `direct` | OE modes `sentence`/`free` only |
| 2026-05 | `[E]` = `1` with options `A. point-1` | `[E]` = `ABCD[idx]` in `multiview_correspondence.py` |
