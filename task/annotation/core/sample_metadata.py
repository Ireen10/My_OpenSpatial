"""
Sample-level metadata helpers (plan M3+).

prompt_struct is filled from PromptRenderRecord at sample time — no reverse lookup,
no default-first-template, no A/B/C alphabet assumptions for placeholder semantics.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .prompt_template import PLACEHOLDER_RE, PromptRenderRecord, PromptTemplate, TemplateRegistry

import task.prompt_templates  # noqa: F401

_SLOT_PATTERN_RE = re.compile(r"-\(\w+\s+(box|mask|point)\)", re.I)
# Placeholders used for MCQ options / type labels — not scene-object referents.
_NON_OBJECT_PLACEHOLDERS = frozenset({"Y", "O", "Z", "T", "D"})


def split_prompt_qa(prompt: str) -> Tuple[str, str]:
    """Split canonical annotation prompt into question and answer (matches message_builder)."""
    for sep in (" Answer: ", "Answer: "):
        if sep in prompt:
            q, a = prompt.split(sep, 1)
            return q.strip(), a.strip()
    return prompt.strip(), ""


def template_to_pattern(template_line: str) -> str:
    return PLACEHOLDER_RE.sub(r"{{\1}}", template_line)


def template_family_id(template_id: str) -> str:
    if "." in template_id:
        parts = template_id.split(".")
        if len(parts) >= 2:
            return ".".join(parts[:2])
    return template_id


def slots_from_tag_bindings(bindings: Dict[str, Tuple[int, str]]) -> Dict[str, Dict[str, Any]]:
    """bindings: slot_id -> (obj_idx, tag)."""
    return {sid: {"obj_idx": idx, "tag": tag} for sid, (idx, tag) in bindings.items()}


def slots_from_mark_spec(mark_spec: Optional[dict]) -> Dict[str, Dict[str, Any]]:
    if not mark_spec:
        return {}
    from .mark_spec import all_slots_flat
    return {
        s["slot_id"]: {"obj_idx": s["obj_idx"], "tag": s["tag"]}
        for s in all_slots_flat(mark_spec)
        if isinstance(s, dict) and s.get("slot_id")
    }


def align_referent_slots(
    question_line: str,
    bindings: Optional[Dict[str, Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    """
    Map task-provided object bindings onto placeholder keys used in the sampled question.

    Example: counting MCQ uses [X] in the template but tasks may pass {"A": {obj_idx, tag}}.
    """
    if not bindings:
        return {}
    clean = {k: v for k, v in bindings.items() if isinstance(v, dict)}
    if not clean:
        return {}
    ph_keys = PromptTemplate.placeholders_in_line(question_line)
    if not ph_keys:
        return {}
    object_ph = [k for k in ph_keys if k not in _NON_OBJECT_PLACEHOLDERS]
    if len(object_ph) == 1 and len(clean) > 1:
        only_ph = object_ph[0]
        ordered = sorted(clean.keys())
        tags = [clean[k]["tag"] for k in ordered]
        first = clean[ordered[0]]
        return {only_ph: {"obj_idx": first.get("obj_idx", -1), "tag": ", ".join(tags)}}
    out: Dict[str, Dict[str, Any]] = {}
    for key in object_ph:
        if key in clean:
            out[key] = clean[key]
    if out:
        return out
    if len(object_ph) == 1 and len(clean) == 1:
        only_ph = object_ph[0]
        only_binding = next(iter(clean.values()))
        return {only_ph: only_binding}
    if len(object_ph) == len(clean) and len(object_ph) > 1:
        sorted_ph = sorted(object_ph)
        sorted_keys = sorted(clean.keys())
        if sorted_ph == sorted_keys:
            return dict(clean)
    return {}


def _sanitize_bindings(line: str, bindings: Dict[str, str]) -> Dict[str, str]:
    """Keep only placeholders present on the template line with filled literal values."""
    keys = set(PromptTemplate.placeholders_in_line(line))
    out: Dict[str, str] = {}
    for key, val in (bindings or {}).items():
        if key not in keys:
            continue
        if val is None:
            continue
        s = str(val).strip()
        if not s or "{{" in s:
            continue
        if s.startswith("[") and s.endswith("]") and len(s) <= 3:
            continue
        out[key] = s
    return out


def _sanitize_referent_slots(refs: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if not refs:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for key, val in refs.items():
        if isinstance(val, dict) and val.get("tag") is not None:
            out[key] = val
    return out


def build_prompt_struct(
    render: PromptRenderRecord,
    *,
    referent_slots: Optional[Dict[str, Dict[str, Any]]] = None,
    mark_spec: Optional[dict] = None,
) -> dict:
    """
    Build prompt_struct from render-time provenance.

    - question_pattern / answer_pattern: selected template lines (not index 0).
    - question_bindings / answer_bindings: values used when filling [X], [Y], etc.
    - referent_slots: scene-object referents keyed by placeholder letter in the question.
    """
    refs = _sanitize_referent_slots(
        align_referent_slots(render.question_line, referent_slots)
    )
    if not refs and mark_spec:
        refs = _sanitize_referent_slots(slots_from_mark_spec(mark_spec))
    object_ph = [
        k for k in PromptTemplate.placeholders_in_line(render.question_line)
        if k not in _NON_OBJECT_PLACEHOLDERS
    ]
    if object_ph:
        refs = {k: v for k, v in refs.items() if k in object_ph}

    out = {
        "template_id": render.template_id,
        "template_family": template_family_id(render.template_id),
        "question_index": render.question_index,
        "answer_index": render.answer_index,
        "question_pattern": template_to_pattern(render.question_line),
        "answer_pattern": template_to_pattern(render.answer_line) or "{{answer}}",
        "question_bindings": _sanitize_bindings(render.question_line, render.question_bindings) or {},
        "answer_bindings": _sanitize_bindings(render.answer_line, render.answer_bindings) or {},
        "referent_slots": refs,
    }
    if render.question_type:
        out["question_type_enum"] = render.question_type
    if render.instruction_type:
        out["instruction_type"] = render.instruction_type
    if render.introduction_index >= 0:
        out["introduction_index"] = render.introduction_index
    if render.question_instruction_index >= 0:
        out["question_instruction_index"] = render.question_instruction_index
    return out


def build_depth_prompt_struct(template_id: str, mark_spec: Optional[dict], **kwargs) -> dict:
    """Deprecated alias; prefer build_prompt_struct(render=...)."""
    raise NotImplementedError("Use build_prompt_struct(PromptRenderRecord, ...) instead")


def _strip_legacy_segments(text: str) -> str:
    """Remove -(color box/mask/point) suffixes from a filled prompt fragment."""
    out = _SLOT_PATTERN_RE.sub("", text)
    return re.sub(r"  +", " ", out).strip()


def materialize_pattern(
    pattern: str,
    slot_tags: Dict[str, str],
    shared: Optional[Dict[str, str]] = None,
) -> str:
    """Fill {{K}} placeholders; slot_tags supply object names for referent keys."""
    text = pattern
    merged = dict(shared or {})
    if slot_tags and "A" not in merged and "X" not in merged:
        merged.setdefault("A", ", ".join(slot_tags[s] for s in sorted(slot_tags)))
    if shared:
        merged.update(shared)
    for key, val in merged.items():
        text = text.replace(f"{{{{{key}}}}}", str(val))
    for sid, tag in slot_tags.items():
        text = text.replace(f"{{{{{sid}}}}}", tag)
    return text


def build_turn_record(
    *,
    turn_id: int,
    task_name: str,
    sub_task: str,
    question_type: str,
    template_id: str,
    prompt: str,
    render: Optional[PromptRenderRecord] = None,
    mark_spec: Optional[dict] = None,
    referent_mode: str = "semantic",
    instruction_mode: str = "legacy",
    referent_slots: Optional[Dict[str, Dict[str, Any]]] = None,
    extra_slots: Optional[Dict[str, Dict[str, Any]]] = None,
    question_prefix: Optional[str] = None,
    image_placeholder_count: int = 1,
    answer_text: Optional[str] = None,
    sorted_semantic: Optional[List[str]] = None,
    coord_tags: Optional[List[str]] = None,
    type_label: Optional[str] = None,
) -> dict:
    if render is None:
        q, a = split_prompt_qa(prompt)
        render = PromptRenderRecord(
            template_id=template_id,
            question_index=-1,
            answer_index=-1,
            question_line="",
            answer_line="",
            question_text=q,
            answer_text=a,
            question_bindings={},
            answer_bindings={},
        )

    slot_bindings = referent_slots if referent_slots is not None else extra_slots
    if slot_bindings is None and coord_tags is not None:
        slot_bindings = {
            chr(ord("A") + i): {"obj_idx": -1, "tag": str(t)}
            for i, t in enumerate(coord_tags)
        }
    prompt_struct = build_prompt_struct(
        render,
        referent_slots=align_referent_slots(render.question_line, slot_bindings),
        mark_spec=mark_spec,
    )

    if (
        referent_mode == "semantic"
        and _SLOT_PATTERN_RE.search(render.question_text)
        and prompt_struct.get("referent_slots")
    ):
        refs = prompt_struct["referent_slots"]
        object_ph = [
            k for k in PromptTemplate.placeholders_in_line(render.question_line)
            if k not in _NON_OBJECT_PLACEHOLDERS
        ]
        ref_keys = set(refs)
        shared_fill = {}
        for key, val in render.question_bindings.items():
            if key in ref_keys and not (len(object_ph) == 1 and len(refs) > 1):
                continue
            # Keep legacy mark suffixes in display text (messages) so Q/A matches
            # the fully marked visual style (e.g. "door-(red box)").
            shared_fill[key] = str(val)
        if len(object_ph) == 1 and len(refs) > 1:
            ph = object_ph[0]
            raw = render.question_bindings.get(ph)
            if raw and not _SLOT_PATTERN_RE.search(str(raw)):
                shared_fill[ph] = str(raw)
            else:
                ordered = sorted(refs.keys())
                shared_fill[ph] = ", ".join(refs[k]["tag"] for k in ordered)
        else:
            for key, binding in refs.items():
                if key in object_ph:
                    shared_fill[key] = binding["tag"]
        question_text = materialize_pattern(
            prompt_struct["question_pattern"], {}, shared=shared_fill,
        )
    else:
        question_text = render.question_text
    if question_prefix:
        question_text = f"{question_prefix.strip()}\n\n{question_text}"

    if answer_text is not None:
        final_answer = answer_text
    elif sorted_semantic is not None:
        # Prefer full template answer (e.g. depth ordering); comma-join is legacy fallback.
        final_answer = (
            render.answer_text
            if render and render.answer_text
            else ", ".join(str(t) for t in sorted_semantic)
        )
    else:
        final_answer = render.answer_text

    # semantic mode may intentionally keep legacy mark suffixes in display strings.

    turn: Dict[str, Any] = {
        "turn_id": turn_id,
        "task_name": task_name,
        "sub_task": sub_task,
        "question_type": question_type,
        "instruction_mode": (
            "structured" if render and render.instruction_type else instruction_mode
        ),
        "referent_mode": referent_mode,
        "prompt_struct": prompt_struct,
        "question_text": question_text,
        "answer_text": final_answer,
        "image_placeholder_count": image_placeholder_count,
    }
    if mark_spec is not None:
        turn["mark_spec"] = mark_spec
    if question_prefix:
        turn["question_prefix"] = question_prefix
    if type_label:
        turn["type_label"] = type_label
    return turn


def build_visual_anchor(
    example: dict,
    *,
    raw_image_ref: Optional[str] = None,
    view_indices: Optional[List[int]] = None,
) -> dict:
    parent_id = (
        example.get("parent_preprocess_id")
        or example.get("preprocess_id")
        or example.get("id")
        or example.get("scene_id")
    )
    img = example.get("image")
    anchor: Dict[str, Any] = {
        "parent_preprocess_id": str(parent_id) if parent_id is not None else "unknown",
    }
    if isinstance(img, list):
        if view_indices is not None:
            refs = [str(img[vi]) for vi in view_indices if vi < len(img)]
        else:
            refs = [str(p) for p in img]
        anchor["raw_image_refs"] = refs
        anchor["view_group_id"] = str(example.get("scene_id") or parent_id or "unknown")
    else:
        ref = raw_image_ref or example.get("raw_image_ref") or str(img or "raw.jpg")
        anchor["raw_image_ref"] = ref
    if example.get("scene_id") is not None:
        anchor["scene_id"] = str(example["scene_id"])
    if example.get("frame_id") is not None:
        anchor["frame_id"] = str(example["frame_id"])
    return anchor


def _single_image_ref(example: dict) -> Optional[str]:
    img = example.get("image")
    if isinstance(img, list) and img:
        return str(img[0]).replace("\\", "/")
    if isinstance(img, str) and img.strip():
        return str(img).replace("\\", "/")
    return None


def qa_image_refs_for_turn(example: dict, turn: dict) -> List[str]:
    """
    Canonical QA image paths for one turn (dict semantics: one ref per placeholder).

    Priority: mark_spec.views[].image_ref → view_indices into scene list →
    scene list when len matches placeholder count → single-image ref.
    """
    from .mark_spec import mark_spec_views

    n = int(turn.get("image_placeholder_count") or 1)
    ms = turn.get("mark_spec")
    views = mark_spec_views(ms) if ms else []
    refs_from_spec = [
        str(v["image_ref"]).replace("\\", "/")
        for v in sorted(views, key=lambda x: int(x.get("view_index", 0)))
        if v.get("image_ref")
    ]
    if refs_from_spec and len(refs_from_spec) == n:
        return refs_from_spec

    view_indices = turn.get("view_indices")
    img = example.get("image")
    if isinstance(img, list) and view_indices is not None:
        return [str(img[vi]).replace("\\", "/") for vi in view_indices if vi < len(img)]

    if isinstance(img, list) and len(img) == n:
        return [str(p).replace("\\", "/") for p in img]

    ref = _single_image_ref(example)
    return [ref] if ref else []


def build_turn_metadata(example: dict, turn: dict) -> dict:
    """Build export metadata for one QA turn (visual_anchor + mark_spec + turns)."""
    from .mark_spec import align_mark_spec_to_image_refs, mark_spec_views, wrap_single_view_mark_spec

    export_turn = {k: v for k, v in turn.items() if k not in ("mark_spec", "view_indices")}
    ms = turn.get("mark_spec")
    qa_refs = qa_image_refs_for_turn(example, turn)
    n = int(turn.get("image_placeholder_count") or max(1, len(qa_refs)))

    if ms and ms.get("slots") and not ms.get("views"):
        ms = wrap_single_view_mark_spec(
            ms, image_ref=qa_refs[0] if qa_refs else _single_image_ref(example),
        )
    elif ms and ms.get("layout") == "per_view":
        views = mark_spec_views(ms)
        if len(views) == 1 and qa_refs and not views[0].get("image_ref"):
            views = [dict(views[0], image_ref=qa_refs[0])]
            ms = {**ms, "views": views}

    if ms and qa_refs:
        ms = align_mark_spec_to_image_refs(ms, qa_refs)

    parent_id = (
        example.get("parent_preprocess_id")
        or example.get("preprocess_id")
        or example.get("id")
        or example.get("scene_id")
    )
    anchor: Dict[str, Any] = {
        "parent_preprocess_id": str(parent_id) if parent_id is not None else "unknown",
    }
    if n == 1 and qa_refs:
        anchor["raw_image_ref"] = qa_refs[0]
    elif qa_refs:
        anchor["raw_image_refs"] = qa_refs
        anchor["view_group_id"] = str(example.get("scene_id") or parent_id or "unknown")
    else:
        anchor = build_visual_anchor(
            example,
            raw_image_ref=_single_image_ref(example),
            view_indices=turn.get("view_indices"),
        )

    if example.get("scene_id") is not None:
        anchor["scene_id"] = str(example["scene_id"])
    if example.get("frame_id") is not None:
        anchor["frame_id"] = str(example["frame_id"])

    return {"visual_anchor": anchor, "mark_spec": ms, "turns": [export_turn]}


def build_example_metadata(
    example: dict,
    turn_records: List[dict],
    *,
    mark_spec: Optional[dict] = None,
) -> dict:
    from .mark_spec import mark_spec_has_slots

    if mark_spec is None:
        for tr in turn_records:
            ms = tr.get("mark_spec")
            if mark_spec_has_slots(ms):
                mark_spec = ms
                break

    if len(turn_records) == 1:
        meta = build_turn_metadata(example, turn_records[0])
        if mark_spec is not None and meta.get("mark_spec") is None:
            meta["mark_spec"] = mark_spec
        return meta

    export_turns = [{k: v for k, v in tr.items() if k != "mark_spec"} for tr in turn_records]
    anchor = build_visual_anchor(example, raw_image_ref=_single_image_ref(example))
    return {"visual_anchor": anchor, "mark_spec": mark_spec, "turns": export_turns}
