"""
Sample-level metadata helpers (plan M3+).

Canonical training surface: metadata.turns[].prompt_struct (+ mark_spec, anchors).
Visualization surface only: parquet messages[] (marked Q/A matching on-image marks).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .prompt_template import PLACEHOLDER_RE, PromptRenderRecord, PromptTemplate, TemplateRegistry

import task.prompt_templates  # noqa: F401

MARK_SUFFIX_RE = re.compile(r"-\(\w+\s+(box|mask|point)\)", re.I)
_NON_OBJECT_PLACEHOLDERS = frozenset({"Y", "O", "Z", "T", "D"})


def split_prompt_qa(prompt: str) -> Tuple[str, str]:
    """Split legacy combined prompt string into question and answer."""
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


def has_mark_suffix(text: str) -> bool:
    return bool(text and MARK_SUFFIX_RE.search(text))


def display_label_from_mark_slot(slot: dict) -> str:
    tag = str(slot.get("tag", "")).strip()
    kind = slot.get("mark_kind")
    color = slot.get("color_name")
    if tag and kind and color:
        return f"{tag}-({color} {kind})"
    return tag


def apply_mark_spec_labels_to_text(text: str, mark_spec: Optional[dict]) -> str:
    """Upgrade bare object tags to marked surface forms wherever mark_spec defines a slot."""
    if not text or not mark_spec:
        return text or ""
    from .mark_spec import all_slots_flat

    out = text
    for slot in all_slots_flat(mark_spec):
        tag = str(slot.get("tag", "")).strip()
        if not tag:
            continue
        label = display_label_from_mark_slot(slot)
        if label == tag or label.lower() in out.lower():
            continue
        out = re.sub(
            rf"\b{re.escape(tag)}\b(?!\s*-\()",
            label,
            out,
            flags=re.IGNORECASE,
        )
    return re.sub(r"  +", " ", out).strip()


def marked_surface_label(marked) -> str:
    """Text shown when marks are on-image (tag or tag-(color box|point))."""
    if not isinstance(marked, (list, tuple)) or len(marked) < 1:
        return str(marked).strip().lower()
    return str(marked[0]).strip().lower()


def semantic_object_label(marked) -> str:
    """Semantic tag only (for prompt_struct.referent_slots, not for messages)."""
    from .scene_graph import SceneNode

    if not isinstance(marked, (list, tuple)) or len(marked) < 2:
        raw = str(marked).strip()
        return raw.split("-(", 1)[0].strip().lower() if "-(" in raw else raw.lower()
    desc, node = marked[0], marked[1]
    if isinstance(node, SceneNode):
        return node.tag.strip().lower()
    raw = str(desc).strip()
    if "-(" in raw:
        return raw.split("-(", 1)[0].strip().lower()
    return raw.lower()


def align_referent_slots(
    question_line: str,
    bindings: Optional[Dict[str, Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
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
    refs = _sanitize_referent_slots(
        align_referent_slots(render.question_line, referent_slots)
    )
    if not refs and mark_spec:
        from .mark_spec import all_slots_flat
        refs = _sanitize_referent_slots({
            s["slot_id"]: {"obj_idx": s["obj_idx"], "tag": s["tag"]}
            for s in all_slots_flat(mark_spec)
            if isinstance(s, dict) and s.get("slot_id")
        })
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
    if render.constraint_mode:
        out["constraint_mode"] = render.constraint_mode
    if render.introduction_index >= 0:
        out["introduction_index"] = render.introduction_index
    if render.question_instruction_index >= 0:
        out["question_instruction_index"] = render.question_instruction_index
    return out


def build_viz_qa(
    render: PromptRenderRecord,
    *,
    mark_spec: Optional[dict] = None,
    question_prefix: Optional[str] = None,
    answer_text: Optional[str] = None,
    sorted_semantic: Optional[List[str]] = None,
    image_placeholder_count: int = 1,
) -> dict:
    """
  Marked Q/A strings for messages[] only (visualization / human review).

  Uses render-time literals (including chair-(red box) when marks were planned).
    """
    q = (render.question_text or "").strip()
    if answer_text is not None:
        a = answer_text
    elif sorted_semantic is not None:
        a = render.answer_text if render.answer_text else ", ".join(str(t) for t in sorted_semantic)
    else:
        a = render.answer_text or ""
    prefix = (question_prefix or "").strip() or None
    if prefix:
        q = f"{prefix}\n\n{q}" if q else prefix
    return {
        "question": apply_mark_spec_labels_to_text(q, mark_spec).strip(),
        "answer": apply_mark_spec_labels_to_text(str(a), mark_spec).strip(),
        "question_prefix": prefix,
        "image_placeholder_count": max(1, int(image_placeholder_count)),
    }


def build_metadata_turn(
    *,
    turn_id: int,
    task_name: str,
    sub_task: str,
    question_type: str,
    template_id: str,
    render: PromptRenderRecord,
    mark_spec: Optional[dict] = None,
    referent_slots: Optional[Dict[str, Dict[str, Any]]] = None,
    extra_slots: Optional[Dict[str, Dict[str, Any]]] = None,
    coord_tags: Optional[List[str]] = None,
    instruction_mode: str = "legacy",
    question_prefix: Optional[str] = None,
    image_placeholder_count: int = 1,
    type_label: Optional[str] = None,
) -> dict:
    """Canonical turn payload for metadata (no rendered Q/A prose)."""
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
    turn: Dict[str, Any] = {
        "turn_id": turn_id,
        "task_name": task_name,
        "sub_task": sub_task,
        "question_type": question_type,
        "instruction_mode": (
            "structured" if render.instruction_type else instruction_mode
        ),
        "prompt_struct": prompt_struct,
        "image_placeholder_count": image_placeholder_count,
    }
    if mark_spec is not None:
        turn["mark_spec"] = mark_spec
    if question_prefix:
        turn["question_prefix"] = question_prefix
    if type_label:
        turn["type_label"] = type_label
    return turn


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
    referent_slots: Optional[Dict[str, Dict[str, Any]]] = None,
    extra_slots: Optional[Dict[str, Dict[str, Any]]] = None,
    question_prefix: Optional[str] = None,
    image_placeholder_count: int = 1,
    answer_text: Optional[str] = None,
    sorted_semantic: Optional[List[str]] = None,
    coord_tags: Optional[List[str]] = None,
    type_label: Optional[str] = None,
    instruction_mode: str = "legacy",
    **_: Any,
) -> Tuple[dict, dict]:
    """Return (metadata_turn, viz_qa) — viz_qa is for messages[] only."""
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
    meta = build_metadata_turn(
        turn_id=turn_id,
        task_name=task_name,
        sub_task=sub_task,
        question_type=question_type,
        template_id=template_id,
        render=render,
        mark_spec=mark_spec,
        referent_slots=referent_slots,
        extra_slots=extra_slots,
        coord_tags=coord_tags,
        instruction_mode=instruction_mode,
        question_prefix=question_prefix,
        image_placeholder_count=image_placeholder_count,
        type_label=type_label,
    )
    viz = build_viz_qa(
        render,
        mark_spec=mark_spec,
        question_prefix=question_prefix,
        answer_text=answer_text,
        sorted_semantic=sorted_semantic,
        image_placeholder_count=image_placeholder_count,
    )
    return meta, viz


def build_visual_anchor(
    example: dict,
    *,
    raw_image_ref: Optional[str] = None,
    view_indices: Optional[List[int]] = None,
) -> dict:
    from .dataset_source import infer_dataset_source

    parent_id = (
        example.get("parent_preprocess_id")
        or example.get("preprocess_id")
        or example.get("id")
        or example.get("scene_id")
    )
    img = example.get("image")
    single_ref = raw_image_ref
    if single_ref is None and not isinstance(img, list) and img:
        single_ref = str(img)
    anchor: Dict[str, Any] = {
        "parent_preprocess_id": str(parent_id) if parent_id is not None else "unknown",
        "dataset_source": infer_dataset_source(
            explicit=example.get("dataset") or example.get("dataset_source"),
            parent_preprocess_id=parent_id,
            raw_image_ref=single_ref,
            raw_image_refs=img if isinstance(img, list) else None,
        ),
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
