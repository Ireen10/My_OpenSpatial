"""Validate sample-level metadata records (plan M1)."""

from __future__ import annotations

import math
import re
from typing import Any, List, Tuple


def _is_missing(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    return False


def _nonempty_field(val: Any) -> bool:
    """Safe truthiness for str / list / numpy array (multiview raw_image_refs)."""
    if _is_missing(val):
        return False
    import numpy as np
    if isinstance(val, np.ndarray):
        return val.size > 0
    if isinstance(val, (list, tuple, str)):
        return len(val) > 0
    return bool(val)

SCHEMA_VERSION = "1.1"
MARK_KINDS = frozenset({"box", "mask", "point"})
MESSAGE_ROLES = frozenset({"human", "gpt"})
REFERENT_MODES = frozenset({"semantic", "marked", "alias", "legacy"})

_LEGACY_MARKED_DESC = re.compile(r"-\(\w+\s+(box|mask|point)\)", re.I)


def _err(path: str, msg: str) -> str:
    return f"{path}: {msg}"


def _require_dict(obj: Any, path: str, errors: List[str]) -> dict | None:
    if not isinstance(obj, dict):
        errors.append(_err(path, "must be object"))
        return None
    return obj


def _validate_visual_anchor(va: Any, path: str, errors: List[str]) -> None:
    d = _require_dict(va, path, errors)
    if d is None:
        return
    if not d.get("parent_preprocess_id"):
        errors.append(_err(path, "parent_preprocess_id required"))
    if not _nonempty_field(d.get("raw_image_ref")) and not _nonempty_field(d.get("raw_image_refs")):
        errors.append(_err(path, "raw_image_ref or raw_image_refs required"))


def _validate_geometry(slot: dict, path: str, errors: List[str]) -> None:
    kind = slot.get("mark_kind")
    geom = slot.get("geometry")
    g = _require_dict(geom, f"{path}.geometry", errors)
    if g is None:
        return
    if kind == "box":
        b = g.get("box_2d")
        if _is_missing(b):
            errors.append(_err(path, "box requires geometry.box_2d[4]"))
        else:
            if hasattr(b, "tolist"):
                b = b.tolist()
            if not isinstance(b, list) or len(b) != 4:
                errors.append(_err(path, "box requires geometry.box_2d[4]"))
    elif kind == "mask":
        ref = g.get("mask_ref")
        if not isinstance(ref, dict) or not ref.get("source"):
            errors.append(_err(path, "mask requires geometry.mask_ref.source"))
        else:
            src = ref.get("source")
            if src == "path" and not ref.get("path"):
                errors.append(_err(path, "mask_ref.path required when source=path"))
            elif src == "preprocess" and ref.get("obj_idx") is None:
                errors.append(_err(path, "mask_ref.obj_idx required when source=preprocess"))
            elif src == "tar" and not ref.get("tar_key"):
                errors.append(_err(path, "mask_ref.tar_key required when source=tar"))
            elif src not in ("path", "preprocess", "tar"):
                errors.append(_err(path, f"unknown mask_ref.source: {src!r}"))
    elif kind == "point":
        uv = g.get("uv")
        if _is_missing(uv):
            errors.append(_err(path, "point requires geometry.uv[2]"))
        else:
            if hasattr(uv, "tolist"):
                uv = uv.tolist()
            if not isinstance(uv, list) or len(uv) != 2:
                errors.append(_err(path, "point requires geometry.uv[2]"))


def _validate_mark_spec(ms: Any, path: str, errors: List[str]) -> None:
    if ms is None:
        return
    d = _require_dict(ms, path, errors)
    if d is None:
        return
    if d.get("version") != 2:
        errors.append(_err(path, "version must be 2"))

    from task.annotation.core.mark_spec import mark_spec_views
    views = mark_spec_views(d)
    if not views:
        errors.append(_err(path, "mark_spec.views must be non-empty (per_view layout)"))
        return
    for vi, view in enumerate(views):
        vp = f"{path}.views[{vi}]"
        vdict = _require_dict(view, vp, errors)
        if vdict is None:
            continue
        if "view_index" not in vdict:
            errors.append(_err(vp, "missing view_index"))
        slots = vdict.get("slots")
        if not isinstance(slots, list) or len(slots) == 0:
            errors.append(_err(vp, "slots must be non-empty array for this view"))
            continue
        for i, slot in enumerate(slots):
            sp = f"{vp}.slots[{i}]"
            s = _require_dict(slot, sp, errors)
            if s is None:
                continue
            for key in ("slot_id", "obj_idx", "tag", "mark_kind", "color_name"):
                if key not in s:
                    errors.append(_err(sp, f"missing {key}"))
            if s.get("mark_kind") not in MARK_KINDS:
                errors.append(_err(sp, f"invalid mark_kind {s.get('mark_kind')!r}"))
            _validate_geometry(s, sp, errors)


def _validate_prompt_struct(ps: Any, path: str, errors: List[str]) -> None:
    d = _require_dict(ps, path, errors)
    if d is None:
        return
    if not d.get("template_id"):
        errors.append(_err(path, "template_id required"))
    if not d.get("question_pattern"):
        errors.append(_err(path, "question_pattern required"))
    if not d.get("answer_pattern"):
        errors.append(_err(path, "answer_pattern required"))
    if "question_index" not in d:
        errors.append(_err(path, "question_index required"))
    if "answer_index" not in d:
        errors.append(_err(path, "answer_index required"))
    q_bind = d.get("question_bindings")
    if not isinstance(q_bind, dict):
        errors.append(_err(path, "question_bindings must be object"))
    else:
        for key, val in q_bind.items():
            if not isinstance(key, str) or not key:
                errors.append(_err(path, f"invalid question_bindings key {key!r}"))
            if _is_missing(val):
                continue
            if "{{" in str(val):
                errors.append(_err(path, f"question_bindings[{key!r}] must be filled literal"))
    a_bind = d.get("answer_bindings")
    if a_bind is not None and not isinstance(a_bind, dict):
        errors.append(_err(path, "answer_bindings must be object"))
    refs = d.get("referent_slots")
    if refs is None:
        refs = d.get("slots")
    if refs is not None and not isinstance(refs, dict):
        errors.append(_err(path, "referent_slots must be object"))
    elif isinstance(refs, dict):
        for sid, binding in refs.items():
            if _is_missing(binding) or not isinstance(binding, dict):
                continue
            bp = f"{path}.referent_slots.{sid}"
            b = _require_dict(binding, bp, errors)
            if b is None:
                continue
            if "obj_idx" not in b or "tag" not in b:
                errors.append(_err(bp, "requires obj_idx and tag"))


def _validate_turn(t: Any, path: str, errors: List[str], *, strict_m3: bool) -> None:
    d = _require_dict(t, path, errors)
    if d is None:
        return
    for key in ("turn_id", "task_name", "sub_task", "question_type", "instruction_mode",
                "question_text", "answer_text"):
        if key not in d:
            errors.append(_err(path, f"missing {key}"))
    mode = d.get("referent_mode", "semantic")
    if mode not in REFERENT_MODES:
        errors.append(_err(path, f"invalid referent_mode {mode!r}"))
    if strict_m3 and mode == "semantic":
        qt = d.get("question_text") or ""
        if _LEGACY_MARKED_DESC.search(qt):
            errors.append(_err(path, "semantic question_text must not contain legacy -(color mark) suffix"))
    if strict_m3 and "prompt_struct" not in d:
        errors.append(_err(path, "prompt_struct required (M3+)"))


def validate_sample_record(record: dict, *, strict_m3: bool = False) -> Tuple[bool, List[str]]:
    """Return (ok, errors)."""
    errors: List[str] = []
    if record.get("schema_version") != SCHEMA_VERSION:
        errors.append(_err("schema_version", f"expected {SCHEMA_VERSION}"))
    for key in ("sample_id", "merge_group_key", "image_refs", "messages", "metadata"):
        if key not in record:
            errors.append(_err(key, "missing"))
    if not isinstance(record.get("image_refs"), list) or len(record["image_refs"]) == 0:
        errors.append(_err("image_refs", "must be non-empty array"))
    msgs = record.get("messages")
    if isinstance(msgs, list):
        for i, m in enumerate(msgs):
            mp = f"messages[{i}]"
            md = _require_dict(m, mp, errors)
            if md is None:
                continue
            if md.get("from") not in MESSAGE_ROLES:
                errors.append(_err(mp, "from must be human or gpt"))
            if not md.get("value"):
                errors.append(_err(mp, "value required"))
    meta = record.get("metadata")
    md = _require_dict(meta, "metadata", errors)
    if md is not None:
        _validate_visual_anchor(md.get("visual_anchor"), "metadata.visual_anchor", errors)
        _validate_mark_spec(md.get("mark_spec"), "metadata.mark_spec", errors)
        turns = md.get("turns")
        if not isinstance(turns, list) or len(turns) == 0:
            errors.append(_err("metadata.turns", "must be non-empty array"))
        elif isinstance(turns, list):
            for i, t in enumerate(turns):
                _validate_turn(t, f"metadata.turns[{i}]", errors, strict_m3=strict_m3)
                if strict_m3:
                    ps = t.get("prompt_struct") if isinstance(t, dict) else None
                    if ps is not None:
                        _validate_prompt_struct(ps, f"metadata.turns[{i}].prompt_struct", errors)
    return len(errors) == 0, errors


def _fixture_minimal() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": "00000000-0000-4000-8000-000000000001",
        "merge_group_key": "abc123",
        "image_refs": ["images.tar/scene/0/raw.jpg"],
        "messages": [
            {"from": "human", "value": "<image> What is the distance between the chair and the table?"},
            {"from": "gpt", "value": "The distance is 2.35 m."},
        ],
        "metadata": {
            "visual_anchor": {
                "parent_preprocess_id": "scene-0-frame-0",
                "scene_id": "scene-0",
                "raw_image_ref": "images.tar/scene/0/raw.jpg",
            },
            "mark_spec": {
                "version": 2,
                "layout": "per_view",
                "views": [{
                    "view_index": 0,
                    "mark_kinds": ["box"],
                    "slots": [
                    {
                        "slot_id": "A",
                        "obj_idx": 0,
                        "tag": "chair",
                        "mark_kind": "box",
                        "color_name": "red",
                        "geometry": {"box_2d": [10, 20, 100, 200]},
                    },
                    {
                        "slot_id": "B",
                        "obj_idx": 1,
                        "tag": "table",
                        "mark_kind": "mask",
                        "color_name": "blue",
                        "geometry": {"mask_ref": {"source": "preprocess", "obj_idx": 1}},
                    },
                ],
                }],
            },
            "turns": [
                {
                    "turn_id": 0,
                    "task_name": "distance",
                    "sub_task": "absolute_distance",
                    "question_type": "OE",
                    "instruction_mode": "legacy",
                    "referent_mode": "semantic",
                    "question_text": "What is the distance between the chair and the table?",
                    "answer_text": "The distance is 2.35 m.",
                    "image_placeholder_count": 1,
                    "prompt_struct": {
                        "template_id": "distance.absolute_m",
                        "template_family": "spatial.distance.absolute",
                        "question_index": 0,
                        "answer_index": 0,
                        "question_pattern": "What is the distance between {{A}} and {{B}}?",
                        "answer_pattern": "{{answer}}",
                        "question_bindings": {"A": "chair", "B": "table"},
                        "answer_bindings": {},
                        "referent_slots": {
                            "A": {"obj_idx": 0, "tag": "chair"},
                            "B": {"obj_idx": 1, "tag": "table"},
                        },
                    },
                },
            ],
        },
    }


def _fixture_grounding_prefix() -> dict:
    rec = _fixture_minimal()
    rec["sample_id"] = "00000000-0000-4000-8000-000000000002"
    prefix = (
        "Focal length f_x=1170.19, f_y=1170.19. "
        "bbox_3d: [x_center,y_center,z_center,x_size,y_size,z_size,roll,pitch,yaw]"
    )
    rec["messages"][0]["value"] = f"<image> {prefix}\nPredict 3D boxes for the chair."
    rec["metadata"]["turns"][0]["task_name"] = "3d_grounding"
    rec["metadata"]["turns"][0]["question_prefix"] = prefix
    if "roll,pitch,yaw" not in prefix.replace(" ", ""):
        raise ValueError("fixture prefix must use roll,pitch,yaw")
    return rec


def _fixture_invalid() -> dict:
    rec = _fixture_minimal()
    rec["metadata"]["turns"][0]["question_text"] = "Distance between chair-(red box) and table?"
    rec["metadata"]["turns"][0]["referent_mode"] = "semantic"
    return rec


def run_self_check() -> bool:
    ok1, e1 = validate_sample_record(_fixture_minimal())
    ok2, e2 = validate_sample_record(_fixture_grounding_prefix())
    ok_bad, e_bad = validate_sample_record(_fixture_invalid(), strict_m3=True)
    if not ok1:
        print("M1 FAIL: minimal fixture:", e1)
        return False
    if not ok2:
        print("M1 FAIL: grounding fixture:", e2)
        return False
    if ok_bad:
        print("M1 FAIL: invalid fixture should fail strict_m3")
        return False
    if not e_bad:
        print("M1 FAIL: expected errors for legacy marked desc in semantic text")
        return False
    print("M1 PASS: schema docs + validate_sample_record + 3 fixtures")
    return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_self_check() else 1)
