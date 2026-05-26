"""Dedup / merge key computation (plan §4.2.1–4.2.2)."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Tuple

_LEGACY_MARKED_DESC = re.compile(r"-\(\w+\s+(box|mask|point)\)", re.I)

REFERENT_PRIORITY = {"semantic": 0, "alias": 1, "marked": 2, "legacy": 3}


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_hex(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_image_ref(path: Any) -> str:
    if path is None:
        return "unknown"
    return str(path).replace("\\", "/")


def image_refs_from_row(row: dict) -> List[str]:
    meta = row.get("metadata")
    if isinstance(meta, list) and meta:
        anchor = meta[0] if isinstance(meta[0], dict) else {}
    elif isinstance(meta, dict):
        anchor = meta
    else:
        anchor = {}
    va = anchor.get("visual_anchor") if isinstance(anchor, dict) else {}
    if isinstance(va, dict):
        refs = va.get("raw_image_refs")
        if refs is not None and len(list(refs)) > 0:
            return [normalize_image_ref(p) for p in refs]
        if va.get("raw_image_ref"):
            return [normalize_image_ref(va["raw_image_ref"])]
    img = row.get("image")
    if isinstance(img, list):
        return [normalize_image_ref(p) for p in img]
    return [normalize_image_ref(img)]


def mark_spec_has_slots(mark_spec: Any) -> bool:
    if not isinstance(mark_spec, dict):
        return False
    from task.annotation.core.mark_spec import mark_spec_has_slots as _has
    return _has(mark_spec)


def _norm_slot(s: dict) -> dict:
    geom = s.get("geometry") or {}
    if not isinstance(geom, dict):
        geom = {}
    gnorm: Dict[str, Any] = {}
    box = geom.get("box_2d")
    if box is not None:
        if hasattr(box, "tolist"):
            box = box.tolist()
        gnorm["box_2d"] = [round(float(v), 4) for v in box]
    uv = geom.get("uv")
    if uv is not None:
        if hasattr(uv, "tolist"):
            uv = uv.tolist()
        gnorm["uv"] = [int(uv[0]), int(uv[1])]
    mask_ref = geom.get("mask_ref")
    if isinstance(mask_ref, dict) and mask_ref:
        gnorm["mask_ref"] = dict(sorted(mask_ref.items()))
    return {
        "slot_id": s.get("slot_id"),
        "obj_idx": s.get("obj_idx"),
        "tag": s.get("tag"),
        "mark_kind": s.get("mark_kind"),
        "color_name": s.get("color_name"),
        "geometry": gnorm,
    }


def mark_spec_norm(mark_spec: Optional[dict]) -> Optional[dict]:
    if not isinstance(mark_spec, dict):
        return None
    from task.annotation.core.mark_spec import mark_spec_views
    views_raw = mark_spec_views(mark_spec)
    if not views_raw:
        return None
    views = []
    for v in sorted(views_raw, key=lambda x: int(x.get("view_index", 0))):
        slots_raw = v.get("slots") or []
        slots = []
        for s in sorted(slots_raw, key=lambda x: (x.get("slot_id", ""), x.get("obj_idx", 0))):
            if isinstance(s, dict):
                slots.append(_norm_slot(s))
        if not slots:
            continue
        views.append({
            "view_index": int(v.get("view_index", 0)),
            "image_ref": v.get("image_ref"),
            "mark_kinds": sorted({s.get("mark_kind") for s in slots if s.get("mark_kind")}),
            "slots": slots,
        })
    if not views:
        return None
    return {"version": mark_spec.get("version", 2), "layout": "per_view", "views": views}


# Template families where OE and MCQ are the same semantic question (dedup collapses _mcq).
_OE_MCQ_COLLAPSE_PREFIXES = (
    "counting.",
    "depth.ordering",
    "depth.farthest",
    "depth.closest",
    "depth.choice",
    "distance.relative",
)


def _normalized_intent_family(ps: dict, sub_task: str) -> str:
    fam = (ps.get("template_family") or ps.get("template_id") or "").strip()
    if fam.endswith("_mcq"):
        fam = fam[:-4]
    st = (sub_task or "").strip()
    if st.endswith("_mcq"):
        st = st[:-4]
    if st.endswith("_oe"):
        st = st[:-3]
    return fam or st


def _tags_from_marked_list(text: str) -> Tuple[str, ...]:
    parts = []
    for piece in str(text).split(","):
        piece = piece.strip()
        if "-(" in piece:
            piece = piece.split("-(", 1)[0].strip()
        if piece:
            parts.append(piece)
    return tuple(parts)


def _referent_signature(ps: dict) -> Any:
    refs = ps.get("referent_slots") or {}
    if isinstance(refs, dict) and refs:
        if len(refs) == 1:
            only = next(iter(refs.values()))
            if isinstance(only, dict):
                tag = str(only.get("tag", ""))
                if "," in tag:
                    return ("tags", _tags_from_marked_list(tag))
        return (
            "slots",
            tuple(
                (k, v.get("obj_idx"))
                for k, v in sorted(refs.items())
                if isinstance(v, dict)
            ),
        )
    qb = ps.get("question_bindings") or {}
    if qb.get("A"):
        tags = _tags_from_marked_list(str(qb["A"]))
        if tags:
            return ("tags", tags)
    for key in sorted(qb.keys()):
        val = qb.get(key)
        if val and key not in ("T", "Y", "O", "Z", "D"):
            return ("binding", key, str(val).split(",")[0].strip())
    return ("none",)


def _counting_target_tag(ps: dict) -> Optional[str]:
    refs = ps.get("referent_slots") or {}
    if isinstance(refs, dict) and refs:
        v = next(iter(refs.values()))
        if isinstance(v, dict) and v.get("tag"):
            return str(v["tag"]).split(",")[0].strip()
    qb = ps.get("question_bindings") or {}
    for key in ("X", "A"):
        if qb.get(key):
            return str(qb[key]).split(",")[0].strip()
    return None


def _should_collapse_oe_mcq(family: str) -> bool:
    return any(family.startswith(p) for p in _OE_MCQ_COLLAPSE_PREFIXES)


def compute_question_core_key(turn: dict) -> str:
    ps = turn.get("prompt_struct")
    if isinstance(ps, dict):
        family = _normalized_intent_family(ps, turn.get("sub_task", ""))
        if family.startswith("counting"):
            body = {
                "intent": "counting",
                "target_tag": _counting_target_tag(ps) or "",
            }
            return _sha256_hex(_canonical_json(body))
        if _should_collapse_oe_mcq(family):
            body = {
                "intent": family,
                "referents": _referent_signature(ps),
            }
            return _sha256_hex(_canonical_json(body))
    refs = None
    if isinstance(ps, dict):
        refs = ps.get("referent_slots") or ps.get("slots")
    if isinstance(ps, dict) and refs:
        bindings = {k: v.get("obj_idx") for k, v in sorted(refs.items()) if isinstance(v, dict)}
        body = {
            "sub_task": turn.get("sub_task", ""),
            "template_family": ps.get("template_family") or ps.get("template_id", ""),
            "question_index": ps.get("question_index"),
            "slot_bindings": bindings,
            "question_type": turn.get("question_type", ""),
        }
        if turn.get("question_type") == "MCQ":
            ab = ps.get("answer_bindings") or {}
            ans = ab.get("X") or ab.get("E") or turn.get("answer_text") or ""
            body["mcq_answer_target"] = _normalize_mcq_answer(str(ans))
        return _sha256_hex(_canonical_json(body))

    q = turn.get("question_text") or ""
    q = _LEGACY_MARKED_DESC.sub("", q).strip()
    body = {
        "sub_task": turn.get("sub_task", ""),
        "question_type": turn.get("question_type", ""),
        "question_text_norm": q,
    }
    return _sha256_hex(_canonical_json(body))


def _normalize_mcq_answer(answer_text: str) -> str:
    a = (answer_text or "").strip()
    if len(a) == 1 and a in "ABCD":
        return a
    return a


def visual_anchor_key(row: dict, turn: Optional[dict] = None) -> str:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    va = meta.get("visual_anchor") if isinstance(meta, dict) else {}
    if isinstance(va, dict) and va.get("parent_preprocess_id"):
        return str(va["parent_preprocess_id"])
    for key in ("id", "scene_id", "parent_preprocess_id"):
        if row.get(key) is not None:
            return str(row[key])
    return normalize_image_ref(row.get("image"))


def compute_dedup_fingerprint(task_name: str, row: dict, turn: dict) -> str:
    body = {
        "task_name": task_name,
        "visual_anchor": visual_anchor_key(row, turn),
        "view_group_id": (row.get("metadata") or {}).get("visual_anchor", {}).get("view_group_id"),
        "question_core_key": turn.get("question_core_key") or compute_question_core_key(turn),
    }
    return _sha256_hex(_canonical_json(body))


def compute_merge_group_key(row: dict, turn: Optional[dict] = None) -> str:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    ms = None
    if turn and turn.get("mark_spec"):
        ms = turn["mark_spec"]
    elif isinstance(meta, dict):
        ms = meta.get("mark_spec")
    va = meta.get("visual_anchor") if isinstance(meta, dict) else {}
    body = {
        "image_refs_ordered": image_refs_from_row(row),
        "view_group_id": va.get("view_group_id") if isinstance(va, dict) else None,
        "mark_spec_norm": mark_spec_norm(ms),
    }
    return _sha256_hex(_canonical_json(body))


def pick_dedup_winner(candidates: List[dict], policy: str = "semantic_first") -> dict:
    if policy != "semantic_first":
        return candidates[0]

    def sort_key(t: dict):
        has_struct = 0 if t.get("prompt_struct") else 1
        tpl = (t.get("prompt_struct") or {}).get("template_id") or ""
        return (has_struct, tpl)

    return min(candidates, key=sort_key)
