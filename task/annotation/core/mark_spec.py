"""
Mark specification (v2) and rendering — decouple mark semantics from pixels.

Multiview layout: mark_spec.views[] — each entry is marks for ONE QA image only.
Single-view: one element in views (or legacy flat slots migrated on read).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from utils.image_utils import convert_pil_to_bytes

from .visual_marker import (
    COLOR_MAP,
    VisualMarker,
    draw_boxes_on_image,
    draw_masks_on_image,
    draw_points_on_image,
)

MARK_SPEC_VERSION = 2
LAYOUT_PER_VIEW = "per_view"


# ─── Layout helpers ────────────────────────────────────────────────────


def _pyify(obj: Any) -> Any:
    """Convert numpy / parquet nested values to plain Python."""
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        return _pyify(obj.tolist())
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _pyify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_pyify(x) for x in obj]
    if hasattr(obj, "tolist") and not isinstance(obj, (str, bytes)):
        return _pyify(obj.tolist())
    return obj


def mark_spec_views(mark_spec: Optional[dict]) -> List[dict]:
    """
    Return per-image view entries: {view_index, image_ref?, mark_kinds?, slots[]}.

    Migrates legacy flat mark_spec.slots with optional slot.view_index.
    """
    if not mark_spec or not isinstance(mark_spec, dict):
        return []
    mark_spec = _pyify(mark_spec)
    if not isinstance(mark_spec, dict):
        return []
    views_raw = mark_spec.get("views")
    if mark_spec.get("layout") == LAYOUT_PER_VIEW and views_raw:
        out = []
        for v in views_raw:
            if isinstance(v, dict):
                out.append(v)
        if out:
            return out

    slots = mark_spec.get("slots") or []
    if not slots:
        return []

    by_view: Dict[int, list] = {}
    for s in slots:
        if not isinstance(s, dict):
            continue
        vi = int(s.get("view_index", 0))
        slot = {k: v for k, v in s.items() if k != "view_index"}
        by_view.setdefault(vi, []).append(slot)

    views = []
    for vi in sorted(by_view):
        sl = by_view[vi]
        kinds = sorted({s.get("mark_kind") for s in sl if s.get("mark_kind")})
        views.append({"view_index": vi, "mark_kinds": kinds, "slots": sl})
    return views


def mark_spec_has_slots(mark_spec: Optional[dict]) -> bool:
    return any((v.get("slots") or []) for v in mark_spec_views(mark_spec))


def view_count(mark_spec: Optional[dict]) -> int:
    return len(mark_spec_views(mark_spec))


def slots_for_view(mark_spec: Optional[dict], view_index: int) -> List[dict]:
    for v in mark_spec_views(mark_spec):
        if int(v.get("view_index", -1)) == int(view_index):
            return list(v.get("slots") or [])
    return []


def all_slots_flat(mark_spec: Optional[dict]) -> List[dict]:
    out = []
    for v in mark_spec_views(mark_spec):
        out.extend(v.get("slots") or [])
    return out


def assemble_per_view_mark_spec(
    view_entries: List[dict],
    *,
    render_hints: Optional[dict] = None,
) -> dict:
    """Build canonical multiview mark_spec: one views[] element per QA image."""
    views = []
    for ent in view_entries:
        if not ent:
            continue
        slots = list(ent.get("slots") or [])
        kinds = ent.get("mark_kinds")
        if not kinds and slots:
            kinds = sorted({s.get("mark_kind") for s in slots if s.get("mark_kind")})
        if not slots and not ent.get("image_ref"):
            continue
        views.append({
            "view_index": int(ent["view_index"]),
            "image_ref": ent.get("image_ref"),
            "mark_kinds": list(kinds),
            "slots": slots,
        })
    if not views:
        return {}
    out: Dict[str, Any] = {
        "version": MARK_SPEC_VERSION,
        "layout": LAYOUT_PER_VIEW,
        "views": views,
    }
    if render_hints:
        out["render_hints"] = dict(render_hints)
    return out


def align_mark_spec_to_image_refs(
    mark_spec: Optional[dict],
    image_refs: List[str],
) -> Optional[dict]:
    """
    Dict semantics: one views[] entry per QA image path (slots may be empty).

    Merges existing per-view slots; inserts empty entries for images without marks.
    """
    if not image_refs:
        return mark_spec
    by_index = {int(v.get("view_index", -1)): v for v in mark_spec_views(mark_spec)}
    entries = []
    for i, ref in enumerate(image_refs):
        prev = by_index.get(i, {})
        entries.append({
            "view_index": i,
            "image_ref": ref,
            "mark_kinds": prev.get("mark_kinds", []),
            "slots": list(prev.get("slots") or []),
        })
    assembled = assemble_per_view_mark_spec(entries)
    return assembled if assembled else mark_spec


def wrap_single_view_mark_spec(
    spec: dict,
    *,
    image_ref: Optional[str] = None,
) -> dict:
    """Convert a single-image plan_mark() dict into per_view layout (views[0])."""
    if spec.get("layout") == LAYOUT_PER_VIEW and spec.get("views"):
        return spec
    return assemble_per_view_mark_spec([{
        "view_index": 0,
        "image_ref": image_ref,
        "mark_kinds": spec.get("mark_kinds", []),
        "slots": spec.get("slots", []),
    }], render_hints=spec.get("render_hints"))


def merge_mark_specs(
    specs: List[dict],
    *,
    view_indices: Optional[List[int]] = None,
    image_refs: Optional[List[str]] = None,
) -> Optional[dict]:
    """Merge per-image plan outputs into one per_view mark_spec (multiview)."""
    entries = []
    for i, spec in enumerate(specs):
        if not spec or not spec.get("slots"):
            continue
        vi = int(view_indices[i]) if view_indices and i < len(view_indices) else i
        ref = image_refs[i] if image_refs and i < len(image_refs) else None
        entries.append({
            "view_index": vi,
            "image_ref": ref,
            "mark_kinds": spec.get("mark_kinds", []),
            "slots": spec.get("slots", []),
        })
    assembled = assemble_per_view_mark_spec(entries)
    return assembled if assembled else None


def encode_slot_key(view_index: int, slot_id: str) -> str:
    """Globally unique slot key for UI / viz (view_index + slot_id)."""
    return f"{int(view_index)}:{slot_id}"


def parse_slot_keys(keys: List[str]) -> List[tuple]:
    """
    Parse slot keys from viz / API.

    - ``"1:A"`` → view 1, slot A
    - ``"A"`` (legacy) → any view that has slot A (ambiguous on multiview)
    """
    out: List[tuple] = []
    for raw in keys:
        k = str(raw).strip()
        if not k:
            continue
        if ":" in k:
            vi_s, sid = k.split(":", 1)
            try:
                out.append((int(vi_s), sid))
            except ValueError:
                out.append((None, k))
        else:
            out.append((None, k))
    return out


def slot_ids_for_frame(
    mark_spec: Optional[dict],
    frame_index: int,
    *,
    n_frames: int = 1,
    user_slot_ids: Optional[List[str]] = None,
) -> List[str]:
    """Slot ids belonging to QA image frame_index (0..n_frames-1)."""
    ids = [
        str(s["slot_id"])
        for s in slots_for_view(mark_spec, frame_index)
        if isinstance(s, dict) and s.get("slot_id")
    ]
    if user_slot_ids is not None:
        wanted = {
            sid for vi, sid in parse_slot_keys(user_slot_ids)
            if vi is None or vi == int(frame_index)
        }
        return [sid for sid in ids if sid in wanted]
    return ids


def view_mark_spec_slice(mark_spec: Optional[dict], view_index: int) -> dict:
    """Single-image spec dict for render_mark (one view only)."""
    slots = slots_for_view(mark_spec, view_index)
    kinds = sorted({s.get("mark_kind") for s in slots if s.get("mark_kind")})
    return {"version": MARK_SPEC_VERSION, "mark_kinds": kinds, "slots": slots}


# ─── Planning ──────────────────────────────────────────────────────────


def _normalize_bbox_2d(bbox_2d: Any) -> Optional[List[float]]:
    if bbox_2d is None:
        return None
    if hasattr(bbox_2d, "tolist"):
        bbox_2d = bbox_2d.tolist()
    try:
        b = [float(v) for v in bbox_2d]
    except (TypeError, ValueError):
        return None
    return b if len(b) == 4 else None


def _bbox_2d_from_mask(mask_pil: Any) -> Optional[List[float]]:
    if mask_pil is None:
        return None
    arr = np.array(mask_pil)
    if arr.ndim < 2:
        return None
    ys, xs = np.where(arr > 0)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def _slot_id(index: int, labels: Optional[List[str]]) -> str:
    if labels and index < len(labels):
        return labels[index]
    return chr(ord("A") + index) if index < 26 else str(index)


def _marked_slot_label(tag: str, color_name: str, mark_kind: str) -> str:
    return f"{tag}-({color_name} {mark_kind})"


def _obj_idx_from_node(node) -> int:
    from .scene_graph import SceneNode
    if isinstance(node, SceneNode):
        try:
            return int(node.node_id)
        except (TypeError, ValueError):
            pass
    return 0


def plan_object_marks(
    marker: VisualMarker,
    objs: list,
    mark_type: Optional[str] = None,
    view_idx: int = 0,
    labels: Optional[List[str]] = None,
) -> Tuple[dict, list]:
    if mark_type is None:
        mark_type = marker.choose_mark_type()

    slots = []
    marked_info = []
    mark_kinds = set()

    for i, obj in enumerate(objs):
        tag, passthrough, bbox_2d, mask_pil = marker._extract(obj, view_idx)
        slot_id = _slot_id(i, labels)

        bbox = _normalize_bbox_2d(bbox_2d) or _bbox_2d_from_mask(mask_pil)
        geometry: Dict[str, Any] = {}
        actual_kind = mark_type

        if mark_type == "point":
            if mask_pil is not None:
                mask = np.array(mask_pil)
                ys, xs = np.where(mask > 0)
                if len(xs) == 0:
                    continue
                cx, cy = int(np.mean(xs)), int(np.mean(ys))
                nearest = np.argmin((xs - cx) ** 2 + (ys - cy) ** 2)
                geometry = {"uv": [int(xs[nearest]), int(ys[nearest])]}
            elif bbox is not None:
                geometry["box_2d"] = bbox
                actual_kind = "box"
            else:
                continue
        elif mark_type == "box":
            if bbox is not None:
                geometry["box_2d"] = bbox
            elif mask_pil is not None:
                ys, xs = np.where(np.array(mask_pil) > 0)
                if len(xs) == 0:
                    continue
                cx, cy = int(np.mean(xs)), int(np.mean(ys))
                nearest = np.argmin((xs - cx) ** 2 + (ys - cy) ** 2)
                geometry = {"uv": [int(xs[nearest]), int(ys[nearest])]}
                actual_kind = "point"
            else:
                continue
        else:
            continue

        if not geometry:
            continue

        color_name, _color = marker.pop_color()
        slots.append({
            "slot_id": slot_id,
            "obj_idx": _obj_idx_from_node(passthrough),
            "tag": tag,
            "mark_kind": actual_kind,
            "color_name": color_name,
            "label_alias": slot_id if labels else None,
            "geometry": geometry,
        })
        mark_kinds.add(actual_kind)
        marked_info.append((_marked_slot_label(tag, color_name, actual_kind), passthrough))

    render_hints = {"alpha": 0.3}
    if marker.config.type_weights:
        render_hints["type_weights"] = dict(marker.config.type_weights)
    if marker.config.shuffle_colors:
        render_hints["shuffle_colors"] = True

    spec = {
        "version": MARK_SPEC_VERSION,
        "mark_kinds": sorted(mark_kinds),
        "slots": slots,
        "render_hints": render_hints,
    }
    return spec, marked_info


def plan_point_marks(
    marker: VisualMarker,
    points: List[List[int]],
    labels: Optional[List[str]] = None,
) -> Tuple[dict, list]:
    slots = []
    for i, uv in enumerate(points):
        color_name, _color = marker.pop_color()
        slots.append({
            "slot_id": _slot_id(i, labels),
            "obj_idx": -1,
            "tag": f"point_{i}",
            "mark_kind": "point",
            "color_name": color_name,
            "label_alias": _slot_id(i, labels) if labels else None,
            "geometry": {"uv": [int(uv[0]), int(uv[1])]},
        })
    spec = {
        "version": MARK_SPEC_VERSION,
        "mark_kinds": ["point"],
        "slots": slots,
    }
    return spec, slots


def _mask_to_bool_array(m: Any) -> Optional[np.ndarray]:
    if m is None:
        return None
    if isinstance(m, str):
        if not m.strip():
            return None
        if os.path.isfile(m):
            m = np.array(Image.open(m).convert("L"))
        else:
            return None
    elif isinstance(m, Image.Image):
        m = np.array(m.convert("L"))
    arr = np.asarray(m)
    if arr.ndim < 2:
        return None
    if arr.dtype == bool:
        return arr
    return arr > 0


def _resolve_mask_array(mask_ref: dict, preprocess_row: Optional[dict], slot: dict):
    if not isinstance(mask_ref, dict):
        return None
    source = mask_ref.get("source", "preprocess")
    if source == "path":
        path = mask_ref.get("path")
        if path:
            return _mask_to_bool_array(str(path))
        return None
    if source == "tar":
        return None
    obj_idx = mask_ref.get("obj_idx", slot.get("obj_idx"))
    if obj_idx is None or preprocess_row is None:
        return None
    try:
        obj_idx = int(obj_idx)
    except (TypeError, ValueError):
        return None
    masks = preprocess_row.get("masks")
    if masks is None:
        return None
    if hasattr(masks, "iloc"):
        masks = masks.tolist() if hasattr(masks, "tolist") else list(masks)
    if obj_idx >= len(masks):
        return None
    return _mask_to_bool_array(masks[obj_idx])


def render_mark(
    image: Image.Image,
    mark_spec: dict,
    labels: Optional[List[str]] = None,
    preprocess_row: Optional[dict] = None,
    *,
    view_index: int = 0,
) -> dict:
    """Render marks for a single QA image (one views[] entry)."""
    slice_spec = (
        mark_spec
        if mark_spec.get("slots") is not None and not mark_spec.get("views")
        else view_mark_spec_slice(mark_spec, view_index)
    )
    if not slice_spec or not slice_spec.get("slots"):
        return {"bytes": convert_pil_to_bytes(image)}

    by_kind: Dict[str, list] = {}
    for slot in slice_spec["slots"]:
        by_kind.setdefault(slot["mark_kind"], []).append(slot)

    overlay = np.array(image.convert("RGB"))

    for mark_kind, slots in by_kind.items():
        colors = [[s["color_name"], COLOR_MAP[s["color_name"]]] for s in slots]
        slot_labels = labels
        if slot_labels is None and any(s.get("label_alias") for s in slots):
            slot_labels = [s.get("label_alias") or s["slot_id"] for s in slots]

        if mark_kind == "mask":
            geometries = []
            for s in slots:
                m = _resolve_mask_array(s["geometry"].get("mask_ref", {}), preprocess_row, s)
                if m is None:
                    m = np.zeros(overlay.shape[:2], dtype=bool)
                geometries.append(m.astype(np.uint8))
            overlay = draw_masks_on_image(Image.fromarray(overlay), geometries, colors, labels=slot_labels)
        elif mark_kind == "point":
            geometries = [s["geometry"]["uv"] for s in slots]
            overlay = draw_points_on_image(Image.fromarray(overlay), geometries, colors, labels=slot_labels)
        else:
            geometries = [np.array(s["geometry"]["box_2d"]) for s in slots]
            overlay = draw_boxes_on_image(Image.fromarray(overlay), geometries, colors, labels=slot_labels)

    return {"bytes": convert_pil_to_bytes(Image.fromarray(overlay.astype(np.uint8)))}


def mark_spec_hash(mark_spec: dict) -> str:
    views = mark_spec_views(mark_spec)
    canonical = {
        "version": mark_spec.get("version", MARK_SPEC_VERSION),
        "layout": LAYOUT_PER_VIEW,
        "views": views,
    }
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    import hashlib
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
