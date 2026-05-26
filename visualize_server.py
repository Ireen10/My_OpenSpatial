"""
Visualization server for OpenSpatial annotation outputs.

Usage:
    python visualize_server.py --port 8888 --data_dir output/debug

Then open http://<host>:8888 in browser. The page header shows the data root
basename (e.g. base_pipeline_demo_singleview_all_frame_rot_m8l2).
"""

import argparse
import ast
import socket
from typing import List

import base64
import copy
import io
import json
import os
import glob
import re

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, render_template_string, request, jsonify

from task.annotation.core.mark_spec import (
    encode_slot_key,
    mark_spec_views,
    render_mark,
    slot_ids_for_frame,
    view_mark_spec_slice,
)
from utils.box_utils import (
    compute_box_3d_corners_from_params,
    convert_box_3d_world_to_camera,
)

app = Flask(__name__)
DATA_DIR = "output/debug"


def data_root_name():
    """Basename of the browsed --data_dir root (e.g. base_pipeline_demo_*_m8l2)."""
    return os.path.basename(os.path.normpath(DATA_DIR or "."))

# Fields / keys whose byte payloads are omitted in the raw-row viewer.
_RAW_OMIT_KEYS = frozenset({"QA_images", "qa_images"})
_BYTES_PLACEHOLDER = "<omitted image bytes>"

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def discover_parquets(data_dir):
    """Scan data_dir for all data.parquet files, return list of (display_name, path)."""
    results = []
    for pq_path in sorted(glob.glob(os.path.join(data_dir, "**/data.parquet"), recursive=True)):
        rel = os.path.relpath(pq_path, data_dir)
        parts = rel.split(os.sep)
        pipeline_name = parts[0] if parts else ""
        if "merged_samples" in parts or "aggregate_stage" in parts:
            task_name = "merged_samples"
            label = "[Merged] all tasks"
            is_multiview = False
            is_3d_grounding = False
        else:
            task_name = parts[-2] if len(parts) >= 2 else rel
            is_multiview = "multiview" in task_name
            is_3d_grounding = "3d_grounding" in task_name.lower()
            label = f"{'[Multi] ' if is_multiview else '[Single] '}{task_name}"
        if is_3d_grounding:
            label += " (3D boxes)"
        results.append({
            "label": label,
            "path": pq_path,
            "task": task_name,
            "multiview": is_multiview,
            "grounding_3d": is_3d_grounding,
        })
    return results


def image_from_bytes(data):
    """Convert bytes/dict to PIL Image."""
    if isinstance(data, dict) and "bytes" in data:
        data = data["bytes"]
    if isinstance(data, bytes):
        return Image.open(io.BytesIO(data))
    return None


def pil_to_base64(img, max_w=800):
    """Convert PIL image to base64 data URI, resize if too large."""
    if img is None:
        return ""
    w, h = img.size
    if w > max_w:
        ratio = max_w / w
        img = img.resize((max_w, int(h * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def load_original_image(image_field):
    """Load original image from path string, bytes, or dict."""
    if isinstance(image_field, str) and os.path.exists(image_field):
        return Image.open(image_field)
    if isinstance(image_field, (bytes, dict)):
        return image_from_bytes(image_field)
    if isinstance(image_field, np.ndarray):
        imgs = []
        for item in image_field:
            if isinstance(item, str) and os.path.exists(item):
                imgs.append(Image.open(item))
            elif isinstance(item, (bytes, dict)):
                img = image_from_bytes(item)
                if img:
                    imgs.append(img)
        return imgs if imgs else None
    if isinstance(image_field, list):
        imgs = []
        for item in image_field:
            if isinstance(item, str) and os.path.exists(item):
                imgs.append(Image.open(item))
            elif isinstance(item, (bytes, dict)):
                img = image_from_bytes(item)
                if img:
                    imgs.append(img)
        return imgs if imgs else None
    return None


def _normalize_meta(meta):
    if meta is None or (isinstance(meta, float) and np.isnan(meta)):
        return None
    if isinstance(meta, list) and meta:
        meta = meta[0]
    if not isinstance(meta, dict):
        return None
    meta = dict(meta)
    turns = meta.get("turns")
    if isinstance(turns, np.ndarray):
        meta["turns"] = turns.tolist()
    elif hasattr(turns, "tolist") and not isinstance(turns, list):
        meta["turns"] = turns.tolist()
    ms = meta.get("mark_spec")
    if isinstance(ms, dict):
        ms = dict(ms)
        if isinstance(ms.get("mark_kinds"), np.ndarray):
            ms["mark_kinds"] = ms["mark_kinds"].tolist()
        if isinstance(ms.get("slots"), np.ndarray):
            ms["slots"] = ms["slots"].tolist()
        views = ms.get("views")
        if isinstance(views, np.ndarray):
            ms["views"] = views.tolist()
        elif views is not None and not isinstance(views, list):
            views = _to_list(views)
            if views:
                ms["views"] = views
        meta["mark_spec"] = ms
    return meta


def _to_list(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    if isinstance(val, np.ndarray):
        return val.tolist()
    if hasattr(val, "tolist") and not isinstance(val, (list, str, dict)):
        return val.tolist()
    if isinstance(val, list):
        return val
    return [val]


def _numpy_to_python(obj):
    """Recursively convert numpy scalars/arrays for JSON serialization."""
    if obj is None or (isinstance(obj, float) and np.isnan(obj)):
        return None
    if isinstance(obj, np.ndarray):
        return _numpy_to_python(obj.tolist())
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _numpy_to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_numpy_to_python(v) for v in obj]
    if isinstance(obj, bytes):
        return _BYTES_PLACEHOLDER
    return obj


def _strip_image_payload(val, key=None):
    """Remove bulky image bytes; keep paths and text intact."""
    if key in _RAW_OMIT_KEYS:
        return "<omitted>"
    if isinstance(val, bytes):
        return _BYTES_PLACEHOLDER
    if isinstance(val, dict):
        if "bytes" in val and isinstance(val["bytes"], (bytes, bytearray)):
            out = {k: v for k, v in val.items() if k != "bytes"}
            out["bytes"] = _BYTES_PLACEHOLDER
            return _strip_image_payload(out)
        return {k: _strip_image_payload(v, k) for k, v in val.items()}
    if isinstance(val, list):
        return [_strip_image_payload(v) for v in val]
    if isinstance(val, np.ndarray):
        return _strip_image_payload(val.tolist())
    if isinstance(val, str) and len(val) > 2000 and re.match(r"^[A-Za-z0-9+/=\s]+$", val[:200]):
        return _BYTES_PLACEHOLDER
    return val


def serialize_row_raw(row_series):
    """Full parquet row as JSON-safe dict (no image byte blobs)."""
    d = row_series.to_dict()
    cleaned = {}
    for k, v in d.items():
        py = _numpy_to_python(v)
        cleaned[k] = _strip_image_payload(py, k)
    meta = _normalize_meta(d.get("metadata"))
    n_qa = _image_placeholder_count({"turns": (meta or {}).get("turns") or [], "meta": meta}, row_series)
    img = cleaned.get("image")
    if isinstance(img, list) and n_qa > 0 and len(img) != n_qa:
        refs = []
        if meta:
            va = meta.get("visual_anchor") or {}
            refs = list(va.get("raw_image_refs") or [])
            if not refs and va.get("raw_image_ref"):
                refs = [va["raw_image_ref"]]
        cleaned["_qa_summary"] = {
            "image_placeholder_count": n_qa,
            "preprocess_image_count": len(img),
            "note": (
                "image[] is the full preprocess scene; this QA row only uses "
                f"{n_qa} frame(s) in QA_images / messages <image> tags."
            ),
            "qa_image_paths": refs,
        }
    return cleaned


def _row_as_preprocess_dict(row_series):
    """Dict suitable for render_mark mask_ref resolution."""
    d = row_series.to_dict()
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist() if v.ndim > 0 else v.item()
        else:
            out[k] = v
    masks = out.get("masks")
    if masks is not None and not isinstance(masks, list):
        if hasattr(masks, "tolist"):
            out["masks"] = masks.tolist()
    return out


# 3D grounding visualization (camera-frame boxes, zxy euler)
_BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)
_BOX_COLORS = (
    (255, 64, 64), (64, 200, 64), (64, 128, 255),
    (255, 180, 40), (200, 64, 255), (40, 220, 220),
)


def _is_3d_grounding_task(task_name, parsed):
    if task_name and "3d_grounding" in task_name.lower():
        return True
    tags = parsed.get("tags") or []
    if any("3D Grounding" in str(t) for t in tags):
        return True
    for turn in parsed.get("turns") or []:
        prefix = turn.get("question_prefix") or ""
        q = turn.get("question") or ""
        a = turn.get("answer") or ""
        if "Camera intrinsic parameters" in prefix or "Camera intrinsic parameters" in q:
            if "bbox_3d" in a:
                return True
        if "bbox_3d" in a:
            return True
    return False


def _parse_float_token(s):
    return float(s.strip().rstrip("."))


def _parse_camera_from_text(text):
    """Parse camera params from grounding_3d question prefix (f_x/f_y or legacy hfov)."""
    if not text:
        return None
    num = r"[0-9]+(?:\.[0-9]+)?"
    m = re.search(
        rf"f_x\s*=\s*({num}).*?f_y\s*=\s*({num}).*?"
        rf"c_x\s*=\s*({num}).*?c_y\s*=\s*({num}).*?"
        rf"width\s*(\d+).*?height\s*(\d+)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        try:
            fx = _parse_float_token(m.group(1))
            fy = _parse_float_token(m.group(2))
            cx = _parse_float_token(m.group(3))
            cy = _parse_float_token(m.group(4))
            w, h = int(m.group(5)), int(m.group(6))
        except ValueError:
            return None
        k = np.eye(4, dtype=np.float64)
        k[0, 0], k[1, 1] = fx, fy
        k[0, 2], k[1, 2] = cx, cy
        return {"intrinsic": k, "width": w, "height": h}

    m = re.search(
        rf"hfov\s*=\s*({num}).*?vfov\s*=\s*({num}).*?"
        rf"(?:Image\s+)?width\s*=\s*(\d+).*?(?:Image\s+)?height\s*=\s*(\d+)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return None
    try:
        hfov = _parse_float_token(m.group(1))
        vfov = _parse_float_token(m.group(2))
        w, h = int(m.group(3)), int(m.group(4))
    except ValueError:
        return None
    return {"hfov": hfov, "vfov": vfov, "width": w, "height": h}


def _intrinsic_from_fov(hfov_deg, vfov_deg, width, height):
    h_rad, v_rad = np.radians(hfov_deg), np.radians(vfov_deg)
    fx = width / (2.0 * np.tan(h_rad / 2.0))
    fy = height / (2.0 * np.tan(v_rad / 2.0))
    k = np.eye(4, dtype=np.float64)
    k[0, 0], k[1, 1] = fx, fy
    k[0, 2], k[1, 2] = width / 2.0, height / 2.0
    return k


def _scale_intrinsic_to_image(intrinsic, ref_size, img_size):
    ref_w, ref_h = ref_size
    img_w, img_h = img_size
    if ref_w <= 0 or ref_h <= 0:
        return intrinsic
    k = intrinsic.copy()
    sx, sy = img_w / ref_w, img_h / ref_h
    k[0, 0] *= sx
    k[0, 2] *= sx
    k[1, 1] *= sy
    k[1, 2] *= sy
    return k


def _load_pose_matrix(pose_field):
    if pose_field is None or (isinstance(pose_field, float) and np.isnan(pose_field)):
        return None
    if isinstance(pose_field, (list, np.ndarray)):
        arr = np.asarray(pose_field, dtype=np.float64)
        if arr.shape == (4, 4):
            return arr
        return None
    if isinstance(pose_field, str) and os.path.isfile(pose_field):
        return np.loadtxt(pose_field, dtype=np.float64)
    return None


def _intrinsic_from_row(row):
    raw = row.get("intrinsic", None)
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    k = np.asarray(raw, dtype=np.float64)
    if k.shape == (3, 3):
        k4 = np.eye(4, dtype=np.float64)
        k4[:3, :3] = k
        return k4
    if k.shape == (4, 4):
        return k
    return None


def _grounding_display_turns(parsed):
    """Prefer 3D-grounding turns when a merged sample has many tasks."""
    turns = parsed.get("turns") or []
    grounding = [
        t for t in turns
        if isinstance(t, dict)
        and (
            "3d_grounding" in (t.get("task_name") or "").lower()
            or str(t.get("sub_task") or "").startswith("grounding")
        )
    ]
    return grounding if grounding else turns


def _bbox_entries_from_prompt_struct(parsed):
    """Upstream / merged rows may store camera-frame boxes only in prompt_struct."""
    entries = []
    meta = parsed.get("meta") or {}
    for turn in meta.get("turns") or []:
        if not isinstance(turn, dict):
            continue
        ps = turn.get("prompt_struct")
        if not isinstance(ps, dict):
            continue
        for val in (ps.get("answer_bindings") or {}).values():
            if val is None:
                continue
            entries.extend(_parse_bbox_entries_from_text(str(val)))
        if entries:
            return entries
    return []


def _parse_bbox_entries_from_text(text):
    if not text:
        return []
    m = re.search(r"(\[\s*\{.*\}\s*\])", text, flags=re.DOTALL)
    if not m:
        return []
    blob = m.group(1)
    for parser in (ast.literal_eval, lambda s: json.loads(s.replace("'", '"'))):
        try:
            data = parser(blob)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
        if not isinstance(data, list):
            continue
        entries = []
        for item in data:
            if isinstance(item, dict) and "bbox_3d" in item:
                box = item["bbox_3d"]
                if isinstance(box, (list, tuple)) and len(box) >= 9:
                    entries.append({
                        "bbox_3d": [float(v) for v in box[:9]],
                        "label": str(item.get("label", "")),
                    })
        if entries:
            return entries
    return []


def _bbox_entries_from_row(row):
    boxes_world = row.get("bboxes_3d_world_coords", None)
    if boxes_world is None or (isinstance(boxes_world, float) and np.isnan(boxes_world)):
        return []
    pose = _load_pose_matrix(row.get("pose", None))
    if pose is None:
        return []

    tags = row.get("obj_tags", None)
    if tags is None:
        tags = []
    if isinstance(tags, np.ndarray):
        tags = tags.tolist()
    if isinstance(boxes_world, np.ndarray):
        boxes_world = boxes_world.tolist()

    entries = []
    for idx, box in enumerate(boxes_world):
        if box is None or len(box) < 9:
            continue
        cam_box = convert_box_3d_world_to_camera(box, pose)
        if cam_box is None:
            continue
        label = ""
        if idx < len(tags):
            tag = tags[idx]
            if isinstance(tag, (list, tuple)) and tag:
                label = str(tag[0])
            else:
                label = str(tag)
        entries.append({"bbox_3d": cam_box, "label": label})
    return entries


def _project_cam_to_2d(points_cam, intrinsic):
    pts = np.asarray(points_cam, dtype=np.float64)
    uv = np.full((pts.shape[0], 2), np.nan)
    valid = pts[:, 2] > 1e-3
    z = pts[valid, 2]
    uv[valid, 0] = intrinsic[0, 0] * pts[valid, 0] / z + intrinsic[0, 2]
    uv[valid, 1] = intrinsic[1, 1] * pts[valid, 1] / z + intrinsic[1, 2]
    return uv, valid


def _turn_camera_text(turn):
    prefix = (turn.get("question_prefix") or "").strip()
    question = (turn.get("question") or "").strip()
    if prefix:
        return prefix + "\n" + question
    return question


def _resolve_grounding_context(row, parsed):
    entries = _bbox_entries_from_row(row)
    intrinsic = _intrinsic_from_row(row)
    ref_size = None

    if not entries:
        for turn in _grounding_display_turns(parsed):
            entries = _parse_bbox_entries_from_text(turn.get("answer") or "")
            if entries:
                break

    if not entries:
        entries = _bbox_entries_from_prompt_struct(parsed)

    if not entries:
        return None, None

    if intrinsic is None:
        cam = None
        for turn in _grounding_display_turns(parsed):
            cam = _parse_camera_from_text(turn.get("question_prefix") or "")
            if cam is None:
                cam = _parse_camera_from_text(_turn_camera_text(turn))
            if cam:
                break
        if cam is None:
            return entries, None
        ref_size = (cam["width"], cam["height"])
        if "intrinsic" in cam:
            intrinsic = cam["intrinsic"]
        else:
            intrinsic = _intrinsic_from_fov(cam["hfov"], cam["vfov"], cam["width"], cam["height"])

    return entries, (intrinsic, ref_size)


def draw_3d_boxes_on_image(img, entries, intrinsic, ref_size=None):
    if not entries or intrinsic is None:
        return img
    out = img.convert("RGB")
    w, h = out.size
    k = intrinsic
    if ref_size is not None:
        k = _scale_intrinsic_to_image(k, ref_size, (w, h))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None

    for i, entry in enumerate(entries):
        color = _BOX_COLORS[i % len(_BOX_COLORS)]
        corners = compute_box_3d_corners_from_params(entry["bbox_3d"])
        uv, valid = _project_cam_to_2d(corners, k)
        for i0, i1 in _BOX_EDGES:
            if not (valid[i0] and valid[i1]):
                continue
            p0 = tuple(uv[i0])
            p1 = tuple(uv[i1])
            if any(np.isnan(c) for c in p0 + p1):
                continue
            draw.line([p0, p1], fill=color, width=2)
        vis = np.where(valid)[0]
        if len(vis) > 0 and font is not None:
            cx = int(np.nanmean(uv[vis, 0]))
            cy = int(np.nanmean(uv[vis, 1]))
            label = entry.get("label") or f"box{i}"
            draw.text((cx + 4, cy + 4), label, fill=color, font=font)
    return out


def _strip_image_prefix(text: str) -> str:
    if not text:
        return ""
    t = str(text).strip()
    # Human turns may start with multiple "<image>" placeholders.
    while t.startswith("<image>"):
        t = t[len("<image>") :].strip()
    return t


def _parse_messages_turns(messages):
    if (
        messages
        and isinstance(messages[0], list)
        and messages[0]
        and isinstance(messages[0][0], dict)
    ):
        turns: list = []
        for conv in messages:
            if isinstance(conv, list):
                turns.extend(_parse_messages_turns(conv))
        return turns

    turns = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if isinstance(msg, dict) and msg.get("from") == "human":
            q = msg.get("value", "")
            a = ""
            if i + 1 < len(messages):
                next_msg = messages[i + 1]
                if isinstance(next_msg, dict) and next_msg.get("from") == "gpt":
                    a = next_msg.get("value", "")
                    i += 1
            # Display should use the fully rendered message text verbatim.
            # Downstream training-data conversion should rely on metadata.prompt_struct,
            # not on UI-normalized strings.
            turns.append({"question": str(q), "answer": str(a)})
        i += 1
    return turns


def _metadata_turns_to_display(meta_turns, msg_turns=None):
    out = []
    for i, tr in enumerate(meta_turns):
        if not isinstance(tr, dict):
            continue
        prefix = (tr.get("question_prefix") or "").strip()
        q_text = (tr.get("question_text") or "").strip()
        answer = tr.get("answer_text") or ""
        if msg_turns and i < len(msg_turns):
            msg_q = (msg_turns[i].get("question") or "").strip()
            msg_a = (msg_turns[i].get("answer") or "").strip()
            # Display: prefer fully rendered messages verbatim when present.
            if msg_q:
                q_text = msg_q
            if msg_a:
                answer = msg_a
        question = q_text
        if prefix:
            question = f"{prefix}\n\n{q_text}" if q_text else prefix
        out.append({
            "turn_id": tr.get("turn_id"),
            "task_name": tr.get("task_name", ""),
            "question": question,
            "question_prefix": prefix,
            "question_text": q_text,
            "answer": answer,
            "question_type": tr.get("question_type", ""),
            "sub_task": tr.get("sub_task", ""),
            "image_placeholder_count": int(tr.get("image_placeholder_count") or 0),
        })
    return out


def _find_active_turn_index(meta_turns, msg_turns):
    if not meta_turns or not msg_turns:
        return None
    msg_a = (msg_turns[-1].get("answer") or "").strip()
    for i, tr in enumerate(meta_turns):
        if not isinstance(tr, dict):
            continue
        if (tr.get("answer_text") or "").strip() == msg_a:
            return i
    return len(meta_turns) - 1 if len(meta_turns) == len(msg_turns) else 0


def parse_row(row):
    """Parse a parquet row: prefer metadata.turns, fallback to messages."""
    meta = _normalize_meta(row.get("metadata"))
    messages = _to_list(row.get("messages", []))
    msg_turns = _parse_messages_turns(messages)

    turns = []
    active_turn_index = None
    if meta and meta.get("turns"):
        meta_turns = meta["turns"]
        turns = _metadata_turns_to_display(meta_turns, msg_turns)
        active_turn_index = _find_active_turn_index(meta_turns, msg_turns)
    if not turns:
        turns = msg_turns

    tags = row.get("question_tags", [])
    if isinstance(tags, np.ndarray):
        tags = tags.tolist()
    qtype = row.get("question_types", "")

    type_label = ""
    if meta and meta.get("turns") and isinstance(meta["turns"], list) and meta["turns"]:
        type_label = (meta["turns"][0].get("type_label") or "").strip()

    mark_slots = []
    if meta and meta.get("mark_spec"):
        for v in mark_spec_views(meta["mark_spec"]):
            vi = v.get("view_index", 0)
            for s in v.get("slots") or []:
                if not isinstance(s, dict):
                    continue
                sid = s.get("slot_id", "")
                mark_slots.append({
                    "slot_id": sid,
                    "slot_key": encode_slot_key(vi, sid),
                    "tag": s.get("tag", ""),
                    "mark_kind": s.get("mark_kind", ""),
                    "color_name": s.get("color_name", ""),
                    "label_alias": s.get("label_alias"),
                    "view_index": vi,
                })

    return {
        "turns": turns,
        "active_turn_index": active_turn_index,
        "tags": tags,
        "question_type": qtype,
        "meta": meta,
        "mark_slots": mark_slots,
        "type_label": type_label,
    }


def _filter_mark_spec(mark_spec, slot_ids, view_index: int = 0):
    if not mark_spec or not slot_ids:
        return None
    wanted = set(slot_ids)
    slots = [
        s for s in view_mark_spec_slice(mark_spec, view_index).get("slots", [])
        if s.get("slot_id") in wanted
    ]
    if not slots:
        return None
    kinds = sorted({s.get("mark_kind") for s in slots if s.get("mark_kind")})
    return {"version": 2, "mark_kinds": kinds, "slots": slots}


def _apply_marks_to_image(img, mark_spec, slot_ids, preprocess_row, view_index: int = 0):
    if not slot_ids or not mark_spec:
        return img
    filtered = _filter_mark_spec(mark_spec, slot_ids, view_index=view_index)
    if not filtered:
        return img
    rendered = render_mark(
        img, filtered, preprocess_row=preprocess_row, view_index=view_index,
    )
    out = image_from_bytes(rendered)
    return out if out is not None else img


def _image_placeholder_count(parsed, row_series) -> int:
    """How many QA images this row uses (<image> placeholders)."""
    n = 0
    for turn in parsed.get("turns") or []:
        if isinstance(turn, dict):
            n = max(n, int(turn.get("image_placeholder_count") or 0))
    meta = parsed.get("meta") or {}
    for turn in meta.get("turns") or []:
        if isinstance(turn, dict):
            n = max(n, int(turn.get("image_placeholder_count") or 0))
    if n <= 0:
        for m in _to_list(row_series.get("messages")):
            if isinstance(m, dict) and m.get("from") == "human":
                n = str(m.get("value", "")).count("<image>")
                break
    if n <= 0:
        qa = row_series.get("QA_images")
        if qa is not None and hasattr(qa, "__len__") and not isinstance(qa, (str, bytes)):
            try:
                n = len(qa)
            except TypeError:
                pass
    return n


def build_display_images(
    row_series,
    parsed,
    task_name,
    slot_ids=None,
    overlay_3d=False,
    *,
    overlay_marks: bool = False,
    marks_mode: str = "off",
):
    """Original image(s) with optional mark overlays and 3D box overlay.

    marks_mode:
      - ``off``: raw QA frames only (list page default)
      - ``selected``: only ``slot_ids`` (keys ``view:slot``); empty list → no marks
      - ``all``: every slot on each frame
    """
    if marks_mode not in ("off", "selected", "all"):
        marks_mode = "off"
    apply_marks = marks_mode in ("selected", "all")
    n_place = _image_placeholder_count(parsed, row_series)

    def _images_from_qa_field(qa_raw):
        if qa_raw is None:
            return None
        if isinstance(qa_raw, dict):
            img = image_from_bytes(qa_raw)
            return [img] if img else None
        imgs = []
        for item in _to_list(qa_raw):
            img = image_from_bytes(item)
            if img:
                imgs.append(img)
        return imgs if imgs else None

    orig = None
    qa_raw = row_series.get("QA_images")
    qa_imgs = _images_from_qa_field(qa_raw)
    if qa_imgs and n_place > 0 and len(qa_imgs) == n_place:
        orig = qa_imgs
    if orig is None:
        orig = load_original_image(row_series.get("image"))
    if orig is None:
        orig = qa_imgs

    if orig is None:
        return [], False

    if not isinstance(orig, list):
        images = [orig]
    else:
        images = orig

    meta = parsed.get("meta") or {}
    mark_spec = meta.get("mark_spec")
    preprocess = _row_as_preprocess_dict(row_series)
    n_frames = len(images)
    out_images = []
    for fi, img in enumerate(images):
        frame = img.copy()
        if apply_marks and mark_spec:
            if marks_mode == "all":
                user_slots = None
            else:
                user_slots = list(slot_ids or [])
            use_ids = slot_ids_for_frame(
                mark_spec, fi, n_frames=n_frames, user_slot_ids=user_slots,
            )
            if use_ids:
                frame = _apply_marks_to_image(
                    frame, mark_spec, use_ids, preprocess, view_index=fi,
                )
        out_images.append(frame)

    has_3d = False
    if overlay_3d and _is_3d_grounding_task(task_name, parsed):
        try:
            entries, ctx = _resolve_grounding_context(row_series.to_dict(), parsed)
            if entries and ctx is not None:
                intrinsic, ref_size = ctx
                out_images[0] = draw_3d_boxes_on_image(
                    out_images[0], entries, intrinsic, ref_size=ref_size,
                )
                has_3d = True
        except Exception as exc:
            print(f"[3d_grounding] overlay failed: {exc}")

    return out_images, has_3d, apply_marks


def _turn_task_names(parsed: dict) -> set:
    names: set = set()
    for t in parsed.get("turns") or []:
        if not isinstance(t, dict):
            continue
        for key in ("task_name", "sub_task"):
            v = (t.get(key) or "").strip()
            if v:
                names.add(v)
    meta = parsed.get("meta") or {}
    prov = meta.get("provenance") or {}
    for st in prov.get("source_tasks") or []:
        v = str(st).strip()
        if v:
            names.add(v)
    for tag in parsed.get("tags") or []:
        v = str(tag).strip()
        if v:
            names.add(v)
    return names


def row_matches_display_filters(
    parsed: dict,
    *,
    filter_task: str = "",
    filter_turns: str = "",
) -> bool:
    turns = parsed.get("turns") or []
    n = len(turns)
    ft = (filter_turns or "").strip()
    if ft == "1" and n != 1:
        return False
    if ft == "2" and n != 2:
        return False
    if ft == "3" and n != 3:
        return False
    if ft == "2+" and n < 2:
        return False
    if ft == "3+" and n < 3:
        return False
    task = (filter_task or "").strip()
    if task and task not in _turn_task_names(parsed):
        return False
    return True


def collect_dataset_task_names(path: str, *, max_rows: int = 5000) -> List[str]:
    if not path or not os.path.exists(path):
        return []
    df = pd.read_parquet(path)
    names: set = set()
    for i in range(min(len(df), max_rows)):
        row = df.iloc[i]
        parsed = parse_row(row)
        parsed["meta"] = _normalize_meta(row.get("metadata"))
        names.update(_turn_task_names(parsed))
    return sorted(names)


def filtered_row_indices(
    path: str,
    *,
    filter_task: str = "",
    filter_turns: str = "",
) -> List[int]:
    if not path or not os.path.exists(path):
        return []
    if not (filter_task or "").strip() and not (filter_turns or "").strip():
        df = pd.read_parquet(path)
        return list(range(len(df)))
    df = pd.read_parquet(path)
    out: List[int] = []
    for i in range(len(df)):
        row = df.iloc[i]
        parsed = parse_row(row)
        parsed["meta"] = _normalize_meta(row.get("metadata"))
        if row_matches_display_filters(
            parsed, filter_task=filter_task, filter_turns=filter_turns,
        ):
            out.append(i)
    return out


# ──────────────────────────────────────────────────────────────────────
# HTML Template
# ──────────────────────────────────────────────────────────────────────

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OpenSpatial Visualizer</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; color: #333; }
  .header { background: #1a1a2e; color: white; padding: 16px 24px; display: flex; align-items: center; gap: 16px; position: sticky; top: 0; z-index: 100; flex-wrap: wrap; }
  .header-title { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
  .header h1 { font-size: 20px; font-weight: 600; }
  .data-root-name { font-size: 13px; font-weight: 500; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: rgba(255,255,255,0.85); word-break: break-all; }
  .header select { padding: 8px 12px; border-radius: 6px; border: none; font-size: 14px; background: #16213e; color: white; cursor: pointer; min-width: 280px; }
  .header select option { background: #16213e; }
  .header .info { margin-left: auto; font-size: 13px; opacity: 0.8; }
  .filters { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .filters label { font-size: 12px; opacity: 0.9; }
  .filters select { padding: 6px 10px; border-radius: 6px; border: none; font-size: 13px; background: #16213e; color: white; }
  .nav { display: flex; align-items: center; gap: 8px; margin-left: 16px; }
  .nav button { padding: 6px 14px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.3); background: transparent; color: white; cursor: pointer; font-size: 13px; transition: all 0.2s; }
  .nav button:hover { background: rgba(255,255,255,0.15); }
  .nav button:disabled { opacity: 0.3; cursor: default; }
  .nav span { color: rgba(255,255,255,0.7); font-size: 13px; min-width: 80px; text-align: center; }
  .container { max-width: 1200px; margin: 24px auto; padding: 0 24px; }
  .card { background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 20px; overflow: hidden; }
  .card-header { padding: 14px 20px; background: #f8f9fa; border-bottom: 1px solid #eee; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .tag { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }
  .tag-task { background: #e3f2fd; color: #1565c0; }
  .tag-type { background: #f3e5f5; color: #7b1fa2; }
  .tag-active { background: #fff8e1; color: #f57f17; }
  .card-body { padding: 20px; }
  .images-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }
  .images-row img { border-radius: 8px; border: 1px solid #eee; cursor: pointer; transition: transform 0.2s; max-height: 400px; object-fit: contain; }
  .images-row img:hover { transform: scale(1.02); }
  .mark-panel { margin-bottom: 14px; padding: 12px; background: #fafafa; border-radius: 8px; border: 1px solid #eee; }
  .mark-panel label { display: inline-flex; align-items: center; gap: 6px; margin-right: 14px; margin-bottom: 6px; font-size: 13px; cursor: pointer; }
  .mark-hint { font-size: 12px; color: #888; margin-bottom: 8px; }
  .qa-block { margin-top: 12px; }
  .qa-label { font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
  .qa-label.q { color: #1565c0; }
  .qa-label.a { color: #2e7d32; }
  .qa-text { padding: 12px 16px; border-radius: 8px; font-size: 14px; line-height: 1.6; white-space: pre-wrap; word-break: break-word; }
  .qa-text.q { background: #e3f2fd; }
  .qa-text.a { background: #e8f5e9; }
  .qa-text.prefix { background: #eceff1; font-size: 13px; color: #455a64; margin-bottom: 8px; }
  .turn-divider { border: none; border-top: 1px dashed #ddd; margin: 12px 0; }
  .turn-badge { display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; background: #fff3e0; color: #e65100; margin-left: 6px; }
  .turn-active { border-left: 3px solid #ff9800; padding-left: 12px; }
  .multi-turn-label { font-size: 12px; color: #888; margin-bottom: 4px; }
  .btn-raw { padding: 4px 12px; border-radius: 6px; border: 1px solid #ccc; background: white; font-size: 12px; cursor: pointer; margin-left: auto; }
  .btn-raw:hover { background: #f0f0f0; }
  .raw-panel { display: none; margin-top: 14px; max-height: 480px; overflow: auto; background: #263238; color: #eceff1; padding: 12px; border-radius: 8px; font-family: ui-monospace, monospace; font-size: 12px; line-height: 1.45; white-space: pre-wrap; word-break: break-word; }
  .raw-panel.open { display: block; }
  .lightbox { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.85); z-index: 200; justify-content: center; align-items: center; cursor: zoom-out; }
  .lightbox.active { display: flex; }
  .lightbox img { max-width: 95vw; max-height: 95vh; object-fit: contain; border-radius: 8px; }
  .empty-state { text-align: center; padding: 80px 20px; color: #999; }
  .empty-state h2 { font-size: 24px; margin-bottom: 8px; }
</style>
</head>
<body>

<div class="header">
  <div class="header-title">
    <h1>OpenSpatial Visualizer</h1>
    <span class="data-root-name" title="--data_dir root">{{ data_root_name }}</span>
  </div>
  <select id="taskSelect" onchange="loadTask()">
    <option value="">-- Select a task --</option>
    {% for t in tasks %}
    <option value="{{ t.path }}" data-grounding="{{ '1' if t.grounding_3d else '0' }}" {{ 'selected' if t.path == selected_path else '' }}>{{ t.label }}</option>
    {% endfor %}
  </select>
  <div class="filters">
    <label>Task <select id="filterTask" onchange="applyFilters()"><option value="">All tasks</option></select></label>
    <label>Turns <select id="filterTurns" onchange="applyFilters()">
      <option value="">Any</option><option value="1">1</option><option value="2">2</option>
      <option value="3">3</option><option value="2+">2+</option><option value="3+">3+</option>
    </select></label>
  </div>
  <div class="nav">
    <button id="prevBtn" onclick="navigate(-1)" disabled>&larr; Prev</button>
    <span id="pageInfo">-</span>
    <button id="nextBtn" onclick="navigate(1)" disabled>Next &rarr;</button>
  </div>
  <div class="info" id="totalInfo"></div>
</div>

<div class="container" id="content">
  <div class="empty-state">
    <h2>Select a task to visualize</h2>
    <p>Choose an annotation output from the dropdown above</p>
  </div>
</div>

<div class="lightbox" id="lightbox" onclick="closeLightbox()">
  <img id="lightboxImg" src="" />
</div>

<script>
const PAGE_SIZE = 10;
let currentPage = 0;
let totalRows = 0;
let filteredTotal = 0;
let currentPath = '';
let is3dTask = false;

function filterParams() {
  const p = new URLSearchParams();
  const ft = document.getElementById('filterTask')?.value || '';
  const fn = document.getElementById('filterTurns')?.value || '';
  if (ft) p.set('filter_task', ft);
  if (fn) p.set('filter_turns', fn);
  return p;
}

function loadFilterTasks(path) {
  const sel = document.getElementById('filterTask');
  if (!sel) return;
  const cur = sel.value;
  fetch(`/api/filter_options?path=${encodeURIComponent(path)}`)
    .then(r => r.json())
    .then(data => {
      const tasks = data.tasks || [];
      sel.innerHTML = '<option value="">All tasks</option>' +
        tasks.map(t => `<option value="${escapeHtml(t)}">${escapeHtml(t)}</option>`).join('');
      if (cur && tasks.includes(cur)) sel.value = cur;
    });
}

function applyFilters() {
  if (!currentPath) return;
  currentPage = 0;
  fetchPage(currentPath, 0);
}

function loadTask() {
  const sel = document.getElementById('taskSelect');
  currentPath = sel.value;
  is3dTask = sel.selectedOptions[0]?.dataset.grounding === '1';
  if (!currentPath) return;
  currentPage = 0;
  loadFilterTasks(currentPath);
  fetchPage(currentPath, 0);
}

function navigate(delta) {
  if (!currentPath) return;
  currentPage += delta;
  fetchPage(currentPath, currentPage);
}

function fetchPage(path, page) {
  const q = filterParams();
  q.set('path', path);
  q.set('page', String(page));
  q.set('page_size', String(PAGE_SIZE));
  fetch('/api/data?' + q.toString())
    .then(r => r.json())
    .then(data => {
      totalRows = data.total;
      filteredTotal = data.filtered_total ?? data.total;
      currentPage = data.page;
      is3dTask = data.is_3d_task;
      renderRows(data.rows);
      updateNav();
    });
}

function updateNav() {
  const totalPages = Math.ceil(filteredTotal / PAGE_SIZE);
  document.getElementById('pageInfo').textContent = totalPages > 0 ? `${currentPage + 1} / ${totalPages}` : '-';
  document.getElementById('prevBtn').disabled = currentPage <= 0;
  document.getElementById('nextBtn').disabled = currentPage >= totalPages - 1;
  const extra = filteredTotal !== totalRows ? ` (${filteredTotal} matched / ${totalRows})` : '';
  document.getElementById('totalInfo').textContent = `${totalRows} rows${extra}`;
}

function renderRows(rows) {
  const container = document.getElementById('content');
  if (rows.length === 0) {
    container.innerHTML = '<div class="empty-state"><h2>No data</h2><p>This task produced no output rows.</p></div>';
    return;
  }
  let html = '';
  rows.forEach((row, idx) => {
    const globalIdx = currentPage * PAGE_SIZE + idx;
    const tagsHtml = (row.tags || []).map(t => `<span class="tag tag-task">${escapeHtml(t)}</span>`).join(' ');
    const typeHtml = row.question_type ? `<span class="tag tag-type">${escapeHtml(row.question_type)}</span>` : '';
    const isMultiTurn = row.turns && row.turns.length > 1;
    const turnBadge = isMultiTurn ? `<span class="turn-badge">${row.turns.length} turns (sample)</span>` : '';

    const imgId = `img-${row.row_index}`;
    let imagesHtml = '';
    if (row.display_images && row.display_images.length > 0) {
      const imgTags = row.display_images.map((src, i) =>
        `<img id="${imgId}-${i}" src="${src}" onclick="openLightbox(this.src)" style="max-width:${row.display_images.length > 1 ? Math.floor(100/Math.min(row.display_images.length, 4)) - 2 : 100}%;" />`
      ).join('');
      imagesHtml = `<div class="images-row" id="${imgId}-row">${imgTags}</div>`;
    }

    let markHtml = '';
    if (row.mark_slots && row.mark_slots.length > 0) {
      const checks = row.mark_slots.map(s => {
        const key = s.slot_key || (String(s.view_index) + ':' + s.slot_id);
        const vf = (s.view_index !== undefined && s.view_index !== null && s.view_index !== '')
          ? `view ${s.view_index} · ` : '';
        const alias = s.label_alias ? ` (${escapeHtml(s.label_alias)})` : '';
        const label = `${vf}${escapeHtml(s.slot_id)}${alias}: ${escapeHtml(s.tag)} (${escapeHtml(s.color_name)} ${escapeHtml(s.mark_kind)})`;
        return `<label><input type="checkbox" class="mark-cb" data-row="${row.row_index}" data-slot-key="${escapeHtml(key)}" checked onchange="refreshImage(${row.row_index})" /> ${label}</label>`;
      }).join('');
      const modeHint = row.type_label ? ` Instruction mode: <strong>${escapeHtml(row.type_label)}</strong>.` : '';
      markHtml = `<div class="mark-panel">
        <div class="mark-hint">QA_images are stored unmarked (raw bytes).${modeHint} Overlay is client-side from mark_spec only when you check slots below:</div>
        <label style="display:block;margin-bottom:6px;font-weight:600;">
          <input type="checkbox" class="mark-select-all" data-row="${row.row_index}" checked onchange="toggleAllMarks(${row.row_index}, this.checked)" /> Select all marks
        </label>
        ${checks}
      </div>`;
    }

    let overlay3dHtml = '';
    if (row.can_3d_overlay) {
      overlay3dHtml = `<label style="font-size:13px;margin-bottom:10px;display:block;">
        <input type="checkbox" class="overlay-3d-cb" data-row="${row.row_index}" checked onchange="refreshImage(${row.row_index})" /> Show 3D bbox overlay
      </label>`;
    }

    let turnsHtml = '';
    if (row.turns && row.turns.length > 0) {
      row.turns.forEach((turn, tIdx) => {
        const isActive = row.active_turn_index === tIdx;
        const activeCls = isActive ? ' turn-active' : '';
        const activeBadge = isActive ? '<span class="tag tag-active">this row</span>' : '';
        const prefix = (turn.question_prefix || '').trim();
        const qBody = (turn.question_text || turn.question || '').replace(/<image>\s*/g, '').trim();
        const turnLabel = isMultiTurn ? `<span class="multi-turn-label">Turn ${tIdx + 1}${turn.turn_id != null ? ' (id ' + turn.turn_id + ')' : ''} ${activeBadge}</span>` : '';
        if (tIdx > 0) turnsHtml += '<hr class="turn-divider">';
        let prefixHtml = '';
        if (prefix && prefix !== qBody) {
          prefixHtml = `<div class="qa-text prefix">${escapeHtml(prefix)}</div>`;
        }
        turnsHtml += `
          <div class="qa-block${activeCls}">
            ${turnLabel}
            <div class="qa-label q">Question</div>
            ${prefixHtml}
            <div class="qa-text q">${escapeHtml(qBody || turn.question || '')}</div>
          </div>
          <div class="qa-block" style="margin-top: 10px;">
            <div class="qa-label a">Answer</div>
            <div class="qa-text a">${escapeHtml(turn.answer || '')}</div>
          </div>`;
      });
    }

    html += `
    <div class="card" data-row-index="${row.row_index}">
      <div class="card-header">
        <strong>#${globalIdx + 1}</strong>
        ${tagsHtml} ${typeHtml} ${turnBadge}
        <button type="button" class="btn-raw" onclick="toggleRaw(${row.row_index}, this)">Raw row</button>
      </div>
      <div class="card-body">
        ${overlay3dHtml}
        ${markHtml}
        ${imagesHtml}
        ${turnsHtml}
        <pre class="raw-panel" id="raw-${row.row_index}"></pre>
      </div>
    </div>`;
  });
  container.innerHTML = html;
  rows.forEach(row => {
    if (row.mark_slots && row.mark_slots.length) refreshImage(row.row_index);
  });
  window.scrollTo(0, 0);
}

function cardForRow(rowIndex) {
  return document.querySelector(`.card[data-row-index="${rowIndex}"]`);
}

function selectedSlots(rowIndex) {
  const card = cardForRow(rowIndex);
  if (!card) return [];
  return Array.from(card.querySelectorAll('.mark-cb:checked')).map(b => b.dataset.slotKey);
}

function toggleAllMarks(rowIndex, checked) {
  const card = cardForRow(rowIndex);
  if (!card) return;
  card.querySelectorAll('.mark-cb').forEach(cb => { cb.checked = checked; });
  const master = card.querySelector('.mark-select-all');
  if (master) master.checked = checked;
  refreshImage(rowIndex);
}

function refreshImage(rowIndex) {
  const card = cardForRow(rowIndex);
  if (!card) return;
  const slots = selectedSlots(rowIndex);
  const allCbs = card.querySelectorAll('.mark-cb');
  const master = card.querySelector('.mark-select-all');
  if (master && allCbs.length) {
    master.checked = slots.length === allCbs.length;
    master.indeterminate = slots.length > 0 && slots.length < allCbs.length;
  }
  const overlay3d = card.querySelector('.overlay-3d-cb')?.checked || false;
  const params = new URLSearchParams({
    path: currentPath,
    index: String(rowIndex),
    slots: slots.join(','),
    marks_mode: slots.length ? 'selected' : 'off',
    overlay_3d: overlay3d ? '1' : '0',
  });
  fetch('/api/render?' + params)
    .then(r => r.json())
    .then(data => {
      const imgs = card.querySelectorAll('.images-row img');
      (data.images || []).forEach((src, i) => {
        if (imgs[i]) imgs[i].src = src;
      });
    });
}

function toggleRaw(rowIndex, btn) {
  const panel = document.getElementById('raw-' + rowIndex);
  if (panel.classList.contains('open')) {
    panel.classList.remove('open');
    btn.textContent = 'Raw row';
    return;
  }
  btn.textContent = 'Hide raw';
  panel.classList.add('open');
  panel.textContent = 'Loading...';
  fetch(`/api/raw_row?path=${encodeURIComponent(currentPath)}&index=${rowIndex}`)
    .then(r => r.json())
    .then(data => {
      panel.textContent = JSON.stringify(data.row, null, 2);
    })
    .catch(() => { panel.textContent = 'Failed to load raw row.'; });
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text == null ? '' : String(text);
  return div.innerHTML;
}

function openLightbox(src) {
  document.getElementById('lightboxImg').src = src;
  document.getElementById('lightbox').classList.add('active');
}
function closeLightbox() {
  document.getElementById('lightbox').classList.remove('active');
}
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeLightbox();
  if (e.key === 'ArrowLeft') navigate(-1);
  if (e.key === 'ArrowRight') navigate(1);
});

window.onload = () => {
  const sel = document.getElementById('taskSelect');
  if (sel.value) loadTask();
};
</script>
</body>
</html>
"""


# ──────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    tasks = discover_parquets(DATA_DIR)
    selected = request.args.get("task", "")
    return render_template_string(
        HTML_TEMPLATE,
        tasks=tasks,
        selected_path=selected,
        data_root_name=data_root_name(),
    )


def _task_name_from_parquet_path(path):
    parts = os.path.normpath(path).split(os.sep)
    return parts[-2] if len(parts) >= 2 else ""


def _load_row(path, index):
    if not path or not os.path.exists(path):
        return None, None
    df = pd.read_parquet(path)
    if index < 0 or index >= len(df):
        return None, df
    return df.iloc[index], df


@app.route("/api/filter_options")
def api_filter_options():
    path = request.args.get("path", "")
    return jsonify({"tasks": collect_dataset_task_names(path)})


@app.route("/api/data")
def api_data():
    path = request.args.get("path", "")
    page = int(request.args.get("page", 0))
    page_size = int(request.args.get("page_size", 10))
    filter_task = request.args.get("filter_task", "")
    filter_turns = request.args.get("filter_turns", "")

    if not path or not os.path.exists(path):
        return jsonify({
            "total": 0, "filtered_total": 0, "page": 0, "rows": [], "is_3d_task": False,
        })

    df = pd.read_parquet(path)
    total = len(df)
    indices = filtered_row_indices(
        path, filter_task=filter_task, filter_turns=filter_turns,
    )
    filtered_total = len(indices)
    page_indices = indices[page * page_size : (page + 1) * page_size]
    task_name = _task_name_from_parquet_path(path)
    is_3d = "3d_grounding" in task_name.lower()

    rows = []
    for i in page_indices:
        row = df.iloc[i]
        parsed = parse_row(row)
        parsed["meta"] = _normalize_meta(row.get("metadata"))

        can_3d = _is_3d_grounding_task(task_name, parsed)
        images, _, marks_applied = build_display_images(
            row, parsed, task_name, overlay_3d=can_3d, marks_mode="all",
        )

        rows.append({
            "row_index": i,
            "turns": parsed["turns"],
            "active_turn_index": parsed["active_turn_index"],
            "display_images": [pil_to_base64(img) for img in images],
            "marks_overlay_applied": marks_applied,
            "tags": parsed["tags"] if isinstance(parsed["tags"], list) else [parsed["tags"]],
            "question_type": parsed["question_type"],
            "mark_slots": parsed["mark_slots"],
            "type_label": parsed.get("type_label", ""),
            "can_3d_overlay": can_3d,
        })

    return jsonify({
        "total": total,
        "filtered_total": filtered_total,
        "page": page,
        "rows": rows,
        "is_3d_task": is_3d,
    })


@app.route("/api/render")
def api_render():
    path = request.args.get("path", "")
    index = int(request.args.get("index", 0))
    slots_raw = request.args.get("slots", "")
    marks_mode = request.args.get("marks_mode", "off")
    overlay_3d = request.args.get("overlay_3d", "0") == "1"
    slot_ids = [s.strip() for s in slots_raw.split(",") if s.strip()]
    if marks_mode not in ("off", "selected", "all"):
        marks_mode = "all" if not slot_ids and request.args.get("slots") is None else "selected"
    if marks_mode == "selected" and not slot_ids:
        marks_mode = "off"

    row, _ = _load_row(path, index)
    if row is None:
        return jsonify({"images": []})

    task_name = _task_name_from_parquet_path(path)
    parsed = parse_row(row)
    parsed["meta"] = _normalize_meta(row.get("metadata"))
    images, _, marks_applied = build_display_images(
        row, parsed, task_name,
        slot_ids=slot_ids,
        overlay_3d=overlay_3d,
        marks_mode=marks_mode,
    )
    return jsonify({
        "images": [pil_to_base64(img) for img in images],
        "marks_overlay_applied": marks_applied,
    })


@app.route("/api/raw_row")
def api_raw_row():
    path = request.args.get("path", "")
    index = int(request.args.get("index", 0))
    row, _ = _load_row(path, index)
    if row is None:
        return jsonify({"row": {}})
    return jsonify({"row": serialize_row_raw(row)})


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def _lan_urls(port: int) -> List[str]:
    seen: set = set()
    urls: List[str] = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            if ip and not ip.startswith("127."):
                seen.add(ip)
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip and not ip.startswith("127."):
                seen.add(ip)
    except OSError:
        pass
    for ip in sorted(seen):
        urls.append(f"http://{ip}:{port}")
    return urls


def _print_listen_info(host: str, port: int) -> None:
    print(f"\nListening on http://{host}:{port}")
    print(f"  This machine: http://127.0.0.1:{port}")
    if host in ("0.0.0.0", "::"):
        lan = _lan_urls(port)
        if lan:
            print("  Other machines on the same network:")
            for url in lan:
                print(f"    {url}")
        else:
            print(f"  Other machines: use this PC's LAN IP, e.g. http://<your-ip>:{port}")
        print(
            f"  If remote browsers cannot connect, allow TCP port {port} in Windows Firewall."
        )
    elif host in ("127.0.0.1", "localhost"):
        print("  WARNING: --host is loopback only; other machines cannot connect.")
        print(f"  Restart with: --host 0.0.0.0 --port {port}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenSpatial Annotation Visualizer")
    parser.add_argument("--port", type=int, default=8888, help="Server port")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Bind address (default 0.0.0.0 = all interfaces, reachable on LAN)",
    )
    parser.add_argument("--data_dir", type=str, default="output/debug", help="Root directory containing parquet outputs")
    args = parser.parse_args()

    DATA_DIR = os.path.abspath(args.data_dir)
    tasks = discover_parquets(DATA_DIR)
    print(f"Data root: {data_root_name()} ({DATA_DIR})")
    print(f"Found {len(tasks)} task outputs:")
    for t in tasks:
        print(f"  {t['label']} -> {t['path']}")
    _print_listen_info(args.host, args.port)

    app.run(host=args.host, port=args.port, debug=False)
