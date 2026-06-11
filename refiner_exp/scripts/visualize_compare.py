#!/usr/bin/env python3
"""Rendering helpers for Raw/SAM2/SAM3 refiner comparison (used by serve_compare)."""

from __future__ import annotations

import base64
import io
import json
import os
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from summarize_runs import (  # noqa: E402
    BRANCHES,
    _as_list,
    _load_mask,
    _load_matrix,
    _mask_bbox,
    collect_records,
    compute_box_3d_corners_from_params,
    enrich_against_raw,
    load_stage_df,
    mask_iou,
    resolve_run_root,
)
try:
    import open3d as o3d
except ImportError:  # pragma: no cover
    o3d = None


# Overlay opacity 0–255; higher = more opaque (less transparent).
MASK_OVERLAY_ALPHA = 180

PALETTE = [
    (255, 60, 60),
    (60, 180, 255),
    (60, 220, 100),
    (255, 200, 60),
    (200, 60, 255),
    (60, 220, 220),
    (255, 130, 60),
    (160, 220, 80),
]


def _font(size: int = 18) -> ImageFont.ImageFont:
    """Return a TrueType font when possible (required for textbbox on Windows)."""
    candidates: List[Path] = []
    if os.name == "nt":
        fonts_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
        for fname in ("arial.ttf", "Arial.ttf", "segoeui.ttf", "calibri.ttf", "msyh.ttc"):
            candidates.append(fonts_dir / fname)
    for path in (
        *candidates,
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("DejaVuSans.ttf"),
        Path("arial.ttf"),
    ):
        if not path.is_file():
            continue
        try:
            return ImageFont.truetype(str(path), size)
        except OSError:
            continue
    try:
        # Pillow >= 10.1: bundled Aileron (FreeType), supports textbbox.
        return ImageFont.load_default(size=size)
    except TypeError:
        pass
    return ImageFont.load_default()


def _text_bbox(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
) -> Tuple[int, int, int, int]:
    if hasattr(font, "getbbox"):
        try:
            left, top, right, bottom = font.getbbox(text)
            return (xy[0] + left, xy[1] + top, xy[0] + right, xy[1] + bottom)
        except (KeyError, OSError, ValueError, AttributeError):
            pass
    try:
        return draw.textbbox(xy, text, font=font)
    except (AttributeError, KeyError, OSError, ValueError):
        w = max(len(text) * 8, 8)
        return (xy[0], xy[1], xy[0] + w, xy[1] + 18)


def _color_for_key(key: str) -> Tuple[int, int, int]:
    return PALETTE[abs(hash(key)) % len(PALETTE)]


def _sanitize_name(text: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return clean[:140] or "sample"


@lru_cache(maxsize=2048)
def _load_mask_cached(mask_path: str) -> Optional[np.ndarray]:
    if not mask_path:
        return None
    return _load_mask(mask_path)


@lru_cache(maxsize=64)
def _load_pose_intrinsic(pose_path: str, intrinsic_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    return _load_matrix(pose_path), _load_matrix(intrinsic_path)


def _pcd_vertex_count(path: Optional[str]) -> Optional[int]:
    if not path or not Path(path).exists():
        return None
    try:
        with Path(path).open("r", encoding="ascii", errors="ignore") as f:
            for line in f:
                if line.startswith("element vertex"):
                    return int(line.split()[-1])
    except Exception:
        return None
    return None


def _log(msg: str) -> None:
    print(msg, flush=True)


def _load_image(image_ref: Any) -> Optional[Image.Image]:
    try:
        if isinstance(image_ref, dict) and image_ref.get("bytes"):
            return Image.open(io.BytesIO(image_ref["bytes"])).convert("RGB")
        if isinstance(image_ref, str) and image_ref:
            return Image.open(image_ref).convert("RGB")
    except Exception:
        return None
    return None


def _draw_label(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, color: Tuple[int, int, int]) -> None:
    font = _font(16)
    box = _text_bbox(draw, xy, text, font)
    draw.rectangle(box, fill=color)
    draw.text(xy, text, fill=(255, 255, 255), font=font)


def _project_box_edges(rec: Dict[str, Any]) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    box = rec.get("box_3d")
    pose_path = rec.get("pose") or ""
    intrinsic_path = rec.get("intrinsic") or ""
    pose, intrinsic = _load_pose_intrinsic(str(pose_path), str(intrinsic_path))
    if not box or pose is None or intrinsic is None:
        return []
    try:
        corners = compute_box_3d_corners_from_params(_as_list(box))
        corners_h = np.concatenate([corners, np.ones((corners.shape[0], 1))], axis=1)
        cam = (np.linalg.inv(pose) @ corners_h.T).T[:, :3]
        valid = cam[:, 2] > 1e-3
        k = intrinsic[:3, :3]
        uv = np.full((8, 2), np.nan, dtype=np.float64)
        uv[valid, 0] = k[0, 0] * cam[valid, 0] / cam[valid, 2] + k[0, 2]
        uv[valid, 1] = k[1, 1] * cam[valid, 1] / cam[valid, 2] + k[1, 2]
    except Exception:
        return []

    edge_idx = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    edges = []
    for a, b in edge_idx:
        if valid[a] and valid[b]:
            edges.append(((float(uv[a, 0]), float(uv[a, 1])), (float(uv[b, 0]), float(uv[b, 1]))))
    return edges


def _resize_panel(panel: Image.Image, max_width: int) -> Image.Image:
    if panel.width <= max_width:
        return panel
    scale = max_width / float(panel.width)
    return panel.resize((max_width, int(panel.height * scale)), resample=Image.BILINEAR)


def _draw_object_panel(
    base_image: Image.Image,
    rec: Dict[str, Any],
    *,
    max_width: int,
) -> Image.Image:
    """Draw one object's mask, 2D bbox, and projected 3D bbox on a fresh RGB frame."""
    canvas = base_image.convert("RGBA")
    draw = ImageDraw.Draw(canvas)
    color = _color_for_key(rec["object_key"])
    mask = _load_mask_cached(str(rec.get("mask_path") or ""))
    if mask is not None:
        if mask.shape[::-1] != canvas.size:
            mask_img = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
            mask_img = mask_img.resize(canvas.size, resample=Image.NEAREST)
            mask = np.array(mask_img) > 0
        alpha = Image.fromarray((mask.astype(np.uint8) * MASK_OVERLAY_ALPHA), mode="L")
        fill = Image.new("RGBA", canvas.size, (*color, MASK_OVERLAY_ALPHA))
        overlay = Image.new("RGBA", canvas.size, (*color, 0))
        overlay.paste(fill, mask=alpha)
        canvas = Image.alpha_composite(canvas, overlay)
        draw = ImageDraw.Draw(canvas)
        bbox = _mask_bbox(mask)
        if bbox:
            draw.rectangle(bbox, outline=color, width=3)
    for p1, p2 in _project_box_edges(rec):
        draw.line([p1, p2], fill=(*color, 230), width=3)
    label = f"{rec.get('object_index', '?')}:{rec.get('tag', '')}"
    _draw_label(draw, (8, 8), label, color)
    return _resize_panel(canvas.convert("RGB"), max_width)


def _camera_points_to_viewer(pts: np.ndarray) -> np.ndarray:
    """OpenCV camera (x right, y down, z forward) → Three.js Y-up viewer."""
    out = np.asarray(pts, dtype=np.float32)
    out[:, 1] *= -1.0
    return out


@lru_cache(maxsize=1024)
def _load_pcd_points(path: str, max_points: int) -> Tuple[np.ndarray, ...]:
    """Load fusion-stage per-object .pcd (camera frame, outlier-cleaned)."""
    empty = np.empty((0, 3), dtype=np.float32)
    if not path or not Path(path).exists() or o3d is None:
        return (empty,)
    try:
        pts = np.asarray(o3d.io.read_point_cloud(path).points, dtype=np.float32)
    except Exception:
        return (empty,)
    if pts.size == 0:
        return (empty,)
    if max_points and len(pts) > max_points:
        rng = np.random.default_rng(0)
        keep = rng.choice(len(pts), size=max_points, replace=False)
        pts = pts[keep]
    return (_camera_points_to_viewer(pts),)


def _pack_branch_from_fusion(
    refine_records: List[Dict[str, Any]],
    fusion_by_key: Dict[str, Dict[str, Any]],
    max_points_per_object: int,
) -> Dict[str, Any]:
    """Pack per-object fusion .pcd clouds for one branch (colors match 2D overlay)."""
    objects: List[Dict[str, Any]] = []
    chunks: List[np.ndarray] = []
    for rec in refine_records:
        fusion = fusion_by_key.get(rec["object_key"], {})
        path = str(fusion.get("pointcloud_path") or "")
        pts = _load_pcd_points(path, max_points_per_object)[0]
        if len(pts) == 0:
            continue
        r, g, b = _color_for_key(rec["object_key"])
        objects.append({
            "tag": rec.get("tag", ""),
            "object_index": rec.get("object_index"),
            "n": int(len(pts)),
            "positions": base64.b64encode(pts.tobytes()).decode("ascii"),
            "color": [int(r), int(g), int(b)],
            "pcd_path": path,
        })
        chunks.append(pts)
    if not chunks:
        return {"objects": [], "n": 0, "centroid": [0.0, 0.0, 1.0], "extent": 1.0}
    all_pts = np.vstack(chunks)
    centroid = all_pts.mean(axis=0)
    extent = float(np.linalg.norm(all_pts.max(axis=0) - all_pts.min(axis=0)))
    return {
        "objects": objects,
        "n": int(len(all_pts)),
        "centroid": centroid.astype(float).tolist(),
        "extent": max(extent, 1e-3),
    }


def _records_by_image(records: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        out.setdefault(rec["image_key"], []).append(rec)
    return out


def _records_by_key(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {rec["object_key"]: rec for rec in records}


def load_branch_records(
    branch: str,
    run_root: Path,
    raw_filter_records: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    refine_task = BRANCHES[branch]["refine_task"]
    if refine_task is None:
        refine_df = load_stage_df(run_root, "filter_stage", "3dbox_filter")
    else:
        refine_df = load_stage_df(run_root, "localization_stage", refine_task)
    fusion_df = load_stage_df(run_root, "scene_fusion_stage", "depth_back_projection")
    refine_records = collect_records(
        refine_df, branch=branch, stage_label="refine", run_root=run_root, include_assets=False,
    )
    fusion_records = collect_records(
        fusion_df, branch=branch, stage_label="fusion", run_root=run_root, include_assets=False,
    )
    enrich_against_raw(refine_records, raw_filter_records, include_assets=False)
    return refine_records, fusion_records


def _enrich_image_records_for_stats(
    branch_records: Dict[str, List[Dict[str, Any]]],
    raw_by_key: Dict[str, Dict[str, Any]],
) -> None:
    """Compute mask IoU/area only for objects shown in this image (lazy)."""
    raw_masks: Dict[str, Optional[np.ndarray]] = {}
    for branch in ("raw", "sam2", "sam3"):
        for rec in branch_records.get(branch, []):
            key = rec["object_key"]
            mask = _load_mask_cached(str(rec.get("mask_path") or ""))
            if mask is not None:
                rec["mask_area"] = int(mask.sum())
                rec["mask_bbox"] = _mask_bbox(mask)
            raw = raw_by_key.get(key)
            if raw is None:
                rec["mask_iou_with_raw"] = None
                continue
            if key not in raw_masks:
                raw_masks[key] = _load_mask_cached(str(raw.get("mask_path") or ""))
            rec["mask_iou_with_raw"] = mask_iou(mask, raw_masks.get(key))


@dataclass
class CompareIndex:
    """In-memory index for refiner comparison (parquet metadata, no bulk IO)."""

    image_keys: List[str]
    name_by_key: Dict[str, str]
    key_by_name: Dict[str, str]
    by_image: Dict[str, Dict[str, List[Dict[str, Any]]]]
    fusion_by_key: Dict[str, Dict[str, Dict[str, Any]]]
    raw_by_key: Dict[str, Dict[str, Any]]


def build_compare_index(
    raw_run: str,
    sam2_run: str,
    sam3_run: str,
    *,
    max_images: Optional[int] = 20,
) -> CompareIndex:
    _log("Resolving run directories...")
    run_roots = {
        "raw": resolve_run_root(raw_run, BRANCHES["raw"]["config_stem"]),
        "sam2": resolve_run_root(sam2_run, BRANCHES["sam2"]["config_stem"]),
        "sam3": resolve_run_root(sam3_run, BRANCHES["sam3"]["config_stem"]),
    }

    _log("Loading parquet metadata (light mode)...")
    raw_filter_df = load_stage_df(run_roots["raw"], "filter_stage", "3dbox_filter")
    raw_filter_records = collect_records(
        raw_filter_df, branch="raw", stage_label="filter",
        run_root=run_roots["raw"], include_assets=False,
    )
    raw_by_key = _records_by_key(raw_filter_records)

    refine_by_branch: Dict[str, List[Dict[str, Any]]] = {}
    fusion_by_branch: Dict[str, List[Dict[str, Any]]] = {}
    for branch, root in run_roots.items():
        _log(f"  indexing branch={branch} ...")
        refine_by_branch[branch], fusion_by_branch[branch] = load_branch_records(
            branch, root, raw_filter_records,
        )

    by_image = {branch: _records_by_image(records) for branch, records in refine_by_branch.items()}
    fusion_by_key = {branch: _records_by_key(records) for branch, records in fusion_by_branch.items()}
    image_keys = sorted(set().union(*(set(items.keys()) for items in by_image.values())))
    if max_images:
        image_keys = image_keys[:max_images]

    name_by_key = {k: _sanitize_name(k) for k in image_keys}
    key_by_name = {v: k for k, v in name_by_key.items()}
    return CompareIndex(
        image_keys=image_keys,
        name_by_key=name_by_key,
        key_by_name=key_by_name,
        by_image=by_image,
        fusion_by_key=fusion_by_key,
        raw_by_key=raw_by_key,
    )


def branch_records_for_image(index: CompareIndex, image_key: str) -> Dict[str, List[Dict[str, Any]]]:
    return {
        branch: index.by_image.get(branch, {}).get(image_key, [])
        for branch in ("raw", "sam2", "sam3")
    }


def _record_for_panel(
    branch_records: Dict[str, List[Dict[str, Any]]],
    branch: str,
    object_index: int,
) -> Optional[Dict[str, Any]]:
    records = branch_records.get(branch, [])
    for rec in records:
        if rec.get("object_index") == object_index:
            return rec
    if 0 <= object_index < len(records):
        return records[object_index]
    return None


def render_object_panel_image(
    index: CompareIndex,
    image_key: str,
    branch: str,
    object_index: int,
    *,
    max_panel_width: int = 640,
) -> Optional[Image.Image]:
    """Render a single per-object 2D panel (mask + 2D bbox + 3D wireframe)."""
    if branch not in ("raw", "sam2", "sam3"):
        return None
    branch_records = branch_records_for_image(index, image_key)
    rec = _record_for_panel(branch_records, branch, object_index)
    if rec is None:
        return None
    image = _load_image(rec.get("image"))
    if image is None:
        return None
    return _draw_object_panel(image, rec, max_width=max_panel_width)


def sample_panels_json(index: CompareIndex, image_key: str) -> Dict[str, Any]:
    branch_records = branch_records_for_image(index, image_key)
    return {
        "image_key": image_key,
        "name": index.name_by_key[image_key],
        "branches": {
            branch: [
                {
                    "object_index": rec.get("object_index"),
                    "tag": rec.get("tag"),
                    "label": f"{rec.get('object_index', '?')}:{rec.get('tag', '')}",
                }
                for rec in branch_records.get(branch, [])
            ]
            for branch in ("raw", "sam2", "sam3")
        },
    }


def sample_stats_json(index: CompareIndex, image_key: str) -> Dict[str, Any]:
    branch_records = branch_records_for_image(index, image_key)
    _enrich_image_records_for_stats(branch_records, index.raw_by_key)
    return {
        "image_key": image_key,
        "name": index.name_by_key[image_key],
        "branches": {
            branch: [
                {
                    "object_key": rec["object_key"],
                    "tag": rec.get("tag"),
                    "object_index": rec.get("object_index"),
                    "mask_area": rec.get("mask_area"),
                    "mask_iou_with_raw": rec.get("mask_iou_with_raw"),
                    "point_count": _pcd_vertex_count(
                        (index.fusion_by_key.get(branch, {}).get(rec["object_key"]) or {}).get("pointcloud_path")
                    ),
                }
                for rec in branch_records[branch]
            ]
            for branch in ("raw", "sam2", "sam3")
        },
    }


