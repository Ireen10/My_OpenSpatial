#!/usr/bin/env python3
"""Local web server for Raw/SAM2/SAM3 refiner comparison (interactive 3D viewer)."""

from __future__ import annotations

import argparse
import base64
import html as html_module
import io
import json
import os
import re
import socket
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

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
    convert_box_3d_world_to_camera,
    enrich_against_raw,
    load_stage_df,
    mask_iou,
    resolve_run_root,
)
try:
    import open3d as o3d
except ImportError:  # pragma: no cover
    o3d = None


# Bump when panel rendering / coloring / 3D coords change (invalidates disk cache).
PANEL_RENDER_VERSION = "v6"

# 12 edges of an 8-corner OBB (same order as compute_box_3d_corners).
_BOX_EDGE_IDX: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)

# Ten high-contrast colors; object_index % 10 picks one (per branch, not cross-column).
OBJECT_COLORS: Tuple[Tuple[int, int, int], ...] = (
    (230, 25, 75),    # red
    (60, 180, 75),    # green
    (0, 130, 200),    # blue
    (255, 165, 0),    # orange
    (145, 30, 180),   # purple
    (0, 200, 200),    # cyan
    (255, 0, 180),    # magenta
    (210, 220, 0),    # yellow-green
    (120, 60, 30),    # brown
    (255, 105, 180),  # pink
)

# Overlay opacity 0–255; higher = more opaque (less transparent).
MASK_OVERLAY_ALPHA = 180


def _log(msg: str) -> None:
    print(msg, flush=True)


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


def _index_color(object_index: int) -> Tuple[int, int, int]:
    """Palette color by object_index; 2D and 3D in the same column use the same index."""
    return OBJECT_COLORS[int(object_index) % len(OBJECT_COLORS)]


def _sanitize_name(text: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return clean[:140] or "sample"


@lru_cache(maxsize=16384)
def _load_mask_cached(mask_path: str) -> Optional[np.ndarray]:
    if not mask_path:
        return None
    return _load_mask(mask_path)


@lru_cache(maxsize=512)
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
    color = _index_color(int(rec.get("object_index", 0)))
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


def opencv_camera_to_viewer(pts: np.ndarray) -> np.ndarray:
    """OpenCV camera frame → Three.js viewer frame.

    Annotation / depth back-projection use OpenCV camera coords:
    +X right, +Y down, +Z forward (into the scene).

    Three.js uses +X right, +Y up, camera looks down -Z. Map with (x, -y, -z).
    """
    out = np.asarray(pts, dtype=np.float32).copy()
    out[:, 1] *= -1.0
    out[:, 2] *= -1.0
    return out


def _box_corners_viewer(rec: Dict[str, Any]) -> Optional[np.ndarray]:
    """World-frame 9-param box → 8 corners in viewer coords (camera frame first)."""
    box = rec.get("box_3d")
    if not box:
        return None
    pose_path = str(rec.get("pose") or "")
    intrinsic_path = str(rec.get("intrinsic") or "")
    pose, _ = _load_pose_intrinsic(pose_path, intrinsic_path)
    if pose is None:
        return None
    try:
        cam_box = convert_box_3d_world_to_camera(_as_list(box), pose)
        if cam_box is None:
            return None
        corners = compute_box_3d_corners_from_params(cam_box)
        return opencv_camera_to_viewer(corners)
    except Exception:
        return None


def _wireframe_segments(corners: np.ndarray) -> np.ndarray:
    """Flatten 12 box edges to (24, 3) line segment endpoints."""
    segs = np.empty((len(_BOX_EDGE_IDX) * 2, 3), dtype=np.float32)
    for i, (a, b) in enumerate(_BOX_EDGE_IDX):
        segs[i * 2] = corners[a]
        segs[i * 2 + 1] = corners[b]
    return segs


@lru_cache(maxsize=8192)
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
    return (opencv_camera_to_viewer(pts),)


def _pack_branch_from_fusion(
    fusion_records: List[Dict[str, Any]],
    max_points_per_object: int,
) -> Dict[str, Any]:
    """Pack per-object fusion .pcd clouds for one branch (colors match 2D overlay)."""
    objects: List[Dict[str, Any]] = []
    chunks: List[np.ndarray] = []
    for rec in fusion_records:
        path = str(rec.get("pointcloud_path") or "")
        pts = _load_pcd_points(path, max_points_per_object)[0]
        if len(pts) == 0:
            continue
        r, g, b = _index_color(int(rec.get("object_index", 0)))
        obj: Dict[str, Any] = {
            "tag": rec.get("tag", ""),
            "object_index": rec.get("object_index"),
            "object_key": rec["object_key"],
            "n": int(len(pts)),
            "positions": base64.b64encode(pts.tobytes()).decode("ascii"),
            "color": [int(r), int(g), int(b)],
            "pcd_path": path,
        }
        corners = _box_corners_viewer(rec)
        if corners is not None:
            wire = _wireframe_segments(corners)
            obj["wireframe"] = base64.b64encode(wire.tobytes()).decode("ascii")
            obj["wireframe_n"] = int(len(wire))
        objects.append(obj)
        chunks.append(pts)
    if not chunks:
        return {"objects": [], "n": 0, "centroid": [0.0, 0.0, -1.0], "extent": 1.0}
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
) -> List[Dict[str, Any]]:
    """Load per-object records from scene_fusion (final retained pipeline output)."""
    fusion_df = load_stage_df(run_root, "scene_fusion_stage", "depth_back_projection")
    fusion_records = collect_records(
        fusion_df, branch=branch, stage_label="fusion", run_root=run_root, include_assets=False,
    )
    enrich_against_raw(fusion_records, raw_filter_records, include_assets=False)
    return fusion_records


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
    """In-memory index for refiner comparison (fusion-stage survivors only)."""

    image_keys: List[str]
    name_by_key: Dict[str, str]
    key_by_name: Dict[str, str]
    by_image: Dict[str, Dict[str, List[Dict[str, Any]]]]
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

    fusion_by_branch: Dict[str, List[Dict[str, Any]]] = {}
    for branch, root in run_roots.items():
        _log(f"  indexing branch={branch} (fusion) ...")
        fusion_by_branch[branch] = load_branch_records(
            branch, root, raw_filter_records,
        )

    by_image = {branch: _records_by_image(records) for branch, records in fusion_by_branch.items()}
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
                    "object_key": rec.get("object_key"),
                    "tag": rec.get("tag"),
                    "label": f"{rec.get('object_index', '?')}:{rec.get('tag', '')}",
                    "color": list(_index_color(int(rec.get("object_index", 0)))),
                }
                for rec in branch_records.get(branch, [])
            ]
            for branch in ("raw", "sam2", "sam3")
        },
    }


def _viewer_nav_html(
    index: CompareIndex,
    image_key: str,
    *,
    query_suffix: str = "",
) -> Tuple[str, str, str]:
    """Prev / position / Next links for the sample viewer page."""
    keys = index.image_keys
    try:
        i = keys.index(image_key)
    except ValueError:
        return (
            '<span class="nav-disabled">← Prev</span>',
            "0 / 0",
            '<span class="nav-disabled">Next →</span>',
        )
    total = len(keys)
    pos = f"{i + 1} / {total}"
    if i > 0:
        prev_name = html_module.escape(index.name_by_key[keys[i - 1]], quote=True)
        prev = f'<a class="nav-btn" href="/view/{prev_name}{query_suffix}">← Prev</a>'
    else:
        prev = '<span class="nav-disabled">← Prev</span>'
    if i + 1 < total:
        next_name = html_module.escape(index.name_by_key[keys[i + 1]], quote=True)
        nxt = f'<a class="nav-btn" href="/view/{next_name}{query_suffix}">Next →</a>'
    else:
        nxt = '<span class="nav-disabled">Next →</span>'
    return prev, pos, nxt


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
                    "point_count": _pcd_vertex_count(rec.get("pointcloud_path")),
                }
                for rec in branch_records[branch]
            ]
            for branch in ("raw", "sam2", "sam3")
        },
    }


_INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Refiner Compare</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; background: #111; color: #eee; }}
    h1 {{ font-size: 1.2rem; }}
    a {{ color: #7eb8ff; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    li {{ margin: 6px 0; }}
    .hint {{ color: #888; font-size: 0.9rem; }}
  </style>
</head>
<body>
  <h1>Refiner 对照实验</h1>
  <p class="hint">共 {count} 个样本（各分支 fusion 并集，仅含最终留存）· 点云来自 depth_back_projection</p>
  <ul>
{rows}
  </ul>
</body>
</html>
"""

_VIEWER_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: system-ui, sans-serif; background: #1a1a1a; color: #eee; }}
    header {{ padding: 12px 16px; background: #242424; border-bottom: 1px solid #333; }}
    header h1 {{ margin: 0 0 6px; font-size: 1.05rem; }}
    header p {{ margin: 0; font-size: 0.85rem; color: #aaa; }}
    a {{ color: #7eb8ff; }}
    .pager {{ display: flex; align-items: center; gap: 12px; margin: 8px 0 4px; }}
    .pager .nav-pos {{ color: #ccc; font-size: 0.9rem; min-width: 4.5rem; text-align: center; }}
    .nav-btn {{ padding: 4px 12px; background: #333; border-radius: 4px; text-decoration: none; }}
    .nav-btn:hover {{ background: #444; text-decoration: none; }}
    .nav-disabled {{ color: #555; padding: 4px 12px; }}
    .panels-2d {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; padding: 12px 16px; background: #111; }}
    .branch-col {{ background: #242424; border: 1px solid #333; border-radius: 6px; padding: 8px; }}
    .branch-col h3 {{ margin: 0 0 8px; font-size: 0.9rem; color: #ccc; }}
    .thumb-list {{ display: flex; flex-direction: column; gap: 8px; }}
    .thumb-list img {{ width: 100%; border: 1px solid #444; border-radius: 4px; cursor: zoom-in; display: block; content-visibility: auto; contain-intrinsic-size: 360px; }}
    .thumb-list .empty {{ color: #888; font-size: 0.85rem; padding: 8px 0; }}
    .lightbox {{ position: fixed; inset: 0; z-index: 1000; display: flex; align-items: center; justify-content: center; }}
    .lightbox.hidden {{ display: none; }}
    .lightbox-backdrop {{ position: absolute; inset: 0; background: rgba(0,0,0,0.88); }}
    #lightbox-img {{ position: relative; max-width: 96vw; max-height: 96vh; object-fit: contain; border-radius: 4px; cursor: zoom-out; }}
    .viewers {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; padding: 12px 16px 20px; }}
    .panel {{ background: #242424; border: 1px solid #333; border-radius: 6px; min-height: 320px; display: flex; flex-direction: column; }}
    .panel h2 {{ margin: 0; padding: 8px 10px; font-size: 0.9rem; background: #2e2e2e; }}
    .panel .meta {{ padding: 6px 10px; font-size: 0.78rem; color: #aaa; }}
    .panel canvas {{ flex: 1; width: 100%; min-height: 280px; cursor: grab; }}
    .stats {{ padding: 12px 16px; font-size: 0.8rem; color: #aaa; max-height: 200px; overflow: auto; }}
    @media (max-width: 960px) {{
      .panels-2d {{ grid-template-columns: 1fr; }}
      .viewers {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <div class="pager">
      {nav_prev}
      <span class="nav-pos">{nav_pos}</span>
      {nav_next}
    </div>
    <p>拖拽旋转 · 滚轮缩放 · 右键平移 · 3D 视图中彩色线框为相机系 3D 标注框 · <span style="color:#ccc">点击图片放大</span></p>
    <p><a href="/">← 样本列表</a> · <a href="?refresh=1">刷新 2D 图</a></p>
  </header>
  <div class="panels-2d">
    <div class="branch-col"><h3>RAW</h3><div class="thumb-list" id="panels-raw"></div></div>
    <div class="branch-col"><h3>SAM2</h3><div class="thumb-list" id="panels-sam2"></div></div>
    <div class="branch-col"><h3>SAM3</h3><div class="thumb-list" id="panels-sam3"></div></div>
  </div>
  <div id="lightbox" class="lightbox hidden">
    <div class="lightbox-backdrop"></div>
    <img id="lightbox-img" alt="放大查看">
  </div>
  <div class="viewers">
    <div class="panel"><h2>RAW</h2><div class="meta" id="meta-raw">加载中…</div><canvas id="view-raw"></canvas></div>
    <div class="panel"><h2>SAM2</h2><div class="meta" id="meta-sam2">加载中…</div><canvas id="view-sam2"></canvas></div>
    <div class="panel"><h2>SAM3</h2><div class="meta" id="meta-sam3">加载中…</div><canvas id="view-sam3"></canvas></div>
  </div>
  <pre class="stats" id="stats"></pre>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
  <script>
    const SAMPLE = {name_json};
    const REFRESH = new URLSearchParams(location.search).has("refresh");
    const lightbox = document.getElementById("lightbox");
    const lightboxImg = document.getElementById("lightbox-img");
    function openLightbox(src) {{
      lightboxImg.src = src;
      lightbox.classList.remove("hidden");
    }}
    lightbox.addEventListener("click", () => lightbox.classList.add("hidden"));
    document.addEventListener("keydown", (e) => {{
      if (e.key === "Escape") lightbox.classList.add("hidden");
    }});

    function panelUrl(branch, objectIndex) {{
      let url = "/api/sample/" + SAMPLE + "/panel/" + branch + "/" + objectIndex;
      if (REFRESH) url += "?refresh=1";
      return url;
    }}

    fetch("/api/sample/" + SAMPLE + "/panels").then(r => r.json()).then((data) => {{
      for (const branch of ["raw", "sam2", "sam3"]) {{
        const list = document.getElementById("panels-" + branch);
        const items = (data.branches && data.branches[branch]) || [];
        if (!items.length) {{
          list.innerHTML = '<div class="empty">无物体</div>';
          continue;
        }}
        for (const item of items) {{
          const img = document.createElement("img");
          img.loading = "lazy";
          img.decoding = "async";
          img.src = panelUrl(branch, item.object_index);
          img.alt = item.label || "";
          img.title = (item.label || "") + " — 点击放大";
          img.addEventListener("click", () => openLightbox(img.src));
          list.appendChild(img);
        }}
      }}
    }}).catch(err => {{
      for (const branch of ["raw", "sam2", "sam3"]) {{
        const list = document.getElementById("panels-" + branch);
        if (list) list.innerHTML = '<div class="empty">加载失败: ' + err + '</div>';
      }}
    }});

    function b64ToBytes(b64) {{
      const bin = atob(b64);
      const out = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
      return out;
    }}

    function mountViewer(canvasId, metaId, packed) {{
      const canvas = document.getElementById(canvasId);
      const meta = document.getElementById(metaId);
      const parent = canvas.parentElement;
      const w = Math.max(parent.clientWidth, 280);
      const h = Math.max(280, Math.floor(w * 0.75));
      const renderer = new THREE.WebGLRenderer({{ canvas, antialias: false, powerPreference: "high-performance" }});
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.25));
      renderer.setSize(w, h, false);
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x141414);
      const camera = new THREE.PerspectiveCamera(50, w / h, 0.001, 500);
      const controls = new THREE.OrbitControls(camera, canvas);
      controls.enableDamping = false;
      let visible = false;
      let framePending = false;
      const objects = (packed && packed.objects) ? packed.objects : [];
      if (!objects.length) {{
        meta.textContent = "无点云";
        controls.target.set(0, 0, -1);
        camera.position.set(0, 0, 1);
        controls.update();
        renderer.render(scene, camera);
        return;
      }}
      const extent = packed.extent || 1;
      const pointSize = Math.max(extent * 0.012, 0.004);
      let totalPts = 0;
      for (const obj of objects) {{
        const n = obj.n || 0;
        if (!n) continue;
        totalPts += n;
        const positions = new Float32Array(b64ToBytes(obj.positions).buffer);
        const rgb = obj.color || [200, 200, 200];
        const colorAttr = new Float32Array(n * 3);
        for (let i = 0; i < n; i++) {{
          colorAttr[i*3] = rgb[0]/255; colorAttr[i*3+1] = rgb[1]/255; colorAttr[i*3+2] = rgb[2]/255;
        }}
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute("color", new THREE.BufferAttribute(colorAttr, 3));
        scene.add(new THREE.Points(geometry, new THREE.PointsMaterial({{
          size: pointSize, vertexColors: true, sizeAttenuation: true,
        }})));
        if (obj.wireframe && obj.wireframe_n) {{
          const wf = new Float32Array(b64ToBytes(obj.wireframe).buffer);
          const wfGeom = new THREE.BufferGeometry();
          wfGeom.setAttribute("position", new THREE.BufferAttribute(wf, 3));
          const wfMat = new THREE.LineBasicMaterial({{
            color: new THREE.Color(rgb[0]/255, rgb[1]/255, rgb[2]/255),
            linewidth: 2,
          }});
          scene.add(new THREE.LineSegments(wfGeom, wfMat));
        }}
      }}
      const c = packed.centroid;
      // Server packs OpenCV camera points as Three.js (x, -y, -z); camera on +Z looks at -Z.
      controls.target.set(c[0], c[1], c[2]);
      camera.position.set(c[0], c[1], c[2] + extent * 1.2);
      controls.update();
      meta.textContent = objects.length + " objects · " + totalPts.toLocaleString() + " pts";
      function renderOnce() {{
        controls.update();
        renderer.render(scene, camera);
      }}
      function scheduleRender() {{
        if (!visible || framePending) return;
        framePending = true;
        requestAnimationFrame(() => {{
          framePending = false;
          if (visible) renderOnce();
        }});
      }}
      controls.addEventListener("change", scheduleRender);
      const io = new IntersectionObserver((entries) => {{
        visible = entries[0].isIntersecting;
        if (visible) scheduleRender();
      }}, {{ threshold: 0.08 }});
      io.observe(parent);
      renderOnce();
    }}

    async function loadBranch(branch, canvasId, metaId) {{
      const res = await fetch("/api/sample/" + SAMPLE + "/points/" + branch);
      const packed = await res.json();
      mountViewer(canvasId, metaId, packed);
    }}

    function load3DViewers() {{
      return Promise.all([
        loadBranch("raw", "view-raw", "meta-raw"),
        loadBranch("sam2", "view-sam2", "meta-sam2"),
        loadBranch("sam3", "view-sam3", "meta-sam3"),
      ]);
    }}

    fetch("/api/sample/" + SAMPLE + "/stats").then(r => r.json()).then((stats) => {{
      document.getElementById("stats").textContent = JSON.stringify(stats, null, 2);
    }}).catch(err => {{
      document.getElementById("stats").textContent = "加载失败: " + err;
    }});

    const viewersSection = document.querySelector(".viewers");
    const io3d = new IntersectionObserver((entries) => {{
      if (!entries[0].isIntersecting) return;
      io3d.disconnect();
      load3DViewers().catch(err => {{
        for (const id of ["meta-raw", "meta-sam2", "meta-sam3"]) {{
          const el = document.getElementById(id);
          if (el) el.textContent = "点云加载失败: " + err;
        }}
      }});
    }}, {{ rootMargin: "120px", threshold: 0.01 }});
    io3d.observe(viewersSection);
  </script>
</body>
</html>
"""


class CompareServerState:
    def __init__(
        self,
        index: CompareIndex,
        *,
        cache_dir: Path,
        max_points_per_object: int,
        max_panel_width: int,
        memory_cache_mb: int = 4096,
    ):
        self.index = index
        self.cache_dir = cache_dir
        self.max_points_per_object = max_points_per_object
        self.max_panel_width = max_panel_width
        self._mem_limit = max(memory_cache_mb, 256) * 1024 * 1024
        self._mem_used = 0
        self._panel_mem: Dict[str, bytes] = {}
        self._points_mem: Dict[str, bytes] = {}
        self._mem_lock = threading.RLock()
        self._panel_locks: Dict[str, threading.Lock] = {}
        self._panel_locks_guard = threading.Lock()
        self.preload_total = 0
        self.preload_done = 0
        self.preload_phase = "idle"
        self._preload_progress_lock = threading.Lock()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def preload_status(self) -> Dict[str, Any]:
        with self._preload_progress_lock:
            total = self.preload_total
            done = self.preload_done
            phase = self.preload_phase
        pct = round(100.0 * done / total, 1) if total else 100.0
        return {
            "phase": phase,
            "done": done,
            "total": total,
            "percent": pct,
            "memory_cache_mb": round(self._mem_used / (1024 * 1024), 1),
        }

    def _preload_tick(self, n: int = 1) -> None:
        with self._preload_progress_lock:
            self.preload_done += n

    def _panel_lock(self, key: str) -> threading.Lock:
        with self._panel_locks_guard:
            if key not in self._panel_locks:
                self._panel_locks[key] = threading.Lock()
            return self._panel_locks[key]

    def _mem_store(self, store: Dict[str, bytes], key: str, data: bytes) -> None:
        with self._mem_lock:
            old = store.get(key)
            if old is not None:
                self._mem_used -= len(old)
            while store and self._mem_used + len(data) > self._mem_limit:
                evict_key = next(iter(store))
                evicted = store.pop(evict_key)
                self._mem_used -= len(evicted)
            store[key] = data
            self._mem_used += len(data)

    def panel_bytes(
        self,
        name: str,
        branch: str,
        object_index: int,
        *,
        refresh: bool,
    ) -> Optional[bytes]:
        image_key = self.index.key_by_name.get(name)
        if image_key is None:
            return None
        mem_key = f"{PANEL_RENDER_VERSION}/{name}/{branch}_{object_index}"
        if not refresh:
            with self._mem_lock:
                cached = self._panel_mem.get(mem_key)
            if cached is not None:
                return cached
        sample_cache = self.cache_dir / name
        sample_cache.mkdir(parents=True, exist_ok=True)
        disk_path = sample_cache / f"{PANEL_RENDER_VERSION}_{branch}_{object_index}.jpg"
        if not refresh and disk_path.is_file():
            data = disk_path.read_bytes()
            self._mem_store(self._panel_mem, mem_key, data)
            return data
        with self._panel_lock(mem_key):
            if not refresh:
                with self._mem_lock:
                    cached = self._panel_mem.get(mem_key)
                if cached is not None:
                    return cached
                if disk_path.is_file():
                    data = disk_path.read_bytes()
                    self._mem_store(self._panel_mem, mem_key, data)
                    return data
            panel = render_object_panel_image(
                self.index,
                image_key,
                branch,
                object_index,
                max_panel_width=self.max_panel_width,
            )
            if panel is None:
                return None
            buf = io.BytesIO()
            panel.save(buf, format="JPEG", quality=90, optimize=True)
            data = buf.getvalue()
            disk_path.write_bytes(data)
            self._mem_store(self._panel_mem, mem_key, data)
            return data

    def points_bytes(self, image_key: str, branch: str, *, refresh: bool = False) -> bytes:
        mem_key = f"{PANEL_RENDER_VERSION}/pts:{image_key}|{branch}"
        if not refresh:
            with self._mem_lock:
                cached = self._points_mem.get(mem_key)
            if cached is not None:
                return cached
        branch_records = branch_records_for_image(self.index, image_key)
        packed = _pack_branch_from_fusion(
            branch_records[branch],
            self.max_points_per_object,
        )
        data = json.dumps(packed, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._mem_store(self._points_mem, mem_key, data)
        return data


def _jobs_for_sample(
    index: CompareIndex,
    image_key: str,
    *,
    preload_panels: bool,
    preload_points: bool,
) -> Tuple[List[Tuple[str, str, int]], List[Tuple[str, str]]]:
    name = index.name_by_key[image_key]
    panel_jobs: List[Tuple[str, str, int]] = []
    point_jobs: List[Tuple[str, str]] = []
    if preload_panels:
        manifest = sample_panels_json(index, image_key)
        for branch in ("raw", "sam2", "sam3"):
            for item in manifest["branches"].get(branch, []):
                panel_jobs.append((name, branch, int(item["object_index"])))
    if preload_points:
        for branch in ("raw", "sam2", "sam3"):
            point_jobs.append((image_key, branch))
    return panel_jobs, point_jobs


def _run_preload_jobs(
    state: CompareServerState,
    panel_jobs: List[Tuple[str, str, int]],
    point_jobs: List[Tuple[str, str]],
    *,
    workers: int,
    quiet: bool,
) -> None:
    total = len(panel_jobs) + len(point_jobs)
    if total == 0:
        return
    done_local = 0
    lock = threading.Lock()

    def _tick() -> None:
        nonlocal done_local
        state._preload_tick()
        if quiet:
            return
        with lock:
            done_local += 1
            if done_local % 50 == 0 or done_local == total:
                _log(f"  preload {done_local}/{total}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(state.panel_bytes, name, branch, obj_idx, refresh=False)
            for name, branch, obj_idx in panel_jobs
        ]
        futures.extend(
            pool.submit(state.points_bytes, image_key, branch, refresh=False)
            for image_key, branch in point_jobs
        )
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as exc:
                if not quiet:
                    _log(f"  preload warning: {exc}")
            _tick()


def _start_background_preload(
    state: CompareServerState,
    panel_jobs: List[Tuple[str, str, int]],
    point_jobs: List[Tuple[str, str]],
    *,
    workers: int,
) -> None:
    total = len(panel_jobs) + len(point_jobs)
    if total == 0:
        with state._preload_progress_lock:
            if state.preload_phase == "blocking":
                state.preload_phase = "done"
        return

    def _worker() -> None:
        with state._preload_progress_lock:
            state.preload_phase = "background"
        try:
            _run_preload_jobs(
                state, panel_jobs, point_jobs, workers=workers, quiet=True,
            )
        except Exception as exc:
            _log(f"Background preload error: {exc}")
        finally:
            with state._preload_progress_lock:
                state.preload_phase = "done"
            with state._mem_lock:
                mem_mb = state._mem_used / (1024 * 1024)
            _log(
                f"Background preload finished. "
                f"Cache ~{mem_mb:.0f} MB / {state._mem_limit // (1024 * 1024)} MB limit."
            )

    threading.Thread(target=_worker, name="preload-background", daemon=True).start()


def _schedule_preload(
    state: CompareServerState,
    *,
    workers: int,
    preload_panels: bool,
    preload_points: bool,
    first_samples: int,
) -> None:
    """Preload first N samples before serving; rest continues in background."""
    blocking_panel: List[Tuple[str, str, int]] = []
    blocking_point: List[Tuple[str, str]] = []
    bg_panel: List[Tuple[str, str, int]] = []
    bg_point: List[Tuple[str, str]] = []

    for i, image_key in enumerate(state.index.image_keys):
        p_jobs, pt_jobs = _jobs_for_sample(
            state.index, image_key,
            preload_panels=preload_panels, preload_points=preload_points,
        )
        if first_samples <= 0 or i < first_samples:
            blocking_panel.extend(p_jobs)
            blocking_point.extend(pt_jobs)
        else:
            bg_panel.extend(p_jobs)
            bg_point.extend(pt_jobs)

    total = (
        len(blocking_panel) + len(blocking_point)
        + len(bg_panel) + len(bg_point)
    )
    if total == 0:
        return

    with state._preload_progress_lock:
        state.preload_total = total
        state.preload_done = 0
        state.preload_phase = "blocking"

    n_block = len(blocking_panel) + len(blocking_point)
    n_bg = len(bg_panel) + len(bg_point)
    _log(
        f"Preload plan: {n_block} job(s) before server opens, "
        f"{n_bg} job(s) in background ({workers} workers)."
    )

    if n_block:
        _run_preload_jobs(
            state, blocking_panel, blocking_point, workers=workers, quiet=False,
        )

    if n_bg:
        _log(f"Server ready — background preload continues ({n_bg} jobs remaining).")
        _start_background_preload(state, bg_panel, bg_point, workers=workers)
    else:
        with state._preload_progress_lock:
            state.preload_phase = "done"
        with state._mem_lock:
            mem_mb = state._mem_used / (1024 * 1024)
        _log(f"Preload done. Memory cache ~{mem_mb:.0f} MB.")


def make_handler(state: CompareServerState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            _log(f"[{self.address_string()}] {fmt % args}")

        def _send_bytes(
            self,
            data: bytes,
            content_type: str,
            *,
            status: int = 200,
            cacheable: bool = False,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            if cacheable:
                self.send_header("Cache-Control", "public, max-age=86400")
            self.end_headers()
            self.wfile.write(data)

        def _send_json(self, obj: Any, *, status: int = 200) -> None:
            data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self._send_bytes(data, "application/json; charset=utf-8", status=status)

        def _send_html(self, text: str, *, status: int = 200) -> None:
            self._send_bytes(text.encode("utf-8"), "text/html; charset=utf-8", status=status)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            qs = parse_qs(parsed.query)
            refresh = "refresh" in qs

            if path == "/":
                rows = "\n".join(
                    f'    <li><a href="/view/{state.index.name_by_key[k]}">'
                    f'{html_module.escape(k)}</a></li>'
                    for k in state.index.image_keys
                )
                self._send_html(_INDEX_HTML.format(count=len(state.index.image_keys), rows=rows))
                return

            if path.startswith("/view/"):
                name = path[len("/view/"):]
                image_key = state.index.key_by_name.get(name)
                if image_key is None:
                    self._send_html("样本不存在", status=404)
                    return
                title = html_module.escape(image_key, quote=True)
                query_suffix = "?refresh=1" if refresh else ""
                nav_prev, nav_pos, nav_next = _viewer_nav_html(
                    state.index, image_key, query_suffix=query_suffix,
                )
                body = _VIEWER_HTML.format(
                    title=title,
                    name=name,
                    name_json=json.dumps(name),
                    nav_prev=nav_prev,
                    nav_pos=nav_pos,
                    nav_next=nav_next,
                )
                self._send_html(body)
                return

            if path == "/api/samples":
                self._send_json([
                    {"name": state.index.name_by_key[k], "image_key": k}
                    for k in state.index.image_keys
                ])
                return

            if path == "/api/preload/status":
                self._send_json(state.preload_status())
                return

            parts = path.strip("/").split("/")
            # /api/sample/{name}/panels | panel/{branch}/{idx} | stats | points/{branch}
            if len(parts) >= 3 and parts[0] == "api" and parts[1] == "sample":
                name = parts[2]
                image_key = state.index.key_by_name.get(name)
                if image_key is None:
                    self._send_json({"error": "not found"}, status=404)
                    return
                if len(parts) == 4 and parts[3] == "panels":
                    self._send_json(sample_panels_json(state.index, image_key))
                    return
                if (
                    len(parts) == 6
                    and parts[3] == "panel"
                    and parts[4] in ("raw", "sam2", "sam3")
                    and parts[5].isdigit()
                ):
                    branch = parts[4]
                    obj_idx = int(parts[5])
                    data = state.panel_bytes(name, branch, obj_idx, refresh=refresh)
                    if data is None:
                        self.send_error(HTTPStatus.NOT_FOUND)
                        return
                    self._send_bytes(data, "image/jpeg", cacheable=not refresh)
                    return
                if len(parts) == 4 and parts[3] == "stats":
                    self._send_json(sample_stats_json(state.index, image_key))
                    return
                if len(parts) == 5 and parts[3] == "points" and parts[4] in ("raw", "sam2", "sam3"):
                    branch = parts[4]
                    data = state.points_bytes(image_key, branch, refresh=refresh)
                    self._send_bytes(data, "application/json; charset=utf-8", cacheable=not refresh)
                    return

            self.send_error(HTTPStatus.NOT_FOUND)

    return Handler


def _lan_urls(port: int) -> List[str]:
    seen: set[str] = set()
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
    return [f"http://{ip}:{port}/" for ip in sorted(seen)]


def _print_listen_urls(host: str, port: int, *, sample_count: int) -> None:
    _log(f"Serving {sample_count} sample(s)")
    _log(f"  Local:   http://127.0.0.1:{port}/")
    _log(f"  Local:   http://localhost:{port}/")
    if host in ("0.0.0.0", "::"):
        lan = _lan_urls(port)
        if lan:
            for url in lan:
                _log(f"  Network: {url}")
        else:
            _log(f"  Network: http://<this-machine-ip>:{port}/")
    else:
        _log(f"  Bind:    http://{host}:{port}/")
    _log("Press Ctrl+C to stop.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind address (default 0.0.0.0 = all interfaces)",
    )
    parser.add_argument("--port", type=int, default=8848)
    parser.add_argument("--raw-run", default="refiner_exp/outputs/raw")
    parser.add_argument("--sam2-run", default="refiner_exp/outputs/sam2")
    parser.add_argument("--sam3-run", default="refiner_exp/outputs/sam3")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--max-panel-width", type=int, default=640)
    parser.add_argument(
        "--max-points-per-object",
        type=int,
        default=8000,
        help="Subsample each fusion .pcd when sending to the browser.",
    )
    parser.add_argument(
        "--cache-dir",
        default="refiner_exp/outputs/compare/cache",
        help="JPEG panel disk cache directory.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=48,
        help="Parallel workers for startup preload (default 48).",
    )
    parser.add_argument(
        "--memory-cache-mb",
        type=int,
        default=4096,
        help="In-memory cache budget for panels and point payloads (default 4096 MB).",
    )
    parser.add_argument(
        "--preload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable preload (default on). Use --no-preload to load only on demand.",
    )
    parser.add_argument(
        "--preload-first-samples",
        type=int,
        default=5,
        help="Fully preload this many samples before opening the server; 0 = open immediately.",
    )
    parser.add_argument(
        "--preload-points",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include point-cloud JSON in preload (default on).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index = build_compare_index(
        args.raw_run, args.sam2_run, args.sam3_run, max_images=args.max_images,
    )
    state = CompareServerState(
        index,
        cache_dir=Path(args.cache_dir),
        max_points_per_object=args.max_points_per_object,
        max_panel_width=args.max_panel_width,
        memory_cache_mb=args.memory_cache_mb,
    )
    if args.preload:
        _schedule_preload(
            state,
            workers=max(1, args.workers),
            preload_panels=True,
            preload_points=args.preload_points,
            first_samples=max(0, args.preload_first_samples),
        )
    handler = make_handler(state)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    server.daemon_threads = True
    server.request_queue_size = max(64, args.workers)
    _print_listen_urls(args.host, args.port, sample_count=len(index.image_keys))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _log("Shutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
