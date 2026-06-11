#!/usr/bin/env python3
"""Create side-by-side Raw/SAM2/SAM3 refiner comparison images."""

from __future__ import annotations

import argparse
import base64
import html as html_module
import io
import json
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import tqdm
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
from utils.image_utils import load_depth_map  # noqa: E402


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
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


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
    try:
        box = draw.textbbox(xy, text, font=font)
    except AttributeError:
        box = (xy[0], xy[1], xy[0] + len(text) * 8, xy[1] + 18)
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


def _draw_branch_panel(
    base_image: Image.Image,
    branch: str,
    records: List[Dict[str, Any]],
    max_width: int,
) -> Image.Image:
    canvas = base_image.convert("RGBA")
    draw = ImageDraw.Draw(canvas)
    for rec in records:
        color = _color_for_key(rec["object_key"])
        mask = _load_mask_cached(str(rec.get("mask_path") or ""))
        if mask is not None:
            if mask.shape[::-1] != canvas.size:
                mask_img = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
                mask_img = mask_img.resize(canvas.size, resample=Image.NEAREST)
                mask = np.array(mask_img) > 0
            overlay = Image.new("RGBA", canvas.size, (*color, 0))
            alpha = Image.fromarray((mask.astype(np.uint8) * 120), mode="L")
            fill = Image.new("RGBA", canvas.size, (*color, 120))
            overlay.paste(fill, mask=alpha)
            canvas = Image.alpha_composite(canvas, overlay)
            draw = ImageDraw.Draw(canvas)
            bbox = _mask_bbox(mask)
            if bbox:
                draw.rectangle(bbox, outline=color, width=4)
                _draw_label(draw, (bbox[0], max(0, bbox[1] - 20)), f"{rec['object_index']}:{rec['tag']}", color)

        # Project the original object-level 3D bbox onto RGB.  Use the same
        # object color as the mask so bbox quality can be checked visually.
        for p1, p2 in _project_box_edges(rec):
            draw.line([p1, p2], fill=(*color, 230), width=3)

    title = f"{branch.upper()} objects={len(records)}"
    _draw_label(draw, (8, 8), title, (30, 30, 30))
    panel = canvas.convert("RGB")
    if panel.width > max_width:
        scale = max_width / float(panel.width)
        panel = panel.resize((max_width, int(panel.height * scale)), resample=Image.BILINEAR)
    return panel


def _resize_rgb_to_depth(image: Image.Image, depth_shape: Tuple[int, int]) -> np.ndarray:
    h, w = depth_shape
    if image.size != (w, h):
        image = image.resize((w, h), resample=Image.BILINEAR)
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _resize_mask_to_depth(mask: np.ndarray, depth_shape: Tuple[int, int]) -> np.ndarray:
    if mask.shape == depth_shape:
        return mask > 0
    h, w = depth_shape
    img = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
    return np.asarray(img.resize((w, h), resample=Image.NEAREST)) > 0


def _branch_colored_points(
    image: Image.Image,
    records: List[Dict[str, Any]],
    max_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    seed = next((rec for rec in records if rec.get("depth_map") and rec.get("intrinsic")), None)
    if seed is None:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    try:
        depth = load_depth_map(seed["depth_map"], seed.get("depth_scale"))
    except Exception:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    intrinsic = _load_matrix(seed.get("intrinsic"))
    if intrinsic is None or depth.size == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    masks = []
    for rec in records:
        mask = _load_mask_cached(str(rec.get("mask_path") or ""))
        if mask is not None:
            masks.append(_resize_mask_to_depth(mask, depth.shape))
    if not masks:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    union_mask = np.logical_or.reduce(masks)
    valid = union_mask & np.isfinite(depth) & (depth > 0)
    ys, xs = np.where(valid)
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    if max_points and len(xs) > max_points:
        rng = np.random.default_rng(0)
        keep = rng.choice(len(xs), size=max_points, replace=False)
        ys, xs = ys[keep], xs[keep]

    z = depth[ys, xs].astype(np.float64)
    k = intrinsic[:3, :3]
    x = (xs.astype(np.float64) - k[0, 2]) * z / k[0, 0]
    y = (ys.astype(np.float64) - k[1, 2]) * z / k[1, 1]
    points = np.stack([x, y, z], axis=1).astype(np.float32)
    rgb = _resize_rgb_to_depth(image, depth.shape)
    colors = rgb[ys, xs]
    return points, colors


def _pack_pointcloud_binary(
    points: np.ndarray,
    colors: np.ndarray,
    max_points: int,
) -> Dict[str, Any]:
    """Pack positions/colors as base64 for embedding in interactive HTML."""
    if len(points) == 0:
        return {"n": 0, "positions": "", "colors": "", "centroid": [0.0, 0.0, 1.0], "extent": 1.0}
    if max_points and len(points) > max_points:
        rng = np.random.default_rng(0)
        keep = rng.choice(len(points), size=max_points, replace=False)
        points = points[keep]
        colors = colors[keep]
    centroid = points.mean(axis=0).astype(np.float32)
    extent = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    extent = max(extent, 1e-3)
    return {
        "n": int(len(points)),
        "positions": base64.b64encode(points.astype(np.float32).tobytes()).decode("ascii"),
        "colors": base64.b64encode(colors.astype(np.uint8).tobytes()).decode("ascii"),
        "centroid": centroid.tolist(),
        "extent": extent,
    }


def _write_interactive_html(
    path: Path,
    *,
    image_key: str,
    overlay_filename: str,
    branch_clouds: Dict[str, Dict[str, Any]],
) -> None:
    """Write a self-contained HTML page with drag-to-rotate point cloud viewers."""
    safe_key = html_module.escape(image_key, quote=True)
    payload = json.dumps(
        {
            "image_key": image_key,
            "overlay": overlay_filename,
            "branches": branch_clouds,
        },
        ensure_ascii=False,
    ).replace("</", "<\\/")
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_key} — Refiner Compare</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
      background: #1a1a1a;
      color: #eee;
    }}
    header {{
      padding: 12px 16px;
      background: #242424;
      border-bottom: 1px solid #333;
    }}
    header h1 {{ margin: 0 0 6px; font-size: 1.05rem; font-weight: 600; }}
    header p {{ margin: 0; font-size: 0.85rem; color: #aaa; }}
    .overlay-wrap {{
      padding: 12px 16px;
      background: #111;
      text-align: center;
    }}
    .overlay-wrap img {{
      max-width: 100%;
      height: auto;
      border: 1px solid #333;
      border-radius: 4px;
    }}
    .viewers {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      padding: 12px 16px 20px;
    }}
    .panel {{
      background: #242424;
      border: 1px solid #333;
      border-radius: 6px;
      overflow: hidden;
      min-height: 320px;
      display: flex;
      flex-direction: column;
    }}
    .panel h2 {{
      margin: 0;
      padding: 8px 10px;
      font-size: 0.9rem;
      background: #2e2e2e;
      border-bottom: 1px solid #333;
    }}
    .panel .meta {{
      padding: 6px 10px;
      font-size: 0.78rem;
      color: #aaa;
    }}
    .panel canvas {{
      flex: 1;
      width: 100%;
      min-height: 280px;
      display: block;
      cursor: grab;
    }}
    .panel canvas:active {{ cursor: grabbing; }}
    .hint {{
      padding: 0 16px 16px;
      font-size: 0.8rem;
      color: #888;
    }}
    a.back {{ color: #7eb8ff; text-decoration: none; }}
    a.back:hover {{ text-decoration: underline; }}
    @media (max-width: 960px) {{
      .viewers {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{safe_key}</h1>
    <p>拖拽旋转 · 滚轮缩放 · 右键平移（RAW / SAM2 / SAM3 点云并列）</p>
    <p><a class="back" href="index.html">← 返回样本列表</a></p>
  </header>
  <div class="overlay-wrap">
    <img src="{overlay_filename}" alt="2D overlay comparison">
  </div>
  <div class="viewers">
    <div class="panel">
      <h2>RAW</h2>
      <div class="meta" id="meta-raw"></div>
      <canvas id="view-raw"></canvas>
    </div>
    <div class="panel">
      <h2>SAM2</h2>
      <div class="meta" id="meta-sam2"></div>
      <canvas id="view-sam2"></canvas>
    </div>
    <div class="panel">
      <h2>SAM3</h2>
      <div class="meta" id="meta-sam3"></div>
      <canvas id="view-sam3"></canvas>
    </div>
  </div>
  <p class="hint">点云为相机坐标系 RGB-D 反投影（mask 并集着色）。交互页为便于加载做了下采样；完整点云见同目录 <code>pointclouds/*.ply</code>。</p>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
  <script>
    const PAGE = {payload};

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

      const renderer = new THREE.WebGLRenderer({{ canvas, antialias: true }});
      renderer.setPixelRatio(window.devicePixelRatio || 1);
      renderer.setSize(w, h, false);

      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x141414);

      const camera = new THREE.PerspectiveCamera(50, w / h, 0.001, 500);
      const controls = new THREE.OrbitControls(camera, canvas);
      controls.enableDamping = true;
      controls.dampingFactor = 0.08;

      if (!packed || !packed.n) {{
        meta.textContent = "无点云";
        camera.position.set(0, 0, 2);
        controls.target.set(0, 0, 1);
        controls.update();
        renderer.render(scene, camera);
        return {{ renderer, scene, camera, controls }};
      }}

      const positions = new Float32Array(b64ToBytes(packed.positions).buffer);
      const colorsRaw = b64ToBytes(packed.colors);
      const n = packed.n;
      const colorAttr = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) {{
        colorAttr[i * 3] = colorsRaw[i * 3] / 255;
        colorAttr[i * 3 + 1] = colorsRaw[i * 3 + 1] / 255;
        colorAttr[i * 3 + 2] = colorsRaw[i * 3 + 2] / 255;
      }}

      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      geometry.setAttribute("color", new THREE.BufferAttribute(colorAttr, 3));

      const extent = packed.extent || 1;
      const material = new THREE.PointsMaterial({{
        size: Math.max(extent * 0.012, 0.004),
        vertexColors: true,
        sizeAttenuation: true,
      }});
      scene.add(new THREE.Points(geometry, material));

      const c = packed.centroid;
      const target = new THREE.Vector3(c[0], c[1], c[2]);
      controls.target.copy(target);
      camera.position.set(c[0], c[1] - extent * 0.15, c[2] + extent * 1.4);
      controls.update();
      meta.textContent = `${{n.toLocaleString()}} points · extent ${{extent.toFixed(2)}}m`;

      function animate() {{
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
      }}
      animate();

      function onResize() {{
        const nw = Math.max(parent.clientWidth, 280);
        const nh = Math.max(280, Math.floor(nw * 0.75));
        camera.aspect = nw / nh;
        camera.updateProjectionMatrix();
        renderer.setSize(nw, nh, false);
      }}
      window.addEventListener("resize", onResize);
      return {{ renderer, scene, camera, controls }};
    }}

    mountViewer("view-raw", "meta-raw", PAGE.branches.raw);
    mountViewer("view-sam2", "meta-sam2", PAGE.branches.sam2);
    mountViewer("view-sam3", "meta-sam3", PAGE.branches.sam3);
  </script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def _write_gallery_index(path: Path, entries: List[Dict[str, str]]) -> None:
    rows = "\n".join(
        f'    <li><a href="{e["html"]}">{e["image_key"]}</a> '
        f'(<a href="{e["jpg"]}">jpg</a>)</li>'
        for e in entries
    )
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Refiner Compare Gallery</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; background: #fafafa; }}
    h1 {{ font-size: 1.25rem; }}
    ul {{ line-height: 1.8; }}
    a {{ color: #1a56db; }}
  </style>
</head>
<body>
  <h1>Refiner 对照实验 — 交互可视化</h1>
  <p>共 {len(entries)} 个样本。打开链接可在页面内拖拽旋转点云。</p>
  <ul>
{rows}
  </ul>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Write binary little-endian PLY (much faster than ASCII for large clouds)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(points)
    pts = np.asarray(points, dtype=np.float32)
    cols = np.asarray(colors, dtype=np.uint8)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        row = np.empty((n, 6), dtype=np.dtype([
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ]))
        row["x"] = pts[:, 0]
        row["y"] = pts[:, 1]
        row["z"] = pts[:, 2]
        row["red"] = cols[:, 0]
        row["green"] = cols[:, 1]
        row["blue"] = cols[:, 2]
        row.tofile(f)


def _render_pointcloud_views(
    points: np.ndarray,
    colors: np.ndarray,
    title: str,
    size: Tuple[int, int] = (420, 220),
) -> Image.Image:
    width, height = size
    panel = Image.new("RGB", size, (20, 20, 20))
    draw = ImageDraw.Draw(panel)
    _draw_label(draw, (8, 8), title, (50, 50, 50))
    if len(points) == 0:
        draw.text((12, 42), "no masked depth points", fill=(230, 230, 230), font=_font(15))
        return panel

    front = _render_scatter(points[:, [0, 1]], colors, (width // 2 - 12, height - 36), invert_y=True)
    top = _render_scatter(points[:, [0, 2]], colors, (width // 2 - 12, height - 36), invert_y=True)
    panel.paste(front, (8, 30))
    panel.paste(top, (width // 2 + 4, 30))
    draw = ImageDraw.Draw(panel)
    draw.text((12, height - 18), "front: x/y", fill=(230, 230, 230), font=_font(13))
    draw.text((width // 2 + 8, height - 18), "top: x/z", fill=(230, 230, 230), font=_font(13))
    return panel


def _render_scatter(
    coords: np.ndarray,
    colors: np.ndarray,
    size: Tuple[int, int],
    invert_y: bool,
) -> Image.Image:
    width, height = size
    out = Image.new("RGB", size, (245, 245, 245))
    if len(coords) == 0:
        return out
    xy = coords.astype(np.float64)
    finite = np.isfinite(xy).all(axis=1)
    xy = xy[finite]
    colors = colors[finite]
    if len(xy) == 0:
        return out
    mins = xy.min(axis=0)
    maxs = xy.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    norm = (xy - mins) / span
    px = np.clip((norm[:, 0] * (width - 1)).astype(np.int32), 0, width - 1)
    py_float = norm[:, 1] * (height - 1)
    if invert_y:
        py_float = (height - 1) - py_float
    py = np.clip(py_float.astype(np.int32), 0, height - 1)
    canvas = np.asarray(out)
    # Draw far-to-near by second coordinate for stable dense scatter.
    order = np.argsort(xy[:, 1])
    canvas[py[order], px[order]] = colors[order]
    return Image.fromarray(canvas, mode="RGB")


def _stats_panel(
    image_key: str,
    branch_records: Dict[str, List[Dict[str, Any]]],
    fusion_records: Dict[str, Dict[str, Dict[str, Any]]],
    height: int,
) -> Image.Image:
    panel = Image.new("RGB", (440, height), (245, 245, 245))
    draw = ImageDraw.Draw(panel)
    font = _font(15)
    y = 10
    draw.text((10, y), image_key, fill=(20, 20, 20), font=_font(16))
    y += 30
    for branch in ("raw", "sam2", "sam3"):
        records = branch_records.get(branch, [])
        draw.text((10, y), f"{branch.upper()} refine objects: {len(records)}", fill=(0, 0, 0), font=font)
        y += 22
        for rec in records[:8]:
            color = _color_for_key(rec["object_key"])
            fusion = fusion_records.get(branch, {}).get(rec["object_key"], {})
            pcd_path = fusion.get("pointcloud_path") if fusion else None
            point_count = _pcd_vertex_count(pcd_path)
            inside = fusion.get("pointcloud_inside_box_ratio") if fusion else None
            mask_iou = rec.get("mask_iou_with_raw")
            text = (
                f"{rec['object_index']} {rec['tag']}: "
                f"area={rec.get('mask_area')} "
                f"IoUraw={_fmt(mask_iou)} "
                f"pts={point_count if point_count is not None else 'n/a'} "
                f"in3d={_fmt(inside)}"
            )
            draw.rectangle((10, y + 4, 20, y + 14), fill=color)
            draw.text((26, y), text[:54], fill=(20, 20, 20), font=font)
            y += 20
        y += 10
    return panel


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.2f}"
    except Exception:
        return str(value)


def _hstack(panels: List[Image.Image], gap: int = 8) -> Image.Image:
    height = max(p.height for p in panels)
    width = sum(p.width for p in panels) + gap * (len(panels) - 1)
    out = Image.new("RGB", (width, height), (35, 35, 35))
    x = 0
    for panel in panels:
        out.paste(panel, (x, 0))
        x += panel.width + gap
    return out


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-run", default="refiner_exp/outputs/raw")
    parser.add_argument("--sam2-run", default="refiner_exp/outputs/sam2")
    parser.add_argument("--sam3-run", default="refiner_exp/outputs/sam3")
    parser.add_argument("--output-dir", default="refiner_exp/outputs/compare/images")
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--max-panel-width", type=int, default=640)
    parser.add_argument(
        "--pointcloud-mode",
        choices=("none", "ply", "render", "interactive", "both", "all"),
        default="interactive",
        help=(
            "Point cloud outputs: ply file, static orthographic thumbnails in JPG, "
            "interactive HTML (drag-rotate), or combinations (all=everything)."
        ),
    )
    parser.add_argument(
        "--max-pointcloud-points",
        type=int,
        default=120000,
        help="Max points for PLY export and static orthographic panels.",
    )
    parser.add_argument(
        "--max-interactive-points",
        type=int,
        default=20000,
        help="Max points embedded per branch in interactive HTML (smaller=faster).",
    )
    return parser.parse_args()


def _pc_mode_uses_ply(mode: str) -> bool:
    return mode in ("ply", "both", "all")


def _pc_mode_uses_render(mode: str) -> bool:
    return mode in ("render", "both", "all")


def _pc_mode_uses_interactive(mode: str) -> bool:
    return mode in ("interactive", "both", "all")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _log("Resolving run directories...")
    run_roots = {
        "raw": resolve_run_root(args.raw_run, BRANCHES["raw"]["config_stem"]),
        "sam2": resolve_run_root(args.sam2_run, BRANCHES["sam2"]["config_stem"]),
        "sam3": resolve_run_root(args.sam3_run, BRANCHES["sam3"]["config_stem"]),
    }

    _log("Loading parquet metadata (light mode, no bulk mask/pcd IO)...")
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
    if args.max_images:
        image_keys = image_keys[: args.max_images]
    _log(f"Rendering {len(image_keys)} image(s), pointcloud_mode={args.pointcloud_mode}")

    written = 0
    gallery_entries: List[Dict[str, str]] = []
    for image_key in tqdm.tqdm(image_keys, desc="visualize"):
        branch_records = {branch: by_image.get(branch, {}).get(image_key, []) for branch in ("raw", "sam2", "sam3")}
        seed_record = next((records[0] for records in branch_records.values() if records), None)
        if seed_record is None:
            continue
        image = _load_image(seed_record.get("image"))
        if image is None:
            continue

        name = _sanitize_name(image_key)
        _enrich_image_records_for_stats(branch_records, raw_by_key)
        panels = [
            _draw_branch_panel(image, branch, branch_records[branch], args.max_panel_width)
            for branch in ("raw", "sam2", "sam3")
        ]
        pointcloud_payload: Dict[str, Any] = {}
        branch_clouds: Dict[str, Dict[str, Any]] = {}
        if args.pointcloud_mode != "none":
            pc_dir = output_dir / "pointclouds"
            pc_panels = []
            pc_cap = args.max_pointcloud_points
            if not _pc_mode_uses_ply(args.pointcloud_mode) and not _pc_mode_uses_render(args.pointcloud_mode):
                pc_cap = min(pc_cap, args.max_interactive_points)
            for branch in ("raw", "sam2", "sam3"):
                points, colors = _branch_colored_points(
                    image,
                    branch_records[branch],
                    max_points=pc_cap,
                )
                entry: Dict[str, Any] = {"num_points": int(len(points))}
                if _pc_mode_uses_ply(args.pointcloud_mode) and len(points):
                    ply_path = pc_dir / f"{name}_{branch}.ply"
                    _write_ply(ply_path, points, colors)
                    entry["ply_path"] = str(ply_path)
                if _pc_mode_uses_render(args.pointcloud_mode):
                    pc_panels.append(_render_pointcloud_views(points, colors, f"{branch.upper()} RGB-D point cloud"))
                if _pc_mode_uses_interactive(args.pointcloud_mode):
                    branch_clouds[branch] = _pack_pointcloud_binary(
                        points, colors, args.max_interactive_points
                    )
                pointcloud_payload[branch] = entry
            if pc_panels:
                panels = [
                    _hstack([panel, pc_panel.resize((panel.width, max(1, int(pc_panel.height * panel.width / pc_panel.width))), resample=Image.BILINEAR)])
                    for panel, pc_panel in zip(panels, pc_panels)
                ]
        stats = _stats_panel(image_key, branch_records, fusion_by_key, max(p.height for p in panels))
        combined = _hstack(panels + [stats])
        jpg_name = f"{name}.jpg"
        combined.save(output_dir / jpg_name, quality=92)

        if _pc_mode_uses_interactive(args.pointcloud_mode):
            html_name = f"{name}.html"
            _write_interactive_html(
                output_dir / html_name,
                image_key=image_key,
                overlay_filename=jpg_name,
                branch_clouds=branch_clouds,
            )
            gallery_entries.append({
                "image_key": image_key,
                "html": html_name,
                "jpg": jpg_name,
            })

        slim_payload = {
            "image_key": image_key,
            "image": seed_record.get("image"),
            "pointclouds": pointcloud_payload,
            "branches": {
                branch: [
                    {
                        "object_key": rec["object_key"],
                        "tag": rec.get("tag"),
                        "mask_area": rec.get("mask_area"),
                        "mask_iou_with_raw": rec.get("mask_iou_with_raw"),
                        "point_count": _pcd_vertex_count(
                            (fusion_by_key.get(branch, {}).get(rec["object_key"]) or {}).get("pointcloud_path")
                        ),
                    }
                    for rec in branch_records[branch]
                ]
                for branch in ("raw", "sam2", "sam3")
            },
        }
        (output_dir / f"{name}.json").write_text(
            json.dumps(slim_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        written += 1

    if gallery_entries:
        _write_gallery_index(output_dir / "index.html", gallery_entries)

    msg = f"Wrote {written} comparison image(s) to {output_dir}"
    if gallery_entries:
        msg += f" (+ {len(gallery_entries)} interactive HTML, open {output_dir / 'index.html'})"
    print(msg)


if __name__ == "__main__":
    main()
