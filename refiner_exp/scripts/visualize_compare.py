#!/usr/bin/env python3
"""Create side-by-side Raw/SAM2/SAM3 refiner comparison images."""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
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
    collect_records,
    compute_box_3d_corners_from_params,
    enrich_against_raw,
    load_stage_df,
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


def _load_image(image_ref: Any) -> Optional[Image.Image]:
    try:
        if isinstance(image_ref, dict) and image_ref.get("bytes"):
            return Image.open(io.BytesIO(image_ref["bytes"])).convert("RGB")
        if isinstance(image_ref, str) and image_ref:
            return Image.open(image_ref).convert("RGB")
    except Exception:
        return None
    return None


def _mask_bbox(mask: Optional[np.ndarray]) -> Optional[List[int]]:
    if mask is None or mask.size == 0:
        return None
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


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
    pose = _load_matrix(rec.get("pose"))
    intrinsic = _load_matrix(rec.get("intrinsic"))
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
        mask = _load_mask(rec.get("mask_path"))
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
        mask = _load_mask(rec.get("mask_path"))
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


def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(points, colors):
            f.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


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
            point_count = fusion.get("point_count")
            inside = fusion.get("pointcloud_inside_box_ratio")
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


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _records_by_image(records: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        out.setdefault(rec["image_key"], []).append(rec)
    return out


def _records_by_key(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {rec["object_key"]: rec for rec in records}


def load_branch_records(branch: str, run_root: Path, raw_filter_records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    refine_task = BRANCHES[branch]["refine_task"]
    if refine_task is None:
        refine_df = load_stage_df(run_root, "filter_stage", "3dbox_filter")
    else:
        refine_df = load_stage_df(run_root, "localization_stage", refine_task)
    fusion_df = load_stage_df(run_root, "scene_fusion_stage", "depth_back_projection")
    refine_records = collect_records(refine_df, branch=branch, stage_label="refine", run_root=run_root)
    fusion_records = collect_records(fusion_df, branch=branch, stage_label="fusion", run_root=run_root)
    enrich_against_raw(refine_records, raw_filter_records)
    return refine_records, fusion_records


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
        choices=("none", "ply", "render", "both"),
        default="both",
        help="Export RGB-D masked point clouds as PLY, render orthographic thumbnails, or both.",
    )
    parser.add_argument("--max-pointcloud-points", type=int, default=120000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_roots = {
        "raw": resolve_run_root(args.raw_run, BRANCHES["raw"]["config_stem"]),
        "sam2": resolve_run_root(args.sam2_run, BRANCHES["sam2"]["config_stem"]),
        "sam3": resolve_run_root(args.sam3_run, BRANCHES["sam3"]["config_stem"]),
    }

    raw_filter_df = load_stage_df(run_roots["raw"], "filter_stage", "3dbox_filter")
    raw_filter_records = collect_records(raw_filter_df, branch="raw", stage_label="filter", run_root=run_roots["raw"])

    refine_by_branch: Dict[str, List[Dict[str, Any]]] = {}
    fusion_by_branch: Dict[str, List[Dict[str, Any]]] = {}
    for branch, root in run_roots.items():
        refine_by_branch[branch], fusion_by_branch[branch] = load_branch_records(branch, root, raw_filter_records)

    by_image = {branch: _records_by_image(records) for branch, records in refine_by_branch.items()}
    fusion_by_key = {branch: _records_by_key(records) for branch, records in fusion_by_branch.items()}
    image_keys = sorted(set().union(*(set(items.keys()) for items in by_image.values())))
    if args.max_images:
        image_keys = image_keys[: args.max_images]

    written = 0
    for image_key in image_keys:
        branch_records = {branch: by_image.get(branch, {}).get(image_key, []) for branch in ("raw", "sam2", "sam3")}
        seed_record = next((records[0] for records in branch_records.values() if records), None)
        if seed_record is None:
            continue
        image = _load_image(seed_record.get("image"))
        if image is None:
            continue

        name = _sanitize_name(image_key)
        panels = [
            _draw_branch_panel(image, branch, branch_records[branch], args.max_panel_width)
            for branch in ("raw", "sam2", "sam3")
        ]
        pointcloud_payload: Dict[str, Any] = {}
        if args.pointcloud_mode != "none":
            pc_dir = output_dir / "pointclouds"
            pc_panels = []
            for branch in ("raw", "sam2", "sam3"):
                points, colors = _branch_colored_points(
                    image,
                    branch_records[branch],
                    max_points=args.max_pointcloud_points,
                )
                entry: Dict[str, Any] = {"num_points": int(len(points))}
                if args.pointcloud_mode in ("ply", "both") and len(points):
                    ply_path = pc_dir / f"{name}_{branch}.ply"
                    _write_ply(ply_path, points, colors)
                    entry["ply_path"] = str(ply_path)
                if args.pointcloud_mode in ("render", "both"):
                    pc_panels.append(_render_pointcloud_views(points, colors, f"{branch.upper()} RGB-D point cloud"))
                pointcloud_payload[branch] = entry
            if pc_panels:
                panels = [
                    _hstack([panel, pc_panel.resize((panel.width, max(1, int(pc_panel.height * panel.width / pc_panel.width))), resample=Image.BILINEAR)])
                    for panel, pc_panel in zip(panels, pc_panels)
                ]
        stats = _stats_panel(image_key, branch_records, fusion_by_key, max(p.height for p in panels))
        combined = _hstack(panels + [stats])
        combined.save(output_dir / f"{name}.jpg", quality=92)

        payload = {
            "image_key": image_key,
            "image": seed_record.get("image"),
            "pointclouds": pointcloud_payload,
            "branches": {
                branch: {
                    "refine_records": branch_records[branch],
                    "fusion_records": [
                        fusion_by_key.get(branch, {}).get(rec["object_key"])
                        for rec in branch_records[branch]
                    ],
                }
                for branch in ("raw", "sam2", "sam3")
            },
        }
        (output_dir / f"{name}.json").write_text(
            json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        written += 1

    print(f"Wrote {written} comparison image(s) to {output_dir}")


if __name__ == "__main__":
    main()
