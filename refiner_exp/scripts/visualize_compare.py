#!/usr/bin/env python3
"""Create side-by-side Raw/SAM2/SAM3 refiner comparison images."""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
from dataclasses import dataclass
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
try:
    import open3d as o3d
except ImportError:  # pragma: no cover
    o3d = None


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
    return (pts,)


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


def render_combined_image(
    index: CompareIndex,
    image_key: str,
    *,
    max_panel_width: int = 640,
) -> Optional[Image.Image]:
    branch_records = branch_records_for_image(index, image_key)
    seed_record = next((records[0] for records in branch_records.values() if records), None)
    if seed_record is None:
        return None
    image = _load_image(seed_record.get("image"))
    if image is None:
        return None
    _enrich_image_records_for_stats(branch_records, index.raw_by_key)
    panels = [
        _draw_branch_panel(image, branch, branch_records[branch], max_panel_width)
        for branch in ("raw", "sam2", "sam3")
    ]
    stats = _stats_panel(image_key, branch_records, index.fusion_by_key, max(p.height for p in panels))
    return _hstack(panels + [stats])


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-run", default="refiner_exp/outputs/raw")
    parser.add_argument("--sam2-run", default="refiner_exp/outputs/sam2")
    parser.add_argument("--sam3-run", default="refiner_exp/outputs/sam3")
    parser.add_argument("--output-dir", default="refiner_exp/outputs/compare/images")
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--max-panel-width", type=int, default=640)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate JPG/JSON even when outputs already exist.",
    )
    return parser.parse_args()


def _image_outputs_exist(output_dir: Path, name: str) -> bool:
    return (output_dir / f"{name}.jpg").is_file() and (output_dir / f"{name}.json").is_file()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index = build_compare_index(
        args.raw_run, args.sam2_run, args.sam3_run, max_images=args.max_images,
    )
    _log(f"Exporting {len(index.image_keys)} overlay JPG(s) to {output_dir}")

    written = 0
    skipped = 0
    for image_key in tqdm.tqdm(index.image_keys, desc="export"):
        name = index.name_by_key[image_key]
        if not args.force and _image_outputs_exist(output_dir, name):
            skipped += 1
            continue
        combined = render_combined_image(index, image_key, max_panel_width=args.max_panel_width)
        if combined is None:
            continue
        combined.save(output_dir / f"{name}.jpg", quality=92)
        (output_dir / f"{name}.json").write_text(
            json.dumps(sample_stats_json(index, image_key), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        written += 1

    msg = f"Wrote {written} comparison image(s) to {output_dir}"
    if skipped:
        msg += f" (skipped {skipped} existing; use --force to refresh)"
    msg += " — for interactive 3D viewing run: python refiner_exp/scripts/serve_compare.py"
    print(msg)


if __name__ == "__main__":
    main()
