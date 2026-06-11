#!/usr/bin/env python3
"""Summarize raw/SAM2/SAM3 refiner experiment outputs."""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.box_utils import (  # noqa: E402
    compute_box_3d_corners_from_params,
    convert_box_3d_world_to_camera,
)
from utils.parquet_io import load_parquet_dataframe  # noqa: E402

try:
    import open3d as o3d
except Exception:  # pragma: no cover - optional at report time
    o3d = None

try:
    from scipy.ndimage import label as cc_label
    from scipy.spatial.transform import Rotation as SciRotation
except Exception:  # pragma: no cover - optional fallback
    cc_label = None
    SciRotation = None


BRANCHES = {
    "raw": {"refine_task": None, "config_stem": "raw_no_refine"},
    "sam2": {"refine_task": "sam2_refiner", "config_stem": "sam2_refine"},
    "sam3": {"refine_task": "sam3_refiner", "config_stem": "sam3_refine"},
}


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _as_list(value: Any) -> List[Any]:
    if _is_missing(value):
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _safe_str(value: Any) -> str:
    if _is_missing(value):
        return ""
    return str(value)


def _load_matrix(path: Any) -> Optional[np.ndarray]:
    if not isinstance(path, str) or not path:
        return None
    try:
        return np.loadtxt(path)
    except Exception:
        return None


def _load_mask(mask_ref: Any, run_root: Optional[Path] = None) -> Optional[np.ndarray]:
    if _is_missing(mask_ref):
        return None
    try:
        if isinstance(mask_ref, dict) and mask_ref.get("bytes"):
            return np.array(Image.open(io.BytesIO(mask_ref["bytes"]))) > 0
        if isinstance(mask_ref, str):
            path = Path(mask_ref)
            if not path.exists() and run_root is not None:
                path = run_root / mask_ref
            return np.array(Image.open(path)) > 0
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


def _bbox_area(box: Optional[Iterable[float]]) -> float:
    if box is None:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in box]
    return max(0.0, x2 - x1 + 1.0) * max(0.0, y2 - y1 + 1.0)


def _bbox_intersection(a: Optional[Iterable[float]], b: Optional[Iterable[float]]) -> float:
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0.0, ix2 - ix1 + 1.0) * max(0.0, iy2 - iy1 + 1.0)


def bbox_iou(a: Optional[Iterable[float]], b: Optional[Iterable[float]]) -> Optional[float]:
    inter = _bbox_intersection(a, b)
    union = _bbox_area(a) + _bbox_area(b) - inter
    if union <= 0:
        return None
    return float(inter / union)


def mask_iou(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None or a.shape != b.shape:
        return None
    aa = a > 0
    bb = b > 0
    union = np.logical_or(aa, bb).sum()
    if union == 0:
        return None
    return float(np.logical_and(aa, bb).sum() / union)


def _connected_components(mask: Optional[np.ndarray]) -> Tuple[int, float]:
    if mask is None or mask.size == 0 or int(mask.sum()) == 0:
        return 0, 0.0
    binary = mask > 0
    total = int(binary.sum())
    if cc_label is None:
        return 1, 1.0
    labels, count = cc_label(binary)
    if count == 0:
        return 0, 0.0
    sizes = np.bincount(labels.ravel())[1:]
    return int(count), float(sizes.max() / total) if total else 0.0


def mask_stats(mask: Optional[np.ndarray]) -> Dict[str, Any]:
    bbox = _mask_bbox(mask)
    area = int(mask.sum()) if mask is not None else 0
    bbox_area = _bbox_area(bbox)
    components, max_component_ratio = _connected_components(mask)
    return {
        "mask_area": area,
        "mask_bbox": bbox,
        "mask_bbox_area": bbox_area,
        "mask_bbox_fill_ratio": float(area / bbox_area) if bbox_area > 0 else None,
        "mask_components": components,
        "mask_max_component_ratio": max_component_ratio,
    }


def _projected_bbox_from_3d(
    box_3d: Any,
    pose_path: Any,
    intrinsic_path: Any,
) -> Optional[List[float]]:
    if _is_missing(box_3d):
        return None
    pose = _load_matrix(pose_path)
    intrinsic = _load_matrix(intrinsic_path)
    if pose is None or intrinsic is None:
        return None
    try:
        corners = compute_box_3d_corners_from_params(_as_list(box_3d))
        corners_h = np.concatenate([corners, np.ones((corners.shape[0], 1))], axis=1)
        cam = (np.linalg.inv(pose) @ corners_h.T).T[:, :3]
        valid = cam[:, 2] > 1e-3
        if not np.any(valid):
            return None
        k = intrinsic[:3, :3]
        uv = np.empty((int(valid.sum()), 2), dtype=np.float64)
        pts = cam[valid]
        uv[:, 0] = k[0, 0] * pts[:, 0] / pts[:, 2] + k[0, 2]
        uv[:, 1] = k[1, 1] * pts[:, 1] / pts[:, 2] + k[1, 2]
        return [float(uv[:, 0].min()), float(uv[:, 1].min()), float(uv[:, 0].max()), float(uv[:, 1].max())]
    except Exception:
        return None


def _box_signature(box: Any) -> str:
    vals = _as_list(box)
    if not vals:
        return ""
    try:
        rounded = [round(float(v), 4) for v in vals[:9]]
    except Exception:
        rounded = [str(v) for v in vals[:9]]
    return json.dumps(rounded, ensure_ascii=False, separators=(",", ":"))


def object_key(row: Dict[str, Any], tag: Any, box: Any, obj_idx: int) -> str:
    scene_id = _safe_str(row.get("scene_id"))
    image_id = _safe_str(row.get("id")) or Path(_safe_str(row.get("image"))).stem
    return "|".join([scene_id, image_id, _safe_str(tag), _box_signature(box) or str(obj_idx)])


def image_key(row: Dict[str, Any]) -> str:
    scene_id = _safe_str(row.get("scene_id"))
    image_id = _safe_str(row.get("id")) or Path(_safe_str(row.get("image"))).stem
    return "|".join([scene_id, image_id])


def _read_pcd_stats(
    pointcloud_ref: Any,
    box_world: Any,
    pose_path: Any,
) -> Dict[str, Any]:
    stats = {
        "pointcloud_path": None,
        "point_count": None,
        "pointcloud_aabb_volume": None,
        "pointcloud_center_distance_to_box": None,
        "pointcloud_inside_box_ratio": None,
    }
    if not isinstance(pointcloud_ref, str) or not pointcloud_ref:
        return stats
    stats["pointcloud_path"] = pointcloud_ref
    if o3d is None or not Path(pointcloud_ref).exists():
        return stats
    try:
        pts = np.asarray(o3d.io.read_point_cloud(pointcloud_ref).points)
    except Exception:
        return stats
    if pts.size == 0:
        stats["point_count"] = 0
        return stats
    stats["point_count"] = int(len(pts))
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    stats["pointcloud_aabb_volume"] = float(np.prod(np.maximum(maxs - mins, 0.0)))

    pose = _load_matrix(pose_path)
    if pose is None or SciRotation is None or _is_missing(box_world):
        return stats
    try:
        box_cam = convert_box_3d_world_to_camera(_as_list(box_world), pose)
        center = np.asarray(box_cam[:3], dtype=np.float64)
        extent = np.asarray(box_cam[3:6], dtype=np.float64)
        rot = SciRotation.from_euler("zxy", box_cam[6:9], degrees=False).as_matrix()
        local = (pts - center) @ rot
        inside = np.all(np.abs(local) <= (extent / 2.0 + 1e-6), axis=1)
        stats["pointcloud_inside_box_ratio"] = float(inside.mean())
        stats["pointcloud_center_distance_to_box"] = float(np.linalg.norm(pts.mean(axis=0) - center))
    except Exception:
        pass
    return stats


def resolve_run_root(path: str, config_stem: str) -> Path:
    root = Path(path)
    if (root / "filter_stage").exists():
        return root
    expected = root / f"base_pipeline_{config_stem}"
    if expected.exists():
        return expected
    candidates = sorted(p for p in root.glob("base_pipeline_*") if p.is_dir())
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"Cannot resolve run root under {path!r}")


def _stage_parquet(run_root: Path, stage: str, task: str) -> Optional[Path]:
    path = run_root / stage / task
    if path.exists():
        return path
    return None


def load_stage_df(run_root: Path, stage: str, task: str) -> pd.DataFrame:
    path = _stage_parquet(run_root, stage, task)
    if path is None:
        return pd.DataFrame()
    return load_parquet_dataframe(str(path))


def collect_records(
    df: pd.DataFrame,
    *,
    branch: str,
    stage_label: str,
    run_root: Path,
    include_assets: bool = True,
) -> List[Dict[str, Any]]:
    """Expand parquet rows to per-object records.

    include_assets=False skips loading masks and point clouds (fast path for
    serve_compare); geometry metrics are left None.
    """
    records: List[Dict[str, Any]] = []
    for row_idx, row_obj in df.iterrows():
        row = row_obj.to_dict()
        tags = _as_list(row.get("obj_tags"))
        masks = _as_list(row.get("masks"))
        boxes = _as_list(row.get("bboxes_3d_world_coords"))
        pcds = _as_list(row.get("pointclouds"))
        n = max(len(tags), len(masks), len(boxes), len(pcds))
        img_k = image_key(row)
        for obj_idx in range(n):
            tag = tags[obj_idx] if obj_idx < len(tags) else ""
            mask_ref = masks[obj_idx] if obj_idx < len(masks) else None
            box = boxes[obj_idx] if obj_idx < len(boxes) else None
            pcd = pcds[obj_idx] if obj_idx < len(pcds) else None
            rec: Dict[str, Any] = {
                "branch": branch,
                "stage": stage_label,
                "row_index": int(row_idx),
                "scene_id": _safe_str(row.get("scene_id")),
                "image_id": _safe_str(row.get("id")),
                "image": _safe_str(row.get("image")),
                "depth_map": _safe_str(row.get("depth_map")),
                "depth_scale": row.get("depth_scale"),
                "pose": _safe_str(row.get("pose")),
                "intrinsic": _safe_str(row.get("intrinsic")),
                "image_key": img_k,
                "object_index": obj_idx,
                "object_key": object_key(row, tag, box, obj_idx),
                "tag": _safe_str(tag),
                "mask_path": _safe_str(mask_ref),
                "box_3d": _as_list(box),
                "box_3d_signature": _box_signature(box),
                "pointcloud_path": _safe_str(pcd) if isinstance(pcd, str) else None,
            }
            if include_assets:
                mask = _load_mask(mask_ref, run_root)
                stats = mask_stats(mask)
                proj_bbox = _projected_bbox_from_3d(box, row.get("pose"), row.get("intrinsic"))
                rec["projected_3d_bbox"] = proj_bbox
                rec.update(stats)
                rec["bbox_projected_iou"] = bbox_iou(stats["mask_bbox"], proj_bbox)
                inter = _bbox_intersection(stats["mask_bbox"], proj_bbox)
                proj_area = _bbox_area(proj_bbox)
                bbox_area = _bbox_area(stats["mask_bbox"])
                rec["bbox_projected_coverage"] = float(inter / proj_area) if proj_area > 0 else None
                rec["bbox_inside_projected_ratio"] = float(inter / bbox_area) if bbox_area > 0 else None
                rec.update(_read_pcd_stats(pcd, box, row.get("pose")))
            else:
                rec["projected_3d_bbox"] = None
                rec.update({
                    "mask_area": None,
                    "mask_bbox": None,
                    "mask_bbox_area": None,
                    "mask_bbox_fill_ratio": None,
                    "mask_components": None,
                    "mask_max_component_ratio": None,
                    "bbox_projected_iou": None,
                    "bbox_projected_coverage": None,
                    "bbox_inside_projected_ratio": None,
                    "point_count": None,
                    "pointcloud_aabb_volume": None,
                    "pointcloud_center_distance_to_box": None,
                    "pointcloud_inside_box_ratio": None,
                })
            records.append(rec)
    return records


def _records_by_key(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {rec["object_key"]: rec for rec in records}


def enrich_against_raw(
    records: List[Dict[str, Any]],
    raw_records: List[Dict[str, Any]],
    *,
    include_assets: bool = True,
) -> None:
    raw_by_key = _records_by_key(raw_records)
    raw_masks: Dict[str, Optional[np.ndarray]] = {}
    if include_assets:
        for raw in raw_records:
            raw_masks[raw["object_key"]] = _load_mask(raw.get("mask_path"))

    for rec in records:
        raw = raw_by_key.get(rec["object_key"])
        if raw is None:
            rec["present_in_raw"] = False
            rec["mask_iou_with_raw"] = None
            rec["mask_area_ratio_vs_raw"] = None
            rec["bbox_center_shift_vs_raw"] = None
            continue
        rec["present_in_raw"] = True
        if not include_assets:
            rec["mask_iou_with_raw"] = None
            rec["mask_area_ratio_vs_raw"] = None
            rec["bbox_center_shift_vs_raw"] = None
            continue
        mask = _load_mask(rec.get("mask_path"))
        rec["mask_iou_with_raw"] = mask_iou(mask, raw_masks.get(rec["object_key"]))
        raw_area = raw.get("mask_area") or 0
        rec["mask_area_ratio_vs_raw"] = float(rec.get("mask_area", 0) / raw_area) if raw_area else None
        rec["bbox_center_shift_vs_raw"] = _bbox_center_shift(rec.get("mask_bbox"), raw.get("mask_bbox"))


def _bbox_center_shift(a: Optional[List[float]], b: Optional[List[float]]) -> Optional[float]:
    if a is None or b is None:
        return None
    ax = (float(a[0]) + float(a[2])) / 2.0
    ay = (float(a[1]) + float(a[3])) / 2.0
    bx = (float(b[0]) + float(b[2])) / 2.0
    by = (float(b[1]) + float(b[3])) / 2.0
    return float(math.hypot(ax - bx, ay - by))


def _mean(records: List[Dict[str, Any]], field: str) -> Optional[float]:
    vals = [rec.get(field) for rec in records if rec.get(field) is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def _stage_summary(df: pd.DataFrame, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "samples": int(len(df)),
        "objects": int(len(records)),
        "mean_objects_per_sample": float(len(records) / len(df)) if len(df) else 0.0,
        "mean_mask_area": _mean(records, "mask_area"),
        "mean_mask_fill_ratio": _mean(records, "mask_bbox_fill_ratio"),
        "mean_bbox_projected_iou": _mean(records, "bbox_projected_iou"),
        "mean_point_count": _mean(records, "point_count"),
        "mean_pointcloud_inside_box_ratio": _mean(records, "pointcloud_inside_box_ratio"),
    }


def summarize_branch(branch: str, run_root: Path, raw_filter_records: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    refine_task = BRANCHES[branch]["refine_task"]
    filter_df = load_stage_df(run_root, "filter_stage", "3dbox_filter")
    refine_df = filter_df if refine_task is None else load_stage_df(run_root, "localization_stage", refine_task)
    fusion_df = load_stage_df(run_root, "scene_fusion_stage", "depth_back_projection")

    filter_records = collect_records(filter_df, branch=branch, stage_label="filter", run_root=run_root)
    refine_records = collect_records(refine_df, branch=branch, stage_label="refine", run_root=run_root)
    fusion_records = collect_records(fusion_df, branch=branch, stage_label="fusion", run_root=run_root)

    baseline = raw_filter_records if raw_filter_records else filter_records
    enrich_against_raw(refine_records, baseline)
    enrich_against_raw(fusion_records, baseline)

    summary = {
        "run_root": str(run_root),
        "filter": _stage_summary(filter_df, filter_records),
        "refine": _stage_summary(refine_df, refine_records),
        "fusion": _stage_summary(fusion_df, fusion_records),
    }
    filter_objects = summary["filter"]["objects"]
    refine_objects = summary["refine"]["objects"]
    fusion_objects = summary["fusion"]["objects"]
    summary["retention"] = {
        "refine_vs_filter": float(refine_objects / filter_objects) if filter_objects else None,
        "fusion_vs_filter": float(fusion_objects / filter_objects) if filter_objects else None,
        "fusion_vs_refine": float(fusion_objects / refine_objects) if refine_objects else None,
    }
    summary["quality_vs_raw"] = {
        "mean_mask_iou_with_raw": _mean(refine_records, "mask_iou_with_raw"),
        "mean_mask_area_ratio_vs_raw": _mean(refine_records, "mask_area_ratio_vs_raw"),
        "mean_bbox_center_shift_vs_raw": _mean(refine_records, "bbox_center_shift_vs_raw"),
    }
    return summary, filter_records + refine_records + fusion_records


def write_summary_md(summary: Dict[str, Any], path: Path) -> None:
    lines = ["# Refiner Experiment Summary", ""]
    lines.append("| Branch | Filter Samples | Refine Samples | Fusion Samples | Filter Objects | Refine Objects | Fusion Objects | Refine/Object Retention | Fusion/Object Retention |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for branch, item in summary["branches"].items():
        retention = item["retention"]
        lines.append(
            "| {branch} | {fs} | {rs} | {fus} | {fo} | {ro} | {fuo} | {rr} | {fr} |".format(
                branch=branch,
                fs=item["filter"]["samples"],
                rs=item["refine"]["samples"],
                fus=item["fusion"]["samples"],
                fo=item["filter"]["objects"],
                ro=item["refine"]["objects"],
                fuo=item["fusion"]["objects"],
                rr=_fmt_ratio(retention["refine_vs_filter"]),
                fr=_fmt_ratio(retention["fusion_vs_filter"]),
            )
        )
    lines.extend(["", "## Quality Versus Raw", ""])
    lines.append("| Branch | Mean Mask IoU | Mean Area Ratio | Mean BBox Center Shift | Mean 3D Projection IoU | Mean Point Count | Mean Point-In-Box Ratio |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for branch, item in summary["branches"].items():
        q = item["quality_vs_raw"]
        ref = item["refine"]
        fus = item["fusion"]
        lines.append(
            "| {branch} | {miou} | {area} | {shift} | {proj} | {pts} | {inside} |".format(
                branch=branch,
                miou=_fmt_ratio(q["mean_mask_iou_with_raw"]),
                area=_fmt_float(q["mean_mask_area_ratio_vs_raw"]),
                shift=_fmt_float(q["mean_bbox_center_shift_vs_raw"]),
                proj=_fmt_ratio(ref["mean_bbox_projected_iou"]),
                pts=_fmt_float(fus["mean_point_count"]),
                inside=_fmt_ratio(fus["mean_pointcloud_inside_box_ratio"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt_ratio(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _fmt_float(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-run", default="refiner_exp/outputs/raw")
    parser.add_argument("--sam2-run", default="refiner_exp/outputs/sam2")
    parser.add_argument("--sam3-run", default="refiner_exp/outputs/sam3")
    parser.add_argument("--output-dir", default="refiner_exp/outputs/compare")
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

    summary = {"branches": {}}
    all_records: List[Dict[str, Any]] = []
    for branch, run_root in run_roots.items():
        branch_summary, branch_records = summarize_branch(branch, run_root, raw_filter_records)
        summary["branches"][branch] = branch_summary
        all_records.extend(branch_records)

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_summary_md(summary, output_dir / "summary.md")
    pd.DataFrame(all_records).to_csv(output_dir / "object_metrics.csv", index=False)
    print(f"Wrote {output_dir / 'summary.json'}")
    print(f"Wrote {output_dir / 'summary.md'}")
    print(f"Wrote {output_dir / 'object_metrics.csv'}")


if __name__ == "__main__":
    main()
