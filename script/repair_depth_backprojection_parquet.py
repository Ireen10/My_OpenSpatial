#!/usr/bin/env python3
"""
Repair historical DepthBackProjecter parquet outputs where invalid object
pointclouds were dropped but object-aligned fields were not filtered.

The repair uses the final object index embedded in pointcloud filenames:
``pointcloud_<row_id>_<object_index>.pcd``. For rows where ``pointclouds`` is
shorter than ``obj_tags``, fields whose length matches the original object
count are filtered to those indices.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.parquet_io import load_parquet_dataframe  # noqa: E402

DEFAULT_STAGE_ROOT = (
    _REPO_ROOT
    / "output/EmbodiedScan/scannet/base_pipeline_demo_preprocessing_embodiedscan_sam3"
    / "scene_fusion_stage"
)
POINTCLOUD_OBJECT_INDEX_RE = re.compile(r"_(\d+)\.pcd$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair depth_back_projection/data.parquet object/pointcloud alignment."
    )
    parser.add_argument(
        "--stage-root",
        type=Path,
        default=DEFAULT_STAGE_ROOT,
        help="Path to scene_fusion_stage (default: current ScanNet output path).",
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=None,
        help="Explicit data.parquet path. Overrides --stage-root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report planned repairs without writing.",
    )
    parser.add_argument(
        "--min-pointclouds",
        type=int,
        default=2,
        help="Drop repaired rows with fewer than this many pointclouds.",
    )
    parser.add_argument(
        "--keep-backup",
        action="store_true",
        help="Keep a .bak copy of the original parquet. By default no backup is retained.",
    )
    return parser.parse_args()


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            value = value.tolist()
        except Exception:
            pass
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def object_index_from_pointcloud(path: Any) -> Optional[int]:
    match = POINTCLOUD_OBJECT_INDEX_RE.search(str(path or ""))
    return int(match.group(1)) if match else None


def filtered_value(value: Any, keep_indices: List[int], original_count: int) -> Any:
    seq = as_list(value)
    if len(seq) != original_count:
        return value
    return [seq[i] for i in keep_indices if i < len(seq)]


def backup_path(path: Path) -> Path:
    candidate = path.with_name(path.name + ".bak")
    if not candidate.exists():
        return candidate
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.name}.bak.{stamp}")


def repair_dataframe(df: pd.DataFrame, *, min_pointclouds: int) -> tuple[pd.DataFrame, dict]:
    repaired_rows = 0
    dropped_rows = 0
    skipped_unparseable = 0
    already_ok = 0
    examples = []
    output_rows = []

    for idx, row in df.iterrows():
        record = row.to_dict()
        tags = as_list(record.get("obj_tags"))
        pointclouds = as_list(record.get("pointclouds"))
        original_count = len(tags)

        if not tags or len(pointclouds) == original_count:
            already_ok += 1
            output_rows.append(record)
            continue

        if len(pointclouds) > original_count:
            skipped_unparseable += 1
            output_rows.append(record)
            continue

        keep_indices = [object_index_from_pointcloud(p) for p in pointclouds]
        if any(i is None for i in keep_indices):
            skipped_unparseable += 1
            output_rows.append(record)
            continue
        keep_indices = [int(i) for i in keep_indices]

        if len(keep_indices) < min_pointclouds:
            dropped_rows += 1
            continue

        repaired = dict(record)
        for key, value in record.items():
            if key == "pointclouds":
                repaired[key] = pointclouds
                continue
            repaired[key] = filtered_value(value, keep_indices, original_count)

        repaired_rows += 1
        if len(examples) < 10:
            examples.append({
                "row_index": int(idx),
                "id": record.get("id"),
                "scene_id": record.get("scene_id"),
                "original_count": original_count,
                "pointcloud_count": len(pointclouds),
                "keep_indices": keep_indices,
                "tags_before": tags,
                "tags_after": as_list(repaired.get("obj_tags")),
            })
        output_rows.append(repaired)

    stats = {
        "rows_in": len(df),
        "rows_out": len(output_rows),
        "already_ok": already_ok,
        "repaired_rows": repaired_rows,
        "dropped_rows": dropped_rows,
        "skipped_unparseable": skipped_unparseable,
        "examples": examples,
    }
    return pd.DataFrame(output_rows), stats


def validate_alignment(df: pd.DataFrame) -> int:
    bad = 0
    for _, row in df.iterrows():
        tags = as_list(row.get("obj_tags"))
        pcs = as_list(row.get("pointclouds"))
        masks = as_list(row.get("masks"))
        boxes = as_list(row.get("bboxes_2d"))
        n = len(tags)
        if len(pcs) != n or len(masks) != n or len(boxes) != n:
            bad += 1
    return bad


def main() -> None:
    args = parse_args()
    parquet_path = (
        args.parquet
        if args.parquet is not None
        else args.stage_root / "depth_back_projection" / "data.parquet"
    ).resolve()

    if not parquet_path.is_file():
        raise SystemExit(f"data.parquet not found: {parquet_path}")

    print(f">>> Loading {parquet_path}", flush=True)
    df = load_parquet_dataframe(str(parquet_path))
    repaired, stats = repair_dataframe(df, min_pointclouds=max(0, args.min_pointclouds))

    print(">>> Repair stats:", {k: v for k, v in stats.items() if k != "examples"}, flush=True)
    for example in stats["examples"]:
        print(">>> Example:", example, flush=True)

    bad_after = validate_alignment(repaired)
    print(f">>> Alignment issues after repair: {bad_after}", flush=True)
    if args.dry_run:
        print(">>> dry-run set; no files written", flush=True)
        return
    if bad_after:
        raise SystemExit("Refusing to write: alignment issues remain after repair.")

    if args.keep_backup:
        bak = backup_path(parquet_path)
        print(f">>> Backing up original parquet -> {bak}", flush=True)
        shutil.copy2(parquet_path, bak)

    tmp = parquet_path.with_name(parquet_path.name + ".repair_tmp")
    print(f">>> Writing repaired parquet -> {tmp}", flush=True)
    try:
        repaired.to_parquet(tmp, index=False)
        tmp.replace(parquet_path)
    finally:
        if tmp.exists():
            tmp.unlink()
    print(f">>> Repaired parquet written -> {parquet_path}", flush=True)


if __name__ == "__main__":
    main()
