import json
import logging
import os
import random
from glob import glob
from typing import List, Optional

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = {
    "id", "dataset", "scene_id", "image", "pose", "depth_map",
    "intrinsic", "depth_scale", "bboxes_3d_world_coords", "obj_tags",
    "axis_align_matrix",
}
VALID_DATASETS = {"scannet", "3rscan", "matterport3d", "arkitscenes"}
VALID_DEPTH_SCALES = {1000, 4000}


def _read_per_image(directory: str) -> List[dict]:
    records = []
    files = sorted(glob(os.path.join(directory, "*.jsonl")))
    for fp in tqdm(files, desc="Loading per-image JSONL files", unit="file"):
        if fp.endswith("_scenes.jsonl"):
            continue
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return records


def _read_per_scene(directory: str) -> List[dict]:
    records = []
    files = sorted(glob(os.path.join(directory, "*_scenes.jsonl")))
    for fp in tqdm(files, desc="Loading per-scene JSONL files", unit="file"):
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return records


def validate_schema(directory: str) -> List[str]:
    """Check all per-image records have the required fields."""
    errors = []
    records = _read_per_image(directory)
    for i, r in enumerate(
        tqdm(records, desc="Validating schema", unit="record")
    ):
        missing = REQUIRED_FIELDS - set(r.keys())
        if missing:
            errors.append(f"Record {i} missing fields: {missing}")
    return errors


def validate_counts(directory: str) -> List[str]:
    """Check per-image count matches sum of per-scene num_images."""
    errors = []
    per_image = _read_per_image(directory)
    per_scene = _read_per_scene(directory)

    total_per_image = len(per_image)
    total_per_scene = sum(s.get("num_images", 0) for s in per_scene)

    if total_per_image != total_per_scene:
        errors.append(
            f"Count mismatch: {total_per_image} per-image records vs "
            f"{total_per_scene} total num_images in per-scene records"
        )
    return errors


def validate_value_ranges(directory: str) -> List[str]:
    """Check value ranges for depth_scale, pose, intrinsic, dataset."""
    errors = []
    records = _read_per_image(directory)
    for i, r in enumerate(
        tqdm(records, desc="Validating value ranges", unit="record")
    ):
        ds = r.get("depth_scale")
        if ds not in VALID_DEPTH_SCALES:
            errors.append(f"Record {i}: invalid depth_scale={ds}")

        dataset = r.get("dataset")
        if dataset not in VALID_DATASETS:
            errors.append(f"Record {i}: invalid dataset={dataset}")

        pose = r.get("pose")
        if pose is not None and not isinstance(pose, str):
            # pose can be a file path (string) or a matrix (list)
            try:
                m = np.array(pose)
                if m.shape != (4, 4):
                    errors.append(f"Record {i}: pose is not 4x4")
                else:
                    det = np.linalg.det(m[:3, :3])
                    if not (0.9 <= abs(det) <= 1.1):
                        errors.append(f"Record {i}: pose rotation det={det:.4f}")
            except (ValueError, TypeError):
                errors.append(f"Record {i}: pose is not a valid matrix")

        intrinsic = r.get("intrinsic")
        if intrinsic is not None and isinstance(intrinsic, list):
            try:
                m = np.array(intrinsic)
                if m.shape != (4, 4):
                    errors.append(f"Record {i}: intrinsic is not 4x4")
                elif np.allclose(m, 0):
                    errors.append(f"Record {i}: intrinsic is all zeros")
            except (ValueError, TypeError):
                pass

    return errors


def validate_paths(directory: str, data_root: str, sample_size: int = 100) -> List[str]:
    """Sample records and check that referenced files exist."""
    errors = []
    records = _read_per_image(directory)
    if not records:
        return ["No per-image records found"]

    sample = random.sample(records, min(sample_size, len(records)))
    for r in tqdm(sample, desc="Validating path reachability", unit="record"):
        for field in ("image", "depth_map", "intrinsic"):
            path = r.get(field)
            if path and isinstance(path, str):
                full = os.path.join(data_root, path)
                if not os.path.exists(full):
                    errors.append(f"Record {r.get('id')}: {field} not found: {path}")
    return errors


def run_all(
    directory: str,
    data_root: Optional[str] = None,
    sample_size: int = 100,
) -> bool:
    """Run all validation checks. Returns True if all pass."""
    all_passed = True

    print("Running schema validation...")
    errors = validate_schema(directory)
    if errors:
        print(f"  FAIL: {len(errors)} schema errors")
        for e in errors[:5]:
            print(f"    {e}")
        all_passed = False
    else:
        print("  PASS: schema check")

    print("Running count validation...")
    errors = validate_counts(directory)
    if errors:
        print(f"  FAIL: {errors[0]}")
        all_passed = False
    else:
        print("  PASS: count check")

    print("Running value range validation...")
    errors = validate_value_ranges(directory)
    if errors:
        print(f"  FAIL: {len(errors)} range errors")
        for e in errors[:5]:
            print(f"    {e}")
        all_passed = False
    else:
        print("  PASS: value range check")

    if data_root:
        print("Running path reachability validation...")
        errors = validate_paths(directory, data_root, sample_size)
        if errors:
            print(f"  FAIL: {len(errors)} unreachable paths")
            for e in errors[:5]:
                print(f"    {e}")
            all_passed = False
        else:
            print("  PASS: path reachability check")

    return all_passed
