"""Compare multiple traj-based sky_direction heuristics against metadata.csv."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Reuse OpenSpatial geometry (run from repo root with venv that has embodiedscan-data / deps).
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "data_preprocessing" / "embodiedscan"))

from embodiedscan_data.arkit_geometry import (  # noqa: E402
    find_scene_orientation,
    get_right_vectors,
    get_up_vectors,
    normalize_sky_direction,
    read_traj,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"

Sky = str  # UP | DOWN | LEFT | RIGHT

DECIDE_POSE_TO_SKY: Dict[int, Sky] = {
    0: "UP",
    1: "LEFT",
    2: "DOWN",
    3: "RIGHT",
}


def decide_pose(pose: np.ndarray) -> int:
    """Official rectify_im.py: camera Z vs world ±X/±Y references."""
    z_vec = pose[2, :3]
    z_orien = np.array(
        [
            [0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    return int(np.argmax(z_orien @ z_vec))


def decide_pose_sky(pose: np.ndarray) -> Sky:
    return DECIDE_POSE_TO_SKY[decide_pose(pose)]


def find_scene_orientation_world_up(
    poses_cam_to_world: List[np.ndarray],
    world_up: Tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> Sky:
    """Same logic as find_scene_orientation but configurable world vertical."""
    if not poses_cam_to_world:
        return "UP"
    up_vector = sum(get_up_vectors(p) for p in poses_cam_to_world) / len(poses_cam_to_world)
    right_vector = sum(get_right_vectors(p) for p in poses_cam_to_world) / len(
        poses_cam_to_world
    )
    up_world = np.array([[world_up[0]], [world_up[1]], [world_up[2]], [0.0]])

    def _angle(v: np.ndarray) -> float:
        return (
            math.acos(float(np.clip(np.dot(np.transpose(up_world), v), -1.0, 1.0)))
            * 180.0
            / math.pi
        )

    device_up_to_world_up_angle = _angle(up_vector)
    device_right_to_world_up_angle = _angle(right_vector)
    up_closest_to_90 = abs(device_up_to_world_up_angle - 90.0) < abs(
        device_right_to_world_up_angle - 90.0
    )
    if up_closest_to_90:
        return "LEFT" if device_right_to_world_up_angle > 90.0 else "RIGHT"
    return "DOWN" if device_up_to_world_up_angle > 90.0 else "UP"


def _majority(labels: List[Sky]) -> Optional[Sky]:
    if not labels:
        return None
    counts = Counter(labels)
    top = counts.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return None  # tie
    return top[0][0]


def _first_fraction_poses(poses: List[np.ndarray], fraction: float) -> List[np.ndarray]:
    if not poses:
        return []
    n = max(1, int(len(poses) * fraction))
    return poses[:n]


def compute_all_methods(poses: List[np.ndarray]) -> Dict[str, Optional[Sky]]:
    if not poses:
        return {name: None for name in METHOD_NAMES}

    per_decide = [decide_pose_sky(p) for p in poses]
    per_stat_z = [find_scene_orientation_world_up([p], (0, 0, 1)) for p in poses]

    return {
        "global_stat_z_up": find_scene_orientation(poses)[0],
        "global_stat_y_up": find_scene_orientation_world_up(poses, (0, 1, 0)),
        "global_stat_neg_y_up": find_scene_orientation_world_up(poses, (0, -1, 0)),
        "first_frame_stat_z_up": find_scene_orientation([poses[0]])[0],
        "first_frame_decide_pose": per_decide[0],
        "majority_decide_pose": _majority(per_decide),
        "majority_stat_z_up": _majority(per_stat_z),
        "first_10pct_stat_z_up": find_scene_orientation_world_up(
            _first_fraction_poses(poses, 0.1), (0, 0, 1)
        ),
        "last_frame_stat_z_up": find_scene_orientation([poses[-1]])[0],
        "last_frame_decide_pose": per_decide[-1],
        "median_frame_decide_pose": per_decide[len(per_decide) // 2],
    }


METHOD_NAMES = [
    "global_stat_z_up",
    "global_stat_y_up",
    "global_stat_neg_y_up",
    "first_frame_stat_z_up",
    "first_frame_decide_pose",
    "majority_decide_pose",
    "majority_stat_z_up",
    "first_10pct_stat_z_up",
    "last_frame_stat_z_up",
    "last_frame_decide_pose",
    "median_frame_decide_pose",
]


def _traj_path(data_dir: Path, fold: str, video_id: str) -> Path:
    return data_dir / "traj" / fold / video_id / "lowres_wide.traj"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    meta_path = data_dir / "metadata.csv"
    splits_path = data_dir / "raw_train_val_splits.csv"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing {meta_path}; run download_all_traj.py first")

    meta = pd.read_csv(meta_path)
    meta["video_id"] = meta["video_id"].astype(int)
    meta["sky_direction"] = meta["sky_direction"].astype(str).str.strip()
    meta = meta[
        meta["sky_direction"].notna()
        & (meta["sky_direction"] != "")
        & (meta["sky_direction"].str.lower() != "nan")
    ].copy()
    meta["sky_meta"] = meta["sky_direction"].map(normalize_sky_direction)

    fold_by_vid: Dict[int, str] = {}
    if splits_path.is_file():
        splits = pd.read_csv(splits_path)
        for _, row in splits.iterrows():
            fold_by_vid[int(row["video_id"])] = str(row["fold"])

    rows = []
    for _, mrow in meta.iterrows():
        vid = int(mrow["video_id"])
        fold = fold_by_vid.get(vid, mrow.get("fold") if "fold" in mrow else None)
        if fold is None or (isinstance(fold, float) and math.isnan(fold)):
            fold = str(mrow["fold"]) if "fold" in mrow and pd.notna(mrow["fold"]) else None
        if fold is None:
            continue
        fold = str(fold)
        traj = _traj_path(data_dir, fold, str(vid))
        if not traj.is_file():
            rows.append(
                {
                    "video_id": vid,
                    "fold": fold,
                    "sky_meta": mrow["sky_meta"],
                    "has_traj": False,
                }
            )
            continue

        _, poses = read_traj(str(traj))
        preds = compute_all_methods(poses)
        row = {
            "video_id": vid,
            "fold": fold,
            "sky_meta": mrow["sky_meta"],
            "has_traj": True,
            "traj_lines": len(poses),
        }
        row.update(preds)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "per_video.csv", index=False)

    eval_df = df[df["has_traj"] == True].copy()  # noqa: E712
    summary: Dict[str, object] = {
        "metadata_rows": int(len(meta)),
        "with_traj": int(len(eval_df)),
        "without_traj": int((~df["has_traj"]).sum()),
        "methods": {},
    }

    for method in METHOD_NAMES:
        if method not in eval_df.columns:
            continue
        valid = eval_df[eval_df[method].notna()]
        match = valid[valid[method] == valid["sky_meta"]]
        rate = float(len(match) / len(valid)) if len(valid) else 0.0
        summary["methods"][method] = {
            "match_count": int(len(match)),
            "evaluated": int(len(valid)),
            "match_rate": rate,
            "ties_or_null": int(eval_df[method].isna().sum()),
        }
        mism = valid[valid[method] != valid["sky_meta"]][
            ["video_id", "fold", "sky_meta", method, "traj_lines"]
        ]
        mism.to_csv(results_dir / f"mismatch_{method}.csv", index=False)

    # Best method
    best = max(
        summary["methods"].items(),
        key=lambda kv: kv[1]["match_rate"],
    )
    summary["best_method"] = {"name": best[0], **best[1]}

    with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {results_dir / 'per_video.csv'}")
    print(f"Best: {best[0]}  match_rate={best[1]['match_rate']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
