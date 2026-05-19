"""Prepare ARKitScenes for OpenSpatial: vga_wide RGB, aligned depth, sky-corrected assets.

Reads official raw layout (flat or ``{scene_id}_frames/``) and writes EmbodiedScan-
compatible ``{scene_id}_frames/vga_wide`` outputs. Frame list follows EmbodiedScan v2 pkls.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from embodiedscan_data.arkit_geometry import (
    build_per_frame_sky_table,
    find_asset_file,
    find_depth_asset,
    load_pincam,
    pincam_to_cam2img,
    read_traj,
    resize_depth_to_rotated_vga,
    resolve_scene_sky_direction,
    rotate_depth_image,
    rotate_rgb_image,
    sky_direction_to_rotated_to_cam,
    transform_pincam,
)
from embodiedscan_data.datasets.arkitscenes import (
    ARKIT_SPLITS,
    _arkit_scene_cameras_index,
)

logger = logging.getLogger(__name__)

SCENE_MANIFEST = ".arkit_scene.json"
RGB_SUBDIR = "vga_wide"
DEPTH_SUBDIR = "vga_depth"
INTR_SUBDIR = "vga_wide_intrinsics"


@dataclass(frozen=True)
class PrepareTask:
    scene_key: str
    split: str
    scene_id: str
    cameras: Tuple[str, ...]


def _project_root_from_data_root(data_root: str) -> str:
    return os.path.dirname(os.path.abspath(data_root))


def _resolve_scene_dir(arkit_root: str, split: str, scene_id: str) -> Optional[str]:
    """Resolve scene folder under ``<root>/{split}/{id}`` or ``<root>/raw/{split}/{id}``."""
    if not arkit_root:
        return None
    for base in (
        os.path.join(arkit_root, split, scene_id),
        os.path.join(arkit_root, "raw", split, scene_id),
    ):
        if os.path.isdir(base):
            return base
    return None


def _asset_search_roots(scene_dir: str, scene_id: str) -> List[str]:
    roots = [
        scene_dir,
        os.path.join(scene_dir, f"{scene_id}_frames"),
        os.path.join(scene_dir, scene_id),
    ]
    out: List[str] = []
    seen = set()
    for root in roots:
        if root not in seen and os.path.isdir(root):
            out.append(root)
            seen.add(root)
    return out


def _asset_dir(scene_dir: str, scene_id: str, name: str) -> Optional[str]:
    for root in _asset_search_roots(scene_dir, scene_id):
        path = os.path.join(root, name)
        if os.path.isdir(path):
            return path
    return None


def _traj_path(scene_dir: str, scene_id: str) -> Optional[str]:
    for root in _asset_search_roots(scene_dir, scene_id):
        path = os.path.join(root, "lowres_wide.traj")
        if os.path.isfile(path):
            return path
    return None


def _frames_out_dir(data_root: str, split: str, scene_id: str) -> str:
    return os.path.join(
        data_root, "arkitscenes", split, scene_id, f"{scene_id}_frames"
    )


def _camera_timestamp(camera_name: str, scene_id: str) -> str:
    prefix = f"{scene_id}_"
    if camera_name.startswith(prefix):
        return camera_name[len(prefix) :]
    parts = camera_name.split("_", 1)
    return parts[1] if len(parts) == 2 else camera_name


def _load_v2_scene_cameras(project_root: str) -> Dict[str, Tuple[str, ...]]:
    return dict(_arkit_scene_cameras_index(project_root))


def _collect_tasks(
    data_root: str,
    project_root: str,
    scene_filter: Optional[Sequence[str]],
    max_scenes: Optional[int],
) -> List[PrepareTask]:
    index = _load_v2_scene_cameras(project_root)
    tasks: List[PrepareTask] = []
    for scene_key, cameras in sorted(index.items()):
        if scene_filter and scene_key not in scene_filter:
            continue
        parts = scene_key.split("/")
        if len(parts) != 3 or parts[0] != "arkitscenes":
            continue
        _, split, scene_id = parts
        if split not in ARKIT_SPLITS:
            continue
        tasks.append(
            PrepareTask(
                scene_key=scene_key,
                split=split,
                scene_id=scene_id,
                cameras=cameras,
            )
        )
    if max_scenes is not None:
        tasks = tasks[:max_scenes]
    return tasks


def _write_matrix(path: str, matrix: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in matrix:
            f.write(" ".join(f"{v:.9f}" for v in row) + "\n")


def _process_scene(
    task: PrepareTask,
    data_root: str,
    raw_root: str,
    apply_sky_correction: bool,
    force: bool,
    sky_source: str,
    sky_granularity: str,
) -> Tuple[str, int, int]:
    scene_dir = _resolve_scene_dir(raw_root, task.split, task.scene_id)
    if scene_dir is None:
        scene_dir = _resolve_scene_dir(
            os.path.join(data_root, "arkitscenes"), task.split, task.scene_id
        )
    if scene_dir is None:
        logger.warning("Missing raw scene directory for %s", task.scene_key)
        return task.scene_key, 0, len(task.cameras)

    traj_path = _traj_path(scene_dir, task.scene_id)
    if traj_path is None:
        logger.warning("Missing lowres_wide.traj for %s", task.scene_key)
        return task.scene_key, 0, len(task.cameras)

    timestamps, poses_cam_to_world = read_traj(traj_path)
    per_frame_sky: dict = {}
    if sky_granularity == "frame" and apply_sky_correction:
        per_frame_sky = build_per_frame_sky_table(
            timestamps,
            poses_cam_to_world,
            task.cameras,
            task.scene_id,
        )
        sky_counts: dict = {}
        for entry in per_frame_sky.values():
            label = entry["sky_direction"]
            sky_counts[label] = sky_counts.get(label, 0) + 1
        sky_direction = max(sky_counts, key=sky_counts.get)
        rotated_to_cam = sky_direction_to_rotated_to_cam(sky_direction)
        sky_resolved_from = "traj_per_frame"
    else:
        sky_direction, rotated_to_cam, sky_resolved_from = resolve_scene_sky_direction(
            task.scene_id, poses_cam_to_world, raw_root, sky_source=sky_source
        )
    if not apply_sky_correction:
        sky_direction = "UP"
        rotated_to_cam = np.eye(4)
        sky_resolved_from = "disabled"
        per_frame_sky = {}

    roots = _asset_search_roots(scene_dir, task.scene_id)
    vga_dirs = [os.path.join(r, "vga_wide") for r in roots]
    vga_intr_dirs = [os.path.join(r, "vga_wide_intrinsics") for r in roots]
    highres_depth_dirs = [os.path.join(r, "highres_depth") for r in roots]
    lowres_depth_dirs = [os.path.join(r, "lowres_depth") for r in roots]

    out_frames = _frames_out_dir(data_root, task.split, task.scene_id)
    out_rgb = os.path.join(out_frames, RGB_SUBDIR)
    out_depth = os.path.join(out_frames, DEPTH_SUBDIR)
    out_intr = os.path.join(out_frames, INTR_SUBDIR)
    os.makedirs(out_rgb, exist_ok=True)
    os.makedirs(out_depth, exist_ok=True)
    os.makedirs(out_intr, exist_ok=True)

    manifest_path = os.path.join(out_frames, SCENE_MANIFEST)
    if force or not os.path.isfile(manifest_path):
        manifest = {
            "sky_granularity": sky_granularity if apply_sky_correction else "none",
            "world_up_assumption": "+Z",
            "sky_direction": sky_direction,
            "sky_source": sky_resolved_from,
            "rotated_to_cam": rotated_to_cam.tolist(),
            "apply_sky_correction": apply_sky_correction,
        }
        if per_frame_sky:
            manifest["frames"] = per_frame_sky
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    ok = 0
    failed = 0
    for camera in task.cameras:
        ts = _camera_timestamp(camera, task.scene_id)
        out_img = os.path.join(out_rgb, f"{camera}.jpg")
        out_depth_path = os.path.join(out_depth, f"{camera}.png")
        out_matrix = os.path.join(out_intr, f"{camera}_matrix.txt")

        if (
            not force
            and os.path.isfile(out_img)
            and os.path.isfile(out_depth_path)
            and os.path.isfile(out_matrix)
        ):
            ok += 1
            continue

        vga_path = find_asset_file(vga_dirs, task.scene_id, ts, ".png")
        intr_path = find_asset_file(vga_intr_dirs, task.scene_id, ts, ".pincam")
        depth_path, depth_source = find_depth_asset(
            highres_depth_dirs, lowres_depth_dirs, task.scene_id, ts
        )

        if not vga_path or not intr_path or not depth_path or depth_source is None:
            logger.debug(
                "Skip %s/%s: vga=%s intr=%s depth=%s",
                task.scene_key,
                camera,
                vga_path,
                intr_path,
                depth_path,
            )
            failed += 1
            continue

        try:
            img = Image.open(vga_path)
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth is None:
                failed += 1
                continue

            frame_sky = sky_direction
            if per_frame_sky and camera in per_frame_sky:
                frame_sky = per_frame_sky[camera]["sky_direction"]

            w, h, fx, fy, cx, cy = load_pincam(intr_path)
            tw, th, tfx, tfy, tcx, tcy = transform_pincam(
                w, h, fx, fy, cx, cy, frame_sky
            )

            img = rotate_rgb_image(img, frame_sky)
            depth = rotate_depth_image(depth, frame_sky)
            depth = resize_depth_to_rotated_vga(
                depth, (img.size[0], img.size[1]), depth_source
            )

            img.save(out_img, quality=95)
            cv2.imwrite(out_depth_path, depth)
            _write_matrix(
                out_matrix,
                pincam_to_cam2img(tw, th, tfx, tfy, tcx, tcy),
            )
            ok += 1
        except Exception as exc:
            logger.warning("Failed %s/%s: %s", task.scene_key, camera, exc)
            failed += 1

    return task.scene_key, ok, failed


def _worker(args_tuple):
    task, data_root, raw_root, apply_sky, force, sky_source, sky_granularity = args_tuple
    return _process_scene(
        task, data_root, raw_root, apply_sky, force, sky_source, sky_granularity
    )


def prepare_arkitscenes(
    data_root: str,
    raw_root: Optional[str] = None,
    workers: int = 8,
    apply_sky_correction: bool = True,
    force: bool = False,
    max_scenes: Optional[int] = None,
    scene_filter: Optional[List[str]] = None,
    only_local_raw: bool = False,
    sky_source: str = "traj",
    sky_granularity: str = "frame",
) -> None:
    data_root = os.path.abspath(data_root)
    if raw_root is None:
        raw_root = os.path.join(data_root, "arkitscenes_highres")
    else:
        raw_root = os.path.abspath(raw_root)
    project_root = _project_root_from_data_root(data_root)
    tasks = _collect_tasks(data_root, project_root, scene_filter, max_scenes)
    if only_local_raw:
        tasks = [
            t
            for t in tasks
            if _resolve_scene_dir(raw_root, t.split, t.scene_id) is not None
        ]
    if not tasks:
        logger.error(
            "No ARKit scenes to prepare. Check embodiedscan-v2 pkls under %s",
            project_root,
        )
        return

    logger.info(
        "Preparing %d ARKit scenes (raw=%s, out=%s, workers=%d, "
        "granularity=%s, sky_source=%s)",
        len(tasks),
        raw_root,
        os.path.join(data_root, "arkitscenes"),
        workers,
        sky_granularity,
        sky_source,
    )
    total_ok = 0
    total_fail = 0

    if workers <= 1:
        iterator = (
            _worker(
                (t, data_root, raw_root, apply_sky_correction, force, sky_source, sky_granularity)
            )
            for t in tasks
        )
        for scene_key, ok, fail in tqdm(iterator, total=len(tasks), desc="ARKit prepare"):
            total_ok += ok
            total_fail += fail
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _worker,
                    (
                        t,
                        data_root,
                        raw_root,
                        apply_sky_correction,
                        force,
                        sky_source,
                        sky_granularity,
                    ),
                )
                for t in tasks
            ]
            for fut in tqdm(
                as_completed(futures), total=len(futures), desc="ARKit prepare"
            ):
                _, ok, fail = fut.result()
                total_ok += ok
                total_fail += fail

    print(f"\n{'=' * 60}")
    print("ARKitScenes prepare complete")
    print(f"  Scenes: {len(tasks)}")
    print(f"  Frames OK: {total_ok}")
    print(f"  Frames failed/skipped: {total_fail}")
    print(f"  Output under: {os.path.join(data_root, 'arkitscenes')}")
    print(f"{'=' * 60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare ARKitScenes vga_wide assets for OpenSpatial EmbodiedScan extract",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="EmbodiedScan data root; prepared assets go to <data-root>/arkitscenes/",
    )
    parser.add_argument(
        "--raw-root",
        default=None,
        help="Downloaded raw ARKitScenes root (default: <data-root>/arkitscenes_highres)",
    )
    parser.add_argument(
        "--only-local-raw",
        action="store_true",
        help="Only process scenes that exist under --raw-root",
    )
    parser.add_argument(
        "-ss",
        "--sky-source",
        choices=("metadata", "traj", "auto"),
        default="traj",
        help="Scene-level sky only (-sg scene): metadata|traj|auto (default: traj)",
    )
    parser.add_argument(
        "-sg",
        "--sky-granularity",
        choices=("scene", "frame"),
        default="frame",
        help="frame=per-frame traj; scene=one sky per video (default: frame)",
    )
    parser.add_argument("-j", "--workers", type=int, default=8, help="Parallel workers")
    parser.add_argument(
        "-ns",
        "--no-sky-correction",
        action="store_true",
        help="Pack vga/depth/intrinsics without sky rotation",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument(
        "--scene",
        action="append",
        default=None,
        help="Only prepare given sample_idx (repeatable), e.g. arkitscenes/Training/40753679",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    prepare_arkitscenes(
        data_root=args.data_root,
        raw_root=args.raw_root,
        workers=args.workers,
        apply_sky_correction=not args.no_sky_correction,
        force=args.force,
        max_scenes=args.max_scenes,
        scene_filter=args.scene,
        only_local_raw=args.only_local_raw,
        sky_source=args.sky_source,
        sky_granularity=args.sky_granularity,
    )


if __name__ == "__main__":
    main()
