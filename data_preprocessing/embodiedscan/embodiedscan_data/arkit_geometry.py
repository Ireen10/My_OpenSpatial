"""ARKitScenes sky-direction and camera geometry (aligned with CUT3R preprocessing)."""

from __future__ import annotations

import math
import os
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

SkyDirection = str  # "UP" | "DOWN" | "LEFT" | "RIGHT"

_SKY_ALIASES = {
    "UP": "UP",
    "DOWN": "DOWN",
    "LEFT": "LEFT",
    "RIGHT": "RIGHT",
}


def normalize_sky_direction(value: str) -> SkyDirection:
    """Map Apple metadata.csv values (Up/Down/Left/Right) to internal labels."""
    key = value.strip().upper()
    if key not in _SKY_ALIASES:
        raise ValueError(f"Unknown sky_direction: {value!r}")
    return _SKY_ALIASES[key]


def sky_direction_to_rotated_to_cam(sky_direction: SkyDirection) -> np.ndarray:
    """4x4 rotated_to_cam for a known sky label (same convention as find_scene_orientation)."""
    if sky_direction == "LEFT":
        cam_to_rotated_q = Rotation.from_euler("z", math.pi / 2.0)
    elif sky_direction == "RIGHT":
        cam_to_rotated_q = Rotation.from_euler("z", -math.pi / 2.0)
    elif sky_direction == "DOWN":
        cam_to_rotated_q = Rotation.from_euler("z", math.pi)
    else:
        cam_to_rotated_q = Rotation.from_matrix(np.eye(3))
    cam_to_rotated = np.eye(4)
    cam_to_rotated[:3, :3] = cam_to_rotated_q.as_matrix()
    return np.linalg.inv(cam_to_rotated)


def load_sky_direction_from_metadata(
    video_id: str,
    raw_root: str,
) -> SkyDirection | None:
    """Read official sky_direction for a video_id from ARKitScenes metadata.csv."""
    candidates = [
        os.path.join(raw_root, "raw", "metadata.csv"),
        os.path.join(raw_root, "metadata.csv"),
    ]
    try:
        import pandas as pd
    except ImportError:
        return None

    for path in candidates:
        if not os.path.isfile(path):
            continue
        df = pd.read_csv(path)
        if "video_id" not in df.columns or "sky_direction" not in df.columns:
            continue
        rows = df[df["video_id"].astype(int) == int(video_id)]
        if rows.empty:
            continue
        return normalize_sky_direction(str(rows["sky_direction"].iloc[0]))
    return None


def resolve_scene_sky_direction(
    scene_id: str,
    poses_cam_to_world: List[np.ndarray],
    raw_root: str,
    sky_source: str = "metadata",
) -> Tuple[SkyDirection, np.ndarray, str]:
    """Resolve sky_direction and rotated_to_cam.

    Returns:
        sky_direction, rotated_to_cam, source tag (``metadata`` | ``traj``)
    """
    if sky_source in ("metadata", "auto"):
        meta_sky = load_sky_direction_from_metadata(scene_id, raw_root)
        if meta_sky is not None:
            return meta_sky, sky_direction_to_rotated_to_cam(meta_sky), "metadata"
        if sky_source == "metadata":
            raise FileNotFoundError(
                f"No sky_direction in metadata.csv for video {scene_id}"
            )

    sky_direction, rotated_to_cam = find_scene_orientation(poses_cam_to_world)
    return sky_direction, rotated_to_cam, "traj"


def get_up_vectors(pose_device_to_world: np.ndarray) -> np.ndarray:
    return pose_device_to_world @ np.array([0.0, -1.0, 0.0, 0.0])


def get_right_vectors(pose_device_to_world: np.ndarray) -> np.ndarray:
    return pose_device_to_world @ np.array([1.0, 0.0, 0.0, 0.0])


def read_traj(traj_path: str) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Parse lowres_wide.traj; return timestamps and cam-to-world 4x4 poses."""
    poses_cam_to_world: List[np.ndarray] = []
    timestamps: List[float] = []
    with open(traj_path, encoding="utf-8") as f:
        for line in f:
            tokens = line.split()
            if len(tokens) != 7:
                continue
            traj_timestamp = float(tokens[0])
            angle_axis = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
            r_w_to_p, _ = cv2.Rodrigues(np.asarray(angle_axis))
            t_w_to_p = np.asarray(
                [float(tokens[4]), float(tokens[5]), float(tokens[6])]
            )
            pose_w_to_p = np.eye(4)
            pose_w_to_p[:3, :3] = r_w_to_p
            pose_w_to_p[:3, 3] = t_w_to_p
            pose_p_to_w = np.linalg.inv(pose_w_to_p)
            timestamps.append(traj_timestamp)
            poses_cam_to_world.append(pose_p_to_w)
    return np.asarray(timestamps, dtype=np.float64), poses_cam_to_world


def find_scene_orientation(
    poses_cam_to_world: List[np.ndarray],
) -> Tuple[SkyDirection, np.ndarray]:
    """Infer per-scene sky direction and rotated_to_cam (4x4)."""
    if poses_cam_to_world:
        up_vector = sum(get_up_vectors(p) for p in poses_cam_to_world) / len(
            poses_cam_to_world
        )
        right_vector = sum(get_right_vectors(p) for p in poses_cam_to_world) / len(
            poses_cam_to_world
        )
        up_world = np.array([[0.0], [0.0], [1.0], [0.0]])
    else:
        up_vector = np.array([[0.0], [-1.0], [0.0], [0.0]])
        right_vector = np.array([[1.0], [0.0], [0.0], [0.0]])
        up_world = np.array([[0.0], [0.0], [1.0], [0.0]])

    device_up_to_world_up_angle = (
        np.arccos(np.clip(np.dot(np.transpose(up_world), up_vector), -1.0, 1.0)).item()
        * 180.0
        / np.pi
    )
    device_right_to_world_up_angle = (
        np.arccos(
            np.clip(np.dot(np.transpose(up_world), right_vector), -1.0, 1.0)
        ).item()
        * 180.0
        / np.pi
    )

    up_closest_to_90 = abs(device_up_to_world_up_angle - 90.0) < abs(
        device_right_to_world_up_angle - 90.0
    )
    if up_closest_to_90:
        if device_right_to_world_up_angle > 90.0:
            sky_direction: SkyDirection = "LEFT"
            cam_to_rotated_q = Rotation.from_euler("z", math.pi / 2.0)
        else:
            sky_direction = "RIGHT"
            cam_to_rotated_q = Rotation.from_euler("z", -math.pi / 2.0)
    else:
        if device_up_to_world_up_angle > 90.0:
            sky_direction = "DOWN"
            cam_to_rotated_q = Rotation.from_euler("z", math.pi)
        else:
            sky_direction = "UP"
            cam_to_rotated_q = Rotation.from_matrix(np.eye(3))

    cam_to_rotated = np.eye(4)
    cam_to_rotated[:3, :3] = cam_to_rotated_q.as_matrix()
    rotated_to_cam = np.linalg.inv(cam_to_rotated)
    return sky_direction, rotated_to_cam


def find_frame_orientation(
    pose_cam_to_world: np.ndarray,
) -> Tuple[SkyDirection, np.ndarray]:
    """Per-frame sky direction from a single cam-to-world pose (Z-up heuristic)."""
    return find_scene_orientation([pose_cam_to_world])


def nearest_traj_pose(
    timestamps: np.ndarray,
    poses_cam_to_world: List[np.ndarray],
    query_ts: float,
    max_delta_s: float = 0.05,
) -> Tuple[np.ndarray, float, float]:
    """Return (pose, traj_timestamp, |delta_ms|) for the closest traj sample."""
    if len(timestamps) == 0:
        raise ValueError("empty trajectory")
    idx = int(np.argmin(np.abs(timestamps - query_ts)))
    delta_s = float(abs(timestamps[idx] - query_ts))
    if delta_s > max_delta_s:
        raise ValueError(
            f"traj nearest delta {delta_s * 1000:.1f}ms > {max_delta_s * 1000:.1f}ms"
        )
    return poses_cam_to_world[idx], float(timestamps[idx]), delta_s * 1000.0


def build_per_frame_sky_table(
    timestamps: np.ndarray,
    poses_cam_to_world: List[np.ndarray],
    camera_names: Sequence[str],
    scene_id: str,
    max_delta_s: float = 0.05,
) -> Dict[str, Dict[str, object]]:
    """Map EmbodiedScan camera name -> sky_direction + rotated_to_cam + traj meta."""
    table: Dict[str, Dict[str, object]] = {}
    for camera in camera_names:
        ts_str = camera[len(f"{scene_id}_") :] if camera.startswith(f"{scene_id}_") else camera
        query_ts = float(ts_str)
        pose, traj_ts, delta_ms = nearest_traj_pose(
            timestamps, poses_cam_to_world, query_ts, max_delta_s=max_delta_s
        )
        sky, rotated_to_cam = find_frame_orientation(pose)
        table[camera] = {
            "sky_direction": sky,
            "rotated_to_cam": rotated_to_cam.tolist(),
            "traj_ts": traj_ts,
            "traj_delta_ms": delta_ms,
        }
    return table


def manifest_rotated_to_cam(
    manifest: Dict[str, object],
    camera: str,
) -> np.ndarray:
    """Resolve rotated_to_cam for a camera from scene or per-frame manifest."""
    if manifest.get("sky_granularity") == "frame":
        frames = manifest.get("frames") or {}
        if camera in frames:
            return np.asarray(frames[camera]["rotated_to_cam"], dtype=np.float64)
    return np.asarray(manifest["rotated_to_cam"], dtype=np.float64)


def load_pincam(path: str) -> Tuple[float, float, float, float, float, float]:
    w, h, fx, fy, cx, cy = np.loadtxt(path)
    return float(w), float(h), float(fx), float(fy), float(cx), float(cy)


def transform_pincam(
    w: float,
    h: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    sky_direction: SkyDirection,
) -> Tuple[float, float, float, float, float, float]:
    if sky_direction in ("LEFT", "RIGHT"):
        return h, w, fy, fx, cy, cx
    return w, h, fx, fy, cx, cy


def pincam_to_cam2img(
    w: float,
    h: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[0, 0] = fx
    mat[1, 1] = fy
    mat[0, 2] = cx
    mat[1, 2] = cy
    return mat


def transform_cam2img(
    cam2img: np.ndarray,
    sky_direction: SkyDirection,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Apply sky-direction correction to a 4x4 cam2img (pincam-style swap)."""
    mat = np.asarray(cam2img, dtype=np.float64)
    if mat.shape == (3, 3):
        mat4 = np.eye(4)
        mat4[:3, :3] = mat
        mat = mat4
    w, h = float(image_width), float(image_height)
    fx, fy, cx, cy = mat[0, 0], mat[1, 1], mat[0, 2], mat[1, 2]
    tw, th, tfx, tfy, tcx, tcy = transform_pincam(
        w, h, fx, fy, cx, cy, sky_direction
    )
    return pincam_to_cam2img(tw, th, tfx, tfy, tcx, tcy)


def apply_extrinsic_sky_correction(
    extrinsic: np.ndarray,
    rotated_to_cam: np.ndarray,
) -> np.ndarray:
    """Map sensor cam2world extrinsic to the upright display camera frame."""
    ext = np.asarray(extrinsic, dtype=np.float64)
    return ext @ rotated_to_cam


def rotate_rgb_image(image: Image.Image, sky_direction: SkyDirection) -> Image.Image:
    if sky_direction == "RIGHT":
        return image.transpose(Image.Transpose.ROTATE_90)
    if sky_direction == "LEFT":
        return image.transpose(Image.Transpose.ROTATE_270)
    if sky_direction == "DOWN":
        return image.transpose(Image.Transpose.ROTATE_180)
    return image


def rotate_depth_image(depth: np.ndarray, sky_direction: SkyDirection) -> np.ndarray:
    if sky_direction == "RIGHT":
        return cv2.rotate(depth, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if sky_direction == "LEFT":
        return cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE)
    if sky_direction == "DOWN":
        return cv2.rotate(depth, cv2.ROTATE_180)
    return depth


DepthSource = str  # "highres_depth" | "lowres_depth"


def find_depth_asset(
    highres_dirs: List[str],
    lowres_dirs: List[str],
    scene_id: str,
    timestamp: str,
    max_delta_s: float = 0.05,
) -> Tuple[str | None, DepthSource | None]:
    """Match depth PNG to the same frame timestamp as vga_wide (CUT3R pairing).

    Prefer ``highres_depth`` (``preprocess_arkitscenes_highres.py``), else
    ``lowres_depth`` at identical ``{scene_id}_{timestamp}.png`` basename
    (``preprocess_arkitscenes.py``).
    """
    path = find_asset_file(
        highres_dirs, scene_id, timestamp, ".png", max_delta_s=max_delta_s
    )
    if path:
        return path, "highres_depth"
    path = find_asset_file(
        lowres_dirs, scene_id, timestamp, ".png", max_delta_s=max_delta_s
    )
    if path:
        return path, "lowres_depth"
    return None, None


def resize_depth_to_rotated_vga(
    depth: np.ndarray,
    target_size: Tuple[int, int],
    depth_source: DepthSource,
) -> np.ndarray:
    """Scale depth to rotated ``vga_wide`` size (W, H). No intrinsics reprojection.

    Interpolation matches CUT3R: ``INTER_NEAREST`` for highres_depth,
    ``INTER_NEAREST_EXACT`` for lowres_depth.
    """
    interp = (
        cv2.INTER_NEAREST
        if depth_source == "highres_depth"
        else cv2.INTER_NEAREST_EXACT
    )
    return cv2.resize(depth, target_size, interpolation=interp)


def find_asset_file(
    search_dirs: List[str],
    scene_id: str,
    timestamp: str,
    suffix: str,
    max_delta_s: float = 0.05,
) -> str | None:
    """Find {scene_id}_{timestamp}{suffix} with ±1ms then nearest-neighbor fallback."""
    candidates = [f"{scene_id}_{timestamp}{suffix}"]
    try:
        ts = float(timestamp)
        for delta in (-0.001, 0.001):
            candidates.append(f"{scene_id}_{ts + delta:.3f}{suffix}")
    except ValueError:
        ts = None
    for directory in search_dirs:
        if not directory or not os.path.isdir(directory):
            continue
        names = set(os.listdir(directory))
        for name in candidates:
            if name in names:
                return os.path.join(directory, name)
    if ts is None:
        return None
    prefix = f"{scene_id}_"
    best_path: str | None = None
    best_delta = max_delta_s + 1.0
    for directory in search_dirs:
        if not directory or not os.path.isdir(directory):
            continue
        for name in os.listdir(directory):
            if not name.startswith(prefix) or not name.endswith(suffix):
                continue
            stem = name[len(prefix) : -len(suffix)]
            try:
                file_ts = float(stem)
            except ValueError:
                continue
            delta = abs(file_ts - ts)
            if delta < best_delta:
                best_delta = delta
                best_path = os.path.join(directory, name)
    if best_path is not None and best_delta <= max_delta_s:
        return best_path
    return None
