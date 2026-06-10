"""3D bounding box geometry utilities."""

import json

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation

# Open-ended 3D grounding prompt: roll, pitch, yaw in (-pi, pi] radians.
EULER_GROUNDING_ORDER = "zxy"

# Relative size: skip pairs when max(D_diag)/min(D_diag) < this threshold.
RELATIVE_SIZE_DIAG_RATIO_MIN = 1.2


def box_3d_diag_extent(box_3d_world) -> float:
    """Spatial diagonal of 3D box extent: D_diag = sqrt(L^2 + W^2 + H^2)."""
    L, W, H = box_3d_world[3:6]
    return float(np.sqrt(L * L + W * W + H * H))


def wrap_angle_to_pi(angle: float) -> float:
    """Wrap one angle to (-pi, pi] radians."""
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def normalize_euler_angles(rotation, euler_order: str = EULER_GROUNDING_ORDER) -> list:
    """Return (roll, pitch, yaw) list with each component in (-pi, pi]."""
    rot = SciRotation.from_euler(euler_order, list(rotation), degrees=False)
    euler = rot.as_euler(euler_order, degrees=False)
    return [wrap_angle_to_pi(float(a)) for a in euler]


def format_bbox_3d_for_grounding(box_params, euler_order: str = EULER_GROUNDING_ORDER) -> list:
    """9-param camera-frame box with center/size rounded and euler in (-pi, pi]."""
    if box_params is None or len(box_params) < 9:
        return None
    center = [round(float(v), 2) for v in box_params[:3]]
    size = [round(float(v), 2) for v in box_params[3:6]]
    euler = [round(v, 2) for v in normalize_euler_angles(box_params[6:9], euler_order)]
    return center + size + euler


def format_grounding_answer_json(entries: list) -> str:
    """
    Serialize GT as a JSON array: [{"label": str, "bbox_3d": [9 floats]}, ...].

    Matches grounding_json_output_instruction (double-quoted keys, outer ``[]``).
    """
    payload = [
        {"label": item["label"], "bbox_3d": item["bbox_3d"]}
        for item in entries
        if item.get("bbox_3d") is not None
    ]
    return json.dumps(payload, ensure_ascii=False)


def compute_box_3d_points(size):
    """Compute 8 corner points of an axis-aligned box centered at the origin.

    Args:
        size: (3,) array — [xl, yl, zl].

    Returns:
        np.ndarray of shape (8, 3).
    """
    hs = np.asarray(size) / 2
    return np.array([
        [-hs[0], -hs[1], -hs[2]],
        [-hs[0], -hs[1],  hs[2]],
        [-hs[0],  hs[1],  hs[2]],
        [-hs[0],  hs[1], -hs[2]],
        [ hs[0], -hs[1], -hs[2]],
        [ hs[0], -hs[1],  hs[2]],
        [ hs[0],  hs[1],  hs[2]],
        [ hs[0],  hs[1], -hs[2]],
    ])


def compute_box_3d_corners(center, size, rotation, euler_order='zxy'):
    """Compute 8 world-space corners of an oriented 3D bounding box.

    Args:
        center: (3,) — [x, y, z].
        size: (3,) — [xl, yl, zl].
        rotation: (3,) — euler angles in radians.
        euler_order: rotation convention (default 'zxy').

    Returns:
        np.ndarray of shape (8, 3).
    """
    center = np.asarray(center)
    rot_mat = SciRotation.from_euler(euler_order, list(rotation), degrees=False).as_matrix()
    corners = compute_box_3d_points(size) @ rot_mat.T + center
    return corners


def rotation_matrix_from_box_euler(rotation, euler_order=EULER_GROUNDING_ORDER):
    """3x3 rotation matrix for a 9-param box's euler angles."""
    return SciRotation.from_euler(euler_order, list(rotation), degrees=False).as_matrix()


def shrunk_oriented_box_geometry(box_params, scale_factor=1.0, euler_order=EULER_GROUNDING_ORDER):
    """Center, extent, and rotation for a scaled oriented box from 9 params."""
    center = np.asarray(box_params[:3], dtype=np.float64)
    extent = np.asarray(box_params[3:6], dtype=np.float64) * float(scale_factor)
    rotation = rotation_matrix_from_box_euler(box_params[6:9], euler_order)
    return center, extent, rotation


def oriented_box_volume(extent):
    """Volume of an oriented box given axis lengths (xl, yl, zl)."""
    extent = np.asarray(extent, dtype=np.float64)
    return float(np.prod(np.maximum(extent, 0.0)))


def points_inside_oriented_box(points, center, extent, rotation, eps=1e-6):
    """Boolean mask of N points inside an oriented box (row-vector convention)."""
    points = np.asarray(points, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    extent = np.asarray(extent, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    local = (points - center) @ rotation
    half = extent * 0.5
    return np.all(np.abs(local) <= half + eps, axis=1)


def obb_volume_from_points(points, min_points=4):
    """PCA-oriented bounding box volume for Nx3 points (np.cov + eigh)."""
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] < min_points:
        return 0.0
    centered = points - points.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    if cov.ndim == 0 or not np.all(np.isfinite(cov)):
        return 0.0
    _, axes = np.linalg.eigh(cov)
    local = centered @ axes
    extent = local.max(axis=0) - local.min(axis=0)
    return oriented_box_volume(extent)


def compute_box_3d_corners_from_params(box_params, euler_order='zxy'):
    """Compute 8 corners from a 9-value box parameter list.

    Args:
        box_params: [cx, cy, cz, xl, yl, zl, roll, pitch, yaw].

    Returns:
        np.ndarray of shape (8, 3).
    """
    return compute_box_3d_corners(
        box_params[:3], box_params[3:6], box_params[6:9], euler_order)


def convert_box_3d_world_to_camera(box_params, pose, euler_order='zxy'):
    """Convert a 9-param world-frame 3D box to camera frame.

    Args:
        box_params: [x, y, z, xl, yl, zl, roll, pitch, yaw].
        pose: 4x4 camera-to-world matrix.
        euler_order: Euler convention (default 'zxy').

    Returns:
        9-element list in camera frame, or None if input is invalid.
    """
    if box_params is None or len(box_params) < 9:
        return None
    center = np.array(box_params[:3])
    size = np.array(box_params[3:6])
    rotation = box_params[6:9]

    rot_mat = SciRotation.from_euler(euler_order, list(rotation), degrees=False).as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = rot_mat
    transform[:3, 3] = center

    cam_transform = np.linalg.inv(pose) @ transform
    cam_center = cam_transform[:3, 3]
    cam_euler = normalize_euler_angles(
        SciRotation.from_matrix(cam_transform[:3, :3]).as_euler(euler_order, degrees=False),
        euler_order,
    )
    return list(cam_center) + list(size) + cam_euler


def check_box_2d_overlap(box1_xy, box2_xy):
    """Check if two 2D polygon projections overlap or are close.

    Args:
        box1_xy: Nx2 array of 2D polygon vertices.
        box2_xy: Mx2 array of 2D polygon vertices.

    Returns:
        True if polygons intersect or are within 50% of the larger box's longest edge.
    """
    from shapely.geometry import Polygon

    poly1 = Polygon(box1_xy)
    poly2 = Polygon(box2_xy)

    intersects_flag = poly1.intersects(poly2)

    distance = poly1.distance(poly2)
    max_size1 = np.max(np.linalg.norm(
        box1_xy - np.roll(box1_xy, shift=-1, axis=0), axis=1))
    max_size2 = np.max(np.linalg.norm(
        box2_xy - np.roll(box2_xy, shift=-1, axis=0), axis=1))
    max_size = max(max_size1, max_size2)
    distance_flag = distance < max_size * 0.5

    return intersects_flag or distance_flag


def check_box_3d_vertical_overlap(box_3d_world_list):
    """Check if any pair of 3D boxes has overlapping XY projections.

    Args:
        box_3d_world_list: list of 9-param boxes [cx,cy,cz,xl,yl,zl,r,p,y].

    Returns:
        True if ANY pair overlaps (has vertical overlap).
    """
    for i, box1 in enumerate(box_3d_world_list):
        for j in range(i + 1, len(box_3d_world_list)):
            corners1_xy = compute_box_3d_corners_from_params(box1)[:, :2]
            corners2_xy = compute_box_3d_corners_from_params(box_3d_world_list[j])[:, :2]
            if check_box_2d_overlap(corners1_xy, corners2_xy):
                return True
    return False
