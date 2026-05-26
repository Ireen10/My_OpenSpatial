"""Camera projection and coordinate transform utilities."""

import numpy as np


def intrinsic_matrix_3x3(intrinsic):
    """Upper-left 3x3 from 4x4 or 3x3 intrinsic."""
    k = np.asarray(intrinsic, dtype=np.float64)
    return k[:3, :3] if k.shape[0] >= 3 else k


def compute_fov_from_intrinsic(intrinsic, img_dim):
    """Compute horizontal and vertical field of view from intrinsic matrix.

    Args:
        intrinsic: 4x4 (or 3x3) intrinsic matrix.
        img_dim: (width, height).

    Returns:
        (fov_h, fov_v) in degrees.
    """
    hfov = 2 * np.arctan2(img_dim[0], 2 * intrinsic[0, 0])
    vfov = 2 * np.arctan2(img_dim[1], 2 * intrinsic[1, 1])
    return np.degrees(hfov), np.degrees(vfov)


def transform_points_camera_to_world(camera_coords, pose):
    """Transform Nx3 camera-frame points to world frame.

    Args:
        camera_coords: Nx3 np.ndarray in camera frame.
        pose: 4x4 camera-to-world matrix.

    Returns:
        Nx3 np.ndarray in world frame.
    """
    num_points = camera_coords.shape[0]
    homogeneous_coords = np.hstack((camera_coords, np.ones((num_points, 1))))
    world_coords = (pose @ homogeneous_coords.T).T
    return world_coords[:, :3]


def transform_points_world_to_camera(world_coords, pose):
    """Transform Nx3 world-frame points to camera frame.

    Args:
        world_coords: Nx3 np.ndarray in world frame.
        pose: 4x4 camera-to-world matrix.

    Returns:
        Nx3 np.ndarray in camera frame.
    """
    inv_pose = np.linalg.inv(pose)
    num_points = world_coords.shape[0]
    homogeneous_coords = np.hstack((world_coords, np.ones((num_points, 1))))
    camera_coords = (inv_pose @ homogeneous_coords.T).T
    return camera_coords[:, :3]


def backproject_depth_to_3d(depth, img_dim, intrinsic, pose=None):
    """Back-project a depth map to 3D coordinates.

    Args:
        depth: HxW depth array.
        img_dim: (width, height).
        intrinsic: 4x4 (or 3x3) intrinsic matrix.
        pose: 4x4 camera-to-world matrix. If None, returns camera coordinates.

    Returns:
        np.ndarray of shape (H*W, 3).
    """
    w, h = img_dim
    xmap = np.tile(np.arange(w), (h, 1))
    ymap = np.tile(np.arange(h).reshape(-1, 1), (1, w))

    pts0 = (xmap - intrinsic[0][2]) * depth / intrinsic[0][0]
    pts1 = (ymap - intrinsic[1][2]) * depth / intrinsic[1][1]

    pts_cam = np.stack([pts0, pts1, depth], axis=-1)  # (H, W, 3)

    if pose is None:
        return pts_cam.reshape(-1, 3)

    # Camera → World
    pts_h = np.concatenate([pts_cam, np.ones((*pts_cam.shape[:2], 1))], axis=-1)
    pts_world = (pose @ pts_h.reshape(-1, 4).T).T[:, :3]
    return pts_world


def project_points_3d_to_2d(world_to_camera, points_3d, intrinsic):
    """Project 3D world coordinates to 2D pixel coordinates.

    Args:
        world_to_camera: 4x4 extrinsic matrix (inverse of camera-to-world).
        points_3d: Nx3 array.
        intrinsic: 4x4 intrinsic matrix.

    Returns:
        Nx2 np.ndarray of (u, v) pixel coordinates.
    """
    ones = np.ones((points_3d.shape[0], 1))
    homogeneous_points_3d = np.hstack((points_3d, ones))
    points_camera = world_to_camera @ homogeneous_points_3d.T
    points_2d_homogeneous = intrinsic @ points_camera
    u_v = points_2d_homogeneous[:2] / points_2d_homogeneous[2]
    return u_v.T
