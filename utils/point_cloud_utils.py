"""Point cloud distance, cleaning, loading, and debug I/O utilities."""

import threading

import numpy as np
import open3d as o3d

# Open3D C++ backends are not reliably thread-safe; serialize heavy calls.
_OPEN3D_LOCK = threading.Lock()


def point_cloud_nn_distances(pcd1, pcd2) -> np.ndarray:
    """Per-point nearest-neighbor distances from pcd1 queries to pcd2 targets."""
    with _OPEN3D_LOCK:
        return np.asarray(pcd1.compute_point_cloud_distance(pcd2))


def compute_point_cloud_distance(pcd1, pcd2):
    """Compute minimum distance between two Open3D point clouds.

    Args:
        pcd1, pcd2: open3d.geometry.PointCloud objects.

    Returns:
        float: minimum nearest-neighbor distance, or inf if empty.
    """
    distances = point_cloud_nn_distances(pcd1, pcd2)
    return float(np.min(distances)) if len(distances) else float("inf")


def format_distance_readable(distance_meters):
    """Format distance in meters to human-readable string (cm or m).

    Args:
        distance_meters: distance in meters.

    Returns:
        Formatted string like "12.34 centimeters" or "1.23 meters".
    """
    if distance_meters < 1:
        return f"{round(distance_meters * 100, 2)} centimeters"
    return f"{round(distance_meters, 2)} meters"


def clean_point_cloud(cloud, nb_neighbors=10, std_ratio=1.0):
    """Remove statistical outliers from a point cloud.

    Args:
        cloud: open3d.geometry.PointCloud.
        nb_neighbors: number of neighbors for statistical analysis.
        std_ratio: standard deviation ratio threshold.

    Returns:
        Cleaned open3d.geometry.PointCloud.
    """
    with _OPEN3D_LOCK:
        _, ind = cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        return cloud.select_by_index(ind)


def load_point_clouds(pointcloud_paths, nb_neighbors=5, std_ratio=2.0):
    """Load and clean point clouds from file paths.

    Args:
        pointcloud_paths: list of .pcd file paths (or nested list).
        nb_neighbors: neighbors for outlier removal.
        std_ratio: standard deviation ratio for outlier removal.

    Returns:
        List of cleaned open3d.geometry.PointCloud objects.
    """
    if len(pointcloud_paths) == 1 and isinstance(pointcloud_paths[0], list):
        pointcloud_paths = pointcloud_paths[0]

    restored = []
    for path in pointcloud_paths:
        with _OPEN3D_LOCK:
            ori_pcd = o3d.io.read_point_cloud(path)
        cleaned = clean_point_cloud(ori_pcd, nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        restored.append(cleaned)
    return restored


def write_point_cloud(points, colors, out_filename, fmt="obj"):
    """Write colored point cloud to file.

    Args:
        points: Nx3 array of 3D coordinates.
        colors: Nx3 array of RGB values (0-255 int or 0-1 float).
        out_filename: output file path.
        fmt: output format — "obj" or "ply".
    """
    if fmt == "obj":
        with open(out_filename, 'w') as f:
            for i in range(points.shape[0]):
                c = colors[i]
                f.write('v %f %f %f %d %d %d\n' % (
                    points[i, 0], points[i, 1], points[i, 2],
                    c[0], c[1], c[2]))
    elif fmt == "ply":
        n = points.shape[0]
        with open(out_filename, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {n}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            for i in range(n):
                c = colors[i]
                f.write('%f %f %f %d %d %d\n' % (
                    points[i, 0], points[i, 1], points[i, 2],
                    int(c[0]), int(c[1]), int(c[2])))
    else:
        raise ValueError(f"Unsupported format: {fmt}. Use 'obj' or 'ply'.")
