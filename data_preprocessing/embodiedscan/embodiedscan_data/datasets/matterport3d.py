import os
import pickle
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np

from embodiedscan_data.datasets import register
from embodiedscan_data.datasets.base import DatasetConfig


def _camera_name_from_matterport_path(img_path: str) -> str:
    """Match EmbodiedScanExplorer naming for matterport3d frames."""
    base = img_path.split("/")[-1]
    return base[:-8] + base[-7:-4]


@lru_cache(maxsize=4)
def _matterport_scene_cameras_index(project_root: str) -> Dict[str, Tuple[str, ...]]:
    """Map sample_idx -> camera names listed in EmbodiedScan v1 pkls."""
    ann_rel_paths = (
        "data/embodiedscan_infos_train.pkl",
        "data/embodiedscan_infos_val.pkl",
        "data/embodiedscan_infos_test.pkl",
    )
    index: Dict[str, Tuple[str, ...]] = {}
    for rel in ann_rel_paths:
        path = os.path.join(project_root, rel.replace("/", os.sep))
        if not os.path.isfile(path):
            continue
        with open(path, "rb") as f:
            payload = pickle.load(f)
        for entry in payload["data_list"]:
            sample_idx = entry.get("sample_idx", "")
            if not sample_idx.startswith("matterport3d/"):
                continue
            names = tuple(
                sorted(
                    _camera_name_from_matterport_path(cam["img_path"])
                    for cam in entry.get("images", [])
                )
            )
            index[sample_idx] = names
    return index


@register
class Matterport3DConfig(DatasetConfig):
    name = "matterport3d"
    dataset_key = "matterport3d"
    depth_scale = 4000
    ann_files = [
        "data/embodiedscan_infos_train.pkl",
        "data/embodiedscan_infos_val.pkl",
        "data/embodiedscan_infos_test.pkl",
    ]

    def list_scenes(self, data_root: str) -> List[str]:
        mp3d_dir = os.path.join(data_root, "matterport3d")
        if not os.path.isdir(mp3d_dir):
            return []
        scenes = []
        for building_id in sorted(os.listdir(mp3d_dir)):
            building_dir = os.path.join(mp3d_dir, building_id)
            if not os.path.isdir(building_dir):
                continue
            region_dir = os.path.join(building_dir, "region_segmentations")
            if not os.path.isdir(region_dir):
                continue
            if not os.path.isdir(os.path.join(building_dir, "matterport_color_images")):
                continue
            for f in sorted(os.listdir(region_dir)):
                if f.endswith(".ply"):
                    region_name = f.split(".")[0]
                    scenes.append(f"matterport3d/{building_id}/{region_name}")
        return scenes

    def list_cameras(self, data_root: str, scene: str) -> List[str]:
        project_root = os.path.dirname(os.path.abspath(data_root))
        index = _matterport_scene_cameras_index(project_root)
        cameras = list(index.get(scene, ()))
        if not cameras:
            return []
        return [
            cam for cam in cameras
            if not self.skip_camera(data_root, scene, cam)
        ]

    def get_scene_id(self, scene: str) -> str:
        parts = scene.split("/")
        return f"{parts[1]}__{parts[2]}"

    def get_intrinsic(self, data_root: str, scene: str, camera: str) -> str:
        parts = scene.split("/")
        building_id = parts[1]
        suffix = camera[-3:]
        prefix = camera[:-3]
        intrinsic_filename = f"{prefix}intrinsics_{suffix[0]}.txt"
        intrinsic_path = os.path.join(
            data_root, "matterport3d", building_id, "matterport_camera_intrinsics", intrinsic_filename
        )
        output_path = intrinsic_path.replace(".txt", "_matrix.txt")
        if not os.path.exists(output_path):
            self._parse_intrinsic(intrinsic_path, output_path)
        return os.path.relpath(output_path, data_root)

    def _parse_intrinsic(self, intrinsic_path: str, output_path: str) -> None:
        with open(intrinsic_path, "r") as f:
            values = [float(x) for x in f.read().split()]
        fx, fy, cx, cy = values[2], values[3], values[4], values[5]
        matrix = np.array([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        with open(output_path, "w") as f:
            for row in matrix:
                f.write(" ".join(f"{v:.6f}" for v in row) + "\n")

    def get_depth_map(self, data_root: str, scene: str, camera: str) -> Optional[str]:
        parts = scene.split("/")
        building_id = parts[1]
        suffix = camera[-3:]
        prefix = camera[:-3]
        return os.path.join("matterport3d", building_id, "matterport_depth_images", f"{prefix}d{suffix}.png")

    def skip_scene(self, data_root: str, scene: str) -> bool:
        parts = scene.split("/")
        building_id = parts[1]
        bd = os.path.join(data_root, "matterport3d", building_id)
        return (not os.path.isdir(os.path.join(bd, "region_segmentations"))
                or not os.path.isdir(os.path.join(bd, "matterport_color_images")))

    def skip_camera(self, data_root: str, scene: str, camera: str) -> bool:
        parts = scene.split("/")
        building_id = parts[1]
        suffix = camera[-3:]
        prefix = camera[:-3]
        intrinsic = os.path.join(data_root, "matterport3d", building_id,
                                  "matterport_camera_intrinsics", f"{prefix}intrinsics_{suffix[0]}.txt")
        depth = os.path.join(data_root, "matterport3d", building_id,
                              "matterport_depth_images", f"{prefix}d{suffix}.png")
        return not os.path.exists(intrinsic) or not os.path.exists(depth)
