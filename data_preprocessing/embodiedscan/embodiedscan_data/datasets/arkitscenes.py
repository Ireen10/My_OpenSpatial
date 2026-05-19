import json
import os
import pickle
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np

from embodiedscan_data.arkit_geometry import (
    apply_extrinsic_sky_correction,
    manifest_rotated_to_cam,
)
from embodiedscan_data.datasets import register
from embodiedscan_data.datasets.base import DatasetConfig

# Matches EmbodiedScan data/README.md and upstream EmbodiedScanExplorer.
ARKIT_SPLITS = ("Training", "Validation")

RGB_SUBDIR = "vga_wide"
DEPTH_SUBDIR = "vga_depth"
INTR_SUBDIR = "vga_wide_intrinsics"
SCENE_MANIFEST = ".arkit_scene.json"

# extract --arkit-asset-mode (ARKitScenes only; see ARKitScenesConfig.asset_mode)
ARKIT_ASSET_MODES = ("auto", "vga", "lowres")


def _camera_name_from_arkit_path(img_path: str) -> str:
    """Match EmbodiedScanExplorer naming for ARKitScenes frames."""
    return img_path.split("/")[-1][:-4]


@lru_cache(maxsize=4)
def _arkit_scene_cameras_index(project_root: str) -> Dict[str, Tuple[str, ...]]:
    """Map sample_idx -> camera names listed in EmbodiedScan v2 pkls."""
    ann_rel_paths = (
        "embodiedscan-v2/embodiedscan_infos_train.pkl",
        "embodiedscan-v2/embodiedscan_infos_val.pkl",
        "embodiedscan-v2/embodiedscan_infos_test.pkl",
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
            if not sample_idx.startswith("arkitscenes/"):
                continue
            names = tuple(
                sorted(
                    _camera_name_from_arkit_path(cam["img_path"])
                    for cam in entry.get("images", [])
                )
            )
            index[sample_idx] = names
    return index


@register
class ARKitScenesConfig(DatasetConfig):
    name = "arkitscenes"
    dataset_key = "arkitscenes"
    depth_scale = 1000
    ann_files = [
        "embodiedscan-v2/embodiedscan_infos_train.pkl",
        "embodiedscan-v2/embodiedscan_infos_val.pkl",
        "embodiedscan-v2/embodiedscan_infos_test.pkl",
    ]

    def __init__(self) -> None:
        self.asset_mode = "auto"

    @staticmethod
    def _parse_scene(scene: str) -> Tuple[str, str]:
        """Return (split, scene_id) from e.g. arkitscenes/Training/40753679."""
        parts = scene.split("/")
        if len(parts) == 3 and parts[0] == "arkitscenes" and parts[1] in ARKIT_SPLITS:
            return parts[1], parts[2]
        raise ValueError(
            f"Invalid ARKit scene path: {scene!r}. "
            f"Expected arkitscenes/<Training|Validation>/<scene_id>."
        )

    def _frames_root(self, data_root: str, scene: str) -> str:
        split, scene_id = self._parse_scene(scene)
        return os.path.join(
            data_root, "arkitscenes", split, scene_id, f"{scene_id}_frames"
        )

    def _frames_subdir(self, data_root: str, scene: str, subdir: str) -> str:
        return os.path.join(self._frames_root(data_root, scene), subdir)

    def _scene_manifest_path(self, data_root: str, scene: str) -> str:
        return os.path.join(self._frames_root(data_root, scene), SCENE_MANIFEST)

    def _is_prepared(self, data_root: str, scene: str) -> bool:
        manifest = self._scene_manifest_path(data_root, scene)
        rgb_dir = self._frames_subdir(data_root, scene, RGB_SUBDIR)
        return os.path.isfile(manifest) and os.path.isdir(rgb_dir)

    def _use_vga_assets(self, data_root: str, scene: str) -> bool:
        """Whether extract reads prepare-arkit vga_wide / vga_depth (vs lowres_*)."""
        if self.asset_mode == "lowres":
            return False
        if self.asset_mode == "vga":
            return self._is_prepared(data_root, scene)
        # auto: prepared vga when present
        return self._is_prepared(data_root, scene)

    def list_scenes(self, data_root: str) -> List[str]:
        arkit_dir = os.path.join(data_root, "arkitscenes")
        if not os.path.isdir(arkit_dir):
            return []
        scenes = []
        for split in ARKIT_SPLITS:
            split_dir = os.path.join(arkit_dir, split)
            if not os.path.isdir(split_dir):
                continue
            for scene_id in sorted(os.listdir(split_dir)):
                scene_key = f"arkitscenes/{split}/{scene_id}"
                if not self.skip_scene(data_root, scene_key):
                    scenes.append(scene_key)
        return scenes

    def list_cameras(self, data_root: str, scene: str) -> List[str]:
        project_root = os.path.dirname(os.path.abspath(data_root))
        index = _arkit_scene_cameras_index(project_root)
        cameras = list(index.get(scene, ()))
        if not cameras:
            return []
        return [cam for cam in cameras if not self.skip_camera(data_root, scene, cam)]

    def skip_camera(self, data_root: str, scene: str, camera: str) -> bool:
        if self._use_vga_assets(data_root, scene):
            rgb_jpg = os.path.join(
                self._frames_subdir(data_root, scene, RGB_SUBDIR), f"{camera}.jpg"
            )
            depth_path = os.path.join(
                self._frames_subdir(data_root, scene, DEPTH_SUBDIR), f"{camera}.png"
            )
            matrix_path = os.path.join(
                self._frames_subdir(data_root, scene, INTR_SUBDIR),
                f"{camera}_matrix.txt",
            )
            return not (
                os.path.isfile(rgb_jpg)
                and os.path.isfile(depth_path)
                and os.path.isfile(matrix_path)
            )

        wide_dir = self._frames_subdir(data_root, scene, "lowres_wide")
        depth_path = os.path.join(
            self._frames_subdir(data_root, scene, "lowres_depth"), f"{camera}.png"
        )
        pincam_path = os.path.join(
            self._frames_subdir(data_root, scene, "lowres_wide_intrinsics"),
            f"{camera}.pincam",
        )
        wide_png = os.path.join(wide_dir, f"{camera}.png")
        wide_jpg = os.path.join(wide_dir, f"{camera}.jpg")
        return (
            not (os.path.exists(wide_png) or os.path.exists(wide_jpg))
            or not os.path.exists(depth_path)
            or not os.path.exists(pincam_path)
        )

    def get_scene_id(self, scene: str) -> str:
        return self._parse_scene(scene)[1]

    def get_intrinsic(self, data_root: str, scene: str, camera: str) -> str:
        if self._use_vga_assets(data_root, scene):
            matrix_path = os.path.join(
                self._frames_subdir(data_root, scene, INTR_SUBDIR),
                f"{camera}_matrix.txt",
            )
            return os.path.relpath(matrix_path, data_root)

        intr_dir = self._frames_subdir(data_root, scene, "lowres_wide_intrinsics")
        pincam_path = os.path.join(intr_dir, f"{camera}.pincam")
        output_path = pincam_path.replace(".pincam", "_matrix.txt")
        if not os.path.exists(output_path):
            self._parse_pincam(pincam_path, output_path)
        return os.path.relpath(output_path, data_root)

    def _parse_pincam(self, pincam_path: str, output_path: str) -> None:
        with open(pincam_path, "r", encoding="utf-8") as f:
            content = f.read()
        values = [float(x) for x in content.split()]
        fx, fy, cx, cy = values[2], values[3], values[4], values[5]
        matrix = np.array([
            [fx, 0, cx, 0],
            [0, fy, cy, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for row in matrix:
                f.write(" ".join(f"{v:.6f}" for v in row) + "\n")

    def get_depth_map(self, data_root: str, scene: str, camera: str) -> Optional[str]:
        split, scene_id = self._parse_scene(scene)
        if self._use_vga_assets(data_root, scene):
            return os.path.join(
                "arkitscenes",
                split,
                scene_id,
                f"{scene_id}_frames",
                DEPTH_SUBDIR,
                f"{camera}.png",
            )
        return os.path.join(
            "arkitscenes",
            split,
            scene_id,
            f"{scene_id}_frames",
            "lowres_depth",
            f"{camera}.png",
        )

    def skip_scene(self, data_root: str, scene: str) -> bool:
        try:
            self._parse_scene(scene)
        except ValueError:
            return True
        if self.asset_mode == "vga":
            return not self._is_prepared(data_root, scene)
        if self._use_vga_assets(data_root, scene):
            return False
        frames_dir = self._frames_subdir(data_root, scene, "lowres_wide")
        return not os.path.isdir(frames_dir)

    def get_save_path(self, data_root: str, scene: str) -> str:
        if self._use_vga_assets(data_root, scene):
            return self._frames_subdir(data_root, scene, RGB_SUBDIR)
        return self._frames_subdir(data_root, scene, "lowres_wide")

    def post_process(self, info: dict, data_root: str, scene: str, camera: str) -> dict:
        """Point to prepared vga assets and sync pose/intrinsic with sky correction."""
        if not self._use_vga_assets(data_root, scene):
            return info

        data_root = os.path.abspath(data_root)
        manifest_path = self._scene_manifest_path(data_root, scene)
        with open(manifest_path, encoding="utf-8") as f:
            meta = json.load(f)
        apply_sky = meta.get("apply_sky_correction", True)
        rotated_to_cam = manifest_rotated_to_cam(meta, camera)

        rgb_abs = os.path.join(
            self._frames_subdir(data_root, scene, RGB_SUBDIR), f"{camera}.jpg"
        )
        depth_abs = os.path.join(
            self._frames_subdir(data_root, scene, DEPTH_SUBDIR), f"{camera}.png"
        )
        if os.path.isfile(rgb_abs):
            info["image"] = os.path.relpath(rgb_abs, data_root)
        if os.path.isfile(depth_abs):
            info["depth_map"] = os.path.relpath(depth_abs, data_root)

        pose_abs = os.path.join(
            self._frames_subdir(data_root, scene, RGB_SUBDIR), f"{camera}_pose.txt"
        )
        axis_abs = os.path.join(
            self._frames_subdir(data_root, scene, RGB_SUBDIR),
            f"{camera}_axis_align_matrix.txt",
        )
        if os.path.isfile(pose_abs) and apply_sky:
            extrinsic = np.loadtxt(pose_abs)
            extrinsic = apply_extrinsic_sky_correction(extrinsic, rotated_to_cam)
            np.savetxt(pose_abs, extrinsic, fmt="%.9f")
        if os.path.isfile(pose_abs):
            info["pose"] = os.path.relpath(pose_abs, data_root)
        if os.path.isfile(axis_abs):
            info["axis_align_matrix"] = os.path.relpath(axis_abs, data_root)

        return info
