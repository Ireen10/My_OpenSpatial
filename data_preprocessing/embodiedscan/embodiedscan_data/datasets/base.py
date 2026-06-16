import os
from abc import ABC, abstractmethod
from typing import List, Optional


class DatasetConfig(ABC):
    name: str
    dataset_key: str
    depth_scale: int
    is_metric_depth: bool = True
    ann_files: List[str]

    @abstractmethod
    def list_scenes(self, data_root: str) -> List[str]:
        """Return all scene paths for this dataset."""

    @abstractmethod
    def list_cameras(self, data_root: str, scene: str) -> List[str]:
        """Return all camera names for a given scene."""

    @abstractmethod
    def get_scene_id(self, scene: str) -> str:
        """Return the canonical scene_id."""

    @abstractmethod
    def get_intrinsic(self, data_root: str, scene: str, camera: str) -> str:
        """Parse intrinsic matrix, return saved file path relative to data_root."""

    def get_depth_map(self, data_root: str, scene: str, camera: str) -> Optional[str]:
        """Return depth map path relative to data_root. None if Explorer provides it."""
        return None

    def post_process(self, info: dict, data_root: str, scene: str, camera: str) -> dict:
        """Dataset-specific post-processing. Default: return unchanged."""
        return info

    def skip_scene(self, data_root: str, scene: str) -> bool:
        """Return True to skip this scene. Default: False."""
        return False

    def skip_camera(self, data_root: str, scene: str, camera: str) -> bool:
        """Return True to skip this camera. Default: False."""
        return False

    def get_save_path(self, data_root: str, scene: str) -> str:
        """Return directory where Explorer writes pose/intrinsic files."""
        return os.path.join(data_root, scene)

    def get_explorer_kwargs(self, data_root: str) -> dict:
        """Return kwargs for EmbodiedScanExplorer constructor.

        Args:
            data_root: Absolute path to the data directory (e.g., .../EmbodiedScan/data)
        """
        project_root = os.path.dirname(os.path.abspath(data_root))

        explorer_data_root = {
            "scannet": None, "3rscan": None,
            "matterport3d": None, "arkitscenes": None,
        }
        dataset_dir_map = {
            "scannet": "data/scannet", "3rscan": "data/3rscan",
            "matterport3d": "data/matterport3d", "arkitscenes": "data/arkitscenes",
        }
        explorer_data_root[self.dataset_key] = os.path.join(project_root, dataset_dir_map[self.dataset_key])

        abs_ann_files = [os.path.join(project_root, f) for f in self.ann_files]

        return {
            "data_root": explorer_data_root,
            "ann_file": abs_ann_files,
            "verbose": False,
        }
