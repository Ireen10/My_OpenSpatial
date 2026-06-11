import numpy as np
from PIL import Image
import io
from pathlib import Path
import open3d as o3d
import os

from utils.box_utils import (
    all_box_corners_visible_in_image,
    convert_box_3d_camera_to_world,
    obb_params_from_open3d_pcd,
)
from utils.projection_utils import backproject_depth_to_3d
from utils.image_utils import load_depth_map
from task.base_task import BaseTask


class DepthBackProjecter(BaseTask):
    """Back-project depth maps to per-object 3D point clouds using masks."""

    def __init__(self, args):
        super().__init__(args)
        self.output_dir = self.args.get("output_dir")
        if self.output_dir is None:
            raise ValueError("output_dir must be specified in args.")

    def _load_masks(self, raw_masks):
        """Load masks from file paths or byte dicts into numpy arrays.

        Args:
            raw_masks: list of str (file paths) or list of dict with "bytes" key.

        Returns:
            list of 2D numpy arrays.
        """
        masks = []
        for item in raw_masks:
            if isinstance(item, dict):
                masks.append(np.array(Image.open(io.BytesIO(item["bytes"]))))
            elif isinstance(item, str):
                masks.append(np.array(Image.open(item)))
            else:
                raise ValueError(f"Unsupported mask type: {type(item)}")
        return masks

    @staticmethod
    def _resize_masks_to_depth(masks, depth_shape):
        """Resize masks to match depth map dimensions if needed."""
        h, w = depth_shape
        for i, mask in enumerate(masks):
            if mask.shape != depth_shape:
                masks[i] = np.array(
                    Image.fromarray(mask).resize((w, h), resample=Image.NEAREST)
                )
        return masks

    def _backproject_masks_to_pointclouds(self, depth, intrinsic, masks, img_idx):
        """Back-project masked depth regions to cleaned 3D point clouds.

        Args:
            depth: H x W depth map array.
            intrinsic: 4x4 camera intrinsic matrix.
            masks: list of 2D mask arrays.
            img_idx: index used for naming output files.

        Returns:
            (filepaths, valid_flags, refined_boxes_cam): saved .pcd paths, per-mask
            validity, and camera-frame 9-param OBBs fitted to each cleaned cloud.
        """
        img_dim = depth.shape[::-1]  # (W, H)
        points_3d = backproject_depth_to_3d(depth, img_dim, intrinsic)

        output_dir = os.path.join(self.output_dir, self.args["file_name"], "pointclouds")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        filepaths = []
        valid_flags = []
        refined_boxes_cam = []
        for idx, mask in enumerate(masks):
            masked_pts = points_3d[mask.flatten() > 0]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(masked_pts)

            # Remove statistical outliers
            _, ind = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
            cleaned = pcd.select_by_index(ind)

            if cleaned.is_empty():
                valid_flags.append(False)
                refined_boxes_cam.append(None)
                continue

            valid_flags.append(True)
            refined_boxes_cam.append(obb_params_from_open3d_pcd(cleaned))
            filepath = os.path.join(
                output_dir,
                f"pointcloud_{Path(str(img_idx)).stem}_{idx}.pcd"
            )
            o3d.io.write_point_cloud(filepath, cleaned)
            filepaths.append(filepath)

        return filepaths, valid_flags, refined_boxes_cam

    @staticmethod
    def _filter_by_valid_flags(example, valid_flags):
        """Remove entries corresponding to invalid masks from all list fields."""
        if all(valid_flags):
            return example
        filtered = example.copy()
        for key, value in example.items():
            if isinstance(value, list) and len(value) == len(valid_flags):
                filtered[key] = [v for v, ok in zip(value, valid_flags) if ok]
        return filtered

    def apply_transform(self, example, img_idx):
        """Back-project depth to per-object point clouds.

        Requires: intrinsic, depth_map, depth_scale, masks, obj_tags.
        Populates: pointclouds; updates bboxes_3d_world_coords from refined clouds
        when pose is available and every corner of the original box is in-frame.
        """
        assert "intrinsic" in example, "intrinsic not found in example"
        if "depth_map" not in example:
            raise ValueError("depth_map not found in example")
        if "masks" not in example or "obj_tags" not in example:
            raise ValueError("masks and obj_tags are required")
        if len(example["masks"]) != len(example["obj_tags"]):
            return None, False

        intrinsic = np.loadtxt(example["intrinsic"])
        depth = load_depth_map(example["depth_map"], example["depth_scale"])

        masks = self._load_masks(example["masks"])
        masks = self._resize_masks_to_depth(masks, depth.shape)

        filepaths, valid_flags, refined_boxes_cam = self._backproject_masks_to_pointclouds(
            depth, intrinsic, masks, img_idx
        )

        example["pointclouds"] = filepaths

        if "bboxes_3d_world_coords" in example and "pose" in example:
            pose = np.loadtxt(example["pose"])
            img_dim = depth.shape[::-1]
            refined_boxes_world = []
            for i, cam_box in enumerate(refined_boxes_cam):
                orig_box = example["bboxes_3d_world_coords"][i]
                can_update = all_box_corners_visible_in_image(
                    orig_box, pose, intrinsic, img_dim
                )
                if cam_box is not None and can_update:
                    world_box = convert_box_3d_camera_to_world(cam_box, pose)
                    refined_boxes_world.append(
                        world_box if world_box is not None else orig_box
                    )
                else:
                    refined_boxes_world.append(orig_box)
            example["bboxes_3d_world_coords"] = refined_boxes_world

        # Require at least 2 valid point clouds
        if len(filepaths) <= 1:
            return None, False

        example = self._filter_by_valid_flags(example, valid_flags)
        return example, True
