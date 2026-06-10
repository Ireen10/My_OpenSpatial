import os

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
from shapely.ops import unary_union

from task.base_task import BaseTask
from utils.box_utils import (
    compute_box_3d_corners_from_params,
    obb_volume_from_points,
    oriented_box_volume,
    points_inside_oriented_box,
    shrunk_oriented_box_geometry,
)
from utils.image_utils import load_depth_map
from utils.projection_utils import backproject_depth_to_3d


class ThreeDBoxFilter(BaseTask):
    """Filter objects by validating 3D bounding boxes against depth-based point clouds."""

    EPS = 1e-4
    MIN_POINTS_IN_BOX = 50

    def __init__(self, args):
        super().__init__(args)
        self.proj_mask_threshold = args.get("proj_mask_threshold", 0.1)
        self.box3d_pcd_threshold = args.get("box3d_pcd_threshold", 0.07)
        self.output_dir = os.path.join(args.get("output_dir"), args.get("file_name", "3dbox_filter"))
        self.box_scale_factor = args.get("box_scale_factor", 0.9)
        self.mask_area_threshold = args.get("mask_area_threshold", 0.02)

    @staticmethod
    def _get_box_corners(box):
        """Compute 8 corner points of a 3D bounding box.

        Args:
            box: [cx, cy, cz, w, h, d, yaw, pitch, roll].

        Returns:
            np.ndarray of shape (8, 3).
        """
        return compute_box_3d_corners_from_params(box)

    def _is_box_valid_2d(self, corners, extrinsic_w2c, intrinsic, img_dim):
        """Check if a 3D box projects sufficiently onto the image plane.

        Projects box corners to pixel coordinates, rasterises the face
        polygons with cv2.fillPoly (C-level scan-line fill, ~20× faster than
        matplotlib.path on a full-resolution meshgrid), and checks that the
        visible (in-image) portion is large enough relative to the total
        projected area.

        Returns:
            True if the box passes the 2D projection check.
        """
        w, h = img_dim

        # Project corners: homogeneous 3D → image coordinates
        corners_h = np.concatenate([corners, np.ones((corners.shape[0], 1))], axis=1)
        corners_img = (intrinsic @ extrinsic_w2c @ corners_h.T).T

        # Safe perspective divide (avoid division by near-zero z)
        z = corners_img[:, 2:3]
        z_safe = np.where(np.abs(z) < self.EPS, self.EPS, z)
        corners_px = corners_img[:, :2] / z_safe  # (8, 2) float pixel coords

        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
                 [3, 2, 6, 7], [0, 3, 7, 4], [1, 2, 6, 5]]

        # cv2.fillPoly rasterises each face directly into a uint8 canvas —
        # no W×H point array, no Python-level iteration over pixels.
        canvas = np.zeros((h, w), dtype=np.uint8)
        polygons = []
        for face in faces:
            if (corners_img[face][:, 2] < self.EPS).any():
                continue
            pts_float = corners_px[face, :2]
            polygons.append(Polygon(pts_float.tolist()))
            pts_int = np.round(pts_float).astype(np.int32)
            cv2.fillPoly(canvas, [pts_int], 1)

        union_area = unary_union(polygons).area if polygons else 0
        if union_area == 0:
            return False
        return int(canvas.sum()) / union_area >= self.proj_mask_threshold

    def _is_box_valid_3d(self, points_3d, box_params, img_dim):
        """Check if a 3D box contains enough scene geometry.

        Builds an oriented bounding box (slightly shrunk by box_scale_factor),
        finds points inside, and validates via PCA-OBB volume ratio and mask area.

        Returns:
            (True, mask) if valid, (False, None) otherwise.
        """
        center, extent, rotation = shrunk_oriented_box_geometry(
            box_params, self.box_scale_factor
        )
        inside = points_inside_oriented_box(points_3d, center, extent, rotation)
        inside_idx = np.flatnonzero(inside)
        if inside_idx.size < self.MIN_POINTS_IN_BOX:
            return False, None

        selected = points_3d[inside_idx]
        volume = obb_volume_from_points(selected)
        query_volume = oriented_box_volume(extent)
        if query_volume <= 0 or volume / query_volume < self.box3d_pcd_threshold:
            return False, None

        w, h = img_dim
        mask = np.zeros((h, w), dtype=bool)
        mask.ravel()[inside_idx] = True

        if mask.sum() / (h * w) < self.mask_area_threshold:
            return False, None

        return True, mask

    def _filter_boxes(self, depth, boxes_3d, pose, intrinsic, img_dim):
        """Two-stage filtering: 2D projection validity, then 3D point cloud validity.

        Returns:
            (keep_indices, masks): indices into boxes_3d that passed, and their masks.
        """
        extrinsic_w2c = np.linalg.inv(pose)
        points_3d = backproject_depth_to_3d(depth, img_dim, intrinsic, pose=pose)

        # Stage 1: 2D projection check
        valid_2d = []
        for i, box in enumerate(boxes_3d):
            corners = self._get_box_corners(box)
            if self._is_box_valid_2d(corners, extrinsic_w2c, intrinsic, img_dim):
                valid_2d.append(i)

        # Stage 2: 3D point cloud check
        keep_indices = []
        masks = []
        for idx in valid_2d:
            valid, mask = self._is_box_valid_3d(points_3d, boxes_3d[idx], img_dim)
            if valid:
                keep_indices.append(idx)
                masks.append(mask)

        return keep_indices, masks

    def _filter_by_indices(self, example, indices):
        """Keep only elements at given indices for each field in update_keys."""
        update_keys = self.args.get("update_keys", [])
        if not update_keys or indices is None:
            return example
        for key in update_keys:
            example[key] = [example[key][i] for i in indices]
        return example

    @staticmethod
    def _save_masks(masks, tags, mask_dir, prefix):
        """Save binary masks as grayscale PNG files.

        Returns:
            list of saved file paths.
        """
        os.makedirs(mask_dir, exist_ok=True)
        file_list = []
        for i, mask in enumerate(masks):
            arr = np.where(mask, 255, 0).astype(np.uint8)
            img = Image.fromarray(arr, mode='L')
            fpath = os.path.join(mask_dir, f"example_{prefix}_box_{i}_{tags[i]}_mask.png")
            img.save(fpath)
            file_list.append(fpath)
        return file_list

    def apply_transform(self, example, idx):
        """Filter 3D boxes by 2D/3D validity, generate masks for surviving objects.

        Requires: image, depth_map, depth_scale, obj_tags, bboxes_3d_world_coords,
                  pose, intrinsic.
        Populates: masks (file paths of valid object masks).
        """
        if "image" not in example:
            raise ValueError("image not found in example")

        depth = load_depth_map(example["depth_map"], example["depth_scale"])
        img_dim = depth.shape[::-1]  # (width, height)

        # Stage 0: remove background tags
        obj_tags = example["obj_tags"]
        if len(obj_tags) == 0:
            return None, False

        filter_tags = self.args.get("filter_tags", None)
        if filter_tags is not None:
            keep = [i for i, tag in enumerate(obj_tags) if tag not in filter_tags]
        else:
            keep = list(range(len(obj_tags)))
        if len(keep) == 0:
            return None, False

        assert "bboxes_3d_world_coords" in example
        obj_tags = [obj_tags[i] for i in keep]
        boxes_3d = [example["bboxes_3d_world_coords"][i] for i in keep]
        example = self._filter_by_indices(example, keep)
        example["obj_tags"] = obj_tags
        example["bboxes_3d_world_coords"] = boxes_3d

        # Stage 1+2: 2D/3D box validation
        pose = np.loadtxt(example["pose"])
        intrinsic = np.loadtxt(example["intrinsic"])
        keep2, masks = self._filter_boxes(depth, boxes_3d, pose, intrinsic, img_dim)
        if len(keep2) == 0:
            return None, False

        obj_tags = [obj_tags[i] for i in keep2]
        boxes_3d = [boxes_3d[i] for i in keep2]
        example = self._filter_by_indices(example, keep2)
        example["obj_tags"] = obj_tags
        example["bboxes_3d_world_coords"] = boxes_3d

        # Save masks
        mask_dir = os.path.join(self.output_dir, "masks")
        example["masks"] = self._save_masks(masks, obj_tags, mask_dir, prefix=str(idx))
        return example, True
