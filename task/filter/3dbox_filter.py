import os
import threading
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import cv2
import numpy as np
import pandas as pd
import tqdm
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


def _process_example_worker(task_args, example, idx):
    """Process one sample in a subprocess for process-based parallelism."""
    local_args = dict(task_args or {})
    local_args["use_multi_processing"] = False
    local_args["num_workers"] = 1
    local_args["log_every"] = 0
    task = ThreeDBoxFilter(local_args)
    result, flag = task.apply_transform(example, idx)
    kept_boxes = 0
    if flag and result is not None:
        kept_boxes = len(result.get("bboxes_3d_world_coords", []))
    return idx, result, flag, kept_boxes


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
        self.depth_downsample_factor = max(1, int(args.get("depth_downsample_factor", 1)))
        self.use_candidate_aabb_prefilter = bool(args.get("use_candidate_aabb_prefilter", True))
        default_backend = "process" if self.use_multi_processing else "thread"
        self.parallel_backend = str(args.get("parallel_backend", default_backend)).lower()
        self.log_every = int(args.get("log_every", 1000))
        self._seen_examples = 0
        self._kept_examples = 0
        self._kept_3dbboxes = 0
        self._stats_lock = threading.Lock()

    def run(self, dataset):
        if self.use_multi_processing and self.parallel_backend == "process":
            return self._run_multi_processing_process(dataset)
        return super().run(dataset)

    def _run_multi_processing_process(self, dataset):
        num_workers = int(self.args.get("num_workers", 8))
        n = len(dataset)
        window = max(num_workers * 2, num_workers + 1)
        print(
            f"  [{type(self).__name__}] {n} examples, {num_workers} process workers "
            f"(window={window})",
            flush=True,
        )

        worker_args = dict(self.args)
        worker_args["parallel_backend"] = "thread"

        processed_by_idx = {}
        pending = set()
        next_idx = 0
        errors = 0

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            pbar = tqdm.tqdm(total=n, desc="Processing examples")
            while next_idx < n or pending:
                while next_idx < n and len(pending) < window:
                    example = dataset.iloc[next_idx].to_dict()
                    pending.add(
                        executor.submit(_process_example_worker, worker_args, example, next_idx)
                    )
                    next_idx += 1
                if not pending:
                    break
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    try:
                        idx, result, flag, kept_boxes = fut.result()
                    except Exception as exc:
                        errors += 1
                        self._update_stats_and_log(0)
                        print(f"  [error] worker failed: {exc}", flush=True)
                        pbar.update(1)
                        continue
                    self._update_stats_and_log(kept_boxes)
                    if flag:
                        processed_by_idx[idx] = result
                    pbar.update(1)
            pbar.close()

        processed = [processed_by_idx[i] for i in sorted(processed_by_idx)]
        print(
            f"  [{type(self).__name__}] {len(processed)}/{n} passed"
            + (f", {errors} failed" if errors else ""),
            flush=True,
        )
        return pd.DataFrame(processed).reset_index(drop=True)

    def _update_stats_and_log(self, kept_boxes: int):
        kept_boxes = int(max(0, kept_boxes))
        with self._stats_lock:
            self._seen_examples += 1
            self._kept_3dbboxes += kept_boxes
            if kept_boxes > 0:
                self._kept_examples += 1
            if self.log_every > 0 and self._seen_examples % self.log_every == 0:
                keep_rate = self._kept_examples / self._seen_examples
                print(
                    f"[ThreeDBoxFilter] processed={self._seen_examples}, "
                    f"kept_3dbboxes={self._kept_3dbboxes}, "
                    f"kept_samples={self._kept_examples} ({keep_rate:.2%})"
                )

    @staticmethod
    def _get_box_corners(box):
        """Compute 8 corner points of a 3D bounding box.

        Args:
            box: [cx, cy, cz, w, h, d, yaw, pitch, roll].

        Returns:
            np.ndarray of shape (8, 3).
        """
        return compute_box_3d_corners_from_params(box)

    def _is_box_valid_2d(
        self, corners, extrinsic_w2c, intrinsic, img_dim, return_candidate_indices=False
    ):
        """Check if a 3D box projects sufficiently onto the image plane.

        Projects box corners to pixel coordinates, rasterises the face
        polygons with cv2.fillPoly (C-level scan-line fill, ~20× faster than
        matplotlib.path on a full-resolution meshgrid), and checks that the
        visible (in-image) portion is large enough relative to the total
        projected area.

        Returns:
            - return_candidate_indices=False: bool
            - return_candidate_indices=True: (bool, candidate_flat_indices)
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
            if return_candidate_indices:
                return False, None
            return False
        ratio = int(canvas.sum()) / union_area
        is_valid = ratio >= self.proj_mask_threshold
        if not return_candidate_indices:
            return is_valid
        if not is_valid:
            return False, None
        candidate_flat_idx = np.flatnonzero(canvas.ravel())
        return True, candidate_flat_idx

    def _is_box_valid_3d(self, points_3d, box_params, img_dim, candidate_flat_idx=None):
        """Check if a 3D box contains enough scene geometry.

        Builds an oriented bounding box (slightly shrunk by box_scale_factor),
        finds points inside, and validates via PCA-OBB volume ratio and mask area.

        Returns:
            (True, mask) if valid, (False, None) otherwise.
        """
        center, extent, rotation = shrunk_oriented_box_geometry(
            box_params, self.box_scale_factor
        )
        if candidate_flat_idx is None:
            candidate_flat_idx = np.arange(points_3d.shape[0], dtype=np.int64)
        else:
            candidate_flat_idx = np.asarray(candidate_flat_idx, dtype=np.int64)
        if candidate_flat_idx.size == 0:
            return False, None

        candidate_points = points_3d[candidate_flat_idx]
        if self.use_candidate_aabb_prefilter:
            half_edge = 0.5 * float(np.max(extent))
            coarse = np.all(np.abs(candidate_points - center) <= (half_edge + self.EPS), axis=1)
            if not coarse.any():
                return False, None
            candidate_flat_idx = candidate_flat_idx[coarse]
            candidate_points = candidate_points[coarse]

        inside_local = points_inside_oriented_box(candidate_points, center, extent, rotation)
        inside_idx = candidate_flat_idx[np.flatnonzero(inside_local)]
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

    @staticmethod
    def _rescale_intrinsic(intrinsic, sx, sy):
        """Scale focal length and principal point for resized depth maps."""
        out = np.asarray(intrinsic, dtype=np.float64).copy()
        out[0, 0] *= sx
        out[1, 1] *= sy
        out[0, 2] *= sx
        out[1, 2] *= sy
        return out

    def _prepare_depth_and_intrinsic(self, depth, intrinsic):
        """Downsample depth and adjust intrinsics for faster geometric checks."""
        factor = self.depth_downsample_factor
        if factor <= 1:
            return depth, intrinsic
        h, w = depth.shape[:2]
        new_h = max(1, h // factor)
        new_w = max(1, w // factor)
        if new_h == h and new_w == w:
            return depth, intrinsic
        depth_ds = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        sx = new_w / float(w)
        sy = new_h / float(h)
        intrinsic_ds = self._rescale_intrinsic(intrinsic, sx, sy)
        return depth_ds, intrinsic_ds

    @staticmethod
    def _restore_mask_resolution(masks, target_img_dim):
        """Upsample bool masks back to original image size."""
        if not masks:
            return masks
        w, h = target_img_dim
        restored = []
        for mask in masks:
            up = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            restored.append(up > 0)
        return restored

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
            is_valid_2d, candidate_flat_idx = self._is_box_valid_2d(
                corners,
                extrinsic_w2c,
                intrinsic,
                img_dim,
                return_candidate_indices=True,
            )
            if is_valid_2d:
                valid_2d.append((i, candidate_flat_idx))

        # Stage 2: 3D point cloud check
        keep_indices = []
        masks = []
        for idx, candidate_flat_idx in valid_2d:
            valid, mask = self._is_box_valid_3d(
                points_3d, boxes_3d[idx], img_dim, candidate_flat_idx=candidate_flat_idx
            )
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
        original_img_dim = depth.shape[::-1]  # (width, height)

        # Stage 0: remove background tags
        obj_tags = example["obj_tags"]
        if len(obj_tags) == 0:
            self._update_stats_and_log(0)
            return None, False

        filter_tags = self.args.get("filter_tags", None)
        if filter_tags is not None:
            keep = [i for i, tag in enumerate(obj_tags) if tag not in filter_tags]
        else:
            keep = list(range(len(obj_tags)))
        if len(keep) == 0:
            self._update_stats_and_log(0)
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
        depth_work, intrinsic_work = self._prepare_depth_and_intrinsic(depth, intrinsic)
        img_dim = depth_work.shape[::-1]
        keep2, masks = self._filter_boxes(depth_work, boxes_3d, pose, intrinsic_work, img_dim)
        if len(keep2) == 0:
            self._update_stats_and_log(0)
            return None, False

        obj_tags = [obj_tags[i] for i in keep2]
        boxes_3d = [boxes_3d[i] for i in keep2]
        example = self._filter_by_indices(example, keep2)
        example["obj_tags"] = obj_tags
        example["bboxes_3d_world_coords"] = boxes_3d

        # Save masks
        if img_dim != original_img_dim:
            masks = self._restore_mask_resolution(masks, original_img_dim)
        mask_dir = os.path.join(self.output_dir, "masks")
        example["masks"] = self._save_masks(masks, obj_tags, mask_dir, prefix=str(idx))
        self._update_stats_and_log(len(keep2))
        return example, True
