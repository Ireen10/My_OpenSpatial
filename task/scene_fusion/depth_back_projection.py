import numpy as np
from PIL import Image
import io
from pathlib import Path
import open3d as o3d
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import pandas as pd
import tqdm

from utils.image_utils import load_depth_map
from task.base_task import BaseTask


def _depth_backproject_worker(task_args, example, idx):
    local_args = dict(task_args or {})
    local_args["use_multi_processing"] = False
    local_args["num_workers"] = 1
    task = DepthBackProjecter(local_args)
    result, flag = task.apply_transform(example, idx)
    return idx, result, flag


class DepthBackProjecter(BaseTask):
    """Back-project depth maps to per-object 3D point clouds using masks."""

    def __init__(self, args):
        super().__init__(args)
        self.output_dir = self.args.get("output_dir")
        if self.output_dir is None:
            raise ValueError("output_dir must be specified in args.")
        self.parallel_backend = str(args.get("parallel_backend", "thread")).lower()
        self.max_points_per_object = int(args.get("max_points_per_object", 20000))
        self.min_points_per_object = int(args.get("min_points_per_object", 30))
        self.enable_outlier_removal = bool(args.get("enable_outlier_removal", True))
        self.outlier_nb_neighbors = int(args.get("outlier_nb_neighbors", 10))
        self.outlier_std_ratio = float(args.get("outlier_std_ratio", 2.0))

    def run(self, dataset):
        if self.use_multi_processing and self.parallel_backend == "process":
            return self._run_multi_processing_process(dataset)
        return super().run(dataset)

    def _run_multi_processing_process(self, dataset):
        num_workers = int(self.args.get("num_workers", 8))
        n = len(dataset)
        window = int(self.args.get("max_inflight", num_workers))
        window = max(1, window)
        print(
            f"  [{type(self).__name__}] {n} examples, {num_workers} process workers "
            f"(window={window})",
            flush=True,
        )

        processed_by_idx = {}
        pending = set()
        next_idx = 0
        errors = 0
        worker_args = dict(self.args)
        worker_args["parallel_backend"] = "thread"

        executor = ProcessPoolExecutor(max_workers=num_workers)
        shutdown_now = False
        try:
            pbar = tqdm.tqdm(total=n, desc="Processing examples")
            while next_idx < n or pending:
                while next_idx < n and len(pending) < window:
                    example = dataset.iloc[next_idx].to_dict()
                    pending.add(
                        executor.submit(
                            _depth_backproject_worker,
                            worker_args,
                            example,
                            next_idx,
                        )
                    )
                    next_idx += 1
                if not pending:
                    break
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    try:
                        idx, result, flag = fut.result()
                    except Exception as exc:
                        errors += 1
                        print(f"  [error] worker failed: {exc}", flush=True)
                        pbar.update(1)
                        continue
                    if flag:
                        processed_by_idx[idx] = result
                    pbar.update(1)
            pbar.close()
        except KeyboardInterrupt:
            shutdown_now = True
            for fut in pending:
                fut.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            if not shutdown_now:
                executor.shutdown(wait=True, cancel_futures=False)

        processed = [processed_by_idx[i] for i in sorted(processed_by_idx)]
        print(
            f"  [{type(self).__name__}] {len(processed)}/{n} passed"
            + (f", {errors} failed" if errors else ""),
            flush=True,
        )
        return pd.DataFrame(processed).reset_index(drop=True)

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

    @staticmethod
    def _mask_indices(mask):
        return np.flatnonzero(np.asarray(mask).reshape(-1) > 0)

    @staticmethod
    def _sample_indices(indices, max_points):
        if max_points <= 0 or indices.size <= max_points:
            return indices
        pick = np.linspace(0, indices.size - 1, num=max_points, dtype=np.int64)
        return indices[pick]

    @staticmethod
    def _backproject_flat_indices(depth, intrinsic, flat_idx):
        h, w = depth.shape
        v, u = np.divmod(flat_idx, w)
        z = depth.reshape(-1)[flat_idx].astype(np.float64)
        k = np.asarray(intrinsic, dtype=np.float64)
        x = (u.astype(np.float64) - k[0, 2]) * z / k[0, 0]
        y = (v.astype(np.float64) - k[1, 2]) * z / k[1, 1]
        return np.stack([x, y, z], axis=1)

    def _backproject_masks_to_pointclouds(self, depth, intrinsic, masks, img_idx):
        """Back-project masked depth regions to cleaned 3D point clouds.

        Args:
            depth: H x W depth map array.
            intrinsic: 4x4 camera intrinsic matrix.
            masks: list of 2D mask arrays.
            img_idx: index used for naming output files.

        Returns:
            (filepaths, valid_flags): saved .pcd paths and per-mask validity.
        """
        output_dir = os.path.join(self.output_dir, self.args["file_name"], "pointclouds")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        filepaths = []
        valid_flags = []
        depth_flat = depth.reshape(-1)
        for idx, mask in enumerate(masks):
            flat_idx = self._mask_indices(mask)
            if flat_idx.size:
                z = depth_flat[flat_idx]
                valid_depth = np.isfinite(z) & (z > 0)
                flat_idx = flat_idx[valid_depth]
            if flat_idx.size < self.min_points_per_object:
                valid_flags.append(False)
                continue
            flat_idx = self._sample_indices(flat_idx, self.max_points_per_object)
            masked_pts = self._backproject_flat_indices(depth, intrinsic, flat_idx)
            if masked_pts.shape[0] < self.min_points_per_object:
                valid_flags.append(False)
                continue

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(masked_pts)

            if self.enable_outlier_removal and len(pcd.points) > self.outlier_nb_neighbors:
                _, ind = pcd.remove_statistical_outlier(
                    nb_neighbors=self.outlier_nb_neighbors,
                    std_ratio=self.outlier_std_ratio,
                )
                cleaned = pcd.select_by_index(ind)
            else:
                cleaned = pcd

            if cleaned.is_empty():
                valid_flags.append(False)
                continue

            valid_flags.append(True)
            filepath = os.path.join(
                output_dir,
                f"pointcloud_{Path(str(img_idx)).stem}_{idx}.pcd"
            )
            o3d.io.write_point_cloud(filepath, cleaned)
            filepaths.append(filepath)

        return filepaths, valid_flags

    @staticmethod
    def _filter_by_valid_flags(example, valid_flags):
        """Remove invalid-mask entries from object-aligned list-like fields."""
        if all(valid_flags):
            return example
        filtered = example.copy()
        for key, value in example.items():
            if isinstance(value, (str, bytes, dict)) or value is None:
                continue
            seq = value
            if hasattr(value, "tolist"):
                try:
                    seq = value.tolist()
                except Exception:
                    seq = value
            if isinstance(seq, (list, tuple)) and len(seq) == len(valid_flags):
                filtered[key] = [v for v, ok in zip(seq, valid_flags) if ok]
        return filtered

    def apply_transform(self, example, img_idx):
        """Back-project depth to per-object point clouds.

        Requires: intrinsic, depth_map, depth_scale, masks, obj_tags.
        Populates: pointclouds.
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

        filepaths, valid_flags = self._backproject_masks_to_pointclouds(
            depth, intrinsic, masks, img_idx
        )

        example["pointclouds"] = filepaths

        # Require at least 2 valid point clouds
        if len(filepaths) <= 1:
            return None, False

        example = self._filter_by_valid_flags(example, valid_flags)
        return example, True
