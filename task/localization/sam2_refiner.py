from collections import defaultdict

import numpy as np
import torch
from PIL import Image
import os
import queue
import threading
import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from task.base_task import BaseTask


def _try_import_torch_npu() -> bool:
    try:
        import torch_npu  # noqa: F401
        return True
    except ImportError:
        return False


def _npu_available() -> bool:
    if not _try_import_torch_npu():
        return False
    return bool(hasattr(torch, "npu") and torch.npu.is_available())


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if _npu_available():
        return "npu:0"
    return "cpu"


def _is_npu_device(device: str) -> bool:
    return str(device).startswith("npu")


def _set_npu_device(device: str) -> None:
    if not _is_npu_device(device):
        return
    if not _try_import_torch_npu():
        raise ImportError("torch_npu is required for device='npu:*'.")
    dev_id = int(str(device).split(":")[-1]) if ":" in str(device) else 0
    torch.npu.set_device(dev_id)


def _torch_dtype_from_arg(dtype_name):
    if not dtype_name:
        return None
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = str(dtype_name).lower()
    if key not in mapping:
        raise ValueError(f"Unsupported torch_dtype: {dtype_name!r}")
    return mapping[key]


def _parse_devices(device_raw):
    if isinstance(device_raw, (list, tuple)):
        return [str(d).strip() for d in device_raw if str(d).strip()]
    return [d.strip() for d in str(device_raw).split(",") if d.strip()]


def _load_coarse_mask(path: str) -> np.ndarray:
    mask = np.array(Image.open(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask > 127


def _load_transformers_replica(segmenter_model: str, load_kwargs: dict, device: str):
    _set_npu_device(device)
    try:
        from transformers import Sam2Model, Sam2Processor
    except ImportError as exc:
        raise ImportError(
            "Sam2Model/Sam2Processor not found. Install a recent transformers "
            "version that includes SAM2 support."
        ) from exc

    processor = Sam2Processor.from_pretrained(segmenter_model)
    model = Sam2Model.from_pretrained(segmenter_model, **load_kwargs).to(device)
    model.eval()
    return processor, model


def _normalize_masks_for_image(masks, expected_count: int) -> np.ndarray:
    arr = masks.detach().cpu().numpy() if hasattr(masks, "detach") else np.asarray(masks)
    # Official docs with multimask_output=False return one mask per object:
    #   post_process_masks(...)[image_idx] -> (num_objects, H, W)
    # Some versions keep a singleton mask-choice axis: (num_objects, 1, H, W).
    if arr.ndim == 4:
        arr = arr[:, 0]
    if arr.ndim == 2:
        arr = arr[None, ...]
    return arr[:expected_count]


def _normalize_scores_for_image(scores, image_idx: int, expected_count: int) -> np.ndarray:
    arr = scores.detach().float().cpu().numpy() if hasattr(scores, "detach") else np.asarray(scores)
    # Expected shapes across transformers versions:
    #   (B, num_objects) or (B, num_objects, 1) with multimask_output=False.
    # If a mask-choice axis remains, take the first/best score to mirror
    # multimask_output=False.
    if arr.ndim >= 2:
        arr = arr[image_idx]
    if arr.ndim == 2:
        arr = arr[:, 0]
    if arr.ndim == 0:
        arr = arr[None]
    return arr.reshape(-1)[:expected_count]


def _infer_transformers_forward(processor, model, device, images, boxes_list):
    """One SAM2 forward for a sub-batch with uniform boxes-per-image."""
    _set_npu_device(device)
    inputs = processor(
        images=images,
        input_boxes=boxes_list,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)

    original_sizes = inputs["original_sizes"]
    if hasattr(original_sizes, "cpu"):
        original_sizes = original_sizes.cpu()
    all_masks = processor.post_process_masks(outputs.pred_masks.cpu(), original_sizes)

    results = []
    for out_idx in range(len(images)):
        expected_count = len(boxes_list[out_idx])
        masks_np = _normalize_masks_for_image(all_masks[out_idx], expected_count)
        scores_np = _normalize_scores_for_image(outputs.iou_scores, out_idx, expected_count)
        results.append((list(masks_np), scores_np.tolist()))
    return results


def _infer_transformers_refiner(processor, model, device, images, boxes_per_image):
    """Run SAM2 transformers inference; batch only images with the same box count.

    Unlike the official ``sam2`` predictor, ``Sam2Processor`` / ``Sam2Model`` in
    transformers require every image in one forward to have the same number of
    box prompts. Images are grouped by box count so we still batch when counts
    align (e.g. four images each with 3 boxes).
    """
    valid_indices = [i for i, boxes in enumerate(boxes_per_image) if len(boxes) > 0]
    per_image = [([], []) for _ in images]
    if not valid_indices:
        return per_image

    buckets: dict[int, list[int]] = defaultdict(list)
    for i in valid_indices:
        buckets[len(boxes_per_image[i])].append(i)

    for bucket_indices in buckets.values():
        bucket_images = [images[i] for i in bucket_indices]
        bucket_boxes = [boxes_per_image[i].astype(float).tolist() for i in bucket_indices]
        bucket_results = _infer_transformers_forward(
            processor, model, device, bucket_images, bucket_boxes
        )
        for orig_idx, result in zip(bucket_indices, bucket_results):
            per_image[orig_idx] = result
    return per_image


class Sam2Refiner(BaseTask):
    """Refine coarse masks using SAM2 box-prompt segmentation."""

    MIN_SCORE = 0.6
    MIN_MASK_PIXELS = 20

    def __init__(self, args, device=None):
        super().__init__(args)
        segmenter_model = args.get("segmenter_model", "facebook/sam2-hiera-small")
        device_raw = args.get("device") or device or _default_device()
        devices = _parse_devices(device_raw)
        self.device = devices[0]
        self.segmenter_backend = args.get("segmenter_backend", args.get("backend", "auto"))
        if self.segmenter_backend == "auto":
            self.segmenter_backend = "sam2"

        if self.segmenter_backend == "transformers":
            self._init_transformers_backend(segmenter_model, devices)
        elif self.segmenter_backend == "sam2":
            self._init_sam2_backend(segmenter_model)
        else:
            raise ValueError(
                "segmenter_backend must be one of: auto, sam2, transformers"
            )

        assert "update_keys" in args, "update_keys must be specified in args."
        self.output_dir = os.path.join(self.args.get("output_dir"), self.args.get("file_name"))

    def _init_sam2_backend(self, segmenter_model):
        _set_npu_device(self.device)
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.sam2_model = SAM2ImagePredictor.from_pretrained(
            segmenter_model, trust_remote_code=True, device=self.device
        )

    def _init_transformers_backend(self, segmenter_model, devices):
        load_kwargs = {}
        dtype = _torch_dtype_from_arg(self.args.get("torch_dtype"))
        if dtype is not None:
            load_kwargs["torch_dtype"] = dtype
        replicas_per_device = int(self.args.get("replicas_per_device", 1))
        if any(_is_npu_device(dev) for dev in devices) and replicas_per_device != 1:
            raise ValueError(
                "SAM2 NPU mode supports exactly one transformers replica per NPU. "
                "Set replicas_per_device: 1 to avoid known AICPU errors."
            )

        self._replica_pool: queue.Queue = queue.Queue()
        for dev in devices:
            for _ in range(replicas_per_device):
                processor, model = _load_transformers_replica(segmenter_model, load_kwargs, dev)
                self._replica_pool.put((processor, model, dev))

        total_replicas = len(devices) * replicas_per_device
        if total_replicas > 1:
            self.use_multi_processing = True
            if not self.args.get("num_workers"):
                self.args["num_workers"] = total_replicas
        self.pipeline_batch_size = int(self.args.get("batch_size", 4))
        print(
            f"[Sam2Refiner] transformers backend on {','.join(devices)} | "
            f"replicas={total_replicas} pipeline_batch_size={self.pipeline_batch_size} "
            f"min_score={self.MIN_SCORE} min_mask_pixels={self.MIN_MASK_PIXELS}",
            flush=True,
        )

    @staticmethod
    def _masks_to_bboxes(masks):
        """Compute axis-aligned bounding boxes from binary masks.

        Args:
            masks: list of 2D boolean/uint8 arrays.

        Returns:
            np.ndarray of shape (N, 4) with [x1, y1, x2, y2] per mask.
        """
        boxes = []
        for mask in masks:
            ys, xs = np.where(mask)
            if len(xs) > 0:
                boxes.append([np.min(xs), np.min(ys), np.max(xs), np.max(ys)])
            else:
                boxes.append([0, 0, 0, 0])
        return np.array(boxes)

    @staticmethod
    def _squeeze_mask(arr):
        """Squeeze (1, H, W) → (H, W) if needed."""
        return arr[0] if arr.ndim == 3 else arr

    def refine_masks(self, image, masks):
        """Re-segment each mask region using SAM2 box prompts.

        Args:
            image: RGB PIL Image.
            masks: list of 2D mask arrays (coarse masks to refine).

        Returns:
            (refined_masks, bboxes_2d, keep_indices) where keep_indices
            maps back to the original mask list. Returns ([], [], []) if
            no valid masks survive filtering.
        """
        if self.segmenter_backend == "transformers":
            return self._refine_masks_transformers(image, masks)

        _set_npu_device(self.device)
        self.sam2_model.set_image(image)
        input_boxes = self._masks_to_bboxes(masks)

        raw_masks, scores, _ = self.sam2_model.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        # Two-pass filtering: score threshold, then minimum pixel count
        refined, keep_indices = [], []
        for i, score in enumerate(scores):
            if score < self.MIN_SCORE:
                continue
            arr = self._squeeze_mask(raw_masks[i])
            if np.sum(arr) > self.MIN_MASK_PIXELS:
                refined.append(arr)
                keep_indices.append(i)

        if not keep_indices:
            return [], [], []

        bboxes_2d = self._masks_to_bboxes([m.astype(bool) for m in refined]).tolist()
        return refined, bboxes_2d, keep_indices

    def _refine_masks_transformers(self, image, masks):
        input_boxes = self._masks_to_bboxes(masks).astype(float).tolist()
        if not input_boxes:
            return [], [], []

        masks_np, scores_np = self._infer_transformers_batch([image], [self._masks_to_bboxes(masks)])[0]

        refined, keep_indices = [], []
        for i, arr in enumerate(masks_np):
            score = float(scores_np[i]) if i < len(scores_np) else 1.0
            if score < self.MIN_SCORE:
                continue
            arr = arr > 0
            if np.sum(arr) > self.MIN_MASK_PIXELS:
                refined.append(arr)
                keep_indices.append(i)

        if not keep_indices:
            return [], [], []

        bboxes_2d = self._masks_to_bboxes([m.astype(bool) for m in refined]).tolist()
        return refined, bboxes_2d, keep_indices

    def _infer_transformers_batch(self, images, boxes_per_image):
        processor, model, dev = self._replica_pool.get()
        try:
            return _infer_transformers_refiner(processor, model, dev, images, boxes_per_image)
        finally:
            self._replica_pool.put((processor, model, dev))

    def _results_from_infer(self, per_image):
        out = []
        for pred_masks, scores in per_image:
            refined, keep_indices = [], []
            for i, arr in enumerate(pred_masks):
                score = float(scores[i]) if i < len(scores) else 1.0
                if score < self.MIN_SCORE:
                    continue
                arr = np.asarray(arr) > 0
                if np.sum(arr) > self.MIN_MASK_PIXELS:
                    refined.append(arr)
                    keep_indices.append(i)
            if keep_indices:
                bboxes = self._masks_to_bboxes([m.astype(bool) for m in refined]).tolist()
                out.append((refined, bboxes, keep_indices))
            else:
                out.append(([], [], []))
        return out

    def _process_batch(self, batch_items):
        valid_items = []
        for idx, example in batch_items:
            try:
                self.validate_example(example)
                image = Image.open(example["image"])
                if image.mode != "RGB":
                    image = image.convert("RGB")
                coarse = [_load_coarse_mask(p) for p in example["masks"]]
                valid_items.append((idx, example, image, coarse))
            except Exception:
                pass

        if not valid_items:
            return [], {"samples": len(batch_items), "boxes_in": 0, "boxes_kept": 0, "samples_saved": 0}

        images = [image for _, _, image, _ in valid_items]
        boxes_per_image = [self._masks_to_bboxes(coarse) for _, _, _, coarse in valid_items]
        per_image = self._infer_transformers_batch(images, boxes_per_image)
        batch_results = self._results_from_infer(per_image)
        boxes_in = sum(len(x[3]) for x in valid_items)
        boxes_kept = sum(len(ki) for _, _, ki in batch_results)

        outputs = []
        for (idx, example, _, _), (refined, bboxes_2d, keep_indices) in zip(valid_items, batch_results):
            if not keep_indices:
                continue
            self._filter_by_keep_indices(example, keep_indices)
            mask_files = self._save_masks(refined, os.path.join(self.output_dir, "masks"), str(idx))
            if len(mask_files) != len(example["obj_tags"]):
                continue
            if len(mask_files) != len(example["bboxes_3d_world_coords"]):
                continue
            example["masks"] = mask_files
            example["bboxes_2d"] = bboxes_2d
            outputs.append(example)

        return outputs, {
            "samples": len(batch_items),
            "boxes_in": boxes_in,
            "boxes_kept": boxes_kept,
            "samples_saved": len(outputs),
        }

    def _run_batched(self, dataset):
        num_workers = int(self.args.get("num_workers", 4))
        batch_size = int(self.args.get("batch_size", 4))
        examples = list(enumerate(dataset.to_dict("records")))
        batches = [examples[i : i + batch_size] for i in range(0, len(examples), batch_size)]
        window = max(num_workers * 2, num_workers + 1)

        processed = []
        total_samples = total_boxes_in = total_boxes_kept = total_samples_saved = 0
        next_log_at = 500
        lock = threading.Lock()

        def _log():
            nonlocal next_log_at
            while total_samples >= next_log_at:
                print(
                    f"[Sam2Refiner] {total_samples} samples | "
                    f"bbox in={total_boxes_in} filtered={total_boxes_in - total_boxes_kept} "
                    f"kept={total_boxes_kept} | samples saved={total_samples_saved}",
                    flush=True,
                )
                next_log_at += 500

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            pbar = tqdm.tqdm(total=len(examples), desc="SAM2 samples")
            for chunk_start in range(0, len(batches), window):
                chunk = batches[chunk_start : chunk_start + window]
                futures = [executor.submit(self._process_batch, batch) for batch in chunk]
                for future in as_completed(futures):
                    try:
                        outs, st = future.result()
                        processed.extend(outs)
                        with lock:
                            total_samples += st["samples"]
                            total_boxes_in += st["boxes_in"]
                            total_boxes_kept += st["boxes_kept"]
                            total_samples_saved += st["samples_saved"]
                            pbar.update(st["samples"])
                            _log()
                    except Exception as exc:
                        import traceback as _tb
                        print(f"[WARN] SAM2 batch failed: {exc}")
                        print(_tb.format_exc())
            pbar.close()

        if total_samples > 0 and total_samples % 500 != 0:
            print(
                f"[Sam2Refiner] {total_samples} samples (final) | "
                f"bbox in={total_boxes_in} filtered={total_boxes_in - total_boxes_kept} "
                f"kept={total_boxes_kept} | samples saved={total_samples_saved}",
                flush=True,
            )
        return pd.DataFrame(processed).reset_index(drop=True) if processed else pd.DataFrame()

    def run(self, dataset):
        if self.segmenter_backend == "transformers" and self.use_multi_processing:
            return self._run_batched(dataset)
        return super().run(dataset)

    def _save_masks(self, masks, mask_dir, prefix):
        """Save binary masks as grayscale PNG files.

        Args:
            masks: list of 2D arrays (values treated as boolean).
            mask_dir: output directory.
            prefix: filename prefix for saved files.

        Returns:
            list of saved file paths.
        """
        os.makedirs(mask_dir, exist_ok=True)
        file_list = []
        for i, mask in enumerate(masks):
            binary = (mask * 255).astype(np.uint8)
            img = Image.fromarray(binary, mode='L')
            path = os.path.join(mask_dir, f"example_{prefix}_box_{i}_mask.png")
            img.save(path)
            file_list.append(path)
        return file_list

    def validate_example(self, example):
        """Check that required fields exist and are non-empty."""
        for key in ("image", "masks", "obj_tags"):
            if key not in example:
                raise ValueError(f"{key} not found in example")
        if len(example["obj_tags"]) == 0:
            raise ValueError("obj_tags is empty")

    def _filter_by_keep_indices(self, example, keep_indices):
        """Keep only elements at keep_indices for each field in update_keys."""
        update_keys = self.args.get("update_keys", [])
        if not update_keys or keep_indices is None:
            return example
        for key in update_keys:
            example[key] = [example[key][i] for i in keep_indices]
        return example

    def apply_transform(self, example, idx):
        """Refine masks, filter by quality, save results.

        Populates:
            example["masks"]: list of refined mask file paths.
            example["bboxes_2d"]: list of [x1, y1, x2, y2] bounding boxes.
        """
        self.validate_example(example)

        image = Image.open(example["image"])
        if image.mode != "RGB":
            image = image.convert("RGB")

        coarse_masks = [np.array(Image.open(p)) for p in example["masks"]]
        refined_masks, bboxes_2d, keep_indices = self.refine_masks(image, coarse_masks)

        if not keep_indices:
            return None, False

        self._filter_by_keep_indices(example, keep_indices)

        mask_dir = os.path.join(self.output_dir, "masks")
        mask_files = self._save_masks(refined_masks, mask_dir, prefix=str(idx))

        assert len(mask_files) == len(example["obj_tags"]), (
            f"Mask count ({len(mask_files)}) != obj_tags count ({len(example['obj_tags'])})"
        )
        assert len(mask_files) == len(example["bboxes_3d_world_coords"]), (
            f"Mask count ({len(mask_files)}) != bboxes_3d count ({len(example['bboxes_3d_world_coords'])})"
        )

        example["masks"] = mask_files
        example["bboxes_2d"] = bboxes_2d
        return example, True
