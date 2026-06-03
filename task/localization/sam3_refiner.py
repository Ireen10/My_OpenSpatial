import queue
import warnings
import numpy as np
import torch
from PIL import Image
import os
import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from transformers import Sam3TrackerProcessor, Sam3TrackerModel
from task.base_task import BaseTask


class Sam3Refiner(BaseTask):
    """Refine coarse masks using SAM3 Tracker box-prompt segmentation (via transformers).

    Drop-in replacement for Sam2Refiner. Uses Sam3TrackerModel from the transformers
    library instead of the sam2 package, preserving all filtering and saving logic.
    Recommended: transformers >= 5.0.0

    Multi-card parallel inference
    ─────────────────────────────
    Set ``device`` to a comma-separated list of device strings to enable data-parallel
    processing across multiple cards.  Each device gets ``replicas_per_device`` independent
    model instances (useful on high-memory cards with bfloat16).  Worker threads compete
    for replicas via a thread-safe queue; at most one thread uses each replica at a time.

    Typical configs (in YAML):

      # Single card (default behaviour)
      device: "npu:0"

      # Two 910B cards, one replica each → 2× throughput
      device: "npu:0,npu:1"
      use_multi_processing: true
      num_workers: 2

      # Two 910B cards, two replicas per card → ~4× throughput (recommended)
      device: "npu:0,npu:1"
      replicas_per_device: 2
      torch_dtype: bfloat16      # halves memory; bfloat16 is native on 910B
      use_multi_processing: true
      num_workers: 4
    """

    MIN_SCORE = 0.6
    MIN_MASK_PIXELS = 20

    def __init__(self, args, device=None):
        super().__init__(args)
        segmenter_model = args.get("segmenter_model", "facebook/sam3")

        # ── Device list ──────────────────────────────────────────────────────
        # Accepts a single string ("npu:0"), a comma-separated list
        # ("npu:0,npu:1"), or a Python list/tuple of strings.
        device_raw = args.get("device") or device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if isinstance(device_raw, (list, tuple)):
            devices = [str(d).strip() for d in device_raw]
        else:
            devices = [d.strip() for d in str(device_raw).split(",")]
        self.device = devices[0]

        # ── Optional dtype (e.g. "bfloat16" halves memory on 910B) ──────────
        torch_dtype_str = args.get("torch_dtype", None)
        torch_dtype = getattr(torch, torch_dtype_str) if torch_dtype_str else None

        # ── Build replica pool ───────────────────────────────────────────────
        # Each entry in the pool is a (processor, model, device) tuple.
        # Workers acquire a replica with queue.get(), release with queue.put().
        replicas_per_device = int(args.get("replicas_per_device", 1))
        self._replica_pool: queue.Queue = queue.Queue()

        load_kwargs = {}
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype

        for dev in devices:
            for _ in range(replicas_per_device):
                # Suppress the spurious "model of type sam3_video" warning.
                # The facebook/sam3 checkpoint registers model_type=sam3_video in
                # config.json to support four architectures from one file; loading
                # Sam3TrackerModel (sam3_tracker) from it triggers a type-mismatch
                # warning that is cosmetic only.  Tracked upstream: issue #43408.
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"You are using a model of type sam3_video",
                        category=UserWarning,
                    )
                    proc = Sam3TrackerProcessor.from_pretrained(segmenter_model)
                    model = Sam3TrackerModel.from_pretrained(
                        segmenter_model, **load_kwargs
                    ).to(dev)
                model.eval()
                self._replica_pool.put((proc, model, dev))

        # ── Auto-configure multi-processing when multiple replicas exist ─────
        total_replicas = len(devices) * replicas_per_device
        if total_replicas > 1:
            self.use_multi_processing = True
            if not args.get("num_workers"):
                args["num_workers"] = total_replicas

        assert "update_keys" in args, "update_keys must be specified in args."
        self.output_dir = os.path.join(self.args.get("output_dir"), self.args.get("file_name"))

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
        """Re-segment each mask region using SAM3 Tracker box prompts.

        Thread-safe: acquires a dedicated (processor, model) replica from the
        pool for the duration of the forward pass, then returns it.

        Args:
            image: RGB PIL Image.
            masks: list of 2D mask arrays (coarse masks to refine).

        Returns:
            (refined_masks, bboxes_2d, keep_indices) where keep_indices
            maps back to the original mask list. Returns ([], [], []) if
            no valid masks survive filtering.
        """
        input_boxes_np = self._masks_to_bboxes(masks)

        # Sam3TrackerProcessor expects input_boxes as [batch, num_objects, 4]
        input_boxes_list = [input_boxes_np.tolist()]

        # Acquire a free replica; blocks if all replicas are in use by other threads.
        processor, model, dev = self._replica_pool.get()
        try:
            inputs = processor(
                images=image,
                input_boxes=input_boxes_list,
                return_tensors="pt",
            ).to(dev)

            with torch.no_grad():
                outputs = model(**inputs, multimask_output=False)

            # post_process_masks returns a list of tensors (one per image in batch).
            # Each tensor has shape (num_objects, num_candidates, H, W).
            # With multimask_output=False, num_candidates=1.
            pred_masks = processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"],
            )[0]  # (N, 1, H, W) tensor

            # iou_scores shape: (batch=1, num_objects, num_candidates=1)
            scores = outputs.iou_scores[0, :, 0].cpu().numpy()  # (N,)
        finally:
            # Always return the replica so other threads are not starved.
            self._replica_pool.put((processor, model, dev))

        # Two-pass filtering: score threshold, then minimum pixel count
        refined, keep_indices = [], []
        for i, score in enumerate(scores):
            if score < self.MIN_SCORE:
                continue
            arr = self._squeeze_mask(pred_masks[i].numpy())  # (H, W) bool/float
            if np.sum(arr) > self.MIN_MASK_PIXELS:
                refined.append(arr)
                keep_indices.append(i)

        if not keep_indices:
            return [], [], []

        bboxes_2d = self._masks_to_bboxes([m.astype(bool) for m in refined]).tolist()
        return refined, bboxes_2d, keep_indices

    def refine_masks_batch(self, images_masks_list):
        """Run a single NPU forward pass over a micro-batch of images.

        Args:
            images_masks_list: list of (PIL Image, list-of-mask-arrays) pairs.

        Returns:
            list of (refined_masks, bboxes_2d, keep_indices) — one tuple per input.
            Entries with no surviving masks have ([], [], []).
        """
        batch_images = []
        batch_input_boxes = []
        for image, masks in images_masks_list:
            boxes_np = self._masks_to_bboxes(masks)
            batch_images.append(image)
            # Processor expects [batch_per_image, num_boxes, 4]; outer list = 1 prompt set
            batch_input_boxes.append([boxes_np.tolist()])

        processor, model, dev = self._replica_pool.get()
        try:
            inputs = processor(
                images=batch_images,
                input_boxes=batch_input_boxes,
                return_tensors="pt",
            ).to(dev)

            with torch.no_grad():
                outputs = model(**inputs, multimask_output=False)

            # post_process_masks returns list[Tensor(N_i, 1, H_i, W_i)], one per image
            all_pred_masks = processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"],
            )

            # iou_scores: (B, max_N, 1) — slice to actual N per image after unpadding
            all_scores = [
                outputs.iou_scores[i, : all_pred_masks[i].shape[0], 0].cpu().numpy()
                for i in range(len(batch_images))
            ]
        finally:
            self._replica_pool.put((processor, model, dev))

        results = []
        for pred_masks, scores in zip(all_pred_masks, all_scores):
            refined, keep_indices = [], []
            for i, score in enumerate(scores):
                if score < self.MIN_SCORE:
                    continue
                arr = self._squeeze_mask(pred_masks[i].numpy())
                if np.sum(arr) > self.MIN_MASK_PIXELS:
                    refined.append(arr)
                    keep_indices.append(i)

            if keep_indices:
                bboxes_2d = self._masks_to_bboxes([m.astype(bool) for m in refined]).tolist()
                results.append((refined, bboxes_2d, keep_indices))
            else:
                results.append(([], [], []))

        return results

    def _process_batch(self, batch_items):
        """Prepare, run SAM3, and post-process one micro-batch.

        Args:
            batch_items: list of (idx, example-dict) from the dataset.

        Returns:
            list of completed example dicts (failed / empty examples are dropped).
        """
        valid_items = []
        for idx, example in batch_items:
            try:
                self.validate_example(example)
                image = Image.open(example["image"])
                if image.mode != "RGB":
                    image = image.convert("RGB")
                coarse_masks = [np.array(Image.open(p)) for p in example["masks"]]
                valid_items.append((idx, example, image, coarse_masks))
            except Exception:
                pass

        if not valid_items:
            return []

        images_masks_list = [(item[2], item[3]) for item in valid_items]
        batch_results = self.refine_masks_batch(images_masks_list)

        outputs = []
        for (idx, example, _, _), (refined_masks, bboxes_2d, keep_indices) in zip(
            valid_items, batch_results
        ):
            if not keep_indices:
                continue

            self._filter_by_keep_indices(example, keep_indices)
            mask_dir = os.path.join(self.output_dir, "masks")
            mask_files = self._save_masks(refined_masks, mask_dir, prefix=str(idx))

            if len(mask_files) != len(example["obj_tags"]):
                continue
            if len(mask_files) != len(example["bboxes_3d_world_coords"]):
                continue

            example["masks"] = mask_files
            example["bboxes_2d"] = bboxes_2d
            outputs.append(example)

        return outputs

    def _run_batched(self, dataset):
        """ThreadPoolExecutor-based batched runner.

        Worker threads each handle a micro-batch of ``batch_size`` images,
        issuing one SAM3 forward pass per batch and competing for replicas in
        the pool.  This increases NPU utilisation compared to the one-image-
        per-pass baseline while keeping the thread-safe replica-pool contract.
        """
        num_workers = self.args.get("num_workers", 4)
        batch_size = int(self.args.get("batch_size", 4))

        examples = list(enumerate(dataset.to_dict("records")))
        batches = [
            examples[i : i + batch_size]
            for i in range(0, len(examples), batch_size)
        ]

        processed = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self._process_batch, b): b for b in batches}
            for future in tqdm.tqdm(
                as_completed(futures), total=len(futures), desc="SAM3 batched"
            ):
                try:
                    processed.extend(future.result())
                except Exception as exc:
                    print(f"[WARN] SAM3 batch failed: {exc}")

        if not processed:
            return pd.DataFrame()
        return pd.DataFrame(processed).reset_index(drop=True)

    def run(self, dataset):
        """Override BaseTask.run to use batched NPU inference when parallelism is on."""
        if self.use_multi_processing:
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
