import queue
import threading
import warnings
import numpy as np
import torch
from PIL import Image
import os
import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from task.base_task import BaseTask


def _validate_model_path(model_name_or_path: str) -> None:
    """Raise a clear FileNotFoundError when a local path does not exist.

    HuggingFace raises a cryptic "Repo id must be in the form …" error when a
    local path is passed but the directory is missing.  This check surfaces the
    real problem early.
    """
    is_local = (
        os.path.isabs(model_name_or_path)
        or model_name_or_path.startswith("./")
        or model_name_or_path.startswith("../")
    )
    if is_local and not os.path.isdir(model_name_or_path):
        raise FileNotFoundError(
            f"SAM3 weights directory not found: {model_name_or_path!r}\n"
            "Check the 'segmenter_model' field in your YAML — it must be either:\n"
            "  • an absolute path to a local directory containing config.json "
            "and model weights, OR\n"
            "  • a HuggingFace Hub repo ID such as 'facebook/sam3'."
        )


def _load_sam3_replica(segmenter_model: str, load_kwargs: dict, device: str):
    """Load one SAM3 (processor, model) pair in single-image mode.

    Prefers Sam3Processor + Sam3Model (image predictor, no memory module).
    Falls back to Sam3TrackerProcessor + Sam3TrackerModel if the image-predictor
    classes are not yet available in the installed transformers version.

    Why prefer the image model:
    - Sam3TrackerModel includes a memory encoder that calls torchvision.roi_align,
      which is unsupported on Ascend NPU and silently falls back to CPU.  The
      resulting CPU↔NPU tensor transfer can corrupt the IoU prediction head output,
      causing all scores to cluster around 0.03–0.05 regardless of mask quality.
    - The tracker's IoU score measures temporal-tracking quality, not segmentation
      quality, so single-frame scores are systematically low even without the NPU bug.
    """
    _validate_model_path(segmenter_model)

    suppress = warnings.catch_warnings()
    suppress.__enter__()
    warnings.filterwarnings(
        "ignore",
        message=r"You are using a model of type sam3_video",
        category=UserWarning,
    )
    try:
        from transformers import Sam3Processor, Sam3Model
        proc = Sam3Processor.from_pretrained(segmenter_model)
        model = Sam3Model.from_pretrained(segmenter_model, **load_kwargs).to(device)
        suppress.__exit__(None, None, None)
        return proc, model, "image"
    except (ImportError, AttributeError, OSError, ValueError):
        # ImportError/AttributeError: class not available in this transformers version.
        # OSError/ValueError: model_type mismatch or config incompatibility with
        # Sam3Model — fall back to the tracker variant which accepts sam3_video configs.
        pass

    from transformers import Sam3TrackerProcessor, Sam3TrackerModel
    proc = Sam3TrackerProcessor.from_pretrained(segmenter_model)
    model = Sam3TrackerModel.from_pretrained(segmenter_model, **load_kwargs).to(device)
    suppress.__exit__(None, None, None)
    return proc, model, "tracker"


def _target_sizes_from_inputs(inputs) -> list:
    sizes = inputs["original_sizes"]
    if hasattr(sizes, "tolist"):
        return sizes.tolist()
    return list(sizes)


def _filter_masks_from_scores(pred_masks, scores, min_score: float, min_pixels: int):
    """Apply score + pixel-count filters; return (masks, keep_indices)."""
    refined, keep_indices = [], []
    for i, score in enumerate(scores):
        if float(score) < min_score:
            continue
        arr = pred_masks[i]
        if hasattr(arr, "float"):
            arr = arr.float()
        if hasattr(arr, "cpu"):
            arr = arr.cpu().numpy()
        elif hasattr(arr, "numpy"):
            arr = arr.numpy()
        if arr.ndim == 3:
            arr = arr[0]
        if np.sum(arr) > min_pixels:
            refined.append(arr)
            keep_indices.append(i)
    return refined, keep_indices


def _forward_box_prompts(processor, model, variant: str, images, boxes_per_image: list,
                         device: str, min_score: float):
    """Run one SAM3 forward for box prompts; return list of (masks_tensor, scores).

    Follows HuggingFace docs:
    - Sam3Model: ``post_process_instance_segmentation`` (box + input_boxes_labels)
    - Sam3Tracker: ``multimask_output=False`` + ``post_process_masks`` on full batch
    """
    input_boxes = [b.tolist() if hasattr(b, "tolist") else list(b) for b in boxes_per_image]
    proc_kwargs = dict(images=images, input_boxes=input_boxes, return_tensors="pt")
    if variant == "image":
        proc_kwargs["input_boxes_labels"] = [[1] * len(b) for b in input_boxes]

    inputs = processor(**proc_kwargs).to(device)
    n_per_image = [len(b) for b in input_boxes]
    target_sizes = _target_sizes_from_inputs(inputs)

    with torch.no_grad():
        if variant == "tracker":
            outputs = model(**inputs, multimask_output=False)
        else:
            outputs = model(**inputs)

    per_image = []
    if variant == "tracker":
        # Official tracker batch: one post_process_masks call on full pred_masks tensor.
        all_masks = processor.post_process_masks(
            outputs.pred_masks.float().cpu(), inputs["original_sizes"]
        )
        for i, n_boxes in enumerate(n_per_image):
            scores = outputs.iou_scores[i, :n_boxes, 0].float().cpu().numpy()
            per_image.append((all_masks[i], scores))
    else:
        # Official Sam3Model box path (transformers model_doc/sam3).
        seg = processor.post_process_instance_segmentation(
            outputs,
            threshold=min_score,
            mask_threshold=0.5,
            target_sizes=target_sizes,
        )
        for item in seg:
            masks = item["masks"]
            if hasattr(masks, "float"):
                masks = masks.float()
            scores = item["scores"].float().cpu().numpy()
            per_image.append((masks, scores))

    return per_image


class Sam3Refiner(BaseTask):
    """Refine coarse masks using SAM3 box-prompt segmentation (via transformers).

    Drop-in replacement for Sam2Refiner. Uses Sam3Model (image predictor) from the
    transformers library instead of the sam2 package, preserving all filtering and
    saving logic.  Falls back to Sam3TrackerModel if Sam3Model is not available.
    Recommended: transformers >= 5.0.0

    Multi-card parallel inference
    ─────────────────────────────
    Set ``device`` to a comma-separated list of device strings to enable data-parallel
    processing across multiple cards.  Each device gets ``replicas_per_device`` independent
    model instances per device.  Worker threads compete
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

        # ── Build replica pool (float32 — no torch_dtype override) ─────────
        replicas_per_device = int(args.get("replicas_per_device", 1))
        self._replica_pool: queue.Queue = queue.Queue()

        _variant_logged = False
        for dev in devices:
            for _ in range(replicas_per_device):
                proc, model, variant = _load_sam3_replica(segmenter_model, {}, dev)
                if not _variant_logged:
                    _cls = type(model).__name__
                    print(f"[Sam3Refiner] SAM3 loaded as {_cls} ({variant}) on {dev} "
                          f"(dtype=float32)")
                    _variant_logged = True
                model.eval()
                self._replica_pool.put((proc, model, dev, variant))

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
        """Re-segment each mask region using SAM3 box prompts (HF official API).

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

        processor, model, dev, variant = self._replica_pool.get()
        try:
            pred_masks, scores = _forward_box_prompts(
                processor, model, variant, image, [input_boxes_np], dev, self.MIN_SCORE
            )[0]
        finally:
            self._replica_pool.put((processor, model, dev, variant))

        if variant == "image":
            refined, keep_indices = _filter_masks_from_scores(
                pred_masks, scores, 0.0, self.MIN_MASK_PIXELS
            )
        else:
            refined, keep_indices = _filter_masks_from_scores(
                pred_masks, scores, self.MIN_SCORE, self.MIN_MASK_PIXELS
            )

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
        boxes_per_image = []
        for image, masks in images_masks_list:
            batch_images.append(image)
            boxes_per_image.append(self._masks_to_bboxes(masks))

        processor, model, dev, variant = self._replica_pool.get()
        try:
            # Sam3Tracker batch requires the same box count per image (HF docs).
            n_boxes = [len(b) for b in boxes_per_image]
            if variant == "tracker" and len(set(n_boxes)) > 1:
                per_image = []
                for image, boxes_np in zip(batch_images, boxes_per_image):
                    per_image.extend(
                        _forward_box_prompts(
                            processor, model, variant, image, [boxes_np], dev, self.MIN_SCORE
                        )
                    )
            else:
                per_image = _forward_box_prompts(
                    processor, model, variant, batch_images, boxes_per_image, dev, self.MIN_SCORE
                )
        finally:
            self._replica_pool.put((processor, model, dev, variant))

        results = []
        min_score = 0.0 if variant == "image" else self.MIN_SCORE
        for pred_masks, scores in per_image:
            refined, keep_indices = _filter_masks_from_scores(
                pred_masks, scores, min_score, self.MIN_MASK_PIXELS
            )
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
            return [], {
                "samples": len(batch_items),
                "boxes_in": 0,
                "boxes_kept": 0,
                "samples_saved": 0,
            }

        images_masks_list = [(item[2], item[3]) for item in valid_items]
        batch_results = self.refine_masks_batch(images_masks_list)

        boxes_in = sum(len(item[3]) for item in valid_items)
        boxes_kept = sum(len(ki) for _, _, ki in batch_results)

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

        stats = {
            "samples": len(batch_items),
            "boxes_in": boxes_in,
            "boxes_kept": boxes_kept,
            "samples_saved": len(outputs),
        }
        return outputs, stats

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
        window = num_workers * 2
        n_micro_batches = len(batches)
        n_rounds = (n_micro_batches + window - 1) // window
        print(
            f"[Sam3Refiner] {len(examples)} samples | "
            f"{n_micro_batches} micro-batches (batch_size={batch_size}) | "
            f"tqdm counts scheduler rounds={n_rounds} (window={window})",
            flush=True,
        )

        # Submit at most (num_workers * 2) futures at a time so the pending-
        # futures dict never holds the entire dataset in memory simultaneously.
        processed = []
        total_samples = 0
        total_boxes_in = 0
        total_boxes_kept = 0
        total_samples_saved = 0
        next_log_at = 1000
        stats_lock = threading.Lock()

        def _maybe_log_progress():
            nonlocal next_log_at
            while total_samples >= next_log_at:
                filtered = total_boxes_in - total_boxes_kept
                print(
                    f"[Sam3Refiner] {total_samples} samples | "
                    f"bbox in={total_boxes_in} filtered={filtered} kept={total_boxes_kept} | "
                    f"samples saved={total_samples_saved}",
                    flush=True,
                )
                next_log_at += 1000

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            pbar = tqdm.tqdm(total=len(examples), desc="SAM3 samples")
            for chunk_start in range(0, len(batches), window):
                chunk = batches[chunk_start : chunk_start + window]
                futures = {executor.submit(self._process_batch, b): None for b in chunk}
                for future in as_completed(futures):
                    try:
                        outs, st = future.result()
                        processed.extend(outs)
                        with stats_lock:
                            total_samples += st["samples"]
                            total_boxes_in += st["boxes_in"]
                            total_boxes_kept += st["boxes_kept"]
                            total_samples_saved += st["samples_saved"]
                            pbar.update(st["samples"])
                            _maybe_log_progress()
                    except Exception as exc:
                        print(f"[WARN] SAM3 batch failed: {exc}")
            pbar.close()

        if total_samples > 0 and total_samples % 1000 != 0:
            filtered = total_boxes_in - total_boxes_kept
            print(
                f"[Sam3Refiner] {total_samples} samples (final) | "
                f"bbox in={total_boxes_in} filtered={filtered} kept={total_boxes_kept} | "
                f"samples saved={total_samples_saved}",
                flush=True,
            )

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
