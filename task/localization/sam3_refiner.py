import queue
import threading
import warnings

# Suppress SyntaxWarnings from Ascend CANN internal libraries (invalid escape
# sequences in tbe/dsl/unify_schedule/**).  These are third-party issues that
# do not affect correctness and would otherwise pollute every worker's log.
warnings.filterwarnings(
    "ignore",
    category=SyntaxWarning,
    module=r"tbe\..*",
)

import numpy as np
import torch
from PIL import Image
import os
import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from task.base_task import BaseTask


def _try_patch_roi_align_for_npu() -> bool:
    """Replace torchvision.ops.roi_align with torch_npu.npu_roi_align.

    npu_roi_align API (confirmed from CANN docs + example):
      npu_roi_align(features, rois, spatial_scale, pooled_h, pooled_w,
                    sample_num, roi_end_mode)
      features : [1, C, H, W]  — single image only
      rois     : [N, 5]        — [batch_idx, x0, y0, x1, y1]
                 batch_idx is always 0 since features is single-image

    torchvision roi_align passes boxes in two formats:
      a) tuple/list of per-image tensors, each [K_i, 4] (no batch_idx)
      b) concatenated [N, 5] tensor with batch_idx in column 0

    Both cases are handled by looping over images and calling npu_roi_align
    once per image with features=[1,C,H,W] and rois=[K_i,5] (batch_idx=0).
    """
    try:
        import torchvision.ops as _tvops
        import torch_npu  # noqa: F401
        _npu_fn = torch_npu.npu_roi_align
    except (ImportError, AttributeError):
        return False

    if getattr(_tvops.roi_align, "_npu_patched", False):
        return True

    def _roi_align_npu(
        input, boxes, output_size,
        spatial_scale=1.0, sampling_ratio=-1, aligned=False,
    ):
        if isinstance(output_size, (int, float)):
            pooled_h = pooled_w = int(output_size)
        else:
            pooled_h, pooled_w = int(output_size[0]), int(output_size[1])
        roi_end_mode   = 1 if aligned else 0
        npu_sample_num = max(0, sampling_ratio)   # torchvision -1 → CANN 0

        if isinstance(boxes, (list, tuple)):
            # Format (a): per-image [K_i, 4] tensors, no batch_idx
            per_image_boxes = boxes
        else:
            # Format (b): concatenated [N, 5] with batch_idx in col 0
            if boxes.shape[0] == 0:
                return input.new_zeros((0, input.shape[1], pooled_h, pooled_w))
            per_image_boxes = [
                boxes[boxes[:, 0] == i, 1:] for i in range(input.shape[0])
            ]

        results = []
        for i, b in enumerate(per_image_boxes):
            if b.shape[0] == 0:
                continue
            # npu_roi_align: features=[1,C,H,W], rois=[K,5] with batch_idx=0
            rois_i = torch.cat(
                [b.new_zeros((b.shape[0], 1)), b.float()], dim=1
            )
            results.append(_npu_fn(
                input[i : i + 1], rois_i,
                spatial_scale, pooled_h, pooled_w, npu_sample_num, roi_end_mode,
            ))

        if not results:
            return input.new_zeros((0, input.shape[1], pooled_h, pooled_w))
        return torch.cat(results, dim=0)

    _roi_align_npu._npu_patched = True
    _tvops.roi_align = _roi_align_npu
    try:
        import torchvision.ops.poolers as _poolers
        if hasattr(_poolers, "roi_align"):
            _poolers.roi_align = _roi_align_npu
    except (ImportError, AttributeError):
        pass
    return True


_NPU_ROI_ALIGN_PATCHED = _try_patch_roi_align_for_npu()


def _validate_model_path(model_name_or_path: str) -> None:
    is_local = (
        os.path.isabs(model_name_or_path)
        or model_name_or_path.startswith("./")
        or model_name_or_path.startswith("../")
    )
    if is_local and not os.path.isdir(model_name_or_path):
        raise FileNotFoundError(
            f"SAM3 weights directory not found: {model_name_or_path!r}\n"
            "Check 'segmenter_model' in your YAML — absolute path or Hub id 'facebook/sam3'."
        )


def _load_sam3_replica(segmenter_model: str, load_kwargs: dict, device: str):
    """Load Sam3Processor + Sam3Model. Requires transformers>=5.0.0."""
    _validate_model_path(segmenter_model)
    try:
        from transformers import Sam3Processor, Sam3Model
    except ImportError as e:
        raise ImportError(
            "Sam3Processor/Sam3Model not found. Install transformers>=5.0.0."
        ) from e

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"You are using a model of type sam3_video",
            category=UserWarning,
        )
        try:
            proc = Sam3Processor.from_pretrained(segmenter_model)
            model = Sam3Model.from_pretrained(segmenter_model, **load_kwargs).to(device)
        except (OSError, ValueError) as e:
            raise RuntimeError(
                f"Failed to load Sam3Model from {segmenter_model!r}."
            ) from e
    return proc, model


def _target_sizes(inputs) -> list:
    sizes = inputs["original_sizes"]
    return sizes.tolist() if hasattr(sizes, "tolist") else list(sizes)


def _load_coarse_mask(path: str) -> np.ndarray:
    m = np.array(Image.open(path))
    if m.ndim == 3:
        m = m[..., 0]
    return m > 127


def _tags_per_box(tags, n_boxes: int) -> list[str]:
    """One SAM3 text prompt per box, aligned with obj_tags indices."""
    if n_boxes <= 0:
        return []
    if tags is None:
        return ["object"] * n_boxes
    if isinstance(tags, np.ndarray):
        tags = tags.tolist()
    elif not isinstance(tags, (list, tuple)):
        tags = [tags]
    out: list[str] = []
    for i in range(n_boxes):
        if i < len(tags) and tags[i] is not None and str(tags[i]).strip():
            out.append(str(tags[i]).strip())
        else:
            out.append("object")
    return out


def _post_process(processor, outputs, min_score: float, target_sizes: list):
    return processor.post_process_instance_segmentation(
        outputs,
        threshold=min_score,
        mask_threshold=0.5,
        target_sizes=target_sizes,
    )


# Running keep/drop statistics (logged every SCORE_LOG_INTERVAL images).
_score_log_state: dict = {
    "images": 0, "boxes_in": 0, "boxes_kept": 0,
}
SCORE_LOG_INTERVAL = 500


def _log_score_stats(scores_np: np.ndarray, min_score: float) -> None:
    """Accumulate and periodically print keep/drop statistics by confidence."""
    s = _score_log_state
    s["images"] += 1
    s["boxes_in"] += len(scores_np)
    s["boxes_kept"] += int((np.asarray(scores_np, dtype=np.float32) >= min_score).sum())

    if s["images"] % SCORE_LOG_INTERVAL == 0:
        total_in = s["boxes_in"]
        total_kept = s["boxes_kept"]
        pct = 100.0 * total_kept / total_in if total_in else 0.0
        print(
            f"[Sam3Refiner] after {s['images']} images | "
            f"boxes_in={total_in}  "
            f"score>={min_score:.2f} {total_kept} ({pct:.1f}%)  "
            f"dropped={total_in - total_kept} ({100.0 - pct:.1f}%)",
            flush=True,
        )


def _normalize_masks_from_seg(seg_item, expected_count: int) -> np.ndarray:
    """Masks from post_process_instance_segmentation for one image (num_obj, H, W)."""
    masks_raw = seg_item.get("masks")
    if masks_raw is None:
        return np.empty((0,), dtype=np.float32)
    if hasattr(masks_raw, "detach"):
        arr = masks_raw.detach().float().cpu().numpy()
    elif hasattr(masks_raw, "cpu"):
        arr = masks_raw.float().cpu().numpy()
    else:
        arr = np.asarray(masks_raw, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[:, 0]
    if arr.ndim == 2:
        arr = arr[None, ...]
    return arr[:expected_count]


def _normalize_scores_from_seg(seg_item, expected_count: int) -> np.ndarray:
    """Confidence scores aligned with input boxes for one image."""
    scores = seg_item.get("scores")
    if scores is None:
        return np.zeros((0,), dtype=np.float32)
    if hasattr(scores, "detach"):
        arr = scores.detach().float().cpu().numpy()
    elif hasattr(scores, "cpu"):
        arr = scores.float().cpu().numpy()
    else:
        arr = np.asarray(scores, dtype=np.float32)
    return arr.reshape(-1)[:expected_count]


def _normalize_boxes_array(boxes) -> np.ndarray:
    b = np.asarray(boxes, dtype=np.float32)
    if b.ndim == 1:
        b = b.reshape(1, 4) if b.size == 4 else b.reshape(-1, 4)
    return b


def _prepare_hybrid_batch(images, boxes_per_image, tags_per_image):
    """Build Sam3Processor nested batch: M images, variable boxes/text per image."""
    valid_indices: list[int] = []
    valid_images = []
    valid_boxes: list[list[list[float]]] = []
    valid_texts: list[list[str]] = []
    valid_labels: list[list[int]] = []
    for img_idx, image in enumerate(images):
        boxes = _normalize_boxes_array(boxes_per_image[img_idx])
        if boxes.shape[0] == 0:
            continue
        tags = _tags_per_box(
            tags_per_image[img_idx] if tags_per_image else None,
            boxes.shape[0],
        )
        valid_indices.append(img_idx)
        valid_images.append(image)
        valid_boxes.append(boxes.astype(float).tolist())
        valid_texts.append(tags)
        valid_labels.append([1] * boxes.shape[0])
    return valid_indices, valid_images, valid_boxes, valid_texts, valid_labels


def _infer_refiner(processor, model, device, images, boxes_per_image, tags_per_image):
    """One SAM3 forward for M images; per-box text + positive box prompts.

    ``post_process_instance_segmentation`` returns one result per image with
    masks/boxes/scores aligned to the input prompts (same role as SAM2).
    """
    per_image: list = [([], []) for _ in images]
    (
        valid_indices,
        valid_images,
        valid_boxes,
        valid_texts,
        valid_labels,
    ) = _prepare_hybrid_batch(images, boxes_per_image, tags_per_image)
    if not valid_indices:
        return per_image

    inputs = processor(
        images=valid_images,
        text=valid_texts,
        input_boxes=valid_boxes,
        input_boxes_labels=valid_labels,
        return_tensors="pt",
    ).to(device)
    target_sizes = _target_sizes(inputs)

    with torch.no_grad():
        outputs = model(**inputs)

    seg_list = _post_process(processor, outputs, 0.0, target_sizes)

    for seg_pos, orig_idx in enumerate(valid_indices):
        expected_count = len(_normalize_boxes_array(boxes_per_image[orig_idx]))
        seg = seg_list[seg_pos]
        masks_np = _normalize_masks_from_seg(seg, expected_count)
        scores_np = _normalize_scores_from_seg(seg, expected_count)
        per_image[orig_idx] = (list(masks_np), scores_np.tolist())

    return per_image


class Sam3Refiner(BaseTask):
    """Refine filter-stage masks with Sam3Model hybrid text+box prompts."""

    MIN_SCORE = 0.6
    MIN_MASK_PIXELS = 20

    def __init__(self, args, device=None):
        super().__init__(args)
        segmenter_model = args.get("segmenter_model", "facebook/sam3")
        device_raw = args.get("device") or device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if isinstance(device_raw, (list, tuple)):
            devices = [str(d).strip() for d in device_raw]
        else:
            devices = [d.strip() for d in str(device_raw).split(",")]
        self.device = devices[0]

        self.min_score = float(args.get("min_score", self.MIN_SCORE))
        self.min_mask_pixels = int(args.get("min_mask_pixels", self.MIN_MASK_PIXELS))
        self.pipeline_batch_size = int(args.get("batch_size", 4))

        replicas_per_device = int(args.get("replicas_per_device", 1))
        self._replica_pool: queue.Queue = queue.Queue()
        logged = False
        for dev in devices:
            for _ in range(replicas_per_device):
                proc, model = _load_sam3_replica(segmenter_model, {}, dev)
                if not logged:
                    print(
                        f"[Sam3Refiner] Sam3Model on {dev} | "
                        f"pipeline_batch_size={self.pipeline_batch_size} "
                        f"min_score={self.min_score} "
                        f"min_mask_pixels={self.min_mask_pixels}",
                        flush=True,
                    )
                    logged = True
                model.eval()
                self._replica_pool.put((proc, model, dev))

        total_replicas = len(devices) * replicas_per_device
        if total_replicas > 1:
            self.use_multi_processing = True
            if not args.get("num_workers"):
                args["num_workers"] = total_replicas

        assert "update_keys" in args, "update_keys must be specified in args."
        self.output_dir = os.path.join(self.args.get("output_dir"), self.args.get("file_name"))

    @staticmethod
    def _masks_to_bboxes(masks):
        boxes = []
        for mask in masks:
            ys, xs = np.where(mask)
            if len(xs) > 0:
                boxes.append([np.min(xs), np.min(ys), np.max(xs), np.max(ys)])
            else:
                boxes.append([0, 0, 0, 0])
        return np.array(boxes)

    def _refine(self, images, masks_list, tags_list):
        boxes_per_image = [self._masks_to_bboxes(m) for m in masks_list]
        processor, model, dev = self._replica_pool.get()
        try:
            # torch.npu.set_device is thread-local; bind the thread to the
            # correct device before any CANN call so the device context is
            # correct regardless of which thread in the pool picks this task.
            try:
                dev_id = int(dev.split(":")[-1]) if ":" in dev else 0
                torch.npu.set_device(dev_id)
            except AttributeError:
                pass  # non-NPU environment
            return _infer_refiner(
                processor, model, dev,
                images, boxes_per_image, tags_list,
            )
        finally:
            self._replica_pool.put((processor, model, dev))

    def _results_from_infer(self, per_image):
        out = []
        for pred_masks, scores in per_image:
            _log_score_stats(np.asarray(scores, dtype=np.float32), self.min_score)
            refined, keep_indices = [], []
            for i, arr in enumerate(pred_masks):
                score = float(scores[i]) if i < len(scores) else 1.0
                if score < self.min_score:
                    continue
                arr = np.asarray(arr) > 0
                if np.sum(arr) > self.min_mask_pixels:
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

        per_image = self._refine(
            [x[2] for x in valid_items],
            [x[3] for x in valid_items],
            [x[1].get("obj_tags") for x in valid_items],
        )
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
        num_workers = self.args.get("num_workers", 4)
        batch_size = int(self.args.get("batch_size", 4))
        examples = list(enumerate(dataset.to_dict("records")))
        batches = [examples[i : i + batch_size] for i in range(0, len(examples), batch_size)]
        window = num_workers * 2

        processed = []
        total_samples = total_boxes_in = total_boxes_kept = total_samples_saved = 0
        next_log_at = 1000
        lock = threading.Lock()

        def _log():
            nonlocal next_log_at
            while total_samples >= next_log_at:
                print(
                    f"[Sam3Refiner] {total_samples} samples | "
                    f"bbox in={total_boxes_in} filtered={total_boxes_in - total_boxes_kept} "
                    f"kept={total_boxes_kept} | samples saved={total_samples_saved}",
                    flush=True,
                )
                next_log_at += 1000

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            pbar = tqdm.tqdm(total=len(examples), desc="SAM3 samples")
            for chunk_start in range(0, len(batches), window):
                chunk = batches[chunk_start : chunk_start + window]
                futures = [executor.submit(self._process_batch, b) for b in chunk]
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
                        print(f"[WARN] SAM3 batch failed: {exc}")
                        print(_tb.format_exc())
            pbar.close()

        if total_samples > 0 and total_samples % 1000 != 0:
            print(
                f"[Sam3Refiner] {total_samples} samples (final) | "
                f"bbox in={total_boxes_in} filtered={total_boxes_in - total_boxes_kept} "
                f"kept={total_boxes_kept} | samples saved={total_samples_saved}",
                flush=True,
            )
        return pd.DataFrame(processed).reset_index(drop=True) if processed else pd.DataFrame()

    def run(self, dataset):
        if self.use_multi_processing:
            return self._run_batched(dataset)
        return super().run(dataset)

    def _save_masks(self, masks, mask_dir, prefix):
        os.makedirs(mask_dir, exist_ok=True)
        paths = []
        for i, mask in enumerate(masks):
            path = os.path.join(mask_dir, f"example_{prefix}_box_{i}_mask.png")
            Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(path)
            paths.append(path)
        return paths

    def validate_example(self, example):
        for key in ("image", "masks", "obj_tags"):
            if key not in example:
                raise ValueError(f"{key} not found in example")
        if len(example["obj_tags"]) == 0:
            raise ValueError("obj_tags is empty")

    def _filter_by_keep_indices(self, example, keep_indices):
        update_keys = self.args.get("update_keys", [])
        if not update_keys or keep_indices is None:
            return example
        for key in update_keys:
            example[key] = [example[key][i] for i in keep_indices]
        return example

    def apply_transform(self, example, idx):
        self.validate_example(example)
        image = Image.open(example["image"])
        if image.mode != "RGB":
            image = image.convert("RGB")
        coarse = [_load_coarse_mask(p) for p in example["masks"]]
        refined, bboxes_2d, keep_indices = self._results_from_infer(
            self._refine([image], [coarse], [example.get("obj_tags")])
        )[0]
        if not keep_indices:
            return None, False

        self._filter_by_keep_indices(example, keep_indices)
        mask_files = self._save_masks(refined, os.path.join(self.output_dir, "masks"), str(idx))
        assert len(mask_files) == len(example["obj_tags"])
        assert len(mask_files) == len(example["bboxes_3d_world_coords"])
        example["masks"] = mask_files
        example["bboxes_2d"] = bboxes_2d
        return example, True
