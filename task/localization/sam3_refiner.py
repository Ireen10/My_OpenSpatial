import os
import queue
import threading
import warnings

# Suppress SyntaxWarnings from Ascend CANN internal libraries.
warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"tbe\..*")

import numpy as np
import pandas as pd
import torch
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

from task.base_task import BaseTask


def _try_patch_roi_align_for_npu() -> bool:
    """Replace torchvision.ops.roi_align with torch_npu.npu_roi_align on NPU."""
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
        roi_end_mode = 1 if aligned else 0
        npu_sample_num = max(0, sampling_ratio)

        if isinstance(boxes, (list, tuple)):
            per_image_boxes = boxes
        else:
            if boxes.shape[0] == 0:
                return input.new_zeros((0, input.shape[1], pooled_h, pooled_w))
            per_image_boxes = [
                boxes[boxes[:, 0] == i, 1:] for i in range(input.shape[0])
            ]

        results = []
        for i, b in enumerate(per_image_boxes):
            if b.shape[0] == 0:
                continue
            rois_i = torch.cat([b.new_zeros((b.shape[0], 1)), b.float()], dim=1)
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


_try_patch_roi_align_for_npu()


def _parse_devices(device_raw) -> list[str]:
    if isinstance(device_raw, (list, tuple)):
        return [str(d).strip() for d in device_raw if str(d).strip()]
    return [d.strip() for d in str(device_raw).split(",") if d.strip()]


def _set_npu_device(device: str) -> None:
    if not str(device).startswith("npu"):
        return
    try:
        dev_id = int(str(device).split(":")[-1]) if ":" in str(device) else 0
        torch.npu.set_device(dev_id)
    except AttributeError:
        pass


def _load_sam3_replica(segmenter_model: str, load_kwargs: dict, device: str):
    """Load Sam3Processor + Sam3Model. Requires transformers>=5.0.0."""
    is_local = (
        os.path.isabs(segmenter_model)
        or segmenter_model.startswith("./")
        or segmenter_model.startswith("../")
    )
    if is_local and not os.path.isdir(segmenter_model):
        raise FileNotFoundError(
            f"SAM3 weights directory not found: {segmenter_model!r}\n"
            "Check 'segmenter_model' in your YAML — absolute path or Hub id 'facebook/sam3'."
        )

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


_SEG_LOGGED = False


def _post_process(processor, outputs, min_score: float, target_sizes: list):
    global _SEG_LOGGED
    result = processor.post_process_instance_segmentation(
        outputs,
        threshold=min_score,
        mask_threshold=0.5,
        target_sizes=target_sizes,
    )
    if not _SEG_LOGGED:
        _SEG_LOGGED = True
        d = result[0] if isinstance(result, list) and result else result
        print(f"[Sam3Refiner] post_process -> {type(result).__name__}", flush=True)
        if isinstance(d, dict):
            for k, v in d.items():
                extra = f" {tuple(v.shape)} {v.dtype}" if hasattr(v, "shape") else ""
                print(f"  {k}: {type(v).__name__}{extra}", flush=True)
    return result


def _load_coarse_mask(path: str) -> np.ndarray:
    m = np.array(Image.open(path))
    if m.ndim == 3:
        m = m[..., 0]
    return m > 127


def _tags_per_box(tags, n_boxes: int) -> list[str]:
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
        tag = tags[i] if i < len(tags) else None
        if tag is not None and str(tag).strip():
            out.append(str(tag).strip())
        else:
            out.append("object")
    return out


def _best_mask_from_seg(seg_item) -> tuple[np.ndarray, float]:
    scores, masks = seg_item.get("scores"), seg_item.get("masks")
    if scores is None or masks is None or len(scores) == 0:
        return np.zeros((1, 1), dtype=np.float32), 0.0
    scores_np = scores.float().cpu().numpy() if hasattr(scores, "cpu") else np.asarray(scores)
    best = int(scores_np.argmax())
    mask = masks[best]
    if hasattr(mask, "float"):
        mask = mask.float()
    mask = mask.cpu().numpy() if hasattr(mask, "cpu") else np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[0]
    return mask, float(scores_np[best])


def _infer_refiner(
    processor,
    model,
    device,
    images,
    boxes_per_image,
    tags_per_image,
    prompt_batch_size: int = 32,
):
    """Batched text+positive-box refinement: one SAM3 row per (image, tag, box)."""
    per_image: list[tuple[list, list]] = []
    queries: list[dict] = []

    for img_idx, image in enumerate(images):
        boxes = np.asarray(boxes_per_image[img_idx], dtype=np.float32).reshape(-1, 4)
        n = len(boxes)
        if n == 0:
            per_image.append(([], []))
            continue
        tags = _tags_per_box(
            tags_per_image[img_idx] if tags_per_image else None, n,
        )
        per_image.append(([np.zeros((1, 1), dtype=np.float32)] * n, [0.0] * n))
        for obj_idx, (box, tag) in enumerate(zip(boxes, tags)):
            queries.append({
                "img_idx": img_idx, "obj_idx": obj_idx,
                "image": image, "text": tag, "box": box,
            })

    if not queries:
        return per_image

    batch_size = max(1, int(prompt_batch_size))
    for start in range(0, len(queries), batch_size):
        chunk = queries[start : start + batch_size]
        inputs = processor(
            images=[q["image"] for q in chunk],
            text=[q["text"] for q in chunk],
            input_boxes=[[q["box"].tolist()] for q in chunk],
            input_boxes_labels=[[1] for _ in chunk],
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        seg_list = _post_process(processor, outputs, 0.6, _target_sizes(inputs))
        for seg, q in zip(seg_list, chunk):
            mask, score = _best_mask_from_seg(seg)
            print(f"score: {score}")
            masks, scores = per_image[q["img_idx"]]
            masks[q["obj_idx"]] = mask
            scores[q["obj_idx"]] = score

    return per_image


class Sam3Refiner(BaseTask):
    """Refine filter-stage masks with Sam3Model hybrid text+box prompts."""

    MIN_SCORE = 0.6
    MIN_MASK_PIXELS = 20

    def __init__(self, args, device=None):
        super().__init__(args)
        segmenter_model = args.get("segmenter_model", "facebook/sam3")
        devices = _parse_devices(
            args.get("device") or device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.device = devices[0]
        self.min_score = float(args.get("min_score", self.MIN_SCORE))
        self.min_mask_pixels = int(args.get("min_mask_pixels", self.MIN_MASK_PIXELS))
        self.prompt_batch_size = int(args.get("prompt_batch_size", 32))

        replicas_per_device = int(args.get("replicas_per_device", 1))
        self._replica_pool: queue.Queue = queue.Queue()
        for dev in devices:
            for _ in range(replicas_per_device):
                proc, model = _load_sam3_replica(segmenter_model, {}, dev)
                model.eval()
                self._replica_pool.put((proc, model, dev))

        total_replicas = len(devices) * replicas_per_device
        if total_replicas > 1:
            self.use_multi_processing = True
            if not args.get("num_workers"):
                args["num_workers"] = total_replicas

        print(
            f"[Sam3Refiner] Sam3Model on {','.join(devices)} | "
            f"replicas={total_replicas} prompt_batch_size={self.prompt_batch_size} "
            f"pipeline_batch_size={args.get('batch_size', 4)} "
            f"min_score={self.min_score:.2f} min_mask_pixels={self.min_mask_pixels}",
            flush=True,
        )

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
            _set_npu_device(dev)
            return _infer_refiner(
                processor, model, dev,
                images, boxes_per_image, tags_list,
                prompt_batch_size=self.prompt_batch_size,
            )
        finally:
            self._replica_pool.put((processor, model, dev))

    def _filter_masks(self, pred_masks, scores):
        refined, keep_indices = [], []
        for i, arr in enumerate(pred_masks):
            if float(scores[i]) < self.min_score:
                continue
            arr = np.asarray(arr) > 0
            if np.sum(arr) > self.min_mask_pixels:
                refined.append(arr)
                keep_indices.append(i)
        if not keep_indices:
            return [], [], []
        bboxes = self._masks_to_bboxes([m.astype(bool) for m in refined]).tolist()
        return refined, bboxes, keep_indices

    def _load_example(self, example):
        image = Image.open(example["image"])
        if image.mode != "RGB":
            image = image.convert("RGB")
        coarse = [_load_coarse_mask(p) for p in example["masks"]]
        return image, coarse

    def _save_example(self, example, idx, refined, bboxes_2d, keep_indices):
        self._filter_by_keep_indices(example, keep_indices)
        mask_files = self._save_masks(
            refined, os.path.join(self.output_dir, "masks"), str(idx),
        )
        if len(mask_files) != len(example["obj_tags"]):
            return None
        if len(mask_files) != len(example["bboxes_3d_world_coords"]):
            return None
        example["masks"] = mask_files
        example["bboxes_2d"] = bboxes_2d
        return example

    def _process_batch(self, batch_items):
        valid_items = []
        for idx, example in batch_items:
            try:
                self.validate_example(example)
                image, coarse = self._load_example(example)
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
        boxes_in = sum(len(x[3]) for x in valid_items)
        boxes_kept = 0
        outputs = []

        for (idx, example, _, _), (pred_masks, scores) in zip(valid_items, per_image):
            refined, bboxes_2d, keep_indices = self._filter_masks(pred_masks, scores)
            boxes_kept += len(keep_indices)
            if not keep_indices:
                continue
            saved = self._save_example(example, idx, refined, bboxes_2d, keep_indices)
            if saved is not None:
                outputs.append(saved)

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
        for key in self.args.get("update_keys", []):
            example[key] = [example[key][i] for i in keep_indices]
        return example

    def apply_transform(self, example, idx):
        self.validate_example(example)
        image, coarse = self._load_example(example)
        pred_masks, scores = self._refine([image], [coarse], [example.get("obj_tags")])[0]
        refined, bboxes_2d, keep_indices = self._filter_masks(pred_masks, scores)
        if not keep_indices:
            return None, False

        saved = self._save_example(example, idx, refined, bboxes_2d, keep_indices)
        assert saved is not None
        return saved, True
