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


def _post_process(processor, outputs, min_score: float, target_sizes: list):
    return processor.post_process_instance_segmentation(
        outputs,
        threshold=min_score,
        mask_threshold=0.5,
        target_sizes=target_sizes,
    )


# Running stats for score diagnostics (logged every SCORE_LOG_INTERVAL images).
_score_log_state: dict = {"count": 0, "total": 0, "below_06": 0, "below_03": 0}
SCORE_LOG_INTERVAL = 500


def _log_score_stats(scores_np: np.ndarray, n_boxes: int) -> None:
    """Accumulate and periodically print SAM3 IoU score statistics."""
    s = _score_log_state
    s["count"] += 1
    s["total"] += len(scores_np)
    s["below_06"] += int((scores_np < 0.6).sum())
    s["below_03"] += int((scores_np < 0.3).sum())
    if s["count"] % SCORE_LOG_INTERVAL == 0:
        t = s["total"]
        pct_06 = 100.0 * s["below_06"] / t if t else 0.0
        pct_03 = 100.0 * s["below_03"] / t if t else 0.0
        print(
            f"[Sam3Refiner] score-diag after {s['count']} images | "
            f"masks={t} below_0.6={s['below_06']}({pct_06:.1f}%) "
            f"below_0.3={s['below_03']}({pct_03:.1f}%)",
            flush=True,
        )


def _best_mask_from_seg(seg_item):
    scores = seg_item["scores"]
    if scores is None or len(scores) == 0:
        return None, 0.0
    best = int(scores.argmax())
    m = seg_item["masks"][best]
    if hasattr(m, "float"):
        m = m.float()
    if hasattr(m, "cpu"):
        m = m.cpu().numpy()
    if m.ndim == 3:
        m = m[0]
    return m, float(scores[best])


def _extract_mask(m) -> np.ndarray:
    """Convert a SAM3 mask tensor/array to a 2-D float32 numpy array."""
    if hasattr(m, "float"):
        m = m.float()
    if hasattr(m, "cpu"):
        m = m.cpu().numpy()
    if m.ndim == 3:
        m = m[0]
    return m


def _infer_refiner(processor, model, device, images, boxes_per_image, texts_per_image):
    """Single SAM3 forward for the whole batch: each entry = one image + ALL its boxes.

    Batch layout (correct for SAM3 / video model):
      entry 0 → img_0,  [box_0_0, box_0_1, …]
      entry 1 → img_1,  [box_1_0, box_1_1, …]
      …

    Why this matters: SAM3 applies temporal attention across the batch dimension.
    The original "one-box-per-entry" design repeated every image N_boxes times,
    so the model saw N copies of the same frame as a "video" — corrupting its IoU
    predictions.  Grouping all boxes under their parent image eliminates that
    confusion while keeping the full batch on the NPU in one forward pass.

    Quality gate: threshold=0.0 in post_process keeps every mask candidate; only
    MIN_MASK_PIXELS is used to discard empty results.  Depth-projected box prompts
    are coarser than GroundingDINO boxes, so their IoU self-estimates are lower and
    a strict threshold would silently drop valid masks.
    """
    # Normalise boxes; track which original indices have valid boxes
    all_boxes: list[np.ndarray] = []
    valid_idx: list[int] = []
    for i, boxes in enumerate(boxes_per_image):
        b = np.asarray(boxes, dtype=np.float32)
        if b.ndim == 1:
            b = b.reshape(1, 4)
        all_boxes.append(b)
        if b.shape[0] > 0:
            valid_idx.append(i)

    # Pre-fill output with empty results for images that had no boxes
    per_image: list = [([], np.array([])) for _ in images]

    if not valid_idx:
        return per_image

    batch_images = [images[i] for i in valid_idx]
    batch_boxes = [all_boxes[i].tolist() for i in valid_idx]
    batch_labels = [[1] * len(all_boxes[i]) for i in valid_idx]
    batch_texts = (
        [texts_per_image[i] for i in valid_idx] if texts_per_image else None
    )

    proc_kwargs = dict(
        images=batch_images,
        input_boxes=batch_boxes,
        input_boxes_labels=batch_labels,
        return_tensors="pt",
    )
    if batch_texts and any(t is not None for t in batch_texts):
        proc_kwargs["text"] = [t or "" for t in batch_texts]

    inputs = processor(**proc_kwargs).to(device)
    target_sizes = _target_sizes(inputs)

    with torch.no_grad():
        outputs = model(**inputs)

    # threshold=0.0 — let MIN_MASK_PIXELS be the sole quality gate
    seg_list = _post_process(processor, outputs, 0.0, target_sizes)

    for seg_pos, orig_idx in enumerate(valid_idx):
        seg = seg_list[seg_pos]
        seg_masks = seg.get("masks")
        seg_scores = seg.get("scores")
        n_got = len(seg_masks) if seg_masks is not None else 0
        h_px, w_px = int(target_sizes[seg_pos][0]), int(target_sizes[seg_pos][1])
        boxes = all_boxes[orig_idx]

        if seg_scores is not None and len(seg_scores) > 0:
            scores_np = (
                seg_scores.float().cpu().numpy()
                if hasattr(seg_scores, "cpu")
                else np.asarray(seg_scores, dtype=np.float32)
            )
            _log_score_stats(scores_np, n_boxes=len(boxes))

        masks_out, scores_out = [], []
        for k in range(len(boxes)):
            if k < n_got:
                sc = (
                    float(seg_scores[k])
                    if seg_scores is not None and k < len(seg_scores)
                    else 0.0
                )
                masks_out.append(_extract_mask(seg_masks[k]))
                scores_out.append(sc)
            else:
                masks_out.append(np.zeros((h_px, w_px), dtype=np.float32))
                scores_out.append(0.0)

        per_image[orig_idx] = (masks_out, np.array(scores_out, dtype=np.float32))

    return per_image


def _filter_by_pixels(pred_masks, min_pixels: int):
    refined, keep_indices = [], []
    for i, arr in enumerate(pred_masks):
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


class Sam3Refiner(BaseTask):
    """Refine filter-stage masks with Sam3Model box prompts (Sam2Refiner replacement)."""

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

        replicas_per_device = int(args.get("replicas_per_device", 1))
        self._replica_pool: queue.Queue = queue.Queue()
        logged = False
        for dev in devices:
            for _ in range(replicas_per_device):
                proc, model = _load_sam3_replica(segmenter_model, {}, dev)
                if not logged:
                    print(f"[Sam3Refiner] Sam3Model on {dev}")
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
        texts = [". ".join(t) if t else None for t in tags_list]
        processor, model, dev = self._replica_pool.get()
        try:
            return _infer_refiner(
                processor, model, dev,
                images, boxes_per_image,
                texts if any(texts) else None,
            )
        finally:
            self._replica_pool.put((processor, model, dev))

    def _results_from_infer(self, per_image):
        out = []
        for pred_masks, scores in per_image:
            refined, keep_indices = _filter_by_pixels(pred_masks, self.MIN_MASK_PIXELS)
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
                        print(f"[WARN] SAM3 batch failed: {exc}")
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
