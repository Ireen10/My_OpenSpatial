#!/usr/bin/env python3
"""
Standalone verifier for SAM3 localization modules (transformers model-loading variant).

Supports:
1) Localizer only (GroundingDINO + SAM3)
2) Refiner only (SAM3 box-prompt refine)
3) End-to-end localizer -> refiner on one image

Examples:
  # Localizer only
  python verification/dataset_pipeline/verify_sam3_transformers.py \
    --mode localizer \
    --image /path/to/image.jpg \
    --tags chair,table \
    --device npu:0 \
    --segmenter_checkpoint_path /path/to/sam3

  # Refiner only with existing coarse masks
  python verification/dataset_pipeline/verify_sam3_transformers.py \
    --mode refiner \
    --image /path/to/image.jpg \
    --mask_paths /path/to/mask0.png /path/to/mask1.png \
    --device npu:0 \
    --segmenter_checkpoint_path /path/to/sam3

  # Chain both modules
  python verification/dataset_pipeline/verify_sam3_transformers.py \
    --mode both \
    --image /path/to/image.jpg \
    --tags chair,table \
    --device npu:0 \
    --segmenter_checkpoint_path /path/to/sam3
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from transformers import Sam3Model, Sam3Processor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.data_utils import merge_overlapping_boxes, merge_overlapping_masks


class Localizer:
    """Grounding DINO + SAM3 pipeline using transformers model loading."""

    MIN_SCORE = 0.7

    def __init__(self, args):
        grounding_model = args.get("grounding_model", "IDEA-Research/grounding-dino-base")
        self.device = self._resolve_device(args)
        self._prepare_npu_runtime_if_needed()

        segmenter_model = args.get("segmenter_model", "facebook/sam3")
        segmenter_checkpoint_path = args.get("segmenter_checkpoint_path")
        segmenter_source = segmenter_checkpoint_path or segmenter_model

        self.processor = AutoProcessor.from_pretrained(grounding_model)
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_model).to(
            self.device
        )
        self.detector.eval()

        self.sam3_processor = Sam3Processor.from_pretrained(segmenter_source)
        self.sam3_model = Sam3Model.from_pretrained(segmenter_source).to(self.device)
        self.sam3_model.eval()
        processor_name = type(self.sam3_processor).__name__.lower()
        if "video" in processor_name:
            raise ValueError(
                "Loaded a SAM3 video processor for segmenter_source. "
                "Please pass an image SAM3 model directory/repo for --segmenter_model "
                "or --segmenter_checkpoint_path."
            )
        _log(
            "LOCALIZER",
            (
                f"loaded SAM3 via transformers: "
                f"processor={type(self.sam3_processor).__name__}, "
                f"model={type(self.sam3_model).__name__}"
            ),
        )

    @staticmethod
    def _resolve_device(args):
        explicit = args.get("device")
        if explicit:
            return explicit
        if hasattr(torch, "npu") and torch.npu.is_available():
            return "npu:0"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _prepare_npu_runtime_if_needed(self):
        if isinstance(self.device, str) and self.device.startswith("npu"):
            if not hasattr(torch, "npu"):
                raise RuntimeError(
                    "NPU device requested but torch.npu is unavailable. "
                    "Please install/enable torch_npu for Ascend runtime."
                )
            torch.npu.set_device(self.device)

    @staticmethod
    def _move_to_device(batch, device):
        moved = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(device)
            else:
                moved[key] = value
        return moved

    def _run_sam3_box_prompt(self, image: Image.Image, box_xyxy):
        box = [float(v) for v in box_xyxy]
        inputs = self.sam3_processor(
            images=image,
            text="visual",
            input_boxes=[[box]],
            input_boxes_labels=[[1]],
            return_tensors="pt",
        )
        inputs = self._move_to_device(inputs, self.device)
        with torch.no_grad():
            outputs = self.sam3_model(**inputs, multimask_output=True)

        pred_masks = getattr(outputs, "pred_masks", None)
        pred_logits = getattr(outputs, "pred_logits", None)
        if pred_masks is None or pred_logits is None or pred_logits.shape[0] == 0:
            return None, None

        score_candidates = pred_logits[0].detach().float().sigmoid()
        presence_logits = getattr(outputs, "presence_logits", None)
        if presence_logits is not None and presence_logits.shape[0] > 0:
            score_candidates = score_candidates * presence_logits[0].detach().float().sigmoid()

        results = self.sam3_processor.post_process_instance_segmentation(
            outputs,
            threshold=0.0,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist(),
        )
        if not results:
            return None, None
        result = results[0]
        mask_candidates = result.get("masks")
        if mask_candidates is None or len(mask_candidates) == 0:
            return None, None
        valid_len = min(len(mask_candidates), len(score_candidates))
        if valid_len == 0:
            return None, None
        mask_candidates = mask_candidates[:valid_len]
        score_candidates = score_candidates[:valid_len]

        keep = score_candidates > self.MIN_SCORE
        if not bool(torch.any(keep)):
            return None, None
        kept_indices = torch.nonzero(keep, as_tuple=False).flatten()
        kept_scores = score_candidates[kept_indices]
        best_keep_idx = int(torch.argmax(kept_scores).item())
        best_idx = int(kept_indices[best_keep_idx].item())

        best_score = float(score_candidates[best_idx].item())
        best_mask = mask_candidates[best_idx]
        if best_mask.ndim == 2:
            best_mask = best_mask.unsqueeze(0)
        best_mask_np = (best_mask.detach().cpu().numpy() > 0).astype(np.uint8)
        return best_mask_np, best_score

    def _sam3_masks_from_boxes(self, image, det_boxes):
        masks, scores = [], []
        invalid_box_count = 0
        prompt_fail_count = 0
        for box_xyxy in det_boxes:
            if (box_xyxy[2] <= box_xyxy[0]) or (box_xyxy[3] <= box_xyxy[1]):
                invalid_box_count += 1
                continue
            mask_np, best_score = self._run_sam3_box_prompt(image, box_xyxy)
            if mask_np is None:
                prompt_fail_count += 1
                continue
            masks.append(mask_np)
            scores.append(best_score)

        _log(
            "LOCALIZER",
            (
                f"SAM3 box prompts: input_boxes={len(det_boxes)}, "
                f"invalid_boxes={invalid_box_count}, prompt_failed={prompt_fail_count}, "
                f"masks_generated={len(masks)}"
            ),
        )
        if len(masks) == 0:
            return None, None
        return np.stack(masks, axis=0), np.array(scores)

    def detect_and_segment(self, image, obj_tags):
        text_prompt = ". ".join(obj_tags)
        _log("LOCALIZER", f"running grounding detector with {len(obj_tags)} tags")
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.detector(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.3,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]],
        )
        det_tags = results[0]["text_labels"]
        det_boxes = results[0]["boxes"].cpu().numpy()
        _log(
            "LOCALIZER",
            f"grounding detections before merge: count={len(det_tags)}",
        )

        if len(det_tags) <= 1:
            return None

        before_merge = len(det_tags)
        det_boxes, det_tags = merge_overlapping_boxes(det_tags, det_boxes, overlap_threshold=0.8)
        _log(
            "LOCALIZER",
            f"detections after box merge: {before_merge} -> {len(det_tags)}",
        )
        masks, scores = self._sam3_masks_from_boxes(image, det_boxes)
        if masks is None or scores is None:
            return None

        valid_len = min(len(det_tags), len(det_boxes), len(scores), len(masks))
        det_tags = det_tags[:valid_len]
        det_boxes = det_boxes[:valid_len]
        masks = masks[:valid_len]
        scores = scores[:valid_len]
        _log(
            "LOCALIZER",
            (
                f"raw mask scores before filter: "
                f"{[round(float(s), 4) for s in scores.tolist()]}"
            ),
        )
        if len(scores) > 0:
            _log(
                "LOCALIZER",
                (
                    f"score stats: min={float(np.min(scores)):.4f}, "
                    f"max={float(np.max(scores)):.4f}, mean={float(np.mean(scores)):.4f}"
                ),
            )

        keep = [i for i, score in enumerate(scores) if score >= self.MIN_SCORE]
        _log(
            "LOCALIZER",
            f"masks after score filter (>= {self.MIN_SCORE}): {len(keep)}/{len(scores)}",
        )
        if len(keep) == 0:
            return None

        masks = masks[keep]
        det_tags = [det_tags[i] for i in keep]
        det_boxes = det_boxes[keep]

        pre_mask_merge = len(det_tags)
        masks, det_tags, det_boxes = merge_overlapping_masks(
            masks, det_tags, det_boxes, overlap_threshold=0.8
        )
        _log(
            "LOCALIZER",
            f"masks after overlap merge: {pre_mask_merge} -> {len(det_tags)}",
        )
        if len(det_tags) <= 1:
            return None

        return masks, det_boxes.tolist(), det_tags


class Sam3Refiner:
    """Refine coarse masks using SAM3 geometric box prompts."""

    MIN_SCORE = 0.6
    MIN_MASK_PIXELS = 20

    def __init__(self, args):
        self.device = self._resolve_device(args)
        self._prepare_npu_runtime_if_needed()

        segmenter_model = args.get("segmenter_model", "facebook/sam3")
        segmenter_checkpoint_path = args.get("segmenter_checkpoint_path")
        segmenter_source = segmenter_checkpoint_path or segmenter_model

        self.sam3_processor = Sam3Processor.from_pretrained(segmenter_source)
        self.sam3_model = Sam3Model.from_pretrained(segmenter_source).to(self.device)
        self.sam3_model.eval()
        processor_name = type(self.sam3_processor).__name__.lower()
        if "video" in processor_name:
            raise ValueError(
                "Loaded a SAM3 video processor for segmenter_source. "
                "Please pass an image SAM3 model directory/repo for --segmenter_model "
                "or --segmenter_checkpoint_path."
            )
        _log(
            "REFINER",
            (
                f"loaded SAM3 via transformers: "
                f"processor={type(self.sam3_processor).__name__}, "
                f"model={type(self.sam3_model).__name__}"
            ),
        )

    @staticmethod
    def _resolve_device(args):
        explicit = args.get("device")
        if explicit:
            return explicit
        if hasattr(torch, "npu") and torch.npu.is_available():
            return "npu:0"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _prepare_npu_runtime_if_needed(self):
        if isinstance(self.device, str) and self.device.startswith("npu"):
            if not hasattr(torch, "npu"):
                raise RuntimeError(
                    "NPU device requested but torch.npu is unavailable. "
                    "Please install/enable torch_npu for Ascend runtime."
                )
            torch.npu.set_device(self.device)

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

    @staticmethod
    def _move_to_device(batch, device):
        moved = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(device)
            else:
                moved[key] = value
        return moved

    def _run_sam3_box_prompt(self, image: Image.Image, box_xyxy):
        box = [float(v) for v in box_xyxy]
        inputs = self.sam3_processor(
            images=image,
            text="visual",
            input_boxes=[[box]],
            input_boxes_labels=[[1]],
            return_tensors="pt",
        )
        inputs = self._move_to_device(inputs, self.device)
        with torch.no_grad():
            outputs = self.sam3_model(**inputs, multimask_output=True)

        pred_masks = getattr(outputs, "pred_masks", None)
        pred_logits = getattr(outputs, "pred_logits", None)
        if pred_masks is None or pred_logits is None or pred_logits.shape[0] == 0:
            return None, None

        score_candidates = pred_logits[0].detach().float().sigmoid()
        presence_logits = getattr(outputs, "presence_logits", None)
        if presence_logits is not None and presence_logits.shape[0] > 0:
            score_candidates = score_candidates * presence_logits[0].detach().float().sigmoid()

        results = self.sam3_processor.post_process_instance_segmentation(
            outputs,
            threshold=0.0,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist(),
        )
        if not results:
            return None, None
        result = results[0]
        mask_candidates = result.get("masks")
        if mask_candidates is None or len(mask_candidates) == 0:
            return None, None
        valid_len = min(len(mask_candidates), len(score_candidates))
        if valid_len == 0:
            return None, None
        mask_candidates = mask_candidates[:valid_len]
        score_candidates = score_candidates[:valid_len]

        keep = score_candidates > self.MIN_SCORE
        if not bool(torch.any(keep)):
            return None, None
        kept_indices = torch.nonzero(keep, as_tuple=False).flatten()
        kept_scores = score_candidates[kept_indices]
        best_keep_idx = int(torch.argmax(kept_scores).item())
        best_idx = int(kept_indices[best_keep_idx].item())

        score = float(score_candidates[best_idx].item())
        mask = mask_candidates[best_idx]
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        mask_np = (mask.detach().cpu().numpy() > 0).astype(np.uint8)
        return mask_np, score

    def refine_masks(self, image, masks):
        input_boxes = self._masks_to_bboxes(masks)
        refined, bboxes_2d, keep_indices = [], [], []
        invalid_box_count = 0
        prompt_fail_count = 0
        low_score_count = 0
        tiny_mask_count = 0
        for i, box_xyxy in enumerate(input_boxes):
            if (box_xyxy[2] <= box_xyxy[0]) or (box_xyxy[3] <= box_xyxy[1]):
                invalid_box_count += 1
                continue

            mask_np, score = self._run_sam3_box_prompt(image, box_xyxy)
            if mask_np is None:
                prompt_fail_count += 1
                continue
            if score < self.MIN_SCORE:
                low_score_count += 1
                continue
            if np.sum(mask_np) <= self.MIN_MASK_PIXELS:
                tiny_mask_count += 1
                continue

            refined.append(mask_np)
            bboxes_2d.append(
                [int(box_xyxy[0]), int(box_xyxy[1]), int(box_xyxy[2]), int(box_xyxy[3])]
            )
            keep_indices.append(i)

        _log(
            "REFINER",
            (
                f"refine stats: input_masks={len(masks)}, valid_boxes={len(input_boxes) - invalid_box_count}, "
                f"prompt_failed={prompt_fail_count}, low_score={low_score_count}, "
                f"tiny_mask={tiny_mask_count}, kept={len(keep_indices)}"
            ),
        )
        if not keep_indices:
            return [], [], []
        return refined, bboxes_2d, keep_indices


def _log(stage: str, message: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{stage}] {message}", flush=True)


@contextlib.contextmanager
def _timed_stage(stage: str, start_message: str):
    _log(stage, start_message)
    t0 = time.perf_counter()
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - t0
        _log(stage, f"failed after {elapsed:.2f}s")
        raise
    elapsed = time.perf_counter() - t0
    _log(stage, f"done in {elapsed:.2f}s")


def _to_py_list(value):
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "tolist") and not isinstance(value, (list, tuple, dict, str)):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    return [value]


def _resolve_asset_path(path_value, raw_data_root: str | None) -> str:
    if not isinstance(path_value, str):
        raise ValueError(f"Expected a string asset path, got: {type(path_value)}")
    if os.path.isabs(path_value) or not raw_data_root:
        return path_value
    return os.path.normpath(os.path.join(raw_data_root, path_value))


def _select_view(value, view_idx: int):
    seq = _to_py_list(value)
    if not seq:
        return value
    if isinstance(seq[0], list):
        if view_idx >= len(seq):
            raise ValueError(f"view_idx={view_idx} out of range for sequence length {len(seq)}")
        return seq[view_idx]
    return value


def _infer_input_from_parquet(parsed):
    if parsed.parquet is None:
        return
    if not parsed.parquet.is_file():
        raise ValueError(f"Parquet not found: {parsed.parquet}")

    with _timed_stage("INPUT", f"loading parquet: {parsed.parquet}"):
        df = pd.read_parquet(parsed.parquet)
    if len(df) == 0:
        raise ValueError(f"Parquet is empty: {parsed.parquet}")
    if parsed.row_idx < 0 or parsed.row_idx >= len(df):
        raise ValueError(f"row_idx out of range: {parsed.row_idx}, len={len(df)}")

    row = df.iloc[parsed.row_idx]

    if parsed.image is None:
        if "image" not in row.index:
            raise ValueError("Parquet row does not contain 'image'.")
        row_image = _select_view(row["image"], parsed.view_idx)
        parsed.image = Path(_resolve_asset_path(row_image, parsed.raw_data_root))

    if parsed.tags is None and "obj_tags" in row.index:
        row_tags = _select_view(row["obj_tags"], parsed.view_idx)
        tags_list = [str(x) for x in _to_py_list(row_tags) if str(x).strip()]
        if tags_list:
            parsed.tags = ",".join(tags_list)

    if (parsed.mask_paths is None or len(parsed.mask_paths) == 0) and "masks" in row.index:
        row_masks = _select_view(row["masks"], parsed.view_idx)
        masks = _to_py_list(row_masks)
        if masks:
            parsed.mask_paths = [
                Path(_resolve_asset_path(str(mask_path), parsed.raw_data_root))
                for mask_path in masks
            ]


def _parse_tags(tags_text: str | None) -> list[str]:
    if not tags_text:
        return []
    return [t.strip() for t in tags_text.split(",") if t.strip()]


def _build_common_args(parsed) -> dict:
    args = {
        "device": parsed.device,
        "segmenter_model": parsed.segmenter_model,
        "segmenter_checkpoint_path": parsed.segmenter_checkpoint_path,
        "segmenter_bpe_path": parsed.segmenter_bpe_path,
        "segmenter_resolution": parsed.segmenter_resolution,
    }
    return {k: v for k, v in args.items() if v is not None}


def _load_image(image_path: Path) -> Image.Image:
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def _load_coarse_masks(mask_paths: list[Path]) -> list[np.ndarray]:
    coarse_masks = []
    for p in mask_paths:
        arr = np.array(Image.open(p))
        if arr.ndim == 3:
            arr = arr[..., 0]
        coarse_masks.append((arr > 0).astype(np.uint8))
    return coarse_masks


def _ensure_save_dir(parsed):
    if parsed.save_dir is None:
        return None
    parsed.save_dir.mkdir(parents=True, exist_ok=True)
    return parsed.save_dir


def _save_localizer_outputs(save_dir: Path, image: Image.Image, masks, boxes, det_tags):
    image.save(save_dir / "input_image.png")

    boxed = image.copy()
    draw = ImageDraw.Draw(boxed)
    for i, (box, tag) in enumerate(zip(boxes, det_tags)):
        x1, y1, x2, y2 = [int(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1 + 2, max(0, y1 - 14)), f"{i}:{tag}", fill="red")
    boxed.save(save_dir / "localizer_boxes.png")

    localizer_mask_dir = save_dir / "localizer_masks"
    localizer_mask_dir.mkdir(parents=True, exist_ok=True)
    for i, mask in enumerate(masks):
        arr = mask[0] if mask.ndim == 3 else mask
        binary = (arr > 0).astype(np.uint8) * 255
        Image.fromarray(binary, mode="L").save(localizer_mask_dir / f"mask_{i}.png")

    summary = {
        "localizer": {
            "num_instances": len(det_tags),
            "tags": det_tags,
            "boxes": boxes,
        }
    }
    with open(save_dir / "localizer_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def _save_refiner_outputs(save_dir: Path, refined_masks, bboxes_2d, keep_indices):
    refiner_mask_dir = save_dir / "refiner_masks"
    refiner_mask_dir.mkdir(parents=True, exist_ok=True)
    for i, mask in enumerate(refined_masks):
        binary = (mask > 0).astype(np.uint8) * 255
        Image.fromarray(binary, mode="L").save(refiner_mask_dir / f"mask_{i}.png")

    summary = {
        "refiner": {
            "num_refined_masks": len(refined_masks),
            "boxes": bboxes_2d,
            "keep_indices": keep_indices,
        }
    }
    with open(save_dir / "refiner_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def run_localizer(parsed) -> tuple[np.ndarray, list, list] | None:
    with _timed_stage("LOCALIZER", f"loading image: {parsed.image}"):
        image = _load_image(parsed.image)
    tags = _parse_tags(parsed.tags)
    if not tags:
        print("FAIL: --tags is required for localizer/both mode")
        return None
    _log("LOCALIZER", f"parsed {len(tags)} tags: {tags}")
    _log("LOCALIZER", f"image size: {image.size[0]}x{image.size[1]}")

    with tempfile.TemporaryDirectory(prefix="verify_sam3_localizer_") as tmp_dir:
        localizer_args = {
            **_build_common_args(parsed),
            "grounding_model": parsed.grounding_model,
            "output_dir": tmp_dir,
            "file_name": "verify_sam3_localizer",
        }
        with _timed_stage("LOCALIZER", "initializing Localizer (model load/build)"):
            localizer = Localizer(localizer_args)
        with _timed_stage("LOCALIZER", "running detect_and_segment"):
            result = localizer.detect_and_segment(image, tags)
        if result is None:
            print("FAIL: Localizer returned None (no valid detections/masks).")
            return None

        masks, boxes, det_tags = result
        _log(
            "LOCALIZER",
            f"final outputs: instances={len(det_tags)}, masks={len(masks)}, boxes={len(boxes)}",
        )
        print(
            f"PASS: Localizer produced {len(det_tags)} instances, "
            f"{len(masks)} masks, {len(boxes)} boxes."
        )
        print(f"Detected tags: {det_tags}")
        print(f"Boxes: {json.dumps(boxes)}")
        save_dir = _ensure_save_dir(parsed)
        if save_dir is not None:
            _save_localizer_outputs(save_dir, image, masks, boxes, det_tags)
            print(f"SAVED: localizer outputs -> {save_dir}")
        return masks, boxes, det_tags


def run_refiner(parsed, coarse_masks: list[np.ndarray] | None = None) -> tuple[list, list, list] | None:
    with _timed_stage("REFINER", f"loading image: {parsed.image}"):
        image = _load_image(parsed.image)
    with tempfile.TemporaryDirectory(prefix="verify_sam3_refiner_") as tmp_dir:
        refiner_args = {
            **_build_common_args(parsed),
            "output_dir": tmp_dir,
            "file_name": "verify_sam3_refiner",
            "update_keys": [],
        }
        with _timed_stage("REFINER", "initializing Sam3Refiner (model load/build)"):
            refiner = Sam3Refiner(refiner_args)

        if coarse_masks is None:
            if not parsed.mask_paths:
                print("FAIL: --mask_paths is required for refiner mode.")
                return None
            with _timed_stage("REFINER", f"loading coarse masks ({len(parsed.mask_paths)})"):
                coarse_masks = _load_coarse_masks(parsed.mask_paths)
        else:
            _log("REFINER", f"using localizer-produced coarse masks: {len(coarse_masks)}")

        _log("REFINER", f"coarse mask count before refine: {len(coarse_masks)}")
        with _timed_stage("REFINER", f"running refine_masks on {len(coarse_masks)} masks"):
            refined_masks, bboxes_2d, keep_indices = refiner.refine_masks(image, coarse_masks)
        if len(keep_indices) == 0:
            print("FAIL: Refiner kept 0 masks after score/area filtering.")
            return None

        _log(
            "REFINER",
            f"final outputs: kept={len(keep_indices)}, refined_masks={len(refined_masks)}",
        )
        print(
            f"PASS: Refiner kept {len(keep_indices)}/{len(coarse_masks)} masks, "
            f"produced {len(refined_masks)} refined masks."
        )
        print(f"Refined boxes: {json.dumps(bboxes_2d)}")
        print(f"Keep indices: {keep_indices}")
        save_dir = _ensure_save_dir(parsed)
        if save_dir is not None:
            _save_refiner_outputs(save_dir, refined_masks, bboxes_2d, keep_indices)
            print(f"SAVED: refiner outputs -> {save_dir}")
        return refined_masks, bboxes_2d, keep_indices


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone verifier for SAM3 modules.")
    parser.add_argument(
        "--mode",
        choices=["localizer", "refiner", "both"],
        default="both",
        help="Which SAM3 module path to verify.",
    )
    parser.add_argument("--image", type=Path, required=False, help="Input image path.")
    parser.add_argument(
        "--parquet",
        type=Path,
        default=None,
        help="Optional parquet input; script will infer image/tags/masks from one row.",
    )
    parser.add_argument(
        "--row_idx",
        type=int,
        default=0,
        help="Row index used with --parquet.",
    )
    parser.add_argument(
        "--view_idx",
        type=int,
        default=0,
        help="View index for per-scene rows (list-valued columns) with --parquet.",
    )
    parser.add_argument(
        "--raw_data_root",
        type=str,
        default=None,
        help="Root for resolving relative image/mask paths from parquet.",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=None,
        help="Optional directory to save visual outputs (boxes/masks/summaries).",
    )
    parser.add_argument(
        "--tags",
        type=str,
        default=None,
        help="Comma-separated tags for localizer, e.g. chair,table.",
    )
    parser.add_argument(
        "--mask_paths",
        type=Path,
        nargs="*",
        default=None,
        help="Coarse mask paths for refiner-only mode.",
    )

    parser.add_argument("--device", type=str, default="npu:0", help="Device, e.g. npu:0/cuda/cpu.")
    parser.add_argument(
        "--grounding_model",
        type=str,
        default="IDEA-Research/grounding-dino-base",
        help="GroundingDINO model id.",
    )
    parser.add_argument(
        "--segmenter_model",
        type=str,
        default="facebook/sam3",
        help="SAM3 model id (record only).",
    )
    parser.add_argument(
        "--segmenter_checkpoint_path",
        type=str,
        default=None,
        help="Optional local SAM3 checkpoint path. If omitted, auto-download is used.",
    )
    parser.add_argument(
        "--segmenter_bpe_path",
        type=str,
        default=None,
        help="Optional local bpe_simple_vocab_16e6.txt.gz path.",
    )
    parser.add_argument(
        "--segmenter_resolution",
        type=int,
        default=1008,
        help="SAM3 processor resolution.",
    )
    args = parser.parse_args()
    _log(
        "INIT",
        (
            f"mode={args.mode}, device={args.device}, image={args.image}, parquet={args.parquet}, "
            f"save_dir={args.save_dir}, segmenter_model={args.segmenter_model}"
        ),
    )
    try:
        _infer_input_from_parquet(args)
    except Exception as exc:
        print(f"FAIL: failed to infer inputs from parquet: {exc}")
        traceback.print_exc()
        return 1

    if args.image is None:
        print("FAIL: must provide --image or --parquet.")
        return 1
    if not args.image.is_file():
        print(f"FAIL: image not found: {args.image}")
        return 1
    try:
        if args.mode == "localizer":
            _log("FLOW", "running mode=localizer")
            return 0 if run_localizer(args) is not None else 1

        if args.mode == "refiner":
            _log("FLOW", "running mode=refiner")
            return 0 if run_refiner(args) is not None else 1

        _log("FLOW", "running mode=both (localizer -> refiner)")
        localizer_out = run_localizer(args)
        if localizer_out is None:
            return 1

        masks, _, _ = localizer_out
        coarse_masks = []
        _log("FLOW", f"converting {len(masks)} localizer masks to refiner inputs")
        for m in masks:
            if m.ndim == 3 and m.shape[0] == 1:
                coarse_masks.append((m[0] > 0).astype(np.uint8))
            elif m.ndim == 2:
                coarse_masks.append((m > 0).astype(np.uint8))
            else:
                raise ValueError(f"Unexpected mask shape from localizer: {m.shape}")

        return 0 if run_refiner(args, coarse_masks=coarse_masks) is not None else 1
    except Exception as exc:
        traceback.print_exc()
        print(f"FAIL: exception during verification: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
