#!/usr/bin/env python3
"""
Standalone verifier for SAM3 localization modules.

Supports:
1) Localizer only (GroundingDINO + SAM3)
2) Refiner only (SAM3 box-prompt refine)
3) End-to-end localizer -> refiner on one image

Examples:
  # Localizer only
  python verification/dataset_pipeline/verify_sam3_modules.py \
    --mode localizer \
    --image /path/to/image.jpg \
    --tags chair,table \
    --device npu:0 \
    --segmenter_checkpoint_path /path/to/sam3.pt

  # Refiner only with existing coarse masks
  python verification/dataset_pipeline/verify_sam3_modules.py \
    --mode refiner \
    --image /path/to/image.jpg \
    --mask_paths /path/to/mask0.png /path/to/mask1.png \
    --device npu:0 \
    --segmenter_checkpoint_path /path/to/sam3.pt

  # Chain both modules
  python verification/dataset_pipeline/verify_sam3_modules.py \
    --mode both \
    --image /path/to/image.jpg \
    --tags chair,table \
    --device npu:0 \
    --segmenter_checkpoint_path /path/to/sam3.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from task.localization.grounding_sam3 import Localizer
from task.localization.sam3_refiner import Sam3Refiner


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
    image = _load_image(parsed.image)
    tags = _parse_tags(parsed.tags)
    if not tags:
        print("FAIL: --tags is required for localizer/both mode")
        return None

    with tempfile.TemporaryDirectory(prefix="verify_sam3_localizer_") as tmp_dir:
        localizer_args = {
            **_build_common_args(parsed),
            "grounding_model": parsed.grounding_model,
            "output_dir": tmp_dir,
            "file_name": "verify_sam3_localizer",
        }
        localizer = Localizer(localizer_args)
        result = localizer.detect_and_segment(image, tags)
        if result is None:
            print("FAIL: Localizer returned None (no valid detections/masks).")
            return None

        masks, boxes, det_tags = result
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
    image = _load_image(parsed.image)
    with tempfile.TemporaryDirectory(prefix="verify_sam3_refiner_") as tmp_dir:
        refiner_args = {
            **_build_common_args(parsed),
            "output_dir": tmp_dir,
            "file_name": "verify_sam3_refiner",
            "update_keys": [],
        }
        refiner = Sam3Refiner(refiner_args)

        if coarse_masks is None:
            if not parsed.mask_paths:
                print("FAIL: --mask_paths is required for refiner mode.")
                return None
            coarse_masks = _load_coarse_masks(parsed.mask_paths)

        refined_masks, bboxes_2d, keep_indices = refiner.refine_masks(image, coarse_masks)
        if len(keep_indices) == 0:
            print("FAIL: Refiner kept 0 masks after score/area filtering.")
            return None

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
    try:
        _infer_input_from_parquet(args)
    except Exception as exc:
        print(f"FAIL: failed to infer inputs from parquet: {exc}")
        return 1

    if args.image is None:
        print("FAIL: must provide --image or --parquet.")
        return 1
    if not args.image.is_file():
        print(f"FAIL: image not found: {args.image}")
        return 1
    try:
        if args.mode == "localizer":
            return 0 if run_localizer(args) is not None else 1

        if args.mode == "refiner":
            return 0 if run_refiner(args) is not None else 1

        localizer_out = run_localizer(args)
        if localizer_out is None:
            return 1

        masks, _, _ = localizer_out
        coarse_masks = []
        for m in masks:
            if m.ndim == 3 and m.shape[0] == 1:
                coarse_masks.append((m[0] > 0).astype(np.uint8))
            elif m.ndim == 2:
                coarse_masks.append((m > 0).astype(np.uint8))
            else:
                raise ValueError(f"Unexpected mask shape from localizer: {m.shape}")

        return 0 if run_refiner(args, coarse_masks=coarse_masks) is not None else 1
    except Exception as exc:
        print(f"FAIL: exception during verification: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
