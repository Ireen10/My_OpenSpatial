#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    Sam3Model,
    Sam3Processor,
)


def log(stage: str, message: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [{stage}] {message}", flush=True)


def resolve_device(explicit: str | None) -> str:
    if explicit:
        return explicit
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu:0"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def prepare_npu_if_needed(device: str):
    if not device.startswith("npu"):
        return
    if not hasattr(torch, "npu"):
        raise RuntimeError("Requested NPU device, but torch.npu is unavailable.")
    torch.npu.set_device(device)


def load_image(image_path: Path) -> Image.Image:
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def to_py_list(value):
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


def resolve_asset_path(path_value, raw_data_root: str | None) -> str:
    if not isinstance(path_value, str):
        raise ValueError(f"Expected a string asset path, got: {type(path_value)}")
    if os.path.isabs(path_value) or not raw_data_root:
        return path_value
    return os.path.normpath(os.path.join(raw_data_root, path_value))


def select_view(value, view_idx: int):
    seq = to_py_list(value)
    if not seq:
        return value
    if isinstance(seq[0], list):
        if view_idx >= len(seq):
            raise ValueError(f"view_idx={view_idx} out of range for sequence length {len(seq)}")
        return seq[view_idx]
    return value


def infer_input_from_parquet(parsed):
    if parsed.parquet is None:
        return
    if not parsed.parquet.is_file():
        raise ValueError(f"Parquet not found: {parsed.parquet}")

    log("INPUT", f"loading parquet: {parsed.parquet}")
    df = pd.read_parquet(parsed.parquet)
    if len(df) == 0:
        raise ValueError(f"Parquet is empty: {parsed.parquet}")
    if parsed.row_idx < 0 or parsed.row_idx >= len(df):
        raise ValueError(f"row_idx out of range: {parsed.row_idx}, len={len(df)}")

    row = df.iloc[parsed.row_idx]

    if parsed.image is None:
        if "image" not in row.index:
            raise ValueError("Parquet row does not contain 'image'.")
        row_image = select_view(row["image"], parsed.view_idx)
        parsed.image = Path(resolve_asset_path(row_image, parsed.raw_data_root))

    if parsed.tags is None and "obj_tags" in row.index:
        row_tags = select_view(row["obj_tags"], parsed.view_idx)
        tags_list = [str(x) for x in to_py_list(row_tags) if str(x).strip()]
        if tags_list:
            parsed.tags = ",".join(tags_list)

    if (parsed.mask_paths is None or len(parsed.mask_paths) == 0) and "masks" in row.index:
        row_masks = select_view(row["masks"], parsed.view_idx)
        masks = to_py_list(row_masks)
        if masks:
            parsed.mask_paths = [
                Path(resolve_asset_path(str(mask_path), parsed.raw_data_root))
                for mask_path in masks
            ]


def load_binary_masks(mask_paths: list[Path]) -> list[np.ndarray]:
    masks = []
    for p in mask_paths:
        arr = np.array(Image.open(p))
        if arr.ndim == 3:
            arr = arr[..., 0]
        masks.append((arr > 0).astype(np.uint8))
    return masks


def masks_to_xyxy_boxes(masks: list[np.ndarray]) -> np.ndarray:
    boxes = []
    for mask in masks:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            boxes.append([0.0, 0.0, 0.0, 0.0])
            continue
        boxes.append([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())])
    return np.asarray(boxes, dtype=np.float32)


def np_boxes_to_nested_list(boxes: np.ndarray) -> list[list[float]]:
    return [[float(x1), float(y1), float(x2), float(y2)] for x1, y1, x2, y2 in boxes]


def postprocess_masks_and_scores(post_result: dict) -> tuple[list[np.ndarray], list[float]]:
    raw_masks = post_result.get("masks", [])
    raw_scores = post_result.get("scores")

    if isinstance(raw_masks, torch.Tensor):
        raw_masks = [raw_masks[i] for i in range(raw_masks.shape[0])]

    masks: list[np.ndarray] = []
    for m in raw_masks:
        if isinstance(m, torch.Tensor):
            m = m.detach().cpu().numpy()
        m = np.asarray(m)
        if m.ndim == 3 and m.shape[0] == 1:
            m = m[0]
        masks.append((m > 0).astype(np.uint8))

    if raw_scores is None:
        scores = [1.0] * len(masks)
    elif isinstance(raw_scores, torch.Tensor):
        scores = [float(v) for v in raw_scores.detach().cpu().tolist()]
    else:
        scores = [float(v) for v in raw_scores]

    if len(scores) < len(masks):
        scores.extend([1.0] * (len(masks) - len(scores)))
    return masks, scores


def run_sam3_box_prompt(
    image: Image.Image,
    boxes_xyxy: np.ndarray,
    model: Sam3Model,
    processor: Sam3Processor,
    device: str,
    threshold: float,
    mask_threshold: float,
) -> tuple[list[np.ndarray], list[float]]:
    if boxes_xyxy.shape[0] == 0:
        return [], []

    input_boxes = [np_boxes_to_nested_list(boxes_xyxy)]
    input_boxes_labels = [[1] * len(input_boxes[0])]

    model_inputs = processor(
        images=image,
        input_boxes=input_boxes,
        input_boxes_labels=input_boxes_labels,
        return_tensors="pt",
    )
    model_inputs = {
        k: (v.to(device) if hasattr(v, "to") else v) for k, v in model_inputs.items()
    }

    with torch.no_grad():
        outputs = model(**model_inputs)

    processed = processor.post_process_instance_segmentation(
        outputs=outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=model_inputs["original_sizes"].detach().cpu().tolist(),
    )[0]
    return postprocess_masks_and_scores(processed)


def save_masks(mask_dir: Path, masks: list[np.ndarray]) -> list[str]:
    mask_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for i, m in enumerate(masks):
        out = mask_dir / f"mask_{i}.png"
        Image.fromarray((m > 0).astype(np.uint8) * 255, mode="L").save(out)
        saved.append(str(out))
    return saved


def save_boxes_visualization(
    out_path: Path, image: Image.Image, boxes: np.ndarray, tags: list[str] | None = None
):
    vis = image.copy()
    draw = ImageDraw.Draw(vis)
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in b]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        if tags is not None and i < len(tags):
            draw.text((x1 + 2, max(0, y1 - 12)), f"{i}:{tags[i]}", fill="red")
    vis.save(out_path)


def main() -> int:
    parser = argparse.ArgumentParser("Transformers-based SAM3 verifier")
    parser.add_argument("--mode", choices=["localizer", "refiner", "both"], default="both")
    parser.add_argument("--image", type=Path, required=False)
    parser.add_argument("--parquet", type=Path, default=None)
    parser.add_argument("--row_idx", type=int, default=0)
    parser.add_argument("--view_idx", type=int, default=0)
    parser.add_argument("--raw_data_root", type=str, default=None)
    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument("--mask_paths", type=Path, nargs="*", default=None)

    parser.add_argument("--device", type=str, default=None, help="sam3 device (e.g. npu:0/cuda:0/cpu)")
    parser.add_argument("--grounding_model", type=str, default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--sam3_model", type=str, default="facebook/sam3")

    parser.add_argument("--det_threshold", type=float, default=0.3)
    parser.add_argument("--text_threshold", type=float, default=0.3)
    parser.add_argument("--sam_threshold", type=float, default=0.5)
    parser.add_argument("--sam_mask_threshold", type=float, default=0.5)
    parser.add_argument("--save_dir", type=Path, default=Path("output/verify_sam3_transformers"))
    args = parser.parse_args()

    infer_input_from_parquet(args)
    if args.image is None:
        raise ValueError("must provide --image or --parquet")
    if not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if args.mode == "refiner" and not args.mask_paths:
        raise ValueError("--mask_paths is required for refiner mode")

    sam_device = resolve_device(args.device)
    prepare_npu_if_needed(sam_device)
    det_device = sam_device

    args.save_dir.mkdir(parents=True, exist_ok=True)
    log("INIT", f"mode={args.mode}, device={sam_device}")
    log("INIT", f"grounding_model={args.grounding_model}, sam3_model={args.sam3_model}")

    log("LOAD", f"loading image: {args.image}")
    image = load_image(args.image)

    log("LOAD", f"loading GroundingDINO: {args.grounding_model}")
    detector_processor = AutoProcessor.from_pretrained(args.grounding_model)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(args.grounding_model).to(det_device)

    log("LOAD", f"loading SAM3: {args.sam3_model}")
    sam3_model = Sam3Model.from_pretrained(args.sam3_model).to(sam_device)
    sam3_model.eval()
    sam3_processor = Sam3Processor.from_pretrained(args.sam3_model)

    localizer_masks: list[np.ndarray] = []
    localizer_boxes = np.zeros((0, 4), dtype=np.float32)
    det_tags: list[str] = []
    localizer_scores: list[float] = []

    if args.mode in ("localizer", "both"):
        tags_text = args.tags or "chair,table"
        tags = [t.strip() for t in tags_text.split(",") if t.strip()]
        if not tags:
            raise ValueError("--tags is empty in localizer/both mode")

        text_prompt = ". ".join(tags)
        log("LOCALIZER", f"running detector with prompt: {text_prompt}")
        det_inputs = detector_processor(images=image, text=text_prompt, return_tensors="pt")
        det_inputs = {k: (v.to(det_device) if hasattr(v, "to") else v) for k, v in det_inputs.items()}
        with torch.no_grad():
            det_outputs = detector(**det_inputs)

        det_result = detector_processor.post_process_grounded_object_detection(
            det_outputs,
            det_inputs["input_ids"],
            threshold=args.det_threshold,
            text_threshold=args.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        det_tags = det_result["text_labels"]
        localizer_boxes = det_result["boxes"].detach().cpu().numpy()
        log("LOCALIZER", f"detected boxes: {len(localizer_boxes)}")
        save_boxes_visualization(args.save_dir / "localizer_boxes.png", image, localizer_boxes, det_tags)

        log("LOCALIZER", "running SAM3 with detector boxes")
        localizer_masks, localizer_scores = run_sam3_box_prompt(
            image=image,
            boxes_xyxy=localizer_boxes,
            model=sam3_model,
            processor=sam3_processor,
            device=sam_device,
            threshold=args.sam_threshold,
            mask_threshold=args.sam_mask_threshold,
        )
        log("LOCALIZER", f"sam3 masks: {len(localizer_masks)}")
        save_masks(args.save_dir / "localizer_masks", localizer_masks)

        with open(args.save_dir / "localizer_summary.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "num_detections": len(localizer_boxes),
                    "det_tags": det_tags,
                    "det_boxes": localizer_boxes.tolist(),
                    "num_masks": len(localizer_masks),
                    "scores": localizer_scores,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    if args.mode in ("refiner", "both"):
        if args.mode == "refiner":
            coarse_masks = load_binary_masks(args.mask_paths or [])
        else:
            coarse_masks = localizer_masks
            if len(coarse_masks) == 0:
                raise RuntimeError(
                    "both mode requires localizer to produce masks, but got 0. "
                    f"Detector boxes={len(localizer_boxes)}, detector tags={len(det_tags)}, "
                    f"sam3 masks={len(localizer_masks)}. "
                    "Run with --mode localizer first and inspect localizer_summary.json."
                )

        coarse_boxes = masks_to_xyxy_boxes(coarse_masks)
        log("REFINER", f"running SAM3 refine with {len(coarse_boxes)} coarse boxes")
        refined_masks, refined_scores = run_sam3_box_prompt(
            image=image,
            boxes_xyxy=coarse_boxes,
            model=sam3_model,
            processor=sam3_processor,
            device=sam_device,
            threshold=args.sam_threshold,
            mask_threshold=args.sam_mask_threshold,
        )
        log("REFINER", f"refined masks: {len(refined_masks)}")
        save_masks(args.save_dir / "refiner_masks", refined_masks)

        with open(args.save_dir / "refiner_summary.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "num_input_masks": len(coarse_masks),
                    "input_boxes": coarse_boxes.tolist(),
                    "num_refined_masks": len(refined_masks),
                    "scores": refined_scores,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    log("DONE", f"outputs saved to: {args.save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
