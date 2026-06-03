#!/usr/bin/env python3
"""
Verify the SAM3 grounding localizer (grounding_sam3.Localizer) on parquet data.

The script samples rows from a parquet file, resolves image paths via --data_root,
runs detect_and_segment(), saves an overlay visualization per sample, and prints
a summary report.

Usage:
  python verification/localization/verify_sam3_localizer.py \\
    --parquet   /path/to/data.parquet \\
    --data_root /path/to/raw/data \\
    --grounding_model /path/to/grounding_dino_base \\
    --segmenter_model /path/to/sam3 \\
    --output_dir /tmp/sam3_verify \\
    --n_samples 10 \\
    --device npu

Notes:
  - Only singleview rows (image is a string path) are processed; multiview rows are
    skipped automatically (they require masks from a prior stage).
  - The script calls detect_and_segment() directly, bypassing apply_transform(),
    so no pipeline state (output_dir, masks, etc.) is needed.
  - Requires transformers >= 5.0.0 for Sam3TrackerModel/Sam3TrackerProcessor.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from task.localization.grounding_sam3 import Localizer


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

# Distinct colours for up to 20 object instances (RGBA)
_PALETTE = [
    (255, 59,  59,  120),
    (59,  178, 255, 120),
    (59,  255, 130, 120),
    (255, 200, 59,  120),
    (200, 59,  255, 120),
    (59,  255, 230, 120),
    (255, 130, 59,  120),
    (130, 255, 59,  120),
    (59,  59,  255, 120),
    (255, 59,  200, 120),
    (200, 255, 59,  120),
    (59,  200, 255, 120),
    (255, 59,  130, 120),
    (59,  255, 59,  120),
    (130, 59,  255, 120),
    (255, 255, 59,  120),
    (59,  130, 255, 120),
    (255, 59,  59,  120),
    (59,  255, 200, 120),
    (200, 59,  130, 120),
]


def _parse_tags(val) -> list[str]:
    """Normalise obj_tags from parquet (list, JSON string, or numpy array)."""
    if val is None:
        return []
    if isinstance(val, list):
        return [str(t) for t in val]
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            return [str(t) for t in parsed] if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    if hasattr(val, "tolist"):
        return [str(t) for t in val.tolist()]
    return []


def _resolve_image_path(image_val, data_root: Optional[Path]) -> Optional[Path]:
    """Return absolute image path, handling relative paths via data_root."""
    if isinstance(image_val, list):
        image_val = image_val[0]
    if not isinstance(image_val, str) or not image_val.strip():
        return None
    p = Path(image_val)
    if p.is_absolute():
        return p
    if data_root is not None:
        return data_root / p
    return p


def _draw_overlay(
    image: Image.Image,
    masks: np.ndarray,
    boxes: list,
    tags: list[str],
) -> Image.Image:
    """Composite coloured masks and labelled boxes onto a copy of image.

    Args:
        image: Original RGB PIL Image.
        masks: np.ndarray of shape (N, 1, H, W) — raw SAM3 output.
        boxes: list of N [x1, y1, x2, y2] bounding boxes.
        tags: list of N tag strings.

    Returns:
        Annotated RGB PIL Image.
    """
    canvas = image.convert("RGBA")

    for idx, (mask, box, tag) in enumerate(zip(masks, boxes, tags)):
        colour = _PALETTE[idx % len(_PALETTE)]

        # mask shape: (1, H, W) — squeeze to (H, W)
        mask_2d = (mask[0] > 0).astype(np.uint8)  # 0 or 1

        # Resize mask to match original image if needed
        if mask_2d.shape != (image.height, image.width):
            mask_pil = Image.fromarray(mask_2d * 255, mode="L").resize(
                (image.width, image.height), resample=Image.NEAREST
            )
            mask_2d = np.array(mask_pil) > 127

        # Create coloured RGBA overlay for this mask
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        fill = Image.new("RGBA", canvas.size, colour)
        alpha_mask = Image.fromarray((mask_2d * colour[3]).astype(np.uint8), mode="L")
        overlay.paste(fill, mask=alpha_mask)
        canvas = Image.alpha_composite(canvas, overlay)

    # Draw boxes and labels on top
    result = canvas.convert("RGB")
    draw = ImageDraw.Draw(result)
    try:
        font = ImageFont.truetype("arial.ttf", size=14)
    except IOError:
        font = ImageFont.load_default()

    for idx, (box, tag) in enumerate(zip(boxes, tags)):
        colour_rgb = _PALETTE[idx % len(_PALETTE)][:3]
        x1, y1, x2, y2 = [int(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=colour_rgb, width=2)
        label = f"{idx}: {tag}"
        # Draw label background
        try:
            bbox = draw.textbbox((x1, y1 - 16), label, font=font)
        except AttributeError:
            bbox = (x1, y1 - 16, x1 + len(label) * 8, y1)
        draw.rectangle(bbox, fill=colour_rgb)
        draw.text((x1, y1 - 16), label, fill=(255, 255, 255), font=font)

    return result


def _make_summary_image(
    original: Image.Image,
    annotated: Image.Image,
    tags_input: list[str],
    tags_detected: list[str],
    sample_id: str,
) -> Image.Image:
    """Place original | annotated side by side with a title bar."""
    W, H = original.width, original.height
    gap = 8
    title_h = 40
    tag_h = max(len(tags_input), 1) * 18 + 8

    total_w = W * 2 + gap
    total_h = H + title_h + tag_h
    out = Image.new("RGB", (total_w, total_h), (30, 30, 30))

    # Title
    draw = ImageDraw.Draw(out)
    try:
        font_title = ImageFont.truetype("arial.ttf", size=16)
        font_small = ImageFont.truetype("arial.ttf", size=12)
    except IOError:
        font_title = ImageFont.load_default()
        font_small = font_title

    draw.text((4, 4), f"Sample: {sample_id}", fill=(220, 220, 220), font=font_title)
    draw.text((W + gap + 4, 4), f"Detected ({len(tags_detected)} obj): {', '.join(tags_detected[:8])}", fill=(140, 220, 140), font=font_title)

    # Images
    out.paste(original, (0, title_h))
    out.paste(annotated, (W + gap, title_h))

    # Input tags below
    tags_str = "Input tags: " + ", ".join(tags_input[:20])
    draw.text((4, title_h + H + 4), tags_str, fill=(180, 180, 180), font=font_small)

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Verify SAM3 grounding localizer on parquet data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--parquet", required=True, type=Path,
                   help="Path to input .parquet file.")
    p.add_argument("--data_root", type=Path, default=None,
                   help="Root directory prepended to relative image paths in parquet.")
    p.add_argument("--grounding_model", default="IDEA-Research/grounding-dino-base",
                   help="Local path or HuggingFace hub ID for GroundingDINO.")
    p.add_argument("--segmenter_model", default="facebook/sam3",
                   help="Local path or HuggingFace hub ID for SAM3.")
    p.add_argument("--output_dir", type=Path, default=Path("output/sam3_verify"),
                   help="Directory to save overlay visualizations.")
    p.add_argument("--n_samples", type=int, default=5,
                   help="Number of rows to sample from the parquet file.")
    p.add_argument("--device", default=None,
                   help='Inference device, e.g. "cuda", "npu", "npu:0", "cpu". '
                        'Auto-detected if omitted.')
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for row sampling.")
    p.add_argument("--image_col", default="image",
                   help="Parquet column that holds the image path.")
    p.add_argument("--tags_col", default="obj_tags",
                   help="Parquet column that holds the object tag list.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    # ── Load parquet ───────────────────────────────────────────────────────────
    if not args.parquet.is_file():
        print(f"ERROR: parquet not found: {args.parquet}")
        return 1

    df = pd.read_parquet(args.parquet)
    print(f"Loaded {len(df)} rows from {args.parquet}")
    print(f"Columns: {list(df.columns)}")

    # Filter to singleview rows only (image column is a plain string path)
    singleview_mask = df[args.image_col].apply(
        lambda v: isinstance(v, str) and bool(v.strip())
    )
    df_sv = df[singleview_mask].reset_index(drop=True)
    if len(df_sv) == 0:
        print("ERROR: no singleview rows found (image column must be a string path).")
        print("  If your data uses list[str] for images, this is a multiview dataset;")
        print("  multiview verification requires pre-existing masks and is not yet supported.")
        return 1

    n = min(args.n_samples, len(df_sv))
    indices = random.sample(range(len(df_sv)), n)
    print(f"Sampling {n} / {len(df_sv)} singleview rows (seed={args.seed})")

    # ── Build model args ───────────────────────────────────────────────────────
    model_args: dict = {
        "grounding_model": str(args.grounding_model),
        "segmenter_model": str(args.segmenter_model),
        "output_dir": str(args.output_dir),
        "file_name": "sam3_verify",
        # update_keys is not needed (we call detect_and_segment directly)
        # but Localizer doesn't assert on it, so no problem.
    }
    if args.device:
        model_args["device"] = args.device

    # ── Initialise Localizer ───────────────────────────────────────────────────
    print("\nInitialising Localizer (loading model weights) …")
    t0 = time.perf_counter()
    localizer = Localizer(model_args)
    print(f"  Model loaded in {time.perf_counter() - t0:.1f}s  "
          f"(device={localizer.device})")

    # ── Run on samples ─────────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []  # list of dicts for summary table
    for rank, row_idx in enumerate(indices):
        row = df_sv.iloc[row_idx]
        sample_id = str(row.get("sample_id", row_idx))

        img_path = _resolve_image_path(row.get(args.image_col), args.data_root)
        obj_tags = _parse_tags(row.get(args.tags_col))

        print(f"\n[{rank+1}/{n}] row={row_idx}  sample_id={sample_id}")
        print(f"  image  : {img_path}")
        print(f"  tags   : {obj_tags}")

        # ── Sanity checks ──────────────────────────────────────────────────────
        if img_path is None or not Path(img_path).is_file():
            print(f"  SKIP: image not found at {img_path}")
            results.append({"sample_id": sample_id, "status": "skip_missing_image",
                             "n_detected": 0, "elapsed_s": 0.0})
            continue

        if not obj_tags:
            print(f"  SKIP: no obj_tags in row")
            results.append({"sample_id": sample_id, "status": "skip_no_tags",
                             "n_detected": 0, "elapsed_s": 0.0})
            continue

        # ── Run detect_and_segment ─────────────────────────────────────────────
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as exc:
            print(f"  SKIP: cannot open image — {exc}")
            results.append({"sample_id": sample_id, "status": "skip_image_error",
                             "n_detected": 0, "elapsed_s": 0.0})
            continue

        t1 = time.perf_counter()
        try:
            result = localizer.detect_and_segment(image, obj_tags)
        except Exception as exc:
            elapsed = time.perf_counter() - t1
            print(f"  FAIL: detect_and_segment raised {type(exc).__name__}: {exc}")
            results.append({"sample_id": sample_id, "status": "error",
                             "n_detected": 0, "elapsed_s": elapsed})
            continue
        elapsed = time.perf_counter() - t1

        if result is None:
            print(f"  RESULT: no detections (returned None)  [{elapsed:.2f}s]")
            results.append({"sample_id": sample_id, "status": "no_detections",
                             "n_detected": 0, "elapsed_s": elapsed})
            # Still save the unmodified image so caller can inspect input
            image.save(args.output_dir / f"{rank:03d}_{sample_id}_no_detections.png")
            continue

        masks, boxes, det_tags = result
        print(f"  RESULT: {len(det_tags)} objects  tags={det_tags}  [{elapsed:.2f}s]")
        for i, (box, tag) in enumerate(zip(boxes, det_tags)):
            print(f"    [{i}] {tag:30s}  box={[round(v) for v in box]}")

        results.append({
            "sample_id": sample_id,
            "status": "ok",
            "n_detected": len(det_tags),
            "det_tags": det_tags,
            "elapsed_s": elapsed,
        })

        # ── Visualise ──────────────────────────────────────────────────────────
        try:
            annotated = _draw_overlay(image, masks, boxes, det_tags)
            summary_img = _make_summary_image(
                image, annotated, obj_tags, det_tags, sample_id
            )
            out_path = args.output_dir / f"{rank:03d}_{sample_id}_result.png"
            summary_img.save(out_path)
            print(f"  Saved: {out_path}")
        except Exception as exc:
            print(f"  WARN: visualisation failed — {exc}")

    # ── Summary ────────────────────────────────────────────────────────────────
    total_ok        = sum(1 for r in results if r["status"] == "ok")
    total_no_det    = sum(1 for r in results if r["status"] == "no_detections")
    total_skip      = sum(1 for r in results if r["status"].startswith("skip"))
    total_error     = sum(1 for r in results if r["status"] == "error")
    avg_elapsed     = (
        sum(r["elapsed_s"] for r in results if r["status"] == "ok") / total_ok
        if total_ok > 0 else 0.0
    )
    avg_detections  = (
        sum(r["n_detected"] for r in results if r["status"] == "ok") / total_ok
        if total_ok > 0 else 0.0
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Sampled rows    : {n}")
    print(f"  OK (detections) : {total_ok}")
    print(f"  No detections   : {total_no_det}")
    print(f"  Skipped         : {total_skip}")
    print(f"  Errors          : {total_error}")
    print(f"  Avg detections  : {avg_detections:.1f}  (across OK rows)")
    print(f"  Avg latency     : {avg_elapsed:.2f}s  (per image, model warm)")
    print(f"  Outputs saved to: {args.output_dir.resolve()}")
    print("=" * 60)

    return 0 if total_error == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
