#!/usr/bin/env python3
"""
Diagnostic visualiser for Sam3Refiner: coarse-mask → bbox → SAM3 mask.

Picks N samples from a filter-stage parquet, draws the coarse masks and the
derived bounding boxes on the original image, runs SAM3, then renders the
output masks side-by-side.  Useful for answering:

  1. Are the coarse masks valid (non-empty, correct location)?
  2. Are the derived bboxes reasonable?
  3. Does SAM3 produce any mask at those locations?
  4. What are the actual IoU scores?

Usage
-----
python verification/localization/verify_sam3_refiner.py \
    --parquet  /path/to/filter_stage/3dbox_filter/data.parquet \
    --data_root /path/to/raw/data \
    --segmenter_model facebook/sam3 \
    --output_dir /tmp/refiner_debug \
    --n_samples 5 \
    --device npu:0
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from task.localization.sam3_refiner import (
    Sam3Refiner,
    _extract_mask,
    _load_coarse_mask,
    _load_sam3_replica,
    _match_proposals_to_boxes,
    _post_process,
    _target_sizes,
)
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_COLORS = [
    (255, 60, 60),
    (60, 180, 255),
    (60, 220, 100),
    (255, 200, 60),
    (200, 60, 255),
    (60, 220, 220),
]


def _font():
    try:
        return ImageFont.truetype("arial.ttf", 14)
    except IOError:
        return ImageFont.load_default()


def _draw_boxes_and_masks(image: Image.Image, masks_bool, boxes, tags, title="") -> Image.Image:
    """Overlay coarse masks (semi-transparent) + bbox rectangles on image."""
    canvas = image.convert("RGBA")
    for idx, mask in enumerate(masks_bool):
        color = _COLORS[idx % len(_COLORS)]
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        fill = Image.new("RGBA", canvas.size, (*color, 80))
        alpha = Image.fromarray((mask.astype(np.uint8) * 80), mode="L")
        overlay.paste(fill, mask=alpha)
        canvas = Image.alpha_composite(canvas, overlay)

    result = canvas.convert("RGB")
    draw = ImageDraw.Draw(result)
    font = _font()
    for idx, (box, tag) in enumerate(zip(boxes, tags)):
        color = _COLORS[idx % len(_COLORS)]
        x1, y1, x2, y2 = [int(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{idx}:{tag}"
        try:
            lbox = draw.textbbox((x1, max(0, y1 - 16)), label, font=font)
        except AttributeError:
            lbox = (x1, max(0, y1 - 16), x1 + len(label) * 8, y1)
        draw.rectangle(lbox, fill=color)
        draw.text((x1, max(0, y1 - 16)), label, fill=(255, 255, 255), font=font)

    if title:
        draw.text((4, 4), title, fill=(255, 255, 0), font=font)
    return result


def _draw_sam3_masks(image: Image.Image, masks_float, scores, tags, title="") -> Image.Image:
    """Overlay SAM3 output masks on image with per-mask score."""
    canvas = image.convert("RGBA")
    for idx, mask in enumerate(masks_float):
        color = _COLORS[idx % len(_COLORS)]
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        fill = Image.new("RGBA", canvas.size, (*color, 100))
        alpha = Image.fromarray(((mask > 0.5).astype(np.uint8) * 100), mode="L")
        overlay.paste(fill, mask=alpha)
        canvas = Image.alpha_composite(canvas, overlay)

    result = canvas.convert("RGB")
    draw = ImageDraw.Draw(result)
    font = _font()
    for idx, (score, tag) in enumerate(zip(scores, tags)):
        color = _COLORS[idx % len(_COLORS)]
        label = f"{idx}:{tag} sc={score:.3f}"
        y_off = 4 + idx * 18
        try:
            lbox = draw.textbbox((4, y_off), label, font=font)
        except AttributeError:
            lbox = (4, y_off, 4 + len(label) * 8, y_off + 14)
        draw.rectangle(lbox, fill=color)
        draw.text((4, y_off), label, fill=(255, 255, 255), font=font)

    if title:
        draw.text((4, 4 + len(scores) * 18 + 4), title, fill=(255, 255, 0), font=font)
    return result


def _side_by_side(*panels: Image.Image, gap: int = 6) -> Image.Image:
    h = max(p.height for p in panels)
    w = sum(p.width for p in panels) + gap * (len(panels) - 1)
    out = Image.new("RGB", (w, h), (30, 30, 30))
    x = 0
    for p in panels:
        out.paste(p, (x, 0))
        x += p.width + gap
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Core diagnostic
# ─────────────────────────────────────────────────────────────────────────────

def diagnose_sample(row: dict, data_root: Path | None,
                    processor, model, device: str,
                    output_dir: Path, rank: int,
                    min_coverage: float = 0.10,
                    min_precision: float = 0.40) -> dict:
    """
    Run one sample through the full refiner chain and save a 3-panel image:
      [original + coarse masks + bboxes] | [SAM3 output masks] | [score table]

    Returns a summary dict.
    """
    sample_id = str(row.get("sample_id", row.get("id", rank)))

    # ── Resolve image path ───────────────────────────────────────────────────
    img_val = row.get("image")
    if isinstance(img_val, list):
        img_val = img_val[0]
    img_path = Path(img_val) if img_val else None
    if img_path and not img_path.is_absolute() and data_root:
        img_path = data_root / img_path
    if img_path is None or not img_path.is_file():
        print(f"  SKIP: image not found: {img_path}")
        return {"sample_id": sample_id, "status": "skip_no_image"}

    # ── Load image ───────────────────────────────────────────────────────────
    image = Image.open(img_path).convert("RGB")
    W, H = image.size
    print(f"  image  : {img_path}  ({W}×{H})")

    # ── Load coarse masks ────────────────────────────────────────────────────
    # Parquet columns are often numpy arrays; avoid `if not array` ambiguity.
    mask_paths_raw = row.get("masks")
    mask_paths = (
        list(mask_paths_raw)
        if mask_paths_raw is not None and hasattr(mask_paths_raw, "__len__")
        else (mask_paths_raw if mask_paths_raw is not None else [])
    )
    if len(mask_paths) == 0:
        print("  SKIP: no masks in row")
        return {"sample_id": sample_id, "status": "skip_no_masks"}

    coarse_masks = []
    for p in mask_paths:
        try:
            m = _load_coarse_mask(str(p))
            coarse_masks.append(m)
        except Exception as e:
            print(f"  WARN: cannot load mask {p}: {e}")
    if len(coarse_masks) == 0:
        print("  SKIP: all masks failed to load")
        return {"sample_id": sample_id, "status": "skip_mask_load_error"}

    tags_raw = row.get("obj_tags")
    tags = (
        list(tags_raw)
        if tags_raw is not None and hasattr(tags_raw, "__len__") and len(tags_raw) > 0
        else ["?"] * len(coarse_masks)
    )

    # ── Derive bboxes from coarse masks ──────────────────────────────────────
    boxes = Sam3Refiner._masks_to_bboxes(coarse_masks)
    print(f"  coarse masks : {len(coarse_masks)}  shapes: "
          f"{[m.shape for m in coarse_masks]}")
    print(f"  coarse pixels: {[int(m.sum()) for m in coarse_masks]}")
    print(f"  derived boxes: {boxes.tolist()}")
    for k, (box, tag) in enumerate(zip(boxes, tags)):
        x1, y1, x2, y2 = box
        valid = (0 <= x1 < x2 <= W) and (0 <= y1 < y2 <= H)
        area_px = int((x2 - x1) * (y2 - y1))
        print(f"    [{k}] {tag:25s}  box={[int(v) for v in box]}  "
              f"area={area_px}px  in_bounds={'OK' if valid else 'BAD'}")

    # ── Panel 1: original + coarse overlays ──────────────────────────────────
    panel_coarse = _draw_boxes_and_masks(
        image, coarse_masks, boxes, tags,
        title=f"coarse masks + derived boxes  ({W}x{H})"
    )

    # ── Run SAM3 ─────────────────────────────────────────────────────────────
    text = ". ".join(str(t) for t in tags) if tags else None
    proc_kwargs = dict(
        images=image,
        input_boxes=[boxes.tolist()],
        input_boxes_labels=[[1] * len(boxes)],
        return_tensors="pt",
    )
    if text:
        proc_kwargs["text"] = text

    try:
        inputs = processor(**proc_kwargs).to(device)
        target_sizes = _target_sizes(inputs)
        h_px, w_px = int(target_sizes[0][0]), int(target_sizes[0][1])
        with torch.no_grad():
            outputs = model(**inputs)
        seg_list = _post_process(processor, outputs, 0.0, target_sizes)
        seg = seg_list[0]
        seg_masks_raw = seg.get("masks")
        n_proposals = len(seg_masks_raw) if seg_masks_raw is not None else 0
    except Exception as exc:
        print(f"  ERROR: SAM3 inference failed — {type(exc).__name__}: {exc}")
        return {
            "sample_id": sample_id,
            "status": "error_sam3",
            "n_boxes": len(boxes),
            "error": str(exc),
        }

    print(f"  SAM3 returned {n_proposals} proposals for {len(boxes)} boxes")
    print(f"  (SAM3 is DETR-style: always ~200 proposals; matching by ROI coverage)")

    # Convert all proposals to numpy, then match to boxes by coverage.
    # min_coverage=0.0 here so we always display the best match regardless of
    # threshold — the summary table shows how many would survive the real gate.
    proposals = [_extract_mask(seg_masks_raw[k]) for k in range(n_proposals)]
    sam3_masks_np, sam3_scores, sam3_precisions = _match_proposals_to_boxes(
        proposals, boxes, h_px, w_px, min_coverage=0.0
    )

    for k, (m, cov, prec) in enumerate(zip(sam3_masks_np, sam3_scores, sam3_precisions)):
        px = int((m > 0.5).sum())
        tag = tags[k] if k < len(tags) else "?"
        box = boxes[k]
        box_area = max(int((box[2] - box[0]) * (box[3] - box[1])), 1)
        keeps = cov >= min_coverage and prec >= min_precision and px > 20
        verdict = "KEEP" if keeps else "DROP"
        print(f"    [{k}] {tag:20s}  coverage={cov:.3f} "
              f"({int(cov*box_area)}/{box_area}px)  "
              f"precision={prec:.3f}  mask_pixels={px}  [{verdict}]")

    # ── Panel 2: SAM3 output (coverage-matched) ──────────────────────────────
    panel_sam3 = _draw_sam3_masks(
        image, sam3_masks_np, sam3_scores, tags,
        title=f"SAM3 output (coverage-matched, {n_proposals} proposals)"
    )

    # ── Save composite ───────────────────────────────────────────────────────
    composite = _side_by_side(panel_coarse, panel_sam3)
    out_path = output_dir / f"{rank:03d}_{sample_id}.png"
    composite.save(out_path)
    print(f"  Saved: {out_path}")

    return {
        "sample_id": sample_id,
        "status": "ok",
        "n_boxes": len(boxes),
        "n_proposals": n_proposals,
        "coverages": sam3_scores,
        "precisions": sam3_precisions,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Diagnose Sam3Refiner: visualise coarse-mask → bbox → SAM3 mask.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--parquet", required=True, type=Path,
                   help="filter-stage output parquet (data.parquet)")
    p.add_argument("--data_root", type=Path, default=None,
                   help="root prepended to relative image paths")
    p.add_argument("--segmenter_model", default="facebook/sam3")
    p.add_argument("--output_dir", type=Path, default=Path("/tmp/refiner_debug"))
    p.add_argument("--n_samples", type=int, default=5,
                   help="number of rows to test")
    p.add_argument("--offset", type=int, default=0,
                   help="skip first N rows (pick different slice of the parquet)")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None,
                   help="if set, sample randomly instead of sequentially")
    p.add_argument(
        "--min_coverage", type=float, default=0.10,
        help="Recall gate: matched mask must cover ≥ this fraction of the box "
             "area (mirrors Sam3Refiner.MIN_COVERAGE). Set 0.0 to show all.",
    )
    p.add_argument(
        "--min_precision", type=float, default=0.40,
        help="Purity gate: at least this fraction of the matched mask must lie "
             "inside the box (mirrors Sam3Refiner.MIN_PRECISION). "
             "Rejects masks that drift far outside the box region.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not args.parquet.is_file():
        print(f"ERROR: parquet not found: {args.parquet}")
        return 1

    df = pd.read_parquet(args.parquet)
    print(f"Loaded {len(df)} rows from {args.parquet}")

    if args.seed is not None:
        import random
        random.seed(args.seed)
        indices = random.sample(range(len(df)), min(args.n_samples, len(df)))
    else:
        start = args.offset
        indices = list(range(start, min(start + args.n_samples, len(df))))

    print(f"Diagnosing rows: {indices}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nLoading SAM3 ({args.segmenter_model}) on {device} ...")
    processor, model = _load_sam3_replica(args.segmenter_model, {}, device)
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for rank, row_idx in enumerate(indices):
        row = df.iloc[row_idx].to_dict()
        print(f"\n[{rank+1}/{len(indices)}] row={row_idx}")
        r = diagnose_sample(row, args.data_root, processor, model, device,
                            args.output_dir, rank,
                            min_coverage=args.min_coverage,
                            min_precision=args.min_precision)
        results.append(r)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    ok = [r for r in results if r["status"] == "ok"]
    if ok:
        all_cov = [s for r in ok for s in r["coverages"]]
        total_proposals = sum(r.get("n_proposals", 0) for r in ok)
        print(f"  Samples OK          : {len(ok)}/{len(results)}")
        print(f"  Total boxes         : {sum(r['n_boxes'] for r in ok)}")
        print(f"  Total SAM3 proposals: {total_proposals}")
        if all_cov:
            all_prec = [p for r in ok for p in r.get("precisions", [])]
            cov_thr  = args.min_coverage
            prec_thr = args.min_precision
            kept = sum(
                1 for c, p in zip(all_cov, all_prec)
                if c >= cov_thr and p >= prec_thr
            )
            drop_cov  = sum(1 for c in all_cov if c < cov_thr)
            drop_prec = sum(
                1 for c, p in zip(all_cov, all_prec)
                if c >= cov_thr and p < prec_thr
            )
            print(f"  min_coverage used   : {cov_thr:.2f}")
            print(f"  min_precision used  : {prec_thr:.2f}")
            print(f"  Coverage  range     : {min(all_cov):.3f} – {max(all_cov):.3f}")
            print(f"  Precision range     : {min(all_prec):.3f} – {max(all_prec):.3f}  (if any)")
            print(f"  Would keep (both)   : {kept}/{len(all_cov)}")
            print(f"  Drop (cov<{cov_thr:.0%})    : {drop_cov}/{len(all_cov)}")
            print(f"  Drop (prec<{prec_thr:.0%})  : {drop_prec}/{len(all_cov)}")
    print(f"  Outputs saved to    : {args.output_dir.resolve()}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
