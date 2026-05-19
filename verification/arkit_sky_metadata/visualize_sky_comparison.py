"""Four-way ARKit sky correction comparison (verification only; not pipeline output).

Columns: raw vga_wide | metadata (scene) | traj global stat | per-frame traj.

Outputs under ``verification/arkit_sky_metadata/results/sky_4way_viz/`` by default.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "data_preprocessing" / "embodiedscan"))

from embodiedscan_data.arkit_geometry import (  # noqa: E402
    find_asset_file,
    find_frame_orientation,
    find_scene_orientation,
    load_sky_direction_from_metadata,
    nearest_traj_pose,
    read_traj,
    rotate_rgb_image,
)
from embodiedscan_data.datasets.arkitscenes import (  # noqa: E402
    ARKIT_SPLITS,
    _arkit_scene_cameras_index,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results" / "sky_4way_viz"

Sky = str  # UP | DOWN | LEFT | RIGHT

COLUMNS = (
    ("raw", "Raw vga_wide"),
    ("metadata", "metadata.csv (scene)"),
    ("global", "traj global stat (+Z)"),
    ("frame", "traj per-frame (+Z)"),
)


def _project_root_from_data_root(data_root: str) -> str:
    return os.path.dirname(os.path.abspath(data_root))


def _resolve_scene_dir(raw_root: str, split: str, scene_id: str) -> Optional[str]:
    for base in (
        os.path.join(raw_root, split, scene_id),
        os.path.join(raw_root, "raw", split, scene_id),
    ):
        if os.path.isdir(base):
            return base
    return None


def _asset_search_roots(scene_dir: str, scene_id: str) -> List[str]:
    roots = [
        scene_dir,
        os.path.join(scene_dir, f"{scene_id}_frames"),
        os.path.join(scene_dir, scene_id),
    ]
    out: List[str] = []
    seen = set()
    for root in roots:
        if root not in seen and os.path.isdir(root):
            out.append(root)
            seen.add(root)
    return out


def _traj_path(scene_dir: str, scene_id: str) -> Optional[str]:
    for root in _asset_search_roots(scene_dir, scene_id):
        path = os.path.join(root, "lowres_wide.traj")
        if os.path.isfile(path):
            return path
    return None


def _camera_timestamp(camera_name: str, scene_id: str) -> str:
    prefix = f"{scene_id}_"
    if camera_name.startswith(prefix):
        return camera_name[len(prefix) :]
    parts = camera_name.split("_", 1)
    return parts[1] if len(parts) == 2 else camera_name


def _load_font(size: int = 16) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _annotate_tile(img: Image.Image, title: str, sky: Optional[Sky]) -> Image.Image:
    font = _load_font(15)
    label = title if sky is None else f"{title}\n{sky}"
    pad = 4
    line_h = 18
    lines = label.split("\n")
    banner_h = pad * 2 + line_h * len(lines)
    out = Image.new("RGB", (img.width, img.height + banner_h), (32, 32, 32))
    out.paste(img, (0, banner_h))
    draw = ImageDraw.Draw(out)
    y = pad
    for line in lines:
        draw.text((pad, y), line, fill=(240, 240, 240), font=font)
        y += line_h
    return out


def _compose_row(tiles: List[Image.Image]) -> Image.Image:
    h = max(t.height for t in tiles)
    w = sum(t.width for t in tiles)
    row = Image.new("RGB", (w, h), (0, 0, 0))
    x = 0
    for tile in tiles:
        row.paste(tile, (x, 0))
        x += tile.width
    return row


def _all_correction_up(meta_sky: Sky, global_sky: Sky, frame_sky: Sky) -> bool:
    """True when all three correction labels are UP (raw already upright)."""
    return meta_sky == "UP" and global_sky == "UP" and frame_sky == "UP"


def _collect_scenes(
    project_root: str,
    scene_filter: Optional[Sequence[str]],
    only_local_raw: bool,
    raw_root: str,
) -> List[Tuple[str, str, str]]:
    index = _arkit_scene_cameras_index(project_root)
    scenes: List[Tuple[str, str, str]] = []
    for scene_key, _ in sorted(index.items()):
        if scene_filter and scene_key not in scene_filter:
            continue
        parts = scene_key.split("/")
        if len(parts) != 3 or parts[0] != "arkitscenes" or parts[1] not in ARKIT_SPLITS:
            continue
        _, split, scene_id = parts
        if only_local_raw and _resolve_scene_dir(raw_root, split, scene_id) is None:
            continue
        scenes.append((scene_key, split, scene_id))
    return scenes


def _write_index(
    out_dir: Path,
    entries: List[dict],
    skipped_all_up: int,
    total_candidates: int,
) -> None:
    rows = []
    for e in entries:
        rel = e["relpath"].replace("\\", "/")
        cap = html.escape(
            f"{e['scene_key']} / {e['camera']} | "
            f"meta={e['meta']} global={e['global']} frame={e['frame']}"
        )
        rows.append(
            f'<figure><a href="{rel}"><img src="{rel}" width="960"></a>'
            f"<figcaption>{cap}</figcaption></figure>"
        )
    body = "\n".join(rows) if rows else "<p>No images (all filtered or failed).</p>"
    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>ARKit sky 4-way</title>
<style>
body {{ font-family: sans-serif; background: #111; color: #eee; margin: 1rem; }}
figure {{ margin: 1.5rem 0; }}
figcaption {{ font-size: 0.9rem; color: #aaa; margin-top: 0.25rem; }}
.stats {{ background: #222; padding: 0.75rem 1rem; border-radius: 6px; }}
</style></head><body>
<h1>ARKit sky 4-way comparison</h1>
<div class="stats">
<p>Shown: {len(entries)} / {total_candidates} frames</p>
<p>Skipped (--skip-all-up): {skipped_all_up}</p>
</div>
{body}
</body></html>"""
    (out_dir / "index.html").write_text(page, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    data_root = os.path.abspath(args.data_root)
    raw_root = os.path.abspath(args.raw_root or os.path.join(data_root, "arkitscenes_highres"))
    project_root = _project_root_from_data_root(data_root)
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_filter = list(args.scene) if args.scene else None
    scenes = _collect_scenes(project_root, scene_filter, args.only_local_raw, raw_root)
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]

    if not scenes:
        raise SystemExit("No scenes to visualize (check v2 pkls, raw paths, --only-local-raw).")

    thumb_w = args.thumb_width
    entries: List[dict] = []
    skipped_all_up = 0
    total_candidates = 0
    failed = 0

    for scene_key, split, scene_id in scenes:
        scene_dir = _resolve_scene_dir(raw_root, split, scene_id)
        if scene_dir is None:
            print(f"Skip missing raw: {scene_key}")
            continue
        traj_path = _traj_path(scene_dir, scene_id)
        if traj_path is None:
            print(f"Skip missing traj: {scene_key}")
            continue

        timestamps, poses = read_traj(traj_path)
        global_sky, _ = find_scene_orientation(poses)
        meta_sky = load_sky_direction_from_metadata(scene_id, raw_root)
        if meta_sky is None:
            meta_sky = global_sky

        vga_dirs = [
            os.path.join(r, "vga_wide")
            for r in _asset_search_roots(scene_dir, scene_id)
        ]
        cameras = _arkit_scene_cameras_index(project_root).get(scene_key, ())
        if not cameras:
            continue

        scene_out = out_dir / split / scene_id
        scene_out.mkdir(parents=True, exist_ok=True)

        for camera in tqdm(cameras, desc=scene_id, leave=False):
            total_candidates += 1
            ts = _camera_timestamp(camera, scene_id)
            try:
                pose, _, _ = nearest_traj_pose(timestamps, poses, float(ts))
                frame_sky, _ = find_frame_orientation(pose)
            except (ValueError, IndexError):
                failed += 1
                continue

            if args.skip_all_up and _all_correction_up(meta_sky, global_sky, frame_sky):
                skipped_all_up += 1
                continue

            vga_path = find_asset_file(vga_dirs, scene_id, ts, ".png")
            if not vga_path:
                failed += 1
                continue

            raw_img = Image.open(vga_path).convert("RGB")
            tiles: List[Image.Image] = []
            for col_id, col_title in COLUMNS:
                if col_id == "raw":
                    sky = None
                    tile = raw_img.copy()
                elif col_id == "metadata":
                    sky = meta_sky
                    tile = rotate_rgb_image(raw_img, sky)
                elif col_id == "global":
                    sky = global_sky
                    tile = rotate_rgb_image(raw_img, sky)
                else:
                    sky = frame_sky
                    tile = rotate_rgb_image(raw_img, sky)
                if tile.width > thumb_w:
                    scale = thumb_w / tile.width
                    tile = tile.resize(
                        (thumb_w, max(1, int(tile.height * scale))),
                        Image.Resampling.BILINEAR,
                    )
                tiles.append(_annotate_tile(tile, col_title, sky))

            row = _compose_row(tiles)
            rel_name = f"{camera}.jpg"
            out_path = scene_out / rel_name
            row.save(out_path, quality=92)
            entries.append(
                {
                    "relpath": str(out_path.relative_to(out_dir)).replace("\\", "/"),
                    "scene_key": scene_key,
                    "camera": camera,
                    "meta": meta_sky,
                    "global": global_sky,
                    "frame": frame_sky,
                }
            )

    _write_index(out_dir, entries, skipped_all_up, total_candidates)
    summary = {
        "scenes": len(scenes),
        "shown": len(entries),
        "skipped_all_up": skipped_all_up,
        "failed": failed,
        "total_candidates": total_candidates,
        "output_dir": str(out_dir),
        "skip_all_up": args.skip_all_up,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


def main() -> None:
    default_data = REPO_ROOT / "data_root" / "EmbodiedScan" / "data"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default=str(default_data),
        help="EmbodiedScan data root (for v2 pkl resolution)",
    )
    parser.add_argument(
        "--raw-root",
        default=None,
        help="ARKit raw root (default: <data-root>/arkitscenes_highres)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Visualization output directory",
    )
    parser.add_argument(
        "--only-local-raw",
        action="store_true",
        help="Only scenes present under --raw-root",
    )
    parser.add_argument(
        "--skip-all-up",
        action="store_true",
        help="Skip frames where metadata, global traj, and per-frame labels are all UP",
    )
    parser.add_argument(
        "--scene",
        action="append",
        default=None,
        help="Filter sample_idx, e.g. arkitscenes/Training/40753679 (repeatable)",
    )
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--thumb-width", type=int, default=320)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
