# EmbodiedScan Data Preprocessing

Preprocessing pipeline for extracting multi-view 3D perception data from [EmbodiedScan](https://github.com/OpenRobotLab/EmbodiedScan) datasets and converting them into the OpenSpatial Parquet format.

Supports 4 datasets: **ScanNet**, **3RScan**, **Matterport3D**, **ARKitScenes**.

## Installation

```bash
# Install EmbodiedScan from source (required dependency).
# The [visual] extras pull in open3d, which embodiedscan.explorer imports at module load.
git clone https://github.com/OpenRobotLab/EmbodiedScan.git
cd EmbodiedScan
# Workaround for upstream packaging bug: top-level `embodiedscan/` lacks an
# __init__.py so modern setuptools' find_packages() registers nothing on
# `pip install -e`. Tracked in InternRobotics/EmbodiedScan#117 — drop this
# line once it's merged.
touch embodiedscan/__init__.py
pip install -e ".[visual]"
cd ..

# Install this preprocessing package
cd OpenSpatial/data_preprocessing/embodiedscan
pip install -e .
```

## Prerequisites

1. **Raw dataset files** -- Download the original datasets following the [EmbodiedScan data documentation](https://github.com/OpenRobotLab/EmbodiedScan/blob/main/data/README.md). You only need to download the datasets you plan to use. For ARKitScenes, use the official [ARKitScenes](https://github.com/apple/ARKitScenes) layout under `arkitscenes_highres/raw/{Training,Validation}/<scene_id>/` (default `--raw-root`; see tree below). Do not confuse this with `arkitscenes/`, which `prepare-arkit` creates later.

2. **EmbodiedScan annotation files** -- Download the `.pkl` annotation files and place them as shown below.

## Data Directory Structure

Use **one** directory as `--data-root` (and default `--raw-root` = `<data-root>/arkitscenes_highres`). In this repo that is typically `OpenSpatial/data_root/EmbodiedScan/data` — do not add a second folder or Windows junction alias for the same tree.

The `--data-root` argument should point to a directory with the following layout:

```
<data-root>/
├── scannet/
│   ├── posed_images/
│   │   └── <scene_id>/              # e.g., scene0000_01
│   │       ├── 00000.jpg             # RGB image
│   │       ├── 00000.png             # depth map (16-bit)
│   │       └── ...
│   └── scans/
│       └── <scene_id>/
│           └── intrinsic/
│               └── intrinsic_depth.txt   # 4x4 intrinsic matrix
│
├── 3rscan/
│   └── <scene_uuid>/
│       └── sequence/
│           ├── _info.txt             # contains m_calibrationDepthIntrinsic
│           ├── frame-000000.color.jpg
│           └── ...
│
├── matterport3d/
│   └── <building_id>/
│       ├── region_segmentations/
│       │   └── *.ply
│       ├── matterport_color_images/
│       │   └── *.jpg
│       ├── matterport_camera_intrinsics/
│       │   └── *.txt                 # width height fx fy cx cy ...
│       └── matterport_depth_images/
│           └── *.png                 # 16-bit depth (depth_scale=4000)
│
├── arkitscenes_highres/            # official raw download (default --raw-root)
│   ├── metadata.csv                # optional; also accepted under raw/
│   └── raw/
│       ├── Training/
│       │   └── <scene_id>/
│       │       ├── vga_wide/              # *.png, 640×480 @ 30 FPS
│       │       ├── vga_wide_intrinsics/   # *.pincam
│       │       ├── lowres_depth/            # *.png (default depth source)
│       │       ├── lowres_wide.traj
│       │       └── highres_depth/           # optional; upsampling scenes only
│       └── Validation/
│           └── <scene_id>/                # same layout as Training
│
├── embodiedscan_infos_train.pkl      # EmbodiedScan v1 annotations
├── embodiedscan_infos_val.pkl
└── embodiedscan_infos_test.pkl
```

`prepare-arkit` writes prepared assets under `<data-root>/arkitscenes/{Training,Validation}/<scene_id>/<scene_id>_frames/` (`vga_wide/`, `vga_depth/`, `vga_wide_intrinsics/`, `.arkit_scene.json`). That directory is **not** required before Step 0.

ARKitScenes requires v2 annotations in a sibling directory:

```
<data-root>/../embodiedscan-v2/
├── embodiedscan_infos_train.pkl
├── embodiedscan_infos_val.pkl
└── embodiedscan_infos_test.pkl
```

## Usage

For **ARKitScenes**, run **prepare-arkit** once before extract. **Defaults:** `vga_wide` RGB; depth from `highres_depth` or `lowres_depth` at the **same timestamp**, sky-rotate then resize to rotated vga size (**CUT3R**-aligned: `INTER_NEAREST` / `INTER_NEAREST_EXACT`); **per-frame** sky from `lowres_wide.traj`. See `data_root/EmbodiedScan/arkitscenes说明.md` for the algorithm steps. World 3D boxes unchanged; `extract` applies per-frame `rotated_to_cam` to poses.

The pipeline has 5 steps for ARKit: **prepare-arkit** -> **extract** -> **merge** -> **export** -> **validate**.

### Step 0: Prepare ARKitScenes (vga_wide + per-frame sky correction)

Download [ARKitScenes raw](https://github.com/apple/ARKitScenes) into `<data-root>/arkitscenes_highres/` (official layout: `arkitscenes_highres/raw/Training/<scene_id>/`). On Windows, unzip with `scripts/unzip_arkitscenes_raw.py` after `download_data.py` (no `unzip` CLI).

```bash
# Download (example)
python tools/ARKitScenes/download_data.py raw --split Training --video_id 40753679 \
  --download_dir /path/to/EmbodiedScan/data/arkitscenes_highres \
  --raw_dataset_assets vga_wide vga_wide_intrinsics lowres_depth lowres_wide.traj

python data_preprocessing/embodiedscan/scripts/unzip_arkitscenes_raw.py --raw-root .../arkitscenes_highres

# Prepare -> writes to <data-root>/arkitscenes/.../<scene_id>_frames/
# Example (from OpenSpatial repo root):
python -m embodiedscan_data prepare-arkit \
  --data-root ./data_root/EmbodiedScan/data \
  --raw-root ./data_root/EmbodiedScan/data/arkitscenes_highres \
  --only-local-raw \
  --workers 8
# Defaults: --sky-granularity frame, RGB/depth from vga_wide (+ lowres_depth resize)
# Legacy scene-level sky: add -sg scene [-ss metadata|traj]  (long: --sky-granularity / --sky-source)
```

Annotations: **EmbodiedScan v2** pkls in `<project>/embodiedscan-v2/` (not 3DOD JSON). `bbox_3d` is **world / axis-aligned**; sky correction updates camera pose only (per-frame when using default `frame` granularity).

**`.arkit_scene.json` (default `frame` mode):** `sky_granularity`, `frames[<camera>]` with `sky_direction`, `rotated_to_cam`, `traj_ts`; scene-level `sky_direction` is the dominant label for logging only.

### Step 1: Extract (per-image)

ARKitScenes only — choose sensor layout at extract time (other datasets ignore this flag):

| `--arkit-asset-mode` | Behavior |
|--------------------|----------|
| `auto` (default) | Use prepared `vga_wide` / `vga_depth` if `.arkit_scene.json` exists, else `lowres_*` |
| `vga` | Force prepared vga assets (skip scene if not prepared) |
| `lowres` | Force `lowres_wide` / `lowres_depth` (ignore prepared vga on disk) |

```bash
# Single dataset
python -m embodiedscan_data extract \
  --dataset scannet \
  --data-root /path/to/data \
  --output ./output \
  --workers 24

# All datasets
python -m embodiedscan_data extract \
  --dataset all \
  --data-root /path/to/data \
  --output ./output \
  --workers 24

# Smoke test (limit scenes)
python -m embodiedscan_data extract \
  --dataset scannet \
  --data-root /path/to/data \
  --output ./output \
  --workers 4 \
  --max-scenes 2

# ARKit: lowres, no sky (same disk may also have prepared vga — use lowres to ignore it)
python -m embodiedscan_data extract \
  --dataset arkitscenes \
  --data-root ./data_root/EmbodiedScan/data \
  --output ./output_lowres \
  --arkit-asset-mode lowres
```

Outputs per-image JSONL files (e.g., `scannet.jsonl`). Supports resume -- rerunning skips already-extracted records.

### Step 2: Merge (per-scene)

```bash
python -m embodiedscan_data merge --input ./output
```

Groups per-image records by `scene_id` into per-scene JSONL files (e.g., `scannet_scenes.jsonl`).

### Step 3: Export (to Parquet)

```bash
python -m embodiedscan_data export --input ./output --format both
```

Converts JSONL to sharded Parquet files under `per_image/` and `per_scene/` subdirectories.

### Step 4: Validate

```bash
python -m embodiedscan_data validate \
  --input ./output \
  --data-root /path/to/data
```

Checks schema completeness, record counts, value ranges, and file path reachability.

## Output Schema

### Per-image record

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique record ID (e.g., `scannet__scene0000_01__00000`) |
| `dataset` | `str` | Source dataset name |
| `scene_id` | `str` | Scene identifier |
| `image` | `str` | RGB image path (relative to `--data-root`) |
| `depth_map` | `str` | Depth map path |
| `pose` | `str` | 4x4 camera-to-world extrinsic matrix (txt) |
| `intrinsic` | `str` | 4x4 camera intrinsic matrix (txt) |
| `depth_scale` | `int` | Depth scale factor (1000 or 4000) |
| `is_metric_depth` | `bool` | Always `true`; depth values are metric after `depth_scale` division |
| `bboxes_3d_world_coords` | `list[list[float]]` | 3D OBBs `[cx,cy,cz,xl,yl,zl,roll,pitch,yaw]` |
| `obj_tags` | `list[str]` | Object semantic labels |
| `axis_align_matrix` | `str` | Axis alignment matrix path |

### Per-scene record

Per-image fields are aggregated into lists, with `dataset` and `scene_id` kept as scalars. An additional `num_images` field records the view count.

## Supported Datasets

| Dataset | depth_scale | Annotations | Notes |
|---------|-------------|-------------|-------|
| ScanNet | 1000 | v1 | Images resized to match depth map dimensions |
| 3RScan | 1000 | v1 | Intrinsic parsed from `_info.txt` |
| Matterport3D | 4000 | v1 | Region-level scenes; cameras per region from v1 pkl (not whole-building) |
| ARKitScenes | 1000 | v2 | `prepare-arkit` (default: `vga_wide`, per-frame traj sky) → `extract` uses `vga_wide`/`vga_depth`; per-frame `rotated_to_cam` on pose (boxes unchanged) |
