# 预处理管线输入数据 Schema

本文档约定：要通过 `run.py` 与 `config/preprocessing/*.yaml` 运行 OpenSpatial **处理阶段**（filter → localization → scene_fusion → 可选 group）时，**起始 Parquet** 必须满足哪些条件。

本文**不**涵盖 `data_preprocessing/*` 下的转换脚本（Hypersim、EmbodiedScan、ScanNet++ 等）；那些脚本应产出符合本 Schema 的 Parquet。

坐标系与 OBB 数学定义见 [quick_start.md](quick_start.md) §2.1–2.2。

---

## 1. 配置入口（`dataset` 块）

| 键 | 是否必填 | 说明 |
|----|----------|------|
| `modality` | 是 | 必须为 `image`。 |
| `dataset_name` | 与下两项二选一 | 简写：同时作为读、写后端（兼容旧配置）。 |
| `input_dataset_name` | 可选 | 读入与 stage 间 `depends_on` 加载用的后端，默认 `image_base`。 |
| `output_dataset_name` | 可选 | 落盘用的后端，默认与 `input_dataset_name` 相同。 |
| `data_dir` | 是 | 单个 `.parquet` 文件、parquet 分片目录，或符合 `owner/name` 格式的 HuggingFace 仓库 id。 |
| `raw_data_root` | 建议填写 | 与 Parquet 中 **相对路径** 拼接的根目录，作用于 `image`、`depth_map`、`pose`、`intrinsic`，以及可选的 `axis_align_matrix`。仅当 Parquet 内全部为绝对路径时可省略。 |

已注册后端见 `dataset/__init__.py`：`image_base`（Parquet）、`jsonl_base`（JSONL，见 `dataset/jsonl_base.py`）。

**读写解耦：** 可只改 `output_dataset_name`（例如 `jsonl_base`）。多 stage 时管线仍用 `data.parquet` 在 task 间传递；若读写后端不同，每次 `save_data` 会 **同时** 写 Parquet（供下一 task）和第二种格式（如 `data.jsonl`）。

```yaml
dataset:
  modality: image
  dataset_name: image_base
  data_dir: /path/to/input.parquet
  raw_data_root: /path/to/raw/rgb-d

# 或显式解耦，例如预处理 Parquet 进、最终额外导出 JSONL：
#   input_dataset_name: image_base
#   output_dataset_name: jsonl_base
#   output_path: data.jsonl
```

---

## 2. 两种输入粒度

各 stage 实现可同时支持两种表结构；**预处理 YAML** 必须与 Parquet 的行组织方式一致。

| 模式 | 行含义 | 锚点列形态 | 典型来源 | YAML 流程 |
|------|--------|------------|----------|-----------|
| **Per-image（逐帧）** | 1 行 = 1 帧 | 标量（`str`、`int`、`list` 等） | Hypersim `prepare_hypersim`、EmbodiedScan `per_image/` | `filter` → `sam2` → `fusion`（无 `flatten`） |
| **Per-scene（逐场景）** | 1 行 = 1 场景、多帧 | `image` 为长度 N 的 `list[str]` | ScanNet++ `prepare_scannetpp`、EmbodiedScan `per_scene/` | `flatten` → `filter` → `sam2` → `fusion` → 可选 `group` |

**规则（`SampleFlattener`）：** `split_col_list` 中的每一列必须是 **长度为 N 的 list**（与 `image` 一致）。不在 `split_col_list` 中的列会 **原样复制** 到每条展平后的行（例如共用的 `scene_id`）。

**规则（`SampleGrouper`）：** `group_col_list` 中的每一列在每条 per-image 行上都必须存在（或在创建分组首行时可初始化）。**不要** 列出 Parquet 中不存在的列（例如 Hypersim 没有 `axis_align_matrix`，应从 `group_col_list` 中去掉）。

---

## 3. 管线起点必备字段

以下字段须在进入 `filter_stage` 之前就存在（若使用 `flatten`，则指 **展平后的每条 per-image 行**）。

### 3.1 核心列

| 列名 | 类型（per-image 行） | 说明 |
|------|----------------------|------|
| `image` | `str` | RGB 图像路径（`.jpg` / `.png`），PIL 可读并会转为 RGB。 |
| `depth_map` | `str` | 深度文件：`.npy` 浮点数组，或图像（如 16-bit PNG）。见 §4.2。 |
| `depth_scale` | `int` | 加载深度时的除数（`load_depth_map`：`depth / depth_scale` → 米）。例：`1000`（ScanNet、ARKit、Hypersim），`4000`（Matterport3D）。 |
| `pose` | `str` | 4×4 **camera-to-world** 外参文本路径，空白分隔（`numpy.loadtxt`）。 |
| `intrinsic` | `str` | 4×4 相机内参路径，左上 3×3 有效；文件格式同 `pose`。 |
| `obj_tags` | `list[str]` | 每个物体一个语义标签；长度 = 该帧物体数。 |
| `bboxes_3d_world_coords` | `list[list[float]]` | 每个物体一个 9-DoF OBB，与 `obj_tags` 等长。见 §4.3。 |

### 3.2 强烈建议

| 列名 | 类型 | 说明 |
|------|------|------|
| `scene_id` | `str` | 场景 ID；若 `group_stage` 使用 `group_by: scene_id` 则 **必填**。 |
| `id` | `str` | 记录唯一 id（如 `scene-cam-frame`）；用于日志与 mask 文件名。 |
| `is_metric_depth` | `bool` | 缩放后深度是否为米制；会透传到下游标注。 |

### 3.3 可选（仅 loader 解析路径）

| 列名 | 类型 | 说明 |
|------|------|------|
| `axis_align_matrix` | `str` 或 `null` | 若存在且为相对路径，经 `raw_data_root` 拼接。EmbodiedScan：4×4 txt。ScanNet++/Hypersim 常省略。**`ThreeDBoxFilter` 不读取此列。** |

除非某 task 的 `update_keys` 会过滤列表字段，否则额外列会随记录在各 stage 间保留。

---

## 4. 磁盘文件约定

### 4.1 RGB（`image`）

- 解析后的路径下文件必须存在。
- **宽 × 高** 必须与深度图空间尺寸一致（filter 阶段会断言）。
- 非 RGB 时会自动转换。

### 4.2 深度（`depth_map` + `depth_scale`）

加载逻辑（`utils/image_utils.load_depth_map`）：

- `.npy`：读为 float64；若 `depth_scale` 非 0 则除以该值。
- 图像（如 PNG）：`astype(float64)` 后同样除以 `depth_scale`。

缩放后按 **米制深度** 用于反投影（与 `backproject_depth_to_3d` 及 [quick_start.md](quick_start.md) 中的相机 Z / 平面深度约定一致）。

常见编码：

| 数据集族 | 典型存储 | `depth_scale` |
|----------|----------|---------------|
| ScanNet / Hypersim / ARKit | uint16 PNG，毫米 | `1000` |
| Matterport3D | uint16 PNG，0.25 mm/单位 | `4000` |

### 4.3 位姿与内参（文本文件）

**`pose`：** 4×4 矩阵，**camera-to-world**（列向量）。世界坐标：`P_world = pose @ P_cam_homogeneous`。

**`intrinsic`：** 4×4 矩阵；左上 3×3 为焦距与主点：

```
fx  0  cx  0
0  fy  cy  0
0   0   1  0
0   0   0  1
```

投影使用的 `img_dim` 来自 RGB 图像的 `(width, height)`。

### 4.4 三维框（`bboxes_3d_world_coords`）

每个物体为 **世界坐标系** 下长度 9 的向量：

```text
[cx, cy, cz, xl, yl, zl, roll, pitch, yaw]
```

| 下标 | 含义 |
|------|------|
| 0–2 | 框中心（世界系） |
| 3–5 | 沿框局部轴的 **全长**（非半长） |
| 6–8 | 欧拉角，**zxy** 内旋顺序，**弧度** |

- 进入 filter 的每一行须满足 `len(obj_tags) == len(bboxes_3d_world_coords)`。
- YAML 中 `filter_tags` 列出的标签会在框校验前剔除（如 `ceiling`、`floor`、`wall`、`object`）。
- 剔除后至少保留一个物体，否则该行被丢弃。

### 4.5 标签（`obj_tags`）

- 需要处理的行上，`obj_tags` 为非空列表。
- 字符串为过滤与 QA 用的语义名；若依赖 YAML `filter_tags`，请保持词表一致。

---

## 5. 管线各阶段输入输出（per-image 行）

校验失败时 `apply_transform` 返回 `flag=False`，该行被丢弃；各阶段后行数通常会减少。

```text
INPUT（符合本 Schema）
    │
    ▼
[可选] flatten_stage          per-scene 的 list → per-image 标量
    │
    ▼
filter_stage (ThreeDBoxFilter)    + masks（粗 mask 的 PNG 路径）
    │                             − 无效 2D/3D 框、filter_tags
    ▼
localization_stage (Sam2Refiner)  + bboxes_2d、细化 masks
    │                             依赖 filter 的 masks
    │                             − SAM2 分数过低 / mask 过小
    ▼
scene_fusion_stage (DepthBackProjecter)  + pointclouds（每物体一个 .pcd）
    │                                    − 需多于 1 个有效点云
    ▼
[可选] group_stage (SampleGrouper)   按 scene_id 聚合成 list 列
```

### 5.1 各阶段依赖一览

| 阶段 | 输入要求 | 新增 / 更新 |
|------|----------|-------------|
| `flatten` | Per-scene：`image` 为 list，且 `split_col_list` 各列为等长 list | 输出 per-image 行 |
| `3dbox_filter` | §3.1 | `masks` |
| `sam2_refiner` | `image`、`masks`、`obj_tags`、`bboxes_3d_world_coords` | `masks`（细化）、`bboxes_2d` |
| `depth_back_projection` | `intrinsic`、`depth_map`、`depth_scale`、`masks`、`obj_tags`（等长） | `pointclouds` |
| `group` | Per-image 行；`group_col_list` 所列列均存在 | 每个 `group_by` 键一行；字段变为 list |

### 5.2 `group_col_list`（典型写法）

只列入 Parquet 中实际存在、且需要按场景聚合的列：

```yaml
group_col_list:
  - image
  - id
  - obj_tags
  - depth_map
  - pose
  - intrinsic
  - bboxes_3d_world_coords
  - masks
  - bboxes_2d
  - pointclouds
  - depth_scale
```

除非转换脚本在每一行都提供 `axis_align_matrix`，否则不要写入 `group_col_list`。

---

## 6. `group_stage` 之后的 per-scene（多视图）行形态

每个 `scene_id` 一行（示例：N 帧）：

| 列名 | 类型 |
|------|------|
| `scene_id` | `str` |
| `image` | `list[str]`（长度 N） |
| `pose` | `list[str]` |
| `intrinsic` | `list[str]` |
| `depth_map` | `list[str]` |
| `obj_tags` | `list[list[str]]` |
| `bboxes_3d_world_coords` | `list[list[list[float]]]` |
| `masks` | `list[list[str]]` |
| `bboxes_2d` | `list[list[list[int]]]` |
| `pointclouds` | `list[list[str]]` |
| `depth_scale` | `int` 或 `list`（常为标量重复） |

此为 **多视图** 标注配置（`SceneGraph.from_multiview_example`）的输入形态。

**不做 group** 时，`scene_fusion_stage` 的输出仍为 **per-image**，供 **单视图** 标注配置使用。

---

## 7. 内置转换脚本与 Schema 对应关系

| 转换脚本 | 默认 Parquet 粒度 | 与本 Schema |
|----------|-------------------|-------------|
| `prepare_hypersim.py` | Per-image | §3 逐帧 |
| `embodiedscan_data export per_image/` | Per-image | §3 + `axis_align_matrix` |
| `embodiedscan_data export per_scene/` | Per-scene | filter 前需 `flatten` |
| `prepare_scannetpp.py` | Per-scene | 需 `flatten`；`axis_align_matrix` 常为 null |

EmbodiedScan 的 **merge**（仅数据准备阶段）会得到与 `group` 输出 **形状类似** 的 per-scene JSONL，但 **没有** 管线生成的 `masks` / `pointclouds`，**不能** 替代 fusion 阶段输出。

---

## 8. 接入新数据集检查清单

1. 确定预处理 YAML 粒度：仅 per-image，或 per-scene + `flatten`。
2. 产出含 §3.1 列的 Parquet；确认路径在 `raw_data_root` 下可解析。
3. 深度编码与 `depth_scale` 一致；RGB 与深度尺寸一致。
4. `pose` / `intrinsic` 写为 4×4 文本；确认 `pose` 为 camera-to-world。
5. OBB 为世界系、9-DoF、zxy 弧度；列表长度与 `obj_tags` 对齐。
6. 若使用 `group_stage`，提供 `scene_id`。
7. YAML `filter_tags` 与自有标签词表一致。
8. `group_col_list` 只含真实存在的列，避免 phantom 字段。
9. 冒烟：小 Parquet → `run.py --config config/preprocessing/your.yaml --output_dir ...` → 检查各阶段行数。

---

## 9. 最小示例（一条 per-image 行）

逻辑内容如下：

```json
{
  "scene_id": "my_scene_001",
  "id": "my_scene_001-cam_00-0000",
  "image": "my_dataset/scene_001/rgb/0000.jpg",
  "depth_map": "my_dataset/scene_001/depth/0000.png",
  "depth_scale": 1000,
  "is_metric_depth": true,
  "pose": "my_dataset/scene_001/poses/0000.txt",
  "intrinsic": "my_dataset/scene_001/intrinsics/0000.txt",
  "obj_tags": ["chair", "table"],
  "bboxes_3d_world_coords": [
    [1.0, 2.0, 0.5, 0.6, 0.6, 1.0, 0.0, 0.0, 0.0],
    [2.5, 1.0, 0.4, 1.2, 0.8, 0.75, 0.0, 0.0, 1.57]
  ]
}
```

在 run 配置中设置 `raw_data_root` 时，路径为相对该根目录。

---

## 10. 相关配置文件

| 用途 | 示例配置 |
|------|----------|
| Per-image 预处理 | `config/preprocessing/demo_preprocessing_hypersim.yaml` |
| Per-scene 预处理 | `config/preprocessing/demo_preprocessing_scannetpp.yaml` |
| Per-image 预处理（无 group） | `config/preprocessing/demo_preprocessing_embodiedscan.yaml` |
| 单视图标注输入 | `scene_fusion_stage/.../data.parquet` |
| 多视图标注输入 | `group_stage/group/data.parquet`（per-image 管线 + group 之后） |

标注与预处理数据流详见 [quick_start.md](quick_start.md) §3.4。
