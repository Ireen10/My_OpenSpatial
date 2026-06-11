# Refiner 对照实验

这个目录用于对比三条 preprocessing 分支在少量 ARKitScenes/EmbodiedScan 场景上的效果：

- `raw_no_refine`: `3dbox_filter -> depth_back_projection`
- `sam2_refine`: `3dbox_filter -> sam2_refiner -> depth_back_projection`
- `sam3_refine`: `3dbox_filter -> sam3_refiner -> depth_back_projection`

三条分支统一从生产入口运行，只通过 `--max_samples 5` 截取前 5 条输入样本：

```bash
python run.py --config refiner_exp/configs/raw_no_refine.yaml --output_dir refiner_exp/outputs/raw --max_samples 5
python run.py --config refiner_exp/configs/sam2_refine.yaml --output_dir refiner_exp/outputs/sam2 --max_samples 5
python run.py --config refiner_exp/configs/sam3_refine.yaml --output_dir refiner_exp/outputs/sam3 --max_samples 5
```

`run.py` 会在每个 `--output_dir` 下创建 `base_pipeline_<config_name>` 子目录，例如 `refiner_exp/outputs/raw/base_pipeline_raw_no_refine`。

## SAM2 NPU 后端

`Sam2Refiner` 支持两个后端：

- `segmenter_backend: sam2`: 继续使用官方 `sam2.SAM2ImagePredictor`，适合原 CUDA/CPU 路径。
- `segmenter_backend: transformers`: 使用 `transformers.Sam2Processor/Sam2Model`，适合 NPU，避免官方 SAM2 包里可能写死 CUDA 的路径。

当配置里 `device` 以 `npu` 开头且 `segmenter_backend` 为 `auto` 时，会自动选择 transformers 后端。当前 `refiner_exp/configs/sam2_refine.yaml` 已显式配置：

```yaml
segmenter_model: facebook/sam2-hiera-small
segmenter_backend: transformers
device: "npu:0,npu:1,npu:2,npu:3"
replicas_per_device: 1
use_multi_processing: true
num_workers: 4
```

transformers 后端按官方 SAM2 文档对齐调用：`processor(images=..., input_boxes=..., return_tensors="pt").to(device)`，模型调用传 `multimask_output=False`，再用 `processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])` 还原到原图尺寸。这样和原 `SAM2ImagePredictor.predict(..., multimask_output=False)` 保持同一输入口径：每个 coarse mask 先转成 `[x_min, y_min, x_max, y_max]` box prompt，每个 prompt 只取一个 best mask，并用 `outputs.iou_scores` 对齐原来的 score 过滤逻辑。

NPU 并行策略是四张卡每张卡一个 `Sam2Model` 实例，避免同一张 NPU 卡部署多个实例触发 AICPU 错误。transformers 路径对每张图单独做一次 forward（`Sam2Processor` 无法对不同 object 数量的多图 batch 做 box padding）。并行仍由 `num_workers` 与多卡 replica 提供。该路径需要环境中安装支持 SAM2 的新版 `transformers`，以及 Ascend 运行所需的 `torch_npu`。

## IO 适配结论

raw/no-refine 分支不需要额外的数据适配器阶段。

`ThreeDBoxFilter` 输出 `masks`、`obj_tags`、`bboxes_3d_world_coords`，并保留 `image`、`depth_map`、`depth_scale`、`pose`、`intrinsic` 等上游字段。`DepthBackProjecter` 的硬性输入是 `intrinsic`、`depth_map`、`depth_scale`、`masks`、`obj_tags`，所以 `filter_stage/3dbox_filter` 的输出可以直接接到 `scene_fusion_stage/depth_back_projection`。

raw 分支不会产生 `bboxes_2d`，但这不影响 scene fusion。本实验在汇总和可视化阶段统一从 mask 动态计算 2D bbox，避免为了实验侵入式修改 `filter` 或 `scene_fusion`。如果后续要把 raw 分支直接接入严格依赖 `bboxes_2d` 的 annotation/group 流程，再新增一个独立适配器阶段更合适。

## 汇总与可视化

运行三条分支后生成汇总：

```bash
python refiner_exp/scripts/summarize_runs.py \
  --raw-run refiner_exp/outputs/raw \
  --sam2-run refiner_exp/outputs/sam2 \
  --sam3-run refiner_exp/outputs/sam3 \
  --output-dir refiner_exp/outputs/compare
```

输出：

- `summary.json`: 机器可读的分支级指标。
- `summary.md`: 便于快速查看的文字摘要。
- `object_metrics.csv`: 对象粒度的 mask、bbox、pointcloud 指标（每行一个 object × stage）。

### Web 可视化

启动本地服务后，用浏览器打开（默认绑定 `0.0.0.0:8848`，控制台会打印本机与局域网 URL）：

```bash
python refiner_exp/scripts/serve_compare.py \
  --raw-run refiner_exp/outputs/raw \
  --sam2-run refiner_exp/outputs/sam2 \
  --sam3-run refiner_exp/outputs/sam3 \
  --port 8848
```

浏览器访问 **http://127.0.0.1:8848/**：

| 路径 | 内容 |
|------|------|
| `/` | 样本列表 |
| `/view/{name}` | 单样本页：2D 对照（每物体单独一张）+ RAW/SAM2/SAM3 三列可旋转点云 |

交互操作：左键拖拽旋转、滚轮缩放、右键平移；**点击 2D 对照图放大**（Esc 关闭）。

**3D 点云坐标系**：fusion `.pcd` 为 OpenCV 相机系（X 右、Y 下、Z 朝前、近小远大）；Web 查看器仅翻转 Y 为 `(x, -y, z)`，默认相机在 **-Z** 侧朝 +Z 看，与 RGB 拍摄视角一致。

**数据加载方式**（按需，非预生成大 HTML）：

- 2D 图：`/api/sample/{name}/overlay`（首次渲染后缓存到 `compare/cache/`）
- 点云：`/api/sample/{name}/points/{branch}`，读取 fusion 的 **per-object `.pcd`**
- 指标：`/api/sample/{name}/stats`

重跑 pipeline 后**重启服务**即可看到新数据；2D 缓存可加 `?refresh=1` 强制重绘。

2D 对照图布局：RAW / SAM2 / SAM3 三列，每列按物体纵向排列（每张图仅绘制该物体的 mask、2D 框、3D 线框，避免重叠）。

常用参数：`--max-images`、`--max-points-per-object`（默认 8000）、`--host` / `--port`。

---

## 对照试验：阶段与对齐方式

汇总脚本从各分支 parquet 中读取三个阶段的数据：

| 阶段标签 | raw 分支数据来源 | sam2 / sam3 分支数据来源 |
|----------|------------------|--------------------------|
| `filter` | `filter_stage/3dbox_filter` | 同上 |
| `refine` | `filter_stage/3dbox_filter`（无 refine，与 filter 相同） | `localization_stage/sam2_refiner` 或 `sam3_refiner` |
| `fusion` | `scene_fusion_stage/depth_back_projection` | 同上 |


**相对 raw 的基线**：`enrich_against_raw` 使用 **raw 分支 filter 阶段** 的 object 记录作为 coarse mask 基线（不是 sam2/sam3 各自的 filter 输出）。

---

## 指标含义与计算公式

以下符号约定：

- \(M\)：二值 mask（`mask > 0`），像素数记为 \(|M|\)
- \(B_{\text{mask}}\)：由 mask 外接得到的 axis-aligned 2D bbox，格式 `[x1, y1, x2, y2]`（含边界像素）
- \(B_{\text{proj}}\)：将 `bboxes_3d_world_coords` 的 8 个角点经 `pose⁻¹` 与 `intrinsic` 投影到图像后取 min/max 得到的 2D bbox
- \(\text{area}(\cdot)\)：2D bbox 面积，**含端点**：\((x_2 - x_1 + 1)(y_2 - y_1 + 1)\)
- \(\text{inter}(A, B)\)：两 bbox 交集面积（同样含端点）

### 一、样本级 / 物体级计数

| 指标 | 含义 | 计算 |
|------|------|------|
| `samples` | 该阶段 parquet 行数 | `len(df)` |
| `objects` | 该阶段展开后的物体记录总数 | 各行 `max(len(tags), len(masks), len(boxes), len(pcds))` 之和 |
| `mean_objects_per_sample` | 平均每图物体数 | `objects / samples` |

### 二、保留率（retention）

在**同一分支内**比较各阶段物体数量（按 object 记录条数，不是 sample 数）：

| 指标 | 含义 | 公式 |
|------|------|------|
| `refine_vs_filter` | refine 相对 filter 保留了多少物体 | \(N_{\text{refine}} / N_{\text{filter}}\) |
| `fusion_vs_filter` | fusion 相对 filter 保留了多少物体 | \(N_{\text{fusion}} / N_{\text{filter}}\) |
| `fusion_vs_refine` | fusion 相对 refine 保留了多少物体 | \(N_{\text{fusion}} / N_{\text{refine}}\) |

- 值为 `1.0` 表示该阶段没有因 refine 门控或 fusion 空点云等原因丢失物体。
- 值 `< 1` 表示有物体在中途被丢弃（SAM2 score 过低、SAM3 coverage/precision 不达标、点云为空等）。
- raw 分支的 `refine_vs_filter` 恒为 `1.0`（refine 即 filter）。

### 三、Mask 几何指标（单 object，单 stage）

| 字段 | 含义 | 公式 |
|------|------|------|
| `mask_area` | mask 前景像素数 | \(\|M\| = \sum_{p} M(p)\) |
| `mask_bbox` | mask 轴对齐外接框 | \(x_1=\min x,\; y_1=\min y,\; x_2=\max x,\; y_2=\max y\)（对 \(M(p)>0\) 的像素） |
| `mask_bbox_area` | 外接框面积 | \(\text{area}(B_{\text{mask}})\) |
| `mask_bbox_fill_ratio` | mask 在外接框内的填充率 | \(\|M\| / \text{area}(B_{\text{mask}})\) |
| `mask_components` | 8-连通域个数 | `scipy.ndimage.label`（无 scipy 时退化为 1） |
| `mask_max_component_ratio` | 最大连通域占 mask 面积比例 | \(A_{\max} / \|M\|\)，其中 \(A_{\max}\) 为最大连通域像素数 |

**解读**：

- `mask_bbox_fill_ratio` 低 → mask 碎、空洞多，或形状很细长。
- `mask_max_component_ratio` 低 → 多个离散碎片，refine 可能把背景或邻物并进来。
- `mask_components` 很大 → 噪声点或过分割。

### 四、Mask 与 3D box 投影的一致性

| 字段 | 含义 | 公式 |
|------|------|------|
| `projected_3d_bbox` | 3D 标注框投影到像素的 2D 外接框 \(B_{\text{proj}}\) | 8 角点世界坐标 → 相机系（\(z>10^{-3}\)）→ 针孔投影 → min/max |
| `bbox_projected_iou` | mask 外接框与 3D 投影框的 IoU | \(\text{inter}(B_{\text{mask}}, B_{\text{proj}}) / \text{area}(B_{\text{mask}} \cup B_{\text{proj}})\) |
| `bbox_projected_coverage` | 3D 投影框被 mask 外接框覆盖的比例（召回式） | \(\text{inter}(B_{\text{mask}}, B_{\text{proj}}) / \text{area}(B_{\text{proj}})\) |
| `bbox_inside_projected_ratio` | mask 外接框落在 3D 投影框内的比例（精确式） | \(\text{inter}(B_{\text{mask}}, B_{\text{proj}}) / \text{area}(B_{\text{mask}})\) |

**解读**：

- `bbox_projected_iou` 综合衡量 2D mask 与 3D 标注框在当前视角下的空间一致性。
- `bbox_projected_coverage` 低 → mask 没盖住 3D 框应有的图像区域（欠分割或框偏大）。
- `bbox_inside_projected_ratio` 低 → mask 大量伸出 3D 框外（过分割或漂到背景）。

### 五、相对 raw coarse mask 的变化（refine / fusion 阶段）

仅对能在 raw filter 基线中找到相同 `object_key` 的物体计算：

| 字段 | 含义 | 公式 |
|------|------|------|
| `present_in_raw` | 该 object 是否存在于 raw filter 基线 | 布尔值 |
| `mask_iou_with_raw` | 当前 mask 与 raw coarse mask 的 IoU | \(\|M \cap M_{\text{raw}}\| / \|M \cup M_{\text{raw}}\|\) |
| `mask_area_ratio_vs_raw` | 面积相对 raw 的变化倍率 | \(\|M\| / \|M_{\text{raw}}\|\) |
| `bbox_center_shift_vs_raw` | mask 外接框中心位移（像素） | \(\sqrt{(c_x - c_x^{\text{raw}})^2 + (c_y - c_y^{\text{raw}})^2}\)，其中 \(c\) 为 bbox 中心 |

**解读**：

- `mask_iou_with_raw` 高且 `mask_area_ratio_vs_raw` 接近 1 → refine 轻微修边，语义稳定。
- `mask_iou_with_raw` 低但 `mask_area_ratio_vs_raw` 很大 → 可能扩到邻物或背景（SAM2 常见）。
- `mask_area_ratio_vs_raw` 很小 → 可能过删（SAM3 coverage/precision 门控常见）。
- `bbox_center_shift_vs_raw` 大 → mask 整体平移，即使 IoU 尚可也可能存在对齐问题。

`summary.md` 中 **Quality Versus Raw** 表对上述三列取 refine 阶段所有有效 object 的算术平均。

### 六、点云指标（fusion 阶段）

从 `depth_back_projection` 写出的 `.pcd` 读取（相机坐标系点云）：

| 字段 | 含义 | 公式 |
|------|------|------|
| `point_count` | 点云点数 | \(N\) |
| `pointcloud_aabb_volume` | 点云轴对齐包围盒体积 | \(\prod_d (\max x_d - \min x_d)\) |
| `pointcloud_inside_box_ratio` | 点落在**相机系** 3D 标注框内的比例 | 将 `bboxes_3d_world_coords` 变到相机系后，点 \(p\) 满足 \(\|(p-c)R\|_\infty \le \text{extent}/2\) 的比例 |
| `pointcloud_center_distance_to_box` | 点云质心到 3D 框中心的欧氏距离 | \(\|\bar{p} - c\|_2\) |

**解读**：

- `point_count` 相对 raw 骤降 → mask 收缩或 fusion 空点云剔除。
- `pointcloud_inside_box_ratio` 低 → 反投影点大量落在标注 3D 框外，mask 与 3D 约束不一致。
- `pointcloud_aabb_volume` 结合 `point_count` 可看点的空间分散程度（稀疏 vs 紧凑）。

### 七、分支级汇总（`summary.json` / `summary.md`）

各 stage 的 `_stage_summary` 对 object 级字段取算术平均：

| 汇总字段 | 来源字段 |
|----------|----------|
| `mean_mask_area` | `mask_area` |
| `mean_mask_fill_ratio` | `mask_bbox_fill_ratio` |
| `mean_bbox_projected_iou` | `bbox_projected_iou` |
| `mean_point_count` | `point_count` |
| `mean_pointcloud_inside_box_ratio` | `pointcloud_inside_box_ratio` |

`quality_vs_raw` 块：

| 汇总字段 | 来源字段 |
|----------|----------|
| `mean_mask_iou_with_raw` | `mask_iou_with_raw`（refine 阶段） |
| `mean_mask_area_ratio_vs_raw` | `mask_area_ratio_vs_raw`（refine 阶段） |
| `mean_bbox_center_shift_vs_raw` | `bbox_center_shift_vs_raw`（refine 阶段） |

---

## `object_metrics.csv` 列说明速查

每行对应一个 `(branch, stage, object_key)` 记录。常用列：

| 列名 | 类型 | 说明 |
|------|------|------|
| `branch` | str | `raw` / `sam2` / `sam3` |
| `stage` | str | `filter` / `refine` / `fusion` |
| `image_key` | str | 图像对齐键 |
| `object_key` | str | 跨分支物体对齐键 |
| `object_index` | int | 该行内 list 下标（**仅分支内有效**） |
| `tag` | str | 物体类别标签 |
| `mask_path` | str | mask PNG 路径 |
| `box_3d` | list | 9 维世界系 3D box |
| `mask_area` … `mask_max_component_ratio` | 见第三节 |
| `bbox_projected_iou` … `bbox_inside_projected_ratio` | 见第四节 |
| `present_in_raw` … `bbox_center_shift_vs_raw` | 见第五节 |
| `point_count` … `pointcloud_inside_box_ratio` | 见第六节 |

---

## 推荐解读

好的 refine 应该让 mask 更贴合可见物体，同时不让点云数量异常下降，也不让 bbox 中心大幅偏离原 3D box 投影。可疑结果通常表现为：

| 现象 | 可能原因 | 关注指标 |
|------|----------|----------|
| 物体大量丢失 | refine 门控过严 | `refine_vs_filter` ↓ |
| fusion 后进一步丢失 | 空点云 / 离群点剔除 | `fusion_vs_refine` ↓，`point_count` ↓ |
| mask 扩到背景/邻物 | SAM2 过分割 | `mask_area_ratio_vs_raw` ↑，`mask_iou_with_raw` ↓，`bbox_inside_projected_ratio` ↓ |
| mask 被削掉大半 | SAM3 coverage/precision 门控 | `mask_area_ratio_vs_raw` ↓，`mask_iou_with_raw` ↓ |
| 3D 框与 2D mask 脱节 | 标注框本身偏差或视角问题 | `bbox_projected_iou` ↓ |
| 点云与 3D 框不一致 | mask 错但反投影仍有点 | `pointcloud_inside_box_ratio` ↓ |

**SAM3** 需特别关注 coverage/precision 门控带来的误删；**SAM2** 需特别关注 score 通过但 mask 扩到邻近物体或背景的情况。

**横向对比建议顺序**：

1. 先看 `retention`：refine/fusion 是否过度丢物体。
2. 再看 `mean_mask_iou_with_raw` 与 `mean_mask_area_ratio_vs_raw`：相对 coarse mask 是「修边」还是「重画」。
3. 结合 `mean_bbox_projected_iou` 与可视化中的 3D 线框：判断 2D mask 是否仍被 3D 标注约束。
4. 最后看 `mean_point_count` 与 `mean_pointcloud_inside_box_ratio`：下游 fusion 是否仍产出合理几何。
