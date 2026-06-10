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
batch_size: 4
```

transformers 后端按官方 SAM2 文档对齐调用：`processor(images=..., input_boxes=..., return_tensors="pt").to(device)`，模型调用传 `multimask_output=False`，再用 `processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])` 还原到原图尺寸。这样和原 `SAM2ImagePredictor.predict(..., multimask_output=False)` 保持同一输入口径：每个 coarse mask 先转成 `[x_min, y_min, x_max, y_max]` box prompt，每个 prompt 只取一个 best mask，并用 `outputs.iou_scores` 对齐原来的 score 过滤逻辑。

NPU 并行策略是四张卡每张卡一个 `Sam2Model` 实例，避免同一张 NPU 卡部署多个实例触发 AICPU 错误。每个 worker 会一次处理一批图片和每张图片内的多个 box prompt。该路径需要环境中安装支持 SAM2 的新版 `transformers`，以及 Ascend 运行所需的 `torch_npu`。

## IO 适配结论

raw/no-refine 分支不需要额外的数据适配器阶段。

`ThreeDBoxFilter` 输出 `masks`、`obj_tags`、`bboxes_3d_world_coords`，并保留 `image`、`depth_map`、`depth_scale`、`pose`、`intrinsic` 等上游字段。`DepthBackProjecter` 的硬性输入是 `intrinsic`、`depth_map`、`depth_scale`、`masks`、`obj_tags`，所以 `filter_stage/3dbox_filter` 的输出可以直接接到 `scene_fusion_stage/depth_back_projection`。

raw 分支不会产生 `bboxes_2d`，但这不影响 scene fusion。本实验在汇总和可视化阶段统一从 mask 动态计算 2D bbox，避免为了实验侵入式修改 `filter` 或 `scene_fusion`。如果后续要把 raw 分支直接接入严格依赖 `bboxes_2d` 的 annotation/group 流程，再新增一个独立适配器阶段更合适。

## 汇总指标

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
- `object_metrics.csv`: 对齐到对象粒度的 mask、bbox、pointcloud 指标。

评估分四层：

- 样本级：filter/refine/fusion 后的样本数。
- 物体级：filter/refine/fusion 后的物体数、refiner 保留率、fusion 保留率。
- Mask 质量：面积、连通域数量、最大连通域占比、mask/bbox fill ratio、相对 raw coarse mask 的 IoU 和面积变化率。
- BBox/3D 质量：mask-derived 2D bbox 面积、中心偏移、与 3D box 投影 bbox 的 IoU/coverage、点云点数、点云落在原 3D bbox 内的比例。

## 可视化

生成 Raw/SAM2/SAM3 并排 overlay 和 RGB-D mask 点云视图：

```bash
python refiner_exp/scripts/visualize_compare.py \
  --raw-run refiner_exp/outputs/raw \
  --sam2-run refiner_exp/outputs/sam2 \
  --sam3-run refiner_exp/outputs/sam3 \
  --output-dir refiner_exp/outputs/compare/images \
  --max-images 20 \
  --pointcloud-mode both
```

每张图会展示三条分支的 mask overlay、mask-derived 2D bbox，以及每个 object 的原始 `bboxes_3d_world_coords` 投影到 RGB 后的 3D bbox 线框。mask、2D bbox 和 3D bbox 线框使用同一个 object 颜色，便于直观看出 3D bbox 是否贴合当前视角中的物体。默认还会用当前分支的 mask 从 RGB + depth 反投影出彩色 object point cloud，并追加 front/top 两个正交视图缩略图。

点云相关输出：

- `--pointcloud-mode none`: 只生成 2D overlay。
- `--pointcloud-mode render`: 只在 JPG 中追加点云正交视图。
- `--pointcloud-mode ply`: 只导出可交互查看的 `.ply`。
- `--pointcloud-mode both`: 同时生成缩略图和 `.ply`，默认值。

导出的 `.ply` 位于 `refiner_exp/outputs/compare/images/pointclouds/`，可以用 CloudCompare、MeshLab、Open3D viewer 等常见点云工具打开。旁边会输出同名 `.json`，记录该图内对象的自动指标和每个分支导出的点云路径，便于把人工判断和数值指标对应起来。

## 推荐解读

好的 refine 应该让 mask 更贴合可见物体，同时不让点云数量异常下降，也不让 bbox 中心大幅偏离原 3D box 投影。可疑结果通常表现为 mask 面积剧烈膨胀或收缩、最大连通域占比偏低、bbox 明显漂移到背景、fusion 后物体大量丢失。

SAM3 需要特别关注 coverage/precision 门控带来的误删；SAM2 需要特别关注 score 通过但 mask 扩到邻近物体或背景的情况。
