# Mark 使用备忘（annotation 管线）

> 仅 **point / box**；mask 已禁用。  
> 默认 `emit_marked_images=False`：`mark_spec` 写入 metadata，`QA_images` 为原图 bytes。

## 1. 基础设施

| 组件 | 路径 | 作用 |
|------|------|------|
| `MarkConfig` / `VisualMarker` | `task/annotation/core/visual_marker.py` | point/box 选择与颜色 |
| `plan_mark` / `plan_object_marks` | `task/annotation/core/mark_spec.py` | 构建 `mark_spec` v2 |
| `plan_mark_for_qa` | `task/annotation/core/base_annotation_task.py` | 按策略规划 mark；默认不烙图 |
| `mark_objects_for_qa` | 同上 | 薄封装，等同 `plan_mark_for_qa(objs=…)` |
| `resolve_mark_enabled` | 同上 | 歧义 tag 必开；否则 `OPTIONAL_MARK_ENABLE_PROB`（25%） |
| `SceneGraph` | `task/annotation/core/scene_graph.py` | `count_tag_in_view` / `requires_mark_for_nodes` |

### 启用策略

- 输入：`SceneNode` 列表和/或 `tag` + `view_idx`（`apply_transform` 会设置 `thread_local.scene_graph`）
- 该 view 内任一相关 `tag` 计数 **> 1** → 规划 `mark_spec`
- 否则以 **25%** 概率规划，**75%** 不写 `mark_spec`（`last_mark_spec = None`）
- **点 mark**（`points=`，如 correspondence / depth 随机像素）不经对象歧义分支，始终规划

---

## 2. 会打 mark 的 task

### Singleview — `mark_objects_for_qa` / `plan_mark_for_qa`

| Task | 文件 | MarkConfig | 备注 |
|------|------|------------|------|
| distance | `distance.py` | point 0.25 / box 0.75 | 2–3 物体 |
| position | `position.py` | box, point | 1–2 物体 |
| size | `size.py` | box, point（shuffle 色） | absolute 批量；relative 2 物体 |
| depth_annotation | `depth_annotation.py` | point/box 各 50% | `_mark_and_sort`；另有 `plan_point_marks` 随机像素 |

### Singleview — 无 object mark

| Task | 文件 |
|------|------|
| 3d_grounding | `3d_grounding.py` |
| 3d_scene_caption | `3d_scene_caption.py` |
| counting | `counting.py` |

### Multiview — `_mark_per_view` / `_find_chain_and_mark`

| Task | 文件 |
|------|------|
| multiview_distance | `multiview_distance.py` |
| multiview_size | `multiview_size.py` |
| multiview_distance_obj_cam | `multiview_distance_obj_cam.py` |
| multiview_object_position | `multiview_object_position.py`（`plan_mark_for_qa` + merge） |

### Multiview — 点 mark（任务必需）

| Task | 文件 |
|------|------|
| multiview_correspondence | `multiview_correspondence.py` |

---

## 3. Scene graph

- Singleview：`SceneGraph.from_singleview_example`
- Multiview：`SceneGraph.from_multiview_example`
- 歧义判定统计**该 view 全部可见节点**的同 tag 数量，不限于当前采样子集

---

## 4. 改造检查清单

- [ ] 新 task 走 `plan_mark_for_qa` / `mark_objects_for_qa`，勿直接 `marker.mark_objects()`（除 `emit_marked_images` 路径）
- [ ] `get_mark_config()` 仅 point/box
- [ ] 勿假设 `QA_images` 已带 mark；训练/可视化用 `metadata.mark_spec`
- [ ] 聚合/导出只合并 `mark_spec`，不改变原图落盘契约

---

## 5. 验证

- `verification/dataset_pipeline/check_annotation_mark_paths.py`
- `visualize_server.py` / `visualize_upstream_server.py` 客户端 overlay
