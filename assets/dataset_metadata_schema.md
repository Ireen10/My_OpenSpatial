# 数据落盘 metadata Schema（v1.1）

> 对照 `pipeline_dataset_optimization_plan.md` §4.1、§4.2、§4.7。  
> 校验实现：`verification/dataset_pipeline/validate_metadata.py`

## 1. JSONL 行（sample 级）

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `schema_version` | string | 是 | 当前 `"1.1"` |
| `sample_id` | string | 是 | 全局唯一 |
| `merge_group_key` | string | 是 | 预期渲染图像组合并组（§4.2.2；同/跨 task） |
| `image_refs` | string[] | 是 | tar 内原图 key，长度 = 逻辑视图数 |
| `messages` | object[] | 是 | 合并后多轮；`from` ∈ `human`,`gpt` |
| `metadata` | object | 是 | 见 §2 |

## 2. `metadata` 对象

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `visual_anchor` | object | 是 | §2.1 |
| `mark_spec` | object \| null | 条件 | 有视觉 mark 时必填；§2.2 |
| `turns` | array | 是 | 每轮 QA 溯源；§2.3 |
| `provenance` | object | 否 | `source_tasks`, `merged`, `pipeline_run_id` |

### 2.1 `visual_anchor`

| 字段 | 类型 | 必填 |
|------|------|------|
| `parent_preprocess_id` | string | 是 |
| `scene_id` | string | 推荐 |
| `frame_id` | string | 推荐 |
| `raw_image_ref` | string | 单视图 |
| `raw_image_refs` | string[] | 多视图（与 `image_refs` 对齐） |
| `view_group_id` | string | 多视图合并组 |

### 2.2 `mark_spec`（version 2，**per_view 布局**）

| 字段 | 类型 | 必填 |
|------|------|------|
| `version` | int | 是，固定 `2` |
| `layout` | string | 多视图为 `per_view` |
| `views` | array | 是，**每张 QA 图一条** |
| `render_hints` | object | 否 |

每个 `views[]` 元素（**仅描述该视角图像上的 mark**）：

| 字段 | 类型 | 必填 |
|------|------|------|
| `view_index` | int | 是，与 `QA_images` / `<image>` 占位顺序一致（0 起） |
| `image_ref` | string | 推荐，对应 `visual_anchor.raw_image_refs[view_index]` |
| `mark_kinds` | array | 是 |
| `slots` | array | 是，**只属于本视角** |

每个 `views[].slots[]` 元素：

| 字段 | 类型 | 必填 |
|------|------|------|
| `slot_id` | string | 是，如 `A`,`B` |
| `obj_idx` | int | 是 |
| `tag` | string | 是 |
| `mark_kind` | string | 是，`box`\|`mask`\|`point` |
| `color_name` | string | 是 |
| `geometry` | object | 是 |
| `label_alias` | string | 否 |

单目： `views` 长度为 1。旧版扁平 `slots[]`（无 `views`）在读取时会按 `slot.view_index` 迁移，**新写入一律用 `views[]`**。

`geometry` 按 `mark_kind`：

- `box`: `box_2d` [x1,y1,x2,y2]
- `mask`: `mask_ref` object  
  - **annotation 阶段（推荐）**：`{"source":"path","path":"<pipeline mask 文件路径>","obj_idx":N}` — 自包含叠绘，不依赖 parquet 是否带 `masks` 列  
  - **回退 / 落盘 tar**：`source` ∈ `preprocess`（`obj_idx`）\|`tar`（`tar_key`）
- `point`: `uv` [u,v]

### 2.3 `turns[]` 元素

| 字段 | 类型 | 必填 |
|------|------|------|
| `turn_id` | int | 是 |
| `task_name` | string | 是 |
| `sub_task` | string | 是 |
| `question_type` | string | 是 |
| `instruction_mode` | string | 是 |
| `dedup_fingerprint` | string | M4+ |
| `prompt_struct` | object | M3+ |
| `referent_mode` | string | 默认 `semantic` |
| `question_text` | string | 是 |
| `answer_text` | string | 是 |
| `question_prefix` | string | 否（3D grounding） |
| `image_placeholder_count` | int | 是 |

### 2.4 `prompt_struct`（canonical QA）

| 字段 | 类型 | 必填 |
|------|------|------|
| `template_id` | string | 是 |
| `template_family` | string | 推荐 |
| `question_pattern` | string | 是，含 `{{slot}}` |
| `answer_pattern` | string | 否 |
| `slots` | object | 是，键为 slot_id，值含 `obj_idx`,`tag` |
| `answer_slots` | object | 否 |

## 3. 派生键（聚合阶段写入）

| 键 | 用途 |
|----|------|
| `dedup_fingerprint` | task 内去重（§4.2.1） |
| `question_core_key` | 组成 fingerprint 的一部分 |
| `merge_group_key` | 按渲染图像组合并（§4.2.2） |

## 4. 版本变更

| 版本 | 变更 |
|------|------|
| 1.1 | 初版：sample 行 + mark_spec v2 + prompt_struct |
| 1.1.1 | mask：`mask_ref.source=path` + 管线内 mask 绝对路径（`obj_idx` 保留溯源） |
