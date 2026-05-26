# 数据集管线 — 路标状态

> 对照 `assets/pipeline_dataset_optimization_plan.md` §5.2  
> **`[x]`** 已验收 · **`[pilot]`** 仅模块/单测 · **`[ ]`** 未完成

## 发布分级

| 级别 | 含义 | M2/M3 | M4/M5 |
|------|------|-------|-------|
| **L1** | 10 task 接入 metadata 管线（不含已废弃 counting） | 代码 + 单测 | 代码 + 单测 |
| **L2** | 单目 frame_rot 跑批 + audit + aggregate | audit 绿 | aggregate 真实数据 |
| **L3** | 多目 group parquet + multiview 跑批 | 同 L2 | 同 L2 |

## 路标总表

| 路标 | L1 | L2 | L3 |
|------|:--:|:--:|:--:|
| M0 | [x] | [x] | [x] |
| M1 | [x] | [x] | [x] |
| M2 | [x] | [x] | — |
| M3 | [x] | [x] | — |
| M4 | [x] | [x] | — |
| M5 | [x] | [x] | [x] |
| M6 | [x] | [x] | [x] |
| M7 | [x] | [x] | [x] |
| M8 | [x] | [x] | [ ] |

## M0 — 3D grounding 相机与 bbox 格式

- [x] L1/L2/L3

## M1 — metadata（annotation parquet）

- [x] L1：`sample_metadata` / `BaseAnnotationTask` / `audit_annotation_outputs.py`
- [x] L2：frame_rot 五 task `metadata` 100%；audit 抽检通过
- [x] L3：multiview group parquet + audit（`multiview_distance_obj_cam` pilot 可 0 条 → skip）

## M2 — mark_spec

- [x] L1：`last_mark_spec` / `mask_ref.path`
- [x] L2：frame_rot 重跑 + audit + `visualize_server` 叠 mask

## M3 — 原图 QA_images

- [x] L1/L2：`emit_marked_images: false`；frame_rot 五 task 有 `metadata.turns`

## M4 / M5 — aggregate

- [x] L1：`turn_io` warning；aggregate yaml
- [x] L2：singleview merged（1901 samples / 2466 turns）
- [x] L3：multiview merged（34 samples / 34 turns）

## M6 — 占位符与答案 provenance

- [x] L1：`message_placeholders`；`apply_transform` → `sync_messages_with_turns`
- [x] L2：`check_annotation_mark_paths.py`；未标注路径修复
- [x] L3：aggregate 每轮保留 `<image>`
- [x] `_record_turn` 从 prompt 解析 `answer_text`；`audit --check-answers`

## M7 — 上游 export（JSONL + tar）

- [x] L1：`upstream_export` / `jsonl_base` / `composed` / export yaml
- [x] L2：singleview + multiview bundle；`test_upstream_export`；`verify_milestone M7`
- [x] L3：multiview export 与 audit 一致

## pre-M8 门禁

| 门禁 | 命令 |
|------|------|
| M6 + 源码 | `verify_milestone.py --pre-m8-gates` |
| 答案对账 | `audit_annotation_outputs.py --check-answers` |
| M7 bundle | `verify_milestone.py --milestone M7 --m7-check-l2-bundle` |

重跑 annotation 后须再跑 aggregate → export，并复跑上表。

## M8 — structured prompt + 指令约束

**范围**：`QuestionType`（OE / MCQ / Judgment）+ `introduction` / `stem` / `question_instruction` / `instruction_type` 答案池。L1/L2 仅结构迁入，不改各 task 文案语义。

### L1

- [x] `structured_prompt_template` + `render_structured_prompt`
- [x] `register_structured.py`；`test_prompt_template_structure.py`；`verify_milestone M8`（L1）
- [x] 占位符体系文档 [`STRUCTURED_PLACEHOLDERS.md`](STRUCTURED_PLACEHOLDERS.md) + 正向审计 [`../scripts/audit_structured_placeholder_bindings.py`](../scripts/audit_structured_placeholder_bindings.py)

### L2

- [x] 全 task 迁入 `StructuredTemplateRegistry`
- [x] m8l2 重跑 + `audit --check-answers` PASS + 人工抽查无阻塞（2026-05-22）
- [ ] 可选：aggregate → export → M7 bundle
- [ ] 可选：`compare_annotation_baseline.py`

### L3

- [ ] `instruction_type` 跑批可统计
- [ ] `question_type` × `instruction_type` 答案长度报告
- [ ] `verify_milestone M8`（L3）

### Per-task 内容迭代

> 验收数据：`output/frame_rot/base_pipeline_demo_*_frame_rot_m8l2/`  
> 单目/多目共用模板一并改；仅 `multiview_distance.{farthest,closest,obj_cam*}` 等多目专属 ID 在多目项讨论。  
> 已迁 structured 的 task 删除 flat `TemplateRegistry` 重复注册。

| Task | 状态 |
|------|------|
| distance | 完成 |
| depth | 完成 |
| size | 完成 |
| position | 完成 |
| counting | 已废弃（仅曾有单目；无多目实现） |
| 3d_grounding | 完成 |
| correspondence | 待检视（OE/MCQ 分模板；`[E]`=A–D 已修；review 重跑） |
| multiview_distance | 待检视 |
| multiview_distance_obj_cam | 待检视（模板/introduction 与 distance 同批完成；独立 handler） |
| multiview_size | 待检视（pair 与单目 D_diag+1.2 对齐；N-ary 仅 top vs 2nd 门控） |
| multiview_object_position | 待检视 |
| 3d_scene_caption | 不处理 |

## 命令速查

```powershell
# 跑批
python run.py --config config/annotation/demo_singleview_all_frame_rot.yaml --output_dir E:/GitRepo/OpenSpatial/output/frame_rot
python run.py --config config/annotation/demo_multiview_all_frame_rot.yaml --output_dir E:/GitRepo/OpenSpatial/output/frame_rot
python run.py --config config/annotation/demo_multiview_all_frame_rot_review.yaml --output_dir E:/GitRepo/OpenSpatial/output/frame_rot

# 门禁
python verification/dataset_pipeline/verify_milestone.py --pre-m8-gates
python verification/dataset_pipeline/audit_annotation_outputs.py --root output/frame_rot/base_pipeline_demo_singleview_all_frame_rot_m8l2 --no-check-mask-files --check-answers
python verification/dataset_pipeline/audit_annotation_outputs.py --root output/frame_rot/base_pipeline_demo_multiview_all_frame_rot_m8l2 --no-check-mask-files --check-answers
pytest tests/test_prompt_template_structure.py
python verification/dataset_pipeline/verify_milestone.py --milestone M8

# aggregate / export
python run.py --config config/aggregate/demo_aggregate_singleview_frame_rot.yaml --output_dir E:/GitRepo/OpenSpatial/output/frame_rot
python run.py --config config/export/demo_export_frame_rot.yaml --output_dir E:/GitRepo/OpenSpatial/output/frame_rot

# 可视化（m8l2 / 单目人工检视 review）
python visualize_server.py --data_dir output/frame_rot/base_pipeline_demo_singleview_all_frame_rot_m8l2 --port 8888
python visualize_server.py --data_dir output/frame_rot/base_pipeline_demo_singleview_all_frame_rot_review --port 8890
python visualize_server.py --data_dir output/frame_rot/base_pipeline_demo_multiview_all_frame_rot_m8l2 --port 8889
python visualize_server.py --data_dir output/frame_rot/base_pipeline_demo_multiview_all_frame_rot_review --port 8891
```
