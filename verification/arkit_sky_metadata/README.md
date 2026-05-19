# ARKitScenes `sky_direction` vs traj 验证（可整目录删除）

独立验证区，与主预处理管线无关。删除本目录 `verification/arkit_sky_metadata/` 即可清理全部下载与结果。

## 目录

| 路径 | 说明 |
|------|------|
| `data/metadata.csv` | 官方 metadata（含 `sky_direction`） |
| `data/raw_train_val_splits.csv` | 全量 video_id / fold |
| `data/traj/<Training\|Validation>/<video_id>/lowres_wide.traj` | 仅 traj |
| `results/` | 对比报告、四路天空可视化 |

## 1. 下载全量 traj

```powershell
cd E:\GitRepo\OpenSpatial
.venv311\Scripts\python.exe verification\arkit_sky_metadata\download_all_traj.py --workers 32
```

可选：`--split Training`、`--limit 100`（试跑）、`--retry-failed`

部分 video 无 traj（仅 upsampling、无 3DOD），下载会 404，对比脚本会自动跳过。

## 2. 多方法与 metadata 对比

```powershell
.venv311\Scripts\python.exe verification\arkit_sky_metadata\compare_sky_methods.py
```

输出：

- `results/per_video.csv`：每场景 metadata + 各方法预测
- `results/summary.json`：各方法与 metadata 一致率
- `results/mismatch_<method>.csv`：不一致样本

## 对比方法（见 `compare_sky_methods.py` 内 `METHODS`）

- **global_stat_z_up**：全场 pose 平均 + 世界 **+Z** 为竖直（CUT3R / OpenSpatial 默认）
- **global_stat_y_up**：同上，世界 **+Y** 为竖直
- **first_frame_stat_z_up**：仅 traj **第一行** + Z-up 统计逻辑
- **first_frame_decide_pose**：仅第一行 + 官方 `rectify_im.decide_pose`
- **majority_decide_pose** / **majority_stat_z_up**：逐行 traj 投票众数
- **first_10pct_stat_z_up**：前 10% 时间戳内 pose 平均（近似「开场」）

## 全量跑分结果（5043 条有 traj + 有效 metadata）

| 方法 | 与 metadata 一致率 |
|------|-------------------|
| **global_stat_z_up**（全场统计 +Z） | **83.09%**（最高） |
| majority_decide_pose / majority_stat_z_up | 83.05% |
| median_frame_decide_pose | 82.79% |
| first_10pct_stat_z_up | 81.34% |
| first_frame_stat_z_up / first_frame_decide_pose | **61.47%**（二者完全一致） |
| last_frame_stat_z_up / last_frame_decide_pose | **60.22%**（二者完全一致） |
| global_stat_y_up | 22.37% |
| global_stat_neg_y_up | 8.78% |

**结论（本数据集上）**：

- metadata **不是**「仅第一帧」：第一帧约 **61%**，全场统计约 **83%**，更接近 **全局 traj 统计**（或与之等价的逐帧众数）。
- **没有任何方法 100% 对齐 metadata**（约 17% 不一致），与社区反馈的标签错误、以及中途转屏等因素一致。
- 详细：`results/summary.json`、`results/per_video.csv`、`results/mismatch_<method>.csv`

## 3. 四路天空校正可视化（仅验证，不写主线 `preprocess_output`）

对本地已下载 raw 场景生成四列对比图：**Raw vga_wide** | **metadata.csv（场景级）** | **traj 全场统计** | **traj 逐帧**。

```powershell
cd E:\GitRepo\OpenSpatial
.venv311\Scripts\python.exe verification\arkit_sky_metadata\visualize_sky_comparison.py `
  --data-root .\data_root\EmbodiedScan\data `
  --raw-root .\data_root\EmbodiedScan\data\arkitscenes_highres `
  --only-local-raw `
  --skip-all-up
```

- 输出：`verification/arkit_sky_metadata/results/sky_4way_viz/index.html` 及按场景分目录的 jpg
- `--skip-all-up`：跳过 metadata / global / per-frame **三者标签均为 UP** 的帧（与 raw 无可见差异）
- 可选：`--scene arkitscenes/Training/40753679`（可重复）、`--max-scenes N`
