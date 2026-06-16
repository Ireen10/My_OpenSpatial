const STAGE_LABELS = {
  upstream: "上游 / 官方",
  convert: "转换",
  output: "导出",
  validate: "校验"
};

const PIPELINES = {
  embodiedscan: {
    id: "embodiedscan",
    tabLabel: "EmbodiedScan",
    title: "EmbodiedScan 预处理流程",
    lead:
      "将 ScanNet / 3RScan / Matterport3D / ARKitScenes 转为 OpenSpatial Parquet。CLI：extract → merge → export → validate；ARKit 另需 prepare-arkit。",
    svg: { width: 1680, height: 580, viewBox: "0 0 1680 580" },
    stageBands: [
      { id: "upstream", x: 24, y: 28, w: 1632, h: 175 },
      { id: "convert", x: 24, y: 228, w: 1632, h: 175 },
      { id: "output", x: 24, y: 430, w: 1380, h: 130 }
    ],
    layout: {
      "raw-data": { x: 44, y: 78, w: 260, h: 118 },
      "upstream-images": { x: 330, y: 78, w: 280, h: 118 },
      "arkit-prepare": { x: 636, y: 78, w: 280, h: 118 },
      extract: { x: 44, y: 278, w: 240, h: 118 },
      merge: { x: 310, y: 278, w: 240, h: 118 },
      export: { x: 576, y: 278, w: 240, h: 118 },
      validate: { x: 842, y: 278, w: 240, h: 118 },
      parquet: { x: 1108, y: 278, w: 260, h: 118 },
      "downstream-note": { x: 1108, y: 458, w: 300, h: 88 }
    },
    edges: [
      ["raw-data", "upstream-images", "ScanNet/3RScan"],
      ["raw-data", "arkit-prepare", "ARKit only"],
      ["upstream-images", "extract"],
      ["arkit-prepare", "extract"],
      ["extract", "merge"],
      ["merge", "export"],
      ["export", "validate"],
      ["validate", "parquet"],
      ["parquet", "downstream-note"]
    ],
    steps: [
      {
        id: "raw-data",
        title: "原始数据与标注",
        stage: "upstream",
        summary: "官方 pkl + 各数据集 raw 目录。",
        files: ["data_preprocessing/embodiedscan/README.md", "EmbodiedScan/data/README.md"],
        inputs: [
          "CLI：--data-root = .../EmbodiedScan/data（不是 EmbodiedScan 根目录本身）",
          "pkl 实际路径：dirname(data-root)/data/embodiedscan_infos_{train,val,test}.pkl（v1，与 data-root 同级 data/ 下）",
          "ARKit v2 pkl：dirname(data-root)/embodiedscan-v2/embodiedscan_infos_*.pkl",
          "ScanNet 每帧 pkl 字段：images[].img_path, cam2global, visible_instance_ids[]；场景级 instances[].bbox_3d, axis_align_matrix",
          "磁盘：scannet/posed_images/<scene>/NNNNN.jpg|.png；scans/<scene>/intrinsic/intrinsic_depth.txt"
        ],
        outputs: [
          "无新文件；extract 只读上述路径 + pkl"
        ],
        flow: [
          "pip install -e EmbodiedScan[visual] && pip install -e data_preprocessing/embodiedscan",
          "Explorer 用 project_root=dirname(abs(data-root)) 拼 raw 路径；JSONL 里路径最终相对 data-root",
          "list_cameras() 扫磁盘全部 .jpg；get_info() 只在 pkl['images'] 里按 cam_name 匹配",
          "pkl 有图但无 visible_instance_ids → Rendering box failed → 该帧不进 jsonl（ScanNet 最常见失败原因）"
        ],
        params: [
          "depth_scale：ScanNet/3RScan/ARKit=1000；Matterport3D=4000",
          "不要为 data-root 再建 Windows junction 指向同一棵树（README 明确禁止）"
        ]
      },
      {
        id: "upstream-images",
        title: "上游图像解包",
        stage: "upstream",
        summary: "官方脚本从 .sens 生成 posed_images。",
        files: [
          "EmbodiedScan/embodiedscan/converter/generate_image_scannet.py",
          "EmbodiedScan/embodiedscan/converter/generate_image_3rscan.py"
        ],
        inputs: [
          "ScanNet：scannet/scans/<scene_id>/<scene_id>.sens（在 scannet 目录下 chdir 后跑脚本）",
          "3RScan：各 scene 的 sequence 原始包"
        ],
        outputs: [
          "scannet/posed_images/<scene_id>/NNNNN.jpg + NNNNN.png（与 sens 帧号对齐，5 位补零）",
          "同目录 NNNNN.txt 位姿、intrinsic.txt / depth_intrinsic.txt（extract 主要用 scans/.../intrinsic_depth.txt）",
          "3rscan/<uuid>/sequence/frame-XXXXXX.color.jpg + .depth.pgm"
        ],
        flow: [
          "cd <EmbodiedScan>/data/scannet && python embodiedscan/converter/generate_image_scannet.py --dataset_folder . --fast --nproc 8",
          "--fast：仅解 sens 帧索引 0,10,20,…（与 pkl 采样一致）；无 --fast 会解全帧 → 磁盘暴涨且多数帧不在 pkl",
          "extract 不复制 jpg；ScanNet post_process 仅在 img.size≠depth.size 时写 <stem>_resized.jpg"
        ],
        params: [
          "Matterport3D / ARKit 无此步（原图已在官方目录布局中）",
          "posed_images 与 pkl 帧名必须一致（如 00000.jpg），否则 get_info 返回 None"
        ]
      },
      {
        id: "arkit-prepare",
        title: "prepare-arkit（可选）",
        stage: "upstream",
        summary: "ARKit 专用：vga_wide + 天空校正 depth。",
        files: ["data_preprocessing/embodiedscan/embodiedscan_data/prepare_arkitscenes.py"],
        inputs: [
          "--raw-root 默认 <data-root>/arkitscenes_highres，读 raw/{Training,Validation}/<scene_id>/",
          "需要资产：vga_wide/*.png、vga_wide_intrinsics/*.pincam、lowres_depth、lowres_wide.traj",
          "标注：embodiedscan-v2 pkls（bbox_3d 为世界系，prepare 不改 box）"
        ],
        outputs: [
          "<data-root>/arkitscenes/{Training,Validation}/<scene_id>/<scene_id>_frames/vga_wide/*.jpg",
          "同目录 vga_depth/*.png、vga_wide_intrinsics/*_pose.txt / *_intrinsic.txt",
          ".arkit_scene.json：frames[<cam>].sky_direction, rotated_to_cam, traj_ts"
        ],
        flow: [
          "python -m embodiedscan_data prepare-arkit --data-root ... --raw-root ... --only-local-raw --workers 8",
          "默认 --sky-granularity frame：每帧用 lowres_wide.traj 估天空方向，旋转 depth 对齐 vga 朝向（INTER_NEAREST）",
          "extract --arkit-asset-mode auto：有 .arkit_scene.json 则用 vga_*，否则回退 lowres_wide/lowres_depth",
          "extract 读 v2 pkl 后对 pose 应用 rotated_to_cam；obj_tags / bboxes_3d_world_coords 仍来自 pkl"
        ],
        params: [
          "会新写 RGB+depth（主要磁盘增量）；--arkit-asset-mode lowres 可跳过 vga 产物",
          "仅 ARKit 需要；ScanNet/3RScan/MP3D 不走此节点"
        ]
      },
      {
        id: "extract",
        title: "extract（逐图）",
        stage: "convert",
        summary: "多进程读 pkl，写 JSONL。",
        files: [
          "data_preprocessing/embodiedscan/embodiedscan_data/extract.py",
          "data_preprocessing/embodiedscan/embodiedscan_data/explorer.py"
        ],
        inputs: [
          "任务：(scene, camera) 对；scene 形如 scannet/scene0000_01，camera 为帧 stem（如 00000）",
          "render_box=True 时从 pkl instances[visible_instance_ids[i]] 取 bbox_3d + 语义类"
        ],
        outputs: [
          "output/<dataset>.jsonl，每行一条 per-image record",
          "字段：id, dataset, scene_id, image, depth_map, pose, intrinsic, axis_align_matrix, depth_scale, obj_tags[], bboxes_3d_world_coords[]（9-DoF）",
          "路径均相对 --data-root（extract 末尾 _relpath_under_data_root）",
          "副作用 txt：ScanNet 写在 posed_images/<scene>/<cam>_pose.txt；3RScan 写在图像旁 *_pose.txt；3RScan 另生成 *_depth_map.npy"
        ],
        flow: [
          "python -m embodiedscan_data extract --dataset scannet --data-root .../data --output ./out --workers 24",
          "pose = axis_align_matrix @ cam2global；intrinsic 优先 images[].cam2img 否则场景 cam2img",
          "断点续跑：读已有 jsonl 的 id 集合，重复 id 不追加",
          "失败计数：get_info 返回 None 或 visible_instance_ids 异常 → Failed+1，无 stack trace 中止"
        ],
        params: [
          "ScanNet intrinsic 路径固定：scannet/scans/<scene>/intrinsic/intrinsic_depth.txt（每场景一个，非逐帧）",
          "3RScan depth：pgm → npy 与 jpg 同目录，体积≈原 depth",
          "Matterport depth：matterport_depth_images/*dNNN.png，不经过 explorer 转换"
        ]
      },
      {
        id: "merge",
        title: "merge（逐场景）",
        stage: "convert",
        summary: "JSONL 按 scene_id 聚合成 list 行。",
        files: ["data_preprocessing/embodiedscan/embodiedscan_data/merge.py"],
        inputs: ["output/<dataset>.jsonl（逐图，标量列）"],
        outputs: [
          "output/<dataset>_scenes.jsonl",
          "标量保留：dataset, scene_id, num_images",
          "其余字段变 list：image[i], pose[i], …；obj_tags[i] 为第 i 帧的物体标签列表（list of lists）"
        ],
        flow: [
          "python -m embodiedscan_data merge --input ./out",
          "同一 scene_id 多行按 jsonl 出现顺序 append",
          "与 ScanNet++ 不同：EmbodiedScan per_scene 的 obj_tags 是「每帧一个 list」，不是全场景共享单 list"
        ],
        params: [
          "export --format both 时才需要 merge；只要 per_image parquet 可跳过",
          "OpenSpatial multiview 用 per_scene/ + flatten_stage"
        ]
      },
      {
        id: "export",
        title: "export（Parquet）",
        stage: "output",
        summary: "JSONL → 分片 Parquet。",
        files: ["data_preprocessing/embodiedscan/embodiedscan_data/export.py"],
        inputs: ["output/*.jsonl", "output/*_scenes.jsonl"],
        outputs: [
          "output/per_image/data-00000.parquet（默认 batch_size=3000 条/片）",
          "output/per_scene/data-00000.parquet",
          "列类型与 JSONL 一致；嵌套 list 原样进 parquet（无图像 bytes）"
        ],
        flow: [
          "python -m embodiedscan_data export --input ./out --format both",
          "--format per_image | per_scene | both",
          "可选 --hf-repo 上传 HuggingFace（需 huggingface_hub）"
        ],
        params: [
          "路径仍为相对 data-root 字符串；run.py 需配 dataset.raw_data_root=<data-root>"
        ]
      },
      {
        id: "validate",
        title: "validate",
        stage: "validate",
        summary: "schema + 计数 + 随机路径抽检。",
        files: ["data_preprocessing/embodiedscan/embodiedscan_data/validate.py"],
        inputs: ["--input ./out", "--data-root ...（拼路径做存在性检查）"],
        outputs: ["stdout 错误列表；无写盘"],
        flow: [
          "python -m embodiedscan_data validate --input ./out --data-root .../data",
          "REQUIRED_FIELDS：id, dataset, scene_id, image, pose, depth_map, intrinsic, depth_scale, bboxes_3d_world_coords, obj_tags, axis_align_matrix",
          "validate_counts：sum(per_scene.num_images) == len(per_image lines)",
          "validate_paths：随机抽 50 条，join(data-root, image|depth_map|pose|intrinsic) 必须 isfile",
          "depth_scale 只允许 1000 或 4000；bboxes 每框 len==9"
        ],
        params: [
          "validate 读 jsonl 不读 parquet；export 后应用 jsonl 目录跑",
          "axis_align_matrix 字段必填但可为指向 txt 的路径字符串"
        ]
      },
      {
        id: "parquet",
        title: "OpenSpatial Parquet",
        stage: "output",
        summary: "run.py 预处理入口。",
        files: [
          "config/preprocessing/demo_preprocessing_embodiedscan.yaml",
          "assets/preprocessing_input_schema.md"
        ],
        inputs: [
          "per_image（推荐）：image/depth_map/pose/intrinsic/obj_tags 均为标量 str/list",
          "per_scene：上述相机列为 list[str]，需先 flatten 才能进 filter（demo 默认不用）"
        ],
        outputs: [
          "作为 dataset.data_dir；配合 raw_data_root=<data-root>"
        ],
        flow: [
          "demo_preprocessing_embodiedscan.yaml 直接用 per_image/ 分片，无 flatten_stage",
          "filter → sam2 → fusion 后 group_stage 按 scene_id 聚回 multiview 行",
          "若用 per_scene parquet：须配置 flatten，且 obj_tags 等为「每帧一项」的 list，flatten 时非 split 列整列复制"
        ],
        params: [
          "group_stage 需要 scene_id 列；merge 后 per_scene 行自带",
          "Hypersim/ScanNet++ 的 axis_align_matrix 常为 null；EmbodiedScan 有 txt 路径"
        ]
      },
      {
        id: "downstream-note",
        title: "→ OpenSpatial Pipeline",
        stage: "output",
        summary: "demo_preprocessing_embodiedscan.yaml",
        files: ["run.py", "config/preprocessing/demo_preprocessing_embodiedscan.yaml"],
        inputs: ["per_scene parquet + raw_data_root"],
        outputs: [
          "{output_dir}/flatten_stage/flatten/data.parquet → … → group_stage/group/data.parquet",
          "副产物：filter_stage/3dbox_filter/masks/*.png，scene_fusion_stage/.../pointclouds/*.pcd"
        ],
        flow: [
          "python run.py --config config/preprocessing/demo_preprocessing_embodiedscan.yaml",
          "链：filter → sam2_refiner → depth_back_projection → group（无 flatten）",
          "group 输出供 demo_multiview_* annotation 使用"
        ],
        params: [
          "详见 assets/pipeline/index.html 各 stage 阈值"
        ]
      }
    ]
  },

  hypersim: {
    id: "hypersim",
    tabLabel: "Hypersim",
    title: "Hypersim 预处理流程",
    lead:
      "prepare_hypersim.py 五步串联；Step 3/4 在 input_root 内写派生 JPG/PNG，--output_dir 只收 Parquet。",
    svg: { width: 1760, height: 500, viewBox: "0 0 1760 500" },
    stageBands: [
      { id: "upstream", x: 24, y: 28, w: 1712, h: 140 },
      { id: "convert", x: 24, y: 190, w: 1712, h: 175 },
      { id: "output", x: 24, y: 388, w: 560, h: 90 }
    ],
    layout: {
      "raw-hdf5": { x: 44, y: 62, w: 280, h: 108 },
      "step-intrinsics": { x: 44, y: 232, w: 230, h: 118 },
      "step-extrinsics": { x: 294, y: 232, w: 230, h: 118 },
      "step-tonemap": { x: 544, y: 232, w: 230, h: 118 },
      "step-depth": { x: 794, y: 232, w: 230, h: 118 },
      "step-parquet": { x: 1044, y: 232, w: 230, h: 118 },
      parquet: { x: 1294, y: 232, w: 250, h: 118 },
      "disk-warning": { x: 360, y: 62, w: 420, h: 108 }
    },
    edges: [
      ["raw-hdf5", "step-intrinsics"],
      ["step-intrinsics", "step-extrinsics"],
      ["step-extrinsics", "step-tonemap"],
      ["step-tonemap", "step-depth"],
      ["step-depth", "step-parquet"],
      ["step-parquet", "parquet"]
    ],
    steps: [
      {
        id: "raw-hdf5",
        title: "Hypersim Raw",
        stage: "upstream",
        summary: "HDF5 渲染 + mesh 标注。",
        files: ["data_preprocessing/hypersim/prepare_hypersim.py"],
        inputs: [
          "--input_root/<scene_id>/images/scene_<cam>_final_hdf5/frame.<i>.color.hdf5",
          "…/scene_<cam>_geometry_hdf5/frame.<i>.depth_meters.hdf5（欧氏距离，非平面深度）",
          "…/geometry_hdf5/frame.<i>.semantic_instance.hdf5 + semantic.hdf5",
          "…/_detail/mesh/metadata_objects.csv + mesh_objects_sii.hdf5 + OBB 三个 hdf5",
          "…/_detail/cam_<cam>/camera_keyframe_{orientations,positions}.hdf5",
          "外部 CSV：--camera_params_csv metadata_camera_parameters.csv（按 scene_name 查宽高与 M_proj）"
        ],
        outputs: ["无（本步只读 raw）"],
        flow: [
          "main() 固定顺序执行五步，不可单独跳过（除非改脚本）",
          "Parquet 阶段帧主键：scene_id + cam_id（如 cam_00）+ frame_id（color 文件名中 frame.0007 → 0007）"
        ],
        params: [
          "官方可能已带 final_preview/*.color.jpg；本 pipeline 仍生成 tonemap.jpg 供 Parquet 引用"
        ]
      },
      {
        id: "disk-warning",
        title: "磁盘注意",
        stage: "upstream",
        summary: "派生图像写在 input_root 内。",
        files: [],
        inputs: [],
        outputs: [
          "images/scene_<cam>_final_preview/frame.<i>.tonemap.jpg（Step 3）",
          "images/scene_<cam>_depth/frame.<i>.planar_depth.png uint16 mm（Step 4）",
          "<scene>/masks/<cam>/<frame>/mask_inst_<id>.png 路径写入 parquet（Step 5 逻辑里定义路径）"
        ],
        flow: [
          "Step 3/4 无 skip-if-exists，重跑覆盖",
          "Parquet 在 --output_dir，不把 RGB/depth 打进列",
          "整库空间 ≈ HDF5 + preview + tonemap + planar_depth（后两者为主要增量）"
        ],
        params: [
          "若磁盘紧张：可改脚本对已有 tonemap/depth 跳过，或删 HDF5（Parquet 生成后）"
        ]
      },
      {
        id: "step-intrinsics",
        title: "Step 1 内参",
        stage: "convert",
        summary: "CSV → 每场景一份内参。",
        files: ["prepare_hypersim.py → run_intrinsics()"],
        inputs: ["metadata_camera_parameters.csv 行匹配 scene_name"],
        outputs: [
          "<scene>/_detail/intrinsics.json（3×3 list）",
          "<scene>/_detail/intrinsics.txt（4×4，Parquet 引用此绝对路径）"
        ],
        flow: [
          "fx = M_proj_00 * (W-1)/2，fy = M_proj_11 * (H-1)/2，cx/cy = 图像中心",
          "每场景一个内参文件（所有 cam 共用同一 intrinsics.txt 路径进 Parquet）"
        ],
        params: ["Step 4 用 intrinsics 推图像宽高与 focal 做 distance→planar 缩放"]
      },
      {
        id: "step-extrinsics",
        title: "Step 2 外参",
        stage: "convert",
        summary: "关键帧 HDF5 → 逐帧 txt。",
        files: ["prepare_hypersim.py → run_extrinsics()"],
        inputs: [
          "_detail/cam_<cam>/camera_keyframe_orientations.hdf5",
          "_detail/cam_<cam>/camera_keyframe_positions.hdf5",
          "_detail/metadata_scene.csv → parameter_value 作 position 缩放"
        ],
        outputs: [
          "_detail/cam_<cam>/extrinsics/frame.<i>.extrinsic.json",
          "同路径 frame.<i>.extrinsic.txt（4×4，Parquet pose 列指向 txt）"
        ],
        flow: [
          "R,t 组装 4×4 后乘 diag(1,-1,-1,1) 翻转 y/z 对齐 Open3D",
          "帧索引 i 与 color.hdf5 的 frame.<i> 一致"
        ],
        params: ["无并行参数；场景串行、相机串行"]
      },
      {
        id: "step-tonemap",
        title: "Step 3 Tonemap",
        stage: "convert",
        summary: "HDR → tonemap.jpg",
        files: ["prepare_hypersim.py → run_rgb_tonemap()"],
        inputs: [
          "final_hdf5/frame.<i>.color.hdf5 dataset float HDR",
          "geometry_hdf5/frame.<i>.render_entity_id.hdf5（算 tonemap 缩放）"
        ],
        outputs: [
          "images/scene_<cam>_final_preview/frame.<i>.tonemap.jpg（8-bit BGR via cv2）"
        ],
        flow: [
          "90th percentile 亮度归一化 + gamma=2.2；render_entity_id==0 断言失败则跳过该帧",
          "ThreadPoolExecutor(--tonemap_workers 默认 16) 按帧提交"
        ],
        params: ["不写 color.jpg；Parquet image 列只指向 tonemap.jpg"]
      },
      {
        id: "step-depth",
        title: "Step 4 深度",
        stage: "convert",
        summary: "distance HDF5 → planar PNG。",
        files: ["prepare_hypersim.py → run_depth()"],
        inputs: ["geometry_hdf5/frame.<i>.depth_meters.hdf5", "Step1 intrinsics.json"],
        outputs: [
          "images/scene_<cam>_depth/frame.<i>.planar_depth.png（uint16，毫米）"
        ],
        flow: [
          "planar_depth = euclidean_distance * (focal / ||(x,y,f)||) 逐像素",
          "缺失 intrinsics.json 的场景整场景跳过",
          "ProcessPoolExecutor(--max_workers 默认 32)"
        ],
        params: [
          "Parquet：depth_scale=1000，is_metric_depth=true → load_depth_map 得米制 Z"
        ]
      },
      {
        id: "step-parquet",
        title: "Step 5 Parquet",
        stage: "convert",
        summary: "逐帧一行，绝对路径。",
        files: ["prepare_hypersim.py → run_parquet()", "scannet-labels.combined.tsv"],
        inputs: [
          "tonemap.jpg、planar_depth.png、extrinsic.txt、intrinsics.txt",
          "semantic_instance + semantic hdf5；mesh OBB hdf5 + metadata_objects.csv",
          "可选 --name_filter_json 把非 True 类标为 Unknown"
        ],
        outputs: [
          "--output_dir/batch_<k>.parquet",
          "列：scene_id, id, image, pose, intrinsic, depth_map, depth_scale, is_metric_depth, obj_tags[], bboxes_3d_world_coords[]（9-DoF zxy 欧拉）, masks[], bboxes_2d[], nyu_tags[], instance_ids[]",
          "路径全部为 os.path.abspath（与 EmbodiedScan 相对路径不同）"
        ],
        flow: [
          "先枚举全部 (scene, cam, frame) 再 ProcessPoolExecutor 并行；缺任一 required 文件则该帧 skip",
          "2D：instance map 逐 id 取 bbox + semantic 像素查 NYU40 id→name；过滤 blinds/door/wall 等为 Unknown",
          "3D OBB：instance_id 索引 extents/orientations/positions hdf5，乘 scene scale",
          "chunk 按 scene 边界切：同 scene 不跨 batch 文件"
        ],
        params: [
          "--chunk_size 默认 1000 条记录/文件",
          "demo_preprocessing_hypersim.yaml 无 flatten（已是 per-image）"
        ]
      },
      {
        id: "parquet",
        title: "OpenSpatial Parquet",
        stage: "output",
        summary: "直接进 filter，无 flatten。",
        files: ["config/preprocessing/demo_preprocessing_hypersim.yaml"],
        inputs: ["batch_*.parquet"],
        outputs: ["filter → sam2 → depth_back_projection（无 group）"],
        flow: [
          "dataset.data_dir 指向 parquet 文件或目录",
          "raw_data_root 可省略（路径已是绝对路径）",
          "filter 需要 masks 列存在且与 obj_tags 对齐（Hypersim parquet 自带 masks 路径列表）"
        ],
        params: [
          "group_col_list 不要含 axis_align_matrix（Hypersim 无此列）"
        ]
      }
    ]
  },

  scannetpp: {
    id: "scannetpp",
    tabLabel: "ScanNet++",
    title: "ScanNet++ 预处理流程",
    lead:
      "官方 prepare_iphone_data 解 mkv/bin → prepare_scannetpp.py 写 txt + per-scene Parquet。",
    svg: { width: 1680, height: 600, viewBox: "0 0 1680 600" },
    stageBands: [
      { id: "upstream", x: 24, y: 28, w: 1632, h: 185 },
      { id: "convert", x: 24, y: 238, w: 1632, h: 185 },
      { id: "output", x: 24, y: 448, w: 900, h: 130 }
    ],
    layout: {
      "raw-iphone": { x: 44, y: 82, w: 260, h: 118 },
      "symlink-data": { x: 330, y: 82, w: 250, h: 118 },
      "prepare-iphone": { x: 606, y: 82, w: 280, h: 118 },
      "iphone-ready": { x: 912, y: 82, w: 280, h: 118 },
      "prepare-scannetpp": { x: 260, y: 292, w: 300, h: 118 },
      "camera-txt": { x: 586, y: 292, w: 250, h: 118 },
      parquet: { x: 862, y: 292, w: 260, h: 118 },
      "downstream-note": { x: 44, y: 478, w: 320, h: 88 }
    },
    edges: [
      ["raw-iphone", "symlink-data", "分片 data_1..5"],
      ["symlink-data", "prepare-iphone"],
      ["prepare-iphone", "iphone-ready"],
      ["iphone-ready", "prepare-scannetpp"],
      ["prepare-scannetpp", "camera-txt"],
      ["prepare-scannetpp", "parquet"],
      ["parquet", "downstream-note"]
    ],
    steps: [
      {
        id: "raw-iphone",
        title: "ScanNet++ Raw",
        stage: "upstream",
        summary: "mkv/bin + json + mesh。",
        files: ["github.com/scannetpp/scannetpp"],
        inputs: [
          "iphone/rgb.mkv、depth.bin（192×256 逐帧 lz4/zlib）、rgb_mask.mkv（可选）、colmap/",
          "iphone/pose_intrinsic_imu.json：aligned_poses[]、intrinsic[] 与帧索引对齐",
          "scans/mesh_aligned_0.05.ply、segments_anno.json（segGroups[].label, segments, obb）"
        ],
        outputs: ["无逐帧 pose/intrinsic 磁盘文件（直到 prepare_scannetpp）"],
        flow: [
          "官方下载常分 data_1…data_5；OpenSpatial 脚本读扁平 <input_root>/<scene_id>/",
          "prepare_iphone_data 固定读 <data_root>/data/<scene_id>/（需软链或真实 data/）"
        ],
        params: [
          "无 rgb.mkv 的场景：官方脚本 ffmpeg 失败并 exit(1)，无内置 skip"
        ]
      },
      {
        id: "symlink-data",
        title: "统一 data/ 视图",
        stage: "upstream",
        summary: "分片 → 单一 data/。",
        files: [],
        inputs: ["<root>/data_{1..5}/<scene_id>/"],
        outputs: ["<root>/data/<scene_id> → 软链到某分片（零拷贝）"],
        flow: [
          "for i in 1..5; for scene in data_$i/*; ln -sf \"$(realpath $scene)\" data/$(basename $scene)",
          "prepare_iphone_data.yml：data_root: <root>（parent of data/），不要写 output_root 除非刻意把 RGB 写到别处",
          "prepare_scannetpp：--input_root <root>/data（直接含 scene 文件夹）"
        ],
        params: [
          "同一 scene_id 出现在多分片时后链覆盖前链，需人工去重"
        ]
      },
      {
        id: "prepare-iphone",
        title: "prepare_iphone_data",
        stage: "upstream",
        summary: "官方解包 RGB/depth。",
        files: [
          "scannetpp/iphone/prepare_iphone_data.py",
          "scannetpp/iphone/configs/prepare_iphone_data.yml"
        ],
        inputs: [
          "cfg.data_root/data/<scene_id>/iphone/rgb.mkv",
          "同目录 depth.bin"
        ],
        outputs: [
          "iphone/rgb/frame_%06d.jpg（ffmpeg 全帧；extract_only_rgb_in_colmap 时只解 colmap 列出的帧）",
          "iphone/depth/frame_%06d.png uint16，depth*1000（与 rgb 同帧号）",
          "depth 始终写在输入场景目录，不受 output_root 影响；仅 RGB 可重定向"
        ],
        flow: [
          "cd scannetpp && python -m iphone.prepare_iphone_data",
          "Hydra 覆盖：extract_rgb=true extract_depth=true extract_masks=false",
          "场景列表：splits: [nvs_sem_train, …] 或 scene_ids: [id] 或 scene_list_file",
          "单进程 for scene_id in tqdm；无 max_workers"
        ],
        params: [
          "这是 RGB 首次落盘为 jpg 的步骤（从 mkv 解码，非复制）",
          "缺 mkv：run_command exit_on_error=True 终止整批"
        ]
      },
      {
        id: "iphone-ready",
        title: "解包完成态",
        stage: "upstream",
        summary: "prepare_scannetpp 硬门槛。",
        files: ["data_preprocessing/scannetpp/prepare_scannetpp.py"],
        inputs: [
          "iphone/rgb/frame_XXXXXX.jpg（glob *0.jpg → 仅末位 0 的帧）",
          "iphone/depth/frame_XXXXXX.png 同名",
          "pose_intrinsic_imu.json",
          "scans/mesh_aligned_0.05.ply + segments_anno.json"
        ],
        outputs: ["满足上列即可调用 prepare_scannetpp"],
        flow: [
          "帧对齐：frame_000120.jpg → frame_idx=120 → aligned_poses[120]、intrinsic[120]",
          "缺 depth 或 pose 索引越界：打印 skip，该帧不进入 parquet 列表",
          "不要求 iphone/colmap 或 legacy aligned_pose json"
        ],
        params: [
          "*0.jpg 是采样策略（约 1/10 帧），不是官方 prepare_iphone 的默认输出全集"
        ]
      },
      {
        id: "prepare-scannetpp",
        title: "prepare_scannetpp.py",
        stage: "convert",
        summary: "json→txt + per-scene parquet。",
        files: ["data_preprocessing/scannetpp/prepare_scannetpp.py"],
        inputs: [
          "python prepare_scannetpp.py --input_root .../data --output_dir .../parquet",
          "可选 --selected_tags_file scannet-labels.combined.tsv 语义白名单（segGroups.label）",
          "--chunk_size 100 --max_workers 32"
        ],
        outputs: [
          "iphone/aligned_pose/<frame>.txt、iphone/intrinsic/<frame>.txt（np.savetxt 4×4）",
          "iphone/depth_resized/<frame>.png（仅 depth 尺寸≠rgb 时写；已存在则跳过）",
          "parquet 每行一场景：scene_id, id[], image[], pose[], intrinsic[], depth_map[], obj_tags[], bboxes_3d_world_coords[], depth_scale=1000, is_metric_depth=true, axis_align_matrix=null",
          "image[] 指向原始 rgb 绝对路径；depth_map[] 指向 resized 或原始 depth 绝对路径"
        ],
        flow: [
          "mesh segments_anno segGroups → 子网格 OBB（obb.centroid/axesLengths/normalizedAxes）→ 9-DoF zxy",
          "ProcessPoolExecutor 按 scene 并行；缺 rgb 目录/mesh/anno/json 整场景 return None",
          "batch_*.parquet：每文件最多 chunk_size 条场景记录（不是帧数）"
        ],
        params: [
          "不复制 RGB；额外磁盘主要是 depth_resized + 小 txt",
          "下游 np.loadtxt(pose/intrinsic 路径) 为硬依赖，不能只保留 json"
        ]
      },
      {
        id: "camera-txt",
        title: "相机 txt 产物",
        stage: "convert",
        summary: "Parquet 存路径，不存矩阵。",
        files: ["task/filter/3dbox_filter.py", "task/scene_fusion/depth_back_projection.py"],
        inputs: ["pose_intrinsic_imu.json 逐帧数组"],
        outputs: [
          "aligned_pose/frame_000120.txt：camera-to-world 4×4（来自 aligned_poses[i]）",
          "intrinsic/frame_000120.txt：4×4，3×3 有效（来自 intrinsic[i]）"
        ],
        flow: [
          "3dbox_filter：pose=loadtxt 投影 3D box；intrinsic 同上",
          "depth_back_projection：仅用 intrinsic + depth_map（相机系反投影）",
          "flatten 后每帧一行，pose/intrinsic 变为标量路径字符串"
        ],
        params: [
          "逐帧各一份 intrinsic txt；不共用单文件"
        ]
      },
      {
        id: "parquet",
        title: "OpenSpatial Parquet",
        stage: "output",
        summary: "per-scene list → flatten。",
        files: [
          "config/preprocessing/demo_preprocessing_scannetpp.yaml",
          "assets/preprocessing_input_schema.md"
        ],
        inputs: [
          "list 列：image, id, depth_map, pose, intrinsic（等长 N）",
          "场景级标量列：obj_tags, bboxes_3d_world_coords（各视角共享同一组 3D 框）"
        ],
        outputs: ["flatten 后 per-image 行进入 filter/sam2/fusion/group"],
        flow: [
          "dataset.data_dir = batch_*.parquet 路径",
          "dataset.raw_data_root = scannetpp/data（若 parquet 用绝对路径可省略）",
          "flatten split_col_list 含 image,id,depth_map,pose,intrinsic；obj_tags 整列复制到每帧"
        ],
        params: [
          "与 EmbodiedScan per_scene 不同：ScanNet++ 的 obj_tags 是场景级单 list，非每帧 list"
        ]
      },
      {
        id: "downstream-note",
        title: "→ OpenSpatial Pipeline",
        stage: "output",
        summary: "demo_preprocessing_scannetpp.yaml",
        files: ["run.py", "assets/pipeline/index.html"],
        inputs: ["per-scene parquet"],
        outputs: ["group_stage/group/data.parquet（multiview 标注输入）"],
        flow: [
          "python run.py --config config/preprocessing/demo_preprocessing_scannetpp.yaml --output_dir ...",
          "flatten → 3dbox_filter → sam2_refiner → depth_back_projection → group（group_by: scene_id）"
        ],
        params: [
          "filter 后 masks 写 filter_stage/3dbox_filter/masks/；fusion 写 pointclouds/*.pcd"
        ]
      }
    ]
  }
};

Object.values(PIPELINES).forEach((pipeline) => {
  const byId = Object.fromEntries(pipeline.steps.map((s) => [s.id, s]));
  pipeline.steps.forEach((step) => {
    Object.assign(step, pipeline.layout[step.id] || {});
    step.nodeTitle = step.title;
    step.nodeSummary = step.summary;
  });
  pipeline.byId = byId;
});
