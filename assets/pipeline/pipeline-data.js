const PIPELINE_STEPS = [
  {
    id: "runner",
    title: "Pipeline Runner",
    stage: "orchestration",
    lane: "control",
    x: 56,
    y: 42,
    w: 190,
    h: 92,
    summary: "Loads YAML config, instantiates BasePipeline, resolves task dependencies, and writes each stage output.",
    files: [
      "run.py",
      "pipeline/base_pipeline.py",
      "utils/common.py"
    ],
    inputs: [
      "config/preprocessing/demo_preprocessing_*.yaml",
      "config/annotation/demo_*_all.yaml",
      "config/aggregate/demo_aggregate_*.yaml",
      "--output_dir run root"
    ],
    flow: [
      "Parse CLI arguments and YAML with duplicate-key-safe loader.",
      "Build run directory as {output_root}/{pipeline.file_name}_{config_stem}.",
      "Instantiate pipeline.<file_name>.<class_name> and every task in pipeline.stages.",
      "For each task, resolve depends_on as a file, directory, explicit task ref, or prior stage/task output.",
      "Run task.run(DataFrame) and save data.parquet unless export_stage is configured with skip_parquet."
    ],
    outputs: [
      "{run_root}/{stage}/{task}/data.parquet",
      "Stage-specific side artifacts such as masks, pointclouds, JSONL, and image tar shards"
    ]
  },
  {
    id: "source",
    title: "Source Dataset",
    stage: "preprocess",
    lane: "data",
    x: 56,
    y: 206,
    w: 190,
    h: 92,
    summary: "Input parquet plus raw RGB/depth assets and camera/object metadata.",
    files: [
      "dataset/image_base.py",
      "config/preprocessing/demo_preprocessing_embodiedscan.yaml",
      "config/preprocessing/demo_preprocessing_scannetpp.yaml"
    ],
    inputs: [
      "image",
      "depth_map + depth_scale",
      "pose + intrinsic",
      "obj_tags",
      "bboxes_3d_world_coords"
    ],
    flow: [
      "Build ImageBase dataset from dataset.data_dir.",
      "Optionally join relative asset paths with raw_data_root.",
      "Expose data as pandas DataFrame for the first preprocessing task."
    ],
    outputs: [
      "DataFrame rows consumed by preprocessing tasks"
    ]
  },
  {
    id: "flatten",
    title: "Flatten",
    stage: "preprocess",
    lane: "compute",
    x: 300,
    y: 206,
    w: 190,
    h: 92,
    summary: "Optional conversion from list-valued multi-image samples to one row per image.",
    files: [
      "task/flatten/flatten.py"
    ],
    inputs: [
      "Rows whose anchor_col, usually image, may be a list",
      "split_col_list fields with the same length as anchor_col"
    ],
    flow: [
      "Check whether the anchor field is list-valued.",
      "Validate every split column exists and has the same number of elements.",
      "Create one new row per image index, taking split fields at that index and copying non-split fields.",
      "Leave already single-image rows unchanged."
    ],
    outputs: [
      "Per-image rows with scalar image/depth/pose/intrinsic fields"
    ]
  },
  {
    id: "filter",
    title: "3D Box Filter",
    stage: "preprocess",
    lane: "compute",
    x: 544,
    y: 206,
    w: 190,
    h: 92,
    summary: "Removes background or invalid objects and emits object masks for surviving 3D boxes.",
    files: [
      "task/filter/3dbox_filter.py"
    ],
    inputs: [
      "image, depth_map, depth_scale",
      "pose, intrinsic",
      "obj_tags",
      "bboxes_3d_world_coords"
    ],
    flow: [
      "Load RGB image and metric depth map.",
      "Drop configured background tags such as ceiling, floor, wall, and object.",
      "Project each 3D box into the image and require enough projected area to be visible.",
      "Back-project depth into a scene point cloud and require enough points/volume inside each oriented box.",
      "Save binary mask PNGs for valid objects and filter object-aligned fields by kept indices."
    ],
    outputs: [
      "Filtered obj_tags and bboxes_3d_world_coords",
      "masks stored under filter_stage/3dbox_filter/masks",
      "filter_stage/3dbox_filter/data.parquet"
    ]
  },
  {
    id: "localization",
    title: "Localization",
    stage: "preprocess",
    lane: "compute",
    x: 788,
    y: 206,
    w: 190,
    h: 92,
    summary: "Refines object masks and 2D boxes using SAM2 or SAM3 while preserving existing 3D boxes.",
    files: [
      "task/localization/sam2_refiner.py",
      "task/localization/sam3_refiner.py",
      "config/preprocessing/demo_preprocessing_*_sam3.yaml"
    ],
    inputs: [
      "Filtered image/object rows",
      "obj_tags and bboxes_3d_world_coords",
      "Existing masks or box prompts depending on localizer"
    ],
    flow: [
      "Load the localization model selected by config.",
      "Refine object masks and update 2D boxes from the refined masks.",
      "For SAM3, run text-only prompts per unique object tag and match predicted masks back to input boxes with mask-box similarity.",
      "Do not recompute or overwrite bboxes_3d_world_coords; only remove entries for objects filtered out by localization.",
      "Preserve the object-aligned columns that annotation and scene fusion need."
    ],
    outputs: [
      "Refined masks / bboxes_2d / object-aligned fields",
      "localization_stage/sam*_refiner/data.parquet"
    ]
  },
  {
    id: "fusion",
    title: "Scene Fusion",
    stage: "preprocess",
    lane: "compute",
    x: 1032,
    y: 206,
    w: 190,
    h: 92,
    summary: "Back-projects each object mask through depth into per-object 3D point clouds without changing retained 3D box values.",
    files: [
      "task/scene_fusion/depth_back_projection.py"
    ],
    inputs: [
      "intrinsic",
      "depth_map + depth_scale",
      "masks",
      "obj_tags"
    ],
    flow: [
      "Load masks from paths or embedded bytes and resize them to depth resolution.",
      "Use mask-only backprojection: collect valid depth pixels under each object mask.",
      "Sample at most max_points_per_object points per object, optionally apply statistical outlier removal, and write .pcd files.",
      "Drop invalid mask/object entries by valid_flags and require at least two valid object point clouds.",
      "Filter only explicit object-aligned fields; retained bboxes_3d_world_coords values are copied unchanged."
    ],
    outputs: [
      "pointclouds list on each retained row",
      "scene_fusion_stage/depth_back_projection/pointclouds/*.pcd",
      "scene_fusion_stage/depth_back_projection/data.parquet"
    ]
  },
  {
    id: "group",
    title: "Group Views",
    stage: "preprocess",
    lane: "compute",
    x: 1276,
    y: 206,
    w: 190,
    h: 92,
    summary: "Optional multiview step that groups per-image rows into per-scene list-valued rows.",
    files: [
      "task/group/group.py"
    ],
    inputs: [
      "Per-image fused rows",
      "group_by such as scene_id",
      "group_col_list of image/depth/pose/object fields"
    ],
    flow: [
      "Iterate rows and collect rows with the same group key.",
      "Append group_col_list fields into lists while copying stable scene-level fields once.",
      "Return one row per scene for multiview annotation."
    ],
    outputs: [
      "group_stage/group/data.parquet",
      "List-valued multiview rows"
    ]
  },
  {
    id: "annotation-input",
    title: "Annotation Input",
    stage: "annotation",
    lane: "data",
    x: 56,
    y: 404,
    w: 190,
    h: 92,
    summary: "Preprocessed parquet selected for singleview or multiview annotation configs.",
    files: [
      "config/annotation/demo_singleview_all.yaml",
      "config/annotation/demo_multiview_all.yaml"
    ],
    inputs: [
      "Singleview: scene_fusion_stage/depth_back_projection/data.parquet",
      "Multiview: group_stage/group/data.parquet"
    ],
    flow: [
      "Set dataset.data_dir or a task depends_on path to the preprocessed parquet.",
      "For tasks after the first annotation task, configs point back to the same input rather than chaining from prior annotation output.",
      "BasePipeline loads the appropriate parquet before each task."
    ],
    outputs: [
      "Task-specific annotation DataFrame input"
    ]
  },
  {
    id: "scene-graph",
    title: "SceneGraph",
    stage: "annotation",
    lane: "compute",
    x: 300,
    y: 404,
    w: 190,
    h: 92,
    summary: "Converts each row into object/view structures used by annotation tasks.",
    files: [
      "task/annotation/core/scene_graph.py",
      "task/annotation/core/base_annotation_task.py",
      "task/annotation/core/base_multiview_task.py"
    ],
    inputs: [
      "Singleview or multiview row",
      "Object tags, masks, pointclouds, poses, intrinsics"
    ],
    flow: [
      "Validate required fields and skip rows with missing or empty object lists.",
      "Singleview builds SceneGraph.from_singleview_example.",
      "Multiview builds SceneGraph.from_multiview_example with max_num_views and pose-diversity controls.",
      "Expose nodes, view appearances, camera geometry, and object point clouds to task-specific logic."
    ],
    outputs: [
      "SceneGraph with SceneNode and per-view appearance records"
    ]
  },
  {
    id: "task-logic",
    title: "Task Logic",
    stage: "annotation",
    lane: "compute",
    x: 544,
    y: 404,
    w: 190,
    h: 92,
    summary: "Distance, depth, size, position, grounding, and multiview tasks sample objects/views and compute answers.",
    files: [
      "task/annotation/distance.py",
      "task/annotation/depth_annotation.py",
      "task/annotation/size.py",
      "task/annotation/position.py",
      "task/annotation/3d_grounding.py",
      "task/annotation/multiview_*.py"
    ],
    inputs: [
      "SceneGraph",
      "sub_tasks config",
      "Question templates and metric gates"
    ],
    flow: [
      "Select valid object pairs, target objects, or view sets according to each task.",
      "Compute geometric quantities such as distance, depth ordering, size, position, or correspondence.",
      "Pick OE, MCQ, or grounding templates based on enabled sub_tasks.",
      "Optionally mark referents with box, point, or mask overlays."
    ],
    outputs: [
      "Prompt strings containing question and answer",
      "Processed/marked QA images",
      "Question tags and question types"
    ]
  },
  {
    id: "prompt-metadata",
    title: "Prompt + Metadata",
    stage: "annotation",
    lane: "compute",
    x: 788,
    y: 404,
    w: 190,
    h: 92,
    summary: "Renders structured prompts and records traceable turn metadata.",
    files: [
      "task/annotation/core/structured_prompt_template.py",
      "task/annotation/core/sample_metadata.py",
      "task/annotation/core/visual_marker.py",
      "task/annotation/core/message_builder.py"
    ],
    inputs: [
      "Task prompt + answer",
      "VisualMarker mark_spec",
      "Preprocess row visual anchors"
    ],
    flow: [
      "Render templates and split question/answer text.",
      "Build turn records with task name, sub_task, question_type, template_id, answer text, referent slots, and view indices.",
      "Build visual_anchor linking the annotation row back to its parent preprocess id and image refs.",
      "Create singleview or multiview messages with image placeholders."
    ],
    outputs: [
      "messages",
      "metadata.turns",
      "metadata.visual_anchor",
      "metadata.mark_spec"
    ]
  },
  {
    id: "annotation-output",
    title: "Annotation Parquets",
    stage: "annotation",
    lane: "data",
    x: 1032,
    y: 404,
    w: 190,
    h: 92,
    summary: "Each annotation task writes a separate flattened parquet for aggregation.",
    files: [
      "pipeline/base_pipeline.py",
      "dataset/image_base.py"
    ],
    inputs: [
      "Task output DataFrame from BaseAnnotationTask.apply_transform"
    ],
    flow: [
      "BasePipeline calls dataset.save_data with annotation_flag=True.",
      "Annotation outputs keep compact columns such as messages, metadata, question_tags, and question_types.",
      "Each configured task writes independently under annotation_stage."
    ],
    outputs: [
      "annotation_stage/distance/data.parquet",
      "annotation_stage/depth_annotation/data.parquet",
      "annotation_stage/multiview_correspondence/data.parquet",
      "Other annotation_stage/*/data.parquet"
    ]
  },
  {
    id: "aggregate-input",
    title: "Aggregate Config",
    stage: "aggregate",
    lane: "data",
    x: 56,
    y: 602,
    w: 190,
    h: 92,
    summary: "Selects the set of annotation task outputs to combine.",
    files: [
      "config/aggregate/demo_aggregate_singleview.yaml",
      "config/aggregate/demo_aggregate_multiview.yaml"
    ],
    inputs: [
      "input_tasks_prefix",
      "input_tasks list",
      "dedup and merge policy flags"
    ],
    flow: [
      "Aggregation run defers normal dataset loading when input_tasks is configured.",
      "Resolve each task ref relative to the aggregate run root.",
      "Load all annotation task parquets as turn records."
    ],
    outputs: [
      "List of TurnRecord objects across configured tasks"
    ]
  },
  {
    id: "turn-load",
    title: "Load Turns",
    stage: "aggregate",
    lane: "compute",
    x: 300,
    y: 602,
    w: 190,
    h: 92,
    summary: "Expands annotation rows into per-turn records with computed keys.",
    files: [
      "task/aggregate/turn_io.py",
      "task/aggregate/fingerprint.py"
    ],
    inputs: [
      "messages and metadata from annotation parquet",
      "task_name from input task ref"
    ],
    flow: [
      "Read annotation conversations and metadata turns.",
      "Extract visualization question/answer text for each turn.",
      "Compute question_core_key to identify semantically duplicate questions.",
      "Compute dedup_fingerprint using task name, visual anchor, view group, and question key.",
      "Compute merge_group_key from ordered image refs, view group, and normalized mark_spec."
    ],
    outputs: [
      "TurnRecord list with dedup_fingerprint and merge_group_key"
    ]
  },
  {
    id: "dedup",
    title: "Dedup Turns",
    stage: "aggregate",
    lane: "compute",
    x: 544,
    y: 602,
    w: 190,
    h: 92,
    summary: "Drops duplicate turns within each task while preferring structured semantic records.",
    files: [
      "task/aggregate/sample_aggregator.py",
      "task/aggregate/fingerprint.py"
    ],
    inputs: [
      "TurnRecord groups by task",
      "dedup_keep_policy such as semantic_first"
    ],
    flow: [
      "Group records by dedup_fingerprint within each task.",
      "If a group has multiple candidates, choose a winner with pick_dedup_winner.",
      "Preserve source order and refresh visualization Q/A from the source row when needed."
    ],
    outputs: [
      "Deduplicated TurnRecord list",
      "Aggregate stats: turns_in and dedup_removed"
    ]
  },
  {
    id: "merge",
    title: "Merge Samples",
    stage: "aggregate",
    lane: "compute",
    x: 788,
    y: 602,
    w: 190,
    h: 92,
    summary: "Combines turns that share the same visual input group into one training sample.",
    files: [
      "task/aggregate/sample_aggregator.py"
    ],
    inputs: [
      "Deduplicated TurnRecord list",
      "merge_group_key"
    ],
    flow: [
      "Group turns by merge_group_key.",
      "Sort grounding turns first, then preserve original parquet order.",
      "Flatten visualization Q/A turns into messages.",
      "Carry image_refs, visual_anchor, representative mark_spec, source task provenance, and turn metadata.",
      "Assign a new sample_id."
    ],
    outputs: [
      "schema_version",
      "sample_id",
      "merge_group_key",
      "image_refs",
      "messages_json",
      "metadata_json"
    ]
  },
  {
    id: "merged-parquet",
    title: "Merged Parquet",
    stage: "aggregate",
    lane: "data",
    x: 1032,
    y: 602,
    w: 190,
    h: 92,
    summary: "The canonical aggregated sample table used by the upstream exporter.",
    files: [
      "task/aggregate/sample_aggregator.py"
    ],
    inputs: [
      "Merged sample dictionaries"
    ],
    flow: [
      "Coerce nested numpy values to JSON-safe objects.",
      "Serialize messages and metadata into JSON strings.",
      "Save through BasePipeline as aggregate_stage/sample_aggregator/data.parquet."
    ],
    outputs: [
      "aggregate_stage/sample_aggregator/data.parquet"
    ]
  },
  {
    id: "upstream-export",
    title: "Upstream Export",
    stage: "export",
    lane: "compute",
    x: 300,
    y: 800,
    w: 190,
    h: 92,
    summary: "Writes OpenSpatial upstream sharded metadata and image archives.",
    files: [
      "task/export/dataset_exporter.py",
      "dataset/upstream_export.py"
    ],
    inputs: [
      "aggregate_stage/sample_aggregator/data.parquet",
      "view_scope singleview or multiview",
      "schema_version"
    ],
    flow: [
      "Convert each merged parquet row into an upstream record.",
      "Normalize messages from messages_json.",
      "Collect image_refs and pack referenced images into per-shard tar files.",
      "Write JSONL metadata shards and dataset-level metadata.json.",
      "Skip export_stage data.parquet when skip_parquet is true."
    ],
    outputs: [
      "export_stage/dataset_exporter/jsonl/metadata_*.jsonl",
      "export_stage/dataset_exporter/images/metadata_*.tar",
      "export_stage/dataset_exporter/metadata.json"
    ]
  },
  {
    id: "pangu",
    title: "Pangu ML Convert",
    stage: "export",
    lane: "compute",
    x: 544,
    y: 800,
    w: 190,
    h: 92,
    summary: "Converts the upstream bundle into the downstream Pangu ML training schema.",
    files: [
      "script/export_to_pangu_ml.py",
      "../HW_pangu_ml/docs/pangu_ml_data_schema.md"
    ],
    inputs: [
      "--export-dir pointing to upstream export root",
      "--output-root for Pangu ML dataset",
      "metadata_*.jsonl + metadata_*.tar"
    ],
    flow: [
      "Discover upstream shard pairs and process shard-level jobs, optionally with file-level processes and intra-shard threads.",
      "Resolve images from each upstream tar, assign final mark colors at export time, and render mark overlays from mark_spec when needed.",
      "Move all image content into the first user turn and strip image placeholder tokens from text.",
      "Rewrite legacy mark tokens into natural text and apply export-time text cleanup for capitalization and legacy correspondence/distance fixes.",
      "Map roles from human/gpt to user/assistant.",
      "Write data_*.jsonl and data_*.tar shards with Pangu-safe sample/image paths."
    ],
    outputs: [
      "jsonl/data_*.jsonl",
      "images/data_*.tar",
      "Pangu ML-compatible multimodal training records"
    ]
  },
  {
    id: "final",
    title: "Training Bundle",
    stage: "export",
    lane: "data",
    x: 788,
    y: 800,
    w: 190,
    h: 92,
    summary: "Final downstream dataset bundle ready for the Pangu ML loader.",
    files: [
      "../HW_pangu_ml/docs/pangu_ml_data_schema.md"
    ],
    inputs: [
      "Pangu ML conversion outputs"
    ],
    flow: [
      "Shard names follow data_{shard:06d}.jsonl and data_{shard:06d}.tar.",
      "Each JSONL row references images by tar-relative paths.",
      "Conversation roles and content blocks match downstream training expectations."
    ],
    outputs: [
      "{output_root}/jsonl/data_*.jsonl",
      "{output_root}/images/data_*.tar"
    ]
  }
];

const PIPELINE_EDGES = [
  ["runner", "source"],
  ["source", "flatten"],
  ["flatten", "filter"],
  ["filter", "localization"],
  ["localization", "fusion"],
  ["fusion", "group"],
  ["fusion", "annotation-input", "singleview"],
  ["group", "annotation-input", "multiview"],
  ["annotation-input", "scene-graph"],
  ["scene-graph", "task-logic"],
  ["task-logic", "prompt-metadata"],
  ["prompt-metadata", "annotation-output"],
  ["annotation-output", "aggregate-input"],
  ["aggregate-input", "turn-load"],
  ["turn-load", "dedup"],
  ["dedup", "merge"],
  ["merge", "merged-parquet"],
  ["merged-parquet", "upstream-export"],
  ["upstream-export", "pangu"],
  ["pangu", "final"]
];

const STAGE_LABELS = {
  orchestration: "Orchestration",
  preprocess: "Preprocess",
  annotation: "Annotation",
  aggregate: "Aggregate",
  export: "Export"
};
