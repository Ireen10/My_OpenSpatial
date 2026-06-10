from utils.common import get_task_instance
from utils.parquet_io import list_parquet_shards, resolve_task_output_dir
from dataset import build_dataset
import copy
import os
from collections import Counter


class BasePipeline:
    """Pipeline executor with stage-based task management and dependency resolution."""

    @staticmethod
    def _get_duplicate_suffix(occurrence):
        """Convert occurrence count to duplicate suffix: 2->dup1, 3->dup2."""
        return f"dup{occurrence - 1}" if occurrence > 1 else ""

    @staticmethod
    def _iter_stages(stages_cfg):
        """Yield (stage_name, tasks) from config in order."""
        stage_groups = stages_cfg if isinstance(stages_cfg, (list, tuple)) else [stages_cfg]

        for group in stage_groups:
            if hasattr(group, "items"):
                items = group.items()
            elif hasattr(group, "__dict__"):
                items = group.__dict__.items()
            else:
                raise ValueError("stages must be a mapping or list of mappings")

            for stage_name, tasks in items:
                yield stage_name, tasks

    @staticmethod
    def _format_task_ref(stage_name, task_name, occurrence):
        """Format task reference as stage/task#N."""
        return f"{stage_name}/{task_name}#{occurrence}"

    def _resolve_dependency_path(self, depends_on, current_task_idx=None):
        """Resolve depends_on to a parquet file or task output directory.

        Priority:
        1. Existing file path (single .parquet)
        2. Existing directory with one or more *.parquet files
        3. Explicit task ref with #N → that task's output directory
        4. Bare stage/task (unique prior task) → output directory
        5. Legacy output_root/depends_on (directory)
        """
        if os.path.isfile(depends_on):
            return depends_on

        abs_path = os.path.normpath(os.path.join(self.output_root, depends_on))
        if os.path.isfile(abs_path):
            return abs_path
        if os.path.isdir(abs_path):
            list_parquet_shards(abs_path)
            return abs_path

        # Explicit #N reference
        if depends_on in self.task_ref_to_rel_dir:
            out_dir = os.path.join(self.output_root, self.task_ref_to_rel_dir[depends_on])
            list_parquet_shards(out_dir)
            return out_dir

        # Bare "stage/task" - resolve from tasks before current
        if depends_on in self.base_ref_to_tasks:
            candidates = self.base_ref_to_tasks[depends_on]
            if current_task_idx is not None:
                candidates = [t for t in candidates if t["queue_idx"] < current_task_idx]

            if len(candidates) == 1:
                out_dir = os.path.join(self.output_root, candidates[0]["rel_output_dir"])
                list_parquet_shards(out_dir)
                return out_dir

            if len(candidates) > 1:
                refs = [self._format_task_ref(t["stage_name"], t["task_name"], t["occurrence"])
                        for t in candidates]
                raise ValueError(f"Ambiguous depends_on '{depends_on}'. Use one of: {refs}")

        # Legacy: depends_on is a relative output subdirectory
        try:
            out_dir = resolve_task_output_dir(self.output_root, depends_on)
            list_parquet_shards(out_dir)
            return out_dir
        except FileNotFoundError:
            pass

        raise ValueError(f"Cannot resolve depends_on: {depends_on}")

    def __init__(self, cfg):
        self.cfg = cfg 
        self.data_cfg = self.cfg.dataset 
        self.output_root = self.cfg.output_dir

        # Build task queue with occurrence tracking for duplicates
        self.task_queue = []
        pair_counter = Counter()
        for stage_name, tasks in self._iter_stages(cfg.pipeline.stages):
            for task_cfg in tasks:
                task_name = task_cfg.file_name
                pair_counter[(stage_name, task_name)] += 1
                occurrence = pair_counter[(stage_name, task_name)]
                suffix = self._get_duplicate_suffix(occurrence)
                unique_dir = f"{task_name}_{suffix}" if suffix else task_name
                self.task_queue.append({
                    "stage_name": stage_name,
                    "task_cfg": task_cfg,
                    "task_name": task_name,
                    "task": get_task_instance(stage_name, task_cfg, self.cfg),
                    "queue_idx": len(self.task_queue),
                    "occurrence": occurrence,
                    "base_ref": f"{stage_name}/{task_name}",
                    "full_ref": self._format_task_ref(stage_name, task_name, occurrence),
                    "rel_output_dir": os.path.join(stage_name, unique_dir),
                })

        # Build lookup tables for dependency resolution
        self.base_ref_to_tasks = {}
        self.task_ref_to_rel_dir = {}
        for task in self.task_queue:
            self.base_ref_to_tasks.setdefault(task["base_ref"], []).append(task)
            self.task_ref_to_rel_dir[task["full_ref"]] = task["rel_output_dir"]

        # Reuse models across tasks in same stage
        first_task_by_stage = {}
        for task in self.task_queue:
            first_task_by_stage.setdefault(task["stage_name"], task)

        for task in self.task_queue:
            reuse_stage = getattr(task["task"], "reuse_model", None)
            if reuse_stage and reuse_stage in first_task_by_stage:
                task["task"].model = first_task_by_stage[reuse_stage]["task"].model
                print(f">>> Reusing model from {reuse_stage} for {task['task_name']}")

        # Resolve data_dir relative to this run root (same convention as input_tasks_prefix).
        data_dir = getattr(self.data_cfg, "data_dir", None)
        if isinstance(data_dir, str) and data_dir.startswith(".."):
            self.data_cfg.data_dir = os.path.normpath(
                os.path.join(self.output_root, data_dir)
            )

        # Aggregate with input_tasks never reads dataset.data at init — defer heavy parquet load.
        first_cfg = self.task_queue[0]["task_cfg"] if self.task_queue else None
        if first_cfg and getattr(first_cfg, "input_tasks", None):
            self.data_cfg.defer_load = True

        # Build dataset
        self.dataset = build_dataset(self.data_cfg)

        # Debug: slice dataset when --max_samples is given.
        # Stored so every subsequent override_data call can re-apply the limit.
        self.max_samples = getattr(cfg, "max_samples", None)
        self._truncate_dataset_if_needed()

        # Initialize dataset from first task's depends_on if present
        if not self.task_queue:
            raise ValueError("No tasks found in pipeline stages")
        first_task_cfg = self.task_queue[0]["task_cfg"]
        if hasattr(first_task_cfg, "depends_on"):
            resolved_path = self._resolve_dependency_path(first_task_cfg.depends_on, current_task_idx=0)
            print(f">>> Overriding dataset with {resolved_path}...")
            self.dataset.override_data(resolved_path)
            self._truncate_dataset_if_needed()

    def _truncate_dataset_if_needed(self) -> None:
        """Apply --max_samples truncation (no-op when not set)."""
        if self.dataset.data is None or not self.max_samples:
            return
        full_len = len(self.dataset.data)
        self.dataset.data = self.dataset.data.iloc[: self.max_samples].reset_index(drop=True)
        print(
            f">>> [debug] dataset truncated to {len(self.dataset.data)}/{full_len} rows "
            f"(--max_samples {self.max_samples})."
        )

    def _resolve_output_path(self, stage_name, task_name, task_cfg, default_rel_dir=None):
        """Resolve output directory for task results."""
        output_dir = task_cfg.output_dir
        if output_dir:
            return output_dir if os.path.exists(output_dir) else os.path.join(self.output_root, output_dir)

        rel_dir = default_rel_dir or os.path.join(stage_name, task_name)
        return os.path.join(self.output_root, rel_dir)

    def save_task_data(self, stage_name, task_name, task_cfg, processed_data, default_rel_dir=None):
        """Save processed task data to parquet."""
        output_path = self._resolve_output_path(stage_name, task_name, task_cfg, default_rel_dir)
        os.makedirs(output_path, exist_ok=True)

        skip_parquet = bool(getattr(task_cfg, "skip_parquet", False))
        if stage_name == "export_stage" and skip_parquet:
            print(f">>> Skipping data.parquet for {task_name} (upstream export only)")
            return
        if (
            stage_name == "export_stage"
            and processed_data is not None
            and hasattr(processed_data, "__len__")
            and len(processed_data) == 0
        ):
            print(f">>> Skipping empty data.parquet for {task_name}")
            return

        if stage_name == "annotation_stage":
            batch_size = getattr(task_cfg, "save_batch_size", 8192)
            keep_cols = [
                k for k in getattr(task_cfg, "keep_data_columns",
                                   ["messages", "metadata", "question_tags", "question_types"])
                if k != "QA_images"
            ]
            self.dataset.save_data(os.path.join(output_path, "data.parquet"), processed_data,
                                  annotation_flag=True, batch_size=batch_size, keep_data_columns=keep_cols)
        elif stage_name == "aggregate_stage":
            self.dataset.save_data(os.path.join(output_path, "data.parquet"), processed_data)
        else:
            self.dataset.save_data(os.path.join(output_path, "data.parquet"), processed_data)

    def load_task_data(self, task_name, task_cfg, current_task_idx):
        """Load input data from depends_on path."""
        depends_on = getattr(task_cfg, "depends_on", None)
        if not depends_on:
            raise ValueError(f"Task {task_name} missing depends_on field")

        resolved_path = self._resolve_dependency_path(depends_on, current_task_idx=current_task_idx)
        self.dataset.override_data(resolved_path)
        self._truncate_dataset_if_needed()

    def run(self):
        """Execute all tasks in pipeline sequentially."""
        print(">>> Running Pipeline...")
        for i, task_info in enumerate(self.task_queue):
            stage_name = task_info["stage_name"]
            task_cfg = task_info["task_cfg"]
            task_name = task_info["task_name"]
            task = task_info["task"]

            if i > 0:
                depends_on = getattr(task_cfg, "depends_on", None)
                if depends_on:
                    resolved = self._resolve_dependency_path(depends_on, current_task_idx=i)
                    print(f">>> Loading data for {task_name} from: {resolved}")
                else:
                    print(f">>> Loading data from: {self.task_queue[i-1]['full_ref']}...")
                self.load_task_data(task_name, task_cfg, current_task_idx=i)

            print(f">>> Running Task [{i+1}/{len(self.task_queue)}]: {task_name}...")
            processed_data = task.run(copy.deepcopy(self.dataset.data))
            self.save_task_data(stage_name, task_name, task_cfg, processed_data, task_info["rel_output_dir"])

        print(">>> Pipeline Finished.")

