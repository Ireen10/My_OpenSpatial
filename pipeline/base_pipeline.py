from utils.common import get_task_instance
from utils.parquet_io import list_parquet_shards, resolve_task_output_dir
from dataset import build_dataset
import copy
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone


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

    @staticmethod
    def _utc_now_iso():
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _count_samples(data):
        if data is None:
            return 0
        try:
            return int(len(data))
        except TypeError:
            return 0

    @staticmethod
    def _avg_seconds(total_seconds, sample_count):
        if not sample_count:
            return None
        return float(total_seconds) / float(sample_count)

    @staticmethod
    def _round_seconds(value):
        return round(float(value), 6)

    @staticmethod
    def _write_json(path, payload):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp_path, path)

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

        self.timing_report = {
            "schema_version": "1.0",
            "pipeline": {
                "output_root": self.output_root,
                "task_count": len(self.task_queue),
                "max_samples": self.max_samples,
                "status": "initialized",
                "created_at": self._utc_now_iso(),
            },
            "summary": {},
            "tasks": [],
        }

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

    def _task_output_dir(self, task_info):
        return self._resolve_output_path(
            task_info["stage_name"],
            task_info["task_name"],
            task_info["task_cfg"],
            task_info["rel_output_dir"],
        )

    def _pipeline_timing_path(self):
        return os.path.join(self.output_root, "pipeline_timing_report.json")

    def _write_pipeline_timing_report(self):
        self._write_json(self._pipeline_timing_path(), self.timing_report)

    def _write_task_timing_report(self, task_info, task_report):
        output_dir = self._task_output_dir(task_info)
        self._write_json(os.path.join(output_dir, "timing_report.json"), task_report)

    def _record_task_timing(self, task_info, task_report):
        self.timing_report["tasks"].append(task_report)
        self._write_task_timing_report(task_info, task_report)
        self._write_pipeline_timing_report()

    def _build_task_report(
        self,
        task_info,
        *,
        status,
        input_samples,
        output_samples,
        load_seconds,
        run_seconds,
        save_seconds,
        error=None,
    ):
        total_seconds = load_seconds + run_seconds + save_seconds
        return {
            "stage_name": task_info["stage_name"],
            "task_name": task_info["task_name"],
            "task_ref": task_info["full_ref"],
            "queue_index": task_info["queue_idx"],
            "status": status,
            "output_dir": self._task_output_dir(task_info),
            "input_samples": int(input_samples),
            "output_samples": int(output_samples),
            "timing": {
                "load_seconds": self._round_seconds(load_seconds),
                "run_seconds": self._round_seconds(run_seconds),
                "save_seconds": self._round_seconds(save_seconds),
                "total_seconds": self._round_seconds(total_seconds),
                "avg_total_per_input_sample_seconds": (
                    None if input_samples == 0
                    else self._round_seconds(self._avg_seconds(total_seconds, input_samples))
                ),
                "avg_run_per_input_sample_seconds": (
                    None if input_samples == 0
                    else self._round_seconds(self._avg_seconds(run_seconds, input_samples))
                ),
                "avg_total_per_output_sample_seconds": (
                    None if output_samples == 0
                    else self._round_seconds(self._avg_seconds(total_seconds, output_samples))
                ),
            },
            **({"error": str(error)} if error is not None else {}),
        }

    def run(self, end_to_end_start_time=None):
        """Execute all tasks in pipeline sequentially."""
        print(">>> Running Pipeline...")
        pipeline_start = time.perf_counter()
        self.timing_report["pipeline"].update({
            "status": "running",
            "run_started_at": self._utc_now_iso(),
        })
        self._write_pipeline_timing_report()

        try:
            for i, task_info in enumerate(self.task_queue):
                stage_name = task_info["stage_name"]
                task_cfg = task_info["task_cfg"]
                task_name = task_info["task_name"]
                task = task_info["task"]

                load_seconds = 0.0
                if i > 0:
                    depends_on = getattr(task_cfg, "depends_on", None)
                    if depends_on:
                        resolved = self._resolve_dependency_path(depends_on, current_task_idx=i)
                        print(f">>> Loading data for {task_name} from: {resolved}")
                    else:
                        print(f">>> Loading data from: {self.task_queue[i-1]['full_ref']}...")
                    load_start = time.perf_counter()
                    self.load_task_data(task_name, task_cfg, current_task_idx=i)
                    load_seconds = time.perf_counter() - load_start

                input_samples = self._count_samples(self.dataset.data)
                print(f">>> Running Task [{i+1}/{len(self.task_queue)}]: {task_name}...")

                run_start = time.perf_counter()
                processed_data = None
                try:
                    processed_data = task.run(copy.deepcopy(self.dataset.data))
                    run_seconds = time.perf_counter() - run_start

                    output_samples = self._count_samples(processed_data)
                    save_start = time.perf_counter()
                    self.save_task_data(
                        stage_name,
                        task_name,
                        task_cfg,
                        processed_data,
                        task_info["rel_output_dir"],
                    )
                    save_seconds = time.perf_counter() - save_start

                    task_report = self._build_task_report(
                        task_info,
                        status="success",
                        input_samples=input_samples,
                        output_samples=output_samples,
                        load_seconds=load_seconds,
                        run_seconds=run_seconds,
                        save_seconds=save_seconds,
                    )
                    self._record_task_timing(task_info, task_report)
                    avg = task_report["timing"]["avg_total_per_input_sample_seconds"]
                    print(
                        f">>> Timing {task_info['full_ref']}: "
                        f"{task_report['timing']['total_seconds']:.3f}s total"
                        + (f", {avg:.6f}s/sample" if avg is not None else ""),
                        flush=True,
                    )
                except Exception as exc:
                    run_seconds = time.perf_counter() - run_start
                    task_report = self._build_task_report(
                        task_info,
                        status="failed",
                        input_samples=input_samples,
                        output_samples=self._count_samples(processed_data),
                        load_seconds=load_seconds,
                        run_seconds=run_seconds,
                        save_seconds=0.0,
                        error=exc,
                    )
                    self._record_task_timing(task_info, task_report)
                    raise

            run_seconds_total = time.perf_counter() - pipeline_start
            e2e_seconds = (
                time.perf_counter() - end_to_end_start_time
                if end_to_end_start_time is not None
                else run_seconds_total
            )
            final_output_samples = (
                self.timing_report["tasks"][-1]["output_samples"]
                if self.timing_report["tasks"]
                else 0
            )
            self.timing_report["pipeline"].update({
                "status": "success",
                "finished_at": self._utc_now_iso(),
            })
            self.timing_report["summary"] = {
                "run_seconds": self._round_seconds(run_seconds_total),
                "end_to_end_seconds": self._round_seconds(e2e_seconds),
                "end_to_end_includes_pipeline_initialization": end_to_end_start_time is not None,
                "task_total_seconds_sum": self._round_seconds(
                    sum(t["timing"]["total_seconds"] for t in self.timing_report["tasks"])
                ),
                "final_dataset_samples": final_output_samples,
                "avg_end_to_end_per_final_sample_seconds": (
                    None if final_output_samples == 0
                    else self._round_seconds(self._avg_seconds(e2e_seconds, final_output_samples))
                ),
            }
            self._write_pipeline_timing_report()
        except Exception as exc:
            run_seconds_total = time.perf_counter() - pipeline_start
            e2e_seconds = (
                time.perf_counter() - end_to_end_start_time
                if end_to_end_start_time is not None
                else run_seconds_total
            )
            self.timing_report["pipeline"].update({
                "status": "failed",
                "finished_at": self._utc_now_iso(),
                "error": str(exc),
            })
            self.timing_report["summary"] = {
                "run_seconds": self._round_seconds(run_seconds_total),
                "end_to_end_seconds": self._round_seconds(e2e_seconds),
                "end_to_end_includes_pipeline_initialization": end_to_end_start_time is not None,
                "task_total_seconds_sum": self._round_seconds(
                    sum(t["timing"]["total_seconds"] for t in self.timing_report["tasks"])
                ),
            }
            self._write_pipeline_timing_report()
            raise

        print(">>> Pipeline Finished.")

