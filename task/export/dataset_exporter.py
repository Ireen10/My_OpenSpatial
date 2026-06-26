"""
M7: Export merged aggregate parquet to sharded upstream JSONL + tar (§4.7).

Writes under ``{output_dir}/jsonl`` and ``{output_dir}/images`` plus ``metadata.json``.
Does not require a downstream ``data.parquet`` (set ``skip_parquet: true`` on the task).
"""

from __future__ import annotations

import os

import pandas as pd

from dataset.upstream_export import UPSTREAM_SCHEMA_VERSION, write_sharded_upstream_bundle
from task.base_task import BaseTask


class DatasetExporter(BaseTask):
    """Read merged_samples parquet; write sharded upstream bundle."""

    def __init__(self, args):
        super().__init__(args)
        pipeline_root = args.get("output_root") or "."
        stage_out = args.get("output_dir") or "export_stage/dataset_exporter"
        if not os.path.isabs(stage_out):
            stage_out = os.path.join(pipeline_root, stage_out)
        export_rel = args.get("export_dir")
        if export_rel in (None, "", "."):
            self.export_root = stage_out
        elif os.path.isabs(export_rel):
            self.export_root = export_rel
        else:
            self.export_root = os.path.join(stage_out, export_rel)
        self.schema_version = args.get("schema_version", UPSTREAM_SCHEMA_VERSION)
        self.pipeline_run_id = args.get("pipeline_run_id")
        self.view_scope = args.get("view_scope") or self._infer_view_scope(stage_out)
        self.shard_size = int(args.get("shard_size", 8192))
        self.num_workers = int(args.get("num_workers", 1)) if self.use_multi_processing else 1

    @staticmethod
    def _infer_view_scope(stage_out: str) -> str:
        low = stage_out.replace("\\", "/").lower()
        return "multiview" if "multiview" in low else "singleview"

    def run(self, dataset: pd.DataFrame) -> pd.DataFrame:
        write_sharded_upstream_bundle(
            dataset,
            self.export_root,
            schema_version=self.schema_version,
            pipeline_run_id=self.pipeline_run_id,
            view_scope=self.view_scope,
            shard_size=self.shard_size,
            num_workers=self.num_workers,
        )
        return pd.DataFrame()


class PassThroughExportTask(BaseTask):
    """No-op transform: for ComposedDataset export_bundle save on unchanged aggregate rows."""

    def run(self, dataset: pd.DataFrame) -> pd.DataFrame:
        return dataset
