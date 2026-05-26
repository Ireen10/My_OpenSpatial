"""
M7: Export merged aggregate parquet to upstream JSONL + images.tar + manifest (§4.7).

Output is self-contained upstream data (metadata-rich), not a downstream training format.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from dataset.upstream_export import UPSTREAM_SCHEMA_VERSION, write_upstream_bundle
from task.base_task import BaseTask


class DatasetExporter(BaseTask):
    """Read merged_samples parquet; write upstream bundle via dataset.upstream_export."""

    def __init__(self, args):
        super().__init__(args)
        pipeline_root = args.get("output_root") or "."
        stage_out = args.get("output_dir") or "export_stage/dataset_exporter"
        if not os.path.isabs(stage_out):
            stage_out = os.path.join(pipeline_root, stage_out)
        export_rel = args.get("export_dir", "export")
        self.export_root = export_rel if os.path.isabs(export_rel) else os.path.join(stage_out, export_rel)
        self.schema_version = args.get("schema_version", UPSTREAM_SCHEMA_VERSION)
        self.pipeline_run_id = args.get("pipeline_run_id")

    def run(self, dataset: pd.DataFrame) -> pd.DataFrame:
        summary = write_upstream_bundle(
            dataset,
            self.export_root,
            schema_version=self.schema_version,
            pipeline_run_id=self.pipeline_run_id,
        )
        return pd.DataFrame([summary])


class PassThroughExportTask(BaseTask):
    """No-op transform: for ComposedDataset export_bundle save on unchanged aggregate rows."""

    def run(self, dataset: pd.DataFrame) -> pd.DataFrame:
        return dataset
