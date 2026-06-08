"""JSONL upstream bundle I/O for OpenSpatial pipelines (§4.7 / M7)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from dataset.image_base import (
    ASSET_PATH_FIELDS,
    HF_REPO_PATTERN,
    ImageBaseDataset,
    _resolve_asset_value,
)
from dataset.upstream_export import (
    JSONL_SUBDIR,
    MANIFEST_FILENAME,
    SAMPLES_FILENAME,
    UPSTREAM_SCHEMA_VERSION,
    is_sharded_upstream_root,
    load_manifest,
    merged_row_to_upstream_record,
    read_sharded_upstream,
    read_upstream_bundle,
    read_upstream_jsonl,
    write_sharded_upstream_bundle,
)

PIPELINE_SAVE_BASENAME = "data.jsonl"


class JsonlBaseDataset(ImageBaseDataset):
    """
    Pipeline dataset hook for upstream JSONL + tar bundles.

    Load:
      - sharded export root: ``jsonl/metadata_*.jsonl`` + ``images/metadata_*.tar``
      - legacy ``manifest.json`` + ``samples.jsonl`` + ``images.tar``
      - any standalone ``.jsonl`` file
      - merged ``data.parquet`` (convenience: same rows as aggregate output)

    Save (``export_bundle: true`` in YAML):
      - sharded bundle under ``export_dir`` (default: dataset root, not ``export/``)
      - optional pipeline index at ``data.jsonl`` (one summary line)

    YAML (decoupled with ComposedDataset):
        input_dataset_name: image_base
        output_dataset_name: jsonl_base
        export_bundle: true
        export_dir: export
    """

    MODALITY = "image"

    def __init__(self, cfg, *, _skip_load=False):
        if not cfg.data_dir and not _skip_load:
            raise ValueError("cfg.data_dir is required")
        self.cfg = cfg
        self.data_dir = getattr(cfg, "data_dir", None)
        self.raw_data_root = getattr(cfg, "raw_data_root", None)
        if self.raw_data_root:
            self.raw_data_root = os.path.abspath(self.raw_data_root)
        self.output_path = getattr(cfg, "output_path", None) or PIPELINE_SAVE_BASENAME
        self.export_bundle = bool(getattr(cfg, "export_bundle", False))
        self.export_dir = getattr(cfg, "export_dir", ".")
        self.view_scope = getattr(cfg, "view_scope", None)
        self.schema_version = getattr(cfg, "schema_version", UPSTREAM_SCHEMA_VERSION)
        self.pipeline_run_id = getattr(cfg, "pipeline_run_id", None)
        self.data = None if _skip_load else self._load()

    def _load(self) -> pd.DataFrame:
        return self._load_from_path(self.data_dir)

    def _load_from_path(self, path: str) -> pd.DataFrame:
        path = str(path)
        if HF_REPO_PATTERN.match(path):
            raise NotImplementedError(
                "JsonlBaseDataset does not load HuggingFace Hub repos yet."
            )

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(path)

        if p.is_dir():
            return self._load_export_dir(p)
        if p.suffix.lower() == ".jsonl":
            return self._dataframe_from_upstream_records(read_upstream_jsonl(p))
        if p.suffix.lower() == ".parquet":
            from utils.parquet_io import load_parquet_dataframe
            return self._apply_raw_data_root(load_parquet_dataframe(str(p)))

        raise NotImplementedError(
            f"JsonlBaseDataset._load_from_path: unsupported input '{path}'."
        )

    @staticmethod
    def _load_export_dir(directory: Path) -> pd.DataFrame:
        if is_sharded_upstream_root(directory):
            return JsonlBaseDataset._dataframe_from_upstream_records(
                read_sharded_upstream(directory)
            )
        manifest_path = directory / MANIFEST_FILENAME
        jsonl_path = directory / SAMPLES_FILENAME
        if manifest_path.is_file() and jsonl_path.is_file():
            load_manifest(directory)
            return JsonlBaseDataset._dataframe_from_upstream_records(
                read_upstream_jsonl(jsonl_path)
            )
        try:
            return JsonlBaseDataset._dataframe_from_upstream_records(
                read_upstream_bundle(directory)
            )
        except FileNotFoundError:
            pass
        for child in sorted(directory.glob("*.jsonl")):
            return JsonlBaseDataset._dataframe_from_upstream_records(
                read_upstream_jsonl(child)
            )
        raise FileNotFoundError(
            f"No upstream bundle in {directory} "
            f"(expected sharded {JSONL_SUBDIR}/ or legacy {MANIFEST_FILENAME} + {SAMPLES_FILENAME})"
        )

    @staticmethod
    def _dataframe_from_upstream_records(records: list[dict]) -> pd.DataFrame:
        rows = []
        for rec in records:
            row = {
                "schema_version": rec.get("schema_version"),
                "sample_id": rec.get("sample_id"),
                "merge_group_key": rec.get("merge_group_key"),
                "image_refs": rec.get("image_refs"),
                "messages_json": json.dumps(rec.get("messages") or [], ensure_ascii=False),
                "metadata_json": json.dumps(rec.get("metadata") or {}, ensure_ascii=False),
            }
            if rec.get("_bundle_root"):
                row["bundle_root"] = rec["_bundle_root"]
            if rec.get("_shard_tar"):
                row["shard_tar"] = rec["_shard_tar"]
            rows.append(row)
        return pd.DataFrame(rows)

    def _apply_raw_data_root(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.raw_data_root or df is None or len(df) == 0:
            return df
        for col in ASSET_PATH_FIELDS:
            if col not in df.columns:
                continue
            df[col] = df[col].apply(
                lambda v: _resolve_asset_value(v, self.raw_data_root)
            )
        return df

    def override_data(self, data_path: str) -> None:
        try:
            self.data = self._load_from_path(data_path)
        except Exception as exc:
            raise ValueError(f"Failed to load data: {data_path}") from exc

    def _resolve_jsonl_output_path(self, data_path: str) -> str:
        if self.output_path and os.path.isabs(self.output_path):
            return self.output_path
        if self.output_path and "/" not in self.output_path and "\\" not in self.output_path:
            return os.path.join(os.path.dirname(data_path), self.output_path)
        if data_path.endswith(".parquet"):
            return data_path[: -len(".parquet")] + ".jsonl"
        return data_path if data_path.endswith(".jsonl") else data_path + ".jsonl"

    def _resolve_export_root(self, data_path: str) -> Path:
        rel = self.export_dir
        if rel in (None, "", "."):
            return Path(os.path.dirname(data_path))
        if os.path.isabs(rel):
            return Path(rel)
        return Path(os.path.dirname(data_path)) / rel

    def _infer_view_scope(self, export_root: Path) -> str:
        if self.view_scope:
            return str(self.view_scope)
        parts = {p.lower() for p in export_root.parts}
        if "multiview" in parts:
            return "multiview"
        return "singleview"

    @staticmethod
    def _row_to_jsonable(row: dict[str, Any]) -> dict[str, Any]:
        if "metadata_json" in row or "messages_json" in row:
            return merged_row_to_upstream_record(row)
        return {k: v for k, v in row.items() if not str(k).startswith("_")}

    def _serialize_record(self, record: dict[str, Any]) -> str:
        payload = self._row_to_jsonable(record)
        return json.dumps(payload, ensure_ascii=False, default=str)

    def save_data(
        self,
        data_path: str,
        data: Optional[pd.DataFrame] = None,
        annotation_flag: bool = False,
        batch_size: int = 1000,
        keep_data_columns: Optional[list[str]] = None,
    ) -> None:
        if data is None:
            raise ValueError("Data to save is None")
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Only pandas DataFrame is supported")

        if self.export_bundle:
            export_root = self._resolve_export_root(data_path)
            summary = write_sharded_upstream_bundle(
                data,
                export_root,
                schema_version=self.schema_version,
                pipeline_run_id=self.pipeline_run_id,
                view_scope=self._infer_view_scope(export_root),
            )
            index_path = self._resolve_jsonl_output_path(data_path)
            os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
            with open(index_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(summary, ensure_ascii=False) + "\n")
            return

        out_path = self._resolve_jsonl_output_path(data_path)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for record in data.to_dict(orient="records"):
                f.write(self._serialize_record(record) + "\n")
