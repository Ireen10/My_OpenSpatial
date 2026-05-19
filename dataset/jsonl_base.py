"""JSONL-backed dataset hook for OpenSpatial pipelines (stub — customize load/save)."""

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

# Pipeline always calls save_data(..., "data.parquet"); this hook writes JSONL instead.
PIPELINE_SAVE_BASENAME = "data.jsonl"


class JsonlBaseDataset(ImageBaseDataset):
    """
    Pipeline dataset hook: in-memory table remains a pandas DataFrame; persistence is JSONL.

    YAML:
        dataset:
          modality: image
          input_dataset_name: image_base
          output_dataset_name: jsonl_base
          data_dir: /path/to/input.parquet
          output_path: data.jsonl             # optional; default derived from save path
    """

    MODALITY = "image"

    def __init__(self, cfg, *, _skip_load=False):
        if not cfg.data_dir:
            raise ValueError("cfg.data_dir is required")
        self.cfg = cfg
        self.data_dir = cfg.data_dir
        self.raw_data_root = getattr(cfg, "raw_data_root", None)
        if self.raw_data_root:
            self.raw_data_root = os.path.abspath(self.raw_data_root)
        self.output_path = getattr(cfg, "output_path", None) or PIPELINE_SAVE_BASENAME
        self.data = None if _skip_load else self._load()

    # ------------------------------------------------------------------
    # Load (stub — override in subclass or edit here)
    # ------------------------------------------------------------------

    def _load(self) -> pd.DataFrame:
        return self._load_from_path(self.data_dir)

    def _load_from_path(self, path: str) -> pd.DataFrame:
        """Load records into a DataFrame. Extend for your on-disk JSONL schema."""
        path = str(path)
        if HF_REPO_PATTERN.match(path):
            raise NotImplementedError(
                "JsonlBaseDataset does not load HuggingFace Hub repos yet."
            )

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(path)

        if p.suffix.lower() == ".jsonl":
            df = self._read_jsonl(p)
        elif p.suffix.lower() == ".parquet":
            # Convenience: reuse preprocessing parquet until custom JSONL input is ready.
            df = pd.read_parquet(p, engine="pyarrow", dtype_backend="pyarrow")
        else:
            raise NotImplementedError(
                f"JsonlBaseDataset._load_from_path: unsupported input '{path}'. "
                "Implement .jsonl (or add formats here)."
            )

        return self._apply_raw_data_root(df)

    def _read_jsonl(self, path: Path) -> pd.DataFrame:
        """Parse JSONL file into a DataFrame (stub — customize field parsing)."""
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return pd.DataFrame(records)

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

    # ------------------------------------------------------------------
    # Save (stub — override serialization here)
    # ------------------------------------------------------------------

    def _resolve_jsonl_output_path(self, data_path: str) -> str:
        """Map pipeline's data.parquet path to JSONL output path."""
        if self.output_path and os.path.isabs(self.output_path):
            return self.output_path
        if self.output_path and "/" not in self.output_path and "\\" not in self.output_path:
            return os.path.join(os.path.dirname(data_path), self.output_path)
        if data_path.endswith(".parquet"):
            return data_path[: -len(".parquet")] + ".jsonl"
        return data_path if data_path.endswith(".jsonl") else data_path + ".jsonl"

    @staticmethod
    def _row_to_jsonable(row: dict[str, Any]) -> dict[str, Any]:
        """Convert one DataFrame row to a JSON-serializable dict (stub)."""
        # TODO: custom encoding (bytes, numpy, PIL, nested structures, etc.)
        return row

    def _serialize_record(self, record: dict[str, Any]) -> str:
        """Serialize one record to a single JSONL line (stub)."""
        # TODO: user-defined JSONL schema / field filtering
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
        """
        Write pipeline output as JSONL instead of Parquet.

        annotation_flag / batch_size / keep_data_columns: reserved for parity with
        ImageBaseDataset (implement if you need flattening or sharded JSONL).
        """
        if data is None:
            raise ValueError("Data to save is None")
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Only pandas DataFrame is supported")

        out_path = self._resolve_jsonl_output_path(data_path)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        # TODO: honor annotation_flag via flatten_annotations + sharded writers
        with open(out_path, "w", encoding="utf-8") as f:
            for record in data.to_dict(orient="records"):
                f.write(self._serialize_record(record) + "\n")
