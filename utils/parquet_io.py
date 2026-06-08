"""Load pipeline parquet output (single file or all *.parquet in a directory)."""

from __future__ import annotations

import glob
import os
from typing import List

import pandas as pd


def list_parquet_shards(location: str) -> List[str]:
    """List parquet file(s): a file path, or every ``*.parquet`` in a directory."""
    location = os.path.normpath(str(location))
    if os.path.isfile(location):
        return [location]

    if not os.path.isdir(location):
        raise FileNotFoundError(f"Parquet location not found: {location}")

    files = sorted(glob.glob(os.path.join(location, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No .parquet files under {location}")
    return files


def _is_chunked_nested_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "chunked array" in msg or "nested data conversions" in msg


def _read_parquet(path: str) -> pd.DataFrame:
    """Read one parquet file; fall back when PyArrow chokes on nested chunked columns."""
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        if not _is_chunked_nested_error(exc):
            raise

    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    parts = [pf.read_row_group(i).to_pandas() for i in range(pf.num_row_groups)]
    if not parts:
        return pd.DataFrame()
    if len(parts) == 1:
        return parts[0]
    return pd.concat(parts, ignore_index=True)


def load_parquet_dataframe(location: str) -> pd.DataFrame:
    """Load parquet from a file or concatenate all parquets in a directory."""
    shards = list_parquet_shards(location)
    if len(shards) == 1:
        return _read_parquet(shards[0])
    return pd.concat([_read_parquet(p) for p in shards], ignore_index=True)


def resolve_task_output_dir(output_root: str, task_ref: str) -> str:
    """Resolve stage/task ref or relative path to an on-disk output directory."""
    ref = str(task_ref).strip().replace("\\", "/")
    if os.path.isfile(ref):
        return os.path.dirname(ref)
    if os.path.isdir(ref):
        return os.path.normpath(ref)
    candidate = os.path.normpath(os.path.join(output_root, ref))
    list_parquet_shards(candidate)
    return candidate
