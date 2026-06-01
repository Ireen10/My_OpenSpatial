import json
import logging
import os
from glob import glob
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _read_all_jsonl(directory: str, suffix: str = ".jsonl", exclude_suffix: str = "_scenes.jsonl") -> List[dict]:
    """Read all JSONL files matching pattern from directory."""
    records = []
    files = sorted(glob(os.path.join(directory, f"*{suffix}")))
    for filepath in tqdm(files, desc="Reading per-image JSONL files", unit="file"):
        if exclude_suffix and filepath.endswith(exclude_suffix):
            continue
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return records


def _read_scene_jsonl(directory: str) -> List[dict]:
    """Read all per-scene JSONL files."""
    records = []
    files = sorted(glob(os.path.join(directory, "*_scenes.jsonl")))
    for filepath in tqdm(files, desc="Reading per-scene JSONL files", unit="file"):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return records


def _shard_to_parquet(df: pd.DataFrame, output_dir: str, batch_size: int, prefix: str = "data") -> List[str]:
    """Write DataFrame to sharded Parquet files."""
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for i in tqdm(
        range(0, len(df), batch_size),
        desc=f"Writing {prefix} parquet shards",
        unit="shard",
    ):
        chunk = df.iloc[i:i + batch_size]
        filename = f"{prefix}-{i // batch_size:05d}.parquet"
        filepath = os.path.join(output_dir, filename)
        chunk.to_parquet(filepath, index=False)
        paths.append(filepath)
    return paths


def export_to_parquet(
    input_dir: str,
    output_dir: str,
    batch_size: int = 3000,
    formats: Optional[List[str]] = None,
    hf_repo: Optional[str] = None,
) -> None:
    """Export JSONL files to Parquet, optionally push to HuggingFace.

    Args:
        input_dir: Directory containing JSONL files
        output_dir: Directory for Parquet output
        batch_size: Records per Parquet shard
        formats: List of formats to export: "per_image", "per_scene", or both
        hf_repo: Optional HuggingFace repo ID for upload
    """
    if formats is None:
        formats = ["per_image", "per_scene"]

    if "per_image" in formats:
        records = _read_all_jsonl(input_dir)
        if records:
            df = pd.DataFrame(records)
            per_image_dir = os.path.join(output_dir, "per_image")
            paths = _shard_to_parquet(df, per_image_dir, batch_size)
            logger.info("Exported %d per-image records to %d parquet files", len(df), len(paths))
        else:
            logger.warning("No per-image records found in %s", input_dir)

    if "per_scene" in formats:
        records = _read_scene_jsonl(input_dir)
        if records:
            df = pd.DataFrame(records)
            per_scene_dir = os.path.join(output_dir, "per_scene")
            paths = _shard_to_parquet(df, per_scene_dir, batch_size)
            logger.info("Exported %d per-scene records to %d parquet files", len(df), len(paths))
        else:
            logger.warning("No per-scene records found in %s", input_dir)

    if hf_repo:
        _push_to_huggingface(output_dir, hf_repo)


def _push_to_huggingface(output_dir: str, repo_id: str) -> None:
    """Upload Parquet files to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
        return

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        repo_type="dataset",
    )
    logger.info("Uploaded to https://huggingface.co/datasets/%s", repo_id)
