#!/usr/bin/env python3
"""
M4 L2 spot-check: merged sample with distance + 3d_grounding (chair/table) turns.

Usage:
  python verification/dataset_pipeline/spotcheck_merge_grounding.py \\
    --parquet E:/GitRepo/OpenSpatial/output/frame_rot/base_pipeline_demo_aggregate_singleview/aggregate_stage/merged_samples/data.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_row(row) -> dict:
    if "metadata_json" in row.index:
        meta = json.loads(row["metadata_json"])
        msgs = json.loads(row["messages_json"])
    else:
        meta = row.get("metadata")
        msgs = row.get("messages")
    if isinstance(meta, list) and meta:
        meta = meta[0]
    return {"metadata": meta, "messages": msgs}


def find_chair_table_merged(parquet: Path, limit: int = 5) -> list:
    df = pd.read_parquet(parquet)
    hits = []
    for idx in range(len(df)):
        rec = _load_row(df.iloc[idx])
        meta = rec.get("metadata") or {}
        prov = meta.get("provenance") or {}
        tasks = set(prov.get("source_tasks") or [])
        if not {"distance", "3d_grounding"}.issubset(tasks):
            continue
        text = json.dumps(rec, ensure_ascii=False).lower()
        if "chair" not in text or "table" not in text:
            continue
        hits.append({
            "row": idx,
            "sample_id": df.iloc[idx].get("sample_id"),
            "source_tasks": sorted(tasks),
            "turn_count": prov.get("turn_count"),
            "merge_group_key": df.iloc[idx].get("merge_group_key"),
        })
        if len(hits) >= limit:
            break
    return hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    args = parser.parse_args()
    if not args.parquet.is_file():
        print(f"FAIL: missing {args.parquet}")
        return 1
    hits = find_chair_table_merged(args.parquet)
    if not hits:
        print("WARN: no merged sample with distance+3d_grounding and chair/table in text")
        print("      (pilot may lack both tasks on same visual group — check aggregate inputs)")
        return 0
    print(f"PASS: found {len(hits)} chair/table merged sample(s):")
    for h in hits:
        print(f"  row {h['row']} sample_id={h['sample_id']} turns={h['turn_count']} tasks={h['source_tasks']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
