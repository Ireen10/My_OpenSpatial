"""
Aggregate annotation outputs: per-task dedup then merge by visual input group.

Plan §4.2 — no turn semantic reordering (stable source order only).
"""

from __future__ import annotations

import json
import os
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from task.base_task import BaseTask
from task.aggregate.fingerprint import image_refs_from_row, pick_dedup_winner
from task.aggregate.turn_io import TurnRecord, load_turns_from_parquet

SCHEMA_VERSION = "1.1"


def _json_safe(obj: Any) -> Any:
    """Coerce numpy scalars/arrays for parquet nested structs."""
    if obj is None or (isinstance(obj, float) and pd.isna(obj)):
        return None
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "tolist") and not isinstance(obj, (str, bytes)):
        try:
            return _json_safe(obj.tolist())
        except Exception:
            pass
    return obj


def _json_dumps(obj: Any) -> str:
    return json.dumps(_json_safe(obj), ensure_ascii=False)


class SampleAggregator(BaseTask):
    """M4 + M5: dedup within task_name, merge by merge_group_key."""

    def __init__(self, args):
        super().__init__(args)
        self.output_root = args.get("output_root") or args.get("output_dir", ".")
        self.input_tasks = args.get("input_tasks") or []
        self.dedup_within_task = bool(args.get("dedup_within_task", True))
        self.merge_by_visual_input_group = bool(args.get("merge_by_visual_input_group", True))
        self.dedup_keep_policy = args.get("dedup_keep_policy", "semantic_first")

    def _resolve_task_parquet(self, task_ref: str) -> str:
        if os.path.isfile(task_ref):
            return task_ref
        path = os.path.join(self.output_root, task_ref, "data.parquet")
        if os.path.isfile(path):
            return path
        raise FileNotFoundError(f"Cannot resolve input task parquet: {task_ref} -> {path}")

    def _load_all_turns(self, dataset: pd.DataFrame) -> List[TurnRecord]:
        records: List[TurnRecord] = []
        if self.input_tasks:
            for ref in self.input_tasks:
                path = self._resolve_task_parquet(ref)
                task_name = ref.split("/")[-1].replace(".parquet", "")
                if "/" in ref:
                    task_name = ref.strip("/").split("/")[-1]
                df = pd.read_parquet(path)
                records.extend(load_turns_from_parquet(df, task_name))
            return records

        task_name = self.args.get("task_name", "annotation")
        return load_turns_from_parquet(dataset, task_name)

    @staticmethod
    def dedup_turns(records: List[TurnRecord], policy: str) -> tuple[List[TurnRecord], int]:
        by_fp: Dict[str, List[TurnRecord]] = defaultdict(list)
        for rec in records:
            by_fp[rec.dedup_fingerprint].append(rec)

        kept: List[TurnRecord] = []
        removed = 0
        for group in by_fp.values():
            if len(group) == 1:
                kept.append(group[0])
            else:
                kept.append(TurnRecord(
                    task_name=group[0].task_name,
                    row=group[0].row,
                    turn=pick_dedup_winner([g.turn for g in group], policy=policy),
                    source_order=min(g.source_order for g in group),
                    turn_index=min(g.turn_index for g in group),
                ))
                kept[-1].enrich_keys()
                removed += len(group) - 1
        return kept, removed

    @staticmethod
    def merge_turns(records: List[TurnRecord]) -> List[dict]:
        groups: Dict[str, List[TurnRecord]] = defaultdict(list)
        for rec in records:
            groups[rec.merge_group_key].append(rec)

        samples: List[dict] = []
        for merge_key, group in groups.items():
            group.sort(key=lambda r: (r.source_order, r.turn_index))
            row0 = group[0].row
            image_refs = image_refs_from_row(row0)
            turns = []
            for i, rec in enumerate(group):
                t = {k: v for k, v in rec.turn.items() if k != "mark_spec"}
                t["turn_id"] = i
                turns.append(t)

            messages_flat = SampleAggregator._turns_to_messages(group)
            meta = row0.get("metadata") if isinstance(row0.get("metadata"), dict) else {}
            mark_spec = None
            for rec in group:
                ms = rec.turn.get("mark_spec") or (meta.get("mark_spec") if isinstance(meta, dict) else None)
                from task.aggregate.fingerprint import mark_spec_has_slots
                if mark_spec_has_slots(ms):
                    mark_spec = ms
                    break

            visual_anchor = meta.get("visual_anchor") if isinstance(meta, dict) else {}
            if not visual_anchor:
                from task.annotation.core.sample_metadata import build_visual_anchor
                visual_anchor = build_visual_anchor(row0)

            source_tasks = sorted({rec.task_name for rec in group})
            samples.append({
                "schema_version": SCHEMA_VERSION,
                "sample_id": str(uuid.uuid4()),
                "merge_group_key": merge_key,
                "image_refs": image_refs,
                "messages": [messages_flat] if messages_flat else [],
                "metadata": {
                    "visual_anchor": visual_anchor,
                    "mark_spec": mark_spec,
                    "turns": turns,
                    "provenance": {
                        "source_tasks": source_tasks,
                        "merged": len(group) > 1 or len(source_tasks) > 1,
                        "turn_count": len(turns),
                    },
                },
            })
        return samples

    @staticmethod
    def _turns_to_messages(group: List[TurnRecord]) -> List[dict]:
        """Stable source order → flat human/gpt list; each turn keeps its own <image> tags."""
        out: List[dict] = []
        for rec in group:
            t = rec.turn
            n_img = max(1, int(t.get("image_placeholder_count") or 1))
            q = (t.get("question_text") or "").strip()
            prefix = (t.get("question_prefix") or "").strip()
            if prefix:
                q = f"{prefix}\n\n{q}" if q else prefix
            q = " ".join(["<image>"] * n_img) + " " + q
            out.append({"from": "human", "value": q})
            out.append({"from": "gpt", "value": t.get("answer_text") or ""})
        return out

    def run(self, dataset: pd.DataFrame) -> pd.DataFrame:
        records = self._load_all_turns(dataset)
        stats = {"turns_in": len(records), "dedup_removed": 0, "samples_out": 0}

        if self.dedup_within_task:
            by_task: Dict[str, List[TurnRecord]] = defaultdict(list)
            for rec in records:
                by_task[rec.task_name].append(rec)
            deduped: List[TurnRecord] = []
            for task_recs in by_task.values():
                kept, n = self.dedup_turns(task_recs, self.dedup_keep_policy)
                deduped.extend(kept)
                stats["dedup_removed"] += n
            records = deduped

        if self.merge_by_visual_input_group:
            samples = self.merge_turns(records)
        else:
            samples = self.merge_turns(
                [TurnRecord(
                    task_name=r.task_name, row=r.row, turn=r.turn,
                    source_order=r.source_order, turn_index=r.turn_index,
                    question_core_key=r.question_core_key,
                    dedup_fingerprint=r.dedup_fingerprint,
                    merge_group_key=r.merge_group_key + f":{r.dedup_fingerprint}",
                ) for r in records]
            )

        stats["samples_out"] = len(samples)
        stats["turns_out"] = sum(s["metadata"]["provenance"]["turn_count"] for s in samples)
        print(f">>> Aggregate stats: {stats}")
        rows = []
        for s in samples:
            meta = _json_safe(s.get("metadata"))
            msgs = _json_safe(s.get("messages"))
            rows.append({
                "schema_version": s["schema_version"],
                "sample_id": s["sample_id"],
                "merge_group_key": s["merge_group_key"],
                "image_refs": s["image_refs"],
                "messages_json": _json_dumps(msgs),
                "metadata_json": _json_dumps(meta),
            })
        return pd.DataFrame(rows)
