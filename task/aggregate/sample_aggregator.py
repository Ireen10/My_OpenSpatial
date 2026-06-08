"""
Aggregate annotation outputs: per-task dedup then merge by visual input group.

Plan §4.2 — stable order within non-grounding turns; 3D grounding turns pinned first.
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
from utils.parquet_io import load_parquet_dataframe, resolve_task_output_dir
from task.aggregate.fingerprint import image_refs_from_row, pick_dedup_winner
from task.aggregate.turn_io import (
    TurnRecord,
    _conversations_from_row,
    _message_to_qa,
    load_turns_from_parquet,
)

SCHEMA_VERSION = "1.1"


def _is_grounding_turn(rec: TurnRecord) -> bool:
    """True for 3D grounding annotation turns (task / sub_task / template)."""
    tn = (rec.task_name or "").lower()
    if "grounding" in tn:
        return True
    turn = rec.turn or {}
    st = (turn.get("sub_task") or "").lower()
    if st.startswith("grounding"):
        return True
    ps = turn.get("prompt_struct") if isinstance(turn.get("prompt_struct"), dict) else {}
    tid = (ps.get("template_id") or ps.get("template_family") or "").lower()
    return tid.startswith("grounding")


def _turn_merge_sort_key(rec: TurnRecord) -> tuple:
    """Grounding first, then original parquet order."""
    return (0 if _is_grounding_turn(rec) else 1, rec.source_order, rec.turn_index)


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
        self.input_tasks_prefix = (args.get("input_tasks_prefix") or "").strip()
        self.dedup_within_task = bool(args.get("dedup_within_task", True))
        self.merge_by_visual_input_group = bool(args.get("merge_by_visual_input_group", True))
        self.dedup_keep_policy = args.get("dedup_keep_policy", "semantic_first")

    def _resolve_task_ref(self, task_ref: str) -> str:
        ref = str(task_ref).strip().replace("\\", "/")
        if (
            self.input_tasks_prefix
            and not os.path.isabs(ref)
            and not ref.startswith("..")
            and "/" not in ref
            and not ref.endswith(".parquet")
        ):
            prefix = self.input_tasks_prefix.replace("\\", "/").rstrip("/")
            ref = f"{prefix}/{ref}"
        return ref

    def _resolve_task_output(self, task_ref: str) -> str:
        ref = self._resolve_task_ref(task_ref)
        return resolve_task_output_dir(self.output_root, ref)

    def _load_all_turns(self, dataset: pd.DataFrame) -> List[TurnRecord]:
        records: List[TurnRecord] = []
        if self.input_tasks:
            for ref in self.input_tasks:
                loc = self._resolve_task_output(ref)
                task_name = ref.split("/")[-1].replace(".parquet", "")
                if "/" in ref:
                    task_name = ref.strip("/").split("/")[-1]
                df = load_parquet_dataframe(loc)
                records.extend(load_turns_from_parquet(df, task_name))
            return records

        task_name = self.args.get("task_name", "annotation")
        return load_turns_from_parquet(dataset, task_name)

    @staticmethod
    def _turn_match_signature(turn: dict) -> tuple:
        ps = turn.get("prompt_struct") if isinstance(turn.get("prompt_struct"), dict) else {}
        return (
            turn.get("sub_task"),
            turn.get("question_type"),
            ps.get("template_id"),
        )

    @classmethod
    def _pick_dedup_source_record(
        cls, group: List[TurnRecord], winner_turn: dict,
    ) -> TurnRecord:
        sig = cls._turn_match_signature(winner_turn)
        for rec in group:
            if cls._turn_match_signature(rec.turn) == sig:
                return rec
        return group[0]

    @staticmethod
    def _refresh_viz_from_row(rec: TurnRecord) -> None:
        convs = _conversations_from_row(rec.row)
        idx = int(rec.turn_index)
        if idx < 0 or idx >= len(convs):
            return
        vq, va, vp, vn = _message_to_qa(convs[idx])
        if vq:
            rec.viz_question = vq
        if va:
            rec.viz_answer = va
        if vp:
            rec.viz_prefix = vp
        rec.viz_n_images = vn

    @classmethod
    def dedup_turns(cls, records: List[TurnRecord], policy: str) -> tuple[List[TurnRecord], int]:
        by_fp: Dict[str, List[TurnRecord]] = defaultdict(list)
        for rec in records:
            by_fp[rec.dedup_fingerprint].append(rec)

        kept: List[TurnRecord] = []
        removed = 0
        for group in by_fp.values():
            if len(group) == 1:
                kept.append(group[0])
            else:
                winner_turn = pick_dedup_winner([g.turn for g in group], policy=policy)
                src = cls._pick_dedup_source_record(group, winner_turn)
                kept.append(TurnRecord(
                    task_name=group[0].task_name,
                    row=src.row,
                    turn=winner_turn,
                    source_order=min(g.source_order for g in group),
                    turn_index=src.turn_index,
                    viz_question=src.viz_question,
                    viz_answer=src.viz_answer,
                    viz_prefix=src.viz_prefix,
                    viz_n_images=src.viz_n_images,
                ))
                kept[-1].enrich_keys()
                if not (kept[-1].viz_question or kept[-1].viz_answer):
                    cls._refresh_viz_from_row(kept[-1])
                removed += len(group) - 1
        return kept, removed

    @staticmethod
    def merge_turns(records: List[TurnRecord]) -> List[dict]:
        groups: Dict[str, List[TurnRecord]] = defaultdict(list)
        for rec in records:
            groups[rec.merge_group_key].append(rec)

        samples: List[dict] = []
        for merge_key, group in groups.items():
            group.sort(key=_turn_merge_sort_key)
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
        """Concatenate per-turn visualization Q/A (from annotation messages)."""
        out: List[dict] = []
        for rec in group:
            if not (rec.viz_question or rec.viz_answer):
                SampleAggregator._refresh_viz_from_row(rec)
            n_img = max(1, rec.viz_n_images or int(rec.turn.get("image_placeholder_count") or 1))
            prefix = (rec.viz_prefix or rec.turn.get("question_prefix") or "").strip()
            body = (rec.viz_question or "").strip()
            if prefix and body.startswith(prefix):
                body = body[len(prefix):].lstrip()
            q = body
            if prefix and not body.startswith(prefix):
                q = f"{prefix}\n\n{body}" if body else prefix
            q = " ".join(["<image>"] * n_img) + " " + q
            out.append({"from": "human", "value": q.strip()})
            out.append({"from": "gpt", "value": (rec.viz_answer or "").strip()})
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
