"""
Aggregate annotation outputs: per-task dedup then merge by visual input group.

Plan §4.2 — stable order within non-grounding turns; 3D grounding turns pinned first.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from task.base_task import BaseTask
from utils.parquet_io import load_parquet_dataframe, resolve_task_output_dir
from task.aggregate.fingerprint import image_refs_from_row, mark_spec_norm
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


def _iter_mark_slots(mark_spec: Optional[dict]) -> List[tuple[int, dict]]:
    mark_spec = _json_safe(mark_spec)
    if not isinstance(mark_spec, dict):
        return []
    out: List[tuple[int, dict]] = []
    views = mark_spec.get("views")
    if isinstance(views, list):
        for v in views:
            if not isinstance(v, dict):
                continue
            try:
                vi = int(v.get("view_index", 0))
            except (TypeError, ValueError):
                vi = 0
            for slot in v.get("slots") or []:
                if isinstance(slot, dict):
                    out.append((vi, slot))
        return out
    for slot in mark_spec.get("slots") or []:
        if not isinstance(slot, dict):
            continue
        try:
            vi = int(slot.get("view_index", 0))
        except (TypeError, ValueError):
            vi = 0
        out.append((vi, slot))
    return out


def _slot_identity(view_index: int, slot: dict) -> str:
    geom = slot.get("geometry") or {}
    if not isinstance(geom, dict):
        geom = {}
    box = geom.get("box_2d")
    uv = geom.get("uv")
    body = {
        "view_index": int(view_index),
        "slot_id": slot.get("slot_id"),
        "obj_idx": slot.get("obj_idx"),
        "tag": slot.get("tag"),
        "mark_kind": slot.get("mark_kind"),
        "geometry": {
            "box_2d": [round(float(v), 4) for v in box] if box is not None else None,
            "uv": [int(uv[0]), int(uv[1])] if uv is not None else None,
            "mask_ref": geom.get("mask_ref") if isinstance(geom.get("mask_ref"), dict) else None,
        },
    }
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _mark_surface(slot: dict) -> Optional[str]:
    tag = str(slot.get("tag") or "").strip()
    color = str(slot.get("color_name") or "").strip()
    kind = str(slot.get("mark_kind") or "box").strip()
    if not tag or not color or not kind:
        return None
    return f"{tag}-({color} {kind})"


def _mark_surface_replacements(
    source_mark_spec: Optional[dict],
    target_mark_spec: Optional[dict],
) -> List[tuple[str, str]]:
    target_by_identity = {
        _slot_identity(vi, slot): slot
        for vi, slot in _iter_mark_slots(target_mark_spec)
    }
    replacements: List[tuple[str, str]] = []
    for vi, source_slot in _iter_mark_slots(source_mark_spec):
        target_slot = target_by_identity.get(_slot_identity(vi, source_slot))
        if not target_slot:
            continue
        old = _mark_surface(source_slot)
        new = _mark_surface(target_slot)
        if old and new and old != new:
            replacements.append((old, new))
    return replacements


def _replace_mark_surfaces(value: Any, replacements: List[tuple[str, str]]) -> Any:
    if isinstance(value, str):
        out = value
        for old, new in replacements:
            out = re.sub(re.escape(old), new, out, flags=re.IGNORECASE)
        return out
    if isinstance(value, dict):
        return {k: _replace_mark_surfaces(v, replacements) for k, v in value.items()}
    if isinstance(value, list):
        return [_replace_mark_surfaces(v, replacements) for v in value]
    return value


class SampleAggregator(BaseTask):
    """M4 + M5: dedup within task_name, merge by merge_group_key."""

    def __init__(self, args):
        super().__init__(args)
        self.output_root = args.get("output_root") or args.get("output_dir", ".")
        self.input_tasks = args.get("input_tasks") or []
        self.input_tasks_prefix = (args.get("input_tasks_prefix") or "").strip()
        self.dedup_within_task = bool(args.get("dedup_within_task", True))
        self.merge_by_visual_input_group = bool(args.get("merge_by_visual_input_group", True))
        self.dedup_keep_policy = args.get("dedup_keep_policy", "")

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
            mark_spec = SampleAggregator._sample_mark_spec_for_group(group)
            turns = []
            for i, rec in enumerate(group):
                replacements = _mark_surface_replacements(
                    SampleAggregator._record_mark_spec(rec), mark_spec,
                )
                t = {
                    k: _replace_mark_surfaces(v, replacements)
                    for k, v in rec.turn.items()
                    if k != "mark_spec"
                }
                t["turn_id"] = i
                turns.append(t)

            messages_flat = SampleAggregator._turns_to_messages(group, mark_spec)
            meta = row0.get("metadata") if isinstance(row0.get("metadata"), dict) else {}

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
    def _record_mark_spec(rec: TurnRecord) -> Optional[dict]:
        if isinstance(rec.turn, dict) and isinstance(rec.turn.get("mark_spec"), dict):
            return rec.turn["mark_spec"]
        meta = rec.row.get("metadata") if isinstance(rec.row.get("metadata"), dict) else {}
        ms = meta.get("mark_spec") if isinstance(meta, dict) else None
        return ms if isinstance(ms, dict) else None

    @staticmethod
    def _sample_mark_spec_for_group(group: List[TurnRecord]) -> Optional[dict]:
        """Return the unique sample-level mark intent for this group, ignoring color."""
        chosen: Optional[dict] = None
        seen: Dict[str, dict] = {}
        for rec in group:
            ms = SampleAggregator._record_mark_spec(rec)
            norm = mark_spec_norm(ms)
            if norm is None:
                continue
            key = json.dumps(norm, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            seen.setdefault(key, ms)
            if chosen is None:
                chosen = ms
        if len(seen) > 1:
            details = [
                f"{rec.task_name}[row={rec.source_order},turn={rec.turn_index}]"
                for rec in group
            ]
            raise ValueError(
                "Merge group contains multiple distinct mark intents; "
                f"merge_group_key={group[0].merge_group_key}, records={details}"
            )
        return chosen

    @staticmethod
    def _turns_to_messages(
        group: List[TurnRecord],
        sample_mark_spec: Optional[dict] = None,
    ) -> List[dict]:
        """Concatenate per-turn visualization Q/A (from annotation messages)."""
        out: List[dict] = []
        for rec in group:
            if not (rec.viz_question or rec.viz_answer):
                SampleAggregator._refresh_viz_from_row(rec)
            replacements = _mark_surface_replacements(
                SampleAggregator._record_mark_spec(rec), sample_mark_spec,
            )
            n_img = max(1, rec.viz_n_images or int(rec.turn.get("image_placeholder_count") or 1))
            prefix = (rec.viz_prefix or rec.turn.get("question_prefix") or "").strip()
            body = _replace_mark_surfaces((rec.viz_question or "").strip(), replacements)
            if prefix and body.startswith(prefix):
                body = body[len(prefix):].lstrip()
            q = body
            if prefix and not body.startswith(prefix):
                q = f"{prefix}\n\n{body}" if body else prefix
            q = " ".join(["<image>"] * n_img) + " " + q
            out.append({"from": "human", "value": q.strip()})
            answer = _replace_mark_surfaces((rec.viz_answer or "").strip(), replacements)
            out.append({"from": "gpt", "value": answer})
        return out

    def run(self, dataset: pd.DataFrame) -> pd.DataFrame:
        records = self._load_all_turns(dataset)
        stats = {"turns_in": len(records), "dedup_removed": 0, "samples_out": 0}

        if self.dedup_within_task:
            print(
                ">>> Aggregate notice: dedup_within_task is ignored. "
                "Annotation tasks are responsible for avoiding duplicate turns."
            )

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
