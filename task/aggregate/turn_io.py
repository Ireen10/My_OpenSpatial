"""Load annotation parquet rows and explode into turn records."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from .fingerprint import (
    compute_dedup_fingerprint,
    compute_merge_group_key,
    compute_question_core_key,
)


@dataclass
class TurnRecord:
    task_name: str
    row: dict
    turn: dict
    source_order: int
    turn_index: int
    question_core_key: str = ""
    dedup_fingerprint: str = ""
    merge_group_key: str = ""

    def enrich_keys(self) -> None:
        self.turn.setdefault("referent_mode", "legacy")
        if "prompt_struct" not in self.turn:
            logger.warning(
                "turn missing prompt_struct (legacy path): task=%s sub_task=%s turn_id=%s",
                self.task_name,
                self.turn.get("sub_task"),
                self.turn.get("turn_id"),
            )
        self.question_core_key = compute_question_core_key(self.turn)
        self.turn["question_core_key"] = self.question_core_key
        self.dedup_fingerprint = compute_dedup_fingerprint(self.task_name, self.row, self.turn)
        self.turn["dedup_fingerprint"] = self.dedup_fingerprint
        self.merge_group_key = compute_merge_group_key(self.row, self.turn)
        self.turn["merge_group_key"] = self.merge_group_key


def _message_to_qa(msg: list) -> tuple[str, str, Optional[str]]:
    if not msg or len(msg) < 2:
        return "", "", None
    human = next((m.get("value", "") for m in msg if m.get("from") == "human"), "")
    gpt = next((m.get("value", "") for m in msg if m.get("from") == "gpt"), "")
    prefix = None
    if "<image>" in human:
        parts = human.split("<image>", 1)
        rest = parts[-1].strip()
        if len(parts) > 1 and "Focal length" in rest:
            idx = rest.find("Predict ")
            if idx == -1:
                idx = rest.find("What ")
            if idx > 0:
                prefix = rest[:idx].strip()
                human = rest[idx:].strip()
            else:
                human = rest
        else:
            human = rest
    return human.strip(), gpt.strip(), prefix


def explode_row(row: dict, task_name: str, *, base_order: int) -> List[TurnRecord]:
    """One preprocess row → one or more TurnRecords."""
    meta = row.get("metadata")
    records: List[TurnRecord] = []

    if isinstance(meta, list) and meta:
        meta = meta[0]
    if isinstance(meta, dict) and meta.get("turns"):
        for i, turn in enumerate(meta["turns"]):
            tr = dict(turn)
            records.append(TurnRecord(
                task_name=task_name,
                row=row,
                turn=tr,
                source_order=base_order,
                turn_index=i,
            ))
        return records

    messages = row.get("messages") or []
    qtypes = row.get("question_types") or []
    tags = row.get("question_tags") or []

    convs = messages if (messages and isinstance(messages[0], list)) else [messages]

    for i, conv in enumerate(convs):
        if not conv:
            continue
        q_text, a_text, prefix = _message_to_qa(conv)
        sub = "unknown"
        if isinstance(tags, list) and i < len(tags):
            tag0 = tags[i]
            sub = tag0[0] if isinstance(tag0, list) and tag0 else str(tag0)
        qtype = "OE"
        if isinstance(qtypes, list) and i < len(qtypes):
            qt = qtypes[i]
            qtype = "MCQ" if str(qt).upper() in ("MCQ",) or "mcq" in str(qt).lower() else "OE"

        tr = {
            "turn_id": i,
            "task_name": task_name,
            "sub_task": sub,
            "question_type": qtype,
            "instruction_mode": "legacy",
            "referent_mode": "legacy",
            "question_text": q_text,
            "answer_text": a_text,
            "image_placeholder_count": 1,
        }
        if prefix:
            tr["question_prefix"] = prefix
        records.append(TurnRecord(
            task_name=task_name,
            row=row,
            turn=tr,
            source_order=base_order,
            turn_index=i,
        ))
    return records


def load_turns_from_parquet(df, task_name: str) -> List[TurnRecord]:
    out: List[TurnRecord] = []
    for idx in range(len(df)):
        row = df.iloc[idx].to_dict()
        out.extend(explode_row(row, task_name, base_order=idx * 1000))
    for rec in out:
        rec.enrich_keys()
    return out
