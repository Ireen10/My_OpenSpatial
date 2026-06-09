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
    viz_question: str = ""
    viz_answer: str = ""
    viz_prefix: Optional[str] = field(default=None)
    viz_n_images: int = 1

    def enrich_keys(self) -> None:
        if "prompt_struct" not in self.turn:
            logger.warning(
                "turn missing prompt_struct: task=%s sub_task=%s turn_id=%s",
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


def _message_to_qa(msg: list) -> tuple[str, str, Optional[str], int]:
    if not msg or len(msg) < 2:
        return "", "", None, 1
    human = next((m.get("value", "") for m in msg if m.get("from") == "human"), "")
    gpt = next((m.get("value", "") for m in msg if m.get("from") == "gpt"), "")
    n_img = max(1, human.lower().count("<image>"))
    prefix = None
    if "<image>" in human.lower():
        import re
        rest = re.sub(r"<image>\s*", "", human, flags=re.I).strip()
        if "Focal length" in rest:
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
    return human.strip(), gpt.strip(), prefix, n_img


def _conversations_from_row(row: dict) -> List[list]:
    messages = row.get("messages")
    if messages is None:
        messages = []
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    if not messages:
        return []
    if messages and isinstance(messages[0], dict):
        return [messages]
    if messages and isinstance(messages[0], list):
        return messages
    return []


def explode_row(row: dict, task_name: str, *, base_order: int) -> List[TurnRecord]:
    """One preprocess row → one or more TurnRecords."""
    meta = row.get("metadata")
    records: List[TurnRecord] = []
    convs = _conversations_from_row(row)

    if isinstance(meta, list) and meta:
        convs = _conversations_from_row(row)
        for i, ent in enumerate(meta):
            if not isinstance(ent, dict):
                continue
            ent_turns = ent.get("turns") or []
            tr = dict(ent_turns[0]) if ent_turns and isinstance(ent_turns[0], dict) else {}
            if ent.get("mark_spec"):
                tr["mark_spec"] = ent["mark_spec"]
            vq, va, vp, vn = ("", "", None, int(tr.get("image_placeholder_count") or 1))
            if i < len(convs):
                vq, va, vp, vn = _message_to_qa(convs[i])
            records.append(TurnRecord(
                task_name=task_name,
                row=row,
                turn=tr,
                source_order=base_order,
                turn_index=i,
                viz_question=vq,
                viz_answer=va,
                viz_prefix=vp,
                viz_n_images=vn,
            ))
        if records:
            return records
        meta = meta[0]
    if isinstance(meta, dict) and meta.get("turns"):
        for i, turn in enumerate(meta["turns"]):
            tr = dict(turn)
            vq, va, vp, vn = "", "", None, int(tr.get("image_placeholder_count") or 1)
            if i < len(convs):
                vq, va, vp, vn = _message_to_qa(convs[i])
            records.append(TurnRecord(
                task_name=task_name,
                row=row,
                turn=tr,
                source_order=base_order,
                turn_index=i,
                viz_question=vq,
                viz_answer=va,
                viz_prefix=vp,
                viz_n_images=vn,
            ))
        return records

    qtypes = row.get("question_types") or []
    tags = row.get("question_tags") or []

    for i, conv in enumerate(convs):
        if not conv:
            continue
        vq, va, vp, vn = _message_to_qa(conv)
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
            "image_placeholder_count": vn,
        }
        if vp:
            tr["question_prefix"] = vp
        records.append(TurnRecord(
            task_name=task_name,
            row=row,
            turn=tr,
            source_order=base_order,
            turn_index=i,
            viz_question=vq,
            viz_answer=va,
            viz_prefix=vp,
            viz_n_images=vn,
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
