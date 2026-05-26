"""
Audit and optionally fix image / template placeholders in messages (plan §4.5, M6).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from task.base_task import BaseTask

IMAGE_TAG_RE = re.compile(r"<image>\s*", re.I)
IMAGE_TAG_LEAD_RE = re.compile(r"^\s*<image>\s*", re.I)
TEMPLATE_PLACEHOLDER_RE = re.compile(r"\[([A-Z])\]")


def _flatten_messages(messages: Any) -> List[dict]:
    if not messages:
        return []
    if isinstance(messages, list) and messages and isinstance(messages[0], dict):
        if messages[0].get("from") in ("human", "gpt"):
            return list(messages)
    out: List[dict] = []
    if isinstance(messages, list):
        for conv in messages:
            if isinstance(conv, list):
                for m in conv:
                    if isinstance(m, dict):
                        out.append(m)
            elif isinstance(conv, dict):
                out.append(conv)
    return out


def _expected_image_count(meta: Optional[dict], messages: List[dict]) -> int:
    """Total <image> tags expected across all human turns (sum of per-turn counts)."""
    return sum(_expected_counts_per_human(meta, messages))


def _expected_counts_per_human(meta: Optional[dict], messages: List[dict]) -> List[int]:
    """One expected <image> count per human message, in order."""
    turns = []
    if isinstance(meta, dict):
        turns = meta.get("turns") or []
    expected: List[int] = []
    human_i = 0
    for msg in messages:
        if msg.get("from") != "human":
            continue
        n = 1
        if human_i < len(turns) and isinstance(turns[human_i], dict):
            n = max(0, int(turns[human_i].get("image_placeholder_count") or 1))
        expected.append(n)
        human_i += 1
    if not expected:
        expected = [1]
    return expected


def _image_tags_at_front(value: str, n_expected: int) -> bool:
    if n_expected <= 0:
        return True
    rest = (value or "").strip()
    for _ in range(n_expected):
        m = IMAGE_TAG_LEAD_RE.match(rest)
        if not m:
            return False
        rest = rest[m.end() :]
    return True


def _normalize_human_images(
    value: str,
    n_expected: int,
    *,
    question_prefix: str = "",
    is_first_human: bool,
) -> str:
    body = IMAGE_TAG_RE.sub("", value).strip()
    if n_expected <= 0:
        return body
    img_prefix = " ".join(["<image>"] * n_expected) + " "
    if is_first_human and question_prefix:
        qp = question_prefix.strip()
        if body.startswith(qp):
            rest = body[len(qp):].lstrip()
            return img_prefix + qp + ("\n\n" if rest else "") + rest
        return img_prefix + qp + ("\n\n" if body else "") + body
    return img_prefix + body


def audit_row(
    row: dict,
    *,
    fix: bool = False,
) -> Tuple[dict, List[dict], List[str]]:
    """
    Returns (placeholder_audit, messages possibly fixed, errors).
    """
    errors: List[str] = []
    meta = row.get("metadata")
    if isinstance(meta, list) and meta:
        meta = meta[0]
    messages = _flatten_messages(row.get("messages"))
    n_expected = _expected_image_count(meta if isinstance(meta, dict) else None, messages)

    turns_meta = (meta.get("turns") or []) if isinstance(meta, dict) else []
    per_human = _expected_counts_per_human(meta, messages)

    n_tag = 0
    residuals: List[str] = []
    needs_reposition = False
    human_i = 0
    for msg in messages:
        val = str(msg.get("value", ""))
        residuals.extend(_template_residuals(val))
        if msg.get("from") != "human":
            continue
        n_h = len(IMAGE_TAG_RE.findall(val))
        n_tag += n_h
        n_exp = per_human[human_i] if human_i < len(per_human) else 1
        if n_h != n_exp:
            errors.append(f"human turn {human_i}: {n_h} <image> tags != expected {n_exp}")
        elif n_h and not _image_tags_at_front(val, n_exp):
            needs_reposition = True
            errors.append(f"human turn {human_i}: image tags not at question start")
        human_i += 1

    if residuals:
        errors.append(f"template placeholders remain: {sorted(set(residuals))}")
    if n_tag != n_expected and not any("human turn" in e for e in errors):
        errors.append(f"image tag count {n_tag} != expected {n_expected}")

    fixed = False
    if fix and messages and errors:
        human_idx = 0
        new_messages = []
        for msg in messages:
            m = dict(msg)
            if m.get("from") == "human":
                n_exp = per_human[human_idx] if human_idx < len(per_human) else 1
                prefix = ""
                if human_idx < len(turns_meta) and isinstance(turns_meta[human_idx], dict):
                    prefix = (turns_meta[human_idx].get("question_prefix") or "").strip()
                m["value"] = _normalize_human_images(
                    m["value"],
                    n_exp,
                    question_prefix=prefix,
                    is_first_human=(human_idx == 0),
                )
                human_idx += 1
            new_messages.append(m)
        messages = new_messages
        n_tag = sum(
            len(IMAGE_TAG_RE.findall(str(m.get("value", ""))))
            for m in messages
            if m.get("from") == "human"
        )
        fixed = True

    if fixed:
        per_human = _expected_counts_per_human(meta, messages)
        errors = []
        residuals = []
        human_i = 0
        for msg in messages:
            val = str(msg.get("value", ""))
            residuals.extend(_template_residuals(val))
            if msg.get("from") != "human":
                continue
            n_h = len(IMAGE_TAG_RE.findall(val))
            n_exp = per_human[human_i] if human_i < len(per_human) else 1
            if n_h != n_exp:
                errors.append(f"human turn {human_i}: {n_h} <image> tags != expected {n_exp}")
            elif n_h and not _image_tags_at_front(val, n_exp):
                errors.append(f"human turn {human_i}: image tags not at question start")
            human_i += 1
        if residuals:
            errors.append(f"template placeholders remain: {sorted(set(residuals))}")
        n_tag = sum(
            len(IMAGE_TAG_RE.findall(str(m.get("value", ""))))
            for m in messages
            if m.get("from") == "human"
        )
        n_expected = sum(per_human)

    audit = {
        "fixed": fixed,
        "n_img": n_tag,
        "n_img_expected": n_expected,
        "n_tag": n_tag,
        "template_residuals": sorted(set(residuals)),
        "ok": len(errors) == 0,
    }
    return audit, messages, errors


def audit_turn_placeholders(turn: dict, *, fix: bool = False) -> Tuple[dict, dict, List[str]]:
    """
    Audit one annotation turn (pre-flatten). Uses question_text / prefix / placeholder count.
    """
    n_expected = max(0, int(turn.get("image_placeholder_count") or 1))
    prefix = (turn.get("question_prefix") or "").strip()
    body = (turn.get("question_text") or "").strip()
    value = body
    if prefix:
        value = f"{prefix}\n\n{body}" if body else prefix
    messages = [
        {"from": "human", "value": value},
        {"from": "gpt", "value": turn.get("answer_text") or ""},
    ]
    meta = {"turns": [turn]}
    audit, msgs, errs = audit_row({"metadata": meta, "messages": messages}, fix=fix)
    turn_out = dict(turn)
    if fix and msgs:
        val = msgs[0]["value"]
        if prefix and val.startswith(" ".join(["<image>"] * n_expected)):
            rest = val[len(" ".join(["<image>"] * n_expected)) :].strip()
            if rest.startswith(prefix):
                rest = rest[len(prefix):].lstrip()
                turn_out["question_text"] = rest
            else:
                turn_out["question_text"] = IMAGE_TAG_RE.sub("", rest).strip()
        else:
            turn_out["question_text"] = IMAGE_TAG_RE.sub("", val).strip()
    return audit, turn_out, errs


def _template_residuals(text: str) -> List[str]:
    return TEMPLATE_PLACEHOLDER_RE.findall(text or "")


class PlaceholderAuditPass(BaseTask):
    """M6: scan messages + metadata; optional normalize <image> placement."""

    def __init__(self, args):
        super().__init__(args)
        self.fix = bool(args.get("fix_placeholders", False))
        self.fail_on_error = bool(args.get("fail_on_error", False))

    def run(self, dataset: pd.DataFrame) -> pd.DataFrame:
        rows_out = []
        stats = {"rows": 0, "ok": 0, "fixed": 0, "failed": 0}
        for idx in range(len(dataset)):
            if "messages_json" in dataset.columns:
                row = {
                    "messages": json.loads(dataset["messages_json"].iloc[idx]),
                    "metadata": json.loads(dataset["metadata_json"].iloc[idx]),
                }
            else:
                try:
                    row = dataset.iloc[idx].to_dict()
                except Exception:
                    row = {col: dataset[col].iloc[idx] for col in dataset.columns}
            stats["rows"] += 1
            audit, messages, errs = audit_row(row, fix=self.fix)
            meta = row.get("metadata")
            if isinstance(meta, list) and meta:
                meta = dict(meta[0])
            elif isinstance(meta, dict):
                meta = dict(meta)
            else:
                meta = {}
            meta["placeholder_audit"] = audit
            row = dict(row)
            row["metadata"] = meta
            row["messages"] = messages
            if audit.get("ok"):
                stats["ok"] += 1
            else:
                stats["failed"] += 1
            if audit.get("fixed"):
                stats["fixed"] += 1
            if self.fail_on_error and errs:
                raise ValueError(f"row {idx}: " + "; ".join(errs))
            rows_out.append(row)
        print(f">>> PlaceholderAuditPass stats: {stats}")
        return pd.DataFrame(rows_out)
