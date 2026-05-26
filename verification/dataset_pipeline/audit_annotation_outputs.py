#!/usr/bin/env python3
"""
Audit annotation parquet outputs (plan M1–M3 L1/L2 gate).

Checks each task directory for metadata column coverage and strict_m3 validation
on sampled rows.
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from verification.dataset_pipeline.validate_metadata import (
    SCHEMA_VERSION,
    validate_sample_record,
    _validate_turn,
    _validate_prompt_struct,
    _validate_visual_anchor,
)
from task.aggregate.fingerprint import image_refs_from_row


def _to_list(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    if isinstance(val, np.ndarray):
        return val.tolist()
    if hasattr(val, "tolist") and not isinstance(val, (list, str, dict)):
        return val.tolist()
    if isinstance(val, list):
        return val
    return [val]


def _gpt_answers_from_messages(messages) -> list[str]:
    convs = _to_list(messages)
    if not convs:
        return []
    if convs and isinstance(convs[0], dict):
        convs = [convs]
    answers: list[str] = []
    for conv in convs:
        if not isinstance(conv, list):
            continue
        for msg in conv:
            if isinstance(msg, dict) and msg.get("from") == "gpt":
                answers.append(str(msg.get("value", "")).strip())
    return answers


def _turn_answer_texts(meta: dict | None) -> list[str]:
    if not meta:
        return []
    turns = meta.get("turns") or []
    if isinstance(turns, np.ndarray):
        turns = turns.tolist()
    out: list[str] = []
    for turn in turns:
        if isinstance(turn, dict) and turn.get("answer_text") is not None:
            out.append(str(turn["answer_text"]).strip())
    return out


def audit_answer_consistency(path: Path, task_name: str) -> tuple[bool, list[str]]:
    """Messages must have non-empty gpt values (canonical viz surface)."""
    import pandas as pd

    errors: list[str] = []
    if not path.is_file():
        return False, [f"missing file: {path}"]

    df = pd.read_parquet(path)
    if "messages" not in df.columns:
        return True, [f"{task_name}: skip (no messages column)"]

    for idx, row in df.iterrows():
        gpts = _gpt_answers_from_messages(row.get("messages"))
        for ti, gpt in enumerate(gpts):
            if not gpt.strip():
                errors.append(f"{task_name} row {idx} turn {ti}: empty gpt message")
    return len(errors) == 0, errors


def _normalize_meta(meta) -> dict | None:
    if meta is None or (isinstance(meta, float) and np.isnan(meta)):
        return None
    if isinstance(meta, list) and meta:
        meta = meta[0]
    if not isinstance(meta, dict):
        return None
    meta = dict(meta)
    turns = meta.get("turns")
    if isinstance(turns, np.ndarray):
        meta["turns"] = turns.tolist()
    elif hasattr(turns, "tolist") and not isinstance(turns, list):
        meta["turns"] = turns.tolist()
    ms = meta.get("mark_spec")
    if isinstance(ms, dict) and isinstance(ms.get("mark_kinds"), np.ndarray):
        ms = dict(ms)
        ms["mark_kinds"] = ms["mark_kinds"].tolist()
        meta["mark_spec"] = ms
    if isinstance(ms, dict) and isinstance(ms.get("slots"), np.ndarray):
        ms = dict(ms)
        ms["slots"] = ms["slots"].tolist()
        meta["mark_spec"] = ms
    return meta


def _mark_spec_from_meta(meta: dict) -> dict | None:
    from task.annotation.core.mark_spec import mark_spec_has_slots
    ms = meta.get("mark_spec")
    if isinstance(ms, dict) and mark_spec_has_slots(ms):
        return ms
    for turn in meta.get("turns") or []:
        if isinstance(turn, dict):
            tms = turn.get("mark_spec")
            if isinstance(tms, dict) and mark_spec_has_slots(tms):
                return tms
    return None


def _audit_mask_ref_paths(
    meta: dict, task_name: str, row_idx: int, errors: List[str], *, check_file: bool,
) -> None:
    """Annotation-stage mask slots should embed pipeline mask paths (M2)."""
    from task.annotation.core.mark_spec import all_slots_flat
    ms = _mark_spec_from_meta(meta)
    if not ms:
        return
    for slot in all_slots_flat(ms):
        if not isinstance(slot, dict) or slot.get("mark_kind") != "mask":
            continue
        sid = slot.get("slot_id", "?")
        ref = (slot.get("geometry") or {}).get("mask_ref") or {}
        if ref.get("source") != "path":
            errors.append(
                f"{task_name} row {row_idx} slot {sid}: "
                f"mask_ref.source must be 'path' (got {ref.get('source')!r})"
            )
            continue
        path = ref.get("path")
        if not path or not str(path).strip():
            errors.append(f"{task_name} row {row_idx} slot {sid}: mask_ref.path missing")
        elif check_file and not os.path.isfile(str(path)):
            errors.append(
                f"{task_name} row {row_idx} slot {sid}: mask file not found: {path}"
            )


def _row_to_sample(row: dict, task_name: str) -> dict:
    meta = _normalize_meta(row.get("metadata"))
    refs = image_refs_from_row(row)
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": str(uuid.uuid4()),
        "merge_group_key": "audit",
        "image_refs": refs,
        "messages": _to_list(row.get("messages")),
        "metadata": meta or {},
    }


def audit_parquet(
    path: Path,
    task_name: str,
    *,
    sample_n: int,
    strict_m3: bool,
    require_mask_path: bool,
    check_mask_files: bool,
) -> Tuple[bool, List[str]]:
    import pandas as pd

    errors: List[str] = []
    if not path.is_file():
        return False, [f"missing file: {path}"]

    df = pd.read_parquet(path)
    if len(df) == 0:
        return True, [f"{task_name}: empty parquet (skip)"]
    if "metadata" not in df.columns:
        return True, [f"{task_name}: no metadata column (skip schema audit)"]

    missing = df["metadata"].isna().sum()
    if missing > 0:
        errors.append(f"{task_name}: {missing}/{len(df)} rows missing metadata")

    n = min(sample_n, len(df))
    if n == 0:
        return False, [f"{task_name}: empty parquet"]

    indices = list(range(0, len(df), max(1, len(df) // n)))[:n]
    for idx in indices:
        row = df.iloc[idx].to_dict()
        meta = _normalize_meta(row.get("metadata"))
        turns = meta.get("turns") if meta else None
        if not meta or not turns or len(turns) == 0:
            errors.append(f"{task_name} row {idx}: metadata.turns missing")
            continue
        sample = _row_to_sample(row, task_name)
        ok, errs = validate_sample_record(sample, strict_m3=strict_m3)
        if not ok:
            for e in errs[:5]:
                errors.append(f"{task_name} row {idx}: {e}")
        va = meta.get("visual_anchor")
        ve: List[str] = []
        _validate_visual_anchor(va, "visual_anchor", ve)
        for e in ve:
            errors.append(f"{task_name} row {idx}: {e}")
        turns_list = meta.get("turns") or []
        for ti, turn in enumerate(turns_list):
            te: List[str] = []
            _validate_turn(turn, f"turns[{ti}]", te, strict_m3=strict_m3)
            for e in te:
                errors.append(f"{task_name} row {idx}: {e}")
            if strict_m3 and isinstance(turn, dict) and turn.get("prompt_struct"):
                pe: List[str] = []
                _validate_prompt_struct(turn["prompt_struct"], "prompt_struct", pe)
                for e in pe:
                    errors.append(f"{task_name} row {idx}: {e}")
        if require_mask_path and meta:
            _audit_mask_ref_paths(
                meta, task_name, idx, errors, check_file=check_mask_files,
            )

    return len(errors) == 0, errors


def discover_tasks(root: Path) -> List[str]:
    stage = root / "annotation_stage"
    if not stage.is_dir():
        return []
    return sorted(
        d.name for d in stage.iterdir()
        if d.is_dir() and (d / "data.parquet").is_file()
    )


def audit_root(
    root: Path,
    tasks: List[str] | None,
    *,
    sample_n: int,
    strict_m3: bool,
    require_mask_path: bool,
    check_mask_files: bool,
    check_answers: bool,
) -> bool:
    if not tasks:
        tasks = discover_tasks(root)
    if not tasks:
        print(f"FAIL: no annotation_stage/*/data.parquet under {root}")
        return False

    all_ok = True
    for task in tasks:
        candidates = [
            root / "annotation_stage" / task / "data.parquet",
            root / task / "data.parquet",
        ]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            print(f"FAIL {task}: no data.parquet under {root}")
            all_ok = False
            continue
        ok, errs = audit_parquet(
            path,
            task,
            sample_n=sample_n,
            strict_m3=strict_m3,
            require_mask_path=require_mask_path,
            check_mask_files=check_mask_files,
        )
        ans_ok, ans_errs = True, []
        ans_skips: list[str] = []
        if check_answers:
            ans_ok, ans_errs = audit_answer_consistency(path, task)
            ans_skips = [e for e in ans_errs if "skip answer check" in e]
            ans_errs = [e for e in ans_errs if e not in ans_skips]

        if ok and ans_ok:
            print(f"PASS {task}: {path}")
            for s in ans_skips:
                print(f"  - {s}")
        else:
            all_ok = False
            print(f"FAIL {task}: {path}")
            for e in errs[:20]:
                print(f"  - {e}")
            if len(errs) > 20:
                print(f"  ... and {len(errs) - 20} more")
            for e in ans_errs[:20]:
                print(f"  - [answer] {e}")
            if len(ans_errs) > 20:
                print(f"  ... and {len(ans_errs) - 20} more answer mismatches")
            for s in ans_skips:
                print(f"  - {s}")
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Audit annotation parquet metadata (M1–M3)")
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Pipeline output root (contains annotation_stage/<task>/data.parquet)",
    )
    parser.add_argument("--sample-n", type=int, default=10, help="Rows to sample per task")
    parser.add_argument("--strict-m3", action="store_true", default=True)
    parser.add_argument("--no-strict-m3", action="store_false", dest="strict_m3")
    parser.add_argument(
        "--require-mask-path",
        action="store_true",
        default=True,
        help="Mask slots must use mask_ref.source=path with pipeline mask path (M2)",
    )
    parser.add_argument(
        "--no-require-mask-path",
        action="store_false",
        dest="require_mask_path",
    )
    parser.add_argument(
        "--check-mask-files",
        action="store_true",
        default=True,
        help="Verify mask_ref.path files exist on disk",
    )
    parser.add_argument(
        "--no-check-mask-files",
        action="store_false",
        dest="check_mask_files",
    )
    parser.add_argument(
        "--check-answers",
        action="store_true",
        default=True,
        help="All rows: metadata.turns[].answer_text vs messages gpt (recommended)",
    )
    parser.add_argument(
        "--no-check-answers",
        action="store_false",
        dest="check_answers",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Task subdirs under annotation_stage (default: auto-discover all with data.parquet)",
    )
    args = parser.parse_args()
    ok = audit_root(
        args.root,
        args.tasks,
        sample_n=args.sample_n,
        strict_m3=args.strict_m3,
        require_mask_path=args.require_mask_path,
        check_mask_files=args.check_mask_files,
        check_answers=args.check_answers,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
