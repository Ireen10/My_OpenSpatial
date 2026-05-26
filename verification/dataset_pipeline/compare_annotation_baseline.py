#!/usr/bin/env python3
"""
Compare annotation parquet QA strings against a baseline (M8 L2 parity).

Compares per-row:
  - messages: human + gpt per conversation turn
  - metadata.turns[].answer_text (when present)

Usage:
  python verification/dataset_pipeline/compare_annotation_baseline.py \\
    --baseline output/frame_rot/base_pipeline_demo_singleview_all_frame_rot \\
    --candidate output/frame_rot/base_pipeline_demo_singleview_all_frame_rot_rerun
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np


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


def _qa_signature(messages) -> List[Tuple[str, str]]:
    """(human, gpt) pairs per conversation in row."""
    convs = _to_list(messages)
    if not convs:
        return []
    if convs and isinstance(convs[0], dict):
        convs = [convs]
    sig: List[Tuple[str, str]] = []
    for conv in convs:
        if not isinstance(conv, list):
            continue
        human = gpt = ""
        for msg in conv:
            if not isinstance(msg, dict):
                continue
            role = msg.get("from")
            val = str(msg.get("value", "")).strip()
            if role == "human":
                human = val
            elif role == "gpt":
                gpt = val
        if human or gpt:
            sig.append((human, gpt))
    return sig


def _turn_answers(meta: Any) -> List[str]:
    if not meta:
        return []
    if isinstance(meta, np.ndarray):
        meta = meta.tolist()
    if isinstance(meta, list) and meta and isinstance(meta[0], dict):
        meta = meta[0]
    if not isinstance(meta, dict):
        return []
    turns = meta.get("turns") or []
    if isinstance(turns, np.ndarray):
        turns = turns.tolist()
    return [
        str(t.get("answer_text", "")).strip()
        for t in turns
        if isinstance(t, dict) and t.get("answer_text") is not None
    ]


def _find_parquet(root: Path, task: str) -> Path | None:
    for p in (
        root / "annotation_stage" / task / "data.parquet",
        root / task / "data.parquet",
    ):
        if p.is_file():
            return p
    return None


def _discover_tasks(root: Path) -> List[str]:
    stage = root / "annotation_stage"
    if not stage.is_dir():
        return []
    return sorted(
        d.name for d in stage.iterdir()
        if d.is_dir() and (d / "data.parquet").is_file()
    )


def _row_fingerprint(messages, metadata) -> str:
    payload = {
        "qa": _qa_signature(messages),
        "turn_answers": _turn_answers(metadata),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"),
    ).hexdigest()


def compare_task(
    baseline_path: Path,
    candidate_path: Path,
    task: str,
    *,
    max_mismatches: int = 20,
) -> Tuple[bool, List[str]]:
    import pandas as pd

    errors: List[str] = []
    if not baseline_path.is_file():
        return True, [f"{task}: baseline missing (skip)"]
    if not candidate_path.is_file():
        return False, [f"{task}: candidate missing {candidate_path}"]

    bdf = pd.read_parquet(baseline_path)
    cdf = pd.read_parquet(candidate_path)
    if len(bdf) != len(cdf):
        errors.append(f"{task}: row count {len(bdf)} vs {len(cdf)}")
        return False, errors[:max_mismatches]

    for idx in range(len(bdf)):
        br = bdf.iloc[idx]
        cr = cdf.iloc[idx]
        bf = _row_fingerprint(br.get("messages"), br.get("metadata"))
        cf = _row_fingerprint(cr.get("messages"), cr.get("metadata"))
        if bf != cf:
            bqa = _qa_signature(br.get("messages"))
            cqa = _qa_signature(cr.get("messages"))
            errors.append(
                f"{task} row {idx}: QA mismatch "
                f"(baseline turns={len(bqa)} candidate turns={len(cqa)})"
            )
            if len(bqa) == len(cqa) and bqa:
                for ti, ((bh, bg), (ch, cg)) in enumerate(zip(bqa, cqa)):
                    if bh != ch:
                        errors.append(f"  turn {ti} human diff (len {len(bh)} vs {len(ch)})")
                    if bg != cg:
                        errors.append(f"  turn {ti} gpt diff: {bg!r} vs {cg!r}")
            if len(errors) >= max_mismatches:
                break

    return len(errors) == 0, errors


def main() -> int:
    parser = argparse.ArgumentParser(description="M8 L2: compare annotation QA vs baseline")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--tasks", nargs="*", default=None)
    args = parser.parse_args()

    baseline = args.baseline.resolve()
    candidate = args.candidate.resolve()
    tasks = args.tasks or _discover_tasks(baseline)
    if not tasks:
        print(f"No tasks under {baseline / 'annotation_stage'}")
        return 1

    ok_all = True
    for task in tasks:
        bp = _find_parquet(baseline, task)
        cp = _find_parquet(candidate, task)
        if bp is None:
            print(f"SKIP {task}: no baseline parquet")
            continue
        if cp is None:
            print(f"FAIL {task}: no candidate parquet")
            ok_all = False
            continue
        ok, errs = compare_task(bp, cp, task)
        if ok:
            print(f"OK   {task}")
        else:
            ok_all = False
            print(f"FAIL {task}:")
            for e in errs:
                print(f"  {e}")

    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
