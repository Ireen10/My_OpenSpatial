#!/usr/bin/env python3
"""
Report annotation tasks that call marker.mark_objects() outside plan_mark_for_qa / _mark_per_view.

Usage: python verification/dataset_pipeline/check_annotation_mark_paths.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TASK_DIR = REPO / "task" / "annotation"

ALLOWED = {
    "core/visual_marker.py",
    "core/mark_spec.py",
    "core/base_annotation_task.py",
    "core/base_multiview_task.py",
}


def main() -> int:
    offenders = []
    for path in sorted(TASK_DIR.rglob("*.py")):
        rel = path.relative_to(TASK_DIR).as_posix()
        if rel in ALLOWED or rel.startswith("core/"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "mark_objects":
                offenders.append(f"{rel}:{node.lineno}")

    if offenders:
        print("FAIL: direct mark_objects() in task modules (use mark_objects_for_qa / plan_mark_for_qa):")
        for line in offenders:
            print(" ", line)
        return 1
    print("PASS: no direct marker.mark_objects() in annotation task modules")
    return 0


if __name__ == "__main__":
    sys.exit(main())
