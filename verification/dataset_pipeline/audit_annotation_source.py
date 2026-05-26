#!/usr/bin/env python3
"""
Static audit: annotation task source patterns that cause metadata/messages answer drift.

Root cause class (2026-05):
  - Prompt built via render_qa() or manual concat without updating thread-local
    last_prompt_render before _record_turn().
  - _record_turn then reuses stale render.answer_text while messages use the new prompt.

Safe patterns:
  - render_prompt() / render_structured_prompt() / render_structured_with_options()
    / render_and_record() before _record_turn
  - _record_turn with explicit answer_text, or prompt containing Answer: (split in base)

Run from repo root:
  python verification/dataset_pipeline/audit_annotation_source.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ANNOTATION_DIR = REPO_ROOT / "task" / "annotation"

RENDER_QA_RE = re.compile(r"\.render_qa\s*\(")
RENDER_PROMPT_RE = re.compile(r"\brender_prompt\s*\(")
RENDER_STRUCTURED_RE = re.compile(r"\brender_structured_prompt\s*\(")
RENDER_OPTIONS_RE = re.compile(r"\brender_prompt_with_options\s*\(")
RENDER_STRUCTURED_OPTIONS_RE = re.compile(r"\brender_structured_with_options\s*\(")
RENDER_AND_RECORD_RE = re.compile(r"\brender_and_record\s*\(")
RECORD_TURN_RE = re.compile(r"\b_record_turn\s*\(")
RECORD_MV_RE = re.compile(r"\b_record_multiview_turn\s*\(")

# Files excluded from render_qa warnings (core library or special message builders).
RENDER_QA_ALLOW = {
    "core/prompt_template.py",
    "3d_grounding.py",  # render_structured_prompt (intro + stem + JSON instruction)
}


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ANNOTATION_DIR)).replace("\\", "/")
    except ValueError:
        return str(path)


def audit_file(path: Path) -> list[str]:
    rel = _rel(path)
    text = path.read_text(encoding="utf-8")
    issues: list[str] = []

    if RENDER_QA_RE.search(text) and rel not in RENDER_QA_ALLOW:
        for i, line in enumerate(text.splitlines(), 1):
            if ".render_qa(" in line:
                issues.append(
                    f"{rel}:{i}: render_qa() — ensure last_prompt_render is set or use render_prompt* helpers"
                )

    if RECORD_TURN_RE.search(text) or RECORD_MV_RE.search(text):
        has_safe_render = bool(
            RENDER_PROMPT_RE.search(text)
            or RENDER_STRUCTURED_RE.search(text)
            or RENDER_OPTIONS_RE.search(text)
            or RENDER_STRUCTURED_OPTIONS_RE.search(text)
            or RENDER_AND_RECORD_RE.search(text)
        )
        if not has_safe_render:
            issues.append(
                f"{rel}: calls _record_turn/_record_multiview_turn but no render_prompt/"
                "render_structured_*/render_and_record in file — verify prompt provenance"
            )

    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Static audit of annotation task source")
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=ANNOTATION_DIR,
        help="task/annotation directory",
    )
    args = parser.parse_args()
    root: Path = args.annotation_dir
    if not root.is_dir():
        print(f"Missing annotation dir: {root}", file=sys.stderr)
        return 1

    py_files = sorted(p for p in root.glob("*.py") if p.name != "__init__.py")

    all_issues: list[str] = []
    for path in py_files:
        all_issues.extend(audit_file(path))

    print("Annotation source audit (answer provenance patterns)")
    print(f"Scanned {len(py_files)} task modules under {root}")
    if not all_issues:
        print("PASS: no risky render_qa / missing render_prompt patterns flagged")
        return 0

    print(f"WARN: {len(all_issues)} note(s) — review each (may be intentional):")
    for item in all_issues:
        print(f"  - {item}")
    # Warnings only; grounding camera render_qa is documented allowlist.
    return 0


if __name__ == "__main__":
    sys.exit(main())
