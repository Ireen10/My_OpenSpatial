#!/usr/bin/env python3
"""
Audit upstream export JSONL: compare legacy mark tokens in messages vs mark_spec slots.

Use before/after ``export_to_pangu_ml.py`` to find samples where QA text mentions
``tag-(red box)`` but ``metadata.mark_spec`` cannot render that many marks.

Example:
  python script/audit_export_marks.py --export-dir /path/to/export
  python script/audit_export_marks.py --export-dir /path/to/export --sample-id abc123
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataset.upstream_export import discover_shard_pairs, normalize_messages  # noqa: E402
from script.export_to_pangu_ml import (  # noqa: E402
    LEGACY_MARK_SUFFIX_RE,
    audit_sample_marks,
    count_legacy_mark_tokens,
    flat_mark_spec_for_view,
    parse_qa_pairs,
    slot_has_renderable_geometry,
)
from task.annotation.core.mark_spec import mark_spec_views, slots_for_view  # noqa: E402


def _sample_mark_spec(metadata: dict) -> Optional[dict]:
    ms = metadata.get("mark_spec")
    return ms if isinstance(ms, dict) else None


def per_view_report(
    metadata: dict,
    n_views: int,
) -> List[Dict[str, Any]]:
    ms = _sample_mark_spec(metadata)
    rows = []
    for vi in range(n_views):
        spec_slots = slots_for_view(ms, vi) if ms else []
        flat = flat_mark_spec_for_view(ms, vi)
        renderable = [
            s for s in (flat.get("slots") or []) if slot_has_renderable_geometry(s)
        ]
        rows.append({
            "view_index": vi,
            "spec_slot_count": len(spec_slots),
            "renderable_count": len(renderable),
            "tags": [s.get("tag") for s in spec_slots],
            "colors": [s.get("color_name") for s in spec_slots],
            "kinds": [s.get("mark_kind") for s in spec_slots],
        })
    return rows


def audit_record(record: dict) -> Optional[Dict[str, Any]]:
    pairs = parse_qa_pairs(normalize_messages(record.get("messages")))
    if not pairs:
        return None

    metadata = record.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}

    legacy, renderable, views_no, mismatch = audit_sample_marks(record, pairs)
    refs = list(record.get("image_refs") or [])
    ms = _sample_mark_spec(metadata)

    return {
        "sample_id": record.get("sample_id"),
        "n_views": len(refs),
        "legacy_mark_tokens": legacy,
        "renderable_slots": renderable,
        "views_without_render": views_no,
        "text_render_mismatch": bool(mismatch),
        "mark_layout": ms.get("layout") if ms else None,
        "mark_view_count": len(mark_spec_views(ms)) if ms else 0,
        "per_view": per_view_report(metadata, len(refs)),
        "messages_preview": " | ".join(
            LEGACY_MARK_SUFFIX_RE.findall(q + " " + a)
            for q, a in pairs[:3]
        ),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit mark_spec vs legacy mark text in export JSONL.")
    p.add_argument("--export-dir", type=Path, required=True)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--sample-id", type=str, default=None, help="Inspect one sample_id only.")
    p.add_argument(
        "--only-mismatch",
        action="store_true",
        help="Print only samples where legacy token count > renderable slots.",
    )
    p.add_argument("--json", action="store_true", help="Emit full per-sample JSON lines.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    export_dir = args.export_dir.resolve()
    pairs = discover_shard_pairs(export_dir)
    if not pairs:
        raise SystemExit(f"No shards under {export_dir / 'jsonl'}")

    totals = Counter()
    seen = 0
    printed = 0

    for jf_path, _tar in pairs:
        with jf_path.open("r", encoding="utf-8") as jf:
            for line in jf:
                if args.max_samples is not None and seen >= args.max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                sid = str(record.get("sample_id") or "")
                if args.sample_id and sid != args.sample_id:
                    continue

                report = audit_record(record)
                if report is None:
                    continue
                seen += 1
                totals["samples"] += 1
                totals["legacy_tokens"] += report["legacy_mark_tokens"]
                totals["renderable_slots"] += report["renderable_slots"]
                if report["text_render_mismatch"]:
                    totals["mismatch_samples"] += 1
                if report["views_without_render"]:
                    totals["views_without_render"] += report["views_without_render"]

                show = not args.only_mismatch or report["text_render_mismatch"]
                if args.sample_id or show:
                    printed += 1
                    if args.json:
                        print(json.dumps(report, ensure_ascii=False))
                    else:
                        print(
                            f"[{report['sample_id']}] legacy={report['legacy_mark_tokens']} "
                            f"renderable={report['renderable_slots']} "
                            f"views={report['n_views']} layout={report['mark_layout']!r} "
                            f"mismatch={report['text_render_mismatch']}"
                        )
                        for pv in report["per_view"]:
                            if pv["spec_slot_count"] or pv["renderable_count"]:
                                print(
                                    f"  view {pv['view_index']}: "
                                    f"slots={pv['spec_slot_count']} "
                                    f"renderable={pv['renderable_count']} "
                                    f"tags={pv['tags']} colors={pv['colors']}"
                                )

        if args.max_samples is not None and seen >= args.max_samples:
            break

    print("\n=== summary ===", flush=True)
    print(dict(totals), flush=True)
    if totals.get("mismatch_samples"):
        print(
            f"Found {totals['mismatch_samples']} sample(s) with text_render_mismatch. "
            "Re-run export_to_pangu_ml.py after fixes; use --sample-id to drill into one row.",
            flush=True,
        )
    elif seen and not args.sample_id:
        print("No text/render count mismatches in scanned samples.", flush=True)


if __name__ == "__main__":
    main()
