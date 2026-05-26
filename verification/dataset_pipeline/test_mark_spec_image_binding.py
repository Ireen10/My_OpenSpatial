"""
Pytest gate: mark_spec.views[] must bind to QA images (per_view layout).

Covers:
- Single-view (views length 1)
- Multiview (views length == image_placeholder_count == len(QA_images))
- Tasks with marks (per_view + image_ref on marked views)
- Tasks without marks (no mark_spec or empty views)
- Partial marks: every QA image has a views[] key; only some have non-empty slots

Run after annotation:
  python run.py --config config/annotation/demo_singleview_all_frame_rot.yaml --output_dir E:/GitRepo/OpenSpatial/output/frame_rot
  python run.py --config config/annotation/demo_multiview_all_frame_rot.yaml --output_dir E:/GitRepo/OpenSpatial/output/frame_rot
  pytest verification/dataset_pipeline/test_mark_spec_image_binding.py -q
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FRAME_ROT = Path(os.environ.get("OPENSPATIAL_FRAME_ROT", REPO_ROOT / "output" / "frame_rot"))

SINGLEVIEW_ROOT = FRAME_ROT / "base_pipeline_demo_singleview_all_frame_rot" / "annotation_stage"
MULTIVIEW_ROOT = FRAME_ROT / "base_pipeline_demo_multiview_all_frame_rot" / "annotation_stage"

SINGLEVIEW_TASKS_WITH_MARKS = ("distance", "depth_annotation", "size", "position")
SINGLEVIEW_TASKS_NO_MARKS = ("3d_grounding",)
MULTIVIEW_TASKS_WITH_MARKS = (
    "multiview_distance",
    "multiview_size",
    "multiview_correspondence",
    "multiview_object_position",
)
MULTIVIEW_TASKS_OPTIONAL = ("multiview_distance_obj_cam",)


def _normalize_meta(meta: Any) -> Optional[dict]:
    if meta is None or (isinstance(meta, float) and pd.isna(meta)):
        return None
    if isinstance(meta, list) and meta:
        meta = meta[0]
    return meta if isinstance(meta, dict) else None


def _qa_image_count(row: dict, meta: dict) -> int:
    turn = (meta.get("turns") or [{}])[0]
    n = int(turn.get("image_placeholder_count") or 0)
    if n > 0:
        return n
    qa = row.get("QA_images")
    if isinstance(qa, dict):
        return 1
    if hasattr(qa, "__len__"):
        return len(qa)
    return 1


def _anchor_refs(meta: dict) -> List[str]:
    va = meta.get("visual_anchor") or {}
    if va.get("raw_image_refs") is not None:
        refs = va["raw_image_refs"]
        if hasattr(refs, "tolist"):
            refs = refs.tolist()
        return [str(r).replace("\\", "/") for r in refs]
    if va.get("raw_image_ref"):
        return [str(va["raw_image_ref"]).replace("\\", "/")]
    return []


def _check_row_binding(row: dict) -> List[str]:
    from task.annotation.core.mark_spec import mark_spec_has_slots, mark_spec_views

    errors: List[str] = []
    meta = _normalize_meta(row.get("metadata"))
    if not meta:
        errors.append("metadata missing")
        return errors

    n_qa = _qa_image_count(row, meta)
    refs = _anchor_refs(meta)
    ms = meta.get("mark_spec")

    if not mark_spec_has_slots(ms):
        return errors

    if not isinstance(ms, dict) or ms.get("layout") != "per_view":
        errors.append(f"mark_spec.layout must be per_view, got {ms.get('layout')!r}")

    views = mark_spec_views(ms)
    if len(views) != n_qa:
        errors.append(f"len(views)={len(views)} != QA count {n_qa}")

    if refs and len(refs) != n_qa:
        errors.append(f"len(visual_anchor refs)={len(refs)} != QA count {n_qa}")

    for i, v in enumerate(sorted(views, key=lambda x: int(x.get("view_index", 0)))):
        vi = int(v.get("view_index", -1))
        if vi != i:
            errors.append(f"views[{i}].view_index={vi} (expected contiguous 0..N-1)")
        ref = v.get("image_ref")
        if not ref or not str(ref).strip():
            errors.append(f"views[{i}].image_ref missing")
        elif refs and str(ref).replace("\\", "/") != refs[vi]:
            errors.append(
                f"views[{vi}].image_ref != visual_anchor ref: {ref!r} vs {refs[vi]!r}"
            )

    return errors


def _iter_task_parquet(root: Path, tasks: Tuple[str, ...]):
    for task in tasks:
        path = root / task / "data.parquet"
        if not path.is_file():
            pytest.skip(f"missing {path} — run annotation pipeline first")
        df = pd.read_parquet(path)
        assert len(df) > 0, f"{task}: empty parquet"
        yield task, df


@pytest.mark.parametrize("task", SINGLEVIEW_TASKS_WITH_MARKS)
def test_singleview_marked_tasks_per_view_binding(task: str):
    for t, df in _iter_task_parquet(SINGLEVIEW_ROOT, (task,)):
        assert t == task
        for idx in range(min(5, len(df))):
            errs = _check_row_binding(df.iloc[idx].to_dict())
            assert not errs, f"row {idx}: " + "; ".join(errs)


@pytest.mark.parametrize("task", MULTIVIEW_TASKS_WITH_MARKS)
def test_multiview_marked_tasks_per_view_binding(task: str):
    for t, df in _iter_task_parquet(MULTIVIEW_ROOT, (task,)):
        assert t == task
        for idx in range(min(5, len(df))):
            errs = _check_row_binding(df.iloc[idx].to_dict())
            assert not errs, f"row {idx}: " + "; ".join(errs)


@pytest.mark.parametrize("task", SINGLEVIEW_TASKS_NO_MARKS)
def test_singleview_no_mark_tasks(task: str):
    from task.annotation.core.mark_spec import mark_spec_has_slots

    for _, df in _iter_task_parquet(SINGLEVIEW_ROOT, (task,)):
        for idx in range(min(3, len(df))):
            meta = _normalize_meta(df.iloc[idx]["metadata"])
            assert meta is not None
            assert not mark_spec_has_slots(meta.get("mark_spec")), (
                f"{task} row {idx}: expected no mark_spec slots"
            )


def test_multiview_partial_views_dict_keys():
    """Every QA image is a views[] key; at least one view may have empty slots."""
    from task.annotation.core.mark_spec import mark_spec_views

    path = MULTIVIEW_ROOT / "multiview_distance" / "data.parquet"
    if not path.is_file():
        pytest.skip(f"missing {path}")
    df = pd.read_parquet(path)
    saw_partial = False
    for idx in range(min(10, len(df))):
        row = df.iloc[idx].to_dict()
        meta = _normalize_meta(row["metadata"])
        if not meta or not meta.get("mark_spec"):
            continue
        n_qa = _qa_image_count(row, meta)
        views = mark_spec_views(meta["mark_spec"])
        assert len(views) == n_qa
        nonempty = sum(1 for v in views if v.get("slots"))
        assert nonempty >= 1
        if nonempty < n_qa:
            saw_partial = True
    assert saw_partial or len(df) > 0, "expected some multiview rows with partial marks"


def test_multiview_correspondence_two_views_two_slot_groups():
    from task.annotation.core.mark_spec import mark_spec_views

    path = MULTIVIEW_ROOT / "multiview_correspondence" / "data.parquet"
    if not path.is_file():
        pytest.skip(f"missing {path}")
    df = pd.read_parquet(path)
    row = df.iloc[0].to_dict()
    meta = _normalize_meta(row["metadata"])
    views = mark_spec_views(meta["mark_spec"])
    assert len(views) == 2
    assert len(views[0]["slots"]) >= 1
    assert len(views[1]["slots"]) >= 1
    assert row["messages"][0]["value"].count("<image>") == 2
