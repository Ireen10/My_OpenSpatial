"""Unit tests for Pangu export mark flattening (per_view layout)."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from script.export_to_pangu_ml import (
    apply_marked_text,
    assign_export_colors,
    count_renderable_slots_for_view,
    flat_mark_spec_for_view,
    render_marked_image_bytes,
)
from task.aggregate.fingerprint import mark_spec_norm


def _per_view_spec() -> dict:
    return {
        "version": 2,
        "layout": "per_view",
        "views": [
            {
                "view_index": 0,
                "image_ref": "a.jpg",
                "mark_kinds": ["box"],
                "slots": [
                    {
                        "slot_id": "A",
                        "tag": "chair",
                        "mark_kind": "box",
                        "color_name": "red",
                        "geometry": {"box_2d": [10, 10, 50, 50]},
                    },
                    {
                        "slot_id": "B",
                        "tag": "table",
                        "mark_kind": "box",
                        "color_name": "green",
                        "geometry": {"box_2d": [60, 60, 90, 90]},
                    },
                ],
            },
            {
                "view_index": 1,
                "image_ref": "b.jpg",
                "mark_kinds": ["box"],
                "slots": [
                    {
                        "slot_id": "A",
                        "tag": "sofa",
                        "mark_kind": "box",
                        "color_name": "blue",
                        "geometry": {"box_2d": [20, 20, 40, 40]},
                    },
                ],
            },
        ],
    }


def test_flat_mark_spec_for_view_reads_per_view_layout():
    spec = _per_view_spec()
    v0 = flat_mark_spec_for_view(spec, 0)
    v1 = flat_mark_spec_for_view(spec, 1)
    assert v0 is not None and len(v0["slots"]) == 2
    assert v1 is not None and len(v1["slots"]) == 1
    assert "views" not in v0


def test_flat_mark_spec_returns_sample_level_view_slots():
    spec = _per_view_spec()
    flat = flat_mark_spec_for_view(spec, 0)
    assert flat is not None
    assert len(flat["slots"]) == 2
    assert flat["slots"][0]["tag"] == "chair"


def test_render_marked_image_bytes_draws_boxes():
    from utils.image_utils import convert_pil_to_bytes

    img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    raw = convert_pil_to_bytes(img)
    spec = _per_view_spec()
    view_ms = flat_mark_spec_for_view(spec, 0)
    out = render_marked_image_bytes(raw, view_ms, view_index=0)
    assert out != raw
    assert len(out) > len(raw)


def test_render_marked_image_bytes_accepts_flat_spec_for_nonzero_view_index():
    from utils.image_utils import convert_pil_to_bytes

    img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    raw = convert_pil_to_bytes(img)
    spec = _per_view_spec()
    view_ms = flat_mark_spec_for_view(spec, 1)
    assert view_ms is not None
    out = render_marked_image_bytes(raw, view_ms, view_index=1)
    assert out != raw
    assert len(out) > len(raw)


def test_count_renderable_slots_for_view():
    spec = _per_view_spec()
    assert count_renderable_slots_for_view(spec, 0) == 2
    assert count_renderable_slots_for_view(spec, 1) == 1
    assert count_renderable_slots_for_view(spec, 2) == 0


def test_mark_spec_norm_ignores_color_names():
    spec_a = _per_view_spec()
    spec_b = _per_view_spec()
    spec_b["views"][0]["slots"][0]["color_name"] = "purple"
    spec_b["views"][0]["slots"][1]["color_name"] = "orange"
    assert mark_spec_norm(spec_a) == mark_spec_norm(spec_b)


def test_mark_spec_norm_accepts_parquet_numpy_nested_values():
    spec = _per_view_spec()
    for view in spec["views"]:
        view["slots"] = np.array(view["slots"], dtype=object)
    spec["views"] = np.array(spec["views"], dtype=object)

    norm = mark_spec_norm(spec)

    assert norm is not None
    assert len(norm["views"]) == 2
    assert sum(len(v["slots"]) for v in norm["views"]) == 3


def test_export_color_assignment_rewrites_legacy_mark_text():
    spec = _per_view_spec()
    colored, replacements = assign_export_colors(spec)
    assert colored is not spec
    assert replacements
    assert colored["views"][0]["slots"][0]["color_name"]

    repl = {
        "tag": "chair",
        "kind": "box",
        "old_color": "red",
        "new_color": "purple",
    }
    old_text = f"{repl['tag']}-({repl['old_color']} {repl['kind']})"
    new_text = apply_marked_text(f"Choose {old_text}.", [repl])
    assert repl["new_color"] in new_text
    assert repl["old_color"] not in new_text


def test_export_color_assignment_rewrites_placeholder_mark_tokens():
    spec = _per_view_spec()
    spec["views"][0]["slots"][0]["color_name"] = "mark0"

    colored, replacements = assign_export_colors(spec)
    repl = next(r for r in replacements if r["tag"] == "chair")
    text = apply_marked_text("Choose chair-(mark0 box).", [repl])

    assert colored["views"][0]["slots"][0]["color_name"] == repl["new_color"]
    assert "mark0" not in text
    assert repl["new_color"] in text
