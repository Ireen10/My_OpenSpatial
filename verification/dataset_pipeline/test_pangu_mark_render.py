"""Unit tests for Pangu export mark flattening (per_view layout)."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from script.export_to_pangu_ml import (
    combined_mark_spec_for_view,
    count_renderable_slots_for_view,
    flat_mark_spec_for_view,
    render_marked_image_bytes,
)


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


def test_combined_mark_spec_unions_sample_level_spec():
    spec = _per_view_spec()
    merged = combined_mark_spec_for_view([], spec, 0)
    assert merged is not None
    assert len(merged["slots"]) == 2
    assert merged["slots"][0]["tag"] == "chair"


def test_render_marked_image_bytes_draws_boxes():
    from utils.image_utils import convert_pil_to_bytes

    img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    raw = convert_pil_to_bytes(img)
    spec = _per_view_spec()
    view_ms = combined_mark_spec_for_view([], spec, 0)
    out = render_marked_image_bytes(raw, view_ms, view_index=0)
    assert out != raw
    assert len(out) > len(raw)


def test_count_renderable_slots_for_view():
    spec = _per_view_spec()
    assert count_renderable_slots_for_view([], spec, 0) == 2
    assert count_renderable_slots_for_view([], spec, 1) == 1
    assert count_renderable_slots_for_view([], spec, 2) == 0
