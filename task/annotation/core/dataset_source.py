"""Infer upstream dataset source (arkitscenes, scannet, …) for metadata and export stats."""

from __future__ import annotations

from typing import Any, Optional

# Order: longer names first if we add prefixes later.
_KNOWN_SOURCES = (
    "arkitscenes",
    "matterport3d",
    "scannet",
    "3rscan",
    "hypersim",
)


def normalize_dataset_source(value: Any) -> str:
    if value is None:
        return "unknown"
    s = str(value).strip().lower()
    if not s:
        return "unknown"
    if s in _KNOWN_SOURCES:
        return s
    if s == "rscan" or s == "rscan3d":
        return "3rscan"
    return s


def infer_dataset_source(
    *,
    explicit: Optional[str] = None,
    parent_preprocess_id: Optional[str] = None,
    raw_image_ref: Optional[str] = None,
    raw_image_refs: Optional[list] = None,
) -> str:
    """Resolve dataset source from preprocess fields or path / id conventions."""
    if explicit:
        return normalize_dataset_source(explicit)

    pid = str(parent_preprocess_id or "").replace("\\", "/")
    for ds in _KNOWN_SOURCES:
        if pid.startswith(f"{ds}__") or pid.startswith(f"{ds}/"):
            return ds

    paths = []
    if raw_image_ref:
        paths.append(str(raw_image_ref))
    for ref in raw_image_refs or []:
        paths.append(str(ref))
    for path in paths:
        low = path.replace("\\", "/").lower()
        for ds in _KNOWN_SOURCES:
            if f"/{ds}/" in low or low.startswith(f"{ds}/"):
                return ds

    return "unknown"
