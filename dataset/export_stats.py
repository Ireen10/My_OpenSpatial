"""
Aggregate statistics for upstream export (dataset-level metadata.json).
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

# Megapixel buckets (width * height / 1e6).
MEGAPIXEL_BUCKETS: List[Tuple[str, Optional[float], Optional[float]]] = [
    ("lt_0.05M", None, 0.05),
    ("0.05_0.15M", 0.05, 0.15),
    ("0.15_0.5M", 0.15, 0.5),
    ("0.5_1M", 0.5, 1.0),
    ("1_2M", 1.0, 2.0),
    ("2_4M", 2.0, 4.0),
    ("gte_4M", 4.0, None),
]

# Short-edge buckets (pixels).
SHORT_EDGE_BUCKETS: List[Tuple[str, Optional[int], Optional[int]]] = [
    ("lt_320px", None, 320),
    ("320_479px", 320, 480),
    ("480_639px", 480, 640),
    ("640_767px", 640, 768),
    ("768_1023px", 768, 1024),
    ("gte_1024px", 1024, None),
]


def bucket_megapixels(width: int, height: int) -> str:
    mp = (max(1, int(width)) * max(1, int(height))) / 1_000_000.0
    for label, lo, hi in MEGAPIXEL_BUCKETS:
        if lo is not None and mp < lo:
            continue
        if hi is not None and mp >= hi:
            continue
        return label
    return MEGAPIXEL_BUCKETS[-1][0]


def bucket_short_edge(width: int, height: int) -> str:
    short = min(max(1, int(width)), max(1, int(height)))
    for label, lo, hi in SHORT_EDGE_BUCKETS:
        if lo is not None and short < lo:
            continue
        if hi is not None and short >= hi:
            continue
        return label
    return SHORT_EDGE_BUCKETS[-1][0]


def _dist(counter: Counter) -> Dict[str, Any]:
    total = sum(counter.values())
    if total == 0:
        return {"count": 0, "histogram": {}}
    hist = {k: {"count": v, "fraction": round(v / total, 6)} for k, v in sorted(counter.items())}
    return {"count": total, "histogram": hist}


def _length_histogram(lengths: List[int]) -> Dict[str, Any]:
    if not lengths:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0, "p50": 0, "p90": 0, "p99": 0, "bins": {}}
    lengths = sorted(lengths)
    n = len(lengths)

    def pct(p: float) -> int:
        if n == 1:
            return lengths[0]
        idx = min(n - 1, int(math.ceil(p * n)) - 1)
        return lengths[idx]

    edges = [0, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 10**9]
    labels = [
        "0-31",
        "32-63",
        "64-127",
        "128-255",
        "256-511",
        "512-1023",
        "1024-2047",
        "2048-4095",
        "4096-8191",
        "8192+",
    ]
    bins: Counter = Counter()
    for ln in lengths:
        for i in range(len(labels)):
            if edges[i] <= ln < edges[i + 1]:
                bins[labels[i]] += 1
                break
    return {
        "count": n,
        "min": lengths[0],
        "max": lengths[-1],
        "mean": round(sum(lengths) / n, 2),
        "p50": pct(0.5),
        "p90": pct(0.9),
        "p99": pct(0.99),
        "bins": dict(bins),
    }


def resolve_dataset_source_from_record(record: dict) -> str:
    meta = record.get("metadata") or {}
    va = meta.get("visual_anchor") if isinstance(meta, dict) else {}
    if isinstance(va, dict) and va.get("dataset_source"):
        return str(va["dataset_source"])
    from task.annotation.core.dataset_source import infer_dataset_source

    refs = record.get("image_refs") or []
    raw = va.get("raw_image_ref") if isinstance(va, dict) else None
    return infer_dataset_source(
        parent_preprocess_id=va.get("parent_preprocess_id") if isinstance(va, dict) else None,
        raw_image_ref=raw or (refs[0] if refs else None),
        raw_image_refs=refs,
    )


class ExportStatsCollector:
    """Accumulate dataset-level stats while streaming export records."""

    def __init__(self, *, view_scope: str) -> None:
        self.view_scope = view_scope
        self._turns_by_task: Counter = Counter()
        self._turn_count_per_sample: Counter = Counter()
        self._image_count_per_sample: Counter = Counter()
        self._megapixels: Counter = Counter()
        self._short_edge: Counter = Counter()
        self._dataset_source: Counter = Counter()
        self._prompt_profile: Counter = Counter()
        self._answer_lengths: List[int] = []
        self.n_samples = 0

    def _task_key(self, task_name: str) -> str:
        name = (task_name or "unknown").strip()
        if name.startswith(f"{self.view_scope}_"):
            return name
        return f"{self.view_scope}_{name}"

    def observe_resolution_file(self, path: str) -> None:
        try:
            with Image.open(path) as im:
                w, h = im.size
        except Exception:
            return
        self._megapixels[bucket_megapixels(w, h)] += 1
        self._short_edge[bucket_short_edge(w, h)] += 1

    def observe_record(self, record: dict) -> None:
        from dataset.upstream_export import normalize_messages

        self.n_samples += 1
        meta = record.get("metadata") or {}
        turns = meta.get("turns") or []
        msgs = normalize_messages(record.get("messages"))
        refs = record.get("image_refs") or []
        n_images = len(refs)
        self._image_count_per_sample[str(n_images)] += 1

        n_turns = len(turns) if turns else max(0, len(msgs) // 2)
        self._turn_count_per_sample[str(n_turns)] += 1

        self._dataset_source[resolve_dataset_source_from_record(record)] += 1

        for turn in turns:
            task_name = turn.get("task_name") or turn.get("sub_task") or "unknown"
            key = self._task_key(str(task_name))
            self._turns_by_task[key] += 1

            from dataset.prompt_profile_stats import prompt_profile_stat_key

            self._prompt_profile[prompt_profile_stat_key(turn)] += 1

        for m in msgs:
            if isinstance(m, dict) and m.get("from") == "gpt":
                val = m.get("value") or ""
                if isinstance(val, str) and val.strip():
                    self._answer_lengths.append(len(val))

    def finalize(
        self,
        *,
        schema_version: str,
        pipeline_run_id: Optional[str],
        shard_size: int,
        n_shards: int,
        n_images_packed: int,
        missing_paths: int,
    ) -> Dict[str, Any]:
        return {
            "kind": "openspatial_upstream_dataset_metadata",
            "schema_version": schema_version,
            "view_scope": self.view_scope,
            "pipeline_run_id": pipeline_run_id,
            "shard_size": shard_size,
            "n_samples": self.n_samples,
            "n_shards": n_shards,
            "n_images_packed": n_images_packed,
            "missing_image_paths": missing_paths,
            "task_type": _dist(self._turns_by_task),
            "dataset_source": _dist(self._dataset_source),
            "turn_count_per_sample": _dist(self._turn_count_per_sample),
            "image_count_per_sample": _dist(self._image_count_per_sample),
            "resolution": {
                "by_megapixels": _dist(self._megapixels),
                "by_short_edge": _dist(self._short_edge),
                "megapixel_bucket_rules": [b[0] for b in MEGAPIXEL_BUCKETS],
                "short_edge_bucket_rules": [b[0] for b in SHORT_EDGE_BUCKETS],
            },
            "prompt_profile": _dist(self._prompt_profile),
            "answer_text_length": _length_histogram(self._answer_lengths),
        }
