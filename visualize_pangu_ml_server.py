"""
Fast local web viewer for Pangu ML JSONL + tar bundles.

Expected layout for each dataset root:
  {root}/jsonl/data_{shard:06d}.jsonl
  {root}/images/data_{shard:06d}.tar
  {root}/metadata.json              optional

Performance model:
  - Startup discovery only counts shard pairs.
  - Dataset open builds a lightweight JSONL byte-offset index.
  - Full records are parsed lazily for visible rows and raw JSON.
  - Browser cards use image URLs; images are served as original bytes by default.
  - 3D overlays are SVG layers generated from the current page payload.
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import html
import io
import json
import mimetypes
import os
import random
import socket
import tarfile
import threading
from collections import Counter, OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import quote

from flask import Flask, Response, jsonify, render_template_string, request
from PIL import Image, ImageDraw, ImageOps

import visualize_server as ann_viz


app = Flask(__name__)
DATA_DIR = "output/pangu_ml"

DEFAULT_INDEX_WORKERS = max(1, min(os.cpu_count() or 8, 32))
DEFAULT_PAGE_SIZE = 24
DEFAULT_MAX_SAMPLES_PER_DATASET = 1_000_000
ALL_FILTER = ""
BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)
BOX_COLORS = (
    "#ea580c", "#0284c7", "#16a34a", "#9333ea",
    "#d97706", "#e11d48", "#0d9488", "#4f46e5",
)

_SETTINGS = {
    "index_workers": DEFAULT_INDEX_WORKERS,
    "page_size": DEFAULT_PAGE_SIZE,
    "max_samples_per_dataset": DEFAULT_MAX_SAMPLES_PER_DATASET,
    "tar_member_cache_items": 128,
}

_DATASET_CACHE: dict[str, "DatasetIndex"] = {}
_DATASET_CACHE_LOCK = threading.Lock()
_DATASET_BUILD_LOCKS: dict[str, threading.Lock] = {}
_TAR_MEMBER_CACHE: "OrderedDict[tuple[str, float], dict[str, tarfile.TarInfo]]" = OrderedDict()
_TAR_MEMBER_CACHE_LOCK = threading.Lock()


def _log(message: str) -> None:
    print(message, flush=True)


def _json_response(payload: dict[str, Any], status: int = 200) -> Response:
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if len(raw) >= 4096:
        raw = gzip.compress(raw, compresslevel=6)
        headers["Content-Encoding"] = "gzip"
    return Response(raw, status=status, headers=headers)


def _json_error(message: str, status: int = 400) -> Response:
    return _json_response({"error": message}, status=status)


def _normalize_member_name(value: str) -> str:
    name = str(value or "").replace("\\", "/").lstrip("/")
    name = Path(name).as_posix()
    if name in {"", "."} or name.startswith("../"):
        return ""
    return name


def _flat_image_key(image_key: str) -> str:
    image_key = _normalize_member_name(image_key)
    if not image_key:
        return ""
    parts = [p for p in image_key.split("/") if p]
    if len(parts) >= 2:
        prefix = parts[:-2]
        parent = parts[-2].replace(".", "_")
        return "/".join([*prefix, f"{parent}_{parts[-1]}"])
    return image_key.replace("/", "_")


def _candidate_member_names(image_key: str) -> list[str]:
    raw = _normalize_member_name(image_key)
    flat = _flat_image_key(raw)
    names: list[str] = []
    for candidate in (raw, flat, Path(raw).name if raw else ""):
        if candidate and candidate not in names:
            names.append(candidate)
    return names


def _guess_image_mime_type(image_key: str) -> str:
    mime_type, _ = mimetypes.guess_type(image_key)
    if mime_type and mime_type.startswith("image/"):
        return mime_type
    return "image/png"


def _shard_id_from_stem(stem: str) -> str | None:
    if not stem.startswith("data_"):
        return None
    shard_id = stem.split("_", 1)[1]
    return shard_id if shard_id.isdigit() else None


def _dataset_mtime(root: Path) -> float:
    mtimes: list[float] = []
    for folder, suffix in (("jsonl", "*.jsonl"), ("images", "*.tar")):
        base = root / folder
        if not base.is_dir():
            continue
        for path in base.glob(suffix):
            try:
                mtimes.append(path.stat().st_mtime)
            except OSError:
                continue
    meta = root / "metadata.json"
    if meta.is_file():
        try:
            mtimes.append(meta.stat().st_mtime)
        except OSError:
            pass
    return max(mtimes) if mtimes else 0.0


def _iter_jsonl_offsets(path: Path, *, max_rows: int | None = None):
    offset = 0
    yielded = 0
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            length = len(line)
            if line.strip():
                yield line_number, offset, length, line
                yielded += 1
                if max_rows is not None and yielded >= max_rows:
                    break
            offset += length


def extract_text(turn: dict[str, Any]) -> str:
    chunks: list[str] = []
    for part in turn.get("content") or []:
        if not isinstance(part, dict) or part.get("type") != "text":
            continue
        text_obj = part.get("text")
        if isinstance(text_obj, dict):
            value = text_obj.get("string")
        else:
            value = text_obj
        if isinstance(value, str):
            chunks.append(value)
    return "\n".join(chunks)


def extract_image_entries(sample: dict[str, Any]) -> list[dict[str, Any]]:
    images: list[dict[str, Any]] = []
    for turn_index, turn in enumerate(sample.get("data") or []):
        if not isinstance(turn, dict):
            continue
        for content_index, part in enumerate(turn.get("content") or []):
            if not isinstance(part, dict) or part.get("type") != "image":
                continue
            image = part.get("image")
            if not isinstance(image, dict):
                continue
            item = dict(image)
            item["turn_index"] = turn_index
            item["content_index"] = content_index
            images.append(item)
    return images


def pangu_to_display_turns(sample: dict[str, Any], max_turns: int | None = None) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    data = sample.get("data") or []
    i = 0
    while i < len(data):
        user = data[i]
        if not isinstance(user, dict) or user.get("role") != "user":
            i += 1
            continue
        assistant = data[i + 1] if i + 1 < len(data) else {}
        if not isinstance(assistant, dict) or assistant.get("role") != "assistant":
            i += 1
            continue
        question = extract_text(user)
        answer = extract_text(assistant)
        turns.append({
            "question": question,
            "answer": answer,
            "task_name": _infer_turn_task_name(question, answer),
        })
        if max_turns is not None and len(turns) >= max_turns:
            break
        i += 2
    return turns


def _infer_turn_task_name(question: str, answer: str) -> str:
    if "bbox_3d" in answer or "Camera intrinsic parameters" in question:
        return "3d_grounding"
    return ""


def pangu_to_parsed(sample: dict[str, Any]) -> dict[str, Any]:
    return {"turns": pangu_to_display_turns(sample, None), "tags": [], "meta": {}}


def is_3d_grounding_sample(sample: dict[str, Any]) -> bool:
    return ann_viz._is_3d_grounding_task("", pangu_to_parsed(sample))


@dataclass
class SampleRef:
    id: str
    jsonl_path: Path
    tar_path: Path
    line_number: int
    byte_offset: int
    byte_length: int
    shard_id: str
    image_count: int = 0
    turn_count: int = 0
    is_mcq: bool = False
    is_3d_grounding: bool = False
    first_image_key: str = ""
    image_width: int = 0
    image_height: int = 0
    raw: dict[str, Any] | None = field(default=None, repr=False)

    def raw_record(self) -> dict[str, Any]:
        if self.raw is not None:
            return self.raw
        with self.jsonl_path.open("rb") as handle:
            handle.seek(self.byte_offset)
            line = handle.read(self.byte_length)
        self.raw = json.loads(line.decode("utf-8").strip())
        return self.raw


@dataclass(frozen=True)
class ShardIndex:
    shard_id: str
    jsonl_path: Path
    tar_path: Path
    line_count: int
    start_index: int


@dataclass(frozen=True)
class _ShardBuildResult:
    shard_id: str
    jsonl_path: str
    tar_path: str
    samples: tuple[SampleRef, ...]


def _facts_from_record(
    record: dict[str, Any],
    *,
    sample_id: str,
    jsonl_path: Path,
    tar_path: Path,
    line_number: int,
    byte_offset: int,
    byte_length: int,
    shard_id: str,
) -> SampleRef:
    turns = pangu_to_display_turns(record, None)
    images = extract_image_entries(record)
    first = images[0] if images else {}
    all_text = "\n".join([t["question"] + "\n" + t["answer"] for t in turns])
    return SampleRef(
        id=sample_id,
        jsonl_path=jsonl_path,
        tar_path=tar_path,
        line_number=line_number,
        byte_offset=byte_offset,
        byte_length=byte_length,
        shard_id=shard_id,
        image_count=len(images),
        turn_count=len(turns),
        is_mcq="options:" in all_text.lower(),
        is_3d_grounding=("bbox_3d" in all_text or "Camera intrinsic parameters" in all_text),
        first_image_key=str(first.get("relative_path") or first.get("path") or "").strip(),
        image_width=int(first.get("width") or 0),
        image_height=int(first.get("height") or 0),
    )


def _index_one_shard(job: tuple[str, str, str, int]) -> _ShardBuildResult:
    shard_id, jsonl_path_str, tar_path_str, max_rows = job
    jsonl_path = Path(jsonl_path_str)
    tar_path = Path(tar_path_str)
    samples: list[SampleRef] = []
    for line_number, offset, length, line in _iter_jsonl_offsets(jsonl_path, max_rows=max_rows):
        sample_id = f"{shard_id}:{line_number}"
        try:
            record = json.loads(line.decode("utf-8").strip())
        except json.JSONDecodeError:
            record = {}
        samples.append(
            _facts_from_record(
                record,
                sample_id=sample_id,
                jsonl_path=jsonl_path,
                tar_path=tar_path,
                line_number=line_number,
                byte_offset=offset,
                byte_length=length,
                shard_id=shard_id,
            )
        )
    return _ShardBuildResult(
        shard_id=shard_id,
        jsonl_path=str(jsonl_path),
        tar_path=str(tar_path),
        samples=tuple(samples),
    )


@dataclass
class DatasetIndex:
    root: Path
    mtime: float
    shards: list[ShardIndex]
    samples: list[SampleRef]
    index_seconds: float
    filter_cache: dict[str, list[int]] = field(default_factory=dict)
    order_cache: dict[tuple[str, bool, int], list[int]] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return len(self.samples)

    def summary(self) -> dict[str, Any]:
        image_counts = Counter()
        turn_counts = Counter()
        for sample in self.samples:
            image_counts[str(sample.image_count)] += 1
            turn_counts[str(sample.turn_count)] += 1
        return {
            "root": str(self.root),
            "name": self.root.name,
            "sample_count": self.total,
            "shard_count": len(self.shards),
            "index_seconds": round(self.index_seconds, 3),
            "image_count_histogram": dict(sorted(image_counts.items(), key=lambda x: int(x[0]))),
            "turn_count_histogram": dict(sorted(turn_counts.items(), key=lambda x: int(x[0]))),
        }

    def matching_indices(self, filter_kind: str) -> list[int]:
        key = str(filter_kind or "")
        cached = self.filter_cache.get(key)
        if cached is not None:
            return cached
        if key == ALL_FILTER:
            indices = list(range(self.total))
        else:
            indices = [i for i, sample in enumerate(self.samples) if _fact_matches(sample, key)]
        self.filter_cache[key] = indices
        return indices

    def ordered_indices(self, filter_kind: str, *, shuffle: bool, seed: int) -> list[int]:
        key = (str(filter_kind or ""), bool(shuffle), int(seed) if shuffle else 0)
        cached = self.order_cache.get(key)
        if cached is not None:
            return cached
        indices = list(self.matching_indices(filter_kind))
        if shuffle:
            rng = random.Random(f"{self.root}:{filter_kind}:{seed}")
            rng.shuffle(indices)
        self.order_cache[key] = indices
        return indices


def _fact_matches(sample: SampleRef, filter_kind: str) -> bool:
    if filter_kind == "3d_grounding":
        return sample.is_3d_grounding
    if filter_kind == "mcq":
        return sample.is_mcq
    if filter_kind == "oe":
        return not sample.is_mcq and not sample.is_3d_grounding
    if filter_kind == "turns_1":
        return sample.turn_count == 1
    if filter_kind == "turns_2plus":
        return sample.turn_count >= 2
    if filter_kind == "single_image":
        return sample.image_count == 1
    if filter_kind == "multi_image":
        return sample.image_count >= 2
    return True


def discover_pangu_roots(data_dir: str) -> list[dict[str, Any]]:
    base = Path(data_dir).expanduser()
    candidates = [base]
    if base.is_dir():
        candidates.extend(child for child in sorted(base.iterdir()) if child.is_dir())
    roots: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in candidates:
        jsonl_dir = root / "jsonl"
        images_dir = root / "images"
        if not jsonl_dir.is_dir() or not images_dir.is_dir():
            continue
        tar_ids = {
            sid
            for path in images_dir.glob("data_*.tar")
            if (sid := _shard_id_from_stem(path.stem)) is not None
        }
        jsonl_count = sum(
            1
            for path in jsonl_dir.glob("data_*.jsonl")
            if _shard_id_from_stem(path.stem) in tar_ids
        )
        if jsonl_count <= 0:
            continue
        resolved = str(root.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        roots.append({
            "path": resolved,
            "name": root.name,
            "label": f"{root.name} ({jsonl_count} shards)",
            "shard_count": jsonl_count,
        })
    return roots


def _build_dataset_index(root: Path) -> DatasetIndex:
    started = perf_counter()
    jsonl_dir = root / "jsonl"
    images_dir = root / "images"
    tar_map: dict[str, Path] = {}
    for tar_path in sorted(images_dir.glob("data_*.tar")):
        sid = _shard_id_from_stem(tar_path.stem)
        if sid is not None:
            tar_map[sid] = tar_path.resolve()

    jobs: list[tuple[str, str, str, int]] = []
    remaining = int(_SETTINGS["max_samples_per_dataset"])
    for jsonl_path in sorted(jsonl_dir.glob("data_*.jsonl")):
        sid = _shard_id_from_stem(jsonl_path.stem)
        if sid is None or sid not in tar_map or remaining <= 0:
            continue
        jobs.append((sid, str(jsonl_path.resolve()), str(tar_map[sid]), remaining))

    if not jobs:
        return DatasetIndex(root=root, mtime=_dataset_mtime(root), shards=[], samples=[], index_seconds=0.0)

    workers = max(1, min(int(_SETTINGS["index_workers"]), len(jobs)))
    _log(f"Indexing {root} with {workers} worker(s), {len(jobs)} shard(s)")
    chunks: list[_ShardBuildResult] = []
    if workers == 1:
        for job in jobs:
            chunk = _index_one_shard(job)
            chunks.append(chunk)
            _log(f"  shard {chunk.shard_id}: {len(chunk.samples)} samples")
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_index_one_shard, job): job[0] for job in jobs}
            for future in as_completed(futures):
                chunk = future.result()
                chunks.append(chunk)
                _log(f"  shard {chunk.shard_id}: {len(chunk.samples)} samples")

    chunks.sort(key=lambda c: int(c.shard_id))
    samples: list[SampleRef] = []
    shards: list[ShardIndex] = []
    running = 0
    max_samples = int(_SETTINGS["max_samples_per_dataset"])
    for chunk in chunks:
        if len(samples) >= max_samples:
            break
        take = list(chunk.samples[: max_samples - len(samples)])
        shards.append(
            ShardIndex(
                shard_id=chunk.shard_id,
                jsonl_path=Path(chunk.jsonl_path),
                tar_path=Path(chunk.tar_path),
                line_count=len(take),
                start_index=running,
            )
        )
        samples.extend(take)
        running += len(take)

    return DatasetIndex(
        root=root,
        mtime=_dataset_mtime(root),
        shards=shards,
        samples=samples,
        index_seconds=perf_counter() - started,
    )


def _build_lock_for(root_path: str) -> threading.Lock:
    with _DATASET_CACHE_LOCK:
        lock = _DATASET_BUILD_LOCKS.get(root_path)
        if lock is None:
            lock = threading.Lock()
            _DATASET_BUILD_LOCKS[root_path] = lock
        return lock


def get_dataset(root_path: str) -> DatasetIndex:
    if not root_path:
        raise ValueError("missing root")
    root = Path(root_path).expanduser().resolve()
    if not (root / "jsonl").is_dir() or not (root / "images").is_dir():
        raise FileNotFoundError(f"not a Pangu ML dataset root: {root}")
    key = str(root)
    mtime = _dataset_mtime(root)
    cached = _DATASET_CACHE.get(key)
    if cached is not None and cached.mtime == mtime:
        return cached
    lock = _build_lock_for(key)
    with lock:
        cached = _DATASET_CACHE.get(key)
        if cached is not None and cached.mtime == mtime:
            return cached
        index = _build_dataset_index(root)
        _DATASET_CACHE[key] = index
        _log(f"Index ready: {index.total} samples from {len(index.shards)} shards in {index.index_seconds:.2f}s")
        return index


def _read_metadata(root: Path) -> dict[str, Any] | None:
    path = root / "metadata.json"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else None


def _sample_images(sample: SampleRef, record: dict[str, Any]) -> list[dict[str, Any]]:
    images = []
    for idx, image in enumerate(extract_image_entries(record)):
        key = str(image.get("relative_path") or image.get("path") or "").strip()
        width = int(image.get("width") or 0)
        height = int(image.get("height") or 0)
        images.append({
            "index": idx,
            "image_key": key,
            "width": width,
            "height": height,
            "turn_index": int(image.get("turn_index") or 0),
            "url": f"/sample-image?root={quote(str(sample.jsonl_path.parent.parent), safe='')}&index={{ROW_INDEX}}&image_index={idx}",
            "is_primary": idx == 0,
        })
    return images


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _page_payload(
    dataset: DatasetIndex,
    *,
    page: int,
    per_page: int,
    filter_kind: str,
    shuffle: bool,
    seed: int,
    max_turns: int | None,
) -> dict[str, Any]:
    per_page = max(1, min(per_page, 150))
    if not shuffle and not filter_kind:
        total = dataset.total
        ordered: list[int] | None = None
    else:
        ordered = dataset.ordered_indices(filter_kind, shuffle=shuffle, seed=seed)
        total = len(ordered)
    page_count = max(1, (total + per_page - 1) // per_page)
    page = max(1, min(page, page_count))
    start = (page - 1) * per_page
    if ordered is None:
        selected = list(range(start, min(start + per_page, total)))
    else:
        selected = ordered[start : start + per_page]

    rows = []
    for row_index in selected:
        sample = dataset.samples[row_index]
        record = sample.raw_record()
        turns = pangu_to_display_turns(record, max_turns)
        images = _sample_images(sample, record)
        for item in images:
            item["url"] = (
                f"/sample-image?root={quote(str(dataset.root), safe='')}"
                f"&index={row_index}&image_index={item['index']}"
            )
        primary_size = (
            (images[0]["width"], images[0]["height"])
            if images
            else (sample.image_width, sample.image_height)
        )
        rows.append({
            "row_index": row_index,
            "sample_id": record.get("id") or sample.id,
            "line_number": sample.line_number,
            "shard_id": sample.shard_id,
            "turns": turns,
            "turn_count": sample.turn_count,
            "image_count": len(images),
            "preview_images": images[:4],
            "images": images,
            "is_mcq": sample.is_mcq,
            "can_3d_overlay": sample.is_3d_grounding,
            "overlay_svg": _svg_3d_overlay(record, primary_size) if sample.is_3d_grounding else "",
        })

    return {
        "dataset": dataset.summary(),
        "page": page,
        "per_page": per_page,
        "total": total,
        "page_count": page_count,
        "filter_kind": filter_kind,
        "shuffle": shuffle,
        "seed": seed,
        "rows": rows,
    }


def _svg_3d_overlay(record: dict[str, Any], image_size: tuple[int, int]) -> str:
    width, height = image_size
    if width <= 0 or height <= 0:
        return ""
    try:
        entries, ctx = ann_viz._resolve_grounding_context({}, pangu_to_parsed(record))
        if not entries or ctx is None:
            return ""
        intrinsic, ref_size = ctx
        k = intrinsic
        if ref_size is not None:
            k = ann_viz._scale_intrinsic_to_image(k, ref_size, (width, height))
        parts = [
            f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none" '
            'xmlns="http://www.w3.org/2000/svg">'
        ]
        for idx, entry in enumerate(entries):
            color = BOX_COLORS[idx % len(BOX_COLORS)]
            corners = ann_viz.compute_box_3d_corners_from_params(entry["bbox_3d"])
            uv, valid = ann_viz._project_cam_to_2d(corners, k)
            for i0, i1 in BOX_EDGES:
                if not (valid[i0] and valid[i1]):
                    continue
                x0, y0 = uv[i0]
                x1, y1 = uv[i1]
                parts.append(
                    f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" '
                    f'stroke="{color}" stroke-width="3" vector-effect="non-scaling-stroke" />'
                )
            valid_points = [uv[i] for i, ok in enumerate(valid) if ok]
            if valid_points:
                x, y = valid_points[0]
                label = html.escape(str(entry.get("label") or f"box {idx + 1}"))
                parts.append(
                    f'<text x="{x + 6:.2f}" y="{y - 6:.2f}" fill="{color}" '
                    'font-family="system-ui,sans-serif" font-size="14" font-weight="700" '
                    f'paint-order="stroke" stroke="white" stroke-width="4">{label}</text>'
                )
        parts.append("</svg>")
        return "".join(parts)
    except Exception as exc:
        _log(f"[overlay] failed: {exc}")
        return ""


def _locate_shard(dataset: DatasetIndex, row_index: int) -> ShardIndex | None:
    starts = [shard.start_index for shard in dataset.shards]
    pos = bisect.bisect_right(starts, row_index) - 1
    if pos < 0:
        return None
    shard = dataset.shards[pos]
    if row_index >= shard.start_index + shard.line_count:
        return None
    return shard


def _tar_members(tar_path: Path) -> dict[str, tarfile.TarInfo]:
    try:
        mtime = tar_path.stat().st_mtime
    except OSError:
        mtime = 0.0
    key = (str(tar_path.resolve()), mtime)
    with _TAR_MEMBER_CACHE_LOCK:
        cached = _TAR_MEMBER_CACHE.get(key)
        if cached is not None:
            _TAR_MEMBER_CACHE.move_to_end(key)
            return cached
    members: dict[str, tarfile.TarInfo] = {}
    with tarfile.open(tar_path, "r") as handle:
        for member in handle.getmembers():
            if not member.isfile():
                continue
            normalized = _normalize_member_name(member.name)
            if not normalized:
                continue
            members.setdefault(normalized, member)
            flat = _flat_image_key(normalized)
            if flat:
                members.setdefault(flat, member)
            members.setdefault(Path(normalized).name, member)
    with _TAR_MEMBER_CACHE_LOCK:
        _TAR_MEMBER_CACHE[key] = members
        _TAR_MEMBER_CACHE.move_to_end(key)
        while len(_TAR_MEMBER_CACHE) > int(_SETTINGS["tar_member_cache_items"]):
            _TAR_MEMBER_CACHE.popitem(last=False)
    return members


def _load_image_bytes(tar_path: Path, image_key: str) -> tuple[bytes, str]:
    members = _tar_members(tar_path)
    candidates = _candidate_member_names(image_key)
    with tarfile.open(tar_path, "r") as handle:
        for candidate in candidates:
            member = members.get(candidate)
            if member is None:
                continue
            stream = handle.extractfile(member)
            if stream is None:
                continue
            return stream.read(), _guess_image_mime_type(candidate)
    raise FileNotFoundError(f"image not found in {tar_path}: {image_key}")


def _render_overlay_png(content: bytes, record: dict[str, Any]) -> bytes:
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(content))).convert("RGB")
    parsed = pangu_to_parsed(record)
    entries, ctx = ann_viz._resolve_grounding_context({}, parsed)
    if entries and ctx is not None:
        intrinsic, ref_size = ctx
        image = ann_viz.draw_3d_boxes_on_image(image, entries, intrinsic, ref_size=ref_size)
    else:
        d = ImageDraw.Draw(image)
        d.text((16, 16), "No 3D overlay context", fill=(220, 38, 38))
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def _sanitize_raw(value: Any, *, depth: int = 0, state: dict[str, bool] | None = None) -> Any:
    if state is None:
        state = {"truncated": False}
    if depth > 8:
        state["truncated"] = True
        return "<truncated: depth>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) <= 12000:
            return value
        state["truncated"] = True
        return value[:12000] + f" ... <truncated {len(value) - 12000} chars>"
    if isinstance(value, list):
        limit = 256
        out = [_sanitize_raw(item, depth=depth + 1, state=state) for item in value[:limit]]
        if len(value) > limit:
            state["truncated"] = True
            out.append(f"... <truncated {len(value) - limit} items>")
        return out
    if isinstance(value, dict):
        limit = 256
        items = list(value.items())
        out = {
            str(k): _sanitize_raw(v, depth=depth + 1, state=state)
            for k, v in items[:limit]
        }
        if len(items) > limit:
            state["truncated"] = True
            out["__truncated_keys__"] = len(items) - limit
        return out
    state["truncated"] = True
    return str(value)


HTML_TEMPLATE = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Pangu ML Viewer</title>
<style>
:root { --bg:#f6f7fb; --panel:#fff; --line:#dbe3ee; --muted:#64748b; --accent:#4f46e5; --soft:#eef2ff; --danger:#be123c; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:#111827; font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; font-size:14px; }
header { position:sticky; top:0; z-index:10; background:rgba(255,255,255,.96); border-bottom:1px solid var(--line); backdrop-filter:blur(8px); }
.bar { max-width:1500px; margin:0 auto; padding:14px 18px; display:flex; gap:10px; align-items:end; flex-wrap:wrap; }
h1 { margin:0; font-size:18px; }
label { display:block; color:var(--muted); font-size:12px; margin-bottom:5px; }
select,input,button { height:38px; border:1px solid var(--line); border-radius:8px; background:white; color:#111827; padding:7px 10px; font:inherit; }
button { cursor:pointer; }
button.primary { background:var(--accent); color:white; border-color:var(--accent); }
.grow { flex:1 1 360px; min-width:280px; }
.narrow { width:110px; }
main { max-width:1500px; margin:0 auto; padding:16px 18px 48px; }
.status { color:var(--muted); text-align:center; margin:8px 0 14px; }
.pager { display:flex; justify-content:center; gap:8px; align-items:center; margin:10px 0 16px; flex-wrap:wrap; }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(380px,1fr)); gap:14px; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:12px; overflow:hidden; box-shadow:0 10px 24px rgba(15,23,42,.06); }
.visuals { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; background:#f8fafc; padding:8px; }
.visual { position:relative; min-height:160px; background:#e5e7eb; overflow:hidden; cursor:zoom-in; display:flex; align-items:center; justify-content:center; }
.visual.single { grid-column:1 / -1; }
.visual img { width:100%; display:block; object-fit:contain; }
.visual .overlay, .lightbox-visual .overlay { position:absolute; inset:0; pointer-events:none; }
.visual .overlay svg, .lightbox-visual .overlay svg { width:100%; height:100%; display:block; }
.placeholder { min-height:160px; color:var(--muted); display:flex; align-items:center; justify-content:center; padding:18px; text-align:center; }
.body { padding:12px; }
.meta { color:var(--muted); font-size:12px; line-height:1.45; margin-bottom:10px; word-break:break-word; }
.tag { display:inline-block; border-radius:999px; padding:2px 8px; font-size:12px; font-weight:650; background:var(--soft); color:#3730a3; margin-right:4px; }
.tag.warn { background:#ffedd5; color:#9a3412; }
.turn { margin-top:10px; }
.turn-title { font-size:12px; color:var(--muted); margin-bottom:4px; }
.qa { white-space:pre-wrap; line-height:1.55; padding:9px 10px; border-radius:8px; overflow-wrap:anywhere; }
.qa.q { background:#eef2ff; }
.qa.a { background:#ecfdf5; margin-top:6px; }
.actions { display:flex; gap:8px; flex-wrap:wrap; margin:8px 0 4px; }
.raw { display:none; max-height:460px; overflow:auto; background:#111827; color:#e5e7eb; padding:12px; border-radius:8px; white-space:pre-wrap; font:12px ui-monospace,SFMono-Regular,Menlo,monospace; }
.raw.open { display:block; }
dialog { border:none; border-radius:12px; padding:0; box-shadow:0 20px 60px rgba(0,0,0,.28); }
dialog::backdrop { background:rgba(15,23,42,.72); }
.lightbox { width:96vw; height:92vh; background:#030712; color:white; overflow:hidden; }
.lightbox-head { height:44px; display:flex; align-items:center; justify-content:space-between; padding:0 14px; background:#111827; }
.lightbox-stage { position:relative; width:100%; height:calc(100% - 44px); overflow:auto; display:flex; align-items:center; justify-content:center; }
.lightbox-visual { position:relative; transform-origin:center center; }
.lightbox-visual img { max-width:92vw; max-height:84vh; display:block; }
.stats { background:white; border:1px solid var(--line); border-radius:12px; padding:12px; margin-bottom:14px; display:none; }
.stats.open { display:block; }
.stats-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:8px; }
.metric { background:#f8fafc; border:1px solid #edf2f7; border-radius:10px; padding:10px; }
.metric b { display:block; font-size:18px; margin-top:3px; }
</style>
</head>
<body>
<header>
  <div class="bar">
    <div><h1>Pangu ML Viewer</h1><label>Dataset</label></div>
    <select id="root" class="grow">
      <option value="">Select dataset</option>
      {% for root in roots %}
      <option value="{{ root.path }}" {% if selected == root.path %}selected{% endif %}>{{ root.label }}</option>
      {% endfor %}
    </select>
    <div><label>Filter</label><select id="filter">
      <option value="">All</option><option value="turns_1">Single-turn</option><option value="turns_2plus">Multi-turn</option>
      <option value="mcq">MCQ</option><option value="oe">Open-ended</option><option value="3d_grounding">3D grounding</option>
      <option value="single_image">1 image</option><option value="multi_image">2+ images</option>
    </select></div>
    <div><label>Per page</label><input id="perPage" class="narrow" type="number" min="1" max="150" value="{{ page_size }}" /></div>
    <div><label>Turns</label><input id="turns" class="narrow" value="2" title="Use max for all turns" /></div>
    <div><label>Page</label><input id="page" class="narrow" type="number" min="1" value="1" /></div>
    <div><label>Seed</label><input id="seed" class="narrow" type="number" value="20260414" /></div>
    <label style="height:38px;display:flex;align-items:center;gap:6px;margin:0;"><input id="shuffle" type="checkbox" checked /> Random</label>
    <label style="height:38px;display:flex;align-items:center;gap:6px;margin:0;"><input id="boxes" type="checkbox" checked /> Overlay</label>
    <button id="load" class="primary">Load</button>
  </div>
</header>
<main>
  <div class="status" id="status">Ready.</div>
  <section class="stats" id="stats"></section>
  <div class="pager"><button id="prev">Prev</button><span id="pageInfo">-</span><button id="next">Next</button></div>
  <section class="grid" id="grid"></section>
  <div class="pager"><button id="prev2">Prev</button><span id="pageInfo2">-</span><button id="next2">Next</button></div>
</main>
<dialog class="lightbox" id="lightbox">
  <div class="lightbox-head"><span id="lightboxTitle"></span><button id="closeLightbox">Close</button></div>
  <div class="lightbox-stage"><div class="lightbox-visual" id="lightboxVisual"></div></div>
</dialog>
<script>
const state = { current:null, token:0, imageToken:0 };
const IMAGE_CONCURRENCY = 10;
const els = {
  root: document.getElementById('root'), filter: document.getElementById('filter'), perPage: document.getElementById('perPage'),
  turns: document.getElementById('turns'), page: document.getElementById('page'), seed: document.getElementById('seed'),
  shuffle: document.getElementById('shuffle'), boxes: document.getElementById('boxes'), load: document.getElementById('load'),
  status: document.getElementById('status'), stats: document.getElementById('stats'), grid: document.getElementById('grid'),
  pageInfo: document.getElementById('pageInfo'), pageInfo2: document.getElementById('pageInfo2'),
  prev: document.getElementById('prev'), next: document.getElementById('next'), prev2: document.getElementById('prev2'), next2: document.getElementById('next2'),
  lightbox: document.getElementById('lightbox'), lightboxTitle: document.getElementById('lightboxTitle'), lightboxVisual: document.getElementById('lightboxVisual'),
  closeLightbox: document.getElementById('closeLightbox'),
};
function esc(v) {
  return String(v ?? '').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'","&#39;");
}
async function fetchJson(url) {
  const res = await fetch(url, {cache:'no-store'});
  if (!res.ok) throw new Error(await res.text() || res.statusText);
  return res.json();
}
function params() {
  const p = new URLSearchParams();
  p.set('root', els.root.value);
  p.set('filter_kind', els.filter.value);
  p.set('page', els.page.value || '1');
  p.set('per_page', els.perPage.value || '24');
  p.set('turns', els.turns.value || '2');
  p.set('shuffle', els.shuffle.checked ? '1' : '0');
  p.set('seed', els.seed.value || '20260414');
  return p;
}
async function loadPage() {
  if (!els.root.value) { els.status.textContent = 'Select a dataset.'; return; }
  const token = ++state.token;
  ++state.imageToken;
  els.grid.innerHTML = '<div class="placeholder">Loading page...</div>';
  const started = performance.now();
  try {
    const data = await fetchJson('/api/samples?' + params().toString());
    if (token !== state.token) return;
    state.current = data;
    els.page.value = data.page;
    renderStats(data.dataset);
    renderPager(data);
    els.status.textContent = `${data.dataset.name} · current ${data.rows.length} · total ${data.total} · page ${data.page}/${data.page_count} · loaded ${((performance.now() - started) / 1000).toFixed(2)}s`;
    els.grid.innerHTML = (data.rows || []).map(cardHtml).join('');
    bindCards();
    queueImages();
  } catch (err) {
    if (token !== state.token) return;
    els.grid.innerHTML = `<div class="placeholder">Load failed: ${esc(err.message || err)}</div>`;
    els.status.textContent = 'Load failed.';
  }
}
function renderStats(ds) {
  if (!ds) { els.stats.classList.remove('open'); return; }
  const metric = (label, value) => `<div class="metric"><span>${esc(label)}</span><b>${esc(value)}</b></div>`;
  els.stats.classList.add('open');
  els.stats.innerHTML = `<div class="stats-grid">
    ${metric('Samples', Number(ds.sample_count || 0).toLocaleString())}
    ${metric('Shards', ds.shard_count || 0)}
    ${metric('Index Seconds', ds.index_seconds || 0)}
    ${metric('Root', ds.root || '')}
  </div>`;
}
function renderPager(data) {
  const txt = `page ${data.page} / ${data.page_count}`;
  els.pageInfo.textContent = txt; els.pageInfo2.textContent = txt;
  const atFirst = data.page <= 1, atLast = data.page >= data.page_count;
  for (const b of [els.prev, els.prev2]) b.disabled = atFirst;
  for (const b of [els.next, els.next2]) b.disabled = atLast;
}
function imageHtml(row) {
  const images = row.preview_images || [];
  if (!images.length) return '<div class="placeholder">No images</div>';
  return `<div class="visuals">${images.map((img, i) => {
    const cls = images.length === 1 ? 'visual single' : 'visual';
    const overlay = img.is_primary && row.overlay_svg ? `<div class="overlay" style="display:${els.boxes.checked ? 'block' : 'none'}">${row.overlay_svg}</div>` : '';
    return `<div class="${cls}" data-row="${row.row_index}" data-image="${img.index}">
      <img data-src="${esc(img.url)}" alt="image ${i + 1}" />${overlay}</div>`;
  }).join('')}</div>`;
}
function cardHtml(row) {
  const tags = [
    `<span class="tag">#${row.row_index}</span>`,
    row.can_3d_overlay ? '<span class="tag warn">3D</span>' : '',
    row.is_mcq ? '<span class="tag">MCQ</span>' : '',
  ].join('');
  const turns = (row.turns || []).map((t, i) => `<div class="turn">
    <div class="turn-title">Turn ${i + 1}${t.task_name ? ' · ' + esc(t.task_name) : ''}</div>
    <div class="qa q">${esc(t.question)}</div><div class="qa a">${esc(t.answer)}</div>
  </div>`).join('');
  return `<article class="card" data-card="${row.row_index}">
    ${imageHtml(row)}
    <div class="body">
      <div class="meta">${tags}<br />sample: ${esc(row.sample_id)} · shard ${esc(row.shard_id)} · line ${row.line_number} · turns ${row.turn_count} · images ${row.image_count}</div>
      <div class="actions"><button data-raw="${row.row_index}">Raw JSON</button><button data-images="${row.row_index}">All Images</button></div>
      ${turns}<pre class="raw" id="raw-${row.row_index}"></pre>
    </div>
  </article>`;
}
function queueImages() {
  const token = ++state.imageToken;
  const imgs = Array.from(els.grid.querySelectorAll('img[data-src]'));
  let next = 0, active = 0;
  function pump() {
    if (token !== state.imageToken) return;
    while (active < IMAGE_CONCURRENCY && next < imgs.length) {
      const img = imgs[next++], src = img.dataset.src || '';
      if (!src) continue;
      active++;
      const done = () => { if (token !== state.imageToken) return; active--; pump(); };
      img.addEventListener('load', done, {once:true});
      img.addEventListener('error', done, {once:true});
      img.src = src;
      img.removeAttribute('data-src');
    }
  }
  pump();
}
function rowByIndex(index) {
  return (state.current?.rows || []).find(r => Number(r.row_index) === Number(index));
}
function imageByIndex(row, index) {
  return (row?.images || []).find(img => Number(img.index) === Number(index));
}
function bindCards() {
  for (const visual of els.grid.querySelectorAll('.visual[data-row]')) {
    visual.addEventListener('click', () => openImage(Number(visual.dataset.row), Number(visual.dataset.image)));
  }
  for (const button of els.grid.querySelectorAll('[data-raw]')) {
    button.addEventListener('click', () => toggleRaw(Number(button.dataset.raw), button));
  }
  for (const button of els.grid.querySelectorAll('[data-images]')) {
    button.addEventListener('click', () => openImageList(Number(button.dataset.images)));
  }
}
async function toggleRaw(index, button) {
  const panel = document.getElementById(`raw-${index}`);
  if (!panel) return;
  if (panel.classList.contains('open')) { panel.classList.remove('open'); button.textContent = 'Raw JSON'; return; }
  panel.classList.add('open'); button.textContent = 'Hide Raw'; panel.textContent = 'Loading...';
  try {
    const p = new URLSearchParams({root: els.root.value, index: String(index)});
    const data = await fetchJson('/api/raw?' + p.toString());
    panel.textContent = (data.truncated ? '[truncated]\n\n' : '') + JSON.stringify(data.row, null, 2);
  } catch (err) {
    panel.textContent = `Failed: ${err.message || err}`;
  }
}
function openImage(rowIndex, imageIndex) {
  const row = rowByIndex(rowIndex), img = imageByIndex(row, imageIndex);
  if (!row || !img) return;
  const overlay = img.is_primary && row.overlay_svg ? `<div class="overlay" style="display:${els.boxes.checked ? 'block' : 'none'}">${row.overlay_svg}</div>` : '';
  els.lightboxTitle.textContent = `${row.sample_id} · image ${imageIndex + 1}`;
  els.lightboxVisual.innerHTML = `<img src="${esc(img.url)}" alt="" />${overlay}`;
  if (!els.lightbox.open) els.lightbox.showModal();
}
function openImageList(rowIndex) {
  const row = rowByIndex(rowIndex);
  if (!row || !(row.images || []).length) return;
  const html = `<div style="padding:14px;background:white;color:#111827;max-width:900px;max-height:80vh;overflow:auto;">
    <h3>All images · ${esc(row.sample_id)}</h3>
    <div class="visuals">${row.images.map(img => `<div class="visual" data-list-image="${img.index}">
      <img src="${esc(img.url)}" alt="" /><div style="position:absolute;left:6px;bottom:6px;background:rgba(0,0,0,.55);color:white;padding:2px 6px;border-radius:6px;font-size:12px;">${img.index + 1}</div>
    </div>`).join('')}</div>
  </div>`;
  els.lightboxTitle.textContent = `All images · ${row.sample_id}`;
  els.lightboxVisual.innerHTML = html;
  if (!els.lightbox.open) els.lightbox.showModal();
  for (const item of els.lightboxVisual.querySelectorAll('[data-list-image]')) {
    item.addEventListener('click', event => { event.stopPropagation(); openImage(rowIndex, Number(item.dataset.listImage)); });
  }
}
function movePage(delta) {
  const cur = Number(els.page.value || 1);
  const max = Number(state.current?.page_count || 1);
  els.page.value = String(Math.max(1, Math.min(max, cur + delta)));
  loadPage();
}
function resetAndLoad() { els.page.value = '1'; loadPage(); }
els.load.addEventListener('click', loadPage);
els.root.addEventListener('change', resetAndLoad);
els.filter.addEventListener('change', resetAndLoad);
els.perPage.addEventListener('change', resetAndLoad);
els.shuffle.addEventListener('change', resetAndLoad);
els.boxes.addEventListener('change', () => {
  for (const overlay of document.querySelectorAll('.overlay')) overlay.style.display = els.boxes.checked ? 'block' : 'none';
});
for (const b of [els.prev, els.prev2]) b.addEventListener('click', () => movePage(-1));
for (const b of [els.next, els.next2]) b.addEventListener('click', () => movePage(1));
els.page.addEventListener('keydown', e => { if (e.key === 'Enter') loadPage(); });
els.closeLightbox.addEventListener('click', () => els.lightbox.close());
document.addEventListener('keydown', e => { if (e.key === 'Escape' && els.lightbox.open) els.lightbox.close(); });
if (els.root.value) loadPage();
</script>
</body>
</html>
"""


@app.route("/")
def index() -> str:
    roots = discover_pangu_roots(DATA_DIR)
    return render_template_string(
        HTML_TEMPLATE,
        roots=roots,
        selected=request.args.get("root", ""),
        page_size=int(_SETTINGS["page_size"]),
    )


@app.route("/api/datasets")
def api_datasets() -> Response:
    return _json_response({"datasets": discover_pangu_roots(DATA_DIR)})


@app.route("/api/samples")
def api_samples() -> Response:
    root = request.args.get("root", "")
    filter_kind = request.args.get("filter_kind", "")
    page = _safe_int(request.args.get("page"), 1)
    per_page = _safe_int(request.args.get("per_page"), int(_SETTINGS["page_size"]))
    seed = _safe_int(request.args.get("seed"), 20260414)
    shuffle = request.args.get("shuffle", "1").lower() not in {"0", "false", "no", "off"}
    turns_raw = str(request.args.get("turns", "2")).strip().lower()
    max_turns = None if turns_raw == "max" else max(1, _safe_int(turns_raw, 2))
    try:
        dataset = get_dataset(root)
        return _json_response(
            _page_payload(
                dataset,
                page=page,
                per_page=per_page,
                filter_kind=filter_kind,
                shuffle=shuffle,
                seed=seed,
                max_turns=max_turns,
            )
        )
    except Exception as exc:
        _log(f"[api/samples] {exc}")
        return _json_error(str(exc), 500)


@app.route("/api/raw")
def api_raw() -> Response:
    try:
        dataset = get_dataset(request.args.get("root", ""))
        index_value = _safe_int(request.args.get("index"), -1)
        if index_value < 0 or index_value >= dataset.total:
            return _json_error("sample index out of range", 404)
        sample = dataset.samples[index_value]
        state = {"truncated": False}
        return _json_response({
            "row": _sanitize_raw(sample.raw_record(), state=state),
            "truncated": bool(state["truncated"]),
        })
    except Exception as exc:
        _log(f"[api/raw] {exc}")
        return _json_error(str(exc), 500)


@app.route("/api/metadata")
def api_metadata() -> Response:
    try:
        root = Path(request.args.get("root", "")).expanduser().resolve()
        meta = _read_metadata(root)
        return _json_response({"has_metadata": meta is not None, "metadata": meta or {}})
    except Exception as exc:
        return _json_error(str(exc), 500)


@app.route("/sample-image")
def sample_image() -> Response:
    try:
        dataset = get_dataset(request.args.get("root", ""))
        index_value = _safe_int(request.args.get("index"), -1)
        image_index = _safe_int(request.args.get("image_index"), 0)
        overlay = request.args.get("overlay", "0").lower() in {"1", "true", "yes", "on"}
        if index_value < 0 or index_value >= dataset.total:
            return _json_error("sample index out of range", 404)
        sample = dataset.samples[index_value]
        record = sample.raw_record()
        images = extract_image_entries(record)
        if image_index < 0 or image_index >= len(images):
            return _json_error("image index out of range", 404)
        image_key = str(images[image_index].get("relative_path") or images[image_index].get("path") or "").strip()
        if not image_key:
            return _json_error("image path missing", 404)
        shard = _locate_shard(dataset, index_value)
        if shard is None:
            return _json_error("shard not found", 404)
        content, mime_type = _load_image_bytes(shard.tar_path, image_key)
        if overlay and image_index == 0 and sample.is_3d_grounding:
            content = _render_overlay_png(content, record)
            mime_type = "image/png"
        return Response(content, headers={
            "Content-Type": mime_type,
            "Content-Length": str(len(content)),
            "Cache-Control": "no-store",
        })
    except FileNotFoundError as exc:
        return _json_error(str(exc), 404)
    except Exception as exc:
        _log(f"[sample-image] {exc}")
        return _json_error(str(exc), 500)


def _lan_urls(port: int) -> list[str]:
    urls: list[str] = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            ip = sock.getsockname()[0]
            if ip and not ip.startswith("127."):
                urls.append(f"http://{ip}:{port}")
    except OSError:
        pass
    return urls


def _print_listen_info(host: str, port: int) -> None:
    _log(f"\nListening on http://{host}:{port}")
    _log(f"  This machine: http://127.0.0.1:{port}")
    if host in {"0.0.0.0", "::"}:
        for url in _lan_urls(port):
            _log(f"  LAN: {url}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fast Pangu ML bundle visualizer")
    parser.add_argument("--data_dir", type=str, default="output/pangu_ml")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8891)
    parser.add_argument("--index-workers", type=int, default=DEFAULT_INDEX_WORKERS)
    parser.add_argument("--load-workers", type=int, default=None, help="Alias for --index-workers.")
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--max-samples-per-dataset", type=int, default=DEFAULT_MAX_SAMPLES_PER_DATASET)
    parser.add_argument("--tar-member-cache-items", type=int, default=128)
    parser.add_argument("--warm-index", action="store_true")
    return parser


def main() -> None:
    global DATA_DIR
    args = build_argparser().parse_args()
    DATA_DIR = args.data_dir
    _SETTINGS["index_workers"] = max(1, args.load_workers or args.index_workers)
    _SETTINGS["page_size"] = max(1, args.page_size)
    _SETTINGS["max_samples_per_dataset"] = max(1, args.max_samples_per_dataset)
    _SETTINGS["tar_member_cache_items"] = max(0, args.tar_member_cache_items)

    _log("Pangu ML visualizer starting ...")
    roots = discover_pangu_roots(DATA_DIR)
    _log(f"Found {len(roots)} dataset(s)")
    _log(
        "Settings: "
        f"index_workers={_SETTINGS['index_workers']}, "
        f"page_size={_SETTINGS['page_size']}, "
        f"max_samples_per_dataset={_SETTINGS['max_samples_per_dataset']}, "
        f"tar_member_cache_items={_SETTINGS['tar_member_cache_items']}"
    )
    if args.warm_index:
        for root in roots:
            get_dataset(root["path"])
    _print_listen_info(args.host, args.port)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
