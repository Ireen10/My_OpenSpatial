"""
Flask visualizer for Pangu ML training bundles.

Layout (per dataset root):
  {root}/jsonl/data_{shard:06d}.jsonl
  {root}/images/data_{shard:06d}.tar

Memory / I/O model (large-data safe):
  - Never loads the full dataset JSON bodies into RAM.
  - Index build: parallel per-shard scan -> byte offsets + compact filter facts.
  - Page browse: text returned immediately; images load via separate /api/images.
  - Image decode uses a thread pool (--image-workers).

Usage:
    python visualize_pangu_ml_server.py --data_dir /path/to/pangu_output --port 8891
    python visualize_pangu_ml_server.py --data_dir ... --index-workers 64 --image-workers 48
"""

from __future__ import annotations

import argparse
import io
import json
import os
import socket
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from flask import Flask, jsonify, render_template_string, request
from PIL import Image, ImageDraw

import visualize_server as ann_viz

app = Flask(__name__)
DATA_DIR = "output/pangu_ml"

_WORKERS = {
    "index": 32,
    "read": 16,
    "image": 32,
}


def _log(msg: str) -> None:
    print(msg, flush=True)

# root_path -> {mtime, shards, facts, total, filter_cache}
_ROOT_CACHE: Dict[str, dict] = {}
_ROOT_LOCK = threading.Lock()
_ROOT_BUILD_LOCKS: Dict[str, threading.Lock] = {}


@dataclass(frozen=True)
class SampleFacts:
    is_3d_grounding: bool
    image_count: int


@dataclass(frozen=True)
class ShardInfo:
    shard_id: str
    jsonl_path: Path
    tar_path: Path
    line_count: int
    start_index: int
    line_offsets: Tuple[int, ...]


def _shard_id_from_stem(stem: str) -> Optional[str]:
    if not stem.startswith("data_"):
        return None
    return stem.split("_", 1)[1]


def discover_pangu_roots(data_dir: str) -> List[dict]:
    """Fast discovery: count shard files only (no full jsonl scan)."""
    _log(f"Discovering Pangu ML datasets under {data_dir} ...")
    roots: List[dict] = []
    base = Path(data_dir)
    candidates: List[Path] = [base]
    if base.is_dir():
        for child in sorted(base.iterdir()):
            if child.is_dir():
                candidates.append(child)

    seen: set = set()
    for root in candidates:
        key = str(root.resolve())
        if key in seen:
            continue
        jsonl_dir = root / "jsonl"
        images_dir = root / "images"
        if not jsonl_dir.is_dir() or not images_dir.is_dir():
            continue

        tar_ids = {
            sid
            for p in images_dir.glob("data_*.tar")
            if (sid := _shard_id_from_stem(p.stem)) is not None
        }
        shard_count = sum(
            1
            for jf in jsonl_dir.glob("data_*.jsonl")
            if (_shard_id_from_stem(jf.stem) in tar_ids)
        )

        if shard_count == 0:
            continue
        seen.add(key)
        roots.append({
            "path": key,
            "label": f"{root.name} ({shard_count} shards, sample count on first open)",
        })
        _log(f"  found: {root.name} ({shard_count} shards)")
    _log(f"Discovery done: {len(roots)} dataset(s)")
    return roots


def _facts_from_sample(sample: dict) -> SampleFacts:
    n_img = len(extract_image_entries(sample))
    return SampleFacts(
        is_3d_grounding=is_3d_grounding_sample(sample),
        image_count=n_img,
    )


@dataclass(frozen=True)
class _ShardIndexChunk:
    shard_id: str
    jsonl_path: str
    tar_path: str
    line_offsets: Tuple[int, ...]
    facts: Tuple[SampleFacts, ...]


def _index_one_shard(job: Tuple[str, str, str]) -> _ShardIndexChunk:
    """Thread worker: scan one jsonl shard (offsets + filter facts)."""
    shard_id, jsonl_path, tar_path = job
    offsets: List[int] = []
    facts: List[SampleFacts] = []
    with Path(jsonl_path).open("rb") as f:
        while True:
            pos = f.tell()
            raw = f.readline()
            if not raw:
                break
            if not raw.strip():
                continue
            offsets.append(pos)
            sample = json.loads(raw.decode("utf-8"))
            facts.append(_facts_from_sample(sample))
    return _ShardIndexChunk(
        shard_id=shard_id,
        jsonl_path=jsonl_path,
        tar_path=tar_path,
        line_offsets=tuple(offsets),
        facts=tuple(facts),
    )


def _build_shard_index(root: Path, *, num_workers: int) -> Tuple[List[ShardInfo], List[SampleFacts]]:
    """Parallel per-shard index build (offsets + filter facts, no image I/O)."""
    jsonl_dir = root / "jsonl"
    images_dir = root / "images"

    tar_map: Dict[str, Path] = {}
    for tar_path in sorted(images_dir.glob("data_*.tar")):
        sid = _shard_id_from_stem(tar_path.stem)
        if sid is not None:
            tar_map[sid] = tar_path

    jobs: List[Tuple[str, str, str]] = []
    for jf in sorted(jsonl_dir.glob("data_*.jsonl")):
        sid = _shard_id_from_stem(jf.stem)
        if sid is None:
            continue
        tar_path = tar_map.get(sid)
        if tar_path is None:
            continue
        jobs.append((sid, str(jf.resolve()), str(tar_path.resolve())))

    if not jobs:
        return [], []

    workers = max(1, min(num_workers, len(jobs)))
    chunks: List[_ShardIndexChunk] = []
    _log(f"Indexing {len(jobs)} shard(s) with {workers} worker(s) ...")
    if workers == 1:
        for i, job in enumerate(jobs, 1):
            chunks.append(_index_one_shard(job))
            _log(f"  [{i}/{len(jobs)}] shard {job[0]} done")
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_index_one_shard, job): job[0] for job in jobs}
            done = 0
            for fut in as_completed(futures):
                sid = futures[fut]
                chunks.append(fut.result())
                done += 1
                _log(f"  [{done}/{len(jobs)}] shard {sid} done")

    chunks.sort(key=lambda c: int(c.shard_id))
    shards: List[ShardInfo] = []
    facts: List[SampleFacts] = []
    running = 0
    for chunk in chunks:
        shards.append(ShardInfo(
            shard_id=chunk.shard_id,
            jsonl_path=Path(chunk.jsonl_path),
            tar_path=Path(chunk.tar_path),
            line_count=len(chunk.line_offsets),
            start_index=running,
            line_offsets=chunk.line_offsets,
        ))
        facts.extend(chunk.facts)
        running += len(chunk.line_offsets)

    return shards, facts


def _build_lock_for(root_path: str) -> threading.Lock:
    with _ROOT_LOCK:
        if root_path not in _ROOT_BUILD_LOCKS:
            _ROOT_BUILD_LOCKS[root_path] = threading.Lock()
        return _ROOT_BUILD_LOCKS[root_path]


def _root_mtime(root_path: str) -> float:
    times: List[float] = []
    root = Path(root_path)
    for sub in ("jsonl", "images"):
        d = root / sub
        if not d.is_dir():
            continue
        for p in d.glob("data_*.*"):
            try:
                times.append(p.stat().st_mtime)
            except OSError:
                pass
    return max(times) if times else 0.0


def _get_root_cache(root_path: str) -> dict:
    mtime = _root_mtime(root_path)
    cached = _ROOT_CACHE.get(root_path)
    if cached and cached.get("mtime") == mtime:
        return cached

    lock = _build_lock_for(root_path)
    with lock:
        cached = _ROOT_CACHE.get(root_path)
        if cached and cached.get("mtime") == mtime:
            return cached
        nw = _WORKERS["index"]
        _log(f"Building sample index for {root_path} ...")
        shards, facts = _build_shard_index(Path(root_path), num_workers=nw)
        cache = {
            "mtime": mtime,
            "shards": shards,
            "facts": facts,
            "total": len(facts),
            "filter_cache": {},
        }
        _ROOT_CACHE[root_path] = cache
        _log(f"Index ready: {len(facts)} samples, {len(shards)} shards")
        return cache


def _locate_sample(shards: List[ShardInfo], index: int) -> Optional[Tuple[ShardInfo, int]]:
    if index < 0:
        return None
    for shard in shards:
        if index < shard.start_index + shard.line_count:
            return shard, index - shard.start_index
    return None


def read_sample_at(cache: dict, index: int) -> Optional[dict]:
    loc = _locate_sample(cache["shards"], index)
    if loc is None:
        return None
    shard, line_idx = loc
    if line_idx < 0 or line_idx >= len(shard.line_offsets):
        return None
    offset = shard.line_offsets[line_idx]
    with shard.jsonl_path.open("rb") as f:
        f.seek(offset)
        line = f.readline().decode("utf-8")
    if not line.strip():
        return None
    return json.loads(line)


def read_sample(root_path: str, index: int) -> Optional[dict]:
    cache = _get_root_cache(root_path)
    return read_sample_at(cache, index)


def _text_row_from_sample(cache: dict, index: int, sample: dict) -> dict:
    fact = cache["facts"][index]
    return {
        "row_index": index,
        "sample_id": sample.get("id") or f"row_{index}",
        "turns": pangu_to_display_turns(sample),
        "can_3d_overlay": fact.is_3d_grounding,
        "image_count": fact.image_count,
    }


def _build_page_text_rows(
    root_path: str,
    cache: dict,
    page_indices: List[int],
) -> List[dict]:
    if not page_indices:
        return []

    def _one(index: int) -> Optional[dict]:
        sample = read_sample_at(cache, index)
        if sample is None:
            return None
        return _text_row_from_sample(cache, index, sample)

    workers = max(1, min(_WORKERS["read"], len(page_indices)))
    if workers == 1:
        rows = [_one(i) for i in page_indices]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_one, page_indices))

    return [r for r in rows if r is not None]


def extract_turn_text(turn: dict) -> str:
    chunks: List[str] = []
    for part in turn.get("content") or []:
        if not isinstance(part, dict) or part.get("type") != "text":
            continue
        text_obj = part.get("text")
        if isinstance(text_obj, dict):
            s = text_obj.get("string")
            if isinstance(s, str):
                chunks.append(s)
    return "\n".join(chunks)


def extract_image_entries(sample: dict) -> List[dict]:
    data = sample.get("data") or []
    if not data or data[0].get("role") != "user":
        return []
    out: List[dict] = []
    for part in data[0].get("content") or []:
        if isinstance(part, dict) and part.get("type") == "image":
            img = part.get("image")
            if isinstance(img, dict):
                out.append(img)
    return out


def pangu_to_display_turns(sample: dict) -> List[dict]:
    turns: List[dict] = []
    data = sample.get("data") or []
    i = 0
    while i < len(data):
        user = data[i]
        if not isinstance(user, dict) or user.get("role") != "user":
            i += 1
            continue
        if i + 1 >= len(data):
            break
        assistant = data[i + 1]
        if not isinstance(assistant, dict) or assistant.get("role") != "assistant":
            i += 1
            continue
        q = extract_turn_text(user)
        a = extract_turn_text(assistant)
        turns.append({
            "question": q,
            "answer": a,
            "question_prefix": "",
            "task_name": _infer_turn_task_name(q, a),
        })
        i += 2
    return turns


def _infer_turn_task_name(question: str, answer: str) -> str:
    q = question or ""
    a = answer or ""
    if "Camera intrinsic parameters" in q or "bbox_3d" in a:
        return "3d_grounding"
    return ""


def pangu_to_parsed(sample: dict) -> dict:
    turns = pangu_to_display_turns(sample)
    return {"turns": turns, "tags": [], "meta": {}}


def is_3d_grounding_sample(sample: dict) -> bool:
    parsed = pangu_to_parsed(sample)
    return ann_viz._is_3d_grounding_task("", parsed)


def _fact_matches(fact: SampleFacts, filter_kind: str) -> bool:
    fk = filter_kind.strip().lower()
    if fk == "3d_grounding":
        return fact.is_3d_grounding
    if fk == "single_image":
        return fact.image_count == 1
    if fk == "multi_image":
        return fact.image_count > 1
    return True


def _matching_indices(cache: dict, filter_kind: str) -> List[int]:
    """Cached list of global indices for a filter (built once per filter)."""
    fk = (filter_kind or "").strip()
    fc: Dict[str, List[int]] = cache.setdefault("filter_cache", {})
    if fk not in fc:
        fc[fk] = [
            i for i, fact in enumerate(cache["facts"])
            if _fact_matches(fact, fk)
        ]
    return fc[fk]


def _page_indices(
    cache: dict,
    filter_kind: str,
    page: int,
    page_size: int,
) -> Tuple[List[int], int]:
    """Return (indices for this page, filtered_total). No full-range list when unfiltered."""
    total = cache["total"]
    fk = (filter_kind or "").strip()
    if not fk:
        start = page * page_size
        if start >= total:
            return [], total
        end = min(start + page_size, total)
        return list(range(start, end)), total

    matches = _matching_indices(cache, fk)
    start = page * page_size
    return matches[start : start + page_size], len(matches)


def load_image_bytes(relative_path: str, tar_path: Path, tar_cache: dict) -> Optional[bytes]:
    ref = str(relative_path).replace("\\", "/")
    tar_key = str(tar_path.resolve())
    try:
        if tar_cache.get("shard_tar") != tar_key:
            old = tar_cache.get("tar")
            if old is not None:
                old.close()
            tar_cache["tar"] = tarfile.open(tar_path, "r")
            tar_cache["shard_tar"] = tar_key
        tf = tar_cache["tar"]
        member = tf.getmember(ref)
        f = tf.extractfile(member)
        if f is None:
            return None
        return f.read()
    except (KeyError, tarfile.TarError, OSError):
        return None


def load_pangu_images(
    sample: dict,
    tar_path: Path,
    tar_cache: dict,
) -> List[Image.Image]:
    images: List[Image.Image] = []
    refs = extract_image_entries(sample)
    for idx, img_meta in enumerate(refs):
        rel = img_meta.get("relative_path") or ""
        raw = load_image_bytes(rel, tar_path, tar_cache)
        if raw is None:
            w = int(img_meta.get("width") or 640)
            h = int(img_meta.get("height") or 360)
            frame = Image.new("RGB", (max(w, 64), max(h, 64)), (245, 245, 245))
            d = ImageDraw.Draw(frame)
            d.multiline_text(
                (16, 16),
                f"Missing image {idx + 1}/{len(refs)}\n{rel}",
                fill=(80, 80, 80),
                spacing=6,
            )
            images.append(frame)
        else:
            images.append(Image.open(io.BytesIO(raw)).convert("RGB"))
    return images


def apply_3d_overlay(images: List[Image.Image], sample: dict) -> Tuple[List[Image.Image], bool]:
    if not images:
        return images, False
    parsed = pangu_to_parsed(sample)
    if not ann_viz._is_3d_grounding_task("", parsed):
        return images, False
    try:
        entries, ctx = ann_viz._resolve_grounding_context({}, parsed)
        if not entries or ctx is None:
            return images, False
        intrinsic, ref_size = ctx
        out = list(images)
        out[0] = ann_viz.draw_3d_boxes_on_image(
            out[0], entries, intrinsic, ref_size=ref_size,
        )
        return out, True
    except Exception as exc:
        print(f"[3d_grounding] overlay failed: {exc}")
        return images, False


def build_display_images(
    sample: dict,
    tar_path: Path,
    *,
    overlay_3d: bool,
    tar_cache: dict,
) -> Tuple[List[Image.Image], bool]:
    images = load_pangu_images(sample, tar_path, tar_cache)
    if overlay_3d and images:
        return apply_3d_overlay(images, sample)
    return images, False


def _close_tar_cache(tar_cache: dict) -> None:
    tf = tar_cache.get("tar")
    if tf is not None:
        tf.close()
    tar_cache.clear()


@dataclass(frozen=True)
class _ImageRenderJob:
    root_path: str
    index: int
    overlay_3d: bool


def _render_images_for_index(job: _ImageRenderJob) -> Tuple[int, List[str]]:
    cache = _get_root_cache(job.root_path)
    sample = read_sample_at(cache, job.index)
    if sample is None:
        return job.index, []

    loc = _locate_sample(cache["shards"], job.index)
    if loc is None:
        return job.index, []

    shard, _ = loc
    tar_cache: dict = {}
    try:
        images, _ = build_display_images(
            sample,
            shard.tar_path,
            overlay_3d=job.overlay_3d,
            tar_cache=tar_cache,
        )
        return job.index, [ann_viz.pil_to_base64(img) for img in images]
    finally:
        _close_tar_cache(tar_cache)


def _render_images_parallel(
    root_path: str,
    indices: List[int],
    *,
    overlay_3d: bool,
) -> Dict[int, List[str]]:
    if not indices:
        return {}

    cache = _get_root_cache(root_path)
    jobs: List[_ImageRenderJob] = []
    for index in indices:
        fact = cache["facts"][index]
        jobs.append(_ImageRenderJob(
            root_path=root_path,
            index=index,
            overlay_3d=overlay_3d and fact.is_3d_grounding,
        ))

    workers = max(1, min(_WORKERS["image"], len(jobs)))
    out: Dict[int, List[str]] = {}
    if workers == 1:
        for job in jobs:
            idx, imgs = _render_images_for_index(job)
            out[idx] = imgs
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_render_images_for_index, job) for job in jobs]
            for fut in as_completed(futures):
                idx, imgs = fut.result()
                out[idx] = imgs
    return out


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Pangu ML Visualizer</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f0fa; color: #333; }
  .header { background: #4a148c; color: white; padding: 16px 24px; display: flex; align-items: center; gap: 16px; flex-wrap: wrap; position: sticky; top: 0; z-index: 100; }
  .header h1 { font-size: 18px; font-weight: 600; }
  .header select { padding: 8px 12px; border-radius: 6px; border: none; min-width: 360px; font-size: 14px; }
  .header .info { margin-left: auto; font-size: 13px; opacity: 0.85; }
  .filters { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .filters label { font-size: 12px; }
  .filters select { padding: 6px 10px; border-radius: 6px; border: none; font-size: 13px; }
  .nav button { padding: 6px 14px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.35); background: transparent; color: white; cursor: pointer; }
  .nav button:disabled { opacity: 0.35; }
  .container { max-width: 1200px; margin: 24px auto; padding: 0 24px; }
  .card { background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 20px; }
  .card-header { padding: 12px 18px; background: #ede7f6; border-bottom: 1px solid #ddd; display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  .tag { padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }
  .tag-id { background: #d1c4e9; color: #311b92; }
  .tag-3d { background: #ffccbc; color: #bf360c; }
  .card-body { padding: 18px; }
  .images-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 14px; min-height: 48px; }
  .images-row img { max-height: 380px; border-radius: 8px; border: 1px solid #eee; cursor: zoom-in; object-fit: contain; }
  .img-placeholder { min-width: 160px; min-height: 120px; padding: 16px; background: #f3f3f3; border-radius: 8px; border: 1px dashed #ccc; color: #888; font-size: 13px; display: flex; align-items: center; justify-content: center; }
  .lightbox { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.85); z-index: 200; justify-content: center; align-items: center; cursor: zoom-out; }
  .lightbox.active { display: flex; }
  .lightbox img { max-width: 95vw; max-height: 95vh; object-fit: contain; border-radius: 8px; }
  .qa-text { padding: 10px 14px; border-radius: 8px; font-size: 14px; line-height: 1.55; white-space: pre-wrap; margin-top: 6px; }
  .qa-text.q { background: #ede7f6; }
  .qa-text.a { background: #e8f5e9; }
  .turn-badge { font-size: 11px; background: #fff3e0; color: #e65100; padding: 2px 8px; border-radius: 8px; margin-left: 6px; }
  .btn-raw { margin-left: auto; padding: 4px 12px; border: 1px solid #ccc; border-radius: 6px; background: white; cursor: pointer; font-size: 12px; }
  .raw-panel { display: none; margin-top: 12px; padding: 12px; background: #263238; color: #eceff1; border-radius: 8px; font-family: monospace; font-size: 12px; max-height: 420px; overflow: auto; white-space: pre-wrap; }
  .raw-panel.open { display: block; }
  .loading-state { text-align: center; padding: 48px; color: #666; font-size: 15px; }
</style>
</head>
<body>
<div class="header">
  <h1>Pangu ML</h1>
  <select id="rootSelect" onchange="loadRoot()">
    <option value="">-- select dataset --</option>
    {% for r in roots %}
    <option value="{{ r.path }}" {% if selected == r.path %}selected{% endif %}>{{ r.label }}</option>
    {% endfor %}
  </select>
  <div class="filters">
    <label>Filter
      <select id="filterKind" onchange="applyFilters()">
        <option value="">All samples</option>
        <option value="3d_grounding">3D grounding only</option>
        <option value="single_image">Single image</option>
        <option value="multi_image">Multi image</option>
      </select>
    </label>
  </div>
  <div class="nav">
    <button onclick="navigate(-1)">Prev</button>
    <span id="pageInfo">-</span>
    <button onclick="navigate(1)">Next</button>
    <input id="pageJump" type="number" min="1" style="width: 84px; margin-left: 6px; padding: 6px 10px; border-radius: 6px; border: none; font-size: 13px;" />
    <button onclick="jumpToPage()">Go</button>
  </div>
  <div class="info" id="headerInfo"></div>
</div>
<div class="container" id="cards"></div>
<div class="lightbox" id="lightbox" onclick="closeLightbox()">
  <img id="lightboxImg" src="" alt="" onclick="event.stopPropagation()" />
</div>
<script>
let currentRoot = '', currentPage = 0, totalRows = 0, filteredTotal = 0;
let fetchSeq = 0, imageSeq = 0;
const pageSize = 8;

function filterParams() {
  const p = new URLSearchParams();
  const fk = document.getElementById('filterKind')?.value || '';
  if (fk) p.set('filter_kind', fk);
  return p;
}

function applyFilters() {
  if (!currentRoot) return;
  currentPage = 0;
  fetchPage();
}

function loadRoot() {
  currentRoot = document.getElementById('rootSelect').value;
  currentPage = 0;
  document.getElementById('filterKind').value = '';
  if (!currentRoot) return;
  fetchPage();
}

function fetchPage() {
  const seq = ++fetchSeq;
  imageSeq = seq;
  document.getElementById('cards').innerHTML = '<div class="loading-state">Loading text…</div>';
  document.getElementById('pageInfo').textContent = 'Loading…';
  const q = filterParams();
  q.set('root', currentRoot);
  q.set('page', String(currentPage));
  q.set('page_size', String(pageSize));
  fetch('/api/data?' + q.toString())
    .then(r => r.json())
    .then(data => {
      if (seq !== fetchSeq) return;
      totalRows = data.total;
      filteredTotal = data.filtered_total ?? data.total;
      renderRows(data.rows);
      updateNav();
      loadPageImages(data.rows, seq);
    })
    .catch(() => {
      if (seq !== fetchSeq) return;
      document.getElementById('cards').innerHTML = '<div class="loading-state">Failed to load.</div>';
    });
}

function updateNav() {
  const start = filteredTotal ? currentPage * pageSize + 1 : 0;
  const end = Math.min((currentPage + 1) * pageSize, filteredTotal);
  const note = filteredTotal !== totalRows ? ` (${filteredTotal} matched / ${totalRows})` : '';
  document.getElementById('pageInfo').textContent = `rows ${start}-${end} / ${filteredTotal}${note}`;
  document.getElementById('headerInfo').textContent = currentRoot;
  const jump = document.getElementById('pageJump');
  if (jump) jump.value = String(currentPage + 1);
}

function navigate(delta) {
  const maxPage = Math.max(0, Math.ceil(filteredTotal / pageSize) - 1);
  currentPage = Math.max(0, Math.min(maxPage, currentPage + delta));
  fetchPage();
}

function jumpToPage() {
  const maxPage = Math.max(0, Math.ceil(filteredTotal / pageSize) - 1);
  const el = document.getElementById('pageJump');
  if (!el) return;
  let p = parseInt(el.value || '1', 10);
  if (!Number.isFinite(p)) p = 1;
  p = Math.max(1, Math.min(maxPage + 1, p));
  currentPage = p - 1;
  fetchPage();
}

function renderRows(rows) {
  document.getElementById('cards').innerHTML = rows.map(r => cardHtml(r)).join('');
}

function cardForRow(rowIndex) {
  return document.querySelector(`.card[data-index="${rowIndex}"]`);
}

function imagePlaceholderHtml(count) {
  if (!count) return '';
  return `<div class="img-placeholder">Loading ${count} image(s)…</div>`;
}

function setCardImages(rowIndex, srcList) {
  const card = cardForRow(rowIndex);
  if (!card) return;
  const row = card.querySelector('.images-row');
  if (!row) return;
  if (!srcList || !srcList.length) {
    row.innerHTML = '';
    return;
  }
  row.innerHTML = srcList.map(src =>
    `<img src="${src}" onclick="openLightbox(this.src)" alt="">`).join('');
}

function loadPageImages(rows, seq) {
  const withImages = (rows || []).filter(r => (r.image_count || 0) > 0);
  withImages.forEach(r => {
    const q = new URLSearchParams({
      root: currentRoot,
      indices: String(r.row_index),
      overlay_3d: r.can_3d_overlay ? '1' : '0',
    });
    fetch('/api/images?' + q.toString())
      .then(res => res.json())
      .then(data => {
        if (seq !== imageSeq) return;
        const key = String(r.row_index);
        const imgs = (data.by_index && (data.by_index[key] || data.by_index[r.row_index])) || [];
        setCardImages(r.row_index, imgs);
      })
      .catch(() => {
        if (seq !== imageSeq) return;
        const card = cardForRow(r.row_index);
        const row = card?.querySelector('.images-row');
        if (row) row.innerHTML = '<div class="img-placeholder">Image load failed</div>';
      });
  });
}

function cardHtml(r) {
  const imgBlock = imagePlaceholderHtml(r.image_count || 0);
  const turns = (r.turns || []).map((t, i) => `
    <div class="qa-block">
      <span class="turn-badge">Turn ${i + 1}${t.task_name ? ' · ' + t.task_name : ''}</span>
      <div class="qa-text q">${escapeHtml(t.question || '')}</div>
      <div class="qa-text a">${escapeHtml(t.answer || '')}</div>
    </div>`).join('');
  const overlay3d = r.can_3d_overlay ? `<label style="font-size:13px;display:block;margin-bottom:10px;">
      <input type="checkbox" class="overlay-3d-cb" checked onchange="refreshOverlay(${r.row_index})"> Show 3D bbox overlay
    </label>` : '';
  const tag3d = r.can_3d_overlay ? '<span class="tag tag-3d">3D grounding</span>' : '';
  return `<div class="card" data-index="${r.row_index}">
    <div class="card-header">
      <span class="tag tag-id">${escapeHtml(r.sample_id || '')}</span>
      ${tag3d}
      <span style="font-size:12px;color:#666;">images: ${r.image_count || 0}</span>
      <button class="btn-raw" onclick="toggleRaw(${r.row_index}, this)">Raw JSON</button>
    </div>
    <div class="card-body">
      ${overlay3d}
      <div class="images-row">${imgBlock}</div>
      ${turns}
      <pre class="raw-panel" id="raw-${r.row_index}"></pre>
    </div>
  </div>`;
}

function refreshOverlay(rowIndex) {
  const card = cardForRow(rowIndex);
  if (!card) return;
  const overlay3d = card.querySelector('.overlay-3d-cb')?.checked ?? false;
  const row = card.querySelector('.images-row');
  if (row) row.innerHTML = '<div class="img-placeholder">Rendering overlay…</div>';
  const q = new URLSearchParams({
    root: currentRoot,
    indices: String(rowIndex),
    overlay_3d: overlay3d ? '1' : '0',
  });
  fetch('/api/images?' + q.toString())
    .then(r => r.json())
    .then(data => {
      const key = String(rowIndex);
      setCardImages(rowIndex, (data.by_index && data.by_index[key]) || []);
    });
}

function toggleRaw(idx, btn) {
  const panel = document.getElementById('raw-' + idx);
  if (panel.classList.contains('open')) {
    panel.classList.remove('open');
    btn.textContent = 'Raw JSON';
    return;
  }
  btn.textContent = 'Hide raw';
  panel.classList.add('open');
  panel.textContent = 'Loading...';
  fetch(`/api/raw_row?root=${encodeURIComponent(currentRoot)}&index=${idx}`)
    .then(r => r.json()).then(d => { panel.textContent = JSON.stringify(d.row, null, 2); });
}

function escapeHtml(t) {
  const d = document.createElement('div');
  d.textContent = t == null ? '' : String(t);
  return d.innerHTML;
}

function openLightbox(src) {
  document.getElementById('lightboxImg').src = src;
  document.getElementById('lightbox').classList.add('active');
}

function closeLightbox() {
  document.getElementById('lightbox').classList.remove('active');
}

document.addEventListener('keydown', e => { if (e.key === 'Escape') closeLightbox(); });
window.onload = () => { if (document.getElementById('rootSelect').value) loadRoot(); };
</script>
</body>
</html>
"""


@app.before_request
def _log_request() -> None:
    qs = request.query_string.decode("utf-8", errors="replace")
    tail = f"?{qs}" if qs else ""
    _log(f"[http] {request.method} {request.path}{tail}")


@app.route("/")
def index():
    roots = discover_pangu_roots(DATA_DIR)
    selected = request.args.get("root", "")
    return render_template_string(HTML_TEMPLATE, roots=roots, selected=selected)


@app.route("/api/data")
def api_data():
    """Text/metadata only — images loaded separately via /api/images."""
    root = request.args.get("root", "")
    page = int(request.args.get("page", 0))
    page_size = int(request.args.get("page_size", 8))
    filter_kind = request.args.get("filter_kind", "")

    cache = _get_root_cache(root)
    total = cache["total"]
    page_indices, filtered_total = _page_indices(
        cache, filter_kind, page, page_size,
    )
    rows_out = _build_page_text_rows(root, cache, page_indices)

    return jsonify({
        "total": total,
        "filtered_total": filtered_total,
        "page": page,
        "rows": rows_out,
    })


@app.route("/api/images")
def api_images():
    """Decode images for one or more row indices (parallel on server)."""
    root = request.args.get("root", "")
    raw = request.args.get("indices", "")
    overlay_3d = request.args.get("overlay_3d", "1") == "1"
    indices: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            indices.append(int(part))
    if not indices:
        return jsonify({"by_index": {}})

    rendered = _render_images_parallel(root, indices, overlay_3d=overlay_3d)
    by_index = {str(k): v for k, v in rendered.items()}
    return jsonify({"by_index": by_index})


@app.route("/api/render")
def api_render():
    """Legacy alias for single-row image refresh."""
    root = request.args.get("root", "")
    index = int(request.args.get("index", 0))
    overlay_3d = request.args.get("overlay_3d", "0") == "1"
    rendered = _render_images_parallel(root, [index], overlay_3d=overlay_3d)
    return jsonify({"images": rendered.get(index, [])})


@app.route("/api/raw_row")
def api_raw_row():
    root = request.args.get("root", "")
    index = int(request.args.get("index", 0))
    sample = read_sample(root, index)
    return jsonify({"row": sample or {}})


def _lan_urls(port: int) -> List[str]:
    seen: set = set()
    urls: List[str] = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            if ip and not ip.startswith("127."):
                seen.add(ip)
    except OSError:
        pass
    for ip in sorted(seen):
        urls.append(f"http://{ip}:{port}")
    return urls


def _print_listen_info(host: str, port: int) -> None:
    _log(f"\nListening on http://{host}:{port}")
    _log(f"  This machine: http://127.0.0.1:{port}")
    if host in ("0.0.0.0", "::"):
        for url in _lan_urls(port):
            _log(f"  LAN: {url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pangu ML bundle visualizer")
    parser.add_argument("--data_dir", type=str, default="output/pangu_ml")
    parser.add_argument("--port", type=int, default=8891)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--index-workers",
        type=int,
        default=None,
        help="Parallel workers for shard index build (default: CPU count).",
    )
    parser.add_argument(
        "--image-workers",
        type=int,
        default=None,
        help="Parallel workers for image decode per request (default: min(48, CPU)).",
    )
    parser.add_argument(
        "--read-workers",
        type=int,
        default=None,
        help="Parallel workers for JSONL text reads per page (default: min(32, CPU)).",
    )
    parser.add_argument(
        "--warm-index",
        action="store_true",
        help="Build jsonl index at startup (shows per-shard progress; blocks until done).",
    )
    args = parser.parse_args()

    _cpu = os.cpu_count() or 32
    _WORKERS["index"] = args.index_workers if args.index_workers is not None else _cpu
    _WORKERS["image"] = args.image_workers if args.image_workers is not None else min(_cpu, 48)
    _WORKERS["read"] = args.read_workers if args.read_workers is not None else min(_cpu, 32)

    DATA_DIR = args.data_dir
    _log("Pangu ML visualizer starting ...")
    roots = discover_pangu_roots(DATA_DIR)
    _log(f"Workers: index={_WORKERS['index']}, read={_WORKERS['read']}, "
         f"image={_WORKERS['image']}")
    if args.warm_index:
        for r in roots:
            _log(f"Warm-index: {r['path']}")
            _get_root_cache(r["path"])
    _print_listen_info(args.host, args.port)
    _log("Server ready — open the URL above in a browser.")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
