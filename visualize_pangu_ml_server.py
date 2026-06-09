"""
Flask visualizer for Pangu ML training bundles.

Layout (per dataset root):
  {root}/jsonl/data_{shard:06d}.jsonl
  {root}/images/data_{shard:06d}.tar

3D grounding samples: wireframe overlay via visualize_server.draw_3d_boxes_on_image
(camera f_x/f_y or hfov/vfov parsed from user text; boxes from assistant JSON).

Usage:
    python visualize_pangu_ml_server.py --data_dir /path/to/pangu_output --port 8891
"""

from __future__ import annotations

import argparse
import io
import json
import os
import socket
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, render_template_string, request
from PIL import Image, ImageDraw

import visualize_server as ann_viz

app = Flask(__name__)
DATA_DIR = "output/pangu_ml"

_SHARD_CACHE: Dict[str, dict] = {}
_TAR_CACHE: Dict[str, dict] = {}


@dataclass(frozen=True)
class ShardInfo:
    shard_id: str
    jsonl_path: Path
    tar_path: Path
    line_count: int
    start_index: int


def _shard_id_from_stem(stem: str) -> Optional[str]:
    if not stem.startswith("data_"):
        return None
    return stem.split("_", 1)[1]


def discover_pangu_roots(data_dir: str) -> List[dict]:
    """Find dataset roots that contain paired jsonl/tar shards."""
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
        shards = _build_shard_index(root)
        if not shards:
            continue
        seen.add(key)
        total = sum(s.line_count for s in shards)
        roots.append({
            "path": str(root.resolve()),
            "label": f"{root.name} ({total} samples, {len(shards)} shards)",
        })
    return roots


def _build_shard_index(root: Path) -> List[ShardInfo]:
    jsonl_dir = root / "jsonl"
    images_dir = root / "images"
    if not jsonl_dir.is_dir() or not images_dir.is_dir():
        return []

    tar_map: Dict[str, Path] = {}
    for tar_path in sorted(images_dir.glob("data_*.tar")):
        sid = _shard_id_from_stem(tar_path.stem)
        if sid is not None:
            tar_map[sid] = tar_path

    shards: List[ShardInfo] = []
    running = 0
    for jf in sorted(jsonl_dir.glob("data_*.jsonl")):
        sid = _shard_id_from_stem(jf.stem)
        if sid is None:
            continue
        tar_path = tar_map.get(sid)
        if tar_path is None:
            continue
        n = 0
        with jf.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n += 1
        shards.append(ShardInfo(sid, jf, tar_path, n, running))
        running += n
    return shards


def _get_root_cache(root_path: str) -> dict:
    mtime = _root_mtime(root_path)
    cached = _SHARD_CACHE.get(root_path)
    if cached and cached.get("mtime") == mtime:
        return cached
    root = Path(root_path)
    shards = _build_shard_index(root)
    cache = {
        "mtime": mtime,
        "shards": shards,
        "total": sum(s.line_count for s in shards),
    }
    _SHARD_CACHE[root_path] = cache
    return cache


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


def _locate_sample(shards: List[ShardInfo], index: int) -> Optional[Tuple[ShardInfo, int]]:
    if index < 0:
        return None
    for shard in shards:
        if index < shard.start_index + shard.line_count:
            return shard, index - shard.start_index
    return None


def read_sample(root_path: str, index: int) -> Optional[dict]:
    cache = _get_root_cache(root_path)
    loc = _locate_sample(cache["shards"], index)
    if loc is None:
        return None
    shard, line_idx = loc
    with shard.jsonl_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            if i == line_idx:
                return json.loads(line)
    return None


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


def _tar_cache_for(tar_path: str) -> dict:
    key = str(Path(tar_path).resolve())
    cached = _TAR_CACHE.get(key)
    if cached and cached.get("path") == key:
        return cached
    entry = {"path": key, "tar": None, "shard_tar": key}
    _TAR_CACHE[key] = entry
    return entry


def load_image_bytes(relative_path: str, tar_path: Path, tar_cache: dict) -> Optional[bytes]:
    ref = str(relative_path).replace("\\", "/")
    try:
        if tar_cache.get("shard_tar") != str(tar_path):
            if tar_cache.get("tar") is not None:
                tar_cache["tar"].close()
            tar_cache["tar"] = tarfile.open(tar_path, "r")
            tar_cache["shard_tar"] = str(tar_path)
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


def _sample_matches_filter(sample: dict, filter_kind: str) -> bool:
    fk = (filter_kind or "").strip().lower()
    if not fk:
        return True
    n_img = len(extract_image_entries(sample))
    is_3d = is_3d_grounding_sample(sample)
    if fk == "3d_grounding":
        return is_3d
    if fk == "single_image":
        return n_img == 1
    if fk == "multi_image":
        return n_img > 1
    return True


def _filtered_indices(root_path: str, filter_kind: str) -> List[int]:
    cache = _get_root_cache(root_path)
    fk = (filter_kind or "").strip()
    if not fk:
        return list(range(cache["total"]))
    out: List[int] = []
    for i in range(cache["total"]):
        sample = read_sample(root_path, i)
        if sample and _sample_matches_filter(sample, fk):
            out.append(i)
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
  .images-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 14px; }
  .images-row img { max-height: 380px; border-radius: 8px; border: 1px solid #eee; cursor: zoom-in; object-fit: contain; }
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
let fetchSeq = 0;
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
  document.getElementById('cards').innerHTML = '<div class="loading-state">Loading…</div>';
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

function cardHtml(r) {
  const imgs = (r.display_images || []).map(src =>
    `<img src="${src}" onclick="openLightbox(this.src)" alt="">`).join('');
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
      <div class="images-row">${imgs}</div>
      ${turns}
      <pre class="raw-panel" id="raw-${r.row_index}"></pre>
    </div>
  </div>`;
}

function refreshOverlay(rowIndex) {
  const card = document.querySelector(`.card[data-index="${rowIndex}"]`);
  if (!card) return;
  const overlay3d = card.querySelector('.overlay-3d-cb')?.checked ?? false;
  const q = new URLSearchParams({
    root: currentRoot, index: String(rowIndex), overlay_3d: overlay3d ? '1' : '0',
  });
  fetch('/api/render?' + q.toString()).then(r => r.json()).then(data => {
    const imgs = card.querySelectorAll('.images-row img');
    (data.images || []).forEach((src, i) => { if (imgs[i]) imgs[i].src = src; });
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


@app.route("/")
def index():
    roots = discover_pangu_roots(DATA_DIR)
    selected = request.args.get("root", "")
    return render_template_string(HTML_TEMPLATE, roots=roots, selected=selected)


@app.route("/api/data")
def api_data():
    root = request.args.get("root", "")
    page = int(request.args.get("page", 0))
    page_size = int(request.args.get("page_size", 8))
    filter_kind = request.args.get("filter_kind", "")

    cache = _get_root_cache(root)
    total = cache["total"]
    indices = _filtered_indices(root, filter_kind)
    page_indices = indices[page * page_size : (page + 1) * page_size]

    rows_out: List[dict] = []
    tar_cache: dict = {}

    for i in page_indices:
        sample = read_sample(root, i)
        if sample is None:
            continue
        loc = _locate_sample(cache["shards"], i)
        if loc is None:
            continue
        shard, _ = loc
        can_3d = is_3d_grounding_sample(sample)
        images, _ = build_display_images(
            sample, shard.tar_path, overlay_3d=can_3d, tar_cache=tar_cache,
        )
        rows_out.append({
            "row_index": i,
            "sample_id": sample.get("id") or f"row_{i}",
            "turns": pangu_to_display_turns(sample),
            "display_images": [ann_viz.pil_to_base64(img) for img in images],
            "can_3d_overlay": can_3d,
            "image_count": len(extract_image_entries(sample)),
        })

    if tar_cache.get("tar") is not None:
        tar_cache["tar"].close()

    return jsonify({
        "total": total,
        "filtered_total": len(indices),
        "page": page,
        "rows": rows_out,
    })


@app.route("/api/render")
def api_render():
    root = request.args.get("root", "")
    index = int(request.args.get("index", 0))
    overlay_3d = request.args.get("overlay_3d", "0") == "1"

    sample = read_sample(root, index)
    if sample is None:
        return jsonify({"images": []})

    cache = _get_root_cache(root)
    loc = _locate_sample(cache["shards"], index)
    if loc is None:
        return jsonify({"images": []})
    shard, _ = loc

    tar_cache = _tar_cache_for(str(shard.tar_path))
    images, _ = build_display_images(
        sample, shard.tar_path, overlay_3d=overlay_3d, tar_cache=tar_cache,
    )
    return jsonify({"images": [ann_viz.pil_to_base64(img) for img in images]})


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
    print(f"\nListening on http://{host}:{port}")
    print(f"  This machine: http://127.0.0.1:{port}")
    if host in ("0.0.0.0", "::"):
        for url in _lan_urls(port):
            print(f"  LAN: {url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pangu ML bundle visualizer")
    parser.add_argument("--data_dir", type=str, default="output/pangu_ml")
    parser.add_argument("--port", type=int, default=8891)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    roots = discover_pangu_roots(DATA_DIR)
    print(f"Found {len(roots)} Pangu ML dataset(s) in {DATA_DIR}:")
    for r in roots:
        print(f"  {r['label']}")
    _print_listen_info(args.host, args.port)
    app.run(host=args.host, port=args.port, debug=False)
