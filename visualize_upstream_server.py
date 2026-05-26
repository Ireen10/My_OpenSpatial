"""
Visualizer for aggregate (merged_samples parquet) and export (upstream JSONL + tar).

Same UX as visualize_server.py: QA turns, optional mark overlay, Raw row button.

Usage:
    python visualize_upstream_server.py --data_dir output/frame_rot --port 8890
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from flask import Flask, jsonify, render_template_string, request
from PIL import Image
from PIL import ImageDraw

from dataset.upstream_export import (
    MANIFEST_FILENAME,
    SAMPLES_FILENAME,
    normalize_messages,
    read_upstream_jsonl,
)
import visualize_server as ann_viz

app = Flask(__name__)
DATA_DIR = "output/frame_rot"


def discover_upstream_sources(data_dir: str) -> List[dict]:
    """Find aggregate merged parquets and export bundles under data_dir."""
    sources: List[dict] = []
    data_dir = os.path.abspath(data_dir)

    for pq in sorted(glob.glob(os.path.join(data_dir, "**/merged_samples/data.parquet"), recursive=True)):
        rel = os.path.relpath(pq, data_dir)
        pipeline = rel.split(os.sep)[0] if rel else ""
        kind = "aggregate"
        if "multiview" in pipeline.lower():
            label = f"[Aggregate · Multi] {pipeline}"
        elif "singleview" in pipeline.lower():
            label = f"[Aggregate · Single] {pipeline}"
        else:
            label = f"[Aggregate] {rel}"
        sources.append({
            "label": label,
            "path": pq,
            "kind": kind,
            "bundle_root": None,
            "multiview": "multiview" in rel.lower(),
        })

    seen_manifests = set()
    for manifest in sorted(glob.glob(os.path.join(data_dir, "**/export/manifest.json"), recursive=True)):
        bundle_root = str(Path(manifest).parent)
        if bundle_root in seen_manifests:
            continue
        jsonl = os.path.join(bundle_root, SAMPLES_FILENAME)
        if not os.path.isfile(jsonl):
            continue
        seen_manifests.add(bundle_root)
        rel = os.path.relpath(bundle_root, data_dir)
        pipeline = rel.split(os.sep)[0] if rel else rel
        if "multiview" in rel.lower():
            label = f"[Export · Multi] {rel}"
        elif "singleview" in rel.lower():
            label = f"[Export · Single] {rel}"
        else:
            label = f"[Export] {rel}"
        sources.append({
            "label": label,
            "path": jsonl,
            "kind": "export",
            "bundle_root": bundle_root,
            "multiview": "multiview" in rel.lower(),
        })

    return sources


def _get_source(path: str, kind: str, bundle_root: Optional[str]) -> dict:
    for s in discover_upstream_sources(DATA_DIR):
        if s["path"] == path and s["kind"] == kind:
            return s
    return {"path": path, "kind": kind, "bundle_root": bundle_root, "multiview": False}


def _load_aggregate_df(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def _load_export_records(path: str) -> List[dict]:
    return read_upstream_jsonl(path)


def _json_load_col(val: Any) -> Any:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, str):
        return json.loads(val)
    return val


def aggregate_row_to_parse_dict(row) -> dict:
    """Shape compatible with visualize_server.parse_row / build_display_images."""
    meta = _json_load_col(row.get("metadata_json")) or {}
    msgs = normalize_messages(_json_load_col(row.get("messages_json")) or row.get("messages"))
    refs = row.get("image_refs")
    if hasattr(refs, "tolist"):
        refs = refs.tolist()
    refs = [str(r) for r in (refs or []) if r]
    prov = meta.get("provenance") or {}
    source_tasks = prov.get("source_tasks") or []
    return {
        "metadata": meta,
        "messages": msgs,
        "image": refs,
        "question_tags": source_tasks,
        "question_types": "",
        "sample_id": row.get("sample_id"),
        "merge_group_key": row.get("merge_group_key"),
        "schema_version": row.get("schema_version"),
    }


def export_record_to_parse_dict(rec: dict) -> dict:
    meta = rec.get("metadata") or {}
    msgs = normalize_messages(rec.get("messages"))
    refs = [str(r) for r in (rec.get("image_refs") or []) if r]
    prov = meta.get("provenance") or {}
    return {
        "metadata": meta,
        "messages": msgs,
        "image": refs,
        "question_tags": prov.get("source_tasks") or [],
        "question_types": "",
        "sample_id": rec.get("sample_id"),
        "merge_group_key": rec.get("merge_group_key"),
        "schema_version": rec.get("schema_version"),
    }


def _tar_index_by_member(bundle_root: str) -> Dict[str, dict]:
    manifest_path = os.path.join(bundle_root, MANIFEST_FILENAME)
    if not os.path.isfile(manifest_path):
        return {}
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return {e["tar_path"]: e for e in manifest.get("tar_index", [])}


def resolve_export_image(ref: str, bundle_root: str, tar_cache: dict) -> Optional[Image.Image]:
    ref = str(ref).replace("\\", "/")
    local = os.path.join(bundle_root, ref)
    if os.path.isfile(local):
        return Image.open(local)

    index = tar_cache.get("index")
    if index is None:
        tar_cache["index"] = _tar_index_by_member(bundle_root)
        index = tar_cache["index"]

    entry = index.get(ref)
    if entry and entry.get("source_path") and os.path.isfile(entry["source_path"]):
        return Image.open(entry["source_path"])

    tar_path = os.path.join(bundle_root, "images.tar")
    if not os.path.isfile(tar_path):
        return None
    if "tar" not in tar_cache:
        tar_cache["tar"] = tarfile.open(tar_path, "r")
    tar = tar_cache["tar"]
    try:
        member = tar.getmember(ref)
        data = tar.extractfile(member)
        if data:
            return Image.open(io.BytesIO(data.read()))
    except KeyError:
        pass
    return None


def load_upstream_images(
    parse_dict: dict,
    *,
    kind: str,
    bundle_root: Optional[str],
    tar_cache: dict,
) -> List[Image.Image]:
    refs = parse_dict.get("image") or []
    if not isinstance(refs, list):
        refs = [refs]
    images: List[Image.Image] = []
    for idx, ref in enumerate(refs):
        img = None
        if kind == "export" and bundle_root:
            img = resolve_export_image(ref, bundle_root, tar_cache)
        elif isinstance(ref, str) and os.path.isfile(ref):
            img = Image.open(ref)
        if img is None:
            # Keep image count aligned with refs to match <image> placeholders.
            # This also makes missing refs obvious in the UI.
            frame = Image.new("RGB", (640, 360), (245, 245, 245))
            d = ImageDraw.Draw(frame)
            msg = f"Missing image {idx + 1}/{len(refs)}\\n{ref}"
            d.multiline_text((16, 16), msg, fill=(80, 80, 80), spacing=6)
            images.append(frame)
        else:
            images.append(img.convert("RGB"))
    return images


def build_upstream_display_images(
    parse_dict: dict,
    parsed: dict,
    *,
    kind: str,
    bundle_root: Optional[str],
    slot_ids=None,
    marks_mode: str = "off",
    tar_cache: Optional[dict] = None,
) -> Tuple[List[Image.Image], bool]:
    tar_cache = tar_cache if tar_cache is not None else {}
    images = load_upstream_images(parse_dict, kind=kind, bundle_root=bundle_root, tar_cache=tar_cache)
    if not images:
        return [], False

    apply_marks = marks_mode in ("selected", "all")
    meta = parsed.get("meta") or {}
    mark_spec = meta.get("mark_spec")
    n_frames = len(images)
    out: List[Image.Image] = []

    for fi, img in enumerate(images):
        frame = img.copy()
        if apply_marks and mark_spec:
            if marks_mode == "all":
                user_slots = None
            else:
                user_slots = list(slot_ids or [])
            use_ids = ann_viz.slot_ids_for_frame(
                mark_spec, fi, n_frames=n_frames, user_slot_ids=user_slots,
            )
            if use_ids:
                frame = ann_viz._apply_marks_to_image(
                    frame, mark_spec, use_ids, None, view_index=fi,
                )
        out.append(frame)
    return out, apply_marks


def serialize_upstream_raw(parse_dict: dict, parsed: dict, *, kind: str, bundle_root: Optional[str]) -> dict:
    row = ann_viz._numpy_to_python(parse_dict)
    row["metadata"] = ann_viz._numpy_to_python(parsed.get("meta"))
    row["messages"] = ann_viz._numpy_to_python(parse_dict.get("messages"))
    if kind == "export" and bundle_root:
        row["_bundle_root"] = bundle_root
        row["_manifest"] = MANIFEST_FILENAME
    prov = (parsed.get("meta") or {}).get("provenance")
    if prov:
        row["_provenance"] = ann_viz._numpy_to_python(prov)
    return row


def _row_count(source: dict) -> int:
    if source["kind"] == "export":
        return len(_load_export_records(source["path"]))
    return len(_load_aggregate_df(source["path"]))


def _load_parsed(path: str, kind: str, bundle_root: Optional[str], index: int):
    if kind == "export":
        recs = _load_export_records(path)
        if index < 0 or index >= len(recs):
            return None, None, None
        pd_dict = export_record_to_parse_dict(recs[index])
        series = pd.Series(pd_dict)
        parsed = ann_viz.parse_row(series)
        parsed["meta"] = ann_viz._normalize_meta(pd_dict.get("metadata"))
        return series, pd_dict, parsed
    df = _load_aggregate_df(path)
    if index < 0 or index >= len(df):
        return None, None, None
    row = df.iloc[index]
    pd_dict = aggregate_row_to_parse_dict(row)
    series = pd.Series(pd_dict)
    parsed = ann_viz.parse_row(series)
    parsed["meta"] = ann_viz._normalize_meta(pd_dict.get("metadata"))
    return series, pd_dict, parsed


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>OpenSpatial Upstream Visualizer</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f0f4f8; color: #333; }
  .header { background: #0d47a1; color: white; padding: 16px 24px; display: flex; align-items: center; gap: 16px; flex-wrap: wrap; position: sticky; top: 0; z-index: 100; }
  .header h1 { font-size: 18px; font-weight: 600; }
  .header select { padding: 8px 12px; border-radius: 6px; border: none; min-width: 360px; font-size: 14px; }
  .header .info { margin-left: auto; font-size: 13px; opacity: 0.85; }
  .nav button { padding: 6px 14px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.35); background: transparent; color: white; cursor: pointer; }
  .nav button:disabled { opacity: 0.35; }
  .container { max-width: 1200px; margin: 24px auto; padding: 0 24px; }
  .card { background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 20px; }
  .card-header { padding: 12px 18px; background: #e3f2fd; border-bottom: 1px solid #ddd; display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  .tag { padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }
  .tag-kind { background: #bbdefb; color: #0d47a1; }
  .tag-task { background: #c8e6c9; color: #1b5e20; }
  .card-body { padding: 18px; }
  .images-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 14px; }
  .images-row img { max-height: 380px; border-radius: 8px; border: 1px solid #eee; cursor: pointer; }
  .mark-panel { margin-bottom: 12px; padding: 10px; background: #fafafa; border-radius: 8px; font-size: 13px; }
  .mark-panel label { margin-right: 12px; cursor: pointer; }
  .qa-text { padding: 10px 14px; border-radius: 8px; font-size: 14px; line-height: 1.55; white-space: pre-wrap; margin-top: 6px; }
  .qa-text.q { background: #e3f2fd; }
  .qa-text.a { background: #e8f5e9; }
  .turn-badge { font-size: 11px; background: #fff3e0; color: #e65100; padding: 2px 8px; border-radius: 8px; margin-left: 6px; }
  .btn-raw { margin-left: auto; padding: 4px 12px; border: 1px solid #ccc; border-radius: 6px; background: white; cursor: pointer; font-size: 12px; }
  .raw-panel { display: none; margin-top: 12px; padding: 12px; background: #263238; color: #eceff1; border-radius: 8px; font-family: monospace; font-size: 12px; max-height: 420px; overflow: auto; white-space: pre-wrap; }
  .raw-panel.open { display: block; }
  .meta-line { font-size: 12px; color: #666; margin-bottom: 8px; }
</style>
</head>
<body>
<div class="header">
  <h1>Upstream (Aggregate / Export)</h1>
  <select id="sourceSelect" onchange="loadSource()">
    <option value="">-- select source --</option>
    {% for s in sources %}
    <option value="{{ s.path }}|{{ s.kind }}|{{ s.bundle_root or '' }}"
      {% if selected == s.path %}selected{% endif %}>{{ s.label }}</option>
    {% endfor %}
  </select>
  <div class="nav">
    <button onclick="navigate(-1)">Prev</button>
    <span id="pageInfo">-</span>
    <button onclick="navigate(1)">Next</button>
    <span style="margin-left: 10px; opacity: 0.9;">Page</span>
    <input id="pageJump" type="number" min="1" style="width: 84px; margin-left: 6px; padding: 6px 10px; border-radius: 6px; border: none; font-size: 13px;" />
    <button onclick="jumpToPage()">Go</button>
  </div>
  <div class="info" id="headerInfo"></div>
</div>
<div class="container" id="cards"></div>
<script>
let currentPath = '', currentKind = '', bundleRoot = '', currentPage = 0, totalRows = 0;
const pageSize = 8;

function parseSelectVal() {
  const v = document.getElementById('sourceSelect').value;
  if (!v) return;
  const parts = v.split('|');
  currentPath = parts[0];
  currentKind = parts[1];
  bundleRoot = parts[2] || '';
}

function loadSource() {
  parseSelectVal();
  currentPage = 0;
  if (!currentPath) return;
  fetch(`/api/data?path=${encodeURIComponent(currentPath)}&kind=${currentKind}&bundle_root=${encodeURIComponent(bundleRoot)}&page=0&page_size=${pageSize}`)
    .then(r => r.json())
    .then(data => {
      totalRows = data.total;
      renderRows(data.rows);
      updateNav();
    });
}

function updateNav() {
  const pages = Math.max(1, Math.ceil(totalRows / pageSize));
  document.getElementById('pageInfo').textContent =
    `rows ${currentPage * pageSize + 1}-${Math.min((currentPage+1)*pageSize, totalRows)} / ${totalRows}`;
  document.getElementById('headerInfo').textContent = currentKind + (bundleRoot ? ' · ' + bundleRoot : '');
  const jump = document.getElementById('pageJump');
  if (jump) jump.value = String(currentPage + 1);
}

function navigate(delta) {
  const maxPage = Math.ceil(totalRows / pageSize) - 1;
  currentPage = Math.max(0, Math.min(maxPage, currentPage + delta));
  fetch(`/api/data?path=${encodeURIComponent(currentPath)}&kind=${currentKind}&bundle_root=${encodeURIComponent(bundleRoot)}&page=${currentPage}&page_size=${pageSize}`)
    .then(r => r.json())
    .then(data => { renderRows(data.rows); updateNav(); });
}

function jumpToPage() {
  const maxPage = Math.ceil(totalRows / pageSize);
  const el = document.getElementById('pageJump');
  if (!el) return;
  let p = parseInt(el.value || '1', 10);
  if (!Number.isFinite(p)) p = 1;
  p = Math.max(1, Math.min(maxPage, p));
  currentPage = p - 1;
  fetch(`/api/data?path=${encodeURIComponent(currentPath)}&kind=${currentKind}&bundle_root=${encodeURIComponent(bundleRoot)}&page=${currentPage}&page_size=${pageSize}`)
    .then(r => r.json())
    .then(data => { renderRows(data.rows); updateNav(); });
}

function renderRows(rows) {
  const el = document.getElementById('cards');
  el.innerHTML = rows.map(r => cardHtml(r)).join('');
}

function cardHtml(r) {
  const tasks = (r.source_tasks || []).join(', ');
  const imgInfo = (r.image_refs_count != null && r.images_loaded_count != null)
    ? `images: ${r.images_loaded_count}/${r.image_refs_count}`
    : '';
  const imgs = (r.display_images || []).map(src =>
    `<img src="${src}" onclick="window.open(this.src)">`).join('');
  const turns = (r.turns || []).map((t, i) => `
    <div class="qa-block">
      <span class="turn-badge">Turn ${i + 1}${t.task_name ? ' · ' + t.task_name : ''}${t.sub_task ? ' · ' + t.sub_task : ''}</span>
      <div class="qa-text q">${escapeHtml(t.question || '')}</div>
      <div class="qa-text a">${escapeHtml(t.answer || '')}</div>
    </div>`).join('');
  const marks = (r.mark_slots || []).map(s =>
    `<label><input type="checkbox" class="mark-cb" value="${s.slot_key}" onchange="refreshMarks(${r.row_index})"> ${s.slot_key} (${s.mark_kind || ''})</label>`).join('');
  return `<div class="card" data-index="${r.row_index}">
    <div class="card-header">
      <span class="tag tag-kind">${r.kind}</span>
      <span class="tag tag-task">${escapeHtml(tasks)}</span>
      <span class="meta-line">sample_id: ${escapeHtml(r.sample_id || '')}${imgInfo ? ' · ' + escapeHtml(imgInfo) : ''}</span>
      <button class="btn-raw" onclick="toggleRaw(${r.row_index}, this)">Raw record</button>
    </div>
    <div class="card-body">
      ${marks ? `<div class="mark-panel">${marks}</div>` : ''}
      <div class="images-row">${imgs}</div>
      ${turns}
      <pre class="raw-panel" id="raw-${r.row_index}"></pre>
    </div>
  </div>`;
}

function refreshMarks(rowIndex) {
  const card = document.querySelector(`.card[data-index="${rowIndex}"]`);
  const slots = [...card.querySelectorAll('.mark-cb:checked')].map(c => c.value);
  const params = new URLSearchParams({
    path: currentPath, kind: currentKind, bundle_root: bundleRoot,
    index: String(rowIndex), slots: slots.join(','),
    marks_mode: slots.length ? 'selected' : 'off',
  });
  fetch('/api/render?' + params).then(r => r.json()).then(data => {
    const imgs = card.querySelectorAll('.images-row img');
    (data.images || []).forEach((src, i) => { if (imgs[i]) imgs[i].src = src; });
  });
}

function toggleRaw(idx, btn) {
  const panel = document.getElementById('raw-' + idx);
  if (panel.classList.contains('open')) {
    panel.classList.remove('open');
    btn.textContent = 'Raw record';
    return;
  }
  btn.textContent = 'Hide raw';
  panel.classList.add('open');
  panel.textContent = 'Loading...';
  fetch(`/api/raw_row?path=${encodeURIComponent(currentPath)}&kind=${currentKind}&bundle_root=${encodeURIComponent(bundleRoot)}&index=${idx}`)
    .then(r => r.json()).then(d => { panel.textContent = JSON.stringify(d.row, null, 2); });
}

function escapeHtml(t) {
  const d = document.createElement('div');
  d.textContent = t == null ? '' : String(t);
  return d.innerHTML;
}
window.onload = () => { if (document.getElementById('sourceSelect').value) loadSource(); };
</script>
</body>
</html>
"""


@app.route("/")
def index():
    sources = discover_upstream_sources(DATA_DIR)
    selected = request.args.get("source", "")
    return render_template_string(HTML_TEMPLATE, sources=sources, selected=selected)


@app.route("/api/data")
def api_data():
    path = request.args.get("path", "")
    kind = request.args.get("kind", "aggregate")
    bundle_root = request.args.get("bundle_root") or None
    page = int(request.args.get("page", 0))
    page_size = int(request.args.get("page_size", 8))

    source = _get_source(path, kind, bundle_root)
    total = _row_count(source)
    start = page * page_size
    end = min(start + page_size, total)
    tar_cache: dict = {}

    rows_out = []
    for i in range(start, end):
        _, pd_dict, parsed = _load_parsed(path, kind, bundle_root, i)
        if pd_dict is None:
            continue
        images, marks_applied = build_upstream_display_images(
            pd_dict, parsed, kind=kind, bundle_root=bundle_root, marks_mode="off", tar_cache=tar_cache,
        )
        refs = pd_dict.get("image") or []
        if not isinstance(refs, list):
            refs = [refs]
        prov = (parsed.get("meta") or {}).get("provenance") or {}
        rows_out.append({
            "row_index": i,
            "kind": kind,
            "sample_id": pd_dict.get("sample_id"),
            "merge_group_key": pd_dict.get("merge_group_key"),
            "source_tasks": prov.get("source_tasks") or pd_dict.get("question_tags") or [],
            "turns": parsed["turns"],
            "mark_slots": parsed.get("mark_slots") or [],
            "display_images": [ann_viz.pil_to_base64(img) for img in images],
            "marks_overlay_applied": marks_applied,
            "image_refs_count": len(refs),
            "images_loaded_count": len(images),
        })

    if "tar" in tar_cache:
        tar_cache["tar"].close()

    return jsonify({"total": total, "page": page, "rows": rows_out})


@app.route("/api/render")
def api_render():
    path = request.args.get("path", "")
    kind = request.args.get("kind", "aggregate")
    bundle_root = request.args.get("bundle_root") or None
    index = int(request.args.get("index", 0))
    slots_raw = request.args.get("slots", "")
    marks_mode = request.args.get("marks_mode", "off")
    slot_ids = [s.strip() for s in slots_raw.split(",") if s.strip()]
    if marks_mode == "selected" and not slot_ids:
        marks_mode = "off"

    _, pd_dict, parsed = _load_parsed(path, kind, bundle_root, index)
    if pd_dict is None:
        return jsonify({"images": []})

    images, marks_applied = build_upstream_display_images(
        pd_dict, parsed, kind=kind, bundle_root=bundle_root,
        slot_ids=slot_ids, marks_mode=marks_mode,
    )
    return jsonify({
        "images": [ann_viz.pil_to_base64(img) for img in images],
        "marks_overlay_applied": marks_applied,
    })


@app.route("/api/raw_row")
def api_raw_row():
    path = request.args.get("path", "")
    kind = request.args.get("kind", "aggregate")
    bundle_root = request.args.get("bundle_root") or None
    index = int(request.args.get("index", 0))

    _, pd_dict, parsed = _load_parsed(path, kind, bundle_root, index)
    if pd_dict is None:
        return jsonify({"row": {}})
    return jsonify({"row": serialize_upstream_raw(pd_dict, parsed, kind=kind, bundle_root=bundle_root)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenSpatial aggregate/export visualizer")
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--data_dir", type=str, default="output/frame_rot")
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    sources = discover_upstream_sources(DATA_DIR)
    print(f"Found {len(sources)} upstream sources in {DATA_DIR}:")
    for s in sources:
        print(f"  {s['label']} -> {s['path']}")
    print(f"\nStarting at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
