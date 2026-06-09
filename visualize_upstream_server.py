"""
Visualizer for upstream training artifacts.

Supported sources only:
  1. Sharded export: ``{root}/jsonl/metadata_*.jsonl`` + ``{root}/images/metadata_*.tar``
  2. Parquet tables: ``data.parquet`` with ``metadata_json`` / ``messages_json`` columns

Usage:
    python visualize_upstream_server.py --data_dir output/frame_rot --port 8890
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import socket
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import pandas as pd
from flask import Flask, jsonify, render_template_string, request
from PIL import Image
from PIL import ImageDraw

from dataset.upstream_export import (
    DATASET_METADATA_FILENAME,
    JSONL_SUBDIR,
    is_sharded_upstream_root,
    normalize_messages,
    read_sharded_upstream,
    resolve_shard_image,
)
import visualize_server as ann_viz

app = Flask(__name__)
DATA_DIR = "output/frame_rot"

# (path, kind, bundle_root) -> cached rows + precomputed filter facts
_SOURCE_CACHE: Dict[Tuple[str, str, str], dict] = {}

FilterFacts = Tuple[int, FrozenSet[str]]


def _coarse_task_key(name: str) -> str:
    """Normalize task_name / source_tasks entry without changing task category."""
    s = (name or "").strip().lower()
    if not s:
        return ""
    return s


def _families_from_metadata(meta: dict) -> FrozenSet[str]:
    families: set = set()
    prov = meta.get("provenance") or {}
    for st in prov.get("source_tasks") or []:
        key = _coarse_task_key(str(st))
        if key:
            families.add(key)
    if not families:
        for turn in meta.get("turns") or []:
            if not isinstance(turn, dict):
                continue
            key = _coarse_task_key(turn.get("task_name") or "")
            if key:
                families.add(key)
    return frozenset(families)


def _filter_facts_from_metadata(meta: dict) -> FilterFacts:
    meta = meta if isinstance(meta, dict) else {}
    return len(meta.get("turns") or []), _families_from_metadata(meta)


def _row_matches_upstream_filters(
    n_turns: int,
    families: FrozenSet[str],
    *,
    filter_task: str = "",
    filter_turns: str = "",
) -> bool:
    ft = (filter_turns or "").strip()
    if ft == "1" and n_turns != 1:
        return False
    if ft == "2" and n_turns != 2:
        return False
    if ft == "3" and n_turns != 3:
        return False
    if ft == "2+" and n_turns < 2:
        return False
    if ft == "3+" and n_turns < 3:
        return False
    task = _coarse_task_key(filter_task)
    if task and task not in families:
        return False
    return True


def _source_cache_key(path: str, kind: str, bundle_root: Optional[str]) -> Tuple[str, str, str]:
    return (path, kind, bundle_root or "")


def _file_mtime(path: str) -> Optional[float]:
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def _sharded_bundle_mtime(bundle_root: str) -> Optional[float]:
    times: List[float] = []
    meta = os.path.join(bundle_root, DATASET_METADATA_FILENAME)
    if os.path.isfile(meta):
        t = _file_mtime(meta)
        if t is not None:
            times.append(t)
    jsonl_dir = Path(bundle_root) / JSONL_SUBDIR
    if jsonl_dir.is_dir():
        for jf in jsonl_dir.glob("metadata_*.jsonl"):
            t = _file_mtime(str(jf))
            if t is not None:
                times.append(t)
    return max(times) if times else None


def _normalize_kind(kind: str) -> str:
    """``aggregate`` is a legacy alias for ``parquet``."""
    k = (kind or "parquet").strip().lower()
    return "parquet" if k == "aggregate" else k


def _build_source_cache_entry(path: str, kind: str, bundle_root: Optional[str]) -> dict:
    kind = _normalize_kind(kind)
    if kind == "export":
        root = bundle_root or path
        if not is_sharded_upstream_root(root):
            raise FileNotFoundError(
                f"Not a sharded export bundle (expected {JSONL_SUBDIR}/metadata_*.jsonl): {root}"
            )
        mtime = _sharded_bundle_mtime(root) or _file_mtime(path)
        recs = _load_sharded_export(root)
        facts: List[FilterFacts] = [
            _filter_facts_from_metadata(rec.get("metadata") or {}) for rec in recs
        ]
        return {
            "mtime": mtime,
            "kind": "export",
            "export_recs": recs,
            "facts": facts,
            "total": len(recs),
        }
    df = _load_parquet_df(path)
    facts = []
    for _, row in df.iterrows():
        meta = _json_load_col(row.get("metadata_json")) or {}
        facts.append(_filter_facts_from_metadata(meta))
    mtime = _file_mtime(path)
    return {"mtime": mtime, "kind": "parquet", "df": df, "facts": facts, "total": len(df)}


def _get_source_cache(path: str, kind: str, bundle_root: Optional[str]) -> dict:
    kind = _normalize_kind(kind)
    key = _source_cache_key(path, kind, bundle_root)
    if kind == "export":
        root = bundle_root or path
        mtime = _sharded_bundle_mtime(root) if root else _file_mtime(path)
    else:
        mtime = _file_mtime(path)
    entry = _SOURCE_CACHE.get(key)
    if entry is not None and entry.get("mtime") == mtime:
        return entry
    entry = _build_source_cache_entry(path, kind, bundle_root)
    _SOURCE_CACHE[key] = entry
    return entry


def discover_upstream_sources(data_dir: str) -> List[dict]:
    """Find sharded export bundles and upstream ``data.parquet`` tables under data_dir."""
    sources: List[dict] = []
    data_dir = os.path.abspath(data_dir)
    seen_pq: set = set()
    seen_bundles: set = set()

    for pq in sorted(
        glob.glob(
            os.path.join(data_dir, "**/merged_samples/data.parquet"),
            recursive=True,
        )
    ):
        pq = os.path.abspath(pq)
        if pq in seen_pq:
            continue
        seen_pq.add(pq)
        rel = os.path.relpath(pq, data_dir)
        low = rel.replace("\\", "/").lower()
        if "multiview" in low:
            label = f"[Parquet · Multi] {rel}"
        elif "singleview" in low:
            label = f"[Parquet · Single] {rel}"
        else:
            label = f"[Parquet] {rel}"
        sources.append({
            "label": label,
            "path": pq,
            "kind": "parquet",
            "bundle_root": None,
            "multiview": "multiview" in low,
        })

    for shard_jsonl in sorted(
        glob.glob(
            os.path.join(data_dir, "**", JSONL_SUBDIR, "metadata_*.jsonl"),
            recursive=True,
        )
    ):
        bundle_root = os.path.abspath(str(Path(shard_jsonl).parent.parent))
        if bundle_root in seen_bundles:
            continue
        if not is_sharded_upstream_root(bundle_root):
            continue
        seen_bundles.add(bundle_root)
        rel = os.path.relpath(bundle_root, data_dir)
        low = rel.replace("\\", "/").lower()
        if "multiview" in low:
            label = f"[Export · Sharded · Multi] {rel}"
        elif "singleview" in low:
            label = f"[Export · Sharded · Single] {rel}"
        else:
            label = f"[Export · Sharded] {rel}"
        sources.append({
            "label": label,
            "path": os.path.join(bundle_root, JSONL_SUBDIR),
            "kind": "export",
            "bundle_root": bundle_root,
            "multiview": "multiview" in low,
        })

    return sources


def _get_source(path: str, kind: str, bundle_root: Optional[str]) -> dict:
    for s in discover_upstream_sources(DATA_DIR):
        if s["path"] == path and s["kind"] == kind:
            return s
    return {"path": path, "kind": kind, "bundle_root": bundle_root, "multiview": False}


def _load_parquet_df(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def _load_sharded_export(bundle_root: str) -> List[dict]:
    root = os.path.abspath(bundle_root)
    if not is_sharded_upstream_root(root):
        raise FileNotFoundError(
            f"Expected sharded export at {root}/{JSONL_SUBDIR}/metadata_*.jsonl"
        )
    return read_sharded_upstream(root)


def _json_load_col(val: Any) -> Any:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, str):
        return json.loads(val)
    return val


def parquet_row_to_parse_dict(row) -> dict:
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
        "shard_tar": rec.get("_shard_tar"),
    }


def resolve_export_image(
    ref: str,
    bundle_root: str,
    tar_cache: dict,
    *,
    shard_tar: Optional[str] = None,
) -> Optional[Image.Image]:
    ref = str(ref).replace("\\", "/")
    local = os.path.join(bundle_root, ref)
    if os.path.isfile(local):
        return Image.open(local)

    raw = resolve_shard_image(
        ref,
        bundle_root=bundle_root,
        shard_tar=shard_tar or tar_cache.get("shard_tar"),
        tar_cache=tar_cache,
    )
    if raw:
        return Image.open(io.BytesIO(raw))
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
    shard_tar = parse_dict.get("shard_tar")
    for idx, ref in enumerate(refs):
        img = None
        if kind == "export" and bundle_root:
            img = resolve_export_image(
                ref, bundle_root, tar_cache, shard_tar=shard_tar
            )
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


def _upstream_grounding_task_label(parsed: dict, parse_dict: dict) -> str:
    prov = (parsed.get("meta") or {}).get("provenance") or {}
    for st in prov.get("source_tasks") or []:
        if "3d_grounding" in str(st).lower():
            return "3d_grounding"
    for tag in parse_dict.get("question_tags") or []:
        if "3d_grounding" in str(tag).lower():
            return "3d_grounding"
    for turn in parsed.get("turns") or []:
        tn = (turn.get("task_name") or "").strip()
        if "3d_grounding" in tn.lower():
            return tn
    return ""


def _upstream_can_3d_overlay(parsed: dict, parse_dict: dict) -> bool:
    return ann_viz._is_3d_grounding_task(
        _upstream_grounding_task_label(parsed, parse_dict), parsed,
    )


def build_upstream_display_images(
    parse_dict: dict,
    parsed: dict,
    *,
    kind: str,
    bundle_root: Optional[str],
    slot_ids=None,
    marks_mode: str = "off",
    overlay_3d: bool = False,
    tar_cache: Optional[dict] = None,
    mark_spec=None,
) -> Tuple[List[Image.Image], bool, bool]:
    tar_cache = tar_cache if tar_cache is not None else {}
    images = load_upstream_images(parse_dict, kind=kind, bundle_root=bundle_root, tar_cache=tar_cache)
    if not images:
        return [], False, False

    apply_marks = marks_mode in ("selected", "all")
    meta = parsed.get("meta") or {}
    if mark_spec is None:
        mark_spec = parsed.get("mark_spec") or meta.get("mark_spec")
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

    has_3d = False
    task_label = _upstream_grounding_task_label(parsed, parse_dict)
    if overlay_3d and ann_viz._is_3d_grounding_task(task_label, parsed):
        try:
            row_dict = dict(parse_dict)
            entries, ctx = ann_viz._resolve_grounding_context(row_dict, parsed)
            if entries and ctx is not None:
                intrinsic, ref_size = ctx
                out[0] = ann_viz.draw_3d_boxes_on_image(
                    out[0], entries, intrinsic, ref_size=ref_size,
                )
                has_3d = True
        except Exception as exc:
            print(f"[3d_grounding] overlay failed: {exc}")

    return out, has_3d, apply_marks


def serialize_upstream_raw(parse_dict: dict, parsed: dict, *, kind: str, bundle_root: Optional[str]) -> dict:
    row = ann_viz._numpy_to_python(parse_dict)
    row["metadata"] = ann_viz._numpy_to_python(parsed.get("meta"))
    row["messages"] = ann_viz._numpy_to_python(parse_dict.get("messages"))
    if kind == "export" and bundle_root:
        row["_bundle_root"] = bundle_root
    prov = (parsed.get("meta") or {}).get("provenance")
    if prov:
        row["_provenance"] = ann_viz._numpy_to_python(prov)
    return row


def _row_count(path: str, kind: str, bundle_root: Optional[str]) -> int:
    return _get_source_cache(path, kind, bundle_root)["total"]


def _load_parsed(path: str, kind: str, bundle_root: Optional[str], index: int):
    kind = _normalize_kind(kind)
    cache = _get_source_cache(path, kind, bundle_root)
    if kind == "export":
        recs = cache["export_recs"]
        if index < 0 or index >= len(recs):
            return None, None, None
        pd_dict = export_record_to_parse_dict(recs[index])
        series = pd.Series(pd_dict)
        parsed = ann_viz.parse_row(series)
        parsed["meta"] = ann_viz._normalize_meta(pd_dict.get("metadata"))
        return series, pd_dict, parsed
    df = cache["df"]
    if index < 0 or index >= len(df):
        return None, None, None
    row = df.iloc[index]
    pd_dict = parquet_row_to_parse_dict(row)
    series = pd.Series(pd_dict)
    parsed = ann_viz.parse_row(series)
    parsed["meta"] = ann_viz._normalize_meta(pd_dict.get("metadata"))
    return series, pd_dict, parsed


def _upstream_collect_task_names(path: str, kind: str, bundle_root: Optional[str]) -> List[str]:
    names: set = set()
    for _, families in _get_source_cache(path, kind, bundle_root)["facts"]:
        names.update(families)
    return sorted(names)


def _upstream_filtered_indices(
    path: str,
    kind: str,
    bundle_root: Optional[str],
    *,
    filter_task: str = "",
    filter_turns: str = "",
) -> List[int]:
    facts: List[FilterFacts] = _get_source_cache(path, kind, bundle_root)["facts"]
    if not (filter_task or "").strip() and not (filter_turns or "").strip():
        return list(range(len(facts)))
    out: List[int] = []
    for i, (n_turns, families) in enumerate(facts):
        if _row_matches_upstream_filters(
            n_turns, families, filter_task=filter_task, filter_turns=filter_turns,
        ):
            out.append(i)
    return out


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
  .filters { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .filters label { font-size: 12px; }
  .filters select { padding: 6px 10px; border-radius: 6px; border: none; font-size: 13px; }
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
  .images-row img { max-height: 380px; border-radius: 8px; border: 1px solid #eee; cursor: zoom-in; transition: transform 0.2s; object-fit: contain; }
  .images-row img:hover { transform: scale(1.02); }
  .lightbox { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.85); z-index: 200; justify-content: center; align-items: center; cursor: zoom-out; }
  .lightbox.active { display: flex; }
  .lightbox img { max-width: 95vw; max-height: 95vh; object-fit: contain; border-radius: 8px; cursor: default; }
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
  .loading-state { text-align: center; padding: 48px; color: #666; font-size: 15px; }
  .filters select:disabled { opacity: 0.65; cursor: wait; }
</style>
</head>
<body>
<div class="header">
  <h1>Upstream (Parquet / Sharded Export)</h1>
  <select id="sourceSelect" onchange="loadSource()">
    <option value="">-- select source --</option>
    {% for s in sources %}
    <option value="{{ s.path }}|{{ s.kind }}|{{ s.bundle_root or '' }}"
      {% if selected == s.path %}selected{% endif %}>{{ s.label }}</option>
    {% endfor %}
  </select>
  <div class="filters">
    <label>Task <select id="filterTask" onchange="applyFilters()"><option value="">All task types</option></select></label>
    <label>Turns <select id="filterTurns" onchange="applyFilters()">
      <option value="">Any</option><option value="1">1</option><option value="2">2</option>
      <option value="3">3</option><option value="2+">2+</option><option value="3+">3+</option>
    </select></label>
  </div>
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
<div class="lightbox" id="lightbox" onclick="closeLightbox()">
  <img id="lightboxImg" src="" alt="" onclick="event.stopPropagation()" />
</div>
<script>
let currentPath = '', currentKind = '', bundleRoot = '', currentPage = 0, totalRows = 0, filteredTotal = 0;
let fetchSeq = 0;
const pageSize = 8;

function filterParams() {
  const p = new URLSearchParams();
  const ft = document.getElementById('filterTask')?.value || '';
  const fn = document.getElementById('filterTurns')?.value || '';
  if (ft) p.set('filter_task', ft);
  if (fn) p.set('filter_turns', fn);
  return p;
}

function loadFilterTasks() {
  const sel = document.getElementById('filterTask');
  if (!sel || !currentPath) return Promise.resolve();
  const cur = sel.value;
  const q = new URLSearchParams({ path: currentPath, kind: currentKind, bundle_root: bundleRoot });
  return fetch('/api/filter_options?' + q.toString())
    .then(r => r.json())
    .then(data => {
      const tasks = data.tasks || [];
      sel.innerHTML = '<option value="">All task types</option>' +
        tasks.map(t => `<option value="${escapeHtml(t)}">${escapeHtml(t)}</option>`).join('');
      sel.value = (cur && tasks.includes(cur)) ? cur : '';
    })
    .catch(() => {
      sel.innerHTML = '<option value="">All task types</option>';
      sel.value = '';
    });
}

function setFiltersBusy(busy) {
  ['filterTask', 'filterTurns'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.disabled = busy;
  });
}

function showLoadingCards() {
  const el = document.getElementById('cards');
  if (el) el.innerHTML = '<div class="loading-state">Loading…</div>';
}

function applyFilters() {
  if (!currentPath) return;
  currentPage = 0;
  fetchPage();
}

function parseSelectVal() {
  const v = document.getElementById('sourceSelect').value;
  if (!v) return;
  const i1 = v.indexOf('|');
  const i2 = v.indexOf('|', i1 + 1);
  if (i1 < 0 || i2 < 0) return;
  currentPath = v.slice(0, i1);
  currentKind = v.slice(i1 + 1, i2);
  bundleRoot = v.slice(i2 + 1);
}

function resetFiltersForSourceChange() {
  const taskSel = document.getElementById('filterTask');
  const turnsSel = document.getElementById('filterTurns');
  if (taskSel) taskSel.value = '';
  if (turnsSel) turnsSel.value = '';
}

function loadSource() {
  parseSelectVal();
  currentPage = 0;
  if (!currentPath) return;
  resetFiltersForSourceChange();
  loadFilterTasks().then(() => fetchPage());
}

function fetchPage() {
  const seq = ++fetchSeq;
  setFiltersBusy(true);
  showLoadingCards();
  document.getElementById('pageInfo').textContent = 'Loading…';
  const q = filterParams();
  q.set('path', currentPath);
  q.set('kind', currentKind);
  q.set('bundle_root', bundleRoot);
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
      const el = document.getElementById('cards');
      if (el) el.innerHTML = '<div class="loading-state">Failed to load data.</div>';
    })
    .finally(() => {
      if (seq === fetchSeq) setFiltersBusy(false);
    });
}

function updateNav() {
  const start = filteredTotal ? currentPage * pageSize + 1 : 0;
  const end = Math.min((currentPage + 1) * pageSize, filteredTotal);
  const note = filteredTotal !== totalRows ? ` (${filteredTotal} matched / ${totalRows})` : '';
  document.getElementById('pageInfo').textContent =
    `rows ${start}-${end} / ${filteredTotal}${note}`;
  document.getElementById('headerInfo').textContent = currentKind + (bundleRoot ? ' · ' + bundleRoot : '');
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
  if (maxPage < 0) {
    currentPage = 0;
    fetchPage();
    return;
  }
  p = Math.max(1, Math.min(maxPage + 1, p));
  currentPage = p - 1;
  fetchPage();
}

function renderRows(rows) {
  const el = document.getElementById('cards');
  el.innerHTML = rows.map(r => cardHtml(r)).join('');
  rows.forEach(r => { if (r.mark_slots && r.mark_slots.length) refreshMarks(r.row_index); });
}

function cardForRow(rowIndex) {
  return document.querySelector(`.card[data-index="${rowIndex}"]`);
}

function toggleAllMarks(rowIndex, checked) {
  const card = cardForRow(rowIndex);
  if (!card) return;
  card.querySelectorAll('.mark-cb').forEach(cb => { cb.checked = checked; });
  const master = card.querySelector('.mark-select-all');
  if (master) master.checked = checked;
  refreshMarks(rowIndex);
}

function cardHtml(r) {
  const tasks = (r.source_tasks || []).join(', ');
  const imgInfo = (r.image_refs_count != null && r.images_loaded_count != null)
    ? `images: ${r.images_loaded_count}/${r.image_refs_count}`
    : '';
  const imgs = (r.display_images || []).map(src =>
    `<img src="${src}" onclick="openLightbox(this.src)" alt="">`).join('');
  const turns = (r.turns || []).map((t, i) => `
    <div class="qa-block">
      <span class="turn-badge">Turn ${i + 1}${t.task_name ? ' · ' + t.task_name : ''}${t.sub_task ? ' · ' + t.sub_task : ''}</span>
      <div class="qa-text q">${escapeHtml(t.question || '')}</div>
      <div class="qa-text a">${escapeHtml(t.answer || '')}</div>
    </div>`).join('');
  const marks = (r.mark_slots || []).map(s =>
    `<label><input type="checkbox" class="mark-cb" value="${s.slot_key}" checked onchange="refreshMarks(${r.row_index})"> ${escapeHtml(s.slot_key)} (${escapeHtml(s.mark_kind || '')})</label>`).join('');
  const markPanel = marks ? `<div class="mark-panel">
      <label style="font-weight:600;display:block;margin-bottom:6px;">
        <input type="checkbox" class="mark-select-all" checked onchange="toggleAllMarks(${r.row_index}, this.checked)"> Select all marks
      </label>${marks}</div>` : '';
  const overlay3d = r.can_3d_overlay ? `<label style="font-size:13px;display:block;margin-bottom:10px;">
      <input type="checkbox" class="overlay-3d-cb" checked onchange="refreshMarks(${r.row_index})"> Show 3D bbox overlay
    </label>` : '';
  return `<div class="card" data-index="${r.row_index}">
    <div class="card-header">
      <span class="tag tag-kind">${r.kind}</span>
      <span class="tag tag-task">${escapeHtml(tasks)}</span>
      <span class="meta-line">sample_id: ${escapeHtml(r.sample_id || '')}${imgInfo ? ' · ' + escapeHtml(imgInfo) : ''}</span>
      <button class="btn-raw" onclick="toggleRaw(${r.row_index}, this)">Raw record</button>
    </div>
    <div class="card-body">
      ${overlay3d}
      ${markPanel}
      <div class="images-row">${imgs}</div>
      ${turns}
      <pre class="raw-panel" id="raw-${r.row_index}"></pre>
    </div>
  </div>`;
}

function refreshMarks(rowIndex) {
  const card = cardForRow(rowIndex);
  if (!card) return;
  const slots = [...card.querySelectorAll('.mark-cb:checked')].map(c => c.value);
  const allCbs = card.querySelectorAll('.mark-cb');
  const master = card.querySelector('.mark-select-all');
  if (master && allCbs.length) {
    master.checked = slots.length === allCbs.length;
    master.indeterminate = slots.length > 0 && slots.length < allCbs.length;
  }
  const overlay3d = card.querySelector('.overlay-3d-cb')?.checked ?? false;
  const params = new URLSearchParams({
    path: currentPath, kind: currentKind, bundle_root: bundleRoot,
    index: String(rowIndex), slots: slots.join(','),
    marks_mode: slots.length ? 'selected' : 'off',
    overlay_3d: overlay3d ? '1' : '0',
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

function openLightbox(src) {
  const lb = document.getElementById('lightbox');
  const img = document.getElementById('lightboxImg');
  if (!lb || !img) return;
  img.src = src;
  lb.classList.add('active');
}

function closeLightbox() {
  const lb = document.getElementById('lightbox');
  if (!lb) return;
  lb.classList.remove('active');
}

document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeLightbox();
});

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


@app.route("/api/filter_options")
def api_filter_options():
    path = request.args.get("path", "")
    kind = _normalize_kind(request.args.get("kind", "parquet"))
    bundle_root = request.args.get("bundle_root") or None
    return jsonify({"tasks": _upstream_collect_task_names(path, kind, bundle_root)})


@app.route("/api/data")
def api_data():
    path = request.args.get("path", "")
    kind = _normalize_kind(request.args.get("kind", "parquet"))
    bundle_root = request.args.get("bundle_root") or None
    page = int(request.args.get("page", 0))
    page_size = int(request.args.get("page_size", 8))
    filter_task = request.args.get("filter_task", "")
    filter_turns = request.args.get("filter_turns", "")

    total = _row_count(path, kind, bundle_root)
    indices = _upstream_filtered_indices(
        path, kind, bundle_root,
        filter_task=filter_task, filter_turns=filter_turns,
    )
    filtered_total = len(indices)
    page_indices = indices[page * page_size : (page + 1) * page_size]
    tar_cache: dict = {}

    rows_out = []
    for i in page_indices:
        _, pd_dict, parsed = _load_parsed(path, kind, bundle_root, i)
        if pd_dict is None:
            continue
        can_3d = _upstream_can_3d_overlay(parsed, pd_dict)
        images, _, marks_applied = build_upstream_display_images(
            pd_dict, parsed, kind=kind, bundle_root=bundle_root,
            marks_mode="all", overlay_3d=can_3d, tar_cache=tar_cache,
            mark_spec=parsed.get("mark_spec"),
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
            "active_turn_index": parsed.get("active_turn_index"),
            "marks_differ_by_turn": parsed.get("marks_differ_by_turn", False),
            "mark_slots": parsed.get("mark_slots") or [],
            "display_images": [ann_viz.pil_to_base64(img) for img in images],
            "marks_overlay_applied": marks_applied,
            "can_3d_overlay": can_3d,
            "image_refs_count": len(refs),
            "images_loaded_count": len(images),
        })

    if "tar" in tar_cache:
        tar_cache["tar"].close()

    return jsonify({
        "total": total,
        "filtered_total": filtered_total,
        "page": page,
        "rows": rows_out,
    })


@app.route("/api/render")
def api_render():
    path = request.args.get("path", "")
    kind = _normalize_kind(request.args.get("kind", "parquet"))
    bundle_root = request.args.get("bundle_root") or None
    index = int(request.args.get("index", 0))
    slots_raw = request.args.get("slots", "")
    marks_mode = request.args.get("marks_mode", "off")
    overlay_3d = request.args.get("overlay_3d", "0") == "1"
    slot_ids = [s.strip() for s in slots_raw.split(",") if s.strip()]
    if marks_mode == "selected" and not slot_ids:
        marks_mode = "off"

    _, pd_dict, parsed = _load_parsed(path, kind, bundle_root, index)
    if pd_dict is None:
        return jsonify({"images": []})

    turn_index = request.args.get("turn_index")
    turn_ms = None
    if turn_index is not None and turn_index != "":
        ti = int(turn_index)
        specs = parsed.get("turn_mark_specs") or []
        if 0 <= ti < len(specs):
            turn_ms = specs[ti]

    images, _, marks_applied = build_upstream_display_images(
        pd_dict, parsed, kind=kind, bundle_root=bundle_root,
        slot_ids=slot_ids, marks_mode=marks_mode, overlay_3d=overlay_3d,
        mark_spec=turn_ms,
    )
    return jsonify({
        "images": [ann_viz.pil_to_base64(img) for img in images],
        "marks_overlay_applied": marks_applied,
    })


@app.route("/api/raw_row")
def api_raw_row():
    path = request.args.get("path", "")
    kind = _normalize_kind(request.args.get("kind", "parquet"))
    bundle_root = request.args.get("bundle_root") or None
    index = int(request.args.get("index", 0))

    _, pd_dict, parsed = _load_parsed(path, kind, bundle_root, index)
    if pd_dict is None:
        return jsonify({"row": {}})
    return jsonify({"row": serialize_upstream_raw(pd_dict, parsed, kind=kind, bundle_root=bundle_root)})


def _lan_urls(port: int) -> List[str]:
    """Collect likely LAN URLs for other machines (exclude loopback)."""
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
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
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
        lan = _lan_urls(port)
        if lan:
            print("  Other machines on the same network:")
            for url in lan:
                print(f"    {url}")
        else:
            print("  Other machines: use this PC's LAN IP, e.g. http://<your-ip>:{port}")
        print(
            "  If remote browsers cannot connect, allow this port in Windows Firewall "
            f"(Inbound rule for TCP {port})."
        )
    elif host in ("127.0.0.1", "localhost"):
        print("  WARNING: --host is loopback only; other machines cannot connect.")
        print(f"  Restart with: --host 0.0.0.0 --port {port}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenSpatial parquet / sharded-export visualizer")
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Bind address (default 0.0.0.0 = all interfaces, reachable on LAN)",
    )
    parser.add_argument("--data_dir", type=str, default="output/frame_rot")
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    sources = discover_upstream_sources(DATA_DIR)
    print(f"Found {len(sources)} upstream sources in {DATA_DIR}:")
    for s in sources:
        print(f"  {s['label']} -> {s['path']}")
    _print_listen_info(args.host, args.port)
    app.run(host=args.host, port=args.port, debug=False)
