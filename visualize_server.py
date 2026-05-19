"""
Visualization server for OpenSpatial annotation outputs.

Usage:
    python visualize_server.py --port 8888 --data_dir output/debug

Then open http://<host>:8888 in browser.
"""

import argparse
import ast
import base64
import io
import json
import os
import glob
import re

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, render_template_string, request, jsonify

from utils.box_utils import (
    compute_box_3d_corners_from_params,
    convert_box_3d_world_to_camera,
)

app = Flask(__name__)
DATA_DIR = "output/debug"

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def discover_parquets(data_dir):
    """Scan data_dir for all data.parquet files, return list of (display_name, path)."""
    results = []
    for pq_path in sorted(glob.glob(os.path.join(data_dir, "**/data.parquet"), recursive=True)):
        rel = os.path.relpath(pq_path, data_dir)
        parts = rel.split(os.sep)
        # e.g. base_pipeline_debug_counting/annotation_stage/counting/data.parquet
        # display as: counting  (singleview) or multiview_distance (multiview)
        task_name = parts[-2] if len(parts) >= 2 else rel
        pipeline_name = parts[0] if parts else ""
        is_multiview = "multiview" in task_name
        is_3d_grounding = "3d_grounding" in task_name.lower()
        label = f"{'[Multi] ' if is_multiview else '[Single] '}{task_name}"
        if is_3d_grounding:
            label += " (3D boxes)"
        results.append({
            "label": label,
            "path": pq_path,
            "task": task_name,
            "multiview": is_multiview,
            "grounding_3d": is_3d_grounding,
        })
    return results


def image_from_bytes(data):
    """Convert bytes/dict to PIL Image."""
    if isinstance(data, dict) and "bytes" in data:
        data = data["bytes"]
    if isinstance(data, bytes):
        return Image.open(io.BytesIO(data))
    return None


def pil_to_base64(img, max_w=800):
    """Convert PIL image to base64 data URI, resize if too large."""
    if img is None:
        return ""
    w, h = img.size
    if w > max_w:
        ratio = max_w / w
        img = img.resize((max_w, int(h * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def load_original_image(image_field):
    """Load original image from path string, bytes, or dict."""
    if isinstance(image_field, str) and os.path.exists(image_field):
        return Image.open(image_field)
    if isinstance(image_field, (bytes, dict)):
        return image_from_bytes(image_field)
    if isinstance(image_field, np.ndarray):
        # multiview: list of image paths
        imgs = []
        for item in image_field:
            if isinstance(item, str) and os.path.exists(item):
                imgs.append(Image.open(item))
        return imgs if imgs else None
    return None


# 3D grounding visualization (camera-frame boxes, zxy euler)
_BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)
_BOX_COLORS = (
    (255, 64, 64), (64, 200, 64), (64, 128, 255),
    (255, 180, 40), (200, 64, 255), (40, 220, 220),
)


def _is_3d_grounding_task(task_name, parsed):
    if task_name and "3d_grounding" in task_name.lower():
        return True
    tags = parsed.get("tags") or []
    if any("3D Grounding" in str(t) for t in tags):
        return True
    for turn in parsed.get("turns") or []:
        q = turn.get("question") or ""
        a = turn.get("answer") or ""
        if "Camera intrinsic parameters" in q and "bbox_3d" in a:
            return True
        if "bbox_3d" in a:
            return True
    return False


def _parse_float_token(s):
    """Parse a numeric token; tolerate trailing sentence punctuation (e.g. '48.49.')."""
    return float(s.strip().rstrip("."))


def _parse_camera_from_text(text):
    """Parse hfov, vfov, width, height from grounding_3d camera system prompt."""
    if not text:
        return None
    # vfov is often followed by '.' before "Image width" — avoid greedy [0-9.]+ swallowing it
    num = r"[0-9]+(?:\.[0-9]+)?"
    m = re.search(
        rf"hfov\s*=\s*({num}).*?vfov\s*=\s*({num}).*?"
        rf"(?:Image\s+)?width\s*=\s*(\d+).*?(?:Image\s+)?height\s*=\s*(\d+)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return None
    try:
        hfov = _parse_float_token(m.group(1))
        vfov = _parse_float_token(m.group(2))
        w, h = int(m.group(3)), int(m.group(4))
    except ValueError:
        return None
    return {"hfov": hfov, "vfov": vfov, "width": w, "height": h}


def _intrinsic_from_fov(hfov_deg, vfov_deg, width, height):
    h_rad, v_rad = np.radians(hfov_deg), np.radians(vfov_deg)
    fx = width / (2.0 * np.tan(h_rad / 2.0))
    fy = height / (2.0 * np.tan(v_rad / 2.0))
    k = np.eye(4, dtype=np.float64)
    k[0, 0], k[1, 1] = fx, fy
    k[0, 2], k[1, 2] = width / 2.0, height / 2.0
    return k


def _scale_intrinsic_to_image(intrinsic, ref_size, img_size):
    """Scale K from reference (W,H) in the prompt to the actual PIL image size."""
    ref_w, ref_h = ref_size
    img_w, img_h = img_size
    if ref_w <= 0 or ref_h <= 0:
        return intrinsic
    k = intrinsic.copy()
    sx, sy = img_w / ref_w, img_h / ref_h
    k[0, 0] *= sx
    k[0, 2] *= sx
    k[1, 1] *= sy
    k[1, 2] *= sy
    return k


def _load_pose_matrix(pose_field):
    if pose_field is None or (isinstance(pose_field, float) and np.isnan(pose_field)):
        return None
    if isinstance(pose_field, (list, np.ndarray)):
        arr = np.asarray(pose_field, dtype=np.float64)
        if arr.shape == (4, 4):
            return arr
        return None
    if isinstance(pose_field, str) and os.path.isfile(pose_field):
        return np.loadtxt(pose_field, dtype=np.float64)
    return None


def _intrinsic_from_row(row):
    raw = row.get("intrinsic", None)
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    k = np.asarray(raw, dtype=np.float64)
    if k.shape == (3, 3):
        k4 = np.eye(4, dtype=np.float64)
        k4[:3, :3] = k
        return k4
    if k.shape == (4, 4):
        return k
    return None


def _parse_bbox_entries_from_text(text):
    """Extract [{'bbox_3d': [...], 'label': ...}, ...] from an answer string."""
    if not text:
        return []
    m = re.search(r"(\[\s*\{.*\}\s*\])", text, flags=re.DOTALL)
    if not m:
        return []
    blob = m.group(1)
    for parser in (ast.literal_eval, lambda s: json.loads(s.replace("'", '"'))):
        try:
            data = parser(blob)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
        if not isinstance(data, list):
            continue
        entries = []
        for item in data:
            if isinstance(item, dict) and "bbox_3d" in item:
                box = item["bbox_3d"]
                if isinstance(box, (list, tuple)) and len(box) >= 9:
                    entries.append({
                        "bbox_3d": [float(v) for v in box[:9]],
                        "label": str(item.get("label", "")),
                    })
        if entries:
            return entries
    return []


def _bbox_entries_from_row(row):
    """Use world-frame boxes + pose when present in parquet (pre-flatten rows)."""
    boxes_world = row.get("bboxes_3d_world_coords", None)
    if boxes_world is None or (isinstance(boxes_world, float) and np.isnan(boxes_world)):
        return []
    pose = _load_pose_matrix(row.get("pose", None))
    if pose is None:
        return []

    tags = row.get("obj_tags", None)
    if tags is None:
        tags = []
    if isinstance(tags, np.ndarray):
        tags = tags.tolist()
    if isinstance(boxes_world, np.ndarray):
        boxes_world = boxes_world.tolist()

    entries = []
    for idx, box in enumerate(boxes_world):
        if box is None or len(box) < 9:
            continue
        cam_box = convert_box_3d_world_to_camera(box, pose)
        if cam_box is None:
            continue
        label = ""
        if idx < len(tags):
            tag = tags[idx]
            if isinstance(tag, (list, tuple)) and tag:
                label = str(tag[0])
            else:
                label = str(tag)
        entries.append({"bbox_3d": cam_box, "label": label})
    return entries


def _project_cam_to_2d(points_cam, intrinsic):
    pts = np.asarray(points_cam, dtype=np.float64)
    uv = np.full((pts.shape[0], 2), np.nan)
    valid = pts[:, 2] > 1e-3
    z = pts[valid, 2]
    uv[valid, 0] = intrinsic[0, 0] * pts[valid, 0] / z + intrinsic[0, 2]
    uv[valid, 1] = intrinsic[1, 1] * pts[valid, 1] / z + intrinsic[1, 2]
    return uv, valid


def _resolve_grounding_context(row, parsed):
    """Return (bbox_entries, intrinsic_4x4) or (None, None) if overlay cannot be built."""
    entries = _bbox_entries_from_row(row)
    intrinsic = _intrinsic_from_row(row)
    ref_size = None

    if not entries:
        for turn in parsed.get("turns") or []:
            entries = _parse_bbox_entries_from_text(turn.get("answer") or "")
            if entries:
                break

    if not entries:
        return None, None

    if intrinsic is None:
        cam = None
        for turn in parsed.get("turns") or []:
            cam = _parse_camera_from_text(turn.get("question") or "")
            if cam:
                break
        if cam is None:
            return entries, None
        intrinsic = _intrinsic_from_fov(cam["hfov"], cam["vfov"], cam["width"], cam["height"])
        ref_size = (cam["width"], cam["height"])

    return entries, (intrinsic, ref_size)


def draw_3d_boxes_on_image(img, entries, intrinsic, ref_size=None):
    """Draw projected 3D box wireframes on a PIL image (in-place copy)."""
    if not entries or intrinsic is None:
        return img
    out = img.convert("RGB")
    w, h = out.size
    k = intrinsic
    if ref_size is not None:
        k = _scale_intrinsic_to_image(k, ref_size, (w, h))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None

    for i, entry in enumerate(entries):
        color = _BOX_COLORS[i % len(_BOX_COLORS)]
        corners = compute_box_3d_corners_from_params(entry["bbox_3d"])
        uv, valid = _project_cam_to_2d(corners, k)
        for i0, i1 in _BOX_EDGES:
            if not (valid[i0] and valid[i1]):
                continue
            p0 = tuple(uv[i0])
            p1 = tuple(uv[i1])
            if any(np.isnan(c) for c in p0 + p1):
                continue
            draw.line([p0, p1], fill=color, width=2)
        vis = np.where(valid)[0]
        if len(vis) > 0 and font is not None:
            cx = int(np.nanmean(uv[vis, 0]))
            cy = int(np.nanmean(uv[vis, 1]))
            label = entry.get("label") or f"box{i}"
            draw.text((cx + 4, cy + 4), label, fill=color, font=font)
    return out


def parse_row(row):
    """Parse a single parquet row into a display-friendly dict.

    Supports both single-turn (2 messages) and multi-turn (4+ messages) conversations.
    Returns a list of (question, answer) turns.
    """
    messages = row.get("messages", [])
    if isinstance(messages, np.ndarray):
        messages = messages.tolist()

    # Parse all turns as (question, answer) pairs
    turns = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if isinstance(msg, dict) and msg.get("from") == "human":
            q = msg.get("value", "")
            a = ""
            if i + 1 < len(messages):
                next_msg = messages[i + 1]
                if isinstance(next_msg, dict) and next_msg.get("from") == "gpt":
                    a = next_msg.get("value", "")
                    i += 1
            turns.append({"question": q, "answer": a})
        i += 1

    # QA images
    qa_images_raw = row.get("QA_images", None)
    qa_images = []
    if isinstance(qa_images_raw, dict):
        img = image_from_bytes(qa_images_raw)
        if img:
            qa_images.append(img)
    elif isinstance(qa_images_raw, (list, np.ndarray)):
        for item in qa_images_raw:
            img = image_from_bytes(item)
            if img:
                qa_images.append(img)

    tags = row.get("question_tags", [])
    if isinstance(tags, np.ndarray):
        tags = tags.tolist()
    qtype = row.get("question_types", "")

    return {
        "turns": turns,
        "qa_images": qa_images,
        "tags": tags,
        "question_type": qtype,
    }


# ──────────────────────────────────────────────────────────────────────
# HTML Template
# ──────────────────────────────────────────────────────────────────────

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OpenSpatial Visualizer</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; color: #333; }
  .header { background: #1a1a2e; color: white; padding: 16px 24px; display: flex; align-items: center; gap: 16px; position: sticky; top: 0; z-index: 100; }
  .header h1 { font-size: 20px; font-weight: 600; }
  .header select { padding: 8px 12px; border-radius: 6px; border: none; font-size: 14px; background: #16213e; color: white; cursor: pointer; min-width: 280px; }
  .header select option { background: #16213e; }
  .header .info { margin-left: auto; font-size: 13px; opacity: 0.8; }
  .nav { display: flex; align-items: center; gap: 8px; margin-left: 16px; }
  .nav button { padding: 6px 14px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.3); background: transparent; color: white; cursor: pointer; font-size: 13px; transition: all 0.2s; }
  .nav button:hover { background: rgba(255,255,255,0.15); }
  .nav button:disabled { opacity: 0.3; cursor: default; }
  .nav span { color: rgba(255,255,255,0.7); font-size: 13px; min-width: 80px; text-align: center; }
  .container { max-width: 1200px; margin: 24px auto; padding: 0 24px; }
  .card { background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 20px; overflow: hidden; }
  .card-header { padding: 14px 20px; background: #f8f9fa; border-bottom: 1px solid #eee; display: flex; align-items: center; gap: 10px; }
  .tag { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }
  .tag-task { background: #e3f2fd; color: #1565c0; }
  .tag-type { background: #f3e5f5; color: #7b1fa2; }
  .card-body { padding: 20px; }
  .images-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }
  .images-row img { border-radius: 8px; border: 1px solid #eee; cursor: pointer; transition: transform 0.2s; max-height: 400px; object-fit: contain; }
  .images-row img:hover { transform: scale(1.02); }
  .qa-block { margin-top: 12px; }
  .qa-label { font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
  .qa-label.q { color: #1565c0; }
  .qa-label.a { color: #2e7d32; }
  .qa-text { padding: 12px 16px; border-radius: 8px; font-size: 14px; line-height: 1.6; white-space: pre-wrap; word-break: break-word; }
  .qa-text.q { background: #e3f2fd; }
  .qa-text.a { background: #e8f5e9; }
  .turn-divider { border: none; border-top: 1px dashed #ddd; margin: 12px 0; }
  .turn-badge { display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; background: #fff3e0; color: #e65100; margin-left: 6px; }
  .multi-turn-label { font-size: 12px; color: #888; margin-bottom: 4px; }
  /* Lightbox */
  .lightbox { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.85); z-index: 200; justify-content: center; align-items: center; cursor: zoom-out; }
  .lightbox.active { display: flex; }
  .lightbox img { max-width: 95vw; max-height: 95vh; object-fit: contain; border-radius: 8px; }
  .empty-state { text-align: center; padding: 80px 20px; color: #999; }
  .empty-state h2 { font-size: 24px; margin-bottom: 8px; }
</style>
</head>
<body>

<div class="header">
  <h1>OpenSpatial Visualizer</h1>
  <select id="taskSelect" onchange="loadTask()">
    <option value="">-- Select a task --</option>
    {% for t in tasks %}
    <option value="{{ t.path }}" {{ 'selected' if t.path == selected_path else '' }}>{{ t.label }}</option>
    {% endfor %}
  </select>
  <div class="nav">
    <button id="prevBtn" onclick="navigate(-1)" disabled>&larr; Prev</button>
    <span id="pageInfo">-</span>
    <button id="nextBtn" onclick="navigate(1)" disabled>Next &rarr;</button>
  </div>
  <div class="info" id="totalInfo"></div>
</div>

<div class="container" id="content">
  <div class="empty-state">
    <h2>Select a task to visualize</h2>
    <p>Choose an annotation output from the dropdown above</p>
  </div>
</div>

<!-- Lightbox -->
<div class="lightbox" id="lightbox" onclick="closeLightbox()">
  <img id="lightboxImg" src="" />
</div>

<script>
const PAGE_SIZE = 10;
let currentPage = 0;
let totalRows = 0;

function loadTask() {
  const path = document.getElementById('taskSelect').value;
  if (!path) return;
  currentPage = 0;
  fetchPage(path, 0);
}

function navigate(delta) {
  const path = document.getElementById('taskSelect').value;
  if (!path) return;
  currentPage += delta;
  fetchPage(path, currentPage);
}

function fetchPage(path, page) {
  fetch(`/api/data?path=${encodeURIComponent(path)}&page=${page}&page_size=${PAGE_SIZE}`)
    .then(r => r.json())
    .then(data => {
      totalRows = data.total;
      currentPage = data.page;
      renderRows(data.rows);
      updateNav();
    });
}

function updateNav() {
  const totalPages = Math.ceil(totalRows / PAGE_SIZE);
  document.getElementById('pageInfo').textContent = totalPages > 0 ? `${currentPage + 1} / ${totalPages}` : '-';
  document.getElementById('prevBtn').disabled = currentPage <= 0;
  document.getElementById('nextBtn').disabled = currentPage >= totalPages - 1;
  document.getElementById('totalInfo').textContent = `${totalRows} rows total`;
}

function renderRows(rows) {
  const container = document.getElementById('content');
  if (rows.length === 0) {
    container.innerHTML = '<div class="empty-state"><h2>No data</h2><p>This task produced no output rows.</p></div>';
    return;
  }
  let html = '';
  rows.forEach((row, idx) => {
    const globalIdx = currentPage * PAGE_SIZE + idx;
    const tagsHtml = (row.tags || []).map(t => `<span class="tag tag-task">${t}</span>`).join(' ');
    const typeHtml = row.question_type ? `<span class="tag tag-type">${row.question_type}</span>` : '';
    const isMultiTurn = row.turns && row.turns.length > 1;
    const turnBadge = isMultiTurn ? `<span class="turn-badge">${row.turns.length} turns</span>` : '';
    const overlayBadge = row.has_3d_overlay ? '<span class="tag tag-type">3D bbox overlay</span>' : '';

    let imagesHtml = '';
    if (row.qa_images && row.qa_images.length > 0) {
      const imgTags = row.qa_images.map(src =>
        `<img src="${src}" onclick="openLightbox('${src}')" style="max-width:${row.qa_images.length > 1 ? Math.floor(100/Math.min(row.qa_images.length, 4)) - 2 : 100}%;" />`
      ).join('');
      imagesHtml = `<div class="images-row">${imgTags}</div>`;
    }

    let turnsHtml = '';
    if (row.turns && row.turns.length > 0) {
      row.turns.forEach((turn, tIdx) => {
        const cleanQ = (turn.question || '').replace(/<image>\s*/g, '').trim();
        const turnLabel = isMultiTurn ? `<span class="multi-turn-label">Turn ${tIdx + 1}</span>` : '';
        if (tIdx > 0) turnsHtml += '<hr class="turn-divider">';
        turnsHtml += `
          ${turnLabel}
          <div class="qa-block">
            <div class="qa-label q">Question</div>
            <div class="qa-text q">${escapeHtml(cleanQ)}</div>
          </div>
          <div class="qa-block" style="margin-top: 10px;">
            <div class="qa-label a">Answer</div>
            <div class="qa-text a">${escapeHtml(turn.answer || '')}</div>
          </div>`;
      });
    }

    html += `
    <div class="card">
      <div class="card-header">
        <strong>#${globalIdx + 1}</strong>
        ${tagsHtml} ${typeHtml} ${overlayBadge} ${turnBadge}
      </div>
      <div class="card-body">
        ${imagesHtml}
        ${turnsHtml}
      </div>
    </div>`;
  });
  container.innerHTML = html;
  window.scrollTo(0, 0);
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function openLightbox(src) {
  document.getElementById('lightboxImg').src = src;
  document.getElementById('lightbox').classList.add('active');
}
function closeLightbox() {
  document.getElementById('lightbox').classList.remove('active');
}
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeLightbox();
  if (e.key === 'ArrowLeft') navigate(-1);
  if (e.key === 'ArrowRight') navigate(1);
});

// Auto-load if a task is pre-selected
window.onload = () => {
  const sel = document.getElementById('taskSelect');
  if (sel.value) loadTask();
};
</script>
</body>
</html>
"""


# ──────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    tasks = discover_parquets(DATA_DIR)
    selected = request.args.get("task", "")
    return render_template_string(HTML_TEMPLATE, tasks=tasks, selected_path=selected)


def _task_name_from_parquet_path(path):
    parts = os.path.normpath(path).split(os.sep)
    return parts[-2] if len(parts) >= 2 else ""


@app.route("/api/data")
def api_data():
    path = request.args.get("path", "")
    page = int(request.args.get("page", 0))
    page_size = int(request.args.get("page_size", 10))

    if not path or not os.path.exists(path):
        return jsonify({"total": 0, "page": 0, "rows": []})

    df = pd.read_parquet(path)
    total = len(df)
    start = page * page_size
    end = min(start + page_size, total)
    task_name = _task_name_from_parquet_path(path)

    rows = []
    for i in range(start, end):
        row = df.iloc[i]
        parsed = parse_row(row)
        qa_images = list(parsed["qa_images"])
        has_3d_overlay = False

        if _is_3d_grounding_task(task_name, parsed):
            try:
                entries, ctx = _resolve_grounding_context(row, parsed)
                if entries and ctx is not None:
                    intrinsic, ref_size = ctx
                    if qa_images:
                        qa_images[0] = draw_3d_boxes_on_image(
                            qa_images[0], entries, intrinsic, ref_size=ref_size,
                        )
                        has_3d_overlay = True
                    else:
                        orig = load_original_image(row.get("image"))
                        if orig is not None and not isinstance(orig, list):
                            qa_images = [
                                draw_3d_boxes_on_image(
                                    orig, entries, intrinsic, ref_size=ref_size,
                                ),
                            ]
                            has_3d_overlay = True
            except Exception as exc:
                print(f"[3d_grounding] row {start + i} overlay failed: {exc}")

        img_b64_list = [pil_to_base64(img) for img in qa_images]
        rows.append({
            "turns": parsed["turns"],
            "qa_images": img_b64_list,
            "tags": parsed["tags"] if isinstance(parsed["tags"], list) else [parsed["tags"]],
            "question_type": parsed["question_type"],
            "has_3d_overlay": has_3d_overlay,
        })

    return jsonify({"total": total, "page": page, "rows": rows})


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenSpatial Annotation Visualizer")
    parser.add_argument("--port", type=int, default=8888, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--data_dir", type=str, default="output/debug", help="Root directory containing parquet outputs")
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    tasks = discover_parquets(DATA_DIR)
    print(f"Found {len(tasks)} task outputs in {DATA_DIR}:")
    for t in tasks:
        print(f"  {t['label']} -> {t['path']}")
    print(f"\nStarting server at http://{args.host}:{args.port}")

    app.run(host=args.host, port=args.port, debug=False)
