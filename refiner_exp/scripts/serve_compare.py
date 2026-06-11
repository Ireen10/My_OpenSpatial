#!/usr/bin/env python3
"""Local web server for Raw/SAM2/SAM3 refiner comparison (interactive 3D viewer)."""

from __future__ import annotations

import argparse
import html as html_module
import io
import json
import socket
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from visualize_compare import (  # noqa: E402
    CompareIndex,
    _log,
    _pack_branch_from_fusion,
    branch_records_for_image,
    build_compare_index,
    build_object_color_map,
    render_object_panel_image,
    sample_panels_json,
    sample_stats_json,
)


_INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Refiner Compare</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; background: #111; color: #eee; }}
    h1 {{ font-size: 1.2rem; }}
    a {{ color: #7eb8ff; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    li {{ margin: 6px 0; }}
    .hint {{ color: #888; font-size: 0.9rem; }}
  </style>
</head>
<body>
  <h1>Refiner 对照实验</h1>
  <p class="hint">共 {count} 个样本 · 点云按需从 fusion .pcd 加载</p>
  <ul>
{rows}
  </ul>
</body>
</html>
"""

_VIEWER_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: system-ui, sans-serif; background: #1a1a1a; color: #eee; }}
    header {{ padding: 12px 16px; background: #242424; border-bottom: 1px solid #333; }}
    header h1 {{ margin: 0 0 6px; font-size: 1.05rem; }}
    header p {{ margin: 0; font-size: 0.85rem; color: #aaa; }}
    a {{ color: #7eb8ff; }}
    .panels-2d {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; padding: 12px 16px; background: #111; }}
    .branch-col {{ background: #242424; border: 1px solid #333; border-radius: 6px; padding: 8px; }}
    .branch-col h3 {{ margin: 0 0 8px; font-size: 0.9rem; color: #ccc; }}
    .thumb-list {{ display: flex; flex-direction: column; gap: 8px; }}
    .thumb-list img {{ width: 100%; border: 1px solid #444; border-radius: 4px; cursor: zoom-in; display: block; content-visibility: auto; contain-intrinsic-size: 360px; }}
    .thumb-list .empty {{ color: #888; font-size: 0.85rem; padding: 8px 0; }}
    .lightbox {{ position: fixed; inset: 0; z-index: 1000; display: flex; align-items: center; justify-content: center; }}
    .lightbox.hidden {{ display: none; }}
    .lightbox-backdrop {{ position: absolute; inset: 0; background: rgba(0,0,0,0.88); }}
    #lightbox-img {{ position: relative; max-width: 96vw; max-height: 96vh; object-fit: contain; border-radius: 4px; cursor: zoom-out; }}
    .viewers {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; padding: 12px 16px 20px; }}
    .panel {{ background: #242424; border: 1px solid #333; border-radius: 6px; min-height: 320px; display: flex; flex-direction: column; }}
    .panel h2 {{ margin: 0; padding: 8px 10px; font-size: 0.9rem; background: #2e2e2e; }}
    .panel .meta {{ padding: 6px 10px; font-size: 0.78rem; color: #aaa; }}
    .panel canvas {{ flex: 1; width: 100%; min-height: 280px; cursor: grab; }}
    .stats {{ padding: 12px 16px; font-size: 0.8rem; color: #aaa; max-height: 200px; overflow: auto; }}
    @media (max-width: 960px) {{
      .panels-2d {{ grid-template-columns: 1fr; }}
      .viewers {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <p>拖拽旋转 · 滚轮缩放 · 右键平移 · <span style="color:#ccc">点击图片放大</span></p>
    <p><a href="/">← 样本列表</a> · <a href="?refresh=1">刷新 2D 图</a></p>
  </header>
  <div class="panels-2d">
    <div class="branch-col"><h3>RAW</h3><div class="thumb-list" id="panels-raw"></div></div>
    <div class="branch-col"><h3>SAM2</h3><div class="thumb-list" id="panels-sam2"></div></div>
    <div class="branch-col"><h3>SAM3</h3><div class="thumb-list" id="panels-sam3"></div></div>
  </div>
  <div id="lightbox" class="lightbox hidden">
    <div class="lightbox-backdrop"></div>
    <img id="lightbox-img" alt="放大查看">
  </div>
  <div class="viewers">
    <div class="panel"><h2>RAW</h2><div class="meta" id="meta-raw">加载中…</div><canvas id="view-raw"></canvas></div>
    <div class="panel"><h2>SAM2</h2><div class="meta" id="meta-sam2">加载中…</div><canvas id="view-sam2"></canvas></div>
    <div class="panel"><h2>SAM3</h2><div class="meta" id="meta-sam3">加载中…</div><canvas id="view-sam3"></canvas></div>
  </div>
  <pre class="stats" id="stats"></pre>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
  <script>
    const SAMPLE = {name_json};
    const REFRESH = new URLSearchParams(location.search).has("refresh");
    const lightbox = document.getElementById("lightbox");
    const lightboxImg = document.getElementById("lightbox-img");
    function openLightbox(src) {{
      lightboxImg.src = src;
      lightbox.classList.remove("hidden");
    }}
    lightbox.addEventListener("click", () => lightbox.classList.add("hidden"));
    document.addEventListener("keydown", (e) => {{
      if (e.key === "Escape") lightbox.classList.add("hidden");
    }});

    function panelUrl(branch, objectIndex) {{
      let url = "/api/sample/" + SAMPLE + "/panel/" + branch + "/" + objectIndex;
      if (REFRESH) url += "?refresh=1";
      return url;
    }}

    fetch("/api/sample/" + SAMPLE + "/panels").then(r => r.json()).then((data) => {{
      for (const branch of ["raw", "sam2", "sam3"]) {{
        const list = document.getElementById("panels-" + branch);
        const items = (data.branches && data.branches[branch]) || [];
        if (!items.length) {{
          list.innerHTML = '<div class="empty">无物体</div>';
          continue;
        }}
        for (const item of items) {{
          const img = document.createElement("img");
          img.loading = "lazy";
          img.decoding = "async";
          img.src = panelUrl(branch, item.object_index);
          img.alt = item.label || "";
          img.title = (item.label || "") + " — 点击放大";
          img.addEventListener("click", () => openLightbox(img.src));
          list.appendChild(img);
        }}
      }}
    }}).catch(err => {{
      for (const branch of ["raw", "sam2", "sam3"]) {{
        const list = document.getElementById("panels-" + branch);
        if (list) list.innerHTML = '<div class="empty">加载失败: ' + err + '</div>';
      }}
    }});

    function b64ToBytes(b64) {{
      const bin = atob(b64);
      const out = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
      return out;
    }}

    function mountViewer(canvasId, metaId, packed) {{
      const canvas = document.getElementById(canvasId);
      const meta = document.getElementById(metaId);
      const parent = canvas.parentElement;
      const w = Math.max(parent.clientWidth, 280);
      const h = Math.max(280, Math.floor(w * 0.75));
      const renderer = new THREE.WebGLRenderer({{ canvas, antialias: false, powerPreference: "high-performance" }});
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.25));
      renderer.setSize(w, h, false);
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x141414);
      const camera = new THREE.PerspectiveCamera(50, w / h, 0.001, 500);
      const controls = new THREE.OrbitControls(camera, canvas);
      controls.enableDamping = false;
      let visible = false;
      let framePending = false;
      const objects = (packed && packed.objects) ? packed.objects : [];
      if (!objects.length) {{
        meta.textContent = "无点云";
        camera.position.set(0, 0, 2);
        controls.target.set(0, 0, 1);
        controls.update();
        renderer.render(scene, camera);
        return;
      }}
      const extent = packed.extent || 1;
      const pointSize = Math.max(extent * 0.012, 0.004);
      let totalPts = 0;
      for (const obj of objects) {{
        const n = obj.n || 0;
        if (!n) continue;
        totalPts += n;
        const positions = new Float32Array(b64ToBytes(obj.positions).buffer);
        const rgb = obj.color || [200, 200, 200];
        const colorAttr = new Float32Array(n * 3);
        for (let i = 0; i < n; i++) {{
          colorAttr[i*3] = rgb[0]/255; colorAttr[i*3+1] = rgb[1]/255; colorAttr[i*3+2] = rgb[2]/255;
        }}
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute("color", new THREE.BufferAttribute(colorAttr, 3));
        scene.add(new THREE.Points(geometry, new THREE.PointsMaterial({{
          size: pointSize, vertexColors: true, sizeAttenuation: true,
        }})));
      }}
      const c = packed.centroid;
      controls.target.set(c[0], c[1], c[2]);
      camera.position.set(c[0], c[1] + extent * 0.15, c[2] + extent * 1.4);
      controls.update();
      meta.textContent = objects.length + " objects · " + totalPts.toLocaleString() + " pts";
      function renderOnce() {{
        controls.update();
        renderer.render(scene, camera);
      }}
      function scheduleRender() {{
        if (!visible || framePending) return;
        framePending = true;
        requestAnimationFrame(() => {{
          framePending = false;
          if (visible) renderOnce();
        }});
      }}
      controls.addEventListener("change", scheduleRender);
      const io = new IntersectionObserver((entries) => {{
        visible = entries[0].isIntersecting;
        if (visible) scheduleRender();
      }}, {{ threshold: 0.08 }});
      io.observe(parent);
      renderOnce();
    }}

    async function loadBranch(branch, canvasId, metaId) {{
      const res = await fetch("/api/sample/" + SAMPLE + "/points/" + branch);
      const packed = await res.json();
      mountViewer(canvasId, metaId, packed);
    }}

    function load3DViewers() {{
      return Promise.all([
        loadBranch("raw", "view-raw", "meta-raw"),
        loadBranch("sam2", "view-sam2", "meta-sam2"),
        loadBranch("sam3", "view-sam3", "meta-sam3"),
      ]);
    }}

    fetch("/api/sample/" + SAMPLE + "/stats").then(r => r.json()).then((stats) => {{
      document.getElementById("stats").textContent = JSON.stringify(stats, null, 2);
    }}).catch(err => {{
      document.getElementById("stats").textContent = "加载失败: " + err;
    }});

    const viewersSection = document.querySelector(".viewers");
    const io3d = new IntersectionObserver((entries) => {{
      if (!entries[0].isIntersecting) return;
      io3d.disconnect();
      load3DViewers().catch(err => {{
        for (const id of ["meta-raw", "meta-sam2", "meta-sam3"]) {{
          const el = document.getElementById(id);
          if (el) el.textContent = "点云加载失败: " + err;
        }}
      }});
    }}, {{ rootMargin: "120px", threshold: 0.01 }});
    io3d.observe(viewersSection);
  </script>
</body>
</html>
"""


class CompareServerState:
    def __init__(
        self,
        index: CompareIndex,
        *,
        cache_dir: Path,
        max_points_per_object: int,
        max_panel_width: int,
        memory_cache_mb: int = 4096,
    ):
        self.index = index
        self.cache_dir = cache_dir
        self.max_points_per_object = max_points_per_object
        self.max_panel_width = max_panel_width
        self._mem_limit = max(memory_cache_mb, 256) * 1024 * 1024
        self._mem_used = 0
        self._panel_mem: Dict[str, bytes] = {}
        self._points_mem: Dict[str, bytes] = {}
        self._mem_lock = threading.RLock()
        self._panel_locks: Dict[str, threading.Lock] = {}
        self._panel_locks_guard = threading.Lock()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _panel_lock(self, key: str) -> threading.Lock:
        with self._panel_locks_guard:
            if key not in self._panel_locks:
                self._panel_locks[key] = threading.Lock()
            return self._panel_locks[key]

    def _mem_store(self, store: Dict[str, bytes], key: str, data: bytes) -> None:
        with self._mem_lock:
            old = store.get(key)
            if old is not None:
                self._mem_used -= len(old)
            while store and self._mem_used + len(data) > self._mem_limit:
                evict_key = next(iter(store))
                evicted = store.pop(evict_key)
                self._mem_used -= len(evicted)
            store[key] = data
            self._mem_used += len(data)

    def panel_bytes(
        self,
        name: str,
        branch: str,
        object_index: int,
        *,
        refresh: bool,
    ) -> Optional[bytes]:
        image_key = self.index.key_by_name.get(name)
        if image_key is None:
            return None
        mem_key = f"{name}/{branch}_{object_index}"
        if not refresh:
            with self._mem_lock:
                cached = self._panel_mem.get(mem_key)
            if cached is not None:
                return cached
        sample_cache = self.cache_dir / name
        sample_cache.mkdir(parents=True, exist_ok=True)
        disk_path = sample_cache / f"{branch}_{object_index}.jpg"
        if not refresh and disk_path.is_file():
            data = disk_path.read_bytes()
            self._mem_store(self._panel_mem, mem_key, data)
            return data
        with self._panel_lock(mem_key):
            if not refresh:
                with self._mem_lock:
                    cached = self._panel_mem.get(mem_key)
                if cached is not None:
                    return cached
                if disk_path.is_file():
                    data = disk_path.read_bytes()
                    self._mem_store(self._panel_mem, mem_key, data)
                    return data
            panel = render_object_panel_image(
                self.index,
                image_key,
                branch,
                object_index,
                max_panel_width=self.max_panel_width,
            )
            if panel is None:
                return None
            buf = io.BytesIO()
            panel.save(buf, format="JPEG", quality=90, optimize=True)
            data = buf.getvalue()
            disk_path.write_bytes(data)
            self._mem_store(self._panel_mem, mem_key, data)
            return data

    def points_bytes(self, image_key: str, branch: str, *, refresh: bool = False) -> bytes:
        mem_key = f"pts:{image_key}|{branch}"
        if not refresh:
            with self._mem_lock:
                cached = self._points_mem.get(mem_key)
            if cached is not None:
                return cached
        branch_records = branch_records_for_image(self.index, image_key)
        color_map = build_object_color_map(self.index, image_key)
        packed = _pack_branch_from_fusion(
            branch_records[branch],
            self.index.fusion_by_key[branch],
            self.max_points_per_object,
            color_map=color_map,
        )
        data = json.dumps(packed, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._mem_store(self._points_mem, mem_key, data)
        return data


def _iter_panel_jobs(index: CompareIndex) -> List[Tuple[str, str, int]]:
    jobs: List[Tuple[str, str, int]] = []
    for image_key in index.image_keys:
        name = index.name_by_key[image_key]
        manifest = sample_panels_json(index, image_key)
        for branch in ("raw", "sam2", "sam3"):
            for item in manifest["branches"].get(branch, []):
                jobs.append((name, branch, int(item["object_index"])))
    return jobs


def _preload_assets(state: CompareServerState, *, workers: int, preload_panels: bool, preload_points: bool) -> None:
    panel_jobs = _iter_panel_jobs(state.index) if preload_panels else []
    point_jobs: List[Tuple[str, str]] = []
    if preload_points:
        for image_key in state.index.image_keys:
            for branch in ("raw", "sam2", "sam3"):
                point_jobs.append((image_key, branch))

    total = len(panel_jobs) + len(point_jobs)
    if total == 0:
        return
    _log(f"Preloading {len(panel_jobs)} panel(s) + {len(point_jobs)} point payload(s) with {workers} workers...")

    done = 0
    lock = threading.Lock()

    def _tick() -> None:
        nonlocal done
        with lock:
            done += 1
            if done % 50 == 0 or done == total:
                _log(f"  preload {done}/{total}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = []
        for name, branch, obj_idx in panel_jobs:
            futures.append(pool.submit(state.panel_bytes, name, branch, obj_idx, refresh=False))
        for image_key, branch in point_jobs:
            futures.append(pool.submit(state.points_bytes, image_key, branch, refresh=False))
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as exc:
                _log(f"  preload warning: {exc}")
            _tick()
    with state._mem_lock:
        mem_mb = state._mem_used / (1024 * 1024)
    _log(f"Preload done. Memory cache ~{mem_mb:.0f} MB (limit {state._mem_limit // (1024 * 1024)} MB).")


def make_handler(state: CompareServerState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            _log(f"[{self.address_string()}] {fmt % args}")

        def _send_bytes(
            self,
            data: bytes,
            content_type: str,
            *,
            status: int = 200,
            cacheable: bool = False,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            if cacheable:
                self.send_header("Cache-Control", "public, max-age=86400")
            self.end_headers()
            self.wfile.write(data)

        def _send_json(self, obj: Any, *, status: int = 200) -> None:
            data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self._send_bytes(data, "application/json; charset=utf-8", status=status)

        def _send_html(self, text: str, *, status: int = 200) -> None:
            self._send_bytes(text.encode("utf-8"), "text/html; charset=utf-8", status=status)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            qs = parse_qs(parsed.query)
            refresh = "refresh" in qs

            if path == "/":
                rows = "\n".join(
                    f'    <li><a href="/view/{state.index.name_by_key[k]}">'
                    f'{html_module.escape(k)}</a></li>'
                    for k in state.index.image_keys
                )
                self._send_html(_INDEX_HTML.format(count=len(state.index.image_keys), rows=rows))
                return

            if path.startswith("/view/"):
                name = path[len("/view/"):]
                image_key = state.index.key_by_name.get(name)
                if image_key is None:
                    self._send_html("样本不存在", status=404)
                    return
                title = html_module.escape(image_key, quote=True)
                body = _VIEWER_HTML.format(
                    title=title,
                    name=name,
                    name_json=json.dumps(name),
                )
                self._send_html(body)
                return

            if path == "/api/samples":
                self._send_json([
                    {"name": state.index.name_by_key[k], "image_key": k}
                    for k in state.index.image_keys
                ])
                return

            parts = path.strip("/").split("/")
            # /api/sample/{name}/panels | panel/{branch}/{idx} | stats | points/{branch}
            if len(parts) >= 3 and parts[0] == "api" and parts[1] == "sample":
                name = parts[2]
                image_key = state.index.key_by_name.get(name)
                if image_key is None:
                    self._send_json({"error": "not found"}, status=404)
                    return
                if len(parts) == 4 and parts[3] == "panels":
                    self._send_json(sample_panels_json(state.index, image_key))
                    return
                if (
                    len(parts) == 6
                    and parts[3] == "panel"
                    and parts[4] in ("raw", "sam2", "sam3")
                    and parts[5].isdigit()
                ):
                    branch = parts[4]
                    obj_idx = int(parts[5])
                    data = state.panel_bytes(name, branch, obj_idx, refresh=refresh)
                    if data is None:
                        self.send_error(HTTPStatus.NOT_FOUND)
                        return
                    self._send_bytes(data, "image/jpeg", cacheable=not refresh)
                    return
                if len(parts) == 4 and parts[3] == "stats":
                    self._send_json(sample_stats_json(state.index, image_key))
                    return
                if len(parts) == 5 and parts[3] == "points" and parts[4] in ("raw", "sam2", "sam3"):
                    branch = parts[4]
                    data = state.points_bytes(image_key, branch, refresh=refresh)
                    self._send_bytes(data, "application/json; charset=utf-8", cacheable=not refresh)
                    return

            self.send_error(HTTPStatus.NOT_FOUND)

    return Handler


def _lan_urls(port: int) -> List[str]:
    seen: set[str] = set()
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
    return [f"http://{ip}:{port}/" for ip in sorted(seen)]


def _print_listen_urls(host: str, port: int, *, sample_count: int) -> None:
    _log(f"Serving {sample_count} sample(s)")
    _log(f"  Local:   http://127.0.0.1:{port}/")
    _log(f"  Local:   http://localhost:{port}/")
    if host in ("0.0.0.0", "::"):
        lan = _lan_urls(port)
        if lan:
            for url in lan:
                _log(f"  Network: {url}")
        else:
            _log(f"  Network: http://<this-machine-ip>:{port}/")
    else:
        _log(f"  Bind:    http://{host}:{port}/")
    _log("Press Ctrl+C to stop.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind address (default 0.0.0.0 = all interfaces)",
    )
    parser.add_argument("--port", type=int, default=8848)
    parser.add_argument("--raw-run", default="refiner_exp/outputs/raw")
    parser.add_argument("--sam2-run", default="refiner_exp/outputs/sam2")
    parser.add_argument("--sam3-run", default="refiner_exp/outputs/sam3")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--max-panel-width", type=int, default=640)
    parser.add_argument(
        "--max-points-per-object",
        type=int,
        default=8000,
        help="Subsample each fusion .pcd when sending to the browser.",
    )
    parser.add_argument(
        "--cache-dir",
        default="refiner_exp/outputs/compare/cache",
        help="JPEG panel disk cache directory.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=48,
        help="Parallel workers for startup preload (default 48).",
    )
    parser.add_argument(
        "--memory-cache-mb",
        type=int,
        default=4096,
        help="In-memory cache budget for panels and point payloads (default 4096 MB).",
    )
    parser.add_argument(
        "--preload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pre-render panels and pre-pack point clouds at startup (default on).",
    )
    parser.add_argument(
        "--preload-points",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include point-cloud JSON in startup preload (default on).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index = build_compare_index(
        args.raw_run, args.sam2_run, args.sam3_run, max_images=args.max_images,
    )
    state = CompareServerState(
        index,
        cache_dir=Path(args.cache_dir),
        max_points_per_object=args.max_points_per_object,
        max_panel_width=args.max_panel_width,
        memory_cache_mb=args.memory_cache_mb,
    )
    if args.preload:
        _preload_assets(
            state,
            workers=max(1, args.workers),
            preload_panels=True,
            preload_points=args.preload_points,
        )
    handler = make_handler(state)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    server.daemon_threads = True
    server.request_queue_size = max(64, args.workers)
    _print_listen_urls(args.host, args.port, sample_count=len(index.image_keys))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _log("Shutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
