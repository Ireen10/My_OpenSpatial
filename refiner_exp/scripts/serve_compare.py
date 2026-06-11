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
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
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
    render_combined_image,
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
    .overlay-wrap {{ padding: 12px 16px; background: #111; text-align: center; }}
    .overlay-wrap img {{ max-width: 100%; border: 1px solid #333; border-radius: 4px; cursor: zoom-in; }}
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
    @media (max-width: 960px) {{ .viewers {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <p>拖拽旋转 · 滚轮缩放 · 右键平移 · <span style="color:#ccc">点击图片放大</span></p>
    <p><a href="/">← 样本列表</a> · <a href="/api/sample/{name}/overlay?refresh=1">刷新 2D 图</a></p>
  </header>
  <div class="overlay-wrap">
    <img src="/api/sample/{name}/overlay" alt="2D overlay" id="overlay-img" title="点击放大">
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
    const overlayImg = document.getElementById("overlay-img");
    const lightbox = document.getElementById("lightbox");
    const lightboxImg = document.getElementById("lightbox-img");
    overlayImg.addEventListener("click", () => {{
      lightboxImg.src = overlayImg.src;
      lightbox.classList.remove("hidden");
    }});
    lightbox.addEventListener("click", () => lightbox.classList.add("hidden"));
    document.addEventListener("keydown", (e) => {{
      if (e.key === "Escape") lightbox.classList.add("hidden");
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
      const renderer = new THREE.WebGLRenderer({{ canvas, antialias: true }});
      renderer.setPixelRatio(window.devicePixelRatio || 1);
      renderer.setSize(w, h, false);
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x141414);
      const camera = new THREE.PerspectiveCamera(50, w / h, 0.001, 500);
      const controls = new THREE.OrbitControls(camera, canvas);
      controls.enableDamping = true;
      controls.dampingFactor = 0.08;
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
      (function animate() {{
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
      }})();
    }}

    async function loadBranch(branch, canvasId, metaId) {{
      const res = await fetch("/api/sample/" + SAMPLE + "/points/" + branch);
      const packed = await res.json();
      mountViewer(canvasId, metaId, packed);
    }}

    Promise.all([
      fetch("/api/sample/" + SAMPLE + "/stats").then(r => r.json()),
      loadBranch("raw", "view-raw", "meta-raw"),
      loadBranch("sam2", "view-sam2", "meta-sam2"),
      loadBranch("sam3", "view-sam3", "meta-sam3"),
    ]).then(([stats]) => {{
      document.getElementById("stats").textContent = JSON.stringify(stats, null, 2);
    }}).catch(err => {{
      document.getElementById("stats").textContent = "加载失败: " + err;
    }});
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
    ):
        self.index = index
        self.cache_dir = cache_dir
        self.max_points_per_object = max_points_per_object
        self.max_panel_width = max_panel_width
        self._overlay_lock = threading.Lock()
        self.cache_dir.mkdir(parents=True, exist_ok=True)


def make_handler(state: CompareServerState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            _log(f"[{self.address_string()}] {fmt % args}")

        def _send_bytes(self, data: bytes, content_type: str, *, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_json(self, obj: Any, *, status: int = 200) -> None:
            data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self._send_bytes(data, "application/json; charset=utf-8", status=status)

        def _send_html(self, text: str, *, status: int = 200) -> None:
            self._send_bytes(text.encode("utf-8"), "text/html; charset=utf-8", status=status)

        def _overlay_bytes(self, name: str, *, refresh: bool) -> Optional[bytes]:
            image_key = state.index.key_by_name.get(name)
            if image_key is None:
                return None
            cache_path = state.cache_dir / f"{name}.jpg"
            if not refresh and cache_path.is_file():
                return cache_path.read_bytes()
            with state._overlay_lock:
                if not refresh and cache_path.is_file():
                    return cache_path.read_bytes()
                combined = render_combined_image(
                    state.index, image_key, max_panel_width=state.max_panel_width,
                )
                if combined is None:
                    return None
                buf = io.BytesIO()
                combined.save(buf, format="JPEG", quality=92)
                data = buf.getvalue()
                cache_path.write_bytes(data)
            return data

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
            # /api/sample/{name}/overlay | stats | points/{branch}
            if len(parts) >= 3 and parts[0] == "api" and parts[1] == "sample":
                name = parts[2]
                image_key = state.index.key_by_name.get(name)
                if image_key is None:
                    self._send_json({"error": "not found"}, status=404)
                    return
                if len(parts) == 4 and parts[3] == "overlay":
                    data = self._overlay_bytes(name, refresh=refresh)
                    if data is None:
                        self.send_error(HTTPStatus.NOT_FOUND)
                        return
                    self._send_bytes(data, "image/jpeg")
                    return
                if len(parts) == 4 and parts[3] == "stats":
                    self._send_json(sample_stats_json(state.index, image_key))
                    return
                if len(parts) == 5 and parts[3] == "points" and parts[4] in ("raw", "sam2", "sam3"):
                    branch = parts[4]
                    branch_records = branch_records_for_image(state.index, image_key)
                    packed = _pack_branch_from_fusion(
                        branch_records[branch],
                        state.index.fusion_by_key[branch],
                        state.max_points_per_object,
                    )
                    self._send_json(packed)
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
    parser.add_argument("--max-images", type=int, default=20)
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
        help="JPEG overlay cache directory.",
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
    )
    handler = make_handler(state)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    _print_listen_urls(args.host, args.port, sample_count=len(index.image_keys))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _log("Shutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
