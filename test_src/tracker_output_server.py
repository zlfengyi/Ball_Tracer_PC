# -*- coding: utf-8 -*-
"""tracker_output 静态报告服务器：列出 *.html（新→旧），手机点开即看。

用法：python test_src/tracker_output_server.py [--host 0.0.0.0] [--port 8123]
只读静态服务，无 ROS 依赖；报告 HTML 里的 plotly 来自 CDN，手机需能上外网。
"""

from __future__ import annotations

import argparse
import html
import re
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent.parent / "tracker_output"

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".mp4": "video/mp4",
    ".avi": "video/x-msvideo",
}

PAGE = """<!doctype html><html lang="zh-CN"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tracker Reports</title>
<style>
body{{font-family:-apple-system,"Segoe UI",Roboto,sans-serif;background:#1a1a2e;color:#e0e0e0;margin:0}}
h1{{font-size:18px;color:#e94560;padding:16px 16px 4px}}
p{{color:#a0a0c0;font-size:12px;padding:0 16px;margin:4px 0 12px}}
a.f{{display:block;margin:8px 12px;padding:14px;background:#16213e;border:1px solid #0f3460;
     border-radius:10px;color:#e0e0e0;text-decoration:none;font-size:15px}}
a.f:active{{background:#0f3460}}
a.f small{{display:block;color:#a0a0c0;font-size:12px;margin-top:4px}}
</style></head><body>
<h1>Tracker Reports</h1>
<p>{count} 份报告，新→旧。生成于 {now}，刷新页面更新列表。</p>
{items}
</body></html>"""


def _index() -> str:
    # 布局：每 session 一个子目录 tracker_*/tracker_*.html；根目录平铺为历史遗留
    candidates = list(ROOT.glob("tracker_*.html")) + list(ROOT.glob("tracker_*/tracker_*.html"))
    files = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    items = []
    for p in files:
        st = p.stat()
        mtime = datetime.fromtimestamp(st.st_mtime).strftime("%m-%d %H:%M")
        size_mb = st.st_size / 1e6
        rel = p.relative_to(ROOT).as_posix()
        items.append(
            f'<a class="f" href="/{html.escape(rel)}">{html.escape(p.stem)}'
            f"<small>{mtime} · {size_mb:.1f} MB</small></a>"
        )
    return PAGE.format(count=len(files), now=datetime.now().strftime("%H:%M:%S"),
                       items="\n".join(items) or '<p style="padding:0 16px">（空）</p>')


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", _index().encode("utf-8"))
            return
        name = path.lstrip("/")
        # 只允许 tracker_output 下最多一层子目录的文件名，拒绝路径穿越
        if not re.fullmatch(r"[\w.\-]+(?:/[\w.\-]+)?", name) or ".." in name:
            self._send(404, "text/plain; charset=utf-8", b"not found")
            return
        target = ROOT / name
        if not target.is_file() or target.suffix.lower() not in CONTENT_TYPES:
            self._send(404, "text/plain; charset=utf-8", b"not found")
            return
        self._send(200, CONTENT_TYPES[target.suffix.lower()], target.read_bytes())

    def log_message(self, fmt: str, *args) -> None:
        return

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8123)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[tracker_output_server] serving {ROOT} at http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
