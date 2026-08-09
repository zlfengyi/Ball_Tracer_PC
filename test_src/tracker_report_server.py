# -*- coding: utf-8 -*-
"""Tracker runs 浏览服务器：索引 + tracker 交互视图 + 产物下载。

曾经的 pc_logger 臂报告页（/run/，POE FK + time_sync 表）随 newarm2 线一起
废弃删除（2026-07-16）——臂数据现走 rosbag → extract_arm_bag.py → generate_curve3_html
的 Arm tab（RK 单调钟 + RK→PC 仿射时间映射）。本服务器只负责浏览与分发产物。
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import quote, unquote, urlparse


_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from test_src.generate_curve3_html import HTML_TEMPLATE, _merge_racket_json


# ball_tracer_pc 2026-08-09 起不再是 tennis-man 的 submodule，产物只在本克隆下
DEFAULT_TRACKER_OUTPUT_DIRS: tuple[Path, ...] = (
    Path("D:/tennis-man/data"),
    Path("D:/Ball_Tracer_PC/tracker_output"),
)
_RUN_RE = re.compile(r"^(tracker_\d{8}_\d{6})(.*)$")
_RUN_DIR_RE = re.compile(r"^tracker_\d{8}_\d{6}$")


@dataclass
class RunArtifacts:
    stem: str
    tracker_json: Path | None = None
    tracker_html: Path | None = None
    tracker_video: Path | None = None
    curve4_json: Path | None = None
    extra_jsons: list[Path] = field(default_factory=list)
    extra_htmls: list[Path] = field(default_factory=list)
    extra_videos: list[Path] = field(default_factory=list)
    all_files: list[Path] = field(default_factory=list)

    def latest_mtime_ns(self) -> int:
        latest = 0
        for path in self.all_files:
            try:
                latest = max(latest, int(path.stat().st_mtime_ns))
            except FileNotFoundError:
                continue
        return latest

    def related_files(self) -> list[Path]:
        paths: list[Path] = []
        for item in (
            self.tracker_json,
            self.tracker_html,
            self.tracker_video,
            self.curve4_json,
        ):
            if item is not None:
                paths.append(item)
        paths.extend(sorted(self.extra_jsons))
        paths.extend(sorted(self.extra_htmls))
        paths.extend(sorted(self.extra_videos))
        return paths


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_tracker_view_payload(payload: dict | None, *, car_source_payload: dict | None = None) -> dict | None:
    if not isinstance(payload, dict):
        return None

    config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    car_source = car_source_payload if isinstance(car_source_payload, dict) else payload
    car_config = car_source.get("config") if isinstance(car_source.get("config"), dict) else {}
    car_summary = car_source.get("summary") if isinstance(car_source.get("summary"), dict) else {}

    slim_frames: list[dict[str, Any]] = []
    for frame in payload.get("frames") or []:
        if not isinstance(frame, dict):
            continue
        slim_frame: dict[str, Any] = {}
        for key in (
            "idx",
            "exposure_pc",
            "elapsed_s",
            "video_frame_idx",
            "video_mapping_exact",
        ):
            if key in frame:
                slim_frame[key] = frame[key]
        for key in ("ball3d", "racket3d"):
            value = frame.get(key)
            if isinstance(value, dict):
                slim_frame[key] = value
        if slim_frame:
            slim_frames.append(slim_frame)

    slim_config: dict[str, Any] = {}
    for key in (
        "first_frame_exposure_pc",
        "fps",
        "duration_s",
        "distance_unit",
        "ideal_hit_z",
        "cor",
        "noise_mm",
        "min_stage1_points",
        "video_frame_mapping_exact",
        "replay_source",
    ):
        if key in config:
            slim_config[key] = config[key]
    car_localizer_cfg = car_config.get("car_localizer")
    if isinstance(car_localizer_cfg, dict):
        slim_config["car_localizer"] = {
            key: car_localizer_cfg[key]
            for key in ("sample_every_frames", "result_source", "tracker_sample_every_frames")
            if key in car_localizer_cfg
        }

    slim_summary: dict[str, Any] = {}
    for key in (
        "total_frames",
        "actual_fps",
        "end_to_end_fps",
        "processing_duration_s",
        "end_to_end_duration_s",
        "observations_3d",
        "predictions",
        "state_transitions",
        "reset_times",
        "latency_ms_avg",
        "car_locs",
        "racket_observations_3d",
        "video_frames_mapped",
        "video_frame_mapping_exact",
    ):
        if key in summary:
            slim_summary[key] = summary[key]
    for key in ("car_locs",):
        if key not in slim_summary and key in car_summary:
            slim_summary[key] = car_summary[key]

    return {
        "config": slim_config,
        "summary": slim_summary,
        "observations": payload.get("observations") or [],
        "predictions": payload.get("predictions") or [],
        "car_locs": car_source.get("car_locs") or [],
        "racket_observations": payload.get("racket_observations") or [],
        "frames": slim_frames,
        "video_frame_indices": payload.get("video_frame_indices") or [],
        "state_transitions": payload.get("state_transitions") or [],
    }


def _safe_child(root: Path, name: str) -> Path | None:
    """Resolve `name` (may contain '/') under `root`, rejecting escapes."""
    candidate = (root / name).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def _assign_file_to_record(record: RunArtifacts, path: Path, suffix: str) -> None:
    record.all_files.append(path)
    if suffix == ".json":
        record.tracker_json = path
    elif suffix == ".html":
        record.tracker_html = path
    elif suffix == ".avi":
        record.tracker_video = path
    elif suffix == "_curve4.json":
        record.curve4_json = path
    elif path.suffix.lower() == ".json":
        record.extra_jsons.append(path)
    elif path.suffix.lower() == ".html":
        record.extra_htmls.append(path)
    elif path.suffix.lower() in {".avi", ".mp4"}:
        record.extra_videos.append(path)


def _scan_runs(roots: Iterable[Path]) -> list[RunArtifacts]:
    """Scan one or more `roots` for tracker runs.

    Within each root two layouts are supported simultaneously:
      A. Flat: all `tracker_<stem><suffix>` files live directly under the root.
      B. Nested: each run has its own subdir `<root>/tracker_<stem>/` that
         contains all its artifacts.
    Runs that share the same stem across roots are merged into one record.
    """
    by_stem: dict[str, RunArtifacts] = {}

    for root in roots:
        if not root.is_dir():
            continue
        for path in root.iterdir():
            if path.is_file():
                match = _RUN_RE.match(path.name)
                if match is None:
                    continue
                stem = match.group(1)
                suffix = match.group(2)
                record = by_stem.setdefault(stem, RunArtifacts(stem=stem))
                _assign_file_to_record(record, path, suffix)
            elif path.is_dir():
                if not _RUN_DIR_RE.match(path.name):
                    continue
                stem = path.name
                record = by_stem.setdefault(stem, RunArtifacts(stem=stem))
                for child in path.iterdir():
                    if not child.is_file():
                        continue
                    match = _RUN_RE.match(child.name)
                    if match is None or match.group(1) != stem:
                        continue
                    _assign_file_to_record(record, child, match.group(2))

    runs = [run for run in by_stem.values() if run.tracker_json is not None or run.tracker_html is not None]
    return sorted(runs, key=lambda item: item.latest_mtime_ns(), reverse=True)


class TrackerReportServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        *,
        tracker_output_dirs: Iterable[Path],
    ) -> None:
        super().__init__(server_address, handler_class)
        self.tracker_output_dirs: tuple[Path, ...] = tuple(
            Path(p).resolve() for p in tracker_output_dirs
        )
        if not self.tracker_output_dirs:
            raise ValueError("tracker_output_dirs must contain at least one path")
        self._cache_lock = Lock()
        self._json_cache: dict[tuple[str, int, int], dict] = {}

    def file_signature(self, path: Path) -> tuple[str, int, int]:
        stat = path.stat()
        return (str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))

    def load_json(self, path: Path) -> dict:
        signature = self.file_signature(path)
        with self._cache_lock:
            cached = self._json_cache.get(signature)
        if cached is not None:
            return cached
        payload = _read_json(path)
        with self._cache_lock:
            self._json_cache = {signature: payload}
        return payload

    def list_runs(self) -> list[RunArtifacts]:
        return _scan_runs(self.tracker_output_dirs)

    def get_run(self, stem: str) -> RunArtifacts | None:
        for run in self.list_runs():
            if run.stem == stem:
                return run
        return None

    def choose_tracker_payload(
        self,
        run: RunArtifacts,
        *,
        source: str = "default",
    ) -> tuple[dict | None, str, list[Path]]:
        if source == "curve4" and run.curve4_json is not None and run.curve4_json.exists():
            return self.load_json(run.curve4_json), run.curve4_json.name, [run.curve4_json]

        if run.tracker_json is None and run.extra_jsons:
            preferred = sorted(run.extra_jsons, key=lambda item: item.stat().st_mtime_ns, reverse=True)[0]
            return self.load_json(preferred), preferred.name, [preferred]

        base_payload = self.load_json(run.tracker_json) if run.tracker_json is not None else None

        exact_with_racket = None
        exact_racket = None
        variant_with_racket: list[Path] = []
        variant_racket: list[Path] = []
        for path in run.extra_jsons:
            if path.name == f"{run.stem}_with_racket.json":
                exact_with_racket = path
            elif path.name.endswith("_with_racket.json"):
                variant_with_racket.append(path)
            elif path.name == f"{run.stem}_racket.json":
                exact_racket = path
            elif path.name.endswith("_racket.json") and not path.name.endswith("_with_racket.json"):
                variant_racket.append(path)

        with_candidates = [
            item
            for item in [
                exact_with_racket,
                *sorted(variant_with_racket, key=lambda item: item.stat().st_mtime_ns, reverse=True),
            ]
            if item is not None
        ]
        if with_candidates:
            selected = with_candidates[0]
            return self.load_json(selected), selected.name, [selected]

        racket_candidates = [
            item
            for item in [
                exact_racket,
                *sorted(variant_racket, key=lambda item: item.stat().st_mtime_ns, reverse=True),
            ]
            if item is not None
        ]
        if base_payload is not None and racket_candidates:
            selected = racket_candidates[0]
            merged = _merge_racket_json(base_payload, self.load_json(selected), str(selected))
            return merged, f"{run.tracker_json.name} + {selected.name}", [run.tracker_json, selected]

        if base_payload is not None and run.tracker_json is not None:
            return base_payload, run.tracker_json.name, [run.tracker_json]
        return None, "no tracker json", []


class TrackerReportHandler(BaseHTTPRequestHandler):
    server: TrackerReportServer

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path.rstrip("/") or "/"
        if route == "/":
            self._serve_index()
            return
        if route.startswith("/tracker-view/"):
            from urllib.parse import parse_qs
            query = parse_qs(parsed.query or "")
            source_values = query.get("source", [])
            source = source_values[0] if source_values else "default"
            self._serve_tracker_view(
                unquote(route.removeprefix("/tracker-view/")),
                source=source,
            )
            return
        if route.startswith("/artifact/"):
            self._serve_artifact(unquote(route.removeprefix("/artifact/")))
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Unknown route")

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_html(self, body: str, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_file(self, path: Path, *, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(path.stat().st_size))
        self.end_headers()
        with path.open("rb") as f:
            while True:
                chunk = f.read(1024 * 128)
                if not chunk:
                    break
                self.wfile.write(chunk)

    def _artifact_url(self, path: Path) -> str:
        path_resolved = path.resolve()
        for root in self.server.tracker_output_dirs:
            try:
                rel = path_resolved.relative_to(root)
            except ValueError:
                continue
            parts = [quote(part) for part in rel.parts]
            return "/artifact/" + "/".join(parts)
        return f"/artifact/{quote(path.name)}"

    def _run_rows_html(self, runs: list[RunArtifacts]) -> str:
        rows: list[str] = []
        for run in runs:
            files = run.related_files()
            file_links = " ".join(
                f'<a class="tag" href="{self._artifact_url(path)}">{html.escape(path.name)}</a>'
                for path in files[:8]
            )
            if len(files) > 8:
                file_links += f' <span class="muted">+{len(files) - 8} more</span>'
            summary_bits = []
            if run.curve4_json is not None:
                summary_bits.append("curve4")
            if any(path.name.endswith("_with_racket.json") for path in run.extra_jsons):
                summary_bits.append("with_racket")
            elif any(path.name.endswith("_racket.json") for path in run.extra_jsons):
                summary_bits.append("racket")
            if any(path.name.endswith("_arm.json") for path in run.extra_jsons):
                summary_bits.append("arm")
            if any(path.name.endswith("_rk_tracking.json") for path in run.extra_jsons):
                summary_bits.append("rk")
            if run.tracker_video is not None or run.extra_videos:
                summary_bits.append("video")
            badges = (
                " ".join(f'<span class="pill">{html.escape(bit)}</span>' for bit in summary_bits)
                or '<span class="muted">base tracker</span>'
            )
            rows.append(
                "<tr>"
                f'<td><a href="/tracker-view/{quote(run.stem)}"><strong>{html.escape(run.stem)}</strong></a><div class="row-meta">{badges}</div></td>'
                f"<td>{file_links}</td>"
                f"<td>{len(files)}</td>"
                "</tr>"
            )
        if not rows:
            return '<tr><td colspan="3">No tracker runs found.</td></tr>'
        return "".join(rows)

    def _serve_index(self) -> None:
        runs = self.server.list_runs()
        body = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Tracker Report Server</title>
  <style>
    :root {{
      --bg: #f3efe7;
      --paper: rgba(255,255,255,0.88);
      --ink: #1a2233;
      --muted: #61708a;
      --line: #d9d0c2;
      --accent: #bd4f2b;
      --accent-2: #1d6f8a;
      --pill: #f6dfcf;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: "IBM Plex Sans", "Noto Sans SC", "Segoe UI", sans-serif; background:
      radial-gradient(circle at top left, rgba(29,111,138,0.14), transparent 28%),
      radial-gradient(circle at top right, rgba(189,79,43,0.18), transparent 24%),
      linear-gradient(180deg, #f8f3ea 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    .page {{ max-width: 1440px; margin: 0 auto; padding: 28px 22px 48px; }}
    .hero {{ background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(255,247,240,0.92)); border: 1px solid rgba(217,208,194,0.95); border-radius: 26px; padding: 24px 26px; box-shadow: 0 20px 60px rgba(26,34,51,0.08); margin-bottom: 18px; }}
    .hero h1 {{ margin: 0 0 10px; font-size: 30px; }}
    .muted {{ color: var(--muted); }}
    .card {{ background: var(--paper); border: 1px solid rgba(217,208,194,0.95); border-radius: 22px; padding: 20px 22px; box-shadow: 0 16px 42px rgba(26,34,51,0.08); }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; padding: 14px 10px; border-top: 1px solid rgba(217,208,194,0.95); vertical-align: top; }}
    th {{ color: var(--muted); font-weight: 600; }}
    a {{ color: var(--accent-2); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .tag {{ display: inline-block; background: rgba(255,255,255,0.7); border: 1px solid rgba(217,208,194,0.95); border-radius: 999px; padding: 4px 10px; margin: 0 6px 6px 0; font-size: 12px; }}
    .pill {{ display: inline-block; border-radius: 999px; background: var(--pill); color: #8f3d1f; padding: 3px 10px; margin-right: 6px; font-size: 12px; }}
    .row-meta {{ margin-top: 8px; }}
    code {{ background: rgba(255,255,255,0.78); border: 1px solid rgba(217,208,194,0.95); border-radius: 8px; padding: 1px 6px; }}
  </style>
</head>
<body>
  <div class="page">
    <div class="hero">
      <h1>Tracker Report Server</h1>
      <p class="muted">选择一个 tracker run 打开交互视图；页面会自动聚合同名的 base tracker JSON/HTML 与标注生成的 <code>*_racket.json</code> / <code>*_with_racket.json</code> / <code>*_arm.json</code> / <code>*_rk_tracking.json</code> 等伴随文件。</p>
      <p class="muted">tracker_output: {" ".join(f"<code>{html.escape(str(p))}</code>" for p in self.server.tracker_output_dirs)}</p>
    </div>
    <div class="card">
      <table>
        <thead><tr><th>Run</th><th>Artifacts</th><th>Count</th></tr></thead>
        <tbody>{self._run_rows_html(runs)}</tbody>
      </table>
    </div>
  </div>
</body>
</html>"""
        self._send_html(body)

    def _serve_tracker_view(self, stem: str, *, source: str = "default") -> None:
        run = self.server.get_run(stem)
        if run is None:
            self.send_error(HTTPStatus.NOT_FOUND, "Tracker run not found")
            return
        payload, _, _ = self.server.choose_tracker_payload(run, source=source)
        if payload is not None:
            base_tracker_payload = (
                self.server.load_json(run.tracker_json)
                if run.tracker_json is not None and run.tracker_json.exists()
                else None
            )
            view_payload = _build_tracker_view_payload(payload, car_source_payload=base_tracker_payload)
            body = HTML_TEMPLATE.replace(
                "%%DATA_JSON%%",
                json.dumps(view_payload, ensure_ascii=False),
            )
            self._send_html(body)
            return
        if run.tracker_html is not None and run.tracker_html.exists():
            self._send_file(run.tracker_html, content_type="text/html; charset=utf-8")
            return
        self.send_error(HTTPStatus.NOT_FOUND, "No tracker html/json source found")

    def _serve_artifact(self, name: str) -> None:
        path: Path | None = None
        for root in self.server.tracker_output_dirs:
            candidate = _safe_child(root, name)
            if candidate is not None and candidate.exists():
                path = candidate
                break
        if path is None:
            self.send_error(HTTPStatus.NOT_FOUND, "Artifact not found")
            return
        suffix = path.suffix.lower()
        if suffix == ".json":
            self._send_file(path, content_type="application/json; charset=utf-8")
            return
        if suffix == ".html":
            self._send_file(path, content_type="text/html; charset=utf-8")
            return
        if suffix == ".avi":
            self._send_file(path, content_type="video/x-msvideo")
            return
        if suffix == ".mp4":
            self._send_file(path, content_type="video/mp4")
            return
        self._send_file(path, content_type="application/octet-stream")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve tracker run reports from tracker_output.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--tracker-output-dir",
        type=Path,
        action="append",
        dest="tracker_output_dirs",
        default=None,
        help=(
            "Directory containing tracker_<stamp>.* artifacts (flat) or "
            "tracker_<stamp>/ subdirs (nested). May be passed multiple times. "
            f"Default: {', '.join(str(p) for p in DEFAULT_TRACKER_OUTPUT_DIRS)}"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    output_dirs = tuple(args.tracker_output_dirs) if args.tracker_output_dirs else DEFAULT_TRACKER_OUTPUT_DIRS
    server = TrackerReportServer(
        (str(args.host), int(args.port)),
        TrackerReportHandler,
        tracker_output_dirs=output_dirs,
    )
    print(f"Tracker report server listening on http://{args.host}:{args.port}")
    for p in server.tracker_output_dirs:
        marker = "" if p.exists() else "  (missing)"
        print(f"tracker_output: {p}{marker}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
