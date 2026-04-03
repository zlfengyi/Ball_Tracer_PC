# -*- coding: utf-8 -*-
"""Shared helpers for the four-camera calibration workflow."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
FOUR_CAMERA_CALIB_ROOT = DATA_ROOT / "four_camera_calibration"
CAMERA_CONFIG_PATH = PROJECT_ROOT / "src" / "config" / "camera.json"
DEFAULT_INTRINSICS_CONFIG_PATH = PROJECT_ROOT / "src" / "config" / "four_camera_intrinsics.json"

DEFAULT_BOARD_COLS = 9
DEFAULT_BOARD_ROWS = 12
DEFAULT_INNER_COLS = DEFAULT_BOARD_COLS - 1
DEFAULT_INNER_ROWS = DEFAULT_BOARD_ROWS - 1
DEFAULT_SQUARE_SIZE_MM = 45.0


def load_camera_config(config_path: Path = CAMERA_CONFIG_PATH) -> dict:
    with open(config_path, encoding="utf-8") as inp:
        return json.load(inp)


def load_sync_serials(config_path: Path = CAMERA_CONFIG_PATH, overrides: dict | None = None) -> list[str]:
    cfg = load_camera_config(config_path)
    if overrides:
        cfg.update(overrides)
    master = cfg.get("master_serial")
    slaves = list(cfg.get("slave_serials", []))
    serials: list[str] = []
    if master and not cfg.get("master_min_bandwidth"):
        serials.append(master)
    serials.extend(slaves)
    return serials


def sanitize_session_name(raw: str) -> str:
    candidate = raw.strip()
    if not candidate:
        return ""
    safe = []
    for ch in candidate:
        if ch.isalnum() or ch in "-_":
            safe.append(ch)
        elif ch.isspace():
            safe.append("_")
        else:
            safe.append("_")
    return "".join(safe)


def auto_session_dir(root: Path, content: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    existing = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        try:
            existing.append(int(entry.name.split("_")[0]))
        except (ValueError, IndexError):
            pass
    next_num = max(existing, default=0) + 1
    timestamp = datetime.now().strftime("%m%d%H%M")
    content_tag = sanitize_session_name(content) or "run"
    return root / f"{next_num:03d}_{content_tag}_{timestamp}"


def resolve_session_dir(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    direct = PROJECT_ROOT / path
    if direct.exists():
        return direct
    return FOUR_CAMERA_CALIB_ROOT / path


def latest_session_dir(root: Path = FOUR_CAMERA_CALIB_ROOT) -> Path:
    sessions = [entry for entry in root.iterdir() if entry.is_dir()]
    if not sessions:
        raise FileNotFoundError(f"No calibration sessions found under {root}")
    return max(sessions, key=lambda entry: entry.stat().st_mtime)


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)
