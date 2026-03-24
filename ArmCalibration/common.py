# -*- coding: utf-8 -*-
"""Shared helpers for the ArmCalibration project."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARM_ROOT = PROJECT_ROOT / "ArmCalibration"
ARM_DATA_ROOT = ARM_ROOT / "data"
DEFAULT_CAMERA_CONFIG_PATH = PROJECT_ROOT / "src" / "config" / "camera.json"


def load_json(path: Path | str) -> Any:
    with open(path, "r", encoding="utf-8") as inp:
        return json.load(inp)


def save_json(path: Path | str, payload: Any) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as out:
            json.dump(payload, out, indent=2, ensure_ascii=False)
        os.replace(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def sanitize_session_name(raw: str) -> str:
    candidate = raw.strip()
    if not candidate:
        return ""
    safe: list[str] = []
    for ch in candidate:
        if ch.isalnum() or ch in "-_":
            safe.append(ch)
        elif ch.isspace():
            safe.append("_")
        else:
            safe.append("_")
    return "".join(safe)


def auto_session_dir(root: Path, label: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    existing: list[int] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        try:
            existing.append(int(entry.name.split("_", 1)[0]))
        except (IndexError, ValueError):
            continue
    next_num = max(existing, default=0) + 1
    timestamp = datetime.now().strftime("%m%d%H%M")
    tag = sanitize_session_name(label) or "session"
    return root / f"{next_num:03d}_{tag}_{timestamp}"


def load_camera_config(path: Path | str = DEFAULT_CAMERA_CONFIG_PATH) -> dict:
    return load_json(Path(path))
