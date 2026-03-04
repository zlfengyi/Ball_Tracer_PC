# -*- coding: utf-8 -*-
"""
MVS SDK 环境初始化（路径、DLL 加载）。

在 import ball_grabber 之前自动完成：
1) MvImport 目录加入 sys.path
2) MvCameraControl.dll 目录加入 DLL 搜索路径
"""

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]  # ball_tracer_pc

# ── MvImport（Python 绑定）路径 ────────────────────────────
_MVIMPORT_CANDIDATES = [
    _ROOT / "SDK_Development" / "Samples" / "Python" / "MvImport",
]
_env_mvimport = os.environ.get("MVS_MVIMPORT_DIR", "").strip()
if _env_mvimport:
    _MVIMPORT_CANDIDATES.insert(0, Path(_env_mvimport))

for _p in _MVIMPORT_CANDIDATES:
    if _p.exists():
        s = str(_p.resolve())
        if s not in sys.path:
            sys.path.insert(0, s)
        break
else:
    raise ImportError(
        "找不到 MVS MvImport 目录。请确认 SDK_Development/Samples/Python/MvImport 存在，"
        "或通过环境变量 MVS_MVIMPORT_DIR 指定。"
    )

# ── MvCameraControl.dll 搜索路径 ──────────────────────────
_DLL_CANDIDATES = [
    Path(r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64"),
    Path(r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win32_i86"),
]
_env_dll = os.environ.get("MVS_DLL_DIR", "").strip()
if _env_dll:
    _DLL_CANDIDATES.insert(0, Path(_env_dll))

_dll_handles = []  # 保活 os.add_dll_directory 句柄

for _dp in _DLL_CANDIDATES:
    if _dp.exists():
        if hasattr(os, "add_dll_directory"):
            try:
                _dll_handles.append(os.add_dll_directory(str(_dp)))
            except OSError:
                pass
        os.environ["PATH"] = str(_dp) + os.pathsep + os.environ.get("PATH", "")
        break
