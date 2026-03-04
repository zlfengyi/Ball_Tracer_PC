# -*- coding: utf-8 -*-

"""MVS ctypes 绑定加载与 DLL 搜索路径处理。

该模块的职责仅限于：
1) 把 MVS 官方 Python 示例的 `MvImport/` 放进 `sys.path`（让其可被 import）；
2) 在 Windows 上把 DLL 目录加入搜索路径（让 `MvCameraControl.dll` 可被加载）；
3) 把官方示例里分散的符号收拢成一个 `MvsBinding`，便于其它模块注入依赖。

设计要点：
- 延迟 import MvImport（因为 import 时会立即尝试加载 MvCameraControl.dll）。
- 在无 DLL 的环境下也能 import 本包，并输出清晰错误信息。
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


class MvsDllNotFoundError(RuntimeError):
    pass


# Windows 下 `os.add_dll_directory()` 返回的句柄需要保活；
# 否则对象被 GC 回收后，目录会自动从 DLL 搜索路径中移除。
_DLL_DIR_HANDLES: list[Any] = []


def _ensure_dll_dir(dll_dir: Path) -> None:
    """把 DLL 目录加入搜索路径。

    Windows：优先用 os.add_dll_directory（Python 3.8+），并同步追加到 PATH。
    """

    if not dll_dir.exists():
        return

    try:
        if hasattr(os, "add_dll_directory"):
            handle = os.add_dll_directory(str(dll_dir))
            # 句柄保活，避免目录被悄悄移除。
            _DLL_DIR_HANDLES.append(handle)
    except OSError:
        # 某些环境可能不允许添加目录；兜底走 PATH。
        pass

    os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")


def _ensure_mvimport_on_syspath(mvimport_dir: Path) -> None:
    # MvImport 目录内部使用同目录 import（不是标准包结构），必须把目录加到 sys.path。
    if str(mvimport_dir) not in sys.path:
        sys.path.insert(0, str(mvimport_dir))


def _resolve_mvimport_dir(*, mvimport_dir: Optional[str]) -> Path:
    """解析 MvImport 目录位置。
    """

    candidates: list[Path] = []

    if mvimport_dir is not None and str(mvimport_dir).strip():
        candidates.append(Path(str(mvimport_dir)).expanduser())

    env_dir = os.environ.get("MVS_MVIMPORT_DIR")
    if env_dir and str(env_dir).strip():
        candidates.append(Path(str(env_dir)).expanduser())

    # best-effort：根据仓库内 examples/scratch/mvs_sdk_init.py 给出的安装路径形态
    # （不同 SDK 版本目录名可能存在 Samples/Sample、Python/python 的差异）。
    pf86 = os.environ.get("ProgramFiles(x86)")
    if pf86 and str(pf86).strip():
        base = Path(pf86) / "MVS" / "Development"
        candidates.extend(
            [
                base / "Sample" / "python" / "MvImport",
                base / "Samples" / "Python" / "MvImport",
            ]
        )

    for c in candidates:
        try:
            p = c.resolve()
        except Exception:
            p = c
        if p.exists() and p.is_dir():
            return p

    tried = "\n".join(f"- {str(x)}" for x in candidates) if candidates else "(none)"
    raise MvsDllNotFoundError(
        "找不到 MVS 官方 Python 示例绑定目录（MvImport）。\n"
        "\n"
        "你需要提供 MvImport 目录（包含 MvCameraControl_class.py 等文件）。\n"
        "支持方式（按优先级）：\n"
        "1) 传参：load_mvs_binding(mvimport_dir=...) / CLI 参数 --mvimport-dir\n"
        "2) 环境变量：MVS_MVIMPORT_DIR=<MvImport目录>\n"
        "3) 将 MvImport 目录加入 sys.path（不推荐，容易污染环境）\n"
        "\n"
        f"已尝试候选目录：\n{tried}"
    )


@dataclass(frozen=True, slots=True)
class MvsBinding:
    """已加载的 MVS 绑定符号集合。"""

    MvCamera: Any
    params: Any
    err: Any

    @property
    def MV_OK(self) -> int:
        return int(self.err.MV_OK)

    @property
    def MV_GIGE_DEVICE(self) -> int:
        return int(self.params.MV_GIGE_DEVICE)

    @property
    def MV_USB_DEVICE(self) -> int:
        return int(self.params.MV_USB_DEVICE)


def load_mvs_binding(*, mvimport_dir: Optional[str] = None, dll_dir: Optional[str] = None) -> MvsBinding:
    """加载 MVS Python 示例绑定。

    Args:
        dll_dir: 包含 MvCameraControl.dll 的目录（可选）。

    Returns:
        MvsBinding: MVS 绑定对象。

    Raises:
        MvsDllNotFoundError: 找不到 MvCameraControl.dll 或其依赖。
    """

    mvimport_path = _resolve_mvimport_dir(mvimport_dir=mvimport_dir)
    _ensure_mvimport_on_syspath(mvimport_path)

    # 1) 用户指定目录
    if dll_dir:
        _ensure_dll_dir(Path(dll_dir))

    # 2) 环境变量
    env_dir = os.environ.get("MVS_DLL_DIR")
    if env_dir:
        _ensure_dll_dir(Path(env_dir))

    try:
        mv = importlib.import_module("MvCameraControl_class")
        params = importlib.import_module("CameraParams_header")
        err = importlib.import_module("MvErrorDefine_const")
    except (FileNotFoundError, ImportError, OSError) as exc:
        raise MvsDllNotFoundError(
            "MVS DLL not found: MvCameraControl.dll (or dependency).\n"
            "找不到 MvCameraControl.dll（或其依赖）。\n"
            "\n"
            "解决方法：\n"
            "1) 安装海康 MVS（Machine Vision Software），确保系统 PATH 可找到 MvCameraControl.dll；\n"
            "2) 或使用参数 dll_dir / 环境变量 MVS_DLL_DIR 指向 DLL 目录；\n"
            "3) 提示：MvCameraControl.dll 通常位于 MVS 安装目录的 Runtime/Bin 之类路径。"
        ) from exc

    return MvsBinding(MvCamera=mv.MvCamera, params=params, err=err)
