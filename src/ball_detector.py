# -*- coding: utf-8 -*-
"""
网球检测模块 — YOLO 推理，返回网球像素坐标。

推理后端（按性能排序）：
  1. TensorRT FP16 engine (*.engine) — 2 张图 batch 推理 ~6ms (RTX 5080)
  2. PyTorch FP16 (*.pt, half=True)  — 2 张图 batch 推理 ~9ms
  3. ONNX Runtime GPU (*.onnx)       — 2 张图 batch 推理 ~9ms
  4. PyTorch FP32 CPU (*.pt)         — 2 张图逐张推理 ~70ms

推荐用法（TensorRT 最快）::

    detector = BallDetector("yolo_model/model.engine")
    results = detector.detect_batch([img1, img2])

模型文件位于 yolo_model/ 目录，由 ultralytics YOLO 导出。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class BallDetection:
    """单个网球检测结果。"""
    x: float           # 中心点 x（像素）
    y: float           # 中心点 y（像素）
    confidence: float   # 置信度 0~1
    x1: float          # 边界框左上角 x
    y1: float          # 边界框左上角 y
    x2: float          # 边界框右下角 x
    y2: float          # 边界框右下角 y


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MODEL_DIR = _PROJECT_ROOT / "yolo_model"


def _find_model(model_dir: Path) -> Path:
    """在目录中查找最新的 .engine > .onnx > .pt 模型文件。"""
    for ext in ("*.engine", "*.onnx", "*.pt"):
        files = sorted(model_dir.glob(ext))
        if files:
            return files[-1]
    raise FileNotFoundError(f"在 {model_dir} 中未找到模型文件 (.engine/.onnx/.pt)")


class BallDetector:
    """
    网球检测器 — 加载 YOLO 模型，返回网球像素位置。

    自动按优先级查找模型：.engine > .onnx > .pt
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        conf_threshold: float = 0.25,
        device: Optional[str] = None,
        half: bool = False,
    ):
        from ultralytics import YOLO

        if model_path is None:
            model_path = _find_model(_DEFAULT_MODEL_DIR)
        self._model_path = Path(model_path)

        if not self._model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self._model_path}")

        self._conf = conf_threshold
        self._half = half
        self._device = device

        # .engine / .onnx 需要显式指定 task
        if self._model_path.suffix in (".engine", ".onnx"):
            self._model = YOLO(str(self._model_path), task="detect")
        else:
            self._model = YOLO(str(self._model_path))

    def _predict(self, source, conf: float) -> list:
        kwargs = {"conf": conf, "verbose": False}
        if self._device is not None:
            kwargs["device"] = self._device
        if self._half:
            kwargs["half"] = True
        return self._model.predict(source, **kwargs)

    @staticmethod
    def _parse_boxes(result) -> list[BallDetection]:
        detections: list[BallDetection] = []
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return detections
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            detections.append(BallDetection(
                x=float((x1 + x2) / 2.0),
                y=float((y1 + y2) / 2.0),
                confidence=float(boxes.conf[i].cpu()),
                x1=float(x1), y1=float(y1),
                x2=float(x2), y2=float(y2),
            ))
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def detect(self, image: np.ndarray, conf: Optional[float] = None) -> list[BallDetection]:
        """对单张图片推理，返回 BallDetection 列表（按置信度降序）。"""
        results = self._predict(image, conf if conf is not None else self._conf)
        out: list[BallDetection] = []
        for r in results:
            out.extend(self._parse_boxes(r))
        return out

    def detect_batch(
        self, images: list[np.ndarray], conf: Optional[float] = None
    ) -> list[list[BallDetection]]:
        """批量推理 — 多张图片单次 GPU 调用，返回每张图的检测列表。"""
        results = self._predict(images, conf if conf is not None else self._conf)
        return [self._parse_boxes(r) for r in results]

    @property
    def model_path(self) -> Path:
        return self._model_path
