# -*- coding: utf-8 -*-
"""
网球/球拍检测模块。

优先级：
1. TensorRT engine
2. PyTorch / Ultralytics
3. ONNX Runtime

对于 `.onnx`，优先直接走 ONNX Runtime，避免 Ultralytics 在 CPU 环境下的额外开销。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Optional

import cv2
import numpy as np


@dataclass
class BallDetection:
    """单个检测结果。"""
    x: float
    y: float
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    label: str = "tennis_ball"

    @property
    def is_tennis_ball(self) -> bool:
        return self.label == "tennis_ball"

    @property
    def is_stationary_object(self) -> bool:
        return self.label == "stationary_object"

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def aspect_ratio(self) -> float:
        width = self.width
        height = self.height
        if width <= 0.0 or height <= 0.0:
            return math.inf
        return max(width, height) / min(width, height)


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MODEL_DIR = _PROJECT_ROOT / "yolo_model"


def _has_tensorrt() -> bool:
    try:
        import tensorrt  # noqa: F401
    except Exception:
        return False
    return True


def _find_model(model_dir: Path) -> Path:
    """Choose the fastest model format supported by the current runtime."""
    patterns = ("*.engine", "*.pt", "*.onnx") if _has_tensorrt() else (
        "*.pt", "*.onnx", "*.engine"
    )
    for ext in patterns:
        files = sorted(model_dir.glob(ext))
        if files:
            return files[-1]
    raise FileNotFoundError(f"在 {model_dir} 中未找到模型文件 (.engine/.onnx/.pt)")


def _infer_fixed_engine_batch(model_path: Path) -> int | None:
    if model_path.suffix != ".engine":
        return None
    match = re.search(r"_b(\d+)(?:_|\.engine$)", model_path.name)
    if not match:
        return None
    batch = int(match.group(1))
    return batch if batch > 0 else None


def _infer_engine_imgsz(model_path: Path) -> int | None:
    if model_path.suffix != ".engine":
        return None
    match = re.search(r"_b\d+_(\d+)(?:_|\.engine$)", model_path.name)
    if not match:
        return None
    imgsz = int(match.group(1))
    return imgsz if imgsz > 0 else None


class BallDetector:
    """
    通用检测器。

    `.onnx` 默认走 ONNX Runtime 直推；`.pt/.engine` 继续走 Ultralytics。
    """

    # 形状门诊断（进程级累计，纯计数不改行为）。
    # 运动拖影会把球框沿运动方向拉长：blur_px ≈ v_img(px/s) × 曝光(s)，
    # 回球段 v_img 可达来球的 2~3 倍，长宽比随之冲高，可能整段被形状门吃掉。
    # 事后只看 3D 观测是分不清「没检出」和「检出后被门拦掉」的 —— 这组计数就是
    # 为了把这两种情况分开，落进 session json 的 summary.detection_shape_gate。
    shape_gate_stats: dict[str, float] = {
        "ball_kept": 0,
        "ball_rejected": 0,
        "rejected_aspect_max": 0.0,
        "rejected_aspect_sum": 0.0,
    }

    @classmethod
    def reset_shape_gate_stats(cls) -> None:
        cls.shape_gate_stats = {
            "ball_kept": 0,
            "ball_rejected": 0,
            "rejected_aspect_max": 0.0,
            "rejected_aspect_sum": 0.0,
        }

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        conf_threshold: float = 0.25,
        device: Optional[str] = None,
        half: bool = False,
        duplicate_iou_threshold: Optional[float] = 0.95,
        max_box_aspect_ratio: Optional[float] = 1.2,
        onnx_providers: Optional[list[str]] = None,
    ):
        if model_path is None:
            model_path = _find_model(_DEFAULT_MODEL_DIR)
        self._model_path = Path(model_path)

        if not self._model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self._model_path}")

        self._conf = conf_threshold
        self._half = half
        self._device = device
        self._duplicate_iou_threshold = duplicate_iou_threshold
        self._max_box_aspect_ratio = max_box_aspect_ratio
        self._onnx_providers = list(onnx_providers) if onnx_providers is not None else None
        self._model = None
        self._onnx_session = None
        self._onnx_input_name: str | None = None
        self._onnx_input_hw: tuple[int, int] | None = None
        self._fixed_engine_batch = _infer_fixed_engine_batch(self._model_path)
        self._engine_imgsz = _infer_engine_imgsz(self._model_path)
        self._pinned_input = None      # 复用的 pinned H2D 缓冲，见 _engine_batch_detections
        self._engine_fast_path = None  # None=未判定 / True=可用 / False=已回退

        if self._model_path.suffix == ".onnx":
            self._init_onnx_runtime()
        else:
            self._init_ultralytics()

    def _init_onnx_runtime(self) -> None:
        try:
            import os
            import torch
            import onnxruntime as ort
        except Exception:
            self._init_ultralytics()
            return

        if self._onnx_providers is not None:
            providers = list(self._onnx_providers)
        else:
            providers = ["CPUExecutionProvider"]
            try:
                if torch.cuda.is_available():
                    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
                    if hasattr(os, "add_dll_directory") and os.path.isdir(torch_lib):
                        os.add_dll_directory(torch_lib)
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            except Exception:
                providers = ["CPUExecutionProvider"]

        self._onnx_session = ort.InferenceSession(
            str(self._model_path),
            providers=providers,
        )
        model_input = self._onnx_session.get_inputs()[0]
        self._onnx_input_name = model_input.name
        self._onnx_input_hw = (
            int(model_input.shape[3]),
            int(model_input.shape[2]),
        )

    def _init_ultralytics(self) -> None:
        from ultralytics import YOLO

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
        if self._engine_imgsz is not None:
            kwargs["imgsz"] = self._engine_imgsz
        return self._model.predict(source, **kwargs)

    def _engine_batch_detections(
        self, images: list[np.ndarray], conf: float
    ) -> Optional[list[list[BallDetection]]]:
        """Direct engine path, bypassing ultralytics' generic pre/post-processing.

        This engine carries its own end2end NMS head: the raw output is
        (B, 300, 6) = [x1, y1, x2, y2, conf, cls] already in input-pixel space,
        so ultralytics' postprocess only rebuilds Results objects around numbers
        the engine already produced. Skipping it — plus filling a pinned buffer
        and doing BGR->RGB / HWC->CHW on the GPU instead of a strided CPU copy —
        is what buys the per-frame budget at 60 Hz (0906 measured yolo_avg 12.8 ms
        at 60 Hz and 14.8 ms at 70 Hz, against a 16.7 / 14.3 ms whole-frame budget).

        Returns None whenever the setup is not the shape this path assumes, so
        the caller falls back to the ultralytics path.
        """
        predictor = getattr(self._model, "predictor", None)
        size = self._engine_imgsz
        if predictor is None or size is None:
            return None
        if not all(
            image.ndim == 3 and image.shape == (size, size, 3) for image in images
        ):
            return None

        import torch

        buffer = self._pinned_input
        if buffer is None or buffer.shape[0] != len(images):
            buffer = torch.empty(
                (len(images), size, size, 3), dtype=torch.uint8, pin_memory=True
            )
            self._pinned_input = buffer
        staging = buffer.numpy()
        for index, image in enumerate(images):
            staging[index] = image

        device_batch = buffer.to(predictor.device, non_blocking=True)
        # BGR HWC uint8 -> RGB CHW float, all on the GPU
        tensor = device_batch.permute(0, 3, 1, 2).flip(1).contiguous()
        tensor = tensor.half() if predictor.model.fp16 else tensor.float()
        tensor = tensor.div_(255.0)

        raw = predictor.inference(tensor)
        if isinstance(raw, (list, tuple)):
            if not raw:
                return None
            raw = raw[0]
        if raw.ndim != 3 or raw.shape[0] != len(images) or raw.shape[2] != 6:
            return None      # not the end2end head — let ultralytics handle it

        keep = raw[:, :, 4] >= conf
        results: list[list[BallDetection]] = [[] for _ in images]
        if bool(keep.any()):
            rows = raw[keep].cpu().numpy()          # only survivors cross PCIe
            owners = keep.nonzero(as_tuple=False)[:, 0].cpu().numpy()
            for owner, row in zip(owners, rows):
                x1, y1, x2, y2, confidence = (float(v) for v in row[:5])
                results[int(owner)].append(BallDetection(
                    x=(x1 + x2) / 2.0,
                    y=(y1 + y2) / 2.0,
                    confidence=confidence,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                ))
        for detections in results:
            detections.sort(key=lambda det: det.confidence, reverse=True)
        return results

    def _predict_fixed_batch(self, images: list[np.ndarray], conf: float) -> list:
        if not images:
            return []

        fixed_batch = self._fixed_engine_batch
        if fixed_batch is None or self._onnx_session is not None:
            return self._predict(images, conf)

        predictor = getattr(self._model, "predictor", None)
        results = []
        for start in range(0, len(images), fixed_batch):
            chunk = list(images[start:start + fixed_batch])
            actual_n = len(chunk)
            while len(chunk) < fixed_batch:
                chunk.append(chunk[-1])
            exact_engine_input = (
                predictor is not None
                and self._engine_imgsz is not None
                and all(
                    image.ndim == 3
                    and image.shape == (
                        self._engine_imgsz,
                        self._engine_imgsz,
                        3,
                    )
                    for image in chunk
                )
            )
            if exact_engine_input:
                import torch

                batch = np.ascontiguousarray(
                    np.stack(chunk)[..., ::-1].transpose((0, 3, 1, 2))
                )
                tensor = torch.from_numpy(batch).to(predictor.device)
                tensor = tensor.half() if predictor.model.fp16 else tensor.float()
                tensor.div_(255.0)
                predictor.args.conf = conf
                predictions = predictor.inference(tensor)
                chunk_results = predictor.postprocess(predictions, tensor, chunk)
            else:
                chunk_results = self._predict(chunk, conf)
                predictor = getattr(self._model, "predictor", None)
            results.extend(chunk_results[:actual_n])
        return results

    def _predict_onnx(self, image: np.ndarray) -> np.ndarray:
        assert self._onnx_session is not None
        assert self._onnx_input_name is not None
        assert self._onnx_input_hw is not None

        input_w, input_h = self._onnx_input_hw
        resized = cv2.resize(image, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        tensor = np.ascontiguousarray(tensor[None])
        return self._onnx_session.run(None, {self._onnx_input_name: tensor})[0]

    @staticmethod
    def _parse_boxes(result) -> list[BallDetection]:
        detections: list[BallDetection] = []
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return detections
        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), confidence in zip(xyxy, confidences):
            detections.append(BallDetection(
                x=float((x1 + x2) / 2.0),
                y=float((y1 + y2) / 2.0),
                confidence=float(confidence),
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
            ))
        detections.sort(key=lambda det: det.confidence, reverse=True)
        return detections

    def _parse_onnx_output(
        self,
        output: np.ndarray,
        image_shape: tuple[int, int],
        conf: float,
    ) -> list[BallDetection]:
        assert self._onnx_input_hw is not None

        pred = np.asarray(output)
        if pred.ndim == 3:
            pred = pred[0]
        if pred.ndim != 2:
            return []
        if pred.shape[0] < pred.shape[1]:
            pred = pred.transpose(1, 0)
        if pred.shape[1] < 5:
            return []

        boxes_xywh = pred[:, :4]
        class_scores = pred[:, 4:]
        if class_scores.size == 0:
            return []

        if class_scores.ndim == 1:
            confidences = class_scores
        else:
            confidences = class_scores.max(axis=1)

        mask = confidences >= conf
        if not np.any(mask):
            return []

        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]

        image_h, image_w = image_shape[:2]
        input_w, input_h = self._onnx_input_hw
        scale_x = image_w / float(input_w)
        scale_y = image_h / float(input_h)

        nms_boxes: list[list[float]] = []
        candidates: list[tuple[float, float, float, float, float]] = []
        for box_xywh, score in zip(boxes_xywh, confidences):
            cx, cy, bw, bh = box_xywh.tolist()
            x1 = max(0.0, (cx - bw / 2.0) * scale_x)
            y1 = max(0.0, (cy - bh / 2.0) * scale_y)
            x2 = min(float(image_w), (cx + bw / 2.0) * scale_x)
            y2 = min(float(image_h), (cy + bh / 2.0) * scale_y)
            if x2 <= x1 or y2 <= y1:
                continue
            nms_boxes.append([x1, y1, x2 - x1, y2 - y1])
            candidates.append((x1, y1, x2, y2, float(score)))

        if not candidates:
            return []

        indices = cv2.dnn.NMSBoxes(
            nms_boxes,
            [candidate[4] for candidate in candidates],
            conf,
            0.45,
        )
        if len(indices) == 0:
            return []

        detections: list[BallDetection] = []
        for idx in np.array(indices).reshape(-1):
            x1, y1, x2, y2, score = candidates[int(idx)]
            detections.append(BallDetection(
                x=float((x1 + x2) / 2.0),
                y=float((y1 + y2) / 2.0),
                confidence=float(score),
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
            ))
        detections.sort(key=lambda det: det.confidence, reverse=True)
        return detections

    @staticmethod
    def _box_iou(lhs: BallDetection, rhs: BallDetection) -> float:
        inter_x1 = max(lhs.x1, rhs.x1)
        inter_y1 = max(lhs.y1, rhs.y1)
        inter_x2 = min(lhs.x2, rhs.x2)
        inter_y2 = min(lhs.y2, rhs.y2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        if inter_w <= 0.0 or inter_h <= 0.0:
            return 0.0

        inter_area = inter_w * inter_h
        union_area = lhs.width * lhs.height + rhs.width * rhs.height - inter_area
        if union_area <= 0.0:
            return 0.0
        return inter_area / union_area

    @staticmethod
    def _passes_box_shape(
        det: BallDetection,
        max_box_aspect_ratio: Optional[float],
    ) -> bool:
        if det.width <= 0.0 or det.height <= 0.0:
            return False
        if max_box_aspect_ratio is None:
            return True
        return det.aspect_ratio <= max_box_aspect_ratio

    @classmethod
    def postprocess_detections(
        cls,
        detections: list[BallDetection],
        *,
        duplicate_iou_threshold: Optional[float] = 0.95,
        max_box_aspect_ratio: Optional[float] = 1.2,
    ) -> list[BallDetection]:
        filtered: list[BallDetection] = []
        stats = cls.shape_gate_stats
        for det in detections:
            if cls._passes_box_shape(det, max_box_aspect_ratio):
                filtered.append(det)
                if det.is_tennis_ball:
                    stats["ball_kept"] += 1
            elif det.is_tennis_ball:
                ar = det.aspect_ratio
                stats["ball_rejected"] += 1
                if math.isfinite(ar):
                    stats["rejected_aspect_sum"] += ar
                    stats["rejected_aspect_max"] = max(stats["rejected_aspect_max"], ar)
        filtered.sort(key=lambda det: det.confidence, reverse=True)

        kept: list[BallDetection] = []
        for det in filtered:
            if duplicate_iou_threshold is not None and any(
                det.label == kept_det.label
                and cls._box_iou(det, kept_det) >= duplicate_iou_threshold
                for kept_det in kept
            ):
                continue
            kept.append(det)
        return kept

    def detect(self, image: np.ndarray, conf: Optional[float] = None) -> list[BallDetection]:
        """单张图推理。"""
        target_conf = conf if conf is not None else self._conf

        if self._onnx_session is not None:
            detections = self._parse_onnx_output(
                self._predict_onnx(image),
                image.shape,
                target_conf,
            )
            return self.postprocess_detections(
                detections,
                duplicate_iou_threshold=self._duplicate_iou_threshold,
                max_box_aspect_ratio=self._max_box_aspect_ratio,
            )

        results = self._predict_fixed_batch([image], target_conf)
        out: list[BallDetection] = []
        for result in results[:1]:
            out.extend(self._parse_boxes(result))
        return self.postprocess_detections(
            out,
            duplicate_iou_threshold=self._duplicate_iou_threshold,
            max_box_aspect_ratio=self._max_box_aspect_ratio,
        )

    def detect_batch(
        self,
        images: list[np.ndarray],
        conf: Optional[float] = None,
    ) -> list[list[BallDetection]]:
        """批量推理。ONNX 直推时按单张循环，避免固定 batch 限制。"""
        if self._onnx_session is not None:
            return [self.detect(image, conf=conf) for image in images]

        threshold = conf if conf is not None else self._conf
        # engine 自带 end2end NMS 时走直连快路径；第一次调用 predictor 还没建，
        # 会返回 None 落回下面的 ultralytics 路径（那一次顺便把 predictor 建出来）。
        if self._engine_fast_path is not False:
            fast = self._engine_batch_detections(images, threshold)
            if fast is not None:
                self._engine_fast_path = True
                return [
                    self.postprocess_detections(
                        detections,
                        duplicate_iou_threshold=self._duplicate_iou_threshold,
                        max_box_aspect_ratio=self._max_box_aspect_ratio,
                    )
                    for detections in fast
                ]

        results = self._predict_fixed_batch(images, threshold)
        return [
            self.postprocess_detections(
                self._parse_boxes(result),
                duplicate_iou_threshold=self._duplicate_iou_threshold,
                max_box_aspect_ratio=self._max_box_aspect_ratio,
            )
            for result in results
        ]

    @property
    def model_path(self) -> Path:
        return self._model_path
