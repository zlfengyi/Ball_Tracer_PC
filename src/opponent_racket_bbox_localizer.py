"""Single-class opponent racket-head bbox inference in native image pixels."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


@dataclass(frozen=True)
class OpponentRacketBBoxDetection:
    serial: str
    bbox_confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    center_xy: tuple[float, float]


class OpponentRacketBBoxLocalizer:
    """Decode a fixed 320px YOLO detector and return bbox geometric centres."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        bbox_confidence_min: float,
        providers: list[str],
        nms_iou: float = 0.45,
    ) -> None:
        if not 0.0 <= bbox_confidence_min <= 1.0:
            raise ValueError("bbox_confidence_min must be in [0, 1]")
        if not providers:
            raise ValueError("at least one ONNX provider is required")
        if not 0.0 <= nms_iou <= 1.0:
            raise ValueError("nms_iou must be in [0, 1]")

        self._bbox_confidence_min = float(bbox_confidence_min)
        self._nms_iou = float(nms_iou)
        self._session = ort.InferenceSession(str(model_path), providers=providers)

        metadata = self._session.get_modelmeta().custom_metadata_map
        if metadata.get("task") != "detect":
            raise ValueError("opponent racket model must declare task=detect")
        try:
            names = ast.literal_eval(metadata["names"])
        except (KeyError, SyntaxError, ValueError) as exc:
            raise ValueError("opponent racket model has invalid names metadata") from exc
        if names not in ({0: "racket_head"}, {"0": "racket_head"}):
            raise ValueError(f"unexpected opponent racket classes: {names}")

        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        input_shapes = [list(item.shape) for item in inputs]
        output_shapes = [list(item.shape) for item in outputs]
        if len(inputs) != 1 or input_shapes[0] != [1, 3, 320, 320]:
            raise ValueError(f"unexpected opponent racket input: {input_shapes}")
        if (
            len(outputs) != 1
            or len(output_shapes[0]) != 3
            or output_shapes[0][:2] != [1, 5]
            or not isinstance(output_shapes[0][2], int)
            or output_shapes[0][2] <= 0
        ):
            raise ValueError(f"unexpected opponent racket output: {output_shapes}")

        self._input_name = inputs[0].name
        self._output_name = outputs[0].name
        self._output_count = output_shapes[0][2]

    @property
    def provider_info(self) -> dict[str, list[str]]:
        return {"opponent_racket_bbox": list(self._session.get_providers())}

    @staticmethod
    def _letterbox(image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        input_size = 320
        height, width = image.shape[:2]
        scale = min(input_size / width, input_size / height)
        resized_width = round(width * scale)
        resized_height = round(height * scale)
        resized = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
        left = round((input_size - resized_width) / 2.0 - 0.1)
        top = round((input_size - resized_height) / 2.0 - 0.1)
        padded = cv2.copyMakeBorder(
            resized,
            top,
            input_size - resized_height - top,
            left,
            input_size - resized_width - left,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        tensor = np.ascontiguousarray(
            padded[:, :, ::-1].transpose(2, 0, 1),
            dtype=np.float32,
        )
        tensor *= 1.0 / 255.0
        return tensor[None], scale, left, top

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
        order = np.argsort(scores)[::-1]
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        keep: list[int] = []
        while order.size:
            current = int(order[0])
            keep.append(current)
            if order.size == 1:
                break
            rest = order[1:]
            x0 = np.maximum(boxes[current, 0], boxes[rest, 0])
            y0 = np.maximum(boxes[current, 1], boxes[rest, 1])
            x1 = np.minimum(boxes[current, 2], boxes[rest, 2])
            y1 = np.minimum(boxes[current, 3], boxes[rest, 3])
            intersection = np.maximum(0.0, x1 - x0) * np.maximum(0.0, y1 - y0)
            union = areas[current] + areas[rest] - intersection
            iou = np.divide(
                intersection,
                union,
                out=np.zeros_like(intersection),
                where=union > 0.0,
            )
            order = rest[iou <= threshold]
        return keep

    def detect_candidates(
        self,
        image: np.ndarray,
        *,
        serial: str,
        image_origin_xy: tuple[int, int] = (0, 0),
        max_candidates: int = 3,
    ) -> list[OpponentRacketBBoxDetection]:
        if (
            image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
            or image.shape[0] < 16
            or image.shape[1] < 16
        ):
            raise ValueError("opponent racket detector requires a uint8 BGR crop")
        if not serial:
            raise ValueError("camera serial is required")
        if max_candidates < 1:
            raise ValueError("max_candidates must be positive")

        tensor, scale, left, top = self._letterbox(image)
        raw = self._session.run(
            [self._output_name],
            {self._input_name: tensor},
        )[0]
        if raw.shape != (1, 5, self._output_count):
            raise ValueError(f"unexpected opponent racket runtime output: {raw.shape}")

        predictions = raw[0].T
        predictions = predictions[
            np.isfinite(predictions).all(axis=1)
            & (predictions[:, 4] >= self._bbox_confidence_min)
            & (predictions[:, 2] > 0.0)
            & (predictions[:, 3] > 0.0)
        ]
        if predictions.size == 0:
            return []

        cx, cy, width, height = predictions[:, :4].T
        boxes = np.column_stack(
            (
                cx - width / 2.0,
                cy - height / 2.0,
                cx + width / 2.0,
                cy + height / 2.0,
            )
        )
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / scale
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, image.shape[1] - 1.0)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, image.shape[0] - 1.0)
        positive_area = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[positive_area]
        scores = predictions[positive_area, 4]
        if not len(boxes):
            return []

        origin_x, origin_y = (float(value) for value in image_origin_xy)
        offset = np.asarray((origin_x, origin_y, origin_x, origin_y))
        detections = []
        for index in self._nms(boxes, scores, self._nms_iou)[:max_candidates]:
            bbox = boxes[index] + offset
            detections.append(
                OpponentRacketBBoxDetection(
                    serial=serial,
                    bbox_confidence=float(scores[index]),
                    bbox_xyxy=tuple(float(value) for value in bbox),
                    center_xy=(
                        float((bbox[0] + bbox[2]) / 2.0),
                        float((bbox[1] + bbox[3]) / 2.0),
                    ),
                )
            )
        return detections


__all__ = [
    "OpponentRacketBBoxDetection",
    "OpponentRacketBBoxLocalizer",
]
