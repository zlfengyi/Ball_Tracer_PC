# -*- coding: utf-8 -*-
"""
src — 网球追踪系统核心模块。

子模块：
  ball_grabber   相机采集（SyncCapture 同步取图、ImageGrabber、frame_to_numpy）（文件名保留）
  ball_detector  YOLO 网球检测（支持 TensorRT/ONNX/PyTorch 后端）
  ball_localizer 双目三角测量 3D 定位（网球）
  car_localizer  AprilTag 双目定位（车辆）
  curve3         轨迹拟合与击球点预测

标定模块已移至 calibration/ 子项目。
"""
from .ball_grabber import (
    Frame,
    ImageGrabber,
    SyncCapture,
    close_camera,
    frame_to_numpy,
    list_devices,
    open_camera,
)
from .ball_detector import BallDetection, BallDetector
from .ball_localizer import Ball3D, BallLocalizer
from .car_localizer import CarDetection, CarLoc, CarLocalizer
from .curve3 import (
    BallObservation, PredictHitPos, Curve3Tracker, FittedCurve, fit_curve,
    TrackerState, TrackerResult,
)

__all__ = [
    "Frame",
    "ImageGrabber",
    "SyncCapture",
    "frame_to_numpy",
    "open_camera",
    "close_camera",
    "list_devices",
    "BallDetection",
    "BallDetector",
    "Ball3D",
    "BallLocalizer",
    "CarDetection",
    "CarLoc",
    "CarLocalizer",
    "BallObservation",
    "PredictHitPos",
    "Curve3Tracker",
    "FittedCurve",
    "fit_curve",
    "TrackerState",
    "TrackerResult",
]
