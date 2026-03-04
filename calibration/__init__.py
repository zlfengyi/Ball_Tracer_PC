# -*- coding: utf-8 -*-
"""
calibration — 双目相机标定子项目。

包含标定板图片采集、双目标定、地面标注、大地坐标系注册等完整流程。
"""
from .stereo_calibrator import BoardConfig, StereoCalibrator, StereoCalibResult

__all__ = [
    "BoardConfig",
    "StereoCalibrator",
    "StereoCalibResult",
]
