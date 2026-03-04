# -*- coding: utf-8 -*-
"""
BallDetector 测试脚本 — 输入图片，打印检测结果，并生成标注了网球位置的 output 图片。

用法:
    python test_ball_detector.py <图片路径> [--conf 0.25] [--device cpu]

示例:
    python test_ball_detector.py test.jpg
    python test_ball_detector.py photo.png --conf 0.5
    python test_ball_detector.py photo.png --output result.jpg
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2

from src.ball_detector import BallDetector


def draw_detections(image, detections):
    """在图片上绘制检测结果：边界框 + 中心点 + 标签。"""
    output = image.copy()
    for i, d in enumerate(detections, 1):
        cx, cy = int(d.x), int(d.y)
        x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)

        # 绿色边界框
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 红色中心点
        cv2.circle(output, (cx, cy), 5, (0, 0, 255), -1)
        # 标签：序号 + 坐标 + 置信度
        label = f"#{i} ({cx},{cy}) {d.confidence:.0%}"
        cv2.putText(output, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return output


def main():
    parser = argparse.ArgumentParser(description="BallDetector 测试：检测网球并生成标注图片")
    parser.add_argument("image", help="输入图片路径")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值 (默认 0.25)")
    parser.add_argument("--device", type=str, default=None, help="推理设备，如 cuda:0 或 cpu")
    parser.add_argument("--output", type=str, default=None, help="输出图片路径 (默认: <原文件名>_output.<扩展名>)")
    args = parser.parse_args()

    # --- 读取图片 ---
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"错误: 图片文件不存在 — {image_path}")
        sys.exit(1)

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"错误: 无法读取图片 — {image_path}")
        sys.exit(1)

    # --- 检测 ---
    detector = BallDetector(conf_threshold=args.conf, device=args.device)
    detections = detector.detect(img)
    positions = [(d.x, d.y) for d in detections]

    # --- 打印信息 ---
    print(f"模型: {detector.model_path.name}")
    print(f"图片: {image_path}  ({img.shape[1]}x{img.shape[0]})")
    print(f"检测到 {len(detections)} 个网球\n")

    if not detections:
        print("未检测到网球，不生成输出图片。")
        sys.exit(0)

    for i, d in enumerate(detections, 1):
        print(f"  [{i}] 中心坐标: ({d.x:.1f}, {d.y:.1f})  "
              f"置信度: {d.confidence:.2%}  "
              f"边界框: ({d.x1:.0f},{d.y1:.0f})-({d.x2:.0f},{d.y2:.0f})")

    print(f"\n坐标列表: {positions}")

    # --- 生成标注图片 ---
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = image_path.parent / f"{image_path.stem}_output{image_path.suffix}"

    output_img = draw_detections(img, detections)
    cv2.imwrite(str(output_path), output_img)
    print(f"\n标注图片已保存: {output_path}")


if __name__ == "__main__":
    main()
