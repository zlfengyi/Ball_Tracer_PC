# -*- coding: utf-8 -*-
"""
棋盘格角点检测 v2 — 更精准 + 质量评分 + 坏帧筛选。

改进：
  1. findChessboardCornersSB（更鲁棒）+ 回退到经典方法
  2. cornerSubPix 更精细参数
  3. 网格规整度评分：行列间距一致性
  4. 自动标记并导出坏帧

用法：
  python -m calibration.detect_corners_v2 --images images/005_checker_030321

输出：
  corner_detections.json  — 含角点 + 每帧质量评分
  {serial}/det_v2/        — 可视化（含评分）
  bad_corners/            — 评分低于阈值的帧
"""

from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np


# ================================================================
#  角点检测（多策略）
# ================================================================

def detect_corners(gray: np.ndarray, pattern: tuple[int, int]
                   ) -> np.ndarray | None:
    """
    检测棋盘格角点，返回 (N,1,2) float32 或 None。
    使用经典方法 + 精细 cornerSubPix（保证角点排列顺序一致）。
    """
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
             | cv2.CALIB_CB_NORMALIZE_IMAGE
             | cv2.CALIB_CB_FAST_CHECK)
    ret, corners = cv2.findChessboardCorners(gray, pattern, None, flags)
    if not ret or corners is None or len(corners) != pattern[0] * pattern[1]:
        # 去掉 FAST_CHECK 重试（更慢但更鲁棒）
        flags2 = (cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        ret, corners = cv2.findChessboardCorners(gray, pattern, None, flags2)
        if not ret or corners is None or len(corners) != pattern[0] * pattern[1]:
            return None

    # 精细亚像素（迭代次数更多、收敛阈值更小、窗口更合适）
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                100, 0.0001)
    corners = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
    return corners


# ================================================================
#  网格质量评分
# ================================================================

def grid_quality_score(corners: np.ndarray, cols: int, rows: int) -> float:
    """
    评估检测到的角点是否构成规则网格。

    方法：将角点 reshape 成 (rows, cols, 2) 网格，
    计算行方向和列方向的相邻间距，
    评分 = 1 - (间距标准差 / 间距均值)。
    完美网格 → 1.0，畸形网格 → 接近 0。

    Returns: 0~1 之间的评分
    """
    pts = corners.reshape(-1, 2)
    if len(pts) != rows * cols:
        return 0.0

    grid = pts.reshape(rows, cols, 2)

    dists = []
    # 行方向间距（同一行相邻角点）
    for r in range(rows):
        for c in range(cols - 1):
            d = np.linalg.norm(grid[r, c + 1] - grid[r, c])
            dists.append(d)
    # 列方向间距（同一列相邻角点）
    for r in range(rows - 1):
        for c in range(cols):
            d = np.linalg.norm(grid[r + 1, c] - grid[r, c])
            dists.append(d)

    dists = np.array(dists)
    if len(dists) == 0 or dists.mean() < 1e-6:
        return 0.0

    # 分别计算行间距和列间距的一致性
    row_dists = []
    for r in range(rows):
        for c in range(cols - 1):
            row_dists.append(np.linalg.norm(grid[r, c + 1] - grid[r, c]))
    col_dists = []
    for r in range(rows - 1):
        for c in range(cols):
            col_dists.append(np.linalg.norm(grid[r + 1, c] - grid[r, c]))

    row_dists = np.array(row_dists)
    col_dists = np.array(col_dists)

    # 每行内的间距应接近等距
    row_cv = row_dists.std() / row_dists.mean() if row_dists.mean() > 0 else 1.0
    col_cv = col_dists.std() / col_dists.mean() if col_dists.mean() > 0 else 1.0

    # 综合评分：cv 越小越好，cv=0 → score=1
    avg_cv = (row_cv + col_cv) / 2.0
    score = max(0.0, 1.0 - avg_cv * 5.0)  # cv=0.2 → score=0
    return round(score, 4)


# ================================================================
#  处理单张图片
# ================================================================

def process_image(path: Path, pattern: tuple[int, int],
                  det_dir: Path | None, cols: int, rows: int
                  ) -> dict | None:
    """检测角点 + 评分 + 保存可视化。"""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    h, w = img.shape[:2]
    corners = detect_corners(img, pattern)

    # 可视化
    if det_dir is not None:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if corners is not None:
            score = grid_quality_score(corners, cols, rows)
            cv2.drawChessboardCorners(vis, pattern, corners, True)
            color = (0, 255, 0) if score > 0.8 else (0, 165, 255) if score > 0.5 else (0, 0, 255)
            cv2.putText(vis, f"score={score:.3f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        else:
            cv2.putText(vis, "NOT DETECTED", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        det_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(det_dir / f"{path.stem}_det.jpg"), vis,
                    [cv2.IMWRITE_JPEG_QUALITY, 80])

    if corners is None:
        return None

    score = grid_quality_score(corners, cols, rows)
    return {
        "corners": corners.reshape(-1, 2).tolist(),
        "image_size": [w, h],
        "score": float(score),
    }


# ================================================================
#  主程序
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="棋盘格角点检测 v2")
    parser.add_argument("--images", type=str, required=True,
                        help="图片目录（相对于 calibration/）")
    parser.add_argument("--serials", type=str, nargs="+", default=None)
    parser.add_argument("--range-start", type=int, default=1)
    parser.add_argument("--range-end", type=int, default=500)
    parser.add_argument("--inner-cols", type=int, default=8)
    parser.add_argument("--inner-rows", type=int, default=11)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=0.8,
                        help="低于此评分视为坏帧 (默认: 0.8)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    calib_root = Path(__file__).resolve().parent
    image_dir = calib_root / args.images

    if args.serials:
        serials = args.serials
    else:
        camera_json = project_root / "src" / "config" / "camera.json"
        with open(camera_json, encoding="utf-8") as f:
            cam_cfg = json.load(f)
        serials = cam_cfg.get("slave_serials", [])

    pattern = (args.inner_cols, args.inner_rows)
    cols, rows = args.inner_cols, args.inner_rows
    start, end = args.range_start, args.range_end
    threshold = args.score_threshold

    print("=" * 60)
    print("       棋盘格角点检测 v2")
    print("=" * 60)
    print(f"  目录: {image_dir}")
    print(f"  相机: {serials}")
    print(f"  范围: {start:03d}~{end:03d}")
    print(f"  棋盘: {cols}x{rows}")
    print(f"  评分阈值: {threshold}")

    # 收集任务
    tasks = []
    for sn in serials:
        det_dir = image_dir / sn / "det_v2"
        for idx in range(start, end + 1):
            path = image_dir / sn / f"{idx:03d}.png"
            if path.exists():
                tasks.append((sn, idx, path, det_dir))

    print(f"  图片: {len(tasks)} 张")
    print()

    # 并行检测
    results = {sn: {} for sn in serials}
    stats = {sn: {"total": 0, "detected": 0, "good": 0, "bad": 0, "scores": []}
             for sn in serials}
    processed = 0
    t0 = time.monotonic()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for sn, idx, path, det_dir in tasks:
            fut = pool.submit(process_image, path, pattern, det_dir, cols, rows)
            futures[fut] = (sn, idx, path)

        for fut in as_completed(futures):
            sn, idx, path = futures[fut]
            stats[sn]["total"] += 1
            processed += 1

            try:
                det = fut.result()
            except Exception as e:
                print(f"  错误: {sn}/{idx:03d} — {e}")
                det = None

            if det is not None:
                results[sn][str(idx)] = det
                stats[sn]["detected"] += 1
                score = det["score"]
                stats[sn]["scores"].append(score)
                if score >= threshold:
                    stats[sn]["good"] += 1
                else:
                    stats[sn]["bad"] += 1

            if processed % 100 == 0 or processed == len(tasks):
                elapsed = time.monotonic() - t0
                speed = processed / elapsed if elapsed > 0 else 0
                print(f"  [{processed}/{len(tasks)}] "
                      f"{elapsed:.0f}s, {speed:.1f} img/s")

    # ── 统计 ──
    elapsed = time.monotonic() - t0
    print(f"\n{'='*60}")
    print(f"  检测完成 ({elapsed:.1f}s)")
    print(f"{'='*60}")

    for sn in serials:
        s = stats[sn]
        scores = np.array(s["scores"]) if s["scores"] else np.array([0])
        print(f"\n  {sn}:")
        print(f"    检测: {s['detected']}/{s['total']}")
        print(f"    评分: median={np.median(scores):.3f} "
              f"mean={np.mean(scores):.3f} "
              f"min={np.min(scores):.3f}")
        print(f"    好帧(≥{threshold}): {s['good']}")
        print(f"    坏帧(<{threshold}): {s['bad']}")

    # ── 导出坏帧 ──
    import shutil
    bad_dir = image_dir / "bad_corners"
    if bad_dir.exists():
        shutil.rmtree(bad_dir)

    bad_count = 0
    for sn in serials:
        for idx_s, det in results[sn].items():
            if det["score"] < threshold:
                idx = int(idx_s)
                sn_dir = bad_dir / sn
                sn_dir.mkdir(parents=True, exist_ok=True)
                # 复制原图
                src = image_dir / sn / f"{idx:03d}.png"
                if src.exists():
                    tag = f"score{det['score']:.3f}"
                    shutil.copy2(src, sn_dir / f"{idx:03d}_{tag}.png")
                # 复制检测标注图
                src_det = image_dir / sn / "det_v2" / f"{idx:03d}_det.jpg"
                if src_det.exists():
                    shutil.copy2(src_det, sn_dir / f"{idx:03d}_{tag}_det.jpg")
                bad_count += 1

    print(f"\n  坏帧共 {bad_count} 张，已导出到: {bad_dir}")

    # ── 保存 JSON ──
    output = {
        "board": {"inner_cols": cols, "inner_rows": rows},
        "cameras": results,
    }
    out_path = image_dir / "corner_detections.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  已保存: {out_path} ({size_mb:.1f}MB)")


if __name__ == "__main__":
    main()
