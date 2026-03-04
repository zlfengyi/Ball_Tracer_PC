# -*- coding: utf-8 -*-
"""
三目相机标定 v2 — 简洁版。

思路：
  1. 从 corner_detections.json 加载角点
  2. calibrateCamera 分别求 3 台相机内参（固定不动）
  3. stereoCalibrate 两两求相对外参，验证基线距离
  4. 以参考相机为原点，solvePnP 求每帧棋盘位姿（固定不动）
  5. 简化 BA：仅优化 2 台从相机的 6DOF 外参（共 12 参数）

用法：
  python -m calibration.run_calibration_v2 --images images/005_checker_030321
"""

from __future__ import annotations
import argparse, json, logging, sys, time
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares

log = logging.getLogger(__name__)


# ── 棋盘格 3D 点 ──
def make_obj_pts(cols, rows, sq):
    pts = np.zeros((cols * rows, 3), np.float32)
    for r in range(rows):
        for c in range(cols):
            pts[r * cols + c] = [c * sq, r * sq, 0.0]
    return pts


# ── 从 corner_detections.json 加载 ──
def load_corners(cache_path, serials, start, end, score_threshold=0.8):
    with open(cache_path, encoding="utf-8") as f:
        cache = json.load(f)
    board = cache["board"]
    cam_cache = cache["cameras"]
    detections = {}   # {idx: {serial: corners(N,1,2)}}
    image_sizes = {}
    skipped = {sn: 0 for sn in serials}
    for sn in serials:
        for idx_s, det in cam_cache.get(sn, {}).items():
            idx = int(idx_s)
            if idx < start or idx > end:
                continue
            score = det.get("score", 1.0)
            if score < score_threshold:
                skipped[sn] += 1
                continue
            corners = np.array(det["corners"], np.float32).reshape(-1, 1, 2)
            if sn not in image_sizes:
                image_sizes[sn] = tuple(det["image_size"])
            detections.setdefault(idx, {})[sn] = corners
    for sn in serials:
        if skipped[sn] > 0:
            print(f"  {sn}: 跳过 {skipped[sn]} 帧 (score<{score_threshold})")
    return detections, image_sizes, board


# ── Step 1: calibrateCamera 求内参 ──
def calibrate_intrinsics(detections, image_sizes, serials, obj_pts, max_frames=80):
    K, D, per_frame_rvecs, per_frame_tvecs = {}, {}, {}, {}
    for sn in serials:
        idxs, objs, imgs = [], [], []
        for idx in sorted(detections):
            if sn in detections[idx]:
                idxs.append(idx)
                objs.append(obj_pts)
                imgs.append(detections[idx][sn])
        # 抽样
        if len(objs) > max_frames:
            step = len(objs) / max_frames
            sel = [int(i * step) for i in range(max_frames)]
            idxs = [idxs[i] for i in sel]
            objs = [objs[i] for i in sel]
            imgs = [imgs[i] for i in sel]
        rms, k, d, rvecs, tvecs = cv2.calibrateCamera(
            objs, imgs, image_sizes[sn], None, None)
        K[sn] = k
        D[sn] = d.ravel()[:5]
        # 保存每帧位姿（用于后面固定棋盘位姿）
        frame_rv, frame_tv = {}, {}
        for i, idx in enumerate(idxs):
            frame_rv[idx] = rvecs[i].ravel()
            frame_tv[idx] = tvecs[i].ravel()
        per_frame_rvecs[sn] = frame_rv
        per_frame_tvecs[sn] = frame_tv
        print(f"  {sn}: fx={k[0,0]:.1f} fy={k[1,1]:.1f} "
              f"cx={k[0,2]:.1f} cy={k[1,2]:.1f} rms={rms:.3f} ({len(objs)}帧)")
    return K, D, per_frame_rvecs, per_frame_tvecs


# ── Step 2: stereoCalibrate 两两校准 ──
def pairwise_stereo(detections, K, D, image_sizes, serials, obj_pts, max_frames=60):
    results = {}
    for i in range(len(serials)):
        for j in range(i + 1, len(serials)):
            sn1, sn2 = serials[i], serials[j]
            objs, imgs1, imgs2 = [], [], []
            for idx in sorted(detections):
                if sn1 in detections[idx] and sn2 in detections[idx]:
                    objs.append(obj_pts)
                    imgs1.append(detections[idx][sn1])
                    imgs2.append(detections[idx][sn2])
            if len(objs) > max_frames:
                step = len(objs) / max_frames
                sel = [int(i * step) for i in range(max_frames)]
                objs = [objs[s] for s in sel]
                imgs1 = [imgs1[s] for s in sel]
                imgs2 = [imgs2[s] for s in sel]
            flags = cv2.CALIB_FIX_INTRINSIC
            rms, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
                objs, imgs1, imgs2,
                K[sn1], D[sn1], K[sn2], D[sn2],
                image_sizes[sn1], flags=flags)
            baseline = np.linalg.norm(T)
            results[(sn1, sn2)] = (R, T, rms, baseline)
            print(f"  {sn1} <-> {sn2}: RMS={rms:.3f}px "
                  f"baseline={baseline:.0f}mm={baseline/1000:.2f}m "
                  f"T=[{T[0,0]:.1f}, {T[1,0]:.1f}, {T[2,0]:.1f}]")
    return results


# ── Step 3: 简化 BA，仅优化 2 从相机外参 ──
def simple_ba(detections, ref_serial, slave_serials, K, D,
              ref_rvecs, ref_tvecs, obj_pts, max_frames=50):
    """
    参数仅 12 个：2 台从相机各 6DOF (rvec + tvec)。
    棋盘位姿来自参考相机 solvePnP，固定不动。
    """
    # 选取三目共同帧，且参考相机有位姿的帧
    common = sorted(
        idx for idx in detections
        if all(sn in detections[idx] for sn in [ref_serial] + slave_serials)
        and idx in ref_rvecs
    )
    if len(common) > max_frames:
        step = len(common) / max_frames
        common = [common[int(i * step)] for i in range(max_frames)]

    n_corners = obj_pts.shape[0]
    print(f"  BA 使用 {len(common)} 帧, {len(slave_serials)} 从相机, "
          f"共 {len(common) * len(slave_serials) * n_corners * 2} 残差, 12 参数")

    # 预计算：参考相机坐标系下的棋盘 3D 点
    board_pts_ref = {}  # {idx: (N,3) 棋盘点在参考相机坐标系}
    for idx in common:
        rv = ref_rvecs[idx]
        tv = ref_tvecs[idx]
        R_b, _ = cv2.Rodrigues(rv)
        # P_ref = R_b @ P_board + tv
        pts_ref = (R_b @ obj_pts.T).T + tv.reshape(1, 3)
        board_pts_ref[idx] = pts_ref.astype(np.float64)

    # 初始化：用 solvePnP 从每帧估计从相机外参，然后取中位数
    init_params = []
    for sn in slave_serials:
        rvecs_est, tvecs_est = [], []
        for idx in common:
            ok, rv, tv = cv2.solvePnP(
                board_pts_ref[idx], detections[idx][sn],
                K[sn], D[sn], flags=cv2.SOLVEPNP_SQPNP)
            if ok:
                rvecs_est.append(rv.ravel())
                tvecs_est.append(tv.ravel())
        # 中位数
        rv_init = np.median(rvecs_est, axis=0)
        tv_init = np.median(tvecs_est, axis=0)
        init_params.extend(rv_init.tolist())
        init_params.extend(tv_init.tolist())
        print(f"  {sn} 初始化: t=[{tv_init[0]:.1f}, {tv_init[1]:.1f}, {tv_init[2]:.1f}]")

    x0 = np.array(init_params, np.float64)

    # 残差函数
    def residuals(x):
        res = []
        for si, sn in enumerate(slave_serials):
            rv_cam = x[si*6:si*6+3]
            tv_cam = x[si*6+3:si*6+6]
            R_cam, _ = cv2.Rodrigues(rv_cam)
            for idx in common:
                pts_ref = board_pts_ref[idx]
                # P_slave = R_cam @ P_ref + tv_cam
                pts_slave = (R_cam @ pts_ref.T).T + tv_cam.reshape(1, 3)
                proj, _ = cv2.projectPoints(
                    pts_slave, np.zeros(3), np.zeros(3), K[sn], D[sn])
                obs = detections[idx][sn].reshape(-1, 2)
                res.append((proj.reshape(-1, 2) - obs).ravel())
        return np.concatenate(res)

    t0 = time.monotonic()
    result = least_squares(residuals, x0, method='lm', max_nfev=500, verbose=1)
    elapsed = time.monotonic() - t0
    total_rms = np.sqrt(np.mean(result.fun ** 2))
    print(f"\n  BA 完成: {elapsed:.1f}s, RMS={total_rms:.3f}px, "
          f"nfev={result.nfev}")

    # 提取结果
    cam_ext = {}
    for si, sn in enumerate(slave_serials):
        rv = result.x[si*6:si*6+3]
        tv = result.x[si*6+3:si*6+6]
        R, _ = cv2.Rodrigues(rv)
        cam_ext[sn] = (R, tv.reshape(3, 1))
        dist = np.linalg.norm(tv)
        print(f"  {sn} → ref: t=[{tv[0]:.1f}, {tv[1]:.1f}, {tv[2]:.1f}] "
              f"dist={dist:.0f}mm")

        # 每从相机 RMS
        cam_res = []
        for idx in common:
            pts_slave = (R @ board_pts_ref[idx].T).T + tv.reshape(1, 3)
            proj, _ = cv2.projectPoints(
                pts_slave, np.zeros(3), np.zeros(3), K[sn], D[sn])
            obs = detections[idx][sn].reshape(-1, 2)
            cam_res.append(((proj.reshape(-1, 2) - obs) ** 2).sum(axis=1))
        rms = np.sqrt(np.mean(np.concatenate(cam_res)))
        print(f"    per-camera RMS: {rms:.3f}px")

    return cam_ext, total_rms


def main():
    parser = argparse.ArgumentParser(description="三目标定 v2（简化BA）")
    parser.add_argument("--images", required=True,
                        help="图片目录（相对于 calibration/）")
    parser.add_argument("--reference", default="DA8199285")
    parser.add_argument("--serials", nargs="+",
                        default=["DA8199285", "DA8199402", "DA8199243"])
    parser.add_argument("--inner-cols", type=int, default=8)
    parser.add_argument("--inner-rows", type=int, default=11)
    parser.add_argument("--square-size", type=float, default=45.0)
    parser.add_argument("--output", default="src/config/multi_calib.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    project_root = Path(__file__).resolve().parent.parent
    calib_root = Path(__file__).resolve().parent
    image_dir = calib_root / args.images
    output_path = project_root / args.output
    cache_path = image_dir / "corner_detections.json"

    if not cache_path.exists():
        print(f"ERROR: {cache_path} 不存在，请先运行 detect_corners.py")
        sys.exit(1)

    obj_pts = make_obj_pts(args.inner_cols, args.inner_rows, args.square_size)
    ref = args.reference
    slaves = [s for s in args.serials if s != ref]

    print("=" * 60)
    print("       三目标定 v2（简化 BA）")
    print("=" * 60)
    print(f"  参考: {ref}")
    print(f"  从机: {slaves}")
    print(f"  棋盘: {args.inner_cols}x{args.inner_rows}, {args.square_size}mm")

    # 1. 加载角点
    print(f"\n[1/4] 加载角点...")
    detections, image_sizes, _ = load_corners(
        cache_path, args.serials, 1, 500)
    for sn in args.serials:
        n = sum(1 for d in detections.values() if sn in d)
        print(f"  {sn}: {n} 帧 ({image_sizes[sn]})")

    # 2. 内参
    print(f"\n[2/4] calibrateCamera 内参...")
    K, D, ref_rvecs, ref_tvecs = calibrate_intrinsics(
        detections, image_sizes, args.serials, obj_pts, max_frames=80)

    # 3. 两两 stereoCalibrate
    print(f"\n[3/4] stereoCalibrate 两两校准...")
    stereo_results = pairwise_stereo(
        detections, K, D, image_sizes, args.serials, obj_pts, max_frames=60)

    # 4. 简化 BA
    print(f"\n[4/4] 简化 BA（仅优化 2 从相机外参）...")
    cam_ext, total_rms = simple_ba(
        detections, ref, slaves, K, D,
        ref_rvecs[ref], ref_tvecs[ref], obj_pts, max_frames=50)

    # ── 保存 ──
    print(f"\n{'='*60}")
    print(f"  最终结果")
    print(f"{'='*60}")
    print(f"  BA total RMS: {total_rms:.3f}px")

    cameras = {}
    for sn in args.serials:
        cam = {
            "K": K[sn].tolist(),
            "D": D[sn].tolist(),
            "image_size": list(image_sizes[sn]),
        }
        if sn == ref:
            cam["R_to_ref"] = np.eye(3).tolist()
            cam["t_to_ref"] = np.zeros((3, 1)).tolist()
        else:
            R, t = cam_ext[sn]
            cam["R_to_ref"] = R.tolist()
            cam["t_to_ref"] = t.tolist()
        cameras[sn] = cam

    # 配对基线
    print(f"\n  配对基线:")
    for sn in slaves:
        R, t = cam_ext[sn]
        dist = np.linalg.norm(t)
        print(f"    {ref} <-> {sn}: {dist:.0f}mm = {dist/1000:.2f}m")
    for (s1, s2), (R, T, rms, bl) in stereo_results.items():
        if s1 != ref and s2 != ref:
            print(f"    {s1} <-> {s2}: stereo={bl:.0f}mm (BA需从各自到ref推算)")

    data = {
        "reference_serial": ref,
        "cameras": cameras,
        "diagnostics": {
            "total_rms": float(total_rms),
            "num_images": len([idx for idx in detections
                               if all(sn in detections[idx] for sn in args.serials)]),
            "stereo_baselines": {
                f"{s1}_{s2}": {"rms": float(rms), "baseline_mm": float(bl)}
                for (s1, s2), (_, _, rms, bl) in stereo_results.items()
            },
        },
        "board": {
            "inner_cols": args.inner_cols,
            "inner_rows": args.inner_rows,
            "square_size": args.square_size,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"\n  已保存: {output_path}")

    # 同时备份到 005 目录
    backup_path = image_dir / "multi_calib_v2.json"
    with open(backup_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"  备份: {backup_path}")


if __name__ == "__main__":
    main()
