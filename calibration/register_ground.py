# -*- coding: utf-8 -*-
"""
多目相机大地坐标系注册：基于人工标注的地面点，将相机外参校准到世界坐标系。

优化每台相机的独立 6 DOF 世界位姿，同时添加配对一致性软约束
（相邻相机间的相对位姿应与 BA 标定的 R_to_ref/t_to_ref 一致）。

用法：
  python -m calibration.register_ground

输入：
  src/config/multi_calib.json                              — 多目标定结果
  calibration/images/{serial}/ground_001_annotations.json  — 各相机标注

输出：
  更新 multi_calib.json 中每台相机的 R_world, t_world, pos_world
"""

import json
import numpy as np
import cv2
from scipy.optimize import least_squares
from pathlib import Path


def load_annotations(path):
    """加载标注 JSON，返回 world_pts (Nx3) 和 img_pts (Nx2)"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    world_pts = []
    img_pts = []
    for key in sorted(data.keys(), key=int):
        world_xyz, pixel = data[key]
        world_pts.append(world_xyz)
        img_pts.append(pixel)
    return np.array(world_pts, dtype=np.float64), np.array(img_pts, dtype=np.float64)


def reproj_rms(world, img_obs, rvec, tvec, K, D):
    """计算重投影 RMS"""
    proj, _ = cv2.projectPoints(world, rvec, tvec, K, D)
    return np.sqrt(np.mean((proj.reshape(-1, 2) - img_obs) ** 2))


def try_solvepnp(world, img, K, D, method, name):
    """尝试一种 solvePnP 方法"""
    try:
        ok, rv, tv = cv2.solvePnP(world, img, K, D, flags=method)
        if not ok:
            return None
        rms = reproj_rms(world, img, rv.flatten(), tv.flatten(), K, D)
        print(f"    {name:20s}: RMS = {rms:.2f} px")
        return rv.flatten(), tv.flatten(), rms
    except Exception as e:
        print(f"    {name:20s}: failed ({e})")
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="多目相机大地坐标系注册")
    parser.add_argument("--ground-index", type=int, default=1,
                        help="地面图片编号 (默认: 1, 即 ground_001)")
    parser.add_argument("--images", type=str, default="images",
                        help="图片目录 (相对于 calibration/，默认: images)")
    args = parser.parse_args()

    ground_idx = f"{args.ground_index:03d}"

    project_root = Path(__file__).resolve().parent.parent
    calib_root = Path(__file__).resolve().parent
    calib_path = project_root / "src" / "config" / "multi_calib.json"

    # --- 加载标定 ---
    with open(calib_path, encoding="utf-8") as f:
        calib = json.load(f)

    ref_serial = calib["reference_serial"]
    cam_data = calib["cameras"]
    serials = list(cam_data.keys())
    n_cams = len(serials)

    # 解析内参和相对外参
    K = {}
    D = {}
    R_to_ref = {}
    t_to_ref = {}
    for sn, cd in cam_data.items():
        K[sn] = np.array(cd["K"], dtype=np.float64).reshape(3, 3)
        D[sn] = np.array(cd["D"], dtype=np.float64).ravel()
        R_to_ref[sn] = np.array(cd["R_to_ref"], dtype=np.float64).reshape(3, 3)
        t_to_ref[sn] = np.array(cd["t_to_ref"], dtype=np.float64).reshape(3, 1)

    # --- 加载各相机标注 ---
    ann_dir = calib_root / args.images
    world_pts = {}  # {sn: Nx3}
    img_pts = {}    # {sn: Nx2}

    print(f"=== Ground Registration ({n_cams} cameras) ===")
    print(f"  Reference: {ref_serial}")
    print(f"  Cameras: {serials}")
    print()

    for sn in serials:
        ann_path = ann_dir / sn / f"ground_{ground_idx}_annotations.json"
        if not ann_path.exists():
            print(f"  WARNING: {sn} 标注文件不存在: {ann_path}")
            continue
        wp, ip = load_annotations(ann_path)
        world_pts[sn] = wp
        img_pts[sn] = ip
        print(f"  {sn}: {len(wp)} 标注点")

    if not world_pts:
        print("ERROR: 无可用标注")
        return

    # 验证世界坐标一致性
    ref_world = None
    for sn in serials:
        if sn not in world_pts:
            continue
        if ref_world is None:
            ref_world = world_pts[sn]
        else:
            if not np.allclose(world_pts[sn], ref_world):
                print(f"  WARNING: {sn} 的世界坐标与其他相机不一致")

    # 标注坐标：米 → 毫米
    world_mm = {}
    for sn in world_pts:
        world_mm[sn] = world_pts[sn] * 1000.0

    # --- 每台相机独立 solvePnP 初始化 ---
    print(f"\n--- Per-camera solvePnP ---")
    methods = [
        (cv2.SOLVEPNP_SQPNP, "SQPNP"),
        (cv2.SOLVEPNP_ITERATIVE, "ITERATIVE"),
    ]

    init_rvecs = {}
    init_tvecs = {}

    for sn in serials:
        if sn not in world_mm:
            continue
        print(f"\n  {sn}:")
        best = None
        for method, name in methods:
            res = try_solvepnp(world_mm[sn], img_pts[sn], K[sn], D[sn], method, name)
            if res is not None:
                if best is None or res[2] < best[2]:
                    best = res
        if best:
            init_rvecs[sn] = best[0]
            init_tvecs[sn] = best[1]

    available = [sn for sn in serials if sn in init_rvecs]
    if not available:
        print("ERROR: 所有相机 solvePnP 均失败")
        return

    # 也尝试从其他相机推导（利用相对外参）
    print(f"\n--- Cross-camera inference ---")
    for sn_from in available:
        for sn_to in serials:
            if sn_to in init_rvecs or sn_to not in world_mm:
                continue
            # T_to_world = T_to_ref @ T_ref_world
            # T_from 在世界坐标下: R_from, t_from
            R_from, _ = cv2.Rodrigues(init_rvecs[sn_from])
            t_from = init_tvecs[sn_from].reshape(3, 1)
            # T_to_ref @ T_from^-1 = T_to_world @ T_from_world^-1
            # => R_to = R_to_ref @ R_ref_from^-1 @ R_from
            # 但更简单: T_to = T_to_ref @ inv(T_from_ref) @ T_from
            R_from_ref = R_to_ref[sn_from]
            t_from_ref = t_to_ref[sn_from]
            R_to_ref_cam = R_to_ref[sn_to]
            t_to_ref_cam = t_to_ref[sn_to]

            # T_ref_world = inv(T_from_ref) @ T_from_world
            R_ref = R_from_ref.T @ R_from
            t_ref = R_from_ref.T @ (t_from - t_from_ref)

            R_to = R_to_ref_cam @ R_ref
            t_to = R_to_ref_cam @ t_ref + t_to_ref_cam

            rv_to, _ = cv2.Rodrigues(R_to)
            rms = reproj_rms(world_mm[sn_to], img_pts[sn_to],
                             rv_to.flatten(), t_to.flatten(), K[sn_to], D[sn_to])
            print(f"  {sn_from} → {sn_to}: RMS = {rms:.2f} px")
            if sn_to not in init_rvecs or rms < reproj_rms(
                    world_mm[sn_to], img_pts[sn_to],
                    init_rvecs[sn_to], init_tvecs[sn_to], K[sn_to], D[sn_to]):
                init_rvecs[sn_to] = rv_to.flatten()
                init_tvecs[sn_to] = t_to.flatten()

    available = [sn for sn in serials if sn in init_rvecs]

    # --- 联合优化 ---
    # 仅优化参考相机的 6 DOF 世界位姿，从相机位姿由 R_to_ref/t_to_ref 推导
    print(f"\n--- Joint optimization (ref={ref_serial}, 6 params, "
          f"{sum(len(img_pts[sn]) for sn in available)} points) ---")

    def derive_slave_pose(R_ref_world, t_ref_world, sn):
        """从参考相机世界位姿 + 固定相对外参，推导从相机世界位姿。
        T_slave_world = T_slave_ref @ T_ref_world
        """
        R_s2r = R_to_ref[sn]
        t_s2r = t_to_ref[sn].reshape(3, 1)
        R_slave = R_s2r @ R_ref_world
        t_slave = R_s2r @ t_ref_world + t_s2r
        return R_slave, t_slave

    def make_residuals():
        def fn(params):
            rv_ref = params[:3]
            tv_ref = params[3:6].reshape(3, 1)
            R_ref, _ = cv2.Rodrigues(rv_ref)
            residuals = []
            for sn in available:
                if sn == ref_serial:
                    rv, tv = rv_ref, tv_ref
                else:
                    R_s, t_s = derive_slave_pose(R_ref, tv_ref, sn)
                    rv_s, _ = cv2.Rodrigues(R_s)
                    rv, tv = rv_s.ravel(), t_s
                proj, _ = cv2.projectPoints(world_mm[sn], rv, tv, K[sn], D[sn])
                res = (proj.reshape(-1, 2) - img_pts[sn]).flatten()
                residuals.append(res)
            return np.concatenate(residuals)
        return fn

    # 初始参数：参考相机的 solvePnP 结果
    if ref_serial in init_rvecs:
        x0 = np.concatenate([init_rvecs[ref_serial], init_tvecs[ref_serial]])
    else:
        print("ERROR: 参考相机 solvePnP 失败")
        return

    residuals_fn = make_residuals()
    result = least_squares(residuals_fn, x0, method='lm')

    # --- 提取结果 ---
    print(f"\n=== Results ===")
    rv_ref = result.x[:3]
    tv_ref = result.x[3:6].reshape(3, 1)
    R_ref_world, _ = cv2.Rodrigues(rv_ref)

    total_errs = []
    cam_poses = {}  # {sn: (R, t, pos)}

    for sn in available:
        if sn == ref_serial:
            R, t = R_ref_world, tv_ref
        else:
            R, t = derive_slave_pose(R_ref_world, tv_ref, sn)
        rv_sn, _ = cv2.Rodrigues(R)
        pos = (-R.T @ t).flatten()
        cam_poses[sn] = (R, t, pos)

        rms = reproj_rms(world_mm[sn], img_pts[sn], rv_sn.ravel(), t, K[sn], D[sn])
        total_errs.append(rms)

        print(f"  {sn}: RMS={rms:.2f}px  pos=[{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]mm  "
              f"height={pos[2]:.0f}mm={pos[2]/1000:.2f}m")

        # 更新 calib
        cam_data[sn]["R_world"] = R.tolist()
        cam_data[sn]["t_world"] = t.reshape(3, 1).tolist()
        cam_data[sn]["pos_world"] = pos.reshape(3, 1).tolist()

    err_total = np.sqrt(np.mean(np.array(total_errs) ** 2))
    print(f"\n  Total RMS: {err_total:.2f} px")

    # 配对基线检查
    print(f"\n--- Pairwise baselines ---")
    avail_list = list(cam_poses.keys())
    for i, sn_i in enumerate(avail_list):
        pos_i = cam_poses[sn_i][2]
        for j, sn_j in enumerate(avail_list):
            if j <= i:
                continue
            pos_j = cam_poses[sn_j][2]
            dist = np.linalg.norm(pos_i - pos_j)
            print(f"  {sn_i} <-> {sn_j}: {dist:.0f}mm = {dist/1000:.2f}m")

    # 逐点重投影
    print(f"\n--- Per-point reprojection ---")
    for sn in available:
        R, t, _ = cam_poses[sn]
        rv_sn, _ = cv2.Rodrigues(R)
        proj, _ = cv2.projectPoints(world_mm[sn], rv_sn.ravel(), t, K[sn], D[sn])
        proj = proj.reshape(-1, 2)
        print(f"  {sn}:")
        for k in range(len(img_pts[sn])):
            e = np.linalg.norm(proj[k] - img_pts[sn][k])
            wx, wy, wz = world_pts[sn][k]
            print(f"    #{k+1} ({wx:.1f},{wy:.2f},{wz})m: "
                  f"err={e:.2f}px  proj=[{proj[k,0]:.0f},{proj[k,1]:.0f}] "
                  f"vs [{img_pts[sn][k,0]:.0f},{img_pts[sn][k,1]:.0f}]")

    # --- 保存 ---
    if "diagnostics" not in calib:
        calib["diagnostics"] = {}
    calib["diagnostics"]["ground_reproj_error"] = float(err_total)

    with open(calib_path, "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=4, ensure_ascii=False)

    print(f"\nUpdated: {calib_path}")


if __name__ == "__main__":
    main()
