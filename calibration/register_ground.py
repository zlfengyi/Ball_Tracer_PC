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
from datetime import datetime


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


def _resolve_from_project(project_root: Path, raw_path: str) -> Path:
    """Resolve a user-supplied path against project root first, then calibration/."""
    path = Path(raw_path)
    if path.is_absolute():
        return path
    calib_root = project_root / "calibration"
    candidates = [project_root / path, calib_root / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if path.parts and path.parts[0] in {"data", "src", "calibration"}:
        return project_root / path
    return calib_root / path


def _rel_to_project(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve())).replace("\\", "/")
    except Exception:
        return str(path.resolve())


def _rt_to_mat(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def _mat_to_rt(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return T[:3, :3].copy(), T[:3, 3:4].copy()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="多目相机大地坐标系注册")
    parser.add_argument("--ground-index", type=int, default=1,
                        help="地面图片编号 (默认: 1, 即 ground_001)")
    parser.add_argument("--images", type=str, default="images",
                        help="图片目录 (相对于 calibration/，默认: images)")
    parser.add_argument("--calib", type=str, default="src/config/multi_calib.json",
                        help="输入标定配置路径 (相对于项目根目录，默认: src/config/multi_calib.json)")
    parser.add_argument("--output", type=str, default=None,
                        help="输出标定配置路径 (默认: 覆盖 --calib)")
    args = parser.parse_args()

    ground_idx = f"{args.ground_index:03d}"

    project_root = Path(__file__).resolve().parent.parent
    calib_path = _resolve_from_project(project_root, args.calib)
    output_path = calib_path if args.output is None else _resolve_from_project(project_root, args.output)
    ann_dir = _resolve_from_project(project_root, args.images)

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
    world_pts = {}  # {sn: Nx3}
    img_pts = {}    # {sn: Nx2}
    annotation_paths = {}

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
        annotation_paths[sn] = ann_path
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

    annotated_serials = [sn for sn in serials if sn in init_rvecs]
    if not annotated_serials:
        print("ERROR: 所有相机 solvePnP 均失败")
        return

    # 也尝试从其他相机推导（利用相对外参）
    print(f"\n--- Cross-camera inference ---")
    for sn_from in annotated_serials:
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

    annotated_serials = [sn for sn in serials if sn in init_rvecs and sn in world_mm]
    if not annotated_serials:
        print("ERROR: 无可用的地面标注相机初始化结果")
        return
    anchor_serial = ref_serial if ref_serial in annotated_serials else annotated_serials[0]

    # --- 联合优化 ---
    # 仅优化 1 台已标注相机的 6 DOF 世界位姿，其他相机由固定相对外参推导。
    print(f"\n--- Joint optimization (anchor={anchor_serial}, ref={ref_serial}, 6 params, "
          f"{sum(len(img_pts[sn]) for sn in annotated_serials)} points) ---")

    def derive_pose_from_anchor(R_anchor_world, t_anchor_world, target_sn):
        """从 anchor 相机世界位姿 + 固定相对外参，推导任意目标相机世界位姿。"""
        if target_sn == anchor_serial:
            return R_anchor_world, t_anchor_world
        T_anchor_world = _rt_to_mat(R_anchor_world, t_anchor_world)
        T_anchor_ref = _rt_to_mat(R_to_ref[anchor_serial], t_to_ref[anchor_serial])
        T_ref_world = np.linalg.inv(T_anchor_ref) @ T_anchor_world
        T_target_ref = _rt_to_mat(R_to_ref[target_sn], t_to_ref[target_sn])
        T_target_world = T_target_ref @ T_ref_world
        return _mat_to_rt(T_target_world)

    def make_residuals():
        def fn(params):
            rv_anchor = params[:3]
            tv_anchor = params[3:6].reshape(3, 1)
            R_anchor, _ = cv2.Rodrigues(rv_anchor)
            residuals = []
            for sn in annotated_serials:
                if sn == anchor_serial:
                    rv, tv = rv_anchor, tv_anchor
                else:
                    R_s, t_s = derive_pose_from_anchor(R_anchor, tv_anchor, sn)
                    rv_s, _ = cv2.Rodrigues(R_s)
                    rv, tv = rv_s.ravel(), t_s
                proj, _ = cv2.projectPoints(world_mm[sn], rv, tv, K[sn], D[sn])
                res = (proj.reshape(-1, 2) - img_pts[sn]).flatten()
                residuals.append(res)
            return np.concatenate(residuals)
        return fn

    # 初始参数：anchor 相机的 solvePnP 结果
    x0 = np.concatenate([init_rvecs[anchor_serial], init_tvecs[anchor_serial]])

    residuals_fn = make_residuals()
    result = least_squares(residuals_fn, x0, method='lm')

    # --- 提取结果 ---
    print(f"\n=== Results ===")
    rv_anchor = result.x[:3]
    tv_anchor = result.x[3:6].reshape(3, 1)
    R_anchor_world, _ = cv2.Rodrigues(rv_anchor)

    total_errs = []
    cam_poses = {}  # {sn: (R, t, pos)}
    per_camera_rms = {}

    for sn in serials:
        R, t = derive_pose_from_anchor(R_anchor_world, tv_anchor, sn)
        rv_sn, _ = cv2.Rodrigues(R)
        pos = (-R.T @ t).flatten()
        cam_poses[sn] = (R, t, pos)

        if sn in world_mm:
            rms = reproj_rms(world_mm[sn], img_pts[sn], rv_sn.ravel(), t, K[sn], D[sn])
            total_errs.append(rms)
            per_camera_rms[sn] = float(rms)
            rms_text = f"{rms:.2f}px"
        else:
            rms_text = "n/a (no ground points)"

        print(f"  {sn}: RMS={rms_text}  pos=[{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]mm  "
              f"height={pos[2]:.0f}mm={pos[2]/1000:.2f}m")

        # 更新 calib
        cam_data[sn]["R_world"] = R.tolist()
        cam_data[sn]["t_world"] = t.reshape(3, 1).tolist()
        cam_data[sn]["pos_world"] = pos.reshape(3, 1).tolist()

    err_total = np.sqrt(np.mean(np.array(total_errs) ** 2))
    print(f"\n  Total RMS: {err_total:.2f} px")

    # 配对基线检查
    print(f"\n--- Pairwise baselines ---")
    registered_list = list(cam_poses.keys())
    for i, sn_i in enumerate(registered_list):
        pos_i = cam_poses[sn_i][2]
        for j, sn_j in enumerate(registered_list):
            if j <= i:
                continue
            pos_j = cam_poses[sn_j][2]
            dist = np.linalg.norm(pos_i - pos_j)
            print(f"  {sn_i} <-> {sn_j}: {dist:.0f}mm = {dist/1000:.2f}m")

    # 逐点重投影
    print(f"\n--- Per-point reprojection ---")
    for sn in annotated_serials:
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
    calib["diagnostics"]["ground_registration"] = {
        "registered_at": datetime.now().isoformat(timespec="seconds"),
        "ground_index": int(args.ground_index),
        "images_dir": _rel_to_project(ann_dir, project_root),
        "reference_serial": ref_serial,
        "anchor_serial": anchor_serial,
        "num_cameras_annotated": len(annotated_serials),
        "num_cameras_registered": len(serials),
        "missing_ground_annotations": [sn for sn in serials if sn not in annotation_paths],
        "total_rms_px": float(err_total),
        "per_camera_rms_px": per_camera_rms,
    }
    calib["config_written_at"] = datetime.now().isoformat(timespec="seconds")

    sources = calib.setdefault("sources", {})
    ground_sources = sources.setdefault("ground_registration", {})
    ground_sources.update({
        "images_dir": _rel_to_project(ann_dir, project_root),
        "ground_index": int(args.ground_index),
        "annotation_files": {
            sn: _rel_to_project(path, project_root) for sn, path in annotation_paths.items()
        },
        "registered_at": datetime.now().isoformat(timespec="seconds"),
        "input_calib": _rel_to_project(calib_path, project_root),
    })

    for sn in serials:
        cam_sources = cam_data[sn].setdefault("sources", {})
        if sn in annotation_paths:
            cam_sources["ground_annotation"] = _rel_to_project(annotation_paths[sn], project_root)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=4, ensure_ascii=False)

    print(f"\nUpdated: {output_path}")


if __name__ == "__main__":
    main()
