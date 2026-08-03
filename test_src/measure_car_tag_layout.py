# -*- coding: utf-8 -*-
"""车载 AprilTag 布局标定：实测各 tag 相对车体系的中心与完整安装旋转。

用途：
  换场地无关（tag 布局绑定在车上）；每次重新粘贴/移动车上的 tag 后运行一次，
  把输出的 vehicle_reference.apriltags 配置块粘回
  src/config/arm_poe_racket_center.json。

原理：
  1. 采集多帧同步图像，检测所有已配置 tag 的四角（逐帧平均降噪）
  2. 每块 tag 做 6 自由度刚体拟合（中心 + 完整旋转），
     已知边长锚定尺度，无竖直假设 → 得到世界系 tag 位姿
  3. 锚定约定：anchor tag（默认 id0）的车体系中心沿用当前配置值（人工实测，
     被视觉验证过）；车 yaw 由两 tag 中心基线的世界方位与配置基线方位差给出
  4. 其余 tag 的车体系中心/旋转由测得的世界位姿反解

注意：
  1. 中心坐标 center_car_m 以手工卷尺实测为准（量到黑方块中心、从地面起）——
     本脚本输出的中心值继承相机外参误差（2026-08-03 实证：414 外参漂移使
     视觉中心偏 1~2cm，替换回手工值后重投影 2.21→1.18px），只用于交叉验证。
  2. 本脚本的主要产出是安装旋转 R_tag_car（含倒贴/微倾角），这个手工量不了。
  3. 车 yaw 的绝对基准依赖配置里两 tag 中心连线的方位角，锚定 tag 配置错则
     全体带偏。

用法（18 楼场地，相机正装需关翻转）：
  $env:BALL_TRACER_CAMERA_REVERSE_180='0'
  python test_src/measure_car_tag_layout.py --field 18
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src import SyncCapture, frame_to_numpy
from src.car_localizer import CarLocalizer


def nearest_rotation(m: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(m)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def rotvec_matrix(w: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3)
    k = w / theta
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0],
    ])
    return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)


class TagPoseFitter:
    """单块 tag 的 6 自由度多目刚体拟合（边长锚定尺度）。"""

    def __init__(self, loc: CarLocalizer, corners_printed_mm: np.ndarray):
        self._loc = loc
        self._printed = corners_printed_mm  # (4,2)

    def world_corners(self, center: np.ndarray, R: np.ndarray) -> np.ndarray:
        return center[None, :] + self._printed @ R[:, :2].T

    def residual(self, center, R, dets):
        wc = self.world_corners(center, R)
        loc = self._loc
        res = []
        for sn, corners_px in dets.items():
            proj, _ = cv2.projectPoints(
                wc, loc._rvec[sn], loc._t[sn], loc._K[sn], loc._D[sn]
            )
            res.append((proj.reshape(4, 2) - corners_px).reshape(-1))
        return np.concatenate(res)

    def fit(self, dets: dict[str, np.ndarray], center0, R0):
        center = np.array(center0, dtype=np.float64)
        R = np.array(R0, dtype=np.float64)
        eps_t, eps_w = 0.5, 2e-4
        for _ in range(40):
            r = self.residual(center, R, dets)
            J = np.empty((r.size, 6))
            for k in range(3):
                d = np.zeros(3)
                d[k] = eps_t
                J[:, k] = (
                    self.residual(center + d, R, dets)
                    - self.residual(center - d, R, dets)
                ) / (2 * eps_t)
            for k in range(3):
                w = np.zeros(3)
                w[k] = eps_w
                J[:, 3 + k] = (
                    self.residual(center, rotvec_matrix(w) @ R, dets)
                    - self.residual(center, rotvec_matrix(-w) @ R, dets)
                ) / (2 * eps_w)
            step, *_ = np.linalg.lstsq(J, -r, rcond=None)
            step[:3] = np.clip(step[:3], -300, 300)
            step[3:] = np.clip(step[3:], -0.3, 0.3)
            center = center + step[:3]
            R = nearest_rotation(rotvec_matrix(step[3:]) @ R)
            if np.linalg.norm(step[:3]) < 0.02 and np.linalg.norm(step[3:]) < 1e-6:
                break
        r = self.residual(center, R, dets)
        rms = float(np.sqrt(np.mean(r ** 2)))
        per_view = {
            sn: float(np.sqrt(np.mean(r[i * 8:(i + 1) * 8] ** 2)))
            for i, sn in enumerate(dets)
        }
        return center, R, rms, per_view


def describe_rotation(R: np.ndarray) -> str:
    e1, e2, n = R[:, 0], R[:, 1], R[:, 2]
    az = math.degrees(math.atan2(n[1], n[0]))
    tilt = math.degrees(math.asin(max(-1.0, min(1.0, float(n[2])))))
    # inplane: 正贴时印面上 e2 ≈ +z（≈0°）；倒贴时 e2 ≈ -z（≈180°）
    inplane = math.degrees(math.atan2(float(-e1[2]), float(e2[2])))
    return f"法线方位 {az:+7.2f}°  仰角 {tilt:+6.2f}°  面内旋转 {inplane:+7.2f}°"


def main():
    ap = argparse.ArgumentParser(description="车载 AprilTag 布局标定")
    ap.add_argument("--field", default="", help="场地后缀（如 18）；空 = 默认场地")
    ap.add_argument("--frames", type=int, default=12, help="采集帧数")
    ap.add_argument("--anchor", type=int, default=0, help="锚定 tag id（配置中心不动）")
    args = ap.parse_args()

    suffix = f"_{args.field}" if args.field else ""
    calib = str(_PROJECT_ROOT / "src" / "config" / f"four_camera_calib{suffix}.json")
    camcfg = str(_PROJECT_ROOT / "src" / "config" / f"camera{suffix}.json")

    loc = CarLocalizer(calib_config_path=calib)
    printed = loc._corners_printed_mm
    layout_cfg_mm = {
        tag_id: spec.center_car_mm.copy() for tag_id, spec in loc._tags.items()
    }
    print(f"标定: {calib}")
    print(f"配置 tag: {sorted(layout_cfg_mm)}  锚定: id{args.anchor}")

    # ── 采集并平均角点 ──
    stacks: dict[int, dict[str, list[np.ndarray]]] = {}
    with SyncCapture.from_config(camcfg) as cap:
        time.sleep(1.0)
        n = 0
        t_end = time.perf_counter() + 3.0 * args.frames
        while n < args.frames and time.perf_counter() < t_end:
            frames = cap.get_frames(timeout_s=1.0)
            if frames is None:
                continue
            images = {
                sn: frame_to_numpy(f)
                for sn, f in frames.items() if sn in loc.serials
            }
            got = False
            for sn, img in images.items():
                for d in loc.detect(img):
                    if d.tag_id not in layout_cfg_mm:
                        continue
                    stacks.setdefault(d.tag_id, {}).setdefault(sn, []).append(
                        d.corners.copy()
                    )
                    got = True
            n += 1 if got else 0
    print(f"采集 {n} 帧")

    # ── 每块 tag 6DoF 拟合 ──
    fitter = TagPoseFitter(loc, printed)
    world: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for tag_id in sorted(stacks):
        cam_stacks = {
            sn: v for sn, v in stacks[tag_id].items() if len(v) >= max(3, n // 2)
        }
        if len(cam_stacks) < 2:
            print(f"id{tag_id}: 稳定相机不足 2 台，跳过")
            continue
        dets = {sn: np.mean(np.stack(v), axis=0) for sn, v in cam_stacks.items()}
        jitter = {
            sn[-3:]: round(float(np.std(np.stack(v), axis=0).mean()), 2)
            for sn, v in cam_stacks.items()
        }
        sns = list(dets)
        center0 = loc._triangulate_point(sns, {
            sn: (float(dets[sn][:, 0].mean()), float(dets[sn][:, 1].mean()))
            for sn in sns
        })
        cs = [
            loc._triangulate_point(sns, {
                sn: (float(dets[sn][k, 0]), float(dets[sn][k, 1])) for sn in sns
            })
            for k in range(4)
        ]
        e1 = (cs[1] - cs[0]) + (cs[2] - cs[3])
        e2 = (cs[0] - cs[3]) + (cs[1] - cs[2])
        R0 = nearest_rotation(np.column_stack([
            e1 / np.linalg.norm(e1),
            e2 / np.linalg.norm(e2),
            np.cross(e1, e2) / np.linalg.norm(np.cross(e1, e2)),
        ]))
        center, R, rms, per_view = fitter.fit(dets, center0, R0)
        world[tag_id] = (center, R)
        print(f"\nid{tag_id}  相机 {sorted(s[-3:] for s in dets)}  角点抖动px {jitter}")
        print(f"  世界中心 mm ({center[0]:8.1f}, {center[1]:8.1f}, {center[2]:7.1f})"
              f"  重投影RMS {rms:.2f}px  各视图 { {k[-3:]: round(v, 2) for k, v in per_view.items()} }")
        print(f"  世界系: {describe_rotation(R)}")

    if args.anchor not in world:
        print(f"\n锚定 tag id{args.anchor} 未测到，无法反解车体系布局。")
        return 1
    others = [tag_id for tag_id in world if tag_id != args.anchor]
    if not others:
        print("\n只测到锚定 tag，仅能校验其配置中心，无法解车 yaw。")
        return 1

    # ── 车 yaw：测得基线方位 vs 配置基线方位（用第一块非锚定 tag）──
    tag_b = others[0]
    w_a, w_b = world[args.anchor][0], world[tag_b][0]
    c_a, c_b = layout_cfg_mm[args.anchor], layout_cfg_mm[tag_b]
    yaw = (
        math.atan2(w_b[1] - w_a[1], w_b[0] - w_a[0])
        - math.atan2(c_b[1] - c_a[1], c_b[0] - c_a[0])
    )
    yaw = math.atan2(math.sin(yaw), math.cos(yaw))
    c, s = math.cos(yaw), math.sin(yaw)
    Rz_inv = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    t_xy = w_a[:2] - np.array([
        c * c_a[0] - s * c_a[1], s * c_a[0] + c * c_a[1],
    ])
    print(f"\n车位姿: x={t_xy[0]:.1f} y={t_xy[1]:.1f} mm  yaw={math.degrees(yaw):+.2f}°"
          f"  （锚定 id{args.anchor} 配置中心 + id{tag_b} 基线方位）")

    # ── 反解各 tag 车体系布局并输出配置块 ──
    apriltags_cfg = {}
    for tag_id in sorted(world):
        center_w, R_w = world[tag_id]
        center_car = Rz_inv[:2, :2] @ (center_w[:2] - t_xy)
        center_car_m = [
            round(float(center_car[0]) / 1000.0, 4),
            round(float(center_car[1]) / 1000.0, 4),
            round(float(center_w[2]) / 1000.0, 4),
        ]
        if tag_id == args.anchor:
            center_car_m = [round(float(v) / 1000.0, 4) for v in c_a]
        R_car = nearest_rotation(Rz_inv @ R_w)
        delta = np.array(center_car_m) * 1000.0 - layout_cfg_mm[tag_id]
        print(f"id{tag_id}  车体系中心 m {center_car_m}"
              f"  vs 配置差 mm ({delta[0]:+.1f}, {delta[1]:+.1f}, {delta[2]:+.1f})")
        print(f"       车体系: {describe_rotation(R_car)}")
        apriltags_cfg[str(tag_id)] = {
            "center_car_m": center_car_m,
            "R_tag_car": [[round(float(v), 6) for v in row] for row in R_car],
            "orientation_note": describe_rotation(R_car).strip(),
        }

    print("\n──── 粘贴到 vehicle_reference.apriltags ────")
    print(json.dumps(apriltags_cfg, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
