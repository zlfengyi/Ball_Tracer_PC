# -*- coding: utf-8 -*-
"""
从录制视频里测量**臂上球拍**的 3D 拍心，产出报告用的 racket_observations 侧车文件。

为什么不直接用 src/racket_localizer.RacketLocalizer.locate：
  1. 它取 detections[0]（每台相机置信度最高的框），场里有两把拍（臂上 + 球员），
     实测会整段锁到 12m 外球员那把（0816 复现，三角化出 y=15.8m）。这里改成
     **把车心投到每台相机、取最近的框**，再用 3D reach 门（离车心 ≤1.8m）兜底。
  2. 它取关键点 0-3 的几何中心。0816 刚体自检发现 kp1/kp3 在拍框环上滑
     （两点距离 182~308mm 乱跳），kp0/kp2 稳定在 325±8mm。这里只用 **kp0/kp2 长轴对**，
     拍心=两端点各自三角化后的中点，并拿 |d02−325mm| 当逐帧质检门——比 SimCC 分数可靠得多。

质检（缺一不写，宁可少给也不给错）：
  - ≥3 台相机的 kp0/kp2 分数都过门（两台解不出相机间一致性，无法自证）
  - |d02 − 标称长轴| ≤ 容差（刚体自检）
  - 相机两两配对解出的拍心，x/z 极差 ≤ 门限（多视角一致性）
  - 拍心落在车心可达范围内

写出的每条观测都是**已经通过质检的**，所以报告端只管拟合、不必再判质量。
时间轴 = 帧的 exposure_pc（PC 绝对轴），与 observations/car_locs 同轴。

⚠ 相机外参必须是最新的：0817 同事修了 DB0260405 的外参后，球的留一相机重投影
从 8.14px 降到 4.29px，拍心的相机对分歧从 5~7cm 降到 1cm 级——外参不对时本工具
会因为过不了一致性门而大面积空白（这是设计意图，不是故障）。

用法：
  python test_src/racket_ht_measure.py --input tracker_output/<S>/<S>.json
  （默认写 <S>_racket.json，再交给 generate_curve3_html.py --racket-json）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.cv_linalg import smallest_right_singular_vector  # noqa: E402
from src.racket_localizer import RacketLocalizer  # noqa: E402
from test_src.annotate_video import extract_fullres_panels  # noqa: E402

D02_NOMINAL_MM = 325.0      # 拍头长轴（kp0-kp2）实测标称，多场中位 322~344
D02_TOL_MM = 45.0
REACH_XY_M = 1.8            # 臂上的拍永远在车心这个半径内；球员那把在 10m 外
Z_LO_M, Z_HI_M = 0.25, 2.30


def _panel_serials(cfg: dict) -> list[str]:
    return list(cfg.get("serials") or cfg.get("camera_serials") or [])


class RacketMeasurer:
    """kp0/kp2 长轴对 → 逐端点三角化 → 拍心中点，带刚体与多视角一致性门。"""

    def __init__(self, calib_path: str, serials: list[str], *, providers: list[str] | None = None,
                 bbox_conf: float = 0.20, score_min: float = 25.0,
                 anchor_max_px: float = 900.0):
        self._loc = RacketLocalizer(
            calib_config_path=calib_path,
            bbox_conf=bbox_conf,
            bbox_onnx_providers=providers,
            pose_providers=providers,
        )
        self.serials = [s for s in serials if s in self._loc.serials]
        self._P = {sn: np.array(self._loc._P[sn], dtype=float) for sn in self.serials}
        self._bbox_conf = bbox_conf
        self.score_min = score_min
        self.anchor_max_px = anchor_max_px

    def project(self, xyz_m, serial: str) -> np.ndarray:
        q = self._P[serial] @ np.append(np.asarray(xyz_m, dtype=float) * 1000.0, 1.0)
        return q[:2] / q[2]

    def triangulate(self, obs: dict[str, tuple[float, float]]) -> np.ndarray:
        rows = []
        for sn, (u, v) in obs.items():
            uu, vv = self._loc._undistort_point(u, v, self._loc._K[sn], self._loc._D[sn])
            P = self._P[sn]
            rows.append(uu * P[2] - P[0])
            rows.append(vv * P[2] - P[1])
        X = smallest_right_singular_vector(np.array(rows, dtype=float))
        return X[:3] / X[3] / 1000.0

    def detect_frame(self, panels: dict[str, np.ndarray], car_xy) -> dict:
        """每台相机选出臂上那把拍，返回 {serial: (kp0_xy, kp2_xy, score_min)}。"""
        anchor = np.array([car_xy[0], car_xy[1], 0.9], dtype=float)
        out = {}
        for sn in self.serials:
            panel = panels.get(sn)
            if panel is None:
                continue
            cands = self._loc._detector.detect(panel, conf=self._bbox_conf)
            if not cands:
                continue
            ax, ay = self.project(anchor, sn)
            best, best_d = None, float("inf")
            for det in cands:
                cx, cy = (det.x1 + det.x2) / 2.0, (det.y1 + det.y2) / 2.0
                d = float(np.hypot(cx - ax, cy - ay))
                if d < best_d:
                    best, best_d = det, d
            if best is None or best_d > self.anchor_max_px:
                continue
            bbox = (float(best.x1), float(best.y1), float(best.x2), float(best.y2))
            kp, sc = self._loc._pose_model(panel, bbox)
            s = float(min(sc[0], sc[2]))
            if s < self.score_min:
                continue
            out[sn] = (np.array(kp[0], dtype=float), np.array(kp[2], dtype=float), s)
        return out

    def solve(self, per_cam: dict, car_xy, *, min_cams: int = 3,
              pair_max_cm: float = 3.0) -> dict | None:
        if len(per_cam) < max(2, min_cams):
            return None
        p0 = self.triangulate({sn: v[0] for sn, v in per_cam.items()})
        p2 = self.triangulate({sn: v[1] for sn, v in per_cam.items()})
        centre = (p0 + p2) / 2.0
        d02_mm = float(np.linalg.norm(p0 - p2) * 1000.0)
        if abs(d02_mm - D02_NOMINAL_MM) > D02_TOL_MM:
            return None
        if np.hypot(centre[0] - car_xy[0], centre[1] - car_xy[1]) > REACH_XY_M:
            return None
        if not (Z_LO_M <= centre[2] <= Z_HI_M):
            return None
        subs = []
        for pair in combinations(per_cam, 2):
            a = self.triangulate({sn: per_cam[sn][0] for sn in pair})
            b = self.triangulate({sn: per_cam[sn][1] for sn in pair})
            subs.append((a + b) / 2.0)
        subs = np.array(subs)
        spread = [(float(subs[:, k].max() - subs[:, k].min()) * 100.0) for k in range(3)]
        if spread[0] > pair_max_cm or spread[2] > pair_max_cm:
            return None
        return {"x": float(centre[0]), "y": float(centre[1]), "z": float(centre[2]),
                "d02_mm": round(d02_mm, 1), "n_cam": len(per_cam),
                "pair_cm": [round(v, 2) for v in spread],
                "score_min": round(min(v[2] for v in per_cam.values()), 1)}


def _select_frames(frames: list[dict], cars: list[dict], near_m: float,
                   window_s: float) -> list[int]:
    """挑出「球接近车」前后一段的帧——真正的击球一定在里面，不依赖 y 反向判定。"""
    if not frames:
        return []
    t = np.array([f["elapsed_s"] for f in frames], dtype=float)
    ct = np.array([c["elapsed_s"] for c in cars], dtype=float) if cars else np.zeros(0)
    hot = []
    for i, f in enumerate(frames):
        b = f.get("ball3d")
        if not b or not cars:
            continue
        j = int(np.argmin(np.abs(ct - t[i])))
        if abs(float(b["y"]) - float(cars[j]["y"])) <= near_m:
            hot.append(t[i])
    if not hot:
        return list(range(len(frames)))
    hot = np.array(hot)
    keep = np.zeros(len(frames), dtype=bool)
    for h in hot:
        keep |= np.abs(t - h) <= window_s
    return [int(i) for i in np.where(keep)[0]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="tracker session json")
    ap.add_argument("--video", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--near-m", type=float, default=5.0, help="球离车多近算候选段")
    ap.add_argument("--window-s", type=float, default=0.70, help="候选段前后各取多久（须 ≥ 报告拟合窗 0.55s）")
    ap.add_argument("--min-cams", type=int, default=3)
    ap.add_argument("--pair-max-cm", type=float, default=3.0)
    ap.add_argument("--score-min", type=float, default=25.0,
                    help="SimCC 分数下限；真正的裁判是 d02 刚体门与相机对分歧门，"
                         "分数门放宽到 25 覆盖率翻倍而质量几乎不变（0817 实测）")
    ap.add_argument("--all-frames", action="store_true")
    ap.add_argument("--gpu", action="store_true", help="用 CUDA（默认 CPU，约 8× 慢）")
    args = ap.parse_args()

    json_path = Path(args.input)
    data = json.load(open(json_path, encoding="utf-8"))
    cfg = data.get("config", {})
    serials = _panel_serials(cfg)
    calib = cfg.get("calib_config_path")
    if not serials or not calib:
        print("[racket] session json 缺 serials/calib_config_path，跳过")
        return 1
    video = Path(args.video) if args.video else json_path.with_suffix(".mp4")
    if not video.exists():
        print(f"[racket] 找不到视频 {video}，跳过")
        return 1
    out_path = Path(args.output) if args.output else json_path.with_name(
        f"{json_path.stem}_racket.json")

    frames = [f for f in data.get("frames", []) if f.get("video_frame_idx") is not None]
    cars = [f["car_loc"] for f in data.get("frames", []) if f.get("car_loc")]
    if not frames or not cars:
        print("[racket] 无可用帧/车定位，跳过")
        return 1
    sel = (list(range(len(frames))) if args.all_frames
           else _select_frames(frames, cars, args.near_m, args.window_s))
    if not sel:
        print("[racket] 没有候选帧，跳过")
        return 1

    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if args.gpu
                 else ["CPUExecutionProvider"])
    m = RacketMeasurer(calib, serials, providers=providers, score_min=args.score_min)
    ct = np.array([c["elapsed_s"] for c in cars], dtype=float)

    cap = cv2.VideoCapture(str(video))
    # h264 随机 seek 每次 3~5 秒，顺序 grab() 才是唯一可用的读法
    want = {int(frames[i]["video_frame_idx"]): i for i in sel}
    order = sorted(want)
    pos = 0
    if order:
        cap.set(cv2.CAP_PROP_POS_FRAMES, order[0])
        pos = order[0]
    obs, t0, n_try = [], time.time(), 0
    for v in order:
        while pos < v:
            cap.grab()
            pos += 1
        ok, img = cap.read()
        pos += 1
        if not ok:
            continue
        i = want[v]
        fr = frames[i]
        car = cars[int(np.argmin(np.abs(ct - float(fr["elapsed_s"]))))]
        panels = extract_fullres_panels(img, serials)
        n_try += 1
        per_cam = m.detect_frame(panels, (car["x"], car["y"]))
        sol = m.solve(per_cam, (car["x"], car["y"]),
                      min_cams=args.min_cams, pair_max_cm=args.pair_max_cm)
        if sol is None:
            continue
        obs.append({**sol,
                    "t": fr.get("exposure_pc"),
                    "elapsed_s": round(float(fr["elapsed_s"]), 4),
                    "frame_idx": fr.get("idx"),
                    "video_frame_idx": int(v)})
    cap.release()

    summary = {
        "racket_frames_processed": n_try,
        "racket_observations_3d": len(obs),
        "racket_min_cams": args.min_cams,
        "racket_pair_max_cm": args.pair_max_cm,
        "racket_d02_nominal_mm": D02_NOMINAL_MM,
        "racket_d02_tol_mm": D02_TOL_MM,
    }
    json.dump({"config": {"racket_model_path": "yolo_model/racket.onnx",
                          "racket_conf_threshold": m._bbox_conf,
                          "racket_measure_gate": {
                              "min_cams": args.min_cams,
                              "pair_max_cm": args.pair_max_cm,
                              "score_min": args.score_min,
                              "d02_mm": [D02_NOMINAL_MM - D02_TOL_MM,
                                         D02_NOMINAL_MM + D02_TOL_MM]}},
               "summary": summary,
               "racket_observations": obs},
              open(out_path, "w", encoding="utf-8"), ensure_ascii=False)
    rate = 100.0 * len(obs) / max(1, n_try)
    print(f"[racket] {len(obs)}/{n_try} 帧通过质检 ({rate:.0f}%)，"
          f"{time.time() - t0:.0f}s → {out_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
