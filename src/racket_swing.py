"""球员挥拍竖直速度：判上旋 / 平击 / 下旋。

两级：
  1. **定位球员**——在球员所在的世界框投影出的像素框里，用跑动背景 EMA 做前景，
     取前景质心。人相对场地是「静止的异物」，背景差比帧差稳，不会被飞过的球带跑。
  2. **测拍头**——在球员周围的小 ROI 里，拍头是**又亮又快**的那一块：帧差 × 亮度，
     再排掉黄绿色（否则质心被球拽走），取前 K 个像素的加权质心。

逐帧压进环形缓冲；ContactSolver 从 `/ball_world_topic` 反解出触球时刻后回看拟合，
斜率即拍头竖直速度。

为什么锚在 RK 而不是 PC 自己的球：PC 切片送 YOLO 前有 3.08x 下采样，远端的球只剩 2.4
像素，要飞到 y≈11m 才看得见（触球后 634ms 中位）；RK 车载相机 81ms 就出第一条 S0。

⚠ 已知失效模式（0811-0812 十一场批验证的结论）：
  1. **移动击球**——触球瞬间人还在跑时，腿/躯干+球场线遮挡闪烁的「运动×亮度」得分
     会淹掉拍头，且污染随视角不同，两相机各拍各的（0811_051552 / 0811_080158 /
     0812_050934 三场坏在这）。⚠ 别试图在测量端切掉低位（见 _measure 里的注释）；
     防线是**双相机 vz 一致性门**——不一致的抛/场不输出。
  2. **回放伪影**——run_tracker 烧在视频左上角的时间戳条每帧都变，是栅格 (0,0) 相机
     （本站 DB0260414）画面里最强的运动源，离线回放必须先遮掉（validate_racket_swing
     已做）；线上喂原始相机图，无此问题。
  3. 旧注释称 0405/0373 掐头故不可用——**错**：掐掉的只是头顶上方，挥拍区仍在框内，
     实测这两台场内 vz 一致性 0.72~0.99、对 e_xy 的留出相关 +0.76~+0.85。
"""

from __future__ import annotations

import json
import math
import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np

# 标定世界系单位是毫米（与 ball_localizer._WORLD_SCALE_M_PER_MM 一致）
_MM_PER_M = 1000.0
_G = 9.8


class ContactSolver:
    """从 RK 的 `/ball_world_topic` 反解「球员把球打出去」的时刻。

    锚点必须是**物理定义**的：0811 那批场次实测，「bot_center 首条 stage 0」到触球的滞后
    比 0812 晚 400ms 以上，用固定偏移读到的是收拍段，逐场相关性直接翻负。这里改成沿弹前
    抛物线反推到 z=1.05m 的较早根——与 bot_center 的发射时机无关。

    时间：拟合在 **RK 单调钟**（载荷 t，精度高）上做，再用**本抛内**的
    median(收包 perf − 载荷 t) 挪到 PC 轴。这是每抛一个的局部偏移，不是跨钟映射，
    不跨抛复用、不外推。
    """

    def __init__(self, cfg: dict) -> None:
        self._z_contact = float(cfg.get("contact_height_m", 1.05))
        self._min_points = int(cfg.get("contact_min_points", 6))
        self._min_speed = float(cfg.get("contact_min_approach_mps", 3.0))
        self._max_gap_s = float(cfg.get("contact_max_gap_s", 0.25))
        self._cooldown_s = float(cfg.get("contact_cooldown_s", 1.5))
        self._leg: list[tuple[float, float, float, float]] = []   # (t_rk, y, z, recv_pc)
        self._fired = False
        self._last_fire = float("-inf")

    def add(self, payload: dict, recv_pc: float) -> float | None:
        """喂一条 ball_world；返回触球时刻（PC perf 轴）或 None。"""
        try:
            t = float(payload["t"]); y = float(payload["y"]); z = float(payload["z"])
        except (KeyError, TypeError, ValueError):
            return None
        if not all(math.isfinite(v) for v in (t, y, z)):
            return None
        if self._leg:
            dt = t - self._leg[-1][0]
            approaching = dt > 0 and (self._leg[-1][1] - y) / max(dt, 1e-6) > 0.0
            if dt <= 0 or dt > self._max_gap_s or not approaching:
                self._leg = []
                self._fired = False
        self._leg.append((t, y, z, recv_pc))
        if len(self._leg) > 40:
            self._leg.pop(0)
        if self._fired or len(self._leg) < self._min_points:
            return None
        if recv_pc - self._last_fire < self._cooldown_s:
            return None

        seg = self._leg[:self._min_points]
        ts = np.array([p[0] for p in seg]); ys = np.array([p[1] for p in seg])
        zs = np.array([p[2] for p in seg])
        span = ts[-1] - ts[0]
        if span <= 0 or (ys[0] - ys[-1]) / span < self._min_speed:
            return None
        tm = ts.mean()
        A = np.c_[np.ones(len(ts)), ts - tm]
        z0, vz = np.linalg.lstsq(A, zs + 0.5 * _G * (ts - tm) ** 2, rcond=None)[0]
        disc = vz * vz + 2.0 * _G * (z0 - self._z_contact)
        if disc < 0.0:
            return None
        t_contact_rk = tm + (vz - math.sqrt(disc)) / _G
        # 反解必须落在观测之前、且不太远：否则这段是**弹后**的上升段，反解会跑到触地之后
        lead = ts[0] - t_contact_rk
        if not (-0.05 < lead < 0.80):
            return None
        offset = float(np.median([p[3] - p[0] for p in seg]))
        self._fired = True
        self._last_fire = recv_pc
        return t_contact_rk + offset


class RacketSwingTracker:
    """单相机、纯图像域。线程安全：update 在全图线程、read 在 ROS2 spin 线程。"""

    def __init__(self, cfg: dict, calib_path: str | Path, serial: str) -> None:
        self.serial = serial
        # downscale = 工作分辨率相对相机原生的倍率；input_scale = 调用方喂进来的图相对
        # 原生的倍率（线上原生=1，离线回放拼格视频=2）。
        self._downscale = int(cfg.get("downscale", 2))
        self._input_scale = int(cfg.get("input_scale", 1))
        if self._downscale % self._input_scale:
            raise ValueError("downscale must be a multiple of input_scale")
        self._step = self._downscale // self._input_scale

        box = cfg.get("player_box_world_m", {})
        self._box_x = tuple(box.get("x", [-2.5, 2.5]))
        self._box_y = tuple(box.get("y", [14.0, 20.0]))
        self._box_z = tuple(box.get("z", [0.0, 2.4]))

        self._roi_half = tuple(int(v) for v in cfg.get("roi_half_px", [95, 95]))
        self._roi_dy = int(cfg.get("roi_offset_v_px", -15))
        self._margin = int(cfg.get("stash_margin_px", 12))
        self._top_k = int(cfg.get("top_k", 90))
        self._bright_floor = float(cfg.get("bright_floor", 40.0))
        self._ball_reject_gb = float(cfg.get("ball_reject_gb", 22.0))
        self._bg_alpha = float(cfg.get("bg_alpha", 0.01))
        self._fg_threshold = float(cfg.get("fg_threshold", 25.0))
        self._locate_alpha = float(cfg.get("locate_alpha", 0.05))
        self._locate_stride = int(cfg.get("locate_stride", 2))
        self._read_delay_s = float(cfg.get("read_delay_s", 0.200))
        self._read_half_s = float(cfg.get("read_half_s", 0.120))
        self._lift_mps = float(cfg.get("lift_threshold_mps", 0.85))
        self._slice_mps = float(cfg.get("slice_threshold_mps", -0.35))
        self._trace_half_s = float(cfg.get("trace_half_s", 0.50))

        cam = json.loads(Path(calib_path).read_text(encoding="utf-8"))["cameras"][serial]
        self._K = np.array(cam["K"], dtype=np.float64).reshape(3, 3)
        self._D = np.array(cam["D"], dtype=np.float64).ravel()
        self._rvec = cv2.Rodrigues(np.array(cam["R_world"], dtype=np.float64).reshape(3, 3))[0]
        self._tvec = np.array(cam["t_world"], dtype=np.float64).reshape(3, 1)
        self._img_w = int(cam["image_size"][0]) // self._downscale
        self._img_h = int(cam["image_size"][1]) // self._downscale

        corners = [(x, y, z) for x in self._box_x for y in self._box_y for z in self._box_z]
        pix = np.array([self._project(*p) for p in corners])
        self.player_box = (
            max(0, int(pix[:, 0].min())), min(self._img_w, int(pix[:, 0].max()) + 1),
            max(0, int(pix[:, 1].min())), min(self._img_h, int(pix[:, 1].max()) + 1),
        )
        cx = 0.5 * (self._box_x[0] + self._box_x[1])
        cy = 0.5 * (self._box_y[0] + self._box_y[1])
        span = abs(self._project(cx, cy, 1.80)[1] - self._project(cx, cy, 0.80)[1])
        self.m_per_px = 1.0 / span
        # 相机看不全球员时这个测量不可信：拍头挥到画面外，质心会跑到身体上
        head = self._project(cx, cy, 1.85)[1]
        reach = self._project(cx, cy, 2.40)[1]
        self.player_fully_visible = bool(head > 0 and reach > 0)

        bx0, bx1, by0, by1 = self.player_box
        self._centre = [0.5 * (bx0 + bx1), 0.5 * (by0 + by1)]
        self._bg: np.ndarray | None = None
        self._lock = threading.Lock()
        self._buf: deque[tuple[float, float, float]] = deque(
            maxlen=int(float(cfg.get("buffer_seconds", 2.5)) * 40.0))
        self._stash: np.ndarray | None = None
        self._stash_org = (0, 0)
        self.frames_seen = 0
        self.samples = 0
        self.located = 0

    def _project(self, x_m: float, y_m: float, z_m: float) -> tuple[float, float]:
        pix, _ = cv2.projectPoints(
            np.array([[x_m, y_m, z_m]], dtype=np.float64) * _MM_PER_M,
            self._rvec, self._tvec, self._K, self._D,
        )
        u, v = pix.reshape(2)
        return u / self._downscale, v / self._downscale

    # ── 每帧 ────────────────────────────────────────────────────────────────
    def update(self, image: np.ndarray, elapsed_s: float) -> None:
        """image: 该相机整幅 BGR，分辨率由 input_scale 声明。"""
        work = image[::self._step, ::self._step] if self._step > 1 else image
        self.frames_seen += 1
        self._locate(work)
        self._measure(work, elapsed_s)

    def _locate(self, work: np.ndarray) -> None:
        bx0, bx1, by0, by1 = self.player_box
        s = self._locate_stride
        box = work[by0:by1:s, bx0:bx1:s].astype(np.float32)
        if self._bg is None or self._bg.shape != box.shape:
            self._bg = box.copy()
            return
        fg = np.abs(box - self._bg).max(axis=2)
        mask = fg > self._fg_threshold
        # 只在**没有前景**的像素上更新背景：球员两次喂球之间站着不动，若无条件 EMA
        # 就会被吸进背景，定位随即漂走（0811 那批场次平均 vz 变负就是这么来的）。
        self._bg[~mask] += self._bg_alpha * (box - self._bg)[~mask]
        if mask.sum() < 30:
            return
        rows, cols = np.nonzero(mask)
        w = fg[mask]
        px = bx0 + s * float((cols * w).sum() / w.sum())
        py = by0 + s * float((rows * w).sum() / w.sum())
        self._centre[0] += self._locate_alpha * (px - self._centre[0])
        self._centre[1] += self._locate_alpha * (py - self._centre[1])
        self.located += 1

    def _roi(self) -> tuple[int, int, int, int]:
        ex, ey = self._centre
        hw, hh = self._roi_half
        return (int(ex - hw), int(ex + hw),
                int(ey - hh + self._roi_dy), int(ey + hh + self._roi_dy))

    def _measure(self, work: np.ndarray, elapsed_s: float) -> None:
        h, w = work.shape[:2]
        x0, x1, y0, y1 = self._roi()
        bx0, bx1, by0, by1 = self.player_box
        # 拍头物理上只可能在球员世界框的投影里：框外上方是亮窗一类的高分噪声源。
        # ⚠ 只做这个静态钳制，别再切低位：「质心以下截断」试过（0813）——上旋的信号
        # 恰恰在拍头从膝腰位拉起来那一段，截掉后语料跨相机一致性 0.65→0.28，已回退。
        x0, x1 = max(x0, bx0 - 6, 0), min(x1, bx1 + 6, w)
        y0, y1 = max(y0, by0 - 6, 0), min(y1, by1 + 6, h)
        if x1 - x0 < 16 or y1 - y0 < 16:
            self._stash = None
            return
        cur = work[y0:y1, x0:x1].astype(np.float32)
        prev = self._aligned_prev(x0, x1, y0, y1)
        self._restash(work, x0, x1, y0, y1)
        if prev is None:
            return
        # 拍框是白/浅蓝，球是黄绿：G−B 大的像素直接排掉，否则质心会被球拽走
        motion = np.abs(cur - prev).max(axis=2)
        not_ball = (cur[:, :, 1] - cur[:, :, 0]) < self._ball_reject_gb
        score = (motion * np.clip(cur.max(axis=2) - self._bright_floor, 0.0, None)
                 * not_ball).ravel()
        if float(score.sum()) <= 0.0:
            return
        k = min(self._top_k, score.size)
        idx = np.argpartition(score, -k)[-k:]
        weight = score[idx]
        rows, cols = np.unravel_index(idx, motion.shape)
        wsum = float(weight.sum())
        ry = y0 + float((rows * weight).sum() / wsum)
        with self._lock:
            self._buf.append((elapsed_s, ry, wsum))
        self.samples += 1

    def _aligned_prev(self, x0: int, x1: int, y0: int, y1: int) -> np.ndarray | None:
        """ROI 会动，所以两帧必须落在**同一个像素网格**上取，否则整幅都被判成运动。
        存的是「ROI + 一圈余量」和它的原点，按整数位移切出来即可。"""
        if self._stash is None:
            return None
        sx = x0 - self._stash_org[0]
        sy = y0 - self._stash_org[1]
        if sx < 0 or sy < 0:
            return None
        if sy + (y1 - y0) > self._stash.shape[0] or sx + (x1 - x0) > self._stash.shape[1]:
            return None
        return self._stash[sy:sy + (y1 - y0), sx:sx + (x1 - x0)]

    def _restash(self, work: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> None:
        h, w = work.shape[:2]
        sx0, sy0 = max(0, x0 - self._margin), max(0, y0 - self._margin)
        sx1, sy1 = min(w, x1 + self._margin), min(h, y1 + self._margin)
        self._stash = work[sy0:sy1, sx0:sx1].astype(np.float32)
        self._stash_org = (sx0, sy0)

    # ── 取数 ────────────────────────────────────────────────────────────────
    def read(self, anchor_elapsed_s: float) -> dict | None:
        """anchor = 收到该抛第一条 stage 0 的时刻（PC perf 轴）。"""
        t_read = anchor_elapsed_s - self._read_delay_s
        with self._lock:
            samples = list(self._buf)
        fit = [(t, ry) for t, ry, _ in samples if abs(t - t_read) <= self._read_half_s]
        if len(fit) < 4:
            return None
        t_arr = np.array([t for t, _ in fit])
        h_arr = np.array([-ry * self.m_per_px for _, ry in fit])
        vz = float(np.polyfit(t_arr, h_arr, 1)[0])
        if not math.isfinite(vz):
            return None
        if vz > self._lift_mps:
            label = "topspin"
        elif vz < self._slice_mps:
            label = "slice"
        else:
            label = "flat"
        return {
            "anchor_elapsed_s": round(anchor_elapsed_s, 4),
            "read_elapsed_s": round(t_read, 4),
            "vz": round(vz, 3),
            "label": label,
            "n_fit": len(fit),
            # top-K 评分质量的中位数：拍头信号强度，弱 = 质心可能不在拍上（可信度门用）
            "score_mass": round(float(np.median([wq for t, _, wq in samples
                                                 if abs(t - t_read) <= self._read_half_s])), 1),
            "serial": self.serial,
            # 原始高度轨迹：离线重调 read_delay_s / read_half_s 不必再解视频
            "trace": [
                [round(t, 4), round(-ry * self.m_per_px, 4)]
                for t, ry, _ in samples
                if abs(t - t_read) <= self._trace_half_s
            ],
        }
