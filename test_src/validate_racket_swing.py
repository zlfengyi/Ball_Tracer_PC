"""批量验证挥拍分类：跑生产 RacketSwingTracker 过每场录像，对比实测反弹系数。

用法：
    python test_src/validate_racket_swing.py align          # 只做 PC↔RK 对齐自检
    python test_src/validate_racket_swing.py run [session…] # 全流程（要解视频，慢）

对齐：绝大多数场次没有 `_tables.json`，所以这里用**球的 z 形状**自对齐——PC 侧
observations 的 z(t) 和 RK 侧 world 的 z(t) 是同一颗球，扫偏移取残差最小。速率项固定
为 1（已知场次实测 |a−1| ≤ 90ppm，300s 累计 27ms，远小于锚点容差）。

分类阈值/读数位置来自 config/tracker.json 的 racket_swing 段——本脚本不再自己调参，
它就是用来判定那组常量在别的场次站不站得住的。
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.racket_swing import ContactSolver, RacketSwingTracker  # noqa: E402

ROOT = Path(r"D:\Ball_Tracer_PC\tracker_output")
CONFIG_DIR = Path(__file__).resolve().parents[1] / "src" / "config"
G = 9.8
ZC = 0.033
QW, QH, COLS = 1024, 768, 2
# bot_center 的 predict 载荷（末尾固定是 s0_gate）
_PRED = re.compile(rb'\{"x":-?\d[^\x00]{50,2500}?"s0_gate":"[^"]*"\}')


def sessions(calib: Path) -> list[Path]:
    """只要相机序列号与标定文件一致的场次（换站点/换车的旧场次直接排除）。"""
    want = set(json.loads(calib.read_text(encoding="utf-8"))["cameras"])
    out = []
    for d in sorted(ROOT.glob("tracker_2026*")):
        name = d.name
        if not (d / f"{name}.mp4").exists():
            continue
        if not (d / f"{name}_rk_tracking.json").exists():
            continue
        if not list(d.glob("*_rosbag/*.mcap")):
            continue
        try:
            cfg = json.loads((d / f"{name}.json").read_text(encoding="utf-8"))["config"]
        except Exception:
            continue
        if set(cfg.get("serials") or []) != want:
            continue
        out.append(d)
    return out


def known_offset(d: Path) -> tuple[float, float] | None:
    tj = d / f"{d.name}_tables.json"
    if not tj.exists():
        return None
    m = re.search(r"PC t = ([\d.]+) . RK t \+ (-?[\d.]+)", tj.read_text(encoding="utf-8"))
    return (float(m.group(1)), float(m.group(2))) if m else None


def align(d: Path) -> float | None:
    """返回 b，使得 PC t ≈ RK t + b（速率项取 1）。"""
    name = d.name
    pc = json.loads((d / f"{name}.json").read_text(encoding="utf-8"))
    t0 = pc["config"]["first_frame_exposure_pc"]
    obs = pc.get("observations") or []
    if len(obs) < 200:
        return None
    tp = np.array([o["t"] for o in obs]) - t0
    zp = np.array([o["z"] for o in obs])
    order = np.argsort(tp)
    tp, zp = tp[order], zp[order]

    rk = json.loads((d / f"{name}_rk_tracking.json").read_text(encoding="utf-8"))
    w = rk.get("world") or {}
    if not w.get("t"):
        return None
    tr = np.array(w["t"])
    zr = np.array(w["y"]["z"])
    order = np.argsort(tr)
    tr, zr = tr[order], zr[order]

    best = None
    for step, span, centre in ((0.050, 25.0, -8.0), (0.005, 0.30, None)):
        if centre is None and best is None:
            return None
        c = centre if centre is not None else best[1]
        for b in np.arange(c - span, c + span + 1e-9, step):
            zi = np.interp(tp - b, tr, zr, left=np.nan, right=np.nan)
            near = np.abs(np.interp(tp - b, tr, tr, left=np.nan, right=np.nan) - (tp - b))
            ok = np.isfinite(zi) & (np.abs(zi - zp) < 1.5)
            if ok.sum() < 100:
                continue
            err = float(np.sqrt(np.mean((zi[ok] - zp[ok]) ** 2))) + 0.4 * (1.0 - ok.mean())
            if best is None or err < best[0]:
                best = (err, float(b), int(ok.sum()))
    return best[1] if best else None


def contacts(d: Path, b: float, cfg: dict) -> list[float]:
    """把 RK 录下的世界系球点喂给生产 ContactSolver，得到每抛触球时刻（PC elapsed 轴）。

    离线没有收包时刻，直接用 recv = t + b —— 于是 solver 内部那个「本抛局部偏移」
    恰好等于对齐系数，与线上路径等价。"""
    rk = json.loads((d / f"{d.name}_rk_tracking.json").read_text(encoding="utf-8"))
    w = rk.get("world") or {}
    if not w.get("t"):
        return []
    T = np.array(w["t"]); X = np.array(w["y"]["x"])
    Y = np.array(w["y"]["y"]); Z = np.array(w["y"]["z"])
    order = np.argsort(T)
    solver = ContactSolver(cfg)
    out = []
    for k in order:
        t = float(T[k])
        got = solver.add({"t": t, "x": float(X[k]), "y": float(Y[k]), "z": float(Z[k])}, t + b)
        if got is not None:
            out.append(got)
    return out


def stage0_anchors(d: Path, b: float) -> list[float]:
    """RK 每抛第一条 stage 0，映射到 PC elapsed 轴（只用于和物理锚点对照）。"""
    raw = list(d.glob("*_rosbag/*.mcap"))[0].read_bytes()
    rk = json.loads((d / f"{d.name}_rk_tracking.json").read_text(encoding="utf-8"))
    t0 = rk["t0"]
    ts = []
    for m in _PRED.finditer(raw):
        try:
            p = json.loads(m.group(0).decode("utf-8"))
        except Exception:
            continue
        if p.get("stage") == 0:
            ts.append(p["ct"] - t0 + b)
    ts.sort()
    out = []
    for t in ts:
        if not out or t - out[-1][1] > 1.5:
            out.append([t, t])
        else:
            out[-1][1] = t
    return [g[0] for g in out]


def bounces(d: Path) -> list[dict]:
    """每次「击球前触地」的切向/法向恢复系数（与 build_ds.py 同一估计器）。"""
    rk = json.loads((d / f"{d.name}_rk_tracking.json").read_text(encoding="utf-8"))
    w = rk["world"]
    T = np.array(w["t"]); X = np.array(w["y"]["x"])
    Y = np.array(w["y"]["y"]); Z = np.array(w["y"]["z"])

    def fit_lin(t, v):
        tm = t.mean(); A = np.c_[np.ones(len(t)), t - tm]
        return tm, np.linalg.lstsq(A, v, rcond=None)[0]

    def fit_z(t, z):
        tm = t.mean(); A = np.c_[np.ones(len(t)), t - tm]
        c = np.linalg.lstsq(A, z + 0.5 * G * (t - tm) ** 2, rcond=None)[0]
        r = A @ c - (z + 0.5 * G * (t - tm) ** 2)
        return tm, c, float(np.sqrt((r ** 2).mean()))

    def zroot(tm, c, latest):
        a = -0.5 * G; bb = c[1]; cc = c[0] - ZC
        dsc = bb * bb - 4 * a * cc
        if dsc < 0:
            return None
        r = np.sqrt(dsc)
        r1, r2 = tm + (-bb - r) / (2 * a), tm + (-bb + r) / (2 * a)
        return max(r1, r2) if latest else min(r1, r2)

    out = []
    k = 3
    while k < len(Z) - 5:
        if not (Z[k] < 0.55 and Z[k] <= Z[k - 1] and Z[k + 1] > Z[k]):
            k += 1
            continue
        pre = np.arange(max(0, k - 8), k + 1)
        post = np.arange(k + 1, min(len(Z), k + 10))
        pre = pre[T[pre] > T[k] - 0.40]
        post = post[T[post] < T[k] + 0.45]
        if len(pre) < 5 or len(post) < 5:
            k += 1
            continue
        tm0, cz0, rz0 = fit_z(T[pre], Z[pre])
        tm1, cz1, rz1 = fit_z(T[post], Z[post])
        ra, rb_ = zroot(tm0, cz0, True), zroot(tm1, cz1, False)
        if rz0 > 0.05 or rz1 > 0.05 or ra is None or rb_ is None:
            k += 1
            continue
        tb = 0.5 * (ra + rb_)
        _, cx0 = fit_lin(T[pre], X[pre]); _, cy0 = fit_lin(T[pre], Y[pre])
        _, cx1 = fit_lin(T[post], X[post]); _, cy1 = fit_lin(T[post], Y[post])
        vin = float(np.hypot(cx0[1], cy0[1])); vout = float(np.hypot(cx1[1], cy1[1]))
        vz_in = float(cz0[1] - G * (tb - tm0)); vz_out = float(cz1[1] - G * (tb - tm1))
        if vin >= 3.0 and vz_in <= -2.0 and vz_out >= 1.0 and cy0[1] < -2.5:
            out.append(dict(tb=tb, e_xy=vout / vin, e_z=-vz_out / vz_in, vin=vin))
        k += 6
    return out


def run_session(d: Path, cfg: dict, calib: Path) -> list[dict] | None:
    b = known_offset(d)
    b = b[1] if b else align(d)
    if b is None:
        print(f"  {d.name}: 对不上，跳过")
        return None
    anch = contacts(d, b, cfg)
    s0 = stage0_anchors(d, b)
    bnc = bounces(d)
    if len(anch) < 4 or len(bnc) < 4:
        print(f"  {d.name}: contacts={len(anch)} bounces={len(bnc)}，样本太少")
        return None
    lag = [min((x - a for x in s0 if x > a - 0.4), default=np.nan) for a in anch]
    lag = np.array([v for v in lag if np.isfinite(v)])
    if len(lag):
        print("  stage0 − 触球: 中位 %+.0f ms (p25 %+.0f / p75 %+.0f)" %
              (np.median(lag) * 1000, np.percentile(lag, 25) * 1000,
               np.percentile(lag, 75) * 1000))

    pc = json.loads((d / f"{d.name}.json").read_text(encoding="utf-8"))
    frames = pc["frames"]
    fel = np.array([f["elapsed_s"] for f in frames])
    fvi = np.array([(f["video_frame_idx"] if f["video_frame_idx"] is not None else -1)
                    for f in frames])
    serials = pc["config"]["serials"]
    order = pc["config"]["video_output"].get("serial_order") or serials
    swing_cfg = dict(cfg, downscale=2, input_scale=2)
    tk = {s: RacketSwingTracker(swing_cfg, calib, s) for s in serials}

    cap = cv2.VideoCapture(str(d / f"{d.name}.mp4"))
    pend = {s: list(anch) for s in serials}
    got = {s: {} for s in serials}
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for vi in range(n):
        ok, im = cap.read()
        if not ok:
            break
        t = float(np.interp(vi, fvi, fel))
        # run_tracker 烧在整帧左上角的时间戳条（x9..636, y9..60）每帧都变，是栅格 (0,0)
        # 相机画面里最强的运动源；线上喂原始图没有它，回放必须遮掉
        im[0:64, 0:660] = 0
        for s in serials:
            i = order.index(s)
            c, r = i % COLS, i // COLS
            tk[s].update(im[r * QH:(r + 1) * QH, c * QW:(c + 1) * QW], t)
            while pend[s] and pend[s][0] <= t:
                a = pend[s].pop(0)
                got[s][a] = tk[s].read(a)
    cap.release()

    rows = []
    for a in anch:
        # 只认「这一抛的击球前触地」：落在锚点之后、下一个锚点之前，且系数在物理范围内。
        # 不设范围时会 admit 机器人回球后的二次触地等，把 e_xy 的散布灌成两倍（0812_115318
        # 曾出现 sd=0.234 vs 语料 0.036）。
        nxt = min([x for x in anch if x > a + 0.5], default=a + 3.0)
        cand = [x for x in bnc
                if 0.2 < x["tb"] + b - a < 2.6 and x["tb"] + b < nxt
                and 0.20 <= x["e_xy"] <= 1.00 and 0.50 <= x["e_z"] <= 1.10]
        if not cand:
            continue
        hit = max(cand, key=lambda x: x["tb"])
        row = dict(session=d.name, anchor=a, e_xy=hit["e_xy"], e_z=hit["e_z"], vin=hit["vin"])
        for s in serials:
            rd = got[s].get(a)
            if rd:
                row[s] = rd["vz"]
                # 存原始高度轨迹：离线扫 read_delay/read_half 不必再解一遍视频
                row[s + "_trace"] = rd["trace"]
        rows.append(row)
    return rows


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "align"
    cfg = json.loads((CONFIG_DIR / "tracker.json").read_text(encoding="utf-8"))["racket_swing"]
    calib = CONFIG_DIR / "four_camera_calib_18.json"

    if mode == "align":
        print("%-26s %12s %12s %10s" % ("session", "known b", "z-shape b", "diff(ms)"))
        for d in sessions(calib):
            k = known_offset(d)
            a = align(d)
            kb = k[1] if k else None
            diff = "" if (kb is None or a is None) else "%+.0f" % ((a - kb) * 1000)
            print("%-26s %12s %12s %10s" % (
                d.name,
                "-" if kb is None else "%.4f" % kb,
                "-" if a is None else "%.4f" % a, diff))
        return

    want = sys.argv[2:]
    allrows = []
    for d in sessions(calib):
        if want and d.name not in want:
            continue
        print(f"[{d.name}]")
        rows = run_session(d, cfg, calib)
        if rows:
            allrows += rows
            print(f"  {len(rows)} 抛")
    out = Path(__file__).resolve().parents[1] / "tracker_output" / "racket_swing_validation.json"
    out.write_text(json.dumps(allrows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n写出 {out}  共 {len(allrows)} 抛 / {len({r['session'] for r in allrows})} 场")

    thr = float(cfg.get("lift_threshold_mps", 0.85))
    e = np.array([r["e_xy"] for r in allrows])
    print("\n%-12s %5s %8s %8s %8s %7s" % ("camera", "n", "topspin", "flat", "diff", "t"))
    for s in sorted({k for r in allrows for k in r
                     if k.startswith("DB") and not k.endswith("_trace")}):
        v = np.array([r.get(s, np.nan) for r in allrows], float)
        k = np.isfinite(v)
        hi = k & (v > thr); lo = k & (v <= thr)
        if hi.sum() < 3 or lo.sum() < 3:
            print("%-12s %5d  样本不足" % (s, k.sum()))
            continue
        sd = np.sqrt(((hi.sum() - 1) * e[hi].var(ddof=1) + (lo.sum() - 1) * e[lo].var(ddof=1))
                     / (k.sum() - 2))
        t = (e[hi].mean() - e[lo].mean()) / (sd * np.sqrt(1 / hi.sum() + 1 / lo.sum()))
        print("%-12s %5d %8.3f %8.3f %+8.3f %7.2f" %
              (s, k.sum(), e[hi].mean(), e[lo].mean(), e[hi].mean() - e[lo].mean(), t))


if __name__ == "__main__":
    main()
