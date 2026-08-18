# -*- coding: utf-8 -*-
"""回球落点一致性：从一场 tracker 数据拟合「击球时刻拍速/拍面 pitch → 出球 → 落点」并反解目标律。

用法（.venv_clean）：
  python test_src\\fit_return_law.py --input tracker_output\\tracker_<ts>\\tracker_<ts>.json ^
      [--targets 15,16,17] [--pitch-out 19] [--plot out.png]

流程：
  1. 逐抛提取（[[pc-return-core]] 同口径 + 扩展）：
     - 触球时刻/接触点：来回弧 y(t) 二次拟合交点（锚=观测 y 最小点；tracker ht 比实际触球晚
       ~0.3s——臂在 z≈1.2-1.4 提前拦截，ht 是球到 ideal_hit_z 的时刻，不能直接当锚）；
     - 出球速度：先出弧 [+20,+400]ms 拟合，再用全可见弧（y 单调向前、跨遮挡拼接）对
       重力+k_drag 二次阻力模型 Gauss-Newton 重拟初速（修短窗低估，rms 典型 2-4cm）；
     - 落点：接触态 RK4 外推到 z=GROUND_Z（本场外推仅比末点多 0.3-0.4s）；
     - 来球速度：入弧 [-380,-25]ms 同法拟合在触球时刻取导；
     - 拍速：_arm.json 触球前窗 [-80,-6]ms TCP 二次拟合取 -6ms 瞬时（拍在加速段触球，
       线性窗中心值低估 0.3-1.2 m/s）；拍面 pitch 同窗 FK（extract_arm_bag.fk 单源）；
     - 车残速：rk_tracking bot_state(IMU) 触球 ±50ms 中位 vy；
     - 臂↔PC 时轴：accepted 触球(t_acc+dur, late-ht 覆盖) ↔ PC 观测触球 共识 offset 逐抛配对
       （_arm.json 里 /predict_hit_pos 是 RK bot_center 自己的 topic，ct/ht 全 RK 轴，
       不能用 t_rk−ct 求跨轴 offset）。
  2. 接触模型（沿世界 y）：v_out_y = (1+e)·v_r − e·v_in_y，v_r = 臂 TCP 前向 + 车残速。
  3. 反解：给定 y_land* 与出球 pitch θ*，对 (y_hit, z_hit) 网格解所需 v_out，换算所需拍速。

2026-08-08 首拟（tracker_20260808_034540，11 抛）：e=0.605±0.074、出球 pitch 19.3±0.9°、
落点(模型) 15.4~19.8m；车残速 1.6 m/s 两抛落最深，retro 修正 ±2 m/s 内。
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_arm_bag as eab  # noqa: E402  车型在 main() 里选定后才有 fk/JOINTS

G = 9.81
HALF_G = 4.905
K_DRAG = 0.024
GROUND_Z = 0.033
RETURN_MAX_STEP_MPS = 20.0
RETURN_TRACK_DEV_M = 0.5   # 出弧防污染门 1：末 3 点线性外推偏差上限（切段）
RETURN_FIT_RESID_M = 0.12  # 出弧防污染门 3：三轴拟合 max|残差| 上限（拒绝出数）
RETURN_MIN_FIT_Z = 0.05    # 出弧防污染门 2：[0,段首] 拟合 z（含重力）下限


# ---------------- [[pc-return-core]] 复刻（generate_curve3_html.py 同口径） ----------------
def quad_fit_u(pts, tc):
    n = len(pts)
    if n < 3:
        return None
    u = np.array([p[0] - tc for p in pts])
    v = np.array([p[1] for p in pts])
    if n >= 6:
        A = np.vstack([np.ones(n), u, u * u]).T
        sol, *_ = np.linalg.lstsq(A, v, rcond=None)
        return {"a": sol[0], "b": sol[1], "c": sol[2], "n": n}
    A = np.vstack([np.ones(n), u]).T
    sol, *_ = np.linalg.lstsq(A, v, rcond=None)
    return {"a": sol[0], "b": sol[1], "c": 0.0, "n": n}


class ReturnExtractor:
    def __init__(self, obs):
        self.obs = sorted(obs, key=lambda o: o["t"])

    def runs(self, lo, hi, zmin=0.12):
        runs, run = [], []
        for p in self.obs:
            if p["t"] < lo:
                continue
            if p["t"] > hi:
                break
            if p["z"] < zmin:
                continue
            if run:
                last = run[-1]
                dt = p["t"] - last["t"]
                step = math.dist((p["x"], p["y"], p["z"]),
                                 (last["x"], last["y"], last["z"])) / max(1e-9, dt)
                if dt > 0.15 or step > RETURN_MAX_STEP_MPS:
                    runs.append(run)
                    run = []
            run.append(p)
        if run:
            runs.append(run)
        return runs

    @staticmethod
    def split_by_track_dev(runs):
        """出弧防污染门 1：run 内已有 ≥3 点时由末 3 点线性外推，偏差 >0.5m 切段。
        只用于出弧（入弧窗内可含真地面反弹，桥接断档会被误切）。"""
        out = []
        for run in runs:
            cur = []
            for p in run:
                if len(cur) >= 3:
                    a, b = cur[-3], cur[-1]
                    dv = max(1e-9, b["t"] - a["t"])
                    dt = p["t"] - b["t"]
                    pred = [b[k] + (b[k] - a[k]) / dv * dt for k in ("x", "y", "z")]
                    if math.dist((p["x"], p["y"], p["z"]), pred) > RETURN_TRACK_DEV_M:
                        out.append(cur)
                        cur = []
                cur.append(p)
            if cur:
                out.append(cur)
        return out

    def hit_time_at(self, t_approx):
        yin = None
        for r in reversed(self.runs(t_approx - 0.38, t_approx - 0.025)):
            if len(r) >= 5:
                yin = r
                break
        if yin is None:
            return None
        fin = quad_fit_u([(p["t"], p["y"]) for p in yin], t_approx)
        if not fin or fin["b"] > -1.0:
            return None
        out_runs = [r for r in self.split_by_track_dev(self.runs(t_approx + 0.03, t_approx + 0.33))
                    if len(r) >= 4]
        out_runs.sort(key=len, reverse=True)
        for run in out_runs:
            fout = quad_fit_u([(p["t"], p["y"]) for p in run], t_approx)
            if not fout or fout["b"] < 0.15:
                continue
            a = fin["c"] - fout["c"]
            b = fin["b"] - fout["b"]
            c = fin["a"] - fout["a"]
            roots = []
            if abs(a) < 1e-9:
                if abs(b) > 1e-9:
                    roots = [-c / b]
            else:
                disc = b * b - 4 * a * c
                if disc >= 0:
                    r_ = math.sqrt(disc)
                    roots = [(-b + r_) / (2 * a), (-b - r_) / (2 * a)]
            roots = [u for u in roots if -0.10 <= u <= 0.14]
            if roots:
                return t_approx + min(roots, key=abs)
        return None

    def hit_time_near(self, t_approx):
        t_hit = None
        for d in (0, -0.06, 0.06, -0.12, 0.12):
            t_hit = self.hit_time_at(t_approx + d)
            if t_hit is not None:
                break
        if t_hit is None:
            return None
        refined = self.hit_time_at(t_hit)
        return refined if refined is not None else t_hit

    @staticmethod
    def bounce_cut(run):
        for i in range(2, len(run)):
            vza = (run[i - 1]["z"] - run[i - 2]["z"]) / max(1e-9, run[i - 1]["t"] - run[i - 2]["t"])
            vzb = (run[i]["z"] - run[i - 1]["z"]) / max(1e-9, run[i]["t"] - run[i - 1]["t"])
            if vza < -0.5 and vzb - vza > 3.0:
                return run[:i], True
        return run, False

    @staticmethod
    def _eval_fit(fit, u):
        return fit["a"] + fit["b"] * u + fit["c"] * u * u

    def return_at(self, t_approx):
        t_hit = self.hit_time_near(t_approx)
        if t_hit is None:
            return None
        cands = [r for r in self.split_by_track_dev(self.runs(t_hit + 0.02, t_hit + 0.40))
                 if len(r) >= 5 and r[0]["t"] - t_hit <= 0.30]
        cands.sort(key=len, reverse=True)
        for run in cands:
            seg, cut = self.bounce_cut(run)
            if len(seg) < 5 or seg[-1]["t"] - seg[0]["t"] < 0.06:
                continue
            fx = quad_fit_u([(p["t"], p["x"]) for p in seg], t_hit)
            fy = quad_fit_u([(p["t"], p["y"]) for p in seg], t_hit)
            fz = quad_fit_u([(p["t"], p["z"] + HALF_G * (p["t"] - t_hit) ** 2) for p in seg], t_hit)
            if not (fx and fy and fz):
                continue
            vx, vy, vz = fx["b"], fy["b"], fz["b"]
            if vy <= 0.5 or math.hypot(vx, vy) < 1.0:
                continue
            start = seg[0]["t"] - t_hit
            # 防污染门 2：段首前拟合 z 触地即整段在反弹之后，倒推跨反弹必错
            min_z = min(self._eval_fit(fz, start * i / 16.0) - HALF_G * (start * i / 16.0) ** 2
                        for i in range(17))
            if min_z < RETURN_MIN_FIT_Z:
                continue
            # 防污染门 3：混轨/乱拟合拒绝出数（不剔点重拟合——高杠杆野点会反把真点顶成最大残差）
            max_res = max(
                max(abs(p["x"] - self._eval_fit(fx, p["t"] - t_hit)),
                    abs(p["y"] - self._eval_fit(fy, p["t"] - t_hit)),
                    abs(p["z"] + HALF_G * (p["t"] - t_hit) ** 2 - self._eval_fit(fz, p["t"] - t_hit)))
                for p in seg)
            if max_res > RETURN_FIT_RESID_M:
                continue
            return {"tHit": t_hit, "x0": fx["a"], "y0": fy["a"], "z0": fz["a"],
                    "vx": vx, "vy": vy, "vz": vz, "bounceCut": cut, "maxRes": max_res,
                    "speed": math.hypot(vx, vy, vz), "n": len(seg)}
        return None

    def incoming_at(self, t_hit):
        yin = None
        for r in reversed(self.runs(t_hit - 0.38, t_hit - 0.025)):
            if len(r) >= 5:
                yin = r
                break
        if yin is None:
            return None
        fx = quad_fit_u([(p["t"], p["x"]) for p in yin], t_hit)
        fy = quad_fit_u([(p["t"], p["y"]) for p in yin], t_hit)
        fz = quad_fit_u([(p["t"], p["z"] + HALF_G * (p["t"] - t_hit) ** 2) for p in yin], t_hit)
        if not (fx and fy and fz):
            return None
        vx, vy, vz = fx["b"], fy["b"], fz["b"]
        return {"vx": vx, "vy": vy, "vz": vz,
                "dive": math.degrees(math.atan2(vz, math.hypot(vx, vy)))}

    def collect_arc(self, ret):
        t_hit, y0 = ret["tHit"], ret["y0"]
        pts, last = [], None
        for p in self.obs:
            if p["t"] < t_hit + 0.02 or p["t"] > t_hit + 3.0:
                continue
            if p["z"] < 0.10 or p["y"] < y0 + 0.05:
                continue
            if last is not None:
                dt = p["t"] - last["t"]
                if dt > 0.5:
                    break
                step = math.dist((p["x"], p["y"], p["z"]),
                                 (last["x"], last["y"], last["z"])) / max(1e-9, dt)
                if step > RETURN_MAX_STEP_MPS or p["y"] < last["y"] - 0.10:
                    break
            pts.append(p)
            last = p
        return pts


# ---------------- 阻力弹道 ----------------
def _deriv(s):
    sp = math.hypot(s[3], s[4], s[5])
    return np.array([s[3], s[4], s[5],
                     -K_DRAG * sp * s[3], -K_DRAG * sp * s[4], -G - K_DRAG * sp * s[5]])


def rk4_to(s, t_from, t_to, dt=0.002):
    t = t_from
    while t < t_to - 1e-9:
        h = min(dt, t_to - t)
        k1 = _deriv(s); k2 = _deriv(s + h / 2 * k1)
        k3 = _deriv(s + h / 2 * k2); k4 = _deriv(s + h * k3)
        s = s + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t += h
    return s


def propagate_landing(state, dt=0.002, tmax=5.0):
    s = np.array(state, float)
    t = 0.0
    while t < tmax:
        k1 = _deriv(s); k2 = _deriv(s + dt / 2 * k1)
        k3 = _deriv(s + dt / 2 * k2); k4 = _deriv(s + dt * k3)
        s2 = s + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        if s2[2] <= GROUND_Z and s2[5] < 0:
            f = (s[2] - GROUND_Z) / max(1e-9, s[2] - s2[2])
            hit = s + f * (s2 - s)
            return {"x": float(hit[0]), "y": float(hit[1]), "t": t + f * dt}
        s = s2
        t += dt
    return None


def fit_arc_velocity(ret, pts):
    """固定接触点，Gauss-Newton 拟合初速使阻力模型过全部可见弧点。"""
    if len(pts) < 8:
        return None
    t_hit = ret["tHit"]
    p0 = np.array([ret["x0"], ret["y0"], ret["z0"]])
    v = np.array([ret["vx"], ret["vy"], ret["vz"]], float)
    ts = np.array([p["t"] - t_hit for p in pts])
    target = np.array([[p["x"], p["y"], p["z"]] for p in pts])

    def traj(v):
        out = []
        s = np.concatenate([p0, v])
        t = 0.0
        for ti in ts:
            s = rk4_to(s, t, ti)
            t = ti
            out.append(s[:3].copy())
        return np.array(out)

    for _ in range(6):
        base = traj(v)
        res = (target - base).ravel()
        J = np.zeros((len(res), 3))
        for k in range(3):
            dv = np.zeros(3)
            dv[k] = 0.02
            J[:, k] = ((traj(v + dv) - base) / 0.02).ravel()
        step, *_ = np.linalg.lstsq(J, res, rcond=None)
        v = v + step
        if np.linalg.norm(step) < 1e-4:
            break
    resid = target - traj(v)
    return {"vx": float(v[0]), "vy": float(v[1]), "vz": float(v[2]),
            "speed": float(np.linalg.norm(v)),
            "rms": float(np.sqrt(np.mean(np.sum(resid ** 2, axis=1)))),
            "n": len(pts), "u_max": float(ts[-1])}


def land_y_of(y0, z0, v, theta_deg):
    th = math.radians(theta_deg)
    r = propagate_landing([0.0, y0, z0, 0.0, v * math.cos(th), v * math.sin(th)])
    return r["y"] if r else float("nan")


def req_speed(y0, z0, theta_deg, y_land_target):
    lo, hi = 4.0, 25.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if land_y_of(y0, z0, mid, theta_deg) < y_land_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------- 臂 / 车 ----------------
def parse_arm_events(arm):
    acc, late = [], []
    for e in arm["events"]:
        txt = e["text"]
        if txt.startswith("accepted hit"):
            kv = dict(m.split("=") for m in txt.split()[2:] if "=" in m)
            acc.append({"t": e["t"], "dur": float(kv["duration"]),
                        "pitch": float(kv["pitch"])})
        elif txt.startswith("late ht saved"):
            m = re.search(r"contact in ([0-9.]+)s", txt)
            if m:
                late.append({"t": e["t"], "dt": float(m.group(1))})
    acc.sort(key=lambda a: a["t"])
    clusters = []
    for a in acc:
        if not clusters or a["t"] - clusters[-1][-1]["t"] > 3.0:
            clusters.append([a])
        else:
            clusters[-1].append(a)
    out = []
    for cl in clusters:
        last = cl[-1]
        contact = last["t"] + last["dur"]
        src_t = last["t"]
        for lh in late:
            if cl[0]["t"] - 0.5 <= lh["t"] <= contact + 0.5 and lh["t"] > src_t:
                contact = lh["t"] + lh["dt"]
                src_t = lh["t"]
        out.append({"contact_rk": contact, "pitch_cmd": last["pitch"], "n_acc": len(cl)})
    return out


def arm_at_contact(states, st_t, cl, bot_t, bot_ch, rk_t0):
    t_c = cl["contact_rk"]
    i0, i1 = np.searchsorted(st_t, [t_c - 0.06, t_c + 0.06])
    t_cross = None
    for a, b in zip(states[i0:i1], states[i0 + 1:i1]):
        ya, yb = a["tcp"][1], b["tcp"][1]
        if ya <= 0.006 < yb and (b["t"] - a["t"]) < 0.05:
            f = (0.006 - ya) / max(1e-9, yb - ya)
            t_cross = a["t"] + f * (b["t"] - a["t"])
            break
    used = t_cross if t_cross is not None else t_c
    i0, i1 = np.searchsorted(st_t, [used - 0.08, used - 0.006])
    win = states[i0:i1]
    if len(win) < 4:
        return None
    tt = np.array([s["t"] - used for s in win])
    tcp = np.array([s["tcp"] for s in win])
    A2 = np.vstack([np.ones(len(win)), tt, tt * tt]).T
    vel_c = []
    for k in range(3):
        sol, *_ = np.linalg.lstsq(A2, tcp[:, k], rcond=None)
        vel_c.append(float(sol[1] + 2 * sol[2] * (-0.006)))
    fps = []
    for s in win:
        # 车型由 main() 的 use_car 定；face_normal 已是该车拍面法向在臂系的单位向量
        n0, n1, n2 = (float(v) for v in eab.fk(s["position"])["face_normal"])
        if n1 < 0:
            n0, n1, n2 = -n0, -n1, -n2
        fps.append(math.degrees(math.asin(max(-1.0, min(1.0, n2)))))
    A1 = np.vstack([np.ones(len(win)), tt]).T
    sol_p, *_ = np.linalg.lstsq(A1, np.array(fps), rcond=None)
    j0, j1_ = np.searchsorted(bot_t, [t_c - rk_t0 - 0.05, t_c - rk_t0 + 0.05])
    car_vy = None
    vals = [v for v in bot_ch["vy"][j0:j1_] if v is not None]
    if vals:
        car_vy = float(np.median(vals))
    return {"t_contact_rk": t_c, "v_fwd_c": vel_c[1],
            "face_pitch_fk": float(sol_p[0]), "pitch_cmd": cl["pitch_cmd"],
            "car_vy": car_vy}


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="tracker_<ts>.json")
    ap.add_argument("--targets", default="15,16,17")
    ap.add_argument("--pitch-out", type=float, default=19.0, help="目标出球 pitch（°）")
    ap.add_argument("--z-hit", type=float, default=1.35, help="表格名义触球高度（m）")
    ap.add_argument("--plot", default=None, help="输出 png 路径")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    base = str(Path(args.input)).removesuffix(".json")
    data = json.load(open(base + ".json", encoding="utf-8"))
    arm = json.load(open(base + "_arm.json", encoding="utf-8"))
    rk = json.load(open(base + "_rk_tracking.json", encoding="utf-8"))

    # 车型决定臂 FK 链（v0.3/v0.4 是两台不同的臂，TCP 差几厘米）。老 arm JSON 的 tcp 是
    # 按 v0.3 链算的，与本场车型不符就地复算——position 才是原始记录。
    car, car_src = eab.car_for_session(arm, base + ".json")
    eab.use_car(car)
    if arm.get("car") != car:
        n = eab.recompute_tcp(arm["states"]) + eab.recompute_tcp(arm.get("commands"))
        print(f"[fit] 车型 {car}（{car_src}）：arm JSON 的 TCP 按 "
              f"{arm.get('car') or '旧版（v03 链）'} 算的，已就地复算 {n} 行")
    else:
        print(f"[fit] 车型 {car}（{car_src}）")

    ex = ReturnExtractor(data["observations"])
    s1 = sorted([p for p in data["predictions"] if p["stage"] == 1], key=lambda p: p["ht"])
    throws = []
    for p in s1:
        if not throws or p["ht"] - throws[-1][-1]["ht"] > 3.0:
            throws.append([p])
        else:
            throws[-1].append(p)

    states = arm["states"]
    st_t = np.array([s["t"] for s in states])
    clusters = parse_arm_events(arm)
    bot_t = np.array(rk["bot"]["t"])
    bot_ch = rk["bot"]["y"]
    rk_t0 = rk["t0"]

    rets = []
    for i, cluster in enumerate(throws):
        ht = float(np.median([p["ht"] for p in cluster[-3:]]))
        win = [p for p in ex.obs if ht - 1.5 <= p["t"] <= ht + 0.4 and p["z"] >= 0.12]
        anchor = min(win, key=lambda p: p["y"])["t"] if win else ht
        ret = ex.return_at(anchor)
        rets.append({"i": i, "ht": ht, "ret": ret})

    pairs = [c["contact_rk"] - r["ret"]["tHit"] for r in rets if r["ret"] for c in clusters]
    pairs = np.array(pairs)
    o_star = max((np.sum(np.abs(pairs - o) < 0.3), o) for o in pairs)[1]
    o_star = float(np.median(pairs[np.abs(pairs - o_star) < 0.3]))
    print(f"throws={len(throws)}  swings={len(clusters)}  offset(RK-PC)={o_star:.3f}s")

    T = []
    for r in rets:
        ret = r["ret"]
        if not ret:
            print(f"i={r['i']:<2d} 无回球认定（挥空/拒绝/丢跟踪）")
            continue
        inc = ex.incoming_at(ret["tHit"])
        fa = fit_arc_velocity(ret, ex.collect_arc(ret))
        src = fa if fa else ret
        land = propagate_landing([ret["x0"], ret["y0"], ret["z0"],
                                  src["vx"], src["vy"], src["vz"]])
        best = min(clusters, key=lambda c: abs(c["contact_rk"] - (ret["tHit"] + o_star)))
        a = None
        if abs(best["contact_rk"] - (ret["tHit"] + o_star)) < 0.8:
            a = arm_at_contact(states, st_t, best, bot_t, bot_ch, rk_t0)
        if not (inc and land and a and a["car_vy"] is not None):
            print(f"i={r['i']:<2d} 提取不完整，跳过（inc={bool(inc)} land={bool(land)} arm={bool(a)}）")
            continue
        vh = math.hypot(src["vx"], src["vy"])
        T.append({"i": r["i"], "y0": ret["y0"], "z0": ret["z0"],
                  "vin_y": inc["vy"], "s_in": -inc["vy"], "dive": inc["dive"],
                  "vout_y": src["vy"], "vout_speed": src["speed"],
                  "pitch_out": math.degrees(math.atan2(src["vz"], vh)),
                  "v_r": a["v_fwd_c"] + a["car_vy"], "car_vy": a["car_vy"],
                  "fp_cmd": a["pitch_cmd"], "land_y": land["y"], "arc_ok": bool(fa)})

    n = len(T)
    vr = np.array([t["v_r"] for t in T])
    s_in = np.array([t["s_in"] for t in T])
    vout = np.array([t["vout_y"] for t in T])
    viny = np.array([t["vin_y"] for t in T])
    e_i = (vout - vr) / (vr - viny)
    coef2, *_ = np.linalg.lstsq(np.vstack([vr + s_in]).T, (vout - vr), rcond=None)
    e_hat = float(coef2[0])
    rms = float(np.sqrt(np.mean((vout - (vr + e_hat * (vr + s_in))) ** 2)))
    pitch = np.array([t["pitch_out"] for t in T])
    dive = np.array([t["dive"] for t in T])
    cp, *_ = np.linalg.lstsq(np.vstack([np.ones(n), dive]).T, pitch, rcond=None)
    print(f"\nn={n}  e={e_hat:.3f} (逐抛 median {np.median(e_i):.3f} std {np.std(e_i):.3f})  "
          f"vout_y rms={rms:.2f} m/s")
    print(f"pitch_out = {cp[0]:.2f} {cp[1]:+.3f}·dive  残差std={np.std(pitch - cp[0] - cp[1]*dive):.2f}°  "
          f"mean={pitch.mean():.2f}° std={pitch.std():.2f}°")

    s_nom = float(np.median(s_in))
    targets = [float(x) for x in args.targets.split(",")]
    th = args.pitch_out
    print(f"\n=== 所需拍速表（θ*={th}°  z_hit={args.z_hit}  s_in名义={s_nom:.2f}，"
          f"v_r*=(v_out_y*−e·s_in)/(1+e)）===")
    print("y_hit | " + " | ".join(f"落点{t:g}m: v_out*  v_r*" for t in targets))
    grid = np.arange(3.0, 8.01, 0.5)
    law = {t: [] for t in targets}
    for y0 in grid:
        cells = []
        for t in targets:
            v_star = req_speed(y0, args.z_hit, th, t)
            vy_star = v_star * math.cos(math.radians(th))
            vr_star = (vy_star - e_hat * s_nom) / (1 + e_hat)
            law[t].append(vr_star)
            cells.append(f"{v_star:6.2f} {vr_star:6.2f}")
        print(f"{y0:5.1f} | " + " | ".join(cells))

    t_mid = targets[len(targets) // 2]
    print(f"\n=== retro（θ*={th}°, y_land*={t_mid}m）===")
    print("i    y_hit  v_r实际  v_r应有   Δv_r   实际land  car_vy残留")
    for t in T:
        v_star = req_speed(t["y0"], t["z0"], th, t_mid)
        vy_star = v_star * math.cos(math.radians(th))
        vr_star = (vy_star - e_hat * t["s_in"]) / (1 + e_hat)
        print(f"{t['i']:<4d} {t['y0']:5.2f}  {t['v_r']:6.2f}  {vr_star:6.2f}  "
              f"{vr_star - t['v_r']:+6.2f}   {t['land_y']:6.2f}    {t['car_vy']:+5.2f}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
        ax = axes[0]
        sc = ax.scatter([t["y0"] for t in T], [t["land_y"] for t in T],
                        c=[t["car_vy"] for t in T], cmap="coolwarm", s=70, zorder=3)
        for t in T:
            ax.annotate(str(t["i"]), (t["y0"], t["land_y"]),
                        textcoords="offset points", xytext=(5, 4), fontsize=8)
        for tv in targets:
            ax.axhline(tv, ls="--", lw=0.8, color="gray")
        fig.colorbar(sc, ax=ax, label="触球时车残速 vy (m/s)")
        ax.set_xlabel("击球点 y_hit (m)"); ax.set_ylabel("落点 y_land (m)")
        ax.set_title("本场：落点 vs 击球点（色=车残速）")
        ax = axes[1]
        x = vr + s_in
        ax.scatter(x, vout - vr, s=60, zorder=3)
        xx = np.linspace(x.min() - 0.3, x.max() + 0.3, 10)
        ax.plot(xx, e_hat * xx, "k--", lw=1,
                label=f"斜率 e={e_hat:.3f}（rms {rms:.2f} m/s）")
        for t, xi, yi in zip(T, x, vout - vr):
            ax.annotate(str(t["i"]), (xi, yi), textcoords="offset points",
                        xytext=(5, 4), fontsize=8)
        ax.set_xlabel("闭合速度 v_r + s_in (m/s)")
        ax.set_ylabel("分离增量 v_out_y − v_r (m/s)")
        ax.set_title("接触模型：v_out_y = (1+e)·v_r − e·v_in_y")
        ax.legend()
        ax = axes[2]
        for tv in targets:
            ax.plot(grid, law[tv], label=f"目标 {tv:g} m")
        sc = ax.scatter([t["y0"] for t in T], [t["v_r"] for t in T],
                        c=[t["land_y"] for t in T], cmap="viridis", s=70, zorder=3,
                        label="实际(色=实际落点)")
        fig.colorbar(sc, ax=ax, label="实际落点 (m)")
        ax.set_xlabel("击球点 y_hit (m)")
        ax.set_ylabel("触球拍速 v_r (m/s，世界系含车速)")
        ax.set_title(f"所需拍速律（θ*={th}°，名义来球）vs 本场实际")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=130)
        print(f"\nplot -> {args.plot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
