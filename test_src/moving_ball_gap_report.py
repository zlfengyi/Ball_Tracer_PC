# -*- coding: utf-8 -*-
"""动球 × 静相机 实验分析（四象限最后一格）。v2：不依赖 AprilTag。

前提：RK 侧 run_moving_ball_static_camera.sh 定角保持相机，手抛球横穿视场；
PC 侧 run_tracker.ps1 照常录（→ _rk_tracking.json）。

对齐策略（tag 可无）：
  1) 时轴：|v|(t) 速度形状 + z(t) 高度形状（均为坐标系不变量）扫 bias；
  2) 空间：2D 刚体 Procrustes 把 RK 世界系配到 PC 世界系（用低 φ̇ 样本加权，
     常数方位偏置会被配准吸收——判决只看随 φ̇ 的斜率，不看常数项）；
  3) 轴心 = 配准后的 bot 位姿 + R(yaw)·(0, 0.365)。
判决回归：
  A. Δ方位 vs φ̇ → δ_eff(ms)   （相机 ω≈0 时若复现 ~20-30ms → 球运动机制坐实）
  B. Δz  vs −vz → δ_z(ms)     （≈δ_eff → 统一时间戳类；≈0 → 方位/水平专属）
  C. Δr  vs −ṙ → δ_r(ms)      （径向/双目通道）
"""
import argparse
import bisect
import csv
import json
import math

DEG = 180.0 / math.pi
PIVOT = 0.365


def interp(ts, ys, t, mg=0.08):
    i = bisect.bisect_left(ts, t)
    if i <= 0 or i >= len(ts):
        return None
    t0, t1 = ts[i - 1], ts[i]
    if t1 - t0 > mg:
        return None
    w = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
    return ys[i - 1] * (1 - w) + ys[i] * w


def median(a):
    a = sorted(a)
    return a[len(a) // 2] if a else float("nan")


def linreg(xy):
    n = len(xy)
    if n < 6:
        return None
    sx = sum(p[0] for p in xy); sy = sum(p[1] for p in xy)
    sxx = sum(p[0] * p[0] for p in xy); sxy = sum(p[0] * p[1] for p in xy)
    d = n * sxx - sx * sx
    if abs(d) < 1e-12:
        return None
    b = (n * sxy - sx * sy) / d
    a = (sy - b * sx) / n
    rms = math.sqrt(sum((p[1] - a - b * p[0]) ** 2 for p in xy) / n)
    return a, b, rms, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="tracker_<ts>.json 路径")
    args = ap.parse_args()
    base = args.input[:-5] if args.input.endswith(".json") else args.input
    tr = json.load(open(args.input))
    rk = json.load(open(base + "_rk_tracking.json"))

    # ── PC 观测（相对轴：不重要，对齐吸收）；同样做轨迹一致性清洗（多球跳变会毁掉插值与评分）──
    raw = sorted(({"t": o["t"], "x": o["x"], "y": o["y"], "z": o["z"]}
                  for o in tr["observations"]), key=lambda o: o["t"])
    t_ref = raw[0]["t"]
    for o in raw:
        o["t"] -= t_ref
    pc_tracks, cur = [], [raw[0]]
    for o in raw[1:]:
        dt = o["t"] - cur[-1]["t"]
        d = math.dist((o["x"], o["y"], o["z"]),
                      (cur[-1]["x"], cur[-1]["y"], cur[-1]["z"]))
        if dt > 0.35 or (dt > 0 and d / dt > 9.0):
            pc_tracks.append(cur)
            cur = [o]
        else:
            cur.append(o)
    pc_tracks.append(cur)
    pc_tracks = [tk for tk in pc_tracks if len(tk) >= 4]
    obs = sorted((o for tk in pc_tracks for o in tk), key=lambda o: o["t"])
    OT = [o["t"] for o in obs]
    OX = [o["x"] for o in obs]; OY = [o["y"] for o in obs]; OZ = [o["z"] for o in obs]
    print("PC obs n=%d(清洗后 %d) span %.1fs | car_locs n=%d%s" % (
        len(raw), len(obs), OT[-1] - OT[0], len(tr.get("car_locs", [])),
        "（无 tag，走无 tag 配准）" if not tr.get("car_locs") else ""))

    def pc_speed(t):
        pa = [interp(OT, A, t - 0.033) for A in (OX, OY, OZ)]
        pb = [interp(OT, A, t + 0.033) for A in (OX, OY, OZ)]
        if None in pa or None in pb:
            return None
        return math.dist(pa, pb) / 0.066

    # ── RK 观测 ────────────────────────────────────────────────
    W = rk["world"]["y"]; WT = rk["world"]["t"]
    n = len(WT)
    CM = rk.get("camera_motor", {})
    if CM:
        cmp_, cmt = CM["y"]["position"], CM["t"]
        oms = [abs(cmp_[i + 5] - cmp_[i]) / (cmt[i + 5] - cmt[i]) * DEG
               for i in range(0, len(cmt) - 6, 5) if cmt[i + 5] > cmt[i]]
        print("相机 |ω| 中位 %.1f°/s p95 %.1f°/s（应≈0）"
              % (median(oms), sorted(oms)[int(len(oms) * 0.95)]))
    bot_x = median(W["bot_x"]); bot_y = median(W["bot_y"]); bot_yaw = median(W["bot_yaw"])

    def rk_v(i):
        j, k = max(0, i - 1), min(n - 1, i + 1)
        dt = WT[k] - WT[j]
        if dt <= 0:
            return 0.0, 0.0
        v3 = math.dist((W["x"][j], W["y"][j], W["z"][j]),
                       (W["x"][k], W["y"][k], W["z"][k])) / dt
        vz = (W["z"][k] - W["z"][j]) / dt
        return v3, vz

    # RK 流是"每帧唯一球"的混合物（静球/手持/飞行轮流上榜）——先按轨迹一致性切开：
    # 相邻点隐含速度 >9m/s 或断档 >0.35s 断开；子轨迹中位速度 0.8~8m/s 才算抛球轨迹。
    def d3(i, j):
        return math.dist((W["x"][i], W["y"][i], W["z"][i]),
                         (W["x"][j], W["y"][j], W["z"][j]))

    tracks, cur = [], [0]
    for i in range(1, n):
        j = cur[-1]
        dt = WT[i] - WT[j]
        if dt > 0.35 or (dt > 0 and d3(i, j) / dt > 9.0):
            tracks.append(cur)
            cur = [i]
        else:
            cur.append(i)
    tracks.append(cur)
    mov, rk_track_segs = [], []
    for tk in tracks:
        if len(tk) < 5:
            continue
        sp = [d3(tk[k], tk[k + 1]) / max(1e-3, WT[tk[k + 1]] - WT[tk[k]])
              for k in range(len(tk) - 1)]
        ms = median(sp)
        if 0.8 <= ms <= 8.0:
            mov += tk
            rk_track_segs.append((WT[tk[0]], WT[tk[-1]], len(tk), ms))
    print("RK obs n=%d → 干净抛球轨迹 %d 条 / 样本 %d" % (n, len(rk_track_segs), len(mov)))
    for a, b, m, ms in rk_track_segs:
        print("   RK 轨迹 t %6.1f +%4.1fs n=%3d 速度中位 %.1f" % (a, b - a, m, ms))
    if len(mov) < 15:
        raise SystemExit("RK 干净抛球样本太少")

    # ── 已知变换路径：车静止 + tag 可见 + bot 位姿恒定 → RK→PC 变换是解析已知的，
    #    免拟合免混叠：p_pc = car_tag ⊕ R(yaw_tag−yaw_bot)·(p_rk − bot)。时轴用全 3D 残差扫。
    cars = tr.get("car_locs", [])
    have_tag = len(cars) >= 20
    if have_tag:
        def medk(k):
            return median([c[k] for c in cars])
        cx_t, cy_t, cyaw_t = medk("x"), medk("y"), medk("yaw")
        bstd = max(abs(v - bot_x) for v in W["bot_x"][:: max(1, n // 200)])
        if bstd > 0.05:
            have_tag = False
            print("⚠ bot 位姿非恒定（漂 %.0fmm），退回免 tag 配准路径" % (bstd * 1000))
    if have_tag:
        dyaw = cyaw_t - bot_yaw
        cth, sth = math.cos(dyaw), math.sin(dyaw)

        def rk2pc_known(x, y):
            dx, dy = x - bot_x, y - bot_y
            return cx_t + cth * dx - sth * dy, cy_t + sth * dx + cth * dy

        def score3d(b):
            errs = []
            for i in mov:
                t = WT[i] + b
                px = interp(OT, OX, t); py = interp(OT, OY, t); pz = interp(OT, OZ, t)
                if px is None:
                    continue
                qx, qy = rk2pc_known(W["x"][i], W["y"][i])
                errs.append(math.dist((qx, qy, W["z"][i]), (px, py, pz)))
            if len(errs) < 12:
                return None
            errs.sort()
            return errs[int(len(errs) * 0.3)], len(errs)

        best3 = (None, 1e9, 0)
        for k in range(-1600, 1601):
            r = score3d(k * 0.05)
            if r and r[0] < best3[1]:
                best3 = (k * 0.05, r[0], r[1])
        if best3[0] is None or best3[1] > 0.25:
            raise SystemExit("已知变换时轴扫描失败（最优 %.2f）" % best3[1])
        fine3 = (best3[0], 1e9)
        for k in range(-12, 13):
            r = score3d(best3[0] + k * 0.005)
            if r and r[0] < fine3[1]:
                fine3 = (best3[0] + k * 0.005, r[0])
        bias = fine3[0]
        r = score3d(bias)
        print("时轴(已知变换全3D): PC t = RK t %+.3fs  30分位残差 %.0fmm n=%d"
              % (bias, fine3[1] * 1000, r[1]))
        rk2pc = rk2pc_known
        px_piv = cx_t - math.sin(cyaw_t) * PIVOT
        py_piv = cy_t + math.cos(cyaw_t) * PIVOT
        print("轴心(tag): (%.3f, %.3f)" % (px_piv, py_piv))
    # ── 免 tag 后备：事件（抛球段）配对生成候选，再用形状评分 ─────────
    if not have_tag:
        def score(bias, need=15):
            # 30 分位（抗多球/杂段污染）：一条被污染的轨迹不再能毁掉正确偏移
            errs = []
            for i in mov[:: max(1, len(mov) // 600)]:
                t = WT[i] + bias
                vp = pc_speed(t)
                zp = interp(OT, OZ, t)
                if vp is None or zp is None:
                    continue
                v3, _ = rk_v(i)
                errs.append(abs(v3 - vp) + abs(W["z"][i] - zp))
            if len(errs) < need:
                return 1e9, len(errs)
            errs.sort()
            return errs[int(len(errs) * 0.3)], len(errs)

        def segments(ts, gap=0.4, min_n=5):
            segs = [[ts[0]]]
            for t in ts[1:]:
                if t - segs[-1][-1] < gap:
                    segs[-1].append(t)
                else:
                    segs.append([t])
            return [(s[0], s[-1]) for s in segs if len(s) >= min_n]

        # PC 抛球段：观测本身就是运动球（静物被 stationary filter 处理）；速度>1 的段
        pc_seg = []
        for a, b in segments(OT):
            mid = (a + b) / 2
            v = pc_speed(mid) or pc_speed(a + 0.1) or 0
            if v > 0.8 or (b - a) > 0.25:
                pc_seg.append((a, b))
        print("事件段: PC %d 段, RK %d 条轨迹" % (len(pc_seg), len(rk_track_segs)))
        cands = set()
        for pa, _pb in pc_seg:
            for ra, _rb, _n2, _ms in rk_track_segs:
                cands.add(round(pa - ra, 2))
        # 两边"看到同一颗球"的入场时刻可差零点几秒（各自视场不同）——候选周围 ±1.5s 粗扫
        best = (None, 1e9, 0)
        tried = set()
        for b0 in sorted(cands):
            for k in range(-150, 151):
                b = round(b0 + k * 0.01, 2)
                if b in tried:
                    continue
                tried.add(b)
                e, m = score(b)
                if e < best[1]:
                    best = (b, e, m)
        if best[0] is None or best[1] > 0.5:
            raise SystemExit("时轴对齐失败（最优残差 %.2f）：两边看到的抛球段对不上？"
                             "多抛带弧线的球、确认 PC 相机拍得到抛球区" % best[1])
        fine = (best[0], 1e9)
        for k in range(-8, 9):
            b0 = best[0] + k * 0.002
            e, m = score(b0)
            if e < fine[1]:
                fine = (b0, e)
        bias = fine[0]
        print("时轴: PC t = RK t %+.3fs（形状残差中位 %.0fmm 级, n=%d；候选 %d 个）"
              % (bias, fine[1] * 1000, score(bias)[1], len(cands)))
        if fine[1] > 0.30:
            print("⚠ 对齐质量一般——结果谨慎解读；多抛带弧线的球能改善")

        # ── 空间配准：2D 刚体 (θ, tx, ty)，低速/低φ̇ 加权，3σ 剔除迭代 ──
        pairs = []
        for i in mov:
            t = WT[i] + bias
            px = interp(OT, OX, t); py = interp(OT, OY, t)
            if px is None or py is None:
                continue
            v3, vz = rk_v(i)
            pairs.append({"i": i, "rx": W["x"][i], "ry": W["y"][i],
                          "px": px, "py": py, "w": 1.0 / (1.0 + v3)})
        if len(pairs) < 12:
            raise SystemExit("配准样本不足 (%d)" % len(pairs))
        use = pairs
        th = tx = ty = 0.0
        for it in range(4):
            sw = sum(p["w"] for p in use)
            rcx = sum(p["rx"] * p["w"] for p in use) / sw
            rcy = sum(p["ry"] * p["w"] for p in use) / sw
            pcx = sum(p["px"] * p["w"] for p in use) / sw
            pcy = sum(p["py"] * p["w"] for p in use) / sw
            sn = sum(p["w"] * ((p["rx"] - rcx) * (p["py"] - pcy) - (p["ry"] - rcy) * (p["px"] - pcx)) for p in use)
            cs = sum(p["w"] * ((p["rx"] - rcx) * (p["px"] - pcx) + (p["ry"] - rcy) * (p["py"] - pcy)) for p in use)
            th = math.atan2(sn, cs)
            c, s = math.cos(th), math.sin(th)
            tx = pcx - (c * rcx - s * rcy)
            ty = pcy - (s * rcx + c * rcy)
            res = []
            for p in pairs:
                ex = c * p["rx"] - s * p["ry"] + tx - p["px"]
                ey = s * p["rx"] + c * p["ry"] + ty - p["py"]
                p["e"] = math.hypot(ex, ey)
                res.append(p["e"])
            thr = 3.0 * median(res) + 0.02
            use = [p for p in pairs if p["e"] < thr]
        print("配准: θ=%.2f° t=(%.3f, %.3f)  内点 %d/%d  残差中位 %.0fmm"
              % (th * DEG, tx, ty, len(use), len(pairs), median([p["e"] for p in use]) * 1000))

        def rk2pc(x, y):
            c, s = math.cos(th), math.sin(th)
            return c * x - s * y + tx, s * x + c * y + ty

        # 轴心（PC 系）
        hx = bot_x - math.sin(bot_yaw) * PIVOT
        hy = bot_y + math.cos(bot_yaw) * PIVOT
        px_piv, py_piv = rk2pc(hx, hy)
        print("轴心(PC系): (%.3f, %.3f)" % (px_piv, py_piv))

    # ── 逐样本指标 ─────────────────────────────────────────────
    rows = []
    for i in mov:
        t = WT[i] + bias
        px = interp(OT, OX, t); py = interp(OT, OY, t); pz = interp(OT, OZ, t)
        pa = [interp(OT, A, t - 0.025) for A in (OX, OY, OZ)]
        pb = [interp(OT, A, t + 0.025) for A in (OX, OY, OZ)]
        if None in (px, py, pz) or None in pa or None in pb:
            continue

        def geo(x, y):
            dx, dy = x - px_piv, y - py_piv
            return math.atan2(dx, dy) * DEG, math.hypot(dx, dy)
        phi_pc, r_pc = geo(px, py)
        phi_a, r_a = geo(pa[0], pa[1])
        phi_b, r_b = geo(pb[0], pb[1])
        rx, ry = rk2pc(W["x"][i], W["y"][i])
        phi_rk, r_rk = geo(rx, ry)
        dphi = (phi_rk - phi_pc + 180.0) % 360.0 - 180.0
        rows.append({"t": WT[i], "phidot": (phi_b - phi_a) / 0.05,
                     "rdot": (r_b - r_a) / 0.05, "vz": (pb[2] - pa[2]) / 0.05,
                     "dphi": dphi, "dr": (r_rk - r_pc) * 1000,
                     "dz": (W["z"][i] - pz) * 1000, "r": r_pc})
    print("有效对比样本 n=%d" % len(rows))
    if len(rows) < 12:
        raise SystemExit("样本不足")

    # ── 判决回归 ───────────────────────────────────────────────
    fast = [r for r in rows if abs(r["phidot"]) > 30]
    print("\nA. Δ方位 vs φ̇（|φ̇|>30°/s, n=%d）——只看斜率，常数项已被配准吸收:" % len(fast))
    for lo, hi in ((-400, -120), (-120, -30), (30, 120), (120, 400)):
        ss = [r for r in fast if lo <= r["phidot"] < hi]
        if len(ss) < 4:
            continue
        dts = [-r["dphi"] / r["phidot"] * 1000 for r in ss]
        print("   φ̇ %4d~%4d: n=%3d  Δφ中位 %+6.2f°  隐含δ中位 %+6.1fms"
              % (lo, hi, len(ss), median([r["dphi"] for r in ss]), median(dts)))
    reg = linreg([(-r["phidot"], r["dphi"]) for r in fast])
    if reg:
        a, b, rms, m = reg
        print("   回归: Δφ = %.1fms×(−φ̇) %+0.2f°  RMS %.2f° n=%d   ← δ_eff(球运动)"
              % (b * 1000, a, rms, m))
    regz = linreg([(-r["vz"] * 1000, r["dz"]) for r in rows if abs(r["vz"]) > 0.8])
    print("\nB. Δz vs −vz:")
    if regz:
        a, b, rms, m = regz
        print("   回归: Δz = %.1fms×(−vz) %+0.0fmm  RMS %.0fmm n=%d   ← δ_z"
              % (b * 1000, a, rms, m))
        print("   判读: δ_z ≈ δ_eff → 统一时间戳；δ_z≈0 且 δ_eff 大 → 方位/水平专属机制")
    else:
        print("   |vz|>0.8 样本不足——补几次带弧线的抛球")
    regr = linreg([(-r["rdot"] * 1000, r["dr"]) for r in rows if abs(r["rdot"]) > 0.8])
    print("\nC. Δr vs −ṙ:")
    if regr:
        a, b, rms, m = regr
        print("   回归: Δr = %.1fms×(−ṙ) %+0.0fmm  RMS %.0fmm n=%d   ← δ_r（径向/双目）"
              % (b * 1000, a, rms, m))
    else:
        print("   |ṙ|>0.8 样本不足——补 1-2 次朝相机方向的抛球")

    print("\nE. Δ距离 vs 距离（双目'远处按比例不准'判读：若 ∝Z² 应逐箱放大）:")
    for lo, hi in ((0.8, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.5), (5.5, 8.0)):
        ss = [r for r in rows if lo <= r["r"] < hi]
        if len(ss) < 4:
            continue
        drs = sorted(x["dr"] for x in ss)
        dphis = sorted(x["dphi"] for x in ss)
        print("   r %3.1f~%3.1fm: n=%3d  Δr中位 %+5.0fmm (p10 %+5.0f p90 %+5.0f)  Δ方位中位 %+6.2f°"
              % (lo, hi, len(ss), drs[len(drs)//2], drs[len(drs)//10], drs[9*len(drs)//10],
                 dphis[len(dphis)//2]))

    print("\nD. 分抛：")
    segs = []
    for r in rows:
        if segs and r["t"] - segs[-1][-1]["t"] < 0.4:
            segs[-1].append(r)
        else:
            segs.append([r])
    for k, s in enumerate(segs):
        if len(s) < 4:
            continue
        print("   #%-2d t %6.1f~%6.1f n=%3d  max|φ̇| %4.0f°/s  Δφ中位 %+6.2f°  Δz中位 %+5.0fmm  Δr中位 %+5.0fmm"
              % (k + 1, s[0]["t"], s[-1]["t"], len(s), max(abs(r["phidot"]) for r in s),
                 median([r["dphi"] for r in s]), median([r["dz"] for r in s]),
                 median([r["dr"] for r in s])))

    out_csv = base + "_moving_ball_gap.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_rk", "phidot_dps", "rdot_mps", "vz_mps", "dphi_deg", "dr_mm", "dz_mm", "r_m"])
        for r in rows:
            w.writerow([round(r["t"], 4), round(r["phidot"], 1), round(r["rdot"], 2),
                        round(r["vz"], 2), round(r["dphi"], 3), round(r["dr"], 1),
                        round(r["dz"], 1), round(r["r"], 3)])
    print("\n明细: %s" % out_csv)


if __name__ == "__main__":
    main()
