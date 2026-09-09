# -*- coding: utf-8 -*-
"""Arm-only swing black marker vs FK: is the J1 link (spring-twist) compensation right?

Input: `<session>_swing_marker.json` (arm_swing_black_marker.py), `<session>.json`, `<session>_arm.json`.

For every accepted marker frame and each joint stream S (meas = /joint_states, cmd = /tennis/motor_command
= motor reference, link = /tennis/link_command = link-side command, only with compensation on) the
world-frame gap Δ = marker − FK_S is fitted jointly (least squares, 2 time-shift iterations) with

    Δ = R_link6·c + R_car·(d_x, d_y, 0) + v_FK·δ + κ·τ1·t̂

c = rigid marker offset in link6 (marker ≠ modelled TCP), d = car-frame base offset (the 4.5 cm arm-base
shift if the FK lacks it), δ = residual time shift (s; the bridge already aligns PC/RK to a few ms),
κ = compliance along the J1 tangent t̂ per N·m of measured J1 torque (mm/Nm; the raw physics is κ≈−2.5…−3.9,
racket behind the encoder when τ1>0).  The fit is identifiable: a time shift scales with velocity
(same sign during accel/coast/decel) while compliance flips sign with torque (memory
v04-fk-vs-vision-time-shift-0905).

Verdict for a compensated session: the motor reference leads by δ_link = τ/k_s, so the marker should track
FK(link) with κ_link ≈ 0 while κ_meas stays at the raw value and κ_cmd ≈ κ_meas + r/k_s.  |κ_link| ≪ |κ_meas|
means the compensation is right; the sign of κ_link says under (same sign as κ_meas) / over compensation.

Also: torque-binned tangential gaps (the "spatial, not temporal" table), a whip/drag velocity check
(v_marker − v_FK along t̂ vs measured τ̇1), and a plant check (rl_arm/env/arm_plant.py run on the link-side
command grid: τ1 model vs measured effort, marker vs plant head residual).

Usage: python test_src/analyze_arm_swing_marker.py --session tracker_20260907_101010 [--no-plant]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import extract_arm_bag as eab  # noqa: E402
from arm_swing_black_marker import session_paths  # noqa: E402

RL_ARM = Path("D:/tennis-man/rl_arm")
FD_H = 0.0025                  # FK velocity central difference half-step (s)
TAU_BINS = [-80, -20, -5, 5, 15, 25, 40, 80]
VEL_PAIR_MAX_DT = 0.045        # consecutive marker frames usable for a velocity pair (s)


def rot_z(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class Stream:
    """Uniformly interpolable joint rows (t, position) + FK cache."""

    def __init__(self, rows: list[dict], name: str):
        rows = sorted([r for r in rows if r.get("position") and r.get("t") is not None], key=lambda r: float(r["t"]))
        self.name = name
        self.t = np.asarray([float(r["t"]) for r in rows])
        self.q = np.asarray([r["position"] for r in rows], dtype=float)
        self.effort = np.asarray([r["effort"] if r.get("effort") else [np.nan] * 6 for r in rows], dtype=float)

    def ok(self, t: float, max_gap: float = 0.05) -> bool:
        i = int(np.searchsorted(self.t, t))
        return 0 < i < len(self.t) and (self.t[i] - self.t[i - 1]) <= max_gap

    def q_at(self, t: float) -> np.ndarray:
        return np.asarray([np.interp(t, self.t, self.q[:, j]) for j in range(6)])

    def effort1_at(self, t: float) -> float:
        return float(np.interp(t, self.t, self.effort[:, 0]))


def fk_world(q: np.ndarray, car: dict, z_off: float) -> dict:
    f = eab.fk(q)
    Rw = rot_z(car["yaw"])
    joint = f["joints"][0]
    T1 = f["joint_frames"][joint["name"]]
    axis = T1[:3, :3] @ np.asarray(joint["axis"], float)
    tang = np.cross(axis, f["tcp"] - T1[:3, 3])
    r_perp = float(np.linalg.norm(tang))          # lever arm of the marker about the J1 axis
    tang = tang / max(r_perp, 1e-9)
    p = np.asarray([car["x"], car["y"], car["z"]]) + Rw @ (f["tcp"] + np.asarray([0.0, 0.0, -z_off]))
    return {"p": p, "tang": Rw @ tang, "R_link6": Rw @ f["link6"][:3, :3], "Rw": Rw, "tcp": f["tcp"], "tang_arm": tang,
            "r_perp": r_perp}


def coarse_delta_scan(obs: list[dict], stream: Stream, car: dict, z_off: float, pc_minus_rk: float,
                      span_s: float = 0.10, step_s: float = 0.01) -> float:
    """Robustness against a PC/RK offset error beyond the linear v·δ range: scan δ, minimise median |marker − FK(t+δ)|."""
    grid = np.arange(-span_s, span_s + 1e-9, step_s)
    best, best_val = 0.0, np.inf
    for dlt in grid:
        ds = []
        for o in obs:
            t = o["t"] - pc_minus_rk + dlt
            if not stream.ok(t):
                continue
            car_o = o.get("car") or car
            ds.append(float(np.linalg.norm(np.asarray([o["x"], o["y"], o["z"]]) - fk_world(stream.q_at(t), car_o, z_off)["p"])))
        if len(ds) >= 8 and np.median(ds) < best_val:
            best, best_val = float(dlt), float(np.median(ds))
    return best


ALIAS_WINDOW_MM = 60.0         # frame gap deviating this much from its window's median gap = another dark blob
ALIAS_RESID_MIN_MM = 40.0      # robust refit: drop 3-D residuals beyond max(this, 4*MAD)


def reject_aliases(obs: list[dict], meas: Stream, car: dict, z_off: float, pc_minus_rk: float) -> tuple[list[dict], int]:
    """Per window, drop marker frames whose (marker − FK_meas) gap deviates > ALIAS_WINDOW_MM from the window's
    median gap. The black-marker solver's 110 mm anchor gate also admits the racket rim / frame logo when the
    true marker is missed (36% of frames on 20260907_062833 before alignment was fixed)."""
    kept, dropped = [], 0
    by_window: dict[int, list[dict]] = {}
    for o in obs:
        by_window.setdefault(o["window"], []).append(o)
    for rows in by_window.values():
        gaps, valid = [], []
        for o in rows:
            t = o["t"] - pc_minus_rk
            if not meas.ok(t):
                gaps.append(None); continue
            p = fk_world(meas.q_at(t), o.get("car") or car, z_off)["p"]
            gaps.append(np.asarray([o["x"], o["y"], o["z"]]) - p)
        g_ok = [g for g in gaps if g is not None]
        if len(g_ok) < 3:
            kept.extend(rows); continue
        med = np.median(np.asarray(g_ok), axis=0)
        for o, g in zip(rows, gaps):
            if g is None or np.linalg.norm(g - med) <= ALIAS_WINDOW_MM * 1e-3:
                kept.append(o)
            else:
                dropped += 1
    return kept, dropped


def fit_stream(obs: list[dict], stream: Stream, meas: Stream, car: dict, z_off: float, pc_minus_rk: float,
               use_kappa: bool = True, iters: int = 3) -> dict:
    delta = coarse_delta_scan(obs, stream, car, z_off, pc_minus_rk)
    rows = []
    for _ in range(iters):
        A_blocks, y_blocks, rows = [], [], []
        for o in obs:
            t = o["t"] - pc_minus_rk + delta
            if not (stream.ok(t - FD_H) and stream.ok(t + FD_H) and meas.ok(t)):
                continue
            car_o = o.get("car") or car
            w0 = fk_world(stream.q_at(t), car_o, z_off)
            pa = fk_world(stream.q_at(t - FD_H), car_o, z_off)["p"]
            pb = fk_world(stream.q_at(t + FD_H), car_o, z_off)["p"]
            v = (pb - pa) / (2 * FD_H)
            tau1 = meas.effort1_at(t)
            dtau1 = ((meas.effort1_at(t + 0.01) - meas.effort1_at(t - 0.01)) / 0.02
                     if meas.ok(t - 0.01) and meas.ok(t + 0.01) else float("nan"))
            marker = np.asarray([o["x"], o["y"], o["z"]])
            gap = marker - w0["p"]
            cols = [w0["R_link6"], w0["Rw"][:, :2], v[:, None]]
            if use_kappa:
                cols.append((tau1 * w0["tang"])[:, None])
            A_blocks.append(np.hstack(cols))
            y_blocks.append(gap)
            rows.append({"t_rk": t, "gap": gap, "tang": w0["tang"], "v": v, "tau1": tau1, "dtau1": dtau1,
                         "window": o["window"], "R_link6": w0["R_link6"], "Rw": w0["Rw"],
                         "speed": float(np.linalg.norm(v)), "r_perp": w0["r_perp"]})
        if len(rows) < 12:
            return {"stream": stream.name, "n": len(rows), "error": "too few usable frames"}
        A = np.vstack(A_blocks)
        y = np.concatenate(y_blocks)
        x, *_ = np.linalg.lstsq(A, y, rcond=None)
        delta += float(x[5])
    resid = (y - A @ x).reshape(-1, 3)
    rn = np.linalg.norm(resid, axis=1)
    thr = max(ALIAS_RESID_MIN_MM * 1e-3, 4.0 * 1.4826 * float(np.median(np.abs(rn - np.median(rn)))))
    keep = rn <= thr
    n_dropped = int((~keep).sum())
    if n_dropped and keep.sum() >= 12:
        rows = [r for r, k in zip(rows, keep) if k]
        A = A.reshape(-1, 3, A.shape[1])[keep].reshape(-1, A.shape[1])
        y = y.reshape(-1, 3)[keep].reshape(-1)
        x, *_ = np.linalg.lstsq(A, y, rcond=None)
        delta += float(x[5])
        resid = (y - A @ x).reshape(-1, 3)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    tang = np.asarray([r["tang"] for r in rows])
    res_t = np.einsum("ij,ij->i", resid, tang)
    # tangential gap after removing only the rigid/time terms (keeps the torque signature for the binned table)
    x_rigid = x.copy()
    if use_kappa:
        x_rigid[6] = 0.0
    gap_rigid = (y - A @ x_rigid).reshape(-1, 3)
    gap_t = np.einsum("ij,ij->i", gap_rigid, tang)
    taus = np.asarray([r["tau1"] for r in rows])
    bins = []
    for lo, hi in zip(TAU_BINS[:-1], TAU_BINS[1:]):
        sel = (taus >= lo) & (taus < hi)
        if sel.sum() >= 3:
            bins.append({"tau_nm": [lo, hi], "n": int(sel.sum()), "tang_gap_mm_median": float(np.median(gap_t[sel]) * 1e3)})
    out = {"stream": stream.name, "n": len(rows), "n_dropped_resid": n_dropped, "resid_gate_mm": float(thr * 1e3),
           "kappa_mm_per_nm": float(x[6] * 1e3) if use_kappa else None,
           "delta_ms": delta * 1e3, "c_link6_mm": (x[:3] * 1e3).tolist(), "d_car_mm": (x[3:5] * 1e3).tolist(),
           "rms_mm": float(np.sqrt((resid ** 2).sum(axis=1).mean()) * 1e3),
           "rms_tangential_mm": float(np.sqrt((res_t ** 2).mean()) * 1e3),
           "rms_z_mm": float(np.sqrt((resid[:, 2] ** 2).mean()) * 1e3), "r2": r2,
           "tau_abs_max_nm": float(np.abs(taus).max()), "torque_bins": bins,
           "z_gap_median_mm": float(np.median(gap_rigid[:, 2]) * 1e3)}
    out["_rows"] = rows
    out["_resid"] = resid
    return out


ZERO_BINS = [-12, -9, -6, -3, -1, 1, 3, 6, 9, 12]
TAU_RATE_MIN = 20.0            # Nm/s: below this the loading/unloading branch is undefined


def gap_vs_spring(fit: dict) -> dict:
    """Is the tangential racket-vs-encoder gap proportional to torque (spring) or a torque-sign step (gap)?

    Every candidate is fitted jointly with the same rigid/time terms (R_link6·c, R_car·d, v·δ) so the models
    compete on equal footing; BIC on the 3-D residual ranks them.  Signature table:
        spring     e_t = κ·τ                (κ<0: racket behind the encoder for τ>0)
        gap        e_t = g·sign(τ)          (|g| = half the total free play, independent of |τ|)
        gap+spring e_t = g·sign(τ) + κ·τ
        two-slope  e_t = κ1·τ + Δκ·sign(τ)·max(|τ|−knee, 0)   (stiffening / softening)
    A gap also shows as a loading/unloading hysteresis loop whose width at τ≈0 equals 2|g|; a spring's loop
    is ~0.  Per-window κ (fixed rigid terms) checks amplitude dependence: a gap inflates κ in low-torque windows."""
    rows = fit["_rows"]
    n = len(rows)
    tang = np.asarray([r["tang"] for r in rows])
    taus = np.asarray([r["tau1"] for r in rows])
    dtau = np.asarray([r["dtau1"] for r in rows])
    y = np.concatenate([r["gap"] for r in rows])
    A_rigid = np.vstack([np.hstack([r["R_link6"], r["Rw"][:, :2], r["v"][:, None]]) for r in rows])
    r_lever = float(np.median([r["r_perp"] for r in rows]))

    def solve(cols, names):
        A = np.hstack([A_rigid] + [(f[:, None] * tang).reshape(-1, 1) for f in cols])
        x, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = (y - A @ x).reshape(-1, 3)
        rss = float((resid ** 2).sum())
        e_t = np.einsum("ij,ij->i", resid, tang)
        return {"params": {k: float(v) for k, v in zip(names, x[6:])}, "rms_mm": float(np.sqrt(rss / n) * 1e3),
                "rms_t_mm": float(np.sqrt((e_t ** 2).mean()) * 1e3),
                "bic": float(3 * n * math.log(max(rss, 1e-12) / (3 * n)) + A.shape[1] * math.log(3 * n)), "_x": x, "_A": A}

    models = {"spring": solve([taus], ["kappa"]), "gap": solve([np.sign(taus)], ["g"]),
              "gap+spring": solve([np.sign(taus), taus], ["g", "kappa"])}
    for tau0 in (2.0, 5.0):
        models[f"gap tanh{tau0:g}"] = solve([np.tanh(taus / tau0)], ["g"])
    for knee in (10.0, 15.0, 20.0, 30.0):
        models[f"two-slope knee{knee:g}"] = solve([taus, np.sign(taus) * np.maximum(np.abs(taus) - knee, 0.0)],
                                                  ["kappa1", "dkappa_above"])
    best = min(models, key=lambda k: models[k]["bic"])
    # tangential gap with only the rigid/time terms of the spring fit removed (torque signature kept)
    x_rigid = models["spring"]["_x"].copy(); x_rigid[6:] = 0.0
    e_t = np.einsum("ij,ij->i", (y - models["spring"]["_A"] @ x_rigid).reshape(-1, 3), tang) * 1e3   # mm
    fine_bins = []
    for lo, hi in zip(ZERO_BINS[:-1], ZERO_BINS[1:]):
        sel = (taus >= lo) & (taus < hi)
        if sel.sum() >= 3:
            fine_bins.append({"tau_nm": [lo, hi], "n": int(sel.sum()), "tang_gap_mm_median": float(np.median(e_t[sel]))})
    # hysteresis: loading (|τ| rising) vs unloading branches
    rising = np.isfinite(dtau) & (taus * dtau > 0) & (np.abs(dtau) > TAU_RATE_MIN)
    falling = np.isfinite(dtau) & (taus * dtau < 0) & (np.abs(dtau) > TAU_RATE_MIN)
    def branch_fit(sel):
        if sel.sum() < 8:
            return None
        A = np.vstack([taus[sel], np.ones(sel.sum())]).T
        (k, b), *_ = np.linalg.lstsq(A, e_t[sel], rcond=None)
        return {"n": int(sel.sum()), "kappa_mm_per_nm": float(k), "intercept_mm": float(b)}
    near0 = np.abs(taus) < 3.0
    loop = None
    if (near0 & rising).sum() >= 3 and (near0 & falling).sum() >= 3:
        loop = float(np.median(e_t[near0 & rising]) - np.median(e_t[near0 & falling]))
    hyst = {"loading": branch_fit(rising), "unloading": branch_fit(falling), "loop_width_at_zero_mm": loop,
            "note": "a gap gives |loop| ≈ 2|g| and a step in the fine bins; a spring gives loop≈0 and a straight line"}
    # per-window κ with the global rigid terms fixed
    per_window = []
    for w in sorted({r["window"] for r in rows}):
        sel = np.asarray([r["window"] == w for r in rows])
        if sel.sum() < 8 or np.abs(taus[sel]).max() < 5.0:
            continue
        A = np.vstack([taus[sel], np.ones(sel.sum())]).T
        (k, b), *_ = np.linalg.lstsq(A, e_t[sel], rcond=None)
        per_window.append({"window": int(w), "n": int(sel.sum()), "tau_abs_max_nm": float(np.abs(taus[sel]).max()),
                           "kappa_mm_per_nm": float(k)})
    # gap-vs-delay separation: per window fit (δ_w, κ_w) with the rigid terms fixed; a backlash g that the
    # global fit could hide in δ shows up as δ_w ≈ δ_true + g/v_peak (∝ 1/v), a real time offset is speed-independent.
    x_sp = models["spring"]["_x"]
    y_r = (y - A_rigid[:, :5] @ x_sp[:5]).reshape(-1, 3)         # rigid/base terms removed, δ and κ effects kept
    per_window_dk = []
    for w in sorted({r["window"] for r in rows}):
        sel = np.asarray([r["window"] == w for r in rows])
        if sel.sum() < 8:
            continue
        vv = np.asarray([r["v"] for r in rows])[sel]
        cols = np.stack([vv.reshape(-1), (taus[sel][:, None] * tang[sel]).reshape(-1)], axis=1)
        (dw, kw), *_ = np.linalg.lstsq(cols, y_r[sel].reshape(-1), rcond=None)
        vpk = float(np.linalg.norm(vv, axis=1).max())
        per_window_dk.append({"window": int(w), "n": int(sel.sum()), "v_peak_m_s": vpk,
                              "tau_abs_max_nm": float(np.abs(taus[sel]).max()), "delta_ms": float(dw * 1e3),
                              "kappa_mm_per_nm": float(kw * 1e3)})
    delay_vs_speed = None
    if len(per_window_dk) >= 4:
        inv_v = np.asarray([1.0 / max(d["v_peak_m_s"], 0.2) for d in per_window_dk])
        dws = np.asarray([d["delta_ms"] * 1e-3 for d in per_window_dk])
        Aiv = np.vstack([inv_v, np.ones_like(inv_v)]).T
        (slope, icpt), *_ = np.linalg.lstsq(Aiv, dws, rcond=None)
        delay_vs_speed = {"g_equiv_mm": float(slope * 1e3), "delta_true_ms": float(icpt * 1e3),
                          "corr": float(np.corrcoef(inv_v, dws)[0, 1]) if np.std(dws) > 0 else 0.0,
                          "note": "δ_w = δ_true + g_equiv/v_peak: g_equiv ≈ 0 ⇒ no backlash hiding in the delay"}
    kap = models["spring"]["params"]["kappa"]
    g3 = models["gap+spring"]["params"]["g"]
    tau_p90 = float(np.percentile(np.abs(taus), 90))
    verdict = {"kappa_spring_mm_per_nm": kap * 1e3, "gap_g_mm": g3 * 1e3,
               "gap_share_at_p90_torque": float(abs(g3) / max(abs(g3) + abs(kap) * tau_p90, 1e-9)),
               "bic_gap_minus_spring": models["gap"]["bic"] - models["spring"]["bic"],
               "bic_gapspring_minus_spring": models["gap+spring"]["bic"] - models["spring"]["bic"],
               "best_by_bic": best, "lever_m": r_lever, "k_s_equiv_nm_per_rad": float(r_lever / max(abs(kap), 1e-9))}
    share = verdict["gap_share_at_p90_torque"] * 100
    if verdict["bic_gap_minus_spring"] <= 0:
        verdict["reading"] = "间隙为主：换向台阶比比例项更能解释数据（纯间隙模型 BIC 更低）"
    elif abs(g3) * 1e3 >= 5.0 and verdict["bic_gapspring_minus_spring"] < -5.0:
        verdict["reading"] = (f"弹簧为主，叠加约 {abs(g3)*1e3:.0f} mm 的换向台阶/软段（占 p90 力矩处总偏差 {share:.0f}%）"
                              "——看细箱表是否在 τ≈0 跳变、迟滞环宽是否≈2|g|")
    else:
        verdict["reading"] = f"弹簧：偏差随力矩成比例，换向台阶 {abs(g3)*1e3:.0f} mm（<5 mm 或统计上不显著）"
    out = {"n": n, "models": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in models.items()},
           "fine_bins": fine_bins, "hysteresis": hyst, "per_window_kappa": per_window,
           "per_window_delta_kappa": per_window_dk, "delay_vs_speed": delay_vs_speed, "verdict": verdict}
    return out


def velocity_check(obs: list[dict], meas: Stream, car: dict, z_off: float, pc_minus_rk: float, delta_s: float) -> dict:
    pairs = []
    by_window: dict[int, list[dict]] = {}
    for o in obs:
        by_window.setdefault(o["window"], []).append(o)
    for rows in by_window.values():
        rows.sort(key=lambda r: r["t"])
        for a, b in zip(rows[:-1], rows[1:]):
            dt = b["t"] - a["t"]
            if not (0.0 < dt <= VEL_PAIR_MAX_DT):
                continue
            ta, tb = a["t"] - pc_minus_rk + delta_s, b["t"] - pc_minus_rk + delta_s
            if not (meas.ok(ta) and meas.ok(tb)):
                continue
            car_o = a.get("car") or car
            wa, wb = fk_world(meas.q_at(ta), car_o, z_off), fk_world(meas.q_at(tb), car_o, z_off)
            v_m = (np.asarray([b["x"], b["y"], b["z"]]) - np.asarray([a["x"], a["y"], a["z"]])) / dt
            v_f = (wb["p"] - wa["p"]) / dt
            tang = wa["tang"] + wb["tang"]
            tang /= max(np.linalg.norm(tang), 1e-9)
            dtau = (meas.effort1_at(tb) - meas.effort1_at(ta)) / dt
            pairs.append({"t_rk": 0.5 * (ta + tb), "dv_t": float((v_m - v_f) @ tang), "dtau1": dtau,
                          "v_fk_t": float(v_f @ tang), "window": a["window"]})
    if len(pairs) < 6:
        return {"n": len(pairs), "error": "too few consecutive marker frames"}
    dv = np.asarray([p["dv_t"] for p in pairs]); dtau = np.asarray([p["dtau1"] for p in pairs])
    A = np.vstack([dtau, np.ones_like(dtau)]).T
    (slope, offset), *_ = np.linalg.lstsq(A, dv, rcond=None)
    resid = dv - A @ np.asarray([slope, offset])
    fast = np.abs(dtau) > 100.0
    return {"n": len(pairs), "slope_mm_per_nm": float(-slope * 1e3), "offset_m_s": float(offset),
            "rms_m_s": float(np.sqrt((resid ** 2).mean())), "corr": float(np.corrcoef(dtau, dv)[0, 1]),
            "n_fast_dtau": int(fast.sum()),
            "fast_dv_median_m_s": float(np.median(dv[fast])) if fast.any() else None,
            "note": "dv_t = (v_marker − v_FK(meas))·t̂ over consecutive frames; model dv_t = −κ·τ̇1 → slope column is κ"}


def plant_check(session_json: dict, arm: dict, obs: list[dict], meas: Stream, car: dict, z_off: float,
                pc_minus_rk: float, delta_s: float, compensated: bool) -> dict:
    sys.path.insert(0, str(RL_ARM))
    from env.arm_plant import ArmPlant  # noqa: E402
    plant = ArmPlant(noise_scale=0.0, compensated=compensated)
    link_rows = arm.get("link_commands") or arm.get("commands")
    cmd = Stream(link_rows, "link" if arm.get("link_commands") else "cmd")
    dt = plant.dt
    out_windows = []
    A_blocks, y_blocks, gaps_fk = [], [], []
    for w in session_json["windows"]:
        ts = np.arange(w["start_rk"], w["end_rk"], dt)
        if not (cmd.ok(ts[0]) and cmd.ok(ts[-1]) and meas.ok(ts[0]) and meas.ok(ts[-1])):
            continue
        Q = np.column_stack([np.interp(ts, cmd.t, cmd.q[:, j]) for j in range(6)])
        V = np.gradient(Q, dt, axis=0)
        pl = plant.run(ts, Q, V)
        tau_meas = np.asarray([meas.effort1_at(t) for t in ts])
        tau_model = pl["tau1"]
        # cross-correlation lag (model vs measured), ±100 ms
        best = (None, -2.0)
        for k in range(-20, 21):
            a = tau_model[max(0, k):len(ts) + min(0, k)]
            b = tau_meas[max(0, -k):len(ts) + min(0, -k)]
            if len(a) > 20 and np.std(a) > 1e-6 and np.std(b) > 1e-6:
                c = float(np.corrcoef(a, b)[0, 1])
                if c > best[1]:
                    best = (k * dt * 1e3, c)
        ss = float(((tau_meas - tau_meas.mean()) ** 2).sum())
        r2 = 1.0 - float(((tau_meas - tau_model) ** 2).sum()) / ss if ss > 0 else float("nan")
        out_windows.append({"window": w["index"], "tau1_r2": r2, "tau1_rms_nm": float(np.sqrt(((tau_meas - tau_model) ** 2).mean())),
                            "tau1_peak_meas": float(tau_meas[np.argmax(np.abs(tau_meas))]),
                            "tau1_peak_model": float(tau_model[np.argmax(np.abs(tau_model))]),
                            "model_lead_ms": best[0], "lag_peak_cm": float(np.abs(pl["lag"]).max() * 100),
                            "delta_peak_mrad": float(np.abs(pl["delta"]).max() * 1e3)})
        # marker vs plant head (bias + lag on FK(q_act)) — rigid/time terms only, no κ
        for o in obs:
            if o["window"] != w["index"]:
                continue
            t = o["t"] - pc_minus_rk + delta_s
            if not (ts[0] <= t - FD_H and t + FD_H <= ts[-1]):
                continue
            q_act = np.asarray([np.interp(t, ts, pl["q_act"][:, j]) for j in range(6)])
            lag = float(np.interp(t, ts, pl["lag"]))
            f = eab.fk(q_act)
            joint = f["joints"][0]
            T1 = f["joint_frames"][joint["name"]]
            tang = np.cross(T1[:3, :3] @ np.asarray(joint["axis"], float), f["tcp"] - T1[:3, 3])
            tang /= max(np.linalg.norm(tang), 1e-9)
            car_o = o.get("car") or car
            head_arm = f["tcp"] - lag * tang + plant.head_bias()
            Rw = rot_z(car_o["yaw"])
            p_head = np.asarray([car_o["x"], car_o["y"], car_o["z"]]) + Rw @ (head_arm + np.asarray([0.0, 0.0, -z_off]))
            p_fk = np.asarray([car_o["x"], car_o["y"], car_o["z"]]) + Rw @ (f["tcp"] + np.asarray([0.0, 0.0, -z_off]))
            pa = fk_world(cmd.q_at(t - FD_H), car_o, z_off)["p"]; pb = fk_world(cmd.q_at(t + FD_H), car_o, z_off)["p"]
            v = (pb - pa) / (2 * FD_H)
            marker = np.asarray([o["x"], o["y"], o["z"]])
            A_blocks.append(np.hstack([Rw @ f["link6"][:3, :3], Rw[:, :2], v[:, None]]))
            y_blocks.append(marker - p_head)
            gaps_fk.append(marker - p_fk)
    result = {"compensated": compensated, "racket": plant.rk, "windows": out_windows}
    if len(y_blocks) >= 12:
        A = np.vstack(A_blocks); y = np.concatenate(y_blocks)
        x, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = (y - A @ x).reshape(-1, 3)
        yf = np.concatenate(gaps_fk)
        xf, *_ = np.linalg.lstsq(A, yf, rcond=None)
        resid_f = (yf - A @ xf).reshape(-1, 3)
        result["marker_vs_plant_head"] = {"n": len(y_blocks), "rms_mm": float(np.sqrt((resid ** 2).sum(axis=1).mean()) * 1e3),
                                          "delta_ms": float(x[5] * 1e3), "c_link6_mm": (x[:3] * 1e3).tolist(),
                                          "d_car_mm": (x[3:5] * 1e3).tolist()}
        result["marker_vs_fk_motor_no_kappa"] = {"rms_mm": float(np.sqrt((resid_f ** 2).sum(axis=1).mean()) * 1e3),
                                                 "delta_ms": float(xf[5] * 1e3)}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", required=True)
    parser.add_argument("--marker-json", type=Path, default=None)
    parser.add_argument("--no-plant", action="store_true")
    parser.add_argument("--emit-model", action="store_true", help="write <session>_compliance_model.json with plant/yaml suggestions")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    paths = session_paths(args.session)
    marker_path = args.marker_json or paths["output"]
    sm = json.loads(marker_path.read_text(encoding="utf-8"))
    arm = json.loads(paths["arm"].read_text(encoding="utf-8"))
    eab.use_car("v04")
    car, z_off, pc_minus_rk = sm["car_pose"], float(sm["z_offset_m"]), float(sm["clock"]["pc_minus_rk"])
    obs_all = [o for o in sm["racket_observations"] if o.get("black_marker")]
    compensated = bool(sm.get("compensated"))
    meas = Stream(arm["states"], "meas")
    obs, n_alias = reject_aliases(obs_all, meas, car, z_off, pc_minus_rk)
    print(f"[analyze] alias filter: {n_alias}/{len(obs_all)} frames dropped (>{ALIAS_WINDOW_MM:.0f} mm from their window's median gap)")
    streams = [meas, Stream(arm["commands"], "cmd")]
    if arm.get("link_commands"):
        streams.append(Stream(arm["link_commands"], "link"))
    print(f"[analyze] {args.session}: {len(obs)} marker frames, {len(sm['windows'])} windows, compensated={compensated}, "
          f"car pose spread p95 {car.get('spread_p95_m', 0)*1e3:.0f} mm")
    print(f"{'stream':6s} {'n':>4s} {'κ mm/Nm':>9s} {'δ ms':>7s} {'rms mm':>7s} {'rms_t':>6s} {'rms_z':>6s} {'R²':>5s} "
          f"{'c_link6 mm':>22s} {'d_car mm':>16s} {'z gap mm':>9s} | pure-δ fit: δ ms / rms mm")
    fits = {}
    for st in streams:
        fit = fit_stream(obs, st, meas, car, z_off, pc_minus_rk, use_kappa=True)
        pure = fit_stream(obs, st, meas, car, z_off, pc_minus_rk, use_kappa=False)
        fits[st.name] = fit
        if "error" in fit:
            print(f"{st.name:6s} {fit['n']:4d} {fit['error']}")
            continue
        c = fit["c_link6_mm"]; d = fit["d_car_mm"]
        print(f"{st.name:6s} {fit['n']:4d} {fit['kappa_mm_per_nm']:+9.2f} {fit['delta_ms']:+7.1f} {fit['rms_mm']:7.1f} "
              f"{fit['rms_tangential_mm']:6.1f} {fit['rms_z_mm']:6.1f} {fit['r2']:5.2f} "
              f"{c[0]:+6.1f}/{c[1]:+6.1f}/{c[2]:+6.1f} {d[0]:+7.1f}/{d[1]:+7.1f} {fit['z_gap_median_mm']:+9.1f} | "
              f"{pure.get('delta_ms', float('nan')):+.1f} / {pure.get('rms_mm', float('nan')):.1f}")
    for name, fit in fits.items():
        if "torque_bins" in fit:
            print(f"  {name}: tangential gap by measured τ1 (mm, median; rigid+time terms removed, κ kept): " +
                  "  ".join(f"[{b['tau_nm'][0]},{b['tau_nm'][1]}):{b['tang_gap_mm_median']:+.0f}({b['n']})" for b in fit["torque_bins"]))
    gaps = {}
    for name in ("meas", "link"):
        if name in fits and "_rows" in fits[name]:
            gs = gap_vs_spring(fits[name])
            gaps[name] = gs
            print(f"  [{name}] gap-vs-spring (n={gs['n']}, lever {gs['verdict']['lever_m']:.2f} m):")
            for mname, m in gs["models"].items():
                par = " ".join(f"{k}={v*1e3:+.2f}" for k, v in m["params"].items())
                print(f"     {mname:20s} rms {m['rms_mm']:5.1f} mm (t {m['rms_t_mm']:5.1f})  ΔBIC {m['bic'] - gs['models']['spring']['bic']:+8.1f}   {par}  [mm or mm/Nm]")
            print("     fine bins (mm, rigid/time removed): " + "  ".join(
                f"[{b['tau_nm'][0]},{b['tau_nm'][1]}):{b['tang_gap_mm_median']:+.0f}({b['n']})" for b in gs["fine_bins"]))
            h = gs["hysteresis"]
            print(f"     hysteresis: loading {h['loading']} | unloading {h['unloading']} | loop width at τ≈0 {h['loop_width_at_zero_mm']} mm")
            print("     per-window κ (mm/Nm @ peak|τ|): " + "  ".join(
                f"w{w['window']}:{w['kappa_mm_per_nm']:+.2f}@{w['tau_abs_max_nm']:.0f}" for w in gs["per_window_kappa"]))
            if gs.get("delay_vs_speed"):
                dv = gs["delay_vs_speed"]
                print("     per-window (δ, κ) @ v_peak: " + "  ".join(
                    f"w{d['window']}:{d['delta_ms']:+.0f}ms/{d['kappa_mm_per_nm']:+.1f}@{d['v_peak_m_s']:.1f}" for d in gs["per_window_delta_kappa"]))
                print(f"     delay-vs-1/v: δ_w = {dv['delta_true_ms']:+.1f} ms + {dv['g_equiv_mm']:+.1f} mm / v  (corr {dv['corr']:+.2f}) "
                      f"→ backlash hidden in the delay ≈ {abs(dv['g_equiv_mm']):.0f} mm")
            v = gs["verdict"]
            print(f"     VERDICT: {v['reading']}; κ {v['kappa_spring_mm_per_nm']:+.2f} mm/Nm ⇒ k_s≈{v['k_s_equiv_nm_per_rad']:.0f} Nm/rad about J1; "
                  f"best by BIC: {v['best_by_bic']}")
    delta_s = fits["meas"].get("delta_ms", 0.0) * 1e-3 if "meas" in fits else 0.0
    vel = velocity_check(obs, meas, car, z_off, pc_minus_rk, delta_s)
    if "error" not in vel:
        print(f"  whip/drag: dv_t = −κ·τ̇1 fit over {vel['n']} frame pairs → κ {vel['slope_mm_per_nm']:+.2f} mm/Nm, "
              f"offset {vel['offset_m_s']:+.2f} m/s, corr {vel['corr']:+.2f}, rms {vel['rms_m_s']:.2f} m/s; "
              f"|τ̇1|>100 Nm/s pairs {vel['n_fast_dtau']} median dv {vel['fast_dv_median_m_s']}")
    else:
        print(f"  whip/drag: {vel['error']} (n={vel['n']})")
    verdict = {}
    km = fits.get("meas", {}).get("kappa_mm_per_nm")
    kl = fits.get("link", {}).get("kappa_mm_per_nm")
    kc = fits.get("cmd", {}).get("kappa_mm_per_nm")
    if km is not None:
        verdict["kappa_meas"] = km
        verdict["reference_plant_kappa"] = "−3.0 mm/Nm (|τ1|≤30) / −1.2 above (plant_params.json, sign: racket behind encoder for τ1>0)"
    if compensated and kl is not None and km is not None:
        ratio = abs(kl) / max(abs(km), 1e-6)
        verdict.update({"kappa_link": kl, "kappa_cmd": kc, "residual_ratio": ratio,
                        "reading": ("补偿有效：黑标跟链侧命令，残余柔度 %.0f%%" % (ratio * 100)) if ratio < 0.3 else
                                   ("补偿不足（同号残余 %.0f%%）" % (ratio * 100) if kl * km > 0 else "补偿过头（反号残余 %.0f%%）" % (ratio * 100))})
        print(f"  VERDICT: κ_meas {km:+.2f} → κ_link {kl:+.2f} mm/Nm (κ_cmd {kc:+.2f}); {verdict['reading']}")
    elif km is not None:
        print(f"  (uncompensated session) κ_meas {km:+.2f} mm/Nm vs plant −3.0/−1.2; κ_cmd {kc:+.2f} "
              f"(command→encoder tracking adds {kc - km:+.2f})")
    plant = None
    if not args.no_plant:
        try:
            plant = plant_check(sm, arm, obs, meas, car, z_off, pc_minus_rk, delta_s, compensated)
            for w in plant["windows"]:
                print(f"  plant window {w['window']}: τ1 model R² {w['tau1_r2']:.2f} rms {w['tau1_rms_nm']:.1f} Nm, "
                      f"peak meas/model {w['tau1_peak_meas']:+.0f}/{w['tau1_peak_model']:+.0f} Nm, model lead {w['model_lead_ms']} ms, "
                      f"lag peak {w['lag_peak_cm']:.1f} cm, δ peak {w['delta_peak_mrad']:.0f} mrad")
            if "marker_vs_plant_head" in plant:
                a, b = plant["marker_vs_plant_head"], plant["marker_vs_fk_motor_no_kappa"]
                print(f"  plant head vs marker: rms {a['rms_mm']:.1f} mm (δ {a['delta_ms']:+.1f} ms, d_car {a['d_car_mm'][0]:+.0f}/{a['d_car_mm'][1]:+.0f} mm) "
                      f"| FK(motor cmd) no-κ: rms {b['rms_mm']:.1f} mm — plant should be ≤ the κ fit above")
        except Exception as exc:  # plant is optional diagnostics
            print(f"  plant check skipped: {exc!r}")
    result = {"session": args.session, "compensated": compensated, "n_obs": len(obs),
              "fits": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in fits.items()},
              "gap_vs_spring": gaps, "velocity": vel, "verdict": verdict, "plant": plant}
    if args.emit_model and "meas" in gaps:
        gs = gaps["meas"]; v = gs["verdict"]; two = gs["models"]
        knee_best = min((k for k in two if k.startswith("two-slope")), key=lambda k: two[k]["bic"])
        kp = two[knee_best]["params"]
        model = {"session": args.session, "compensated": compensated, "n": gs["n"], "reading": v["reading"],
                 "kappa_mm_per_nm": v["kappa_spring_mm_per_nm"], "gap_g_mm": v["gap_g_mm"],
                 "loop_width_at_zero_mm": gs["hysteresis"]["loop_width_at_zero_mm"],
                 "two_slope": {"knee_nm": float(knee_best.split("knee")[1]), "kappa_low_mm_per_nm": kp["kappa1"] * 1e3,
                               "kappa_high_mm_per_nm": (kp["kappa1"] + kp["dkappa_above"]) * 1e3},
                 "lever_m": v["lever_m"], "k_s_equiv_nm_per_rad": v["k_s_equiv_nm_per_rad"],
                 "delta_ms": fits["meas"].get("delta_ms"), "z_gap_median_mm": fits["meas"].get("z_gap_median_mm"),
                 "d_car_mm": fits["meas"].get("d_car_mm"), "c_link6_mm": fits["meas"].get("c_link6_mm"),
                 "suggest": {"rl_arm/assets/v04/plant_params.json racket": {
                                 "kappa_low_mm_per_nm": round(abs(kp["kappa1"]) * 1e3, 2),
                                 "tau_knee_nm": float(knee_best.split("knee")[1]),
                                 "kappa_high_mm_per_nm": round(abs(kp["kappa1"] + kp["dkappa_above"]) * 1e3, 2),
                                 "gap_sign_m (new term if |g|≥5mm)": round(abs(v["gap_g_mm"]) * 1e-3, 4)},
                             "arm_controller config/cars/v04.yaml link_compliance.k_s_nm_per_rad": round(v["k_s_equiv_nm_per_rad"], 0),
                             "note": "k_s here is r/κ (static, about J1); keep I_a from the torque-dynamics fit (ω²·I_a should match)"}}
        mp = paths["output"].with_name(f"{args.session}_compliance_model.json")
        mp.write_text(json.dumps(model, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[analyze] model suggestions → {mp}")
    out = args.output or paths["output"].with_name(f"{args.session}_swing_marker_analysis.json")
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[analyze] saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
