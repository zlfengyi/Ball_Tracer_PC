# -*- coding: utf-8 -*-
"""Black-marker measurement over arm-only swings (car stationary, no ball).

Purpose: verify the J1 link (spring-twist) compensation of arm_controller_cpp
(`run_arm_cpp_ready.sh --car v04 --link-comp`) against the four-camera black
marker.  Unlike `racket_ht_black_marker.py` there are no report tables and no
FinalHT rows:

* swing windows come from the arm bag (`_arm.json`: J1 command speed);
* PC<->RK alignment comes from the tracker's recorded /bot_state clock bridge
  (`config.rk_clock_bridge.pc_minus_rk`, accurate to a few ms; the analysis
  refits the residual time shift jointly with compliance, so this only has to
  put the FK search anchor within the 110 mm gate);
* the car pose is the session median of the AprilTag localisation (the car does
  not move; spread is reported).

Per video frame inside a window: FK anchor (measured joints -> extract_arm_bag
FK -> world), then the same candidate/solve ladder as the HT script (4-cam/FK,
4-cam/trajectory, 3-cam recoveries).  Output `<session>_swing_marker.json`
feeds `analyze_arm_swing_marker.py`.

Usage:
  python test_src/arm_swing_black_marker.py --session tracker_20260907_101010
  (expects tracker_output/<s>/<s>.json, <s>.mp4, <s>_arm.json; the arm json is
   produced by extract_arm_bag.py from the session rosbag as usual)
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import racket_ht_black_marker as ht  # noqa: E402  (shared camera / marker / solver code)

TRACKER_OUTPUT = HERE.parent / "tracker_output"


def _solve_marker_fast(candidates, cameras, expected_xyz_mm, serials, max_expected_mm):
    """Drop-in for racket_ht_black_marker._solve_marker with identical acceptance semantics:
    the pairwise-DLT seed stage (6 camera pairs x 12x12 candidates = 864 SVDs + 4 projections each) is
    vectorised; the refinement/LOO/gate stage over the associated combos is unchanged (it only runs for the
    few combos that survive association, so it was never the bottleneck)."""
    if any(not candidates.get(serial) for serial in serials):
        return None
    und = {}
    uvs = {}
    for serial in serials:
        cam = cameras[serial]
        pts = np.asarray([c.uv for c in candidates[serial]], dtype=np.float64).reshape(-1, 1, 2)
        uvs[serial] = pts.reshape(-1, 2)
        und[serial] = cv2.undistortPoints(pts, cam.K, cam.D, P=cam.K).reshape(-1, 2)
    proposed = {}
    gate = 1.5 * max_expected_mm
    for li, ls in enumerate(serials):
        for rs in serials[li + 1:]:
            Pl, Pr = cameras[ls].P, cameras[rs].P
            ul, ur = und[ls], und[rs]
            nl, nr = len(ul), len(ur)
            if nl == 0 or nr == 0:
                continue
            # rows for every (left, right) pair: 4 x 4 systems
            A = np.empty((nl, nr, 4, 4))
            A[:, :, 0, :] = ul[:, 0][:, None, None] * Pl[2][None, None, :] - Pl[0][None, None, :]
            A[:, :, 1, :] = ul[:, 1][:, None, None] * Pl[2][None, None, :] - Pl[1][None, None, :]
            A[:, :, 2, :] = ur[:, 0][None, :, None] * Pr[2][None, None, :] - Pr[0][None, None, :]
            A[:, :, 3, :] = ur[:, 1][None, :, None] * Pr[2][None, None, :] - Pr[1][None, None, :]
            A = A.reshape(-1, 4, 4)
            try:
                _, _, vt = np.linalg.svd(A)
            except np.linalg.LinAlgError:
                continue
            h = vt[:, -1, :]
            ok = np.abs(h[:, 3]) >= 1e-12
            seeds = np.full((len(h), 3), np.nan)
            seeds[ok] = h[ok, :3] / h[ok, 3:4]
            ok &= np.linalg.norm(seeds - expected_xyz_mm, axis=1) <= gate
            idx = np.nonzero(ok)[0]
            if len(idx) == 0:
                continue
            seeds = seeds[idx]
            choice = np.empty((len(idx), len(serials)), dtype=int)
            total = np.zeros(len(idx))
            alive = np.ones(len(idx), dtype=bool)
            for k, serial in enumerate(serials):
                cam = cameras[serial]
                proj, _ = cv2.projectPoints(seeds.reshape(-1, 1, 3), cam.rvec, cam.t, cam.K, cam.D)
                proj = proj.reshape(-1, 2)
                d = np.linalg.norm(proj[:, None, :] - uvs[serial][None, :, :], axis=2)   # (N, n_cand)
                best = np.argmin(d, axis=1)
                dmin = d[np.arange(len(idx)), best]
                alive &= dmin <= ht.MARKER_ASSOCIATION_PX
                choice[:, k] = best
                total += np.where(alive, dmin, 0.0)
            for j in np.nonzero(alive)[0]:
                key = tuple(int(v) for v in choice[j])
                proposed[key] = min(proposed.get(key, math.inf), float(total[j]))
    best = None
    for choice, _ in sorted(proposed.items(), key=lambda item: item[1])[:32]:
        pixels = {serial: candidates[serial][ci].uv for serial, ci in zip(serials, choice)}
        try:
            fit = ht._triangulate_refined(pixels, cameras)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if fit.max_px > ht.MARKER_MAX_REPROJ_PX:
            continue
        expected_distance = float(np.linalg.norm(fit.xyz_mm - expected_xyz_mm))
        if expected_distance > max_expected_mm:
            continue
        loo_delta, heldout = {}, {}
        try:
            for dropped in serials:
                fit3 = ht._triangulate_refined({sn: uv for sn, uv in pixels.items() if sn != dropped}, cameras)
                loo_delta[dropped] = float(np.linalg.norm(fit3.xyz_mm - fit.xyz_mm))
                heldout[dropped] = float(np.linalg.norm(ht._project_raw(cameras[dropped], fit3.xyz_mm) - pixels[dropped]))
        except (ValueError, np.linalg.LinAlgError):
            continue
        if max(loo_delta.values()) >= ht.MARKER_MAX_LOO_MM or max(heldout.values()) > ht.MARKER_MAX_HELDOUT_PX:
            continue
        score_sum = sum(candidates[serial][ci].score for serial, ci in zip(serials, choice))
        marker = ht.MarkerFit(point=fit, pixels=pixels, loo_delta_mm=loo_delta, loo_heldout_px=heldout,
                              expected_distance_mm=expected_distance)
        rank = (expected_distance, max(loo_delta.values()), fit.rms_px, -score_sum)
        if best is None or rank < best[0]:
            best = (rank, marker)
    return None if best is None else best[1]


ht._solve_marker = _solve_marker_fast   # racket_ht_black_marker._attempt_solve looks the name up at call time


def _component_candidates_fast(image, anchor_uv, *, radius, thresholds, anchor_limit_px, result_limit,
                               prefer_near=False, dedupe_px=5.0):
    """Same filters/centroid/score as racket_ht_black_marker._component_candidates, but every per-component
    statistic is computed on the component's bounding box instead of a full-ROI mask (O(bbox) vs O(ROI) per
    component; the broad pass has a 500x500 ROI and hundreds of dark components per threshold)."""
    height, width = image.shape[:2]
    anchor_x, anchor_y = anchor_uv
    if not np.all(np.isfinite([anchor_x, anchor_y])):
        return []
    x0 = max(0, int(math.floor(anchor_x - radius)))
    y0 = max(0, int(math.floor(anchor_y - radius)))
    x1 = min(width, int(math.ceil(anchor_x + radius)))
    y1 = min(height, int(math.ceil(anchor_y + radius)))
    roi = image[y0:y1, x0:x1]
    if roi.size == 0:
        return []
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    background = float(np.median(gray))
    dark = background - gray                      # float32, same as the original's (background - gray)
    weights_all = np.maximum(dark, 1.0)
    found = []
    for threshold in thresholds:
        count, labels, stats, _ = cv2.connectedComponentsWithStats((gray < threshold).astype(np.uint8), 8)
        if count <= 1:
            continue
        st = stats[1:]
        area = st[:, 4]; w = st[:, 2]; h = st[:, 3]
        keep = (area >= 8) & (area <= 1200) & (w >= 3) & (w <= 50) & (h >= 3) & (h <= 50)
        for label in np.nonzero(keep)[0] + 1:
            x, y, cw, ch, carea = (int(v) for v in stats[label])
            aspect = max(cw, ch) / max(1.0, min(cw, ch))
            fill = carea / float(cw * ch)
            if aspect > 2.4 or fill < 0.30:
                continue
            mask = labels[y:y + ch, x:x + cw] == label
            wsub = np.where(mask, weights_all[y:y + ch, x:x + cw], 0.0)
            mass = float(wsub.sum())
            if mass <= 0.0:
                continue
            xs = np.arange(x, x + cw)[None, :]
            ys = np.arange(y, y + ch)[:, None]
            u = x0 + float((wsub * xs).sum() / mass)
            v = y0 + float((wsub * ys).sum() / mass)
            distance = math.hypot(u - anchor_x, v - anchor_y)
            if anchor_limit_px is not None and distance > anchor_limit_px:
                continue
            contrast = float(np.median(dark[y:y + ch, x:x + cw][mask]))
            scale = 25.0 if anchor_limit_px is not None else 140.0
            found.append(ht.MarkerCandidate(
                uv=(u, v),
                score=max(contrast, 1.0) * fill * min(carea, 250) / (1.0 + (distance / scale) ** 2),
                area=carea, bbox_xywh=(x0 + x, y0 + y, cw, ch)))
    found.sort(key=((lambda item: (math.hypot(item.uv[0] - anchor_x, item.uv[1] - anchor_y), -item.score))
                    if prefer_near else (lambda item: -item.score)))
    deduped = []
    for item in found:
        if all(np.linalg.norm(np.subtract(item.uv, old.uv)) > dedupe_px for old in deduped):
            deduped.append(item)
        if len(deduped) == result_limit:
            break
    return deduped


ht._component_candidates = _component_candidates_fast   # _marker_candidates looks the name up at call time

V04_YAML = Path("D:/tennis-man/arm_controller/cpp/arm_controller_cpp/config/cars/v04.yaml")

J1_SPEED_THRESHOLD = 0.5      # rad/s on the J1 command: inside a swing
WINDOW_MERGE_GAP_S = 0.30
WINDOW_PRE_S = 0.35           # park + swing start before the J1 motion threshold
WINDOW_POST_S = 0.25
WINDOW_MIN_MOTION_S = 0.10
CAR_SPREAD_WARN_M = 0.02
CAR_YAW_SPREAD_WARN_DEG = 1.0
# motion-energy PC/RK alignment (fallback when the tracker recorded no /bot_state clock bridge)
MOTION_SCALE = 8                # grid frame downscale for |ΔI| energy
ALIGN_GRID_S = 0.005            # resampling grid for the cross-correlation
ALIGN_MIN_OVERLAP_S = 60.0


def session_paths(session: str) -> dict[str, Path]:
    base = TRACKER_OUTPUT / session / session
    return {
        "tracker": Path(f"{base}.json"),
        "video": Path(f"{base}.mp4"),
        "arm": Path(f"{base}_arm.json"),
        "output": Path(f"{base}_swing_marker.json"),
    }


def swing_windows(arm: dict, v_thresh: float = J1_SPEED_THRESHOLD, pre_s: float = WINDOW_PRE_S,
                  post_s: float = WINDOW_POST_S) -> list[dict]:
    """Segments of J1 command motion (|v1| > v_thresh), merged and padded. Works for swings, reparks
    (yaw wiggles) and excitation alike — anything that moves J1."""
    cmds = [r for r in arm.get("commands", []) if ht._finite(r.get("t")) and r.get("velocity")]
    cmds.sort(key=lambda r: float(r["t"]))
    if not cmds:
        raise ValueError("arm json has no /tennis/motor_command rows")
    t = np.asarray([float(r["t"]) for r in cmds])
    v1 = np.asarray([float(r["velocity"][0]) for r in cmds])
    moving = np.abs(v1) > v_thresh
    segments: list[list[float]] = []
    for k in range(len(t)):
        if not moving[k]:
            continue
        if segments and t[k] - segments[-1][1] <= WINDOW_MERGE_GAP_S:
            segments[-1][1] = float(t[k])
        else:
            segments.append([float(t[k]), float(t[k])])
    states = [r for r in arm.get("states", []) if ht._finite(r.get("t")) and r.get("effort")]
    st_t = np.asarray([float(r["t"]) for r in states])
    st_e1 = np.asarray([float(r["effort"][0]) for r in states])
    events = [e for e in arm.get("events", []) if e.get("topic") == "/tennis/status"]
    windows = []
    for start, end in segments:
        if end - start < WINDOW_MIN_MOTION_S:
            continue
        sel = (t >= start) & (t <= end)
        sel_s = (st_t >= start) & (st_t <= end)
        labels = [
            e["text"] for e in events
            if start - 0.8 <= float(e["t"]) <= end + 0.2
            and ("accepted hit" in e["text"] or "rl_swing" in e["text"] or "sweep" in e["text"])
        ]
        windows.append({
            "index": len(windows),
            "motion_start_rk": start,
            "motion_end_rk": end,
            "start_rk": start - pre_s,
            "end_rk": end + post_s,
            "peak_abs_v1_rad_s": float(np.max(np.abs(v1[sel]))) if sel.any() else None,
            "peak_abs_tau1_nm": float(np.max(np.abs(st_e1[sel_s]))) if sel_s.any() else None,
            "status": labels[:6],
        })
    if not windows:
        raise ValueError("no J1 swing windows found in the arm json")
    return windows


def manual_windows(text: str, arm: dict) -> list[dict]:
    """--windows \"t0,t1;t0,t1\" in RK seconds (e.g. static push segments); peak values filled from the arm json."""
    states = [r for r in arm.get("states", []) if ht._finite(r.get("t")) and r.get("effort")]
    st_t = np.asarray([float(r["t"]) for r in states]); st_e1 = np.asarray([float(r["effort"][0]) for r in states])
    cmds = [r for r in arm.get("commands", []) if ht._finite(r.get("t")) and r.get("velocity")]
    c_t = np.asarray([float(r["t"]) for r in cmds]); c_v1 = np.asarray([float(r["velocity"][0]) for r in cmds])
    out = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        a, b = (float(v) for v in item.split(","))
        sel_s = (st_t >= a) & (st_t <= b); sel_c = (c_t >= a) & (c_t <= b)
        out.append({"index": len(out), "motion_start_rk": a, "motion_end_rk": b, "start_rk": a, "end_rk": b,
                    "peak_abs_v1_rad_s": float(np.max(np.abs(c_v1[sel_c]))) if sel_c.any() else None,
                    "peak_abs_tau1_nm": float(np.max(np.abs(st_e1[sel_s]))) if sel_s.any() else None,
                    "status": ["manual window"]})
    if not out:
        raise ValueError("--windows given but empty")
    return out


def motion_alignment(video_path: Path, frames: list[dict], exposure_center_offset_s: float, arm: dict,
                     scale: int = MOTION_SCALE) -> dict:
    """PC/RK alignment without bot_center: whole-frame motion energy of the grid video (per exposure centre, PC
    perf axis) cross-correlated with |J1 command speed| (RK axis). The car is static, so the arm is the only
    thing moving; the lag of the correlation peak is pc_minus_rk (parabolic sub-grid refinement, ~ms).
    Compliance does not bias this: it shifts the racket by a few cm, not the timing of the whole-arm motion."""
    t_by_idx = {int(f["video_frame_idx"]): float(f["exposure_pc"]) + exposure_center_offset_s for f in frames}
    cache = Path(str(video_path)[:-4] + "_motion_energy.npz")
    if cache.is_file():
        z = np.load(cache)
        ts, es = z["ts"], z["es"]
        print(f"[swing-marker] motion energy loaded from cache ({len(ts)} frames)", flush=True)
        return _motion_correlate(ts, es, arm)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video {video_path}")
    prev = None
    idx = 0
    ts, es = [], []
    started = time.perf_counter()
    try:
        while True:
            if not capture.grab():
                break
            if idx in t_by_idx:
                ok, image = capture.retrieve()
                if ok:
                    small = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                                       (image.shape[1] // scale, image.shape[0] // scale),
                                       interpolation=cv2.INTER_AREA).astype(np.float32)
                    if prev is not None:
                        ts.append(t_by_idx[idx]); es.append(float(np.mean(np.abs(small - prev))))
                    prev = small
            idx += 1
            if idx % 3000 == 0:
                print(f"[swing-marker] motion energy {idx} frames ({time.perf_counter() - started:.0f}s)", flush=True)
    finally:
        capture.release()
    if len(ts) < 100:
        raise ValueError("motion alignment: too few decodable video frames")
    np.savez(cache, ts=np.asarray(ts), es=np.asarray(es))
    out = _motion_correlate(np.asarray(ts), np.asarray(es), arm)
    out["elapsed_s"] = time.perf_counter() - started
    return out


def _motion_correlate(ts: np.ndarray, es: np.ndarray, arm: dict) -> dict:
    """Normalised cross-correlation of the motion-energy series (PC axis) with |J1 command speed| (RK axis).
    Only lags whose overlap covers at least half of the shorter series count (edge lags with little overlap
    produce spurious peaks once normalised by overlap length — that is how 20260907_062833 first went wrong)."""
    started = time.perf_counter()
    cmds = sorted([r for r in arm.get("commands", []) if ht._finite(r.get("t")) and r.get("velocity")], key=lambda r: float(r["t"]))
    tc = np.asarray([float(r["t"]) for r in cmds]); v1 = np.abs(np.asarray([float(r["velocity"][0]) for r in cmds]))
    g = ALIGN_GRID_S
    es = np.convolve(es, np.ones(3) / 3.0, mode="same")          # 100 Hz mains flicker beats at a 3-frame period @60 fps
    tp = np.arange(ts[0], ts[-1], g); ep = np.interp(tp, ts, es)
    tr = np.arange(tc[0], tc[-1], g); vr = np.interp(tr, tc, v1)
    ep = (ep - ep.mean()) / (ep.std() + 1e-9); vr = (vr - vr.mean()) / (vr.std() + 1e-9)
    nfft = 1 << (len(ep) + len(vr) - 1).bit_length()
    c = np.fft.irfft(np.fft.rfft(ep, nfft) * np.conj(np.fft.rfft(vr, nfft)), nfft)   # c[d] = Σ_i ep[i]·vr[i−d]
    d = np.arange(nfft); d = np.where(d >= nfft // 2, d - nfft, d)
    overlap = np.maximum(0, np.minimum(len(ep), len(vr) + d) - np.maximum(0, d))
    min_overlap = max(ALIGN_MIN_OVERLAP_S / g, 0.5 * min(len(ep), len(vr)))
    score = np.where(overlap >= min_overlap, c / np.maximum(overlap, 1), -np.inf)
    k = int(np.argmax(score))
    if not np.isfinite(score[k]):
        raise ValueError("motion alignment: video and arm bag overlap < 60 s")
    # parabolic refinement on the circular neighbours
    km, kp = (k - 1) % nfft, (k + 1) % nfft
    y0, y1, y2 = score[km], score[k], score[kp]
    frac = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) if np.isfinite(y0) and np.isfinite(y2) and (y0 - 2 * y1 + y2) < 0 else 0.0
    lag_s = (float(d[k]) + float(frac)) * g
    pc_minus_rk = float(tp[0] - tr[0] + lag_s)
    far = np.abs(d - d[k]) > 2.0 / g
    second = float(np.max(score[far])) if np.any(np.isfinite(score[far])) else float("nan")
    k2 = int(np.argmax(np.where(far, score, -np.inf)))
    out = {"pc_minus_rk": pc_minus_rk, "score": float(y1), "second_peak": second,
           "second_peak_pc_minus_rk": float(tp[0] - tr[0] + d[k2] * g),
           "prominence": float(y1 / second) if second > 0 else float("inf"), "overlap_s": float(overlap[k] * g),
           "min_overlap_s": float(min_overlap * g), "n_video": int(len(ts)), "elapsed_s": time.perf_counter() - started}
    print(f"[swing-marker] motion alignment: pc_minus_rk={pc_minus_rk:.4f} score {y1:.2f} (2nd {second:.2f} at "
          f"{out['second_peak_pc_minus_rk'] - pc_minus_rk:+.1f} s), overlap {out['overlap_s']:.0f} s", flush=True)
    return out


_W: dict = {}


def _worker_init(ctx: dict) -> None:
    _W.update(ctx)


def _worker_run(task: tuple) -> list[dict]:
    """Detect markers for one window in a worker process (same passes as measure().detect)."""
    window, pmr = task
    c = _W
    serials, cameras, frames, arm_states = c["serials"], c["cameras"], c["frames"], c["arm_states"]
    jobs = []
    for frame in frames:
        center_abs_pc = float(frame["exposure_pc"]) + c["exposure_center_offset_s"]
        t_rk = center_abs_pc - pmr
        if not (window["start_rk"] <= t_rk <= window["end_rk"]):
            continue
        state = ht._interpolate(arm_states, t_rk, "t", ("tcp",), ht.ARM_STATE_MAX_GAP_S)
        car_here = c["car"]
        if c["car_track"]:
            el = center_abs_pc - c["tracker_t0"]
            pos = ht._interpolate(c["car_pos_rows"], el, "elapsed_s", ("x", "y", "z"), ht.CAR_LOC_MAX_GAP_S)
            yaw = ht._interpolate_yaw(c["car_yaw_rows"], el, ht.CAR_LOC_MAX_GAP_S)
            car_here = None if pos is None or yaw is None else {"x": pos["x"], "y": pos["y"], "z": pos["z"], "yaw": yaw}
        expected = None
        if state is not None and car_here is not None:
            expected = 1000.0 * world_of_local(np.asarray(state["tcp"], dtype=np.float64), car_here, c["z_offset_m"])
        jobs.append({"window": window, "frame": frame, "video_index": int(frame["video_frame_idx"]), "car": car_here,
                     "center_abs_pc": center_abs_pc, "center_elapsed": center_abs_pc - c["tracker_t0"], "t_rk": t_rk,
                     "fk_expected": expected, "candidates": None, "accepted": None})
    if not jobs:
        return []
    capture = cv2.VideoCapture(str(c["video"]))
    if not capture.isOpened():
        raise RuntimeError("could not open video " + str(c["video"]))
    position = [None]

    def read_frames(wanted):
        pos = position[0]
        for job in sorted(wanted, key=lambda item: item["video_index"]):
            vi = job["video_index"]
            if pos is None or vi < pos or vi - pos > 60:
                if not capture.set(cv2.CAP_PROP_POS_FRAMES, vi):
                    raise RuntimeError("could not seek to video frame %d" % vi)
                pos = vi
            while pos < vi:
                if not capture.grab():
                    raise RuntimeError("could not decode through video frame %d" % pos)
                pos += 1
            ok, image = capture.read()
            if not ok:
                raise RuntimeError("could not read video frame %d" % vi)
            pos = vi + 1
            panels = ht._grid_panels(image, serials, cameras)
            anchor = job["fk_expected"] if job["fk_expected"] is not None else job.get("traj_expected")
            job["candidates"] = {sn: ht._marker_candidates(panels[sn], tuple(ht._project_raw(cameras[sn], anchor))) for sn in serials}
        position[0] = pos

    def accept(job, solved):
        fit, n_cam, anchor_name, dropped = solved
        obs = {"x": float(fit.point.xyz_mm[0] / 1000.0), "y": float(fit.point.xyz_mm[1] / 1000.0),
               "z": float(fit.point.xyz_mm[2] / 1000.0), "t": job["center_abs_pc"], "elapsed_s": job["center_elapsed"],
               "t_rk_bridge": job["t_rk"], "frame_idx": int(job["frame"]["idx"]), "video_frame_idx": job["video_index"],
               "window": window["index"], "n_cam": n_cam, "anchor": anchor_name,
               "reproj_err": fit.point.rms_px, "reproj_max_px": fit.point.max_px,
               "loo_max_mm": max(fit.loo_delta_mm.values()), "heldout_max_px": max(fit.loo_heldout_px.values()),
               "expected_distance_mm": fit.expected_distance_mm, "black_marker": True,
               "car": {k: float((job["car"] or c["car"])[k]) for k in ("x", "y", "z", "yaw")}}
        if dropped is not None:
            obs["dropped_serial"] = dropped
        job["accepted"] = obs

    def fits():
        return sorted([(j["center_elapsed"], 1000.0 * np.asarray([j["accepted"]["x"], j["accepted"]["y"], j["accepted"]["z"]]))
                       for j in jobs if j["accepted"] is not None], key=lambda i: i[0])
    try:
        anchored = [j for j in jobs if j["fk_expected"] is not None]
        read_frames(anchored)
        for job in anchored:
            solved = ht._attempt_solve(job["candidates"], cameras, serials, job["fk_expected"], None)
            if solved is not None:
                accept(job, solved)
        for _ in range(6):
            progress = False
            for job in jobs:
                if job["accepted"] is not None:
                    continue
                traj = ht._traj_predict(fits(), job["center_elapsed"])
                if traj is None:
                    continue
                tried = job.get("traj_tried_mm")
                if tried is not None and float(np.linalg.norm(traj - tried)) < 2.0:
                    continue
                job["traj_expected"] = traj
                job["traj_tried_mm"] = traj
                if job["candidates"] is None:
                    read_frames([job])
                solved = ht._attempt_solve(job["candidates"], cameras, serials, None, traj)
                if solved is not None:
                    accept(job, solved)
                    progress = True
            if not progress:
                break
    finally:
        capture.release()
    return [{"window": window["index"], "n_frames": len(jobs), "attempted": sum(1 for j in jobs if j["candidates"] is not None),
             "accepted": [j["accepted"] for j in jobs if j["accepted"] is not None],
             "t_rk_accepted": [j["t_rk"] for j in jobs if j["accepted"] is not None]}]


def bag_alignment(arm: dict) -> dict:
    """PC/RK alignment from the PC-side rosbag: RK stamps vs bag receive time (epoch, PC clock) give rk−epoch;
    epoch−perf is measured on this PC now. Valid only in the same boot with no sleep since the session (perf_counter
    stops during sleep); biased by the one-way RK→PC network latency (a few ms, Wi-Fi) — the analysis' δ fit takes it."""
    cs = arm.get("clock_sync") or {}
    c_js = cs.get("joint_states_stamp_minus_recv_median_s")
    if not ht._finite(c_js):
        raise ValueError("arm json clock_sync lacks joint_states_stamp_minus_recv_median_s")
    k = time.time() - time.perf_counter()
    out = {"pc_minus_rk": float(-float(c_js) - k), "rk_minus_epoch_s": float(c_js), "epoch_minus_perf_now_s": k,
           "drift": cs.get("joint_states_stamp_vs_pc_drift"),
           "caveat": "same PC boot without sleep since the session; +RK→PC latency bias (ms)"}
    print(f"[swing-marker] bag alignment: pc_minus_rk={out['pc_minus_rk']:.4f} (rk−epoch {c_js:.3f}, epoch−perf now {k:.3f})", flush=True)
    return out


def median_car_pose(tracker: dict) -> dict:
    rows = [r for r in tracker.get("car_locs", [])
            if all(ht._finite(r.get(k)) for k in ("x", "y", "z")) and r.get("yaw_valid") and ht._finite(r.get("yaw"))]
    if len(rows) < 20:
        raise ValueError("fewer than 20 valid AprilTag car localisations (yaw_valid)")
    xs = np.asarray([float(r["x"]) for r in rows]); ys = np.asarray([float(r["y"]) for r in rows])
    zs = np.asarray([float(r["z"]) for r in rows]); yaws = np.unwrap(np.asarray([float(r["yaw"]) for r in rows]))
    pose = {"x": float(np.median(xs)), "y": float(np.median(ys)), "z": float(np.median(zs)),
            "yaw": float(np.median(yaws)), "n": len(rows),
            "spread_p95_m": float(np.percentile(np.hypot(xs - np.median(xs), ys - np.median(ys)), 95)),
            "yaw_spread_p95_deg": float(math.degrees(np.percentile(np.abs(yaws - np.median(yaws)), 95)))}
    if pose["spread_p95_m"] > CAR_SPREAD_WARN_M or pose["yaw_spread_p95_deg"] > CAR_YAW_SPREAD_WARN_DEG:
        print(f"[swing-marker] WARNING car pose spread p95 {pose['spread_p95_m']*1000:.0f} mm / "
              f"{pose['yaw_spread_p95_deg']:.2f} deg — was the car really stationary?", flush=True)
    return pose


def world_of_local(local_m: np.ndarray, car: dict, z_offset_m: float) -> np.ndarray:
    """Report convention (racket_ht_black_marker._expected_world_mm): world = car + R(yaw)·[x, y, z − zOff]."""
    c, s = math.cos(car["yaw"]), math.sin(car["yaw"])
    return np.asarray([car["x"] + c * local_m[0] - s * local_m[1],
                       car["y"] + s * local_m[0] + c * local_m[1],
                       car["z"] + local_m[2] - z_offset_m], dtype=np.float64)


def measure(session: str, paths: dict[str, Path], pc_minus_rk: float | None, car_track: bool = False,
            max_windows: int | None = None, align: str = "auto", windows_text: str | None = None,
            v_thresh: float = J1_SPEED_THRESHOLD, check_align: bool = False, refine: bool = False,
            workers: int = 1) -> dict:
    started = time.perf_counter()
    tracker = ht._load_json(paths["tracker"])
    arm = ht._load_json(paths["arm"])
    config = tracker.get("config") or {}
    if str(arm.get("car", "")).lower() != "v04":
        raise ValueError("arm-only black-marker measurement is supported only for V04")
    video_output = config.get("video_output") or {}
    if (video_output.get("layout") != "grid" or int(video_output.get("grid_cols", 0)) != 2
            or int(video_output.get("grid_rows", 0)) != 2 or config.get("video_frame_mapping_exact") is not True):
        raise ValueError("requires exact 2x2 grid video")
    serials = list(video_output.get("serial_order") or config.get("serials") or [])
    if len(serials) != 4 or len(set(serials)) != 4:
        raise ValueError("requires four unique cameras")
    settings = config.get("camera_settings") or {}
    exposures = [float(settings[serial]["exposure_us"]) for serial in serials]
    if max(exposures) - min(exposures) > 1e-6:
        raise ValueError("four cameras must use the same exposure time")
    exposure_center_offset_s = 0.5e-6 * exposures[0]

    tracker_t0 = float(config["first_frame_exposure_pc"])
    frames = sorted([row for row in tracker.get("frames", [])
                     if isinstance(row, dict) and isinstance(row.get("video_frame_idx"), int)
                     and ht._finite(row.get("exposure_pc"))], key=lambda row: int(row["video_frame_idx"]))
    if not frames:
        raise ValueError("session lacks video frames")
    bridge = config.get("rk_clock_bridge") or {}
    alignment: dict = {"bridge": bridge or None}
    if pc_minus_rk is not None:
        alignment["source"] = "manual"
    elif align in ("auto", "bridge") and ht._finite(bridge.get("pc_minus_rk")):
        pc_minus_rk = float(bridge["pc_minus_rk"])
        alignment["source"] = "bridge"
    elif align in ("auto", "bag"):
        bag = bag_alignment(arm)
        pc_minus_rk = bag["pc_minus_rk"]
        alignment.update(source="bag", bag=bag)
    elif align == "motion":
        print("[swing-marker] motion-energy alignment (decodes the whole video once)", flush=True)
        motion = motion_alignment(paths["video"], frames, exposure_center_offset_s, arm)
        pc_minus_rk = motion["pc_minus_rk"]
        alignment.update(source="motion", motion=motion)
    else:
        raise ValueError("no PC/RK alignment: tracker config has no rk_clock_bridge (bot_center /bot_state not recorded); "
                         "use --align bag (same boot, no sleep), --align motion or --pc-minus-rk")
    if check_align and alignment["source"] != "motion":
        motion = motion_alignment(paths["video"], frames, exposure_center_offset_s, arm)
        alignment["motion"] = motion
        alignment["motion_minus_used_ms"] = (motion["pc_minus_rk"] - pc_minus_rk) * 1e3
        print(f"[swing-marker] alignment cross-check: motion − {alignment['source']} = {alignment['motion_minus_used_ms']:+.1f} ms "
              f"(score {motion['score']:.2f}, prominence {motion['prominence']:.2f})", flush=True)
    z_offset_m = float(yaml.safe_load(V04_YAML.read_text(encoding="utf-8"))["tuning"]["hit_pos_z_offset_m"])

    car = median_car_pose(tracker)
    # --car-track: per-frame AprilTag pose (HT-script convention) instead of the session median —
    # for smoke tests on sessions where the car moved, or if the car drifted.
    car_rows = sorted([r for r in tracker.get("car_locs", []) if ht._finite(r.get("elapsed_s"))], key=lambda r: float(r["elapsed_s"]))
    car_pos_rows = [r for r in car_rows if all(ht._finite(r.get(k)) for k in ("x", "y", "z"))]
    car_yaw_rows = [r for r in car_rows if ht._finite(r.get("yaw"))]
    windows = manual_windows(windows_text, arm) if windows_text else swing_windows(arm, v_thresh=v_thresh)
    if max_windows is not None:
        windows = windows[:max_windows]
    compensated = bool(arm.get("link_commands"))
    calibration_path = ht._session_path(config["calib_config_path"], paths["tracker"])
    cameras = ht._load_cameras(calibration_path, serials)
    arm_states = sorted([row for row in arm.get("states", []) if ht._finite(row.get("t")) and row.get("tcp")],
                        key=lambda row: float(row["t"]))
    if not arm_states:
        raise ValueError("session lacks arm states")

    def detect(windows_sel: list[dict], pmr: float) -> list[dict]:
        """FK-anchored marker detection over the given windows with PC-RK offset pmr (all passes)."""
        jobs: list[dict] = []
        for window in windows_sel:
            for frame in frames:
                center_abs_pc = float(frame["exposure_pc"]) + exposure_center_offset_s
                t_rk = center_abs_pc - pmr
                if not (window["start_rk"] <= t_rk <= window["end_rk"]):
                    continue
                state = ht._interpolate(arm_states, t_rk, "t", ("tcp",), ht.ARM_STATE_MAX_GAP_S)
                expected = None
                car_here = car
                if car_track:
                    el = center_abs_pc - tracker_t0
                    pos = ht._interpolate(car_pos_rows, el, "elapsed_s", ("x", "y", "z"), ht.CAR_LOC_MAX_GAP_S)
                    yaw = ht._interpolate_yaw(car_yaw_rows, el, ht.CAR_LOC_MAX_GAP_S)
                    car_here = None if pos is None or yaw is None else {"x": pos["x"], "y": pos["y"], "z": pos["z"], "yaw": yaw}
                if state is not None and car_here is not None:
                    expected = 1000.0 * world_of_local(np.asarray(state["tcp"], dtype=np.float64), car_here, z_offset_m)
                jobs.append({"window": window, "frame": frame, "video_index": int(frame["video_frame_idx"]), "car": car_here,
                             "center_abs_pc": center_abs_pc, "center_elapsed": center_abs_pc - tracker_t0,
                             "t_rk": t_rk, "fk_expected": expected, "candidates": None, "accepted": None})
        print(f"[swing-marker] {session}: {len(windows_sel)} windows, {len(jobs)} frames, "
              f"compensated={compensated}, pmr={pmr:.4f} via {alignment['source']}"
              f"{' (bridge n=%s, mad=%s)' % (bridge.get('n'), bridge.get('mad')) if alignment['source'] == 'bridge' else ''}",
              flush=True)

        video_position: list[int | None] = [None]
        decoded = [0]

        def read_frames(capture: cv2.VideoCapture, wanted: list[dict]) -> None:
            position = video_position[0]
            for job in sorted(wanted, key=lambda item: item["video_index"]):
                decoded[0] += 1
                if decoded[0] % 60 == 0:
                    print(f"[swing-marker] decoded {decoded[0]} frames", flush=True)
                video_index = job["video_index"]
                if position is None or video_index < position or video_index - position > 60:
                    if not capture.set(cv2.CAP_PROP_POS_FRAMES, video_index):
                        raise RuntimeError(f"could not seek to video frame {video_index}")
                    if abs(capture.get(cv2.CAP_PROP_POS_FRAMES) - video_index) > 0.5:
                        raise RuntimeError(f"video seek missed frame {video_index}")
                    position = video_index
                while position < video_index:
                    if not capture.grab():
                        raise RuntimeError(f"could not decode through video frame {position}")
                    position += 1
                ok, image = capture.read()
                if not ok:
                    raise RuntimeError(f"could not read video frame {video_index}")
                position = video_index + 1
                panels = ht._grid_panels(image, serials, cameras)
                anchor = job["fk_expected"] if job["fk_expected"] is not None else job.get("traj_expected")
                job["candidates"] = {serial: ht._marker_candidates(panels[serial], tuple(ht._project_raw(cameras[serial], anchor)))
                                     for serial in serials}
            video_position[0] = position

        def accept(job: dict, solved) -> None:
            fit, n_cam, anchor_name, dropped = solved
            obs = {"x": float(fit.point.xyz_mm[0] / 1000.0), "y": float(fit.point.xyz_mm[1] / 1000.0),
                   "z": float(fit.point.xyz_mm[2] / 1000.0), "t": job["center_abs_pc"], "elapsed_s": job["center_elapsed"],
                   "t_rk_bridge": job["t_rk"], "frame_idx": int(job["frame"]["idx"]), "video_frame_idx": job["video_index"],
                   "window": job["window"]["index"], "n_cam": n_cam, "anchor": anchor_name,
                   "reproj_err": fit.point.rms_px, "reproj_max_px": fit.point.max_px,
                   "loo_max_mm": max(fit.loo_delta_mm.values()), "heldout_max_px": max(fit.loo_heldout_px.values()),
                   "expected_distance_mm": fit.expected_distance_mm, "black_marker": True,
                   "car": {k: float((job["car"] or car)[k]) for k in ("x", "y", "z", "yaw")}}   # traj-anchored frames may lack a per-frame pose
            if dropped is not None:
                obs["dropped_serial"] = dropped
            job["accepted"] = obs

        def window_fits(index: int) -> list[tuple[float, np.ndarray]]:
            return sorted([(job["center_elapsed"], 1000.0 * np.asarray([job["accepted"]["x"], job["accepted"]["y"], job["accepted"]["z"]]))
                           for job in jobs if job["accepted"] is not None and job["window"]["index"] == index], key=lambda i: i[0])

        capture = cv2.VideoCapture(str(paths["video"]))
        if not capture.isOpened():
            raise RuntimeError(f"could not open video {paths['video']}")
        try:
            anchored = [job for job in jobs if job["fk_expected"] is not None]
            read_frames(capture, anchored)
            for job in anchored:
                solved = ht._attempt_solve(job["candidates"], cameras, serials, job["fk_expected"], None)
                if solved is not None:
                    accept(job, solved)
            for _ in range(6):
                progress = False
                for job in jobs:
                    if job["accepted"] is not None:
                        continue
                    traj = ht._traj_predict(window_fits(job["window"]["index"]), job["center_elapsed"])
                    if traj is None:
                        continue
                    tried = job.get("traj_tried_mm")
                    if tried is not None and float(np.linalg.norm(traj - tried)) < 2.0:
                        continue
                    job["traj_expected"] = traj
                    job["traj_tried_mm"] = traj
                    if job["candidates"] is None:
                        read_frames(capture, [job])
                    solved = ht._attempt_solve(job["candidates"], cameras, serials, None, traj)
                    if solved is not None:
                        accept(job, solved)
                        progress = True
                if not progress:
                    break
        finally:
            capture.release()

        return jobs

    def refine_offset(jobs_probe: list[dict]) -> tuple[float, dict | None]:
        """PC/RK offset refinement on VERTICAL motion only: scan delta over +-150 ms minimising the spread of the
        z gap (marker z - FK z) over frames where the FK moves vertically (J2-J4 height adjusts, >0.15 m/s).
        The J1 spring lag lives along the J1 tangent (horizontal), so it cannot bias this; a horizontal criterion
        on slow windows was biased by ~+35 ms on 20260907_062833 while the z criterion landed 5 ms from the bag
        estimate (= the RK->PC one-way latency, the expected sign)."""
        accepted = [job for job in jobs_probe if job["accepted"] is not None]
        if len(accepted) < 20:
            return 0.0, None
        st_t = np.asarray([float(r["t"]) for r in arm_states])
        st_z = np.asarray([float(r["tcp"][2]) for r in arm_states])
        grid = np.arange(-0.15, 0.1501, 0.005)
        spread = []
        n_mov = []
        for dlt in grid:
            trk = np.asarray([job["t_rk"] + dlt for job in accepted])
            fz = np.interp(trk, st_t, st_z)
            vz = (np.interp(trk + 0.01, st_t, st_z) - np.interp(trk - 0.01, st_t, st_z)) / 0.02
            gz = np.asarray([job["accepted"]["z"] - (job["car"] or car)["z"] for job in accepted]) - (fz - z_offset_m)
            moving = np.abs(vz) > 0.15
            n_mov.append(int(moving.sum()))
            spread.append(float(np.median(np.abs(gz[moving] - np.median(gz)))) if moving.sum() >= 20 else np.inf)
        spread = np.asarray(spread)
        k = int(np.argmin(spread))
        if not np.isfinite(spread[k]):
            return 0.0, None
        best = float(grid[k])
        if 0 < k < len(grid) - 1 and np.isfinite(spread[k - 1]) and np.isfinite(spread[k + 1]):
            denom = spread[k - 1] - 2 * spread[k] + spread[k + 1]
            if denom > 0:
                best += float(0.5 * (spread[k - 1] - spread[k + 1]) / denom) * 0.005
        mid = len(grid) // 2
        return best, {"delta_s": best, "spread_mm_at_best": float(spread[k] * 1e3), "spread_mm_at_zero": float(spread[mid] * 1e3),
                      "n": len(accepted), "n_vertical_moving": int(n_mov[k]), "criterion": "z-gap spread on |vz|>0.15 m/s frames"}

    if alignment["source"] == "motion" or refine:
        # probe = every 4th window (>= 12): the height adjusts before/after each swing give the vertical motion
        step = max(1, len(windows) // 12)
        slow = windows[::step]
        print(f"[swing-marker] offset refinement (vertical-motion criterion) on {len(slow)} probe windows", flush=True)
        probe = detect(slow, pc_minus_rk)
        delta_best, info = refine_offset(probe)
        if info is not None:
            pc_minus_rk -= delta_best          # true t_rk = t_pc - pc_minus_rk + delta  =>  new = old - delta
            alignment["refine"] = info
            print(f"[swing-marker] refined pc_minus_rk by {-delta_best*1e3:+.1f} ms -> {pc_minus_rk:.4f} "
                  f"(z-gap spread {info['spread_mm_at_zero']:.1f} -> {info['spread_mm_at_best']:.1f} mm on {info['n_vertical_moving']} frames)", flush=True)
        else:
            print("[swing-marker] offset refinement skipped: too few marker frames in the probe windows", flush=True)
    if workers > 1:
        import multiprocessing as mp
        ctx = {"serials": serials, "cameras": cameras, "frames": frames, "arm_states": arm_states, "car": car,
               "car_track": car_track, "car_pos_rows": car_pos_rows, "car_yaw_rows": car_yaw_rows,
               "exposure_center_offset_s": exposure_center_offset_s, "tracker_t0": tracker_t0, "z_offset_m": z_offset_m,
               "video": str(paths["video"])}
        print("[swing-marker] %d windows on %d worker processes" % (len(windows), workers), flush=True)
        results = []
        with mp.get_context("spawn").Pool(workers, initializer=_worker_init, initargs=(ctx,)) as pool:
            for k, res in enumerate(pool.imap_unordered(_worker_run, [(w, pc_minus_rk) for w in windows]), 1):
                results.extend(res)
                print("[swing-marker] window done %d/%d" % (k, len(windows)), flush=True)
        jobs = []
        for r in results:
            w = windows[r["window"]]
            for obs, t_rk in zip(r["accepted"], r["t_rk_accepted"]):
                jobs.append({"window": w, "accepted": obs, "candidates": True, "t_rk": t_rk})
            for _ in range(r["n_frames"] - len(r["accepted"])):
                jobs.append({"window": w, "accepted": None, "candidates": True if r["attempted"] else None, "t_rk": None})
    else:
        jobs = detect(windows, pc_minus_rk)

    observations = sorted([job["accepted"] for job in jobs if job["accepted"] is not None], key=lambda r: r["t"])
    attempted = sum(1 for job in jobs if job["candidates"] is not None)
    for window in windows:
        n_ok = sum(1 for job in jobs if job["accepted"] is not None and job["window"] is window)
        n_all = sum(1 for job in jobs if job["window"] is window)
        print(f"[swing-marker] window {window['index']}: {n_ok}/{n_all} frames, peak |v1| "
              f"{window['peak_abs_v1_rad_s'] or 0:.1f} rad/s, peak |tau1| {window['peak_abs_tau1_nm'] or 0:.0f} Nm, "
              f"{'; '.join(window['status'][:2])}", flush=True)
    print(f"[swing-marker] {len(observations)}/{attempted} frames accepted in {time.perf_counter() - started:.1f}s")
    return {
        "session": session,
        "config": {"measurement": "V04 fixed black marker center over arm-only swing windows",
                   "timing": f"exposure center = exposure_pc + {exposure_center_offset_s * 1000:.3f} ms; "
                             f"t_rk_bridge = t_pc − pc_minus_rk (tracker rk_clock_bridge)",
                   "anchor": "FK(measured joints, extract_arm_bag) → world via median AprilTag car pose; "
                             f"z − hit_pos_z_offset_m ({z_offset_m:+.6f})",
                   "gates": "same as racket_ht_black_marker (reproj/LOO/held-out/expected distance)"},
        "clock": {"pc_minus_rk": pc_minus_rk, "alignment": alignment},
        "car_pose": car,
        "car_track": car_track,
        "z_offset_m": z_offset_m,
        "compensated": compensated,
        "windows": windows,
        "v_thresh_rad_s": v_thresh,
        "summary": {"observations": len(observations), "frames_processed": attempted,
                    "windows_with_observations": len({r["window"] for r in observations}),
                    "obs_traj_anchor": sum(1 for r in observations if r.get("anchor") == "traj"),
                    "obs_3cam": sum(1 for r in observations if r.get("n_cam") == 3)},
        "racket_observations": observations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", required=True, help="tracker session id, e.g. tracker_20260907_101010")
    parser.add_argument("--pc-minus-rk", type=float, default=None,
                        help="override PC perf − RK mono (s); default = tracker config.rk_clock_bridge")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--car-track", action="store_true", help="per-frame AprilTag car pose instead of the session median")
    parser.add_argument("--max-windows", type=int, default=None, help="only the first N swing windows (smoke test)")
    parser.add_argument("--align", default="auto", choices=["auto", "bridge", "bag", "motion"],
                        help="PC/RK alignment: tracker clock bridge (/bot_state) → PC rosbag receive clock (same boot, no sleep) → "
                             "whole-frame motion energy vs J1 speed (fails under 100 Hz light flicker)")
    parser.add_argument("--check-align", action="store_true", help="also run the motion alignment and report the difference")
    parser.add_argument("--refine", action="store_true", help="marker-based +-150 ms offset refinement even with a clock bridge")
    parser.add_argument("--workers", type=int, default=1, help="worker processes for the main detection pass (one window per task)")
    parser.add_argument("--windows", default=None, help='manual RK windows "t0,t1;t0,t1" instead of J1-speed detection')
    parser.add_argument("--v-thresh", type=float, default=J1_SPEED_THRESHOLD, help="J1 command speed threshold for windows (rad/s)")
    args = parser.parse_args()
    paths = session_paths(args.session)
    for key in ("tracker", "video", "arm"):
        if not paths[key].is_file():
            raise SystemExit(f"missing {key}: {paths[key]}")
    result = measure(args.session, paths, args.pc_minus_rk, car_track=args.car_track, max_windows=args.max_windows,
                     align=args.align, windows_text=args.windows, v_thresh=args.v_thresh, check_align=args.check_align,
                     refine=args.refine, workers=args.workers)
    out = args.output or paths["output"]
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[swing-marker] saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
