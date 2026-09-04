#!/usr/bin/env python3
"""Fit V04 joint zeros and the sweet-spot distance from static FK-vs-vision captures.

Model, per pose, in the arm model frame, x/z components only:

    p_vision_xz = fk(q + dq)["tcp"]_xz + ta * handle_axis_xz + [base_dx, base_dz]

dq = (0, dq2, dq3, dq4, 0, 0); ta corrects TCP_DISTANCE along the handle.  The black
marker sits ON the handle axis (user-confirmed 2026-09-05), so there is no perpendicular
tool offset - that is exactly what makes dq4 identifiable from positions alone.

Why x/z only: the y residual is a lateral wrist-chain effect this model cannot express;
letting it in aliases into absurd dq4/tool values.  y is outside the hit contract anyway.
Not fitted: dq1 (exactly degenerate with the car yaw), dq5 (acts on y), dq6 (does not move
a point on the handle axis).

Prints the numbers to write into config/cars/v04.yaml: motors[].offset_rad
(offset_new = offset_old - direction * dq), kinematics.tool_x (= J6 origin x + TCP_DISTANCE),
tuning.hit_pos_z_offset_m (= old - base_dz: the base sits base_dz higher than modelled).
Captures record q in the offset convention active at capture time, so fit captures taken
with the same yaml as the kinematics being loaded.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.optimize import least_squares

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ArmCalibration import capture_v04_ht_replay as replay

NAMES = ("dq2", "dq3", "dq4", "ta", "base_dx", "base_dz")
SOURCE_KEYS = {"marker": "black_marker", "racket": "racket"}


def load_poses(session_dirs: list[Path], source: str, z_offset: float) -> list[dict[str, Any]]:
    key = SOURCE_KEYS[source]
    poses: list[dict[str, Any]] = []
    for session in session_dirs:
        results = session / "fk_visual_analysis" / "results.jsonl"
        if not results.is_file():
            raise SystemExit(f"{session}: run analyze_v04_ht_replay.py first ({results} missing)")
        for line in results.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            measured = row.get(key)
            if not measured or (key == "racket" and row.get("quality") != "pass"):
                continue
            point = np.asarray(measured["car_m"], dtype=np.float64).copy()
            point[2] += z_offset  # car ground frame -> arm model frame
            poses.append(
                {
                    "session": session.name,
                    "trial_id": row["trial_id"],
                    "point": row.get("point"),
                    "q": np.asarray(row["q_rad"], dtype=np.float64),
                    "measured_model_m": point,
                }
            )
    if not poses:
        raise SystemExit(f"no usable {source} observations in {[str(s) for s in session_dirs]}")
    return poses


def fit(kin: Any, poses: list[dict[str, Any]]) -> dict[str, Any]:
    def predict(theta: np.ndarray, pose: dict[str, Any]) -> np.ndarray:
        fk = kin.fk(pose["q"] + np.array([0.0, theta[0], theta[1], theta[2], 0.0, 0.0]))
        return fk["tcp"] + theta[3] * fk["handle_axis"] + np.array([theta[4], 0.0, theta[5]])

    def residual(theta: np.ndarray) -> np.ndarray:
        out = []
        for pose in poses:
            d = predict(theta, pose) - pose["measured_model_m"]
            out += [d[0], d[2]]
        return np.asarray(out)

    solution = least_squares(residual, np.zeros(6), method="lm", xtol=1e-14, ftol=1e-14)
    dof = max(1, solution.fun.size - solution.x.size)
    sigma2 = float((solution.fun ** 2).sum()) / dof
    sd = np.sqrt(np.diag(np.linalg.inv(solution.jac.T @ solution.jac) * sigma2))
    per_pose = solution.fun.reshape(-1, 2)
    return {
        "theta": solution.x,
        "sd": sd,
        "rms_mm": 1000.0 * float(np.sqrt((solution.fun ** 2).mean())),
        "per_pose_mm": 1000.0 * per_pose,
        "cond": float(np.linalg.cond(solution.jac)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session_dirs", nargs="+", help="v04_ht_replay_* capture directories")
    parser.add_argument("--source", choices=("marker", "racket"), default="marker")
    parser.add_argument("--out", default="", help="write the fit report to this JSON path")
    args = parser.parse_args(argv)

    sessions = [
        p if (p := Path(name)).is_absolute() else PROJECT_ROOT / "arm_controller_data" / p
        for name in args.session_dirs
    ]
    kin, config = replay.load_kinematics()
    z_offset = float(config["tuning"]["hit_pos_z_offset_m"])
    poses = load_poses(sessions, args.source, z_offset)
    result = fit(kin, poses)
    theta, sd = result["theta"], result["sd"]

    print(f"{len(poses)} poses from {len(sessions)} capture(s), source={args.source}, "
          f"kinematics={replay.V04_SOURCE_ROOT} (TCP_DISTANCE {kin.TCP_DISTANCE:.6f}, z_offset {z_offset:+.6f})")
    print(f"  dq2={math.degrees(theta[0]):+.3f}±{math.degrees(sd[0]):.3f}deg  "
          f"dq3={math.degrees(theta[1]):+.3f}±{math.degrees(sd[1]):.3f}deg  "
          f"dq4={math.degrees(theta[2]):+.3f}±{math.degrees(sd[2]):.3f}deg")
    print(f"  ta={1000 * theta[3]:+.2f}±{1000 * sd[3]:.2f}mm (along handle)  "
          f"base_dx={1000 * theta[4]:+.2f}±{1000 * sd[4]:.2f}mm  base_dz={1000 * theta[5]:+.2f}±{1000 * sd[5]:.2f}mm")
    print(f"  rms_xz={result['rms_mm']:.2f}mm  worst={np.linalg.norm(result['per_pose_mm'], axis=1).max():.2f}mm  "
          f"cond(J)={result['cond']:.0f}")
    if len(poses) < 15 or result["cond"] > 500:
        print(f"  WARNING: {len(poses)} poses / cond {result['cond']:.0f} - too few or too similar to pin 6 parameters;"
              " read the residuals, not the parameters (use >=30 D-optimal sweep poses to calibrate)")
    print("  per pose (res_x, res_z mm):")
    for pose, res in zip(poses, result["per_pose_mm"]):
        loft = math.degrees(-pose["q"][1] - pose["q"][2] + pose["q"][3])
        print(f"    {pose['trial_id']} {str(pose['point']):>8} loft={loft:+6.1f}  {res[0]:+6.1f} {res[1]:+6.1f}")

    motors = {m["joint_name"]: m for m in config["motors"]}
    j6_x = float(kin.JOINTS[-1]["local0"][0, 3])
    proposal = {}
    for joint, index in (("joint2", 0), ("joint3", 1), ("joint4", 2)):
        direction = float(motors[joint]["direction"])
        old = float(motors[joint]["offset_rad"])
        proposal[f"{joint}.offset_rad"] = old - direction * float(theta[index])
    proposal["TCP_DISTANCE"] = float(kin.TCP_DISTANCE) + float(theta[3])
    proposal["kinematics.tool_x"] = j6_x + proposal["TCP_DISTANCE"]
    proposal["tuning.hit_pos_z_offset_m"] = z_offset - float(theta[5])
    print("  -> v04.yaml if applied: " + "  ".join(f"{k}={v:.9f}" for k, v in proposal.items()))

    if args.out:
        Path(args.out).write_text(json.dumps({
            "schema": "v04_joint_zero_fit/v2_planar",
            "sessions": [str(s) for s in sessions], "source": args.source, "poses": len(poses),
            "kinematics_source": str(replay.V04_KINEMATICS_PATH),
            "theta": dict(zip(NAMES, theta.tolist())), "sd": dict(zip(NAMES, sd.tolist())),
            "rms_mm": result["rms_mm"], "cond": result["cond"], "proposal": proposal,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"WROTE {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
