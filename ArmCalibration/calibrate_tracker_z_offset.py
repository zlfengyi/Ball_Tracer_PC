# -*- coding: utf-8 -*-
"""Fit a z-only POE correction from tracker runs with vision racket observations."""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ArmCalibration.common import ARM_DATA_ROOT, auto_session_dir, load_json, rel_or_abs, save_json
from src.arm_poe import ArmPoePositionModel


DEFAULT_TRACKER_OUTPUT_DIR = PROJECT_ROOT / "tracker_output"
DEFAULT_POE_CONFIG_PATH = PROJECT_ROOT / "src" / "config" / "arm_poe_racket_center.json"
DEFAULT_RESULT_NAME = "tracker_z_correction_result.json"
DEFAULT_SAMPLES_NAME = "tracker_z_samples.json"
DEFAULT_MANIFEST_NAME = "filtered_tracker_z_manifest.json"
DEFAULT_FILTERED_RUNS_SUBDIR = "filtered_runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a tracker-driven z-only correction on top of the existing POE config.",
    )
    parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        required=True,
        help="Tracker run stem, for example tracker_20260407_175402. Repeat for multiple runs.",
    )
    parser.add_argument(
        "--tracker-output-dir",
        type=str,
        default=str(DEFAULT_TRACKER_OUTPUT_DIR),
        help="Directory containing tracker_*.json and *_pc_logger.json files.",
    )
    parser.add_argument(
        "--poe-config",
        type=str,
        default=str(DEFAULT_POE_CONFIG_PATH),
        help="Existing POE config path to read and update.",
    )
    parser.add_argument(
        "--export-config",
        type=str,
        default=str(DEFAULT_POE_CONFIG_PATH),
        help="Path to write the updated POE config.",
    )
    parser.add_argument(
        "--max-joint-delta-s",
        type=float,
        default=0.08,
        help="Maximum allowed tracker-to-joint timestamp mismatch.",
    )
    parser.add_argument(
        "--max-racket-reproj-px",
        type=float,
        default=10.0,
        help="Keep only racket observations at or below this reprojection error.",
    )
    parser.add_argument(
        "--session-dir",
        type=str,
        default="",
        help="Optional output session directory. Defaults to a new ArmCalibration/data session.",
    )
    parser.add_argument(
        "--result-name",
        type=str,
        default=DEFAULT_RESULT_NAME,
        help="Result JSON filename inside the output session directory.",
    )
    parser.add_argument(
        "--samples-name",
        type=str,
        default=DEFAULT_SAMPLES_NAME,
        help="Raw accepted samples JSON filename inside the output session directory.",
    )
    parser.add_argument(
        "--manifest-name",
        type=str,
        default=DEFAULT_MANIFEST_NAME,
        help="Manifest JSON filename that points to the clean per-run artifacts.",
    )
    return parser.parse_args()


def resolve_session_dir(raw: str) -> Path:
    if raw.strip():
        return Path(raw).resolve()
    return auto_session_dir(ARM_DATA_ROOT, "tracker_z_correction").resolve()


def build_run_paths(tracker_output_dir: Path, run_id: str) -> tuple[Path, Path]:
    tracker_path = tracker_output_dir / f"{run_id}_with_racket.json"
    pc_logger_path = tracker_output_dir / f"{run_id}_pc_logger.json"
    if not tracker_path.exists():
        raise FileNotFoundError(f"Missing tracker racket JSON: {tracker_path}")
    if not pc_logger_path.exists():
        raise FileNotFoundError(f"Missing tracker pc logger JSON: {pc_logger_path}")
    return tracker_path, pc_logger_path


def nearest_index(sorted_values: np.ndarray, target: float) -> int:
    idx = int(np.searchsorted(sorted_values, target))
    if idx <= 0:
        return 0
    if idx >= len(sorted_values):
        return len(sorted_values) - 1
    before = idx - 1
    after = idx
    if abs(sorted_values[after] - target) < abs(sorted_values[before] - target):
        return after
    return before


def compute_summary(values_mm: np.ndarray) -> dict[str, float | int]:
    if len(values_mm) == 0:
        return {
            "count": 0,
            "mean_mm": 0.0,
            "median_mm": 0.0,
            "std_mm": 0.0,
            "p05_mm": 0.0,
            "p95_mm": 0.0,
            "abs_mean_mm": 0.0,
        }
    return {
        "count": int(len(values_mm)),
        "mean_mm": float(np.mean(values_mm)),
        "median_mm": float(np.median(values_mm)),
        "std_mm": float(np.std(values_mm)),
        "p05_mm": float(np.percentile(values_mm, 5)),
        "p95_mm": float(np.percentile(values_mm, 95)),
        "abs_mean_mm": float(np.mean(np.abs(values_mm))),
    }


def extract_run_samples(
    *,
    run_id: str,
    tracker_path: Path,
    pc_logger_path: Path,
    model: ArmPoePositionModel,
    max_joint_delta_s: float,
    max_racket_reproj_px: float,
) -> tuple[list[dict], dict]:
    tracker_payload = load_json(tracker_path)
    pc_logger_payload = load_json(pc_logger_path)

    joint_layout = pc_logger_payload.get("joint_state_layout") or {}
    joint_fields = list(joint_layout.get("sample_fields") or [])
    joint_names = list(joint_layout.get("joint_names") or [])
    joint_rows = list(pc_logger_payload.get("joint_states_matrix") or [])

    stamp_idx = joint_fields.index("stamp_pc_ns")
    position_idx = joint_fields.index("position")
    joint_name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
    required_joint_indices = [joint_name_to_idx[name] for name in model.expected_joint_names]

    joint_times_s = np.array([float(row[stamp_idx]) / 1e9 for row in joint_rows], dtype=np.float64)
    joint_positions = np.array(
        [[float(row[position_idx][idx]) for idx in required_joint_indices] for row in joint_rows],
        dtype=np.float64,
    )
    if len(joint_times_s) == 0:
        raise ValueError(f"{run_id}: joint_states_matrix is empty")

    racket_rows = list(tracker_payload.get("racket_observations") or [])
    accepted_samples: list[dict] = []
    rejected_samples: list[dict] = []
    stats = {
        "run_id": run_id,
        "racket_obs_count": int(len(racket_rows)),
        "accepted_count": 0,
        "rejected_joint_time": 0,
        "rejected_racket_reproj": 0,
    }

    for obs in racket_rows:
        obs_t_s = float(obs["t"])
        base_record = {
            "run_id": run_id,
            "tracker_time_s": obs_t_s,
            "frame_idx": int(obs.get("frame_idx", -1)),
            "video_frame_idx": int(obs.get("video_frame_idx", -1)),
            "racket_reproj_px": float(obs.get("reproj_err", math.inf)),
            "face_keypoint_score_min": float(obs.get("face_keypoint_score_min", 0.0)),
        }
        joint_idx = nearest_index(joint_times_s, obs_t_s)
        joint_dt_s = abs(float(joint_times_s[joint_idx]) - obs_t_s)
        if joint_dt_s > max_joint_delta_s:
            stats["rejected_joint_time"] += 1
            rejected_samples.append(
                {
                    **base_record,
                    "reason": "joint_time",
                    "joint_dt_ms": float(joint_dt_s * 1000.0),
                }
            )
            continue

        racket_reproj = float(obs.get("reproj_err", math.inf))
        if racket_reproj > max_racket_reproj_px:
            stats["rejected_racket_reproj"] += 1
            rejected_samples.append(
                {
                    **base_record,
                    "reason": "racket_reproj",
                    "joint_dt_ms": float(joint_dt_s * 1000.0),
                }
            )
            continue

        q = joint_positions[joint_idx]
        fk = model.forward(q.tolist())
        poe_rel_world_mm = fk.point_world_mm - model.t_base_in_world_mm
        vision_rel_z_mm = float(obs["z"]) * 1000.0
        poe_rel_z_mm = float(poe_rel_world_mm[2])
        diff_mm = vision_rel_z_mm - poe_rel_z_mm

        accepted_samples.append(
            {
                "run_id": run_id,
                "tracker_time_s": obs_t_s,
                "frame_idx": int(obs.get("frame_idx", -1)),
                "video_frame_idx": int(obs.get("video_frame_idx", -1)),
                "joint_dt_ms": float(joint_dt_s * 1000.0),
                "joint_values_rad": [float(value) for value in q.tolist()],
                "vision_rel_z_mm": float(vision_rel_z_mm),
                "poe_rel_z_mm": float(poe_rel_z_mm),
                "diff_mm": float(diff_mm),
                "racket_reproj_px": float(racket_reproj),
                "face_keypoint_score_min": float(obs.get("face_keypoint_score_min", 0.0)),
            }
        )

    stats["accepted_count"] = int(len(accepted_samples))
    diffs_mm = np.array([sample["diff_mm"] for sample in accepted_samples], dtype=np.float64)
    stats["diff_summary_mm"] = compute_summary(diffs_mm)
    stats["rejected_count"] = int(len(rejected_samples))
    return accepted_samples, rejected_samples, stats


def apply_z_offset_to_config(config: dict, offset_base_mm: float, *, result_rel_path: str, source_runs: list[str], sample_count: int, filters: dict) -> dict:
    updated = dict(config)
    updated["z_axis_correction"] = {
        "enabled": True,
        "type": "tracker_constant_offset_base_z",
        "offset_base_mm": float(offset_base_mm),
        "written_at": datetime.now().isoformat(timespec="seconds"),
        "source_runs": list(source_runs),
        "sample_count": int(sample_count),
        "source_result": result_rel_path,
        "note": "z-only correction fitted from high-quality tracker vision vs POE samples; x/y remain unchanged.",
        "filters": filters,
    }
    return updated


def build_run_clean_artifacts(
    *,
    session_dir: Path,
    run_id: str,
    accepted_samples: list[dict],
    rejected_samples: list[dict],
    stats: dict,
) -> dict:
    filtered_runs_dir = session_dir / DEFAULT_FILTERED_RUNS_SUBDIR
    filtered_runs_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = filtered_runs_dir / f"{run_id}_accepted.json"
    rejected_path = filtered_runs_dir / f"{run_id}_rejected.json"
    summary_path = filtered_runs_dir / f"{run_id}_summary.json"
    save_json(accepted_path, accepted_samples)
    save_json(rejected_path, rejected_samples)
    save_json(summary_path, stats)
    return {
        "run_id": run_id,
        "accepted_count": int(len(accepted_samples)),
        "rejected_count": int(len(rejected_samples)),
        "accepted_path": rel_or_abs(accepted_path),
        "rejected_path": rel_or_abs(rejected_path),
        "summary_path": rel_or_abs(summary_path),
    }


def main() -> int:
    args = parse_args()
    tracker_output_dir = Path(args.tracker_output_dir).resolve()
    poe_config_path = Path(args.poe_config).resolve()
    export_config_path = Path(args.export_config).resolve()
    session_dir = resolve_session_dir(args.session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)

    model = ArmPoePositionModel(config_path=poe_config_path)
    all_samples: list[dict] = []
    run_summaries: list[dict] = []
    clean_run_artifacts: list[dict] = []
    for run_id in args.runs:
        tracker_path, pc_logger_path = build_run_paths(tracker_output_dir, run_id)
        samples, rejected_samples, stats = extract_run_samples(
            run_id=run_id,
            tracker_path=tracker_path,
            pc_logger_path=pc_logger_path,
            model=model,
            max_joint_delta_s=args.max_joint_delta_s,
            max_racket_reproj_px=args.max_racket_reproj_px,
        )
        all_samples.extend(samples)
        run_summaries.append(stats)
        clean_run_artifacts.append(
            build_run_clean_artifacts(
                session_dir=session_dir,
                run_id=run_id,
                accepted_samples=samples,
                rejected_samples=rejected_samples,
                stats=stats,
            )
        )

    if not all_samples:
        raise SystemExit("No accepted tracker z samples after filtering.")

    diffs_mm = np.array([sample["diff_mm"] for sample in all_samples], dtype=np.float64)
    fitted_increment_base_mm = float(np.mean(diffs_mm))
    target_z_offset_base_mm = float(model.z_offset_base_mm + fitted_increment_base_mm)
    residual_after_mm = diffs_mm - fitted_increment_base_mm

    samples_path = session_dir / args.samples_name
    result_path = session_dir / args.result_name
    manifest_path = session_dir / args.manifest_name
    save_json(samples_path, all_samples)
    save_json(
        manifest_path,
        {
            "kind": "filtered_tracker_z_manifest",
            "written_at": datetime.now().isoformat(timespec="seconds"),
            "runs": clean_run_artifacts,
            "filters": {
                "max_joint_delta_s": float(args.max_joint_delta_s),
                "max_racket_reproj_px": float(args.max_racket_reproj_px),
            },
            "combined_samples_path": rel_or_abs(samples_path),
        },
    )

    result_payload = {
        "kind": "tracker_z_correction",
        "written_at": datetime.now().isoformat(timespec="seconds"),
        "tracker_output_dir": str(tracker_output_dir),
        "source_poe_config": str(poe_config_path),
        "export_config_path": str(export_config_path),
        "runs": list(args.runs),
        "filters": {
            "max_joint_delta_s": float(args.max_joint_delta_s),
            "max_racket_reproj_px": float(args.max_racket_reproj_px),
        },
        "fitted_correction": {
            "type": "constant_offset_base_z",
            "existing_offset_base_mm": float(model.z_offset_base_mm),
            "increment_offset_base_mm": fitted_increment_base_mm,
            "target_offset_base_mm": target_z_offset_base_mm,
        },
        "summary_before_mm": compute_summary(diffs_mm),
        "summary_after_mm": compute_summary(residual_after_mm),
        "per_run": run_summaries,
        "samples_path": rel_or_abs(samples_path),
        "filtered_manifest_path": rel_or_abs(manifest_path),
        "clean_run_artifacts": clean_run_artifacts,
    }
    save_json(result_path, result_payload)

    config_payload = load_json(poe_config_path)
    updated_config = apply_z_offset_to_config(
        config_payload,
        target_z_offset_base_mm,
        result_rel_path=rel_or_abs(result_path),
        source_runs=list(args.runs),
        sample_count=len(all_samples),
        filters=result_payload["filters"],
    )
    save_json(export_config_path, updated_config)

    print(f"Tracker output dir: {rel_or_abs(tracker_output_dir)}")
    print(f"Source POE config:  {rel_or_abs(poe_config_path)}")
    print(f"Export config:      {rel_or_abs(export_config_path)}")
    print(f"Session dir:        {rel_or_abs(session_dir)}")
    print(f"Samples saved:      {rel_or_abs(samples_path)}")
    print(f"Manifest saved:     {rel_or_abs(manifest_path)}")
    print(f"Result saved:       {rel_or_abs(result_path)}")
    print(f"Accepted samples:   {len(all_samples)}")
    print(f"Existing z offset:  {model.z_offset_base_mm:.3f} mm")
    print(f"Fitted increment:   {fitted_increment_base_mm:.3f} mm")
    print(f"Target z offset:    {target_z_offset_base_mm:.3f} mm")
    print(
        "Residual after fit: "
        f"mean={result_payload['summary_after_mm']['mean_mm']:.3f} mm  "
        f"std={result_payload['summary_after_mm']['std_mm']:.3f} mm  "
        f"abs_mean={result_payload['summary_after_mm']['abs_mean_mm']:.3f} mm"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
