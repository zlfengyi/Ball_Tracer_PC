#!/usr/bin/env python3
"""Offline opponent-racket bbox-centre impact measurement; record-only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import cv2
import onnxruntime as ort

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.opponent_racket_bbox_localizer import (  # noqa: E402
    OpponentRacketBBoxLocalizer,
)
from src.racket_bbox_bundle import BundleGates, load_cameras  # noqa: E402
from src.racket_contact import RacketContactEstimate, StableRacketContactSolver  # noqa: E402
from src.racket_impact import (  # noqa: E402
    RacketImpactEstimator,
    RacketImpactMeasurement,
    SynchronizedRacketFrame,
)


_SCHEMA = "racket_impact/v3"
_CONTROL_USAGE = "record_only"
_FRAME_TIME_SEMANTICS = "mosaic_group_mean_exposure_center_pc_perf_counter"
_VZ_SEMANTICS = "racket_head_bbox_center_world_velocity_proxy"
_DEFAULT_PLAYER_BOX_WORLD_M = {
    "x": [-3.0, 3.0],
    "y": [13.5, 20.5],
    "z": [0.0, 3.2],
}


@dataclass(frozen=True)
class ExactVideoFrame:
    video_frame_idx: int
    exposure_center_pc: float


@dataclass
class ContactJob:
    record: dict
    contact_anchor_pc: float
    frames: list[ExactVideoFrame]


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def iter_rk_world_payloads(rk_tracking: dict) -> Iterable[dict[str, float]]:
    """Yield only the current ``t0 + world.t / world.y.{x,y,z}`` contract."""

    t0 = rk_tracking.get("t0")
    world = rk_tracking.get("world")
    if not isinstance(t0, (int, float)) or not math.isfinite(float(t0)):
        raise ValueError("rk_tracking.t0 must be finite")
    if not isinstance(world, dict) or set(("t", "y")) - set(world):
        raise ValueError("rk_tracking.world must contain t and y")
    relative_t = world["t"]
    values = world["y"]
    if not isinstance(relative_t, list) or not isinstance(values, dict):
        raise ValueError("rk_tracking.world must use t[] and y{}")
    if any(key not in values for key in ("x", "y", "z")):
        raise ValueError("rk_tracking.world.y must contain x/y/z arrays")
    axes = [values[key] for key in ("x", "y", "z")]
    if not all(isinstance(axis, list) for axis in axes):
        raise ValueError("rk_tracking.world.y x/y/z must be arrays")
    length = len(relative_t)
    if length == 0 or any(len(axis) != length for axis in axes):
        raise ValueError("rk_tracking.world arrays must be nonempty and equal length")
    for relative, x_m, y_m, z_m in zip(relative_t, *axes):
        try:
            payload = {
                "t": float(t0) + float(relative),
                "x": float(x_m),
                "y": float(y_m),
                "z": float(z_m),
            }
        except (TypeError, ValueError) as exc:
            raise ValueError("rk_tracking.world contains a nonnumeric value") from exc
        if not all(math.isfinite(value) for value in payload.values()):
            raise ValueError("rk_tracking.world contains a nonfinite value")
        yield payload


def solve_contacts(
    rk_tracking: dict,
    solver: StableRacketContactSolver | None = None,
) -> list[RacketContactEstimate]:
    contact_solver = solver or StableRacketContactSolver()
    results = []
    for payload in iter_rk_world_payloads(rk_tracking):
        result = contact_solver.add(payload)
        if result is not None:
            results.append(result)
    final_result = contact_solver.finish()
    if final_result is not None:
        results.append(final_result)
    return results


def validate_clock_bridge(config: dict) -> tuple[dict | None, str]:
    bridge = config.get("rk_clock_bridge")
    if not isinstance(bridge, dict):
        return None, "missing_rk_clock_bridge"
    if any(key not in bridge for key in ("pc_minus_rk", "mad", "n")):
        return None, "invalid_rk_clock_bridge_fields"
    try:
        pc_minus_rk = float(bridge["pc_minus_rk"])
        mad_s = float(bridge["mad"])
        n_float = float(bridge["n"])
    except (TypeError, ValueError):
        return None, "invalid_rk_clock_bridge_values"
    if not all(math.isfinite(value) for value in (pc_minus_rk, mad_s, n_float)):
        return None, "invalid_rk_clock_bridge_values"
    if n_float < 20.0:
        return None, "rk_clock_bridge_insufficient_samples"
    if mad_s < 0.0 or mad_s > 0.005:
        return None, "rk_clock_bridge_excessive_mad"
    return {
        "source": bridge.get("source"),
        "pc_minus_rk": pc_minus_rk,
        "mad_s": mad_s,
        "n": int(n_float),
    }, ""


def exposure_center_offset_s(config: dict) -> float:
    """Half the group-mean exposure duration added to logged group-mean start."""
    serials = config.get("serials")
    settings = config.get("camera_settings")
    if not isinstance(serials, list) or not serials or not isinstance(settings, dict):
        raise ValueError("config serials/camera_settings are required")
    exposures_us = []
    for serial in serials:
        camera = settings.get(serial)
        if not isinstance(camera, dict) or "exposure_us" not in camera:
            raise ValueError(f"missing exposure_us for camera {serial}")
        exposure_us = float(camera["exposure_us"])
        if not math.isfinite(exposure_us) or exposure_us <= 0.0:
            raise ValueError(f"invalid exposure_us for camera {serial}")
        exposures_us.append(exposure_us)
    return 0.5e-6 * sum(exposures_us) / len(exposures_us)


def exact_video_frames(base_data: dict, center_offset_s: float) -> list[ExactVideoFrame]:
    config = base_data.get("config")
    frames = base_data.get("frames")
    if not isinstance(config, dict) or config.get("video_frame_mapping_exact") is not True:
        raise ValueError("tracker video frame mapping is not exact")
    if not isinstance(frames, list):
        raise ValueError("tracker frames must be an array")
    result = []
    seen = set()
    for frame in frames:
        if not isinstance(frame, dict) or "video_frame_idx" not in frame:
            continue
        if frame.get("video_mapping_exact") is not True:
            raise ValueError("frame has a non-exact video mapping")
        video_frame_idx = frame["video_frame_idx"]
        exposure_start_pc = frame.get("exposure_pc")
        if (
            isinstance(video_frame_idx, bool)
            or not isinstance(video_frame_idx, int)
            or video_frame_idx < 0
            or not isinstance(exposure_start_pc, (int, float))
        ):
            raise ValueError("invalid video_frame_idx/exposure_pc")
        exposure_center_pc = float(exposure_start_pc) + center_offset_s
        if not math.isfinite(exposure_center_pc) or video_frame_idx in seen:
            raise ValueError("duplicate or nonfinite exact video frame")
        seen.add(video_frame_idx)
        result.append(ExactVideoFrame(video_frame_idx, exposure_center_pc))
    if not result:
        raise ValueError("tracker contains no exact video frames")
    return sorted(result, key=lambda frame: frame.video_frame_idx)


def frames_for_contact(
    frames: Sequence[ExactVideoFrame],
    contact_anchor_pc: float,
    *,
    window_before_contact_s: float,
    window_after_contact_s: float,
) -> list[ExactVideoFrame]:
    start = contact_anchor_pc - window_before_contact_s
    end = contact_anchor_pc + window_after_contact_s
    return [
        frame
        for frame in frames
        if start <= frame.exposure_center_pc <= end
    ]


def serialize_contact_estimate(
    estimate: RacketContactEstimate,
    contact_index: int,
    pc_minus_rk: float | None,
) -> dict:
    world = estimate.contact_anchor_world_m
    return {
        "contact_index": int(contact_index),
        "status": "rejected",
        "failure_reason": estimate.failure_reason,
        "contact_anchor_status": "accepted" if estimate.valid else "rejected",
        "vision_evaluated": False,
        "vz_world_mps": None,
        "vz_semantics": _VZ_SEMANTICS,
        "acceptance_mode": estimate.acceptance_mode,
        "contact_anchor_t_rk": estimate.contact_anchor_t_rk,
        "contact_anchor_t_pc": (
            estimate.contact_anchor_t_rk + pc_minus_rk
            if estimate.contact_anchor_t_rk is not None and pc_minus_rk is not None
            else None
        ),
        "contact_anchor_world_m": (
            {"x": world[0], "y": world[1], "z": world[2]}
            if world is not None
            else None
        ),
        "contact_model": estimate.contact_model,
        "contact_height_m": estimate.contact_height_m,
        "prefix_anchor_t_rk": list(estimate.prefix_anchor_t_rk),
        "prefix_spread_s": estimate.prefix_spread_s,
        "contact_point_spread_m": estimate.contact_point_spread_m,
        "ball_fit_rms_m": estimate.ball_fit_rms_m,
        "first_observation_lead_s": estimate.first_observation_lead_s,
        "approach_speed_mps": estimate.approach_speed_mps,
        "contact_fit_n_points": estimate.n_points,
        "contact_window_shift": estimate.window_shift,
    }


def serialize_measurement(measurement: RacketImpactMeasurement) -> dict:
    inlier_indices = set(
        measurement.bundle_diagnostics.get("inlier_observation_indices", [])
    )
    return {
        "accepted": measurement.accepted,
        "reason": measurement.reason,
        "contact_anchor_pc": measurement.contact_anchor_pc,
        "window_start_pc": measurement.window_start_pc,
        "window_end_pc": measurement.window_end_pc,
        "observation_semantics": measurement.observation_semantics,
        "velocity_semantics": measurement.velocity_semantics,
        "bbox_center_vz_world_mps": measurement.bbox_center_vz_world_mps,
        "n_input_frames": measurement.n_input_frames,
        "n_contact_window_frames": measurement.n_contact_window_frames,
        "camera_support_frames": dict(measurement.camera_support_frames),
        "rejection_counts": dict(measurement.rejection_counts),
        "raw_bbox_observations": [
            {
                "bundle_inlier": index in inlier_indices,
                "video_frame_idx": item.video_frame_idx,
                "exposure_center_pc": item.exposure_center_pc,
                "time_to_anchor_s": item.time_to_anchor_s,
                "serial": item.serial,
                "candidate_rank": item.candidate_rank,
                "bbox_confidence": item.bbox_confidence,
                "bbox_native_xyxy": list(item.bbox_native_xyxy),
                "bbox_center_native_xy": list(item.bbox_center_native_xy),
            }
            for index, item in enumerate(measurement.raw_bbox_observations)
        ],
        "bundle_diagnostics": dict(measurement.bundle_diagnostics),
    }


def measure_video_jobs(
    video_path: Path,
    serials: list[str],
    estimator: RacketImpactEstimator,
    jobs: list[ContactJob],
    *,
    capture_factory: Callable[[str], object] = cv2.VideoCapture,
    panel_extractor: Callable[[object, list[str]], dict] | None = None,
) -> None:
    if panel_extractor is None:
        from test_src.annotate_video import extract_fullres_panels

        panel_extractor = extract_fullres_panels
    jobs_with_frames = [job for job in jobs if job.frames]
    if not jobs_with_frames:
        return
    wanted: dict[int, list[tuple[ContactJob, ExactVideoFrame]]] = {}
    completion: dict[int, list[ContactJob]] = {}
    prepared: dict[int, list[SynchronizedRacketFrame]] = {}
    for job in jobs_with_frames:
        for frame in job.frames:
            wanted.setdefault(frame.video_frame_idx, []).append((job, frame))
        completion.setdefault(job.frames[-1].video_frame_idx, []).append(job)
        prepared[id(job)] = []

    capture = capture_factory(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    last_needed = max(wanted)
    decoded_through = -1
    try:
        for video_frame_idx in range(last_needed + 1):
            ok, stitched = capture.read()
            if not ok:
                break
            decoded_through = video_frame_idx
            targets = wanted.get(video_frame_idx, [])
            if targets:
                panels = panel_extractor(stitched, serials)
                for job, frame in targets:
                    prepared[id(job)].append(
                        estimator.prepare_frame(
                            video_frame_idx,
                            frame.exposure_center_pc,
                            panels,
                        )
                    )
            for job in completion.get(video_frame_idx, []):
                measurement = estimator.measure(
                    prepared.pop(id(job)),
                    job.contact_anchor_pc,
                )
                job.record["vision_evaluated"] = True
                job.record["measurement"] = serialize_measurement(measurement)
                job.record["status"] = (
                    "accepted" if measurement.accepted else "rejected"
                )
                job.record["vz_world_mps"] = (
                    measurement.bbox_center_vz_world_mps
                    if measurement.accepted
                    else None
                )
                job.record["failure_reason"] = measurement.reason
    finally:
        capture.release()
    if decoded_through < last_needed:
        for job in jobs_with_frames:
            if id(job) in prepared:
                job.record["status"] = "rejected"
                job.record["failure_reason"] = (
                    "video_decode_ended_before_required_frame"
                )


def _paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    input_path = Path(args.input).resolve()
    rk_path = (
        Path(args.rk_tracking_json).resolve()
        if args.rk_tracking_json
        else input_path.with_name(input_path.stem + "_rk_tracking.json")
    )
    video_path = (
        Path(args.video).resolve() if args.video else input_path.with_suffix(".mp4")
    )
    output_path = (
        Path(args.output).resolve()
        if args.output
        else input_path.with_name(input_path.stem + "_racket_impact.json")
    )
    for path in (input_path, rk_path, video_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    return input_path, rk_path, video_path, output_path


def _ranges(
    player_box_world_m: dict,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    result = []
    for axis in ("x", "y", "z"):
        values = tuple(float(value) for value in player_box_world_m[axis])
        if len(values) != 2 or not all(math.isfinite(value) for value in values):
            raise ValueError(f"player_box_world_m.{axis} must contain two finite values")
        if values[0] >= values[1]:
            raise ValueError(f"player_box_world_m.{axis} must be increasing")
        result.append(values)
    return tuple(result)  # type: ignore[return-value]


def _providers(requested: list[str] | None) -> list[str]:
    if requested:
        providers = list(requested)
    else:
        available = ort.get_available_providers()
        providers = [
            name
            for name in (
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            )
            if name in available
        ]
    if not providers:
        raise RuntimeError("no supported ONNX execution provider is available")
    if any(
        name in {"TensorrtExecutionProvider", "CUDAExecutionProvider"}
        for name in providers
    ):
        # Import loads the bundled CUDA DLLs before ORT creates the session.
        import torch  # noqa: F401
    return providers


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    input_path, rk_path, video_path, output_path = _paths(args)
    base_data = load_json(input_path)
    rk_tracking = load_json(rk_path)
    base_config = base_data.get("config")
    if not isinstance(base_config, dict):
        raise ValueError("tracker config is required")
    serials = base_config.get("serials")
    if not isinstance(serials, list) or len(serials) < 3:
        raise ValueError("tracker config must contain at least three serials")
    impact_config = base_config.get("racket_impact")
    if not isinstance(impact_config, dict):
        impact_config = {}
    player_box_world_m = impact_config.get(
        "player_box_world_m", _DEFAULT_PLAYER_BOX_WORLD_M
    )
    if not isinstance(player_box_world_m, dict):
        raise ValueError("player_box_world_m must be an object")
    x_range, y_range, z_range = _ranges(player_box_world_m)

    contact_config = {
        "contact_height_m": args.contact_height_m,
        "min_approach_mps": args.contact_min_approach_mps,
        "max_gap_s": args.contact_max_gap_s,
        "cooldown_s": args.contact_cooldown_s,
        "max_prefix_spread_s": args.contact_max_prefix_spread_s,
        "max_ball_fit_rms_m": args.contact_max_ball_fit_rms_m,
        "max_step_speed_mps": args.contact_max_step_speed_mps,
        "consensus_max_prefix_spread_s": (
            args.contact_consensus_max_prefix_spread_s
        ),
        "consensus_max_contact_spread_m": (
            args.contact_consensus_max_contact_spread_m
        ),
        "consensus_max_ball_fit_rms_m": (
            args.contact_consensus_max_ball_fit_rms_m
        ),
        "contact_x_range_m": x_range,
        "contact_y_range_m": y_range,
        "contact_reach_margin_m": args.contact_reach_margin_m,
    }
    estimates = solve_contacts(
        rk_tracking,
        StableRacketContactSolver(**contact_config),
    )
    indexed_estimates = list(enumerate(estimates))
    if args.contact_index is not None:
        requested = set(args.contact_index)
        if any(index < 0 or index >= len(indexed_estimates) for index in requested):
            raise IndexError("contact-index is outside the solved contact list")
        indexed_estimates = [
            item for item in indexed_estimates if item[0] in requested
        ]
    if args.max_contacts is not None:
        if args.max_contacts <= 0:
            raise ValueError("max-contacts must be positive")
        indexed_estimates = indexed_estimates[: args.max_contacts]

    bridge, bridge_failure = validate_clock_bridge(base_config)
    pc_minus_rk = bridge["pc_minus_rk"] if bridge is not None else None
    records = [
        serialize_contact_estimate(estimate, index, pc_minus_rk)
        for index, estimate in indexed_estimates
    ]
    jobs: list[ContactJob] = []
    exposure_offset = None
    frame_failure = ""
    exact_frames: list[ExactVideoFrame] = []
    if bridge is not None:
        try:
            exposure_offset = exposure_center_offset_s(base_config)
            exact_frames = exact_video_frames(base_data, exposure_offset)
        except (TypeError, ValueError) as exc:
            frame_failure = str(exc)

    for (_index, estimate), record in zip(indexed_estimates, records):
        if not estimate.valid:
            continue
        if bridge is None:
            record["failure_reason"] = bridge_failure
            continue
        if frame_failure:
            record["failure_reason"] = "invalid_exact_video_frame_timing"
            record["frame_failure_detail"] = frame_failure
            continue
        assert record["contact_anchor_t_pc"] is not None
        selected_frames = frames_for_contact(
            exact_frames,
            record["contact_anchor_t_pc"],
            window_before_contact_s=args.window_before_contact_s,
            window_after_contact_s=args.window_after_contact_s,
        )
        record["video_frame_indices"] = [
            frame.video_frame_idx for frame in selected_frames
        ]
        record["frame_exposure_center_pc"] = [
            frame.exposure_center_pc for frame in selected_frames
        ]
        if not selected_frames:
            record["failure_reason"] = "no_exact_contact_window_video_frames"
            continue
        jobs.append(ContactJob(record, record["contact_anchor_t_pc"], selected_frames))

    bbox_model_path = Path(args.bbox_model).resolve()
    providers = _providers(args.provider)
    model_config = {
        "bbox_model_path": str(bbox_model_path),
        "bbox_model_sha256": (
            _sha256(bbox_model_path) if bbox_model_path.is_file() else None
        ),
        "providers_requested": list(args.provider) if args.provider else ["auto"],
        "providers_selected": providers,
        "providers_actual": None,
        "bbox_confidence_min": args.bbox_confidence_min,
        "bbox_nms_iou": args.bbox_nms_iou,
    }

    margin = args.bundle_player_world_margin_m
    world_volume_m = tuple(
        (lower - margin, upper + margin)
        for lower, upper in (x_range, y_range, z_range)
    )
    bundle_gates = BundleGates(
        world_volume_m=world_volume_m,
        bbox_confidence_min=args.bbox_confidence_min,
        window_before_contact_s=args.window_before_contact_s,
        window_after_contact_s=args.window_after_contact_s,
        reprojection_error_px=args.bundle_reprojection_error_px,
        min_supported_frames=args.bundle_min_supported_frames,
        min_fit_span_s=args.bundle_min_fit_span_s,
        min_abs_vz_mps=args.bundle_min_abs_vz_mps,
        max_speed_mps=args.bundle_max_speed_mps,
    )
    if jobs:
        if not bbox_model_path.is_file():
            raise FileNotFoundError(bbox_model_path)
        calib_path = base_config.get("calib_config_path")
        if not isinstance(calib_path, str):
            raise ValueError("tracker config calib_config_path is required")
        localizer = OpponentRacketBBoxLocalizer(
            bbox_model_path,
            bbox_confidence_min=args.bbox_confidence_min,
            providers=providers,
            nms_iou=args.bbox_nms_iou,
        )
        estimator = RacketImpactEstimator(
            player_box_world_m,
            localizer=localizer,
            cameras=load_cameras(calib_path),
            bundle_gates=bundle_gates,
            max_candidates_per_camera_frame=(
                args.max_candidates_per_camera_frame
            ),
            bbox_edge_margin_px=args.bbox_edge_margin_px,
            player_roi_padding_px=args.player_roi_padding_px,
        )
        model_config["providers_actual"] = estimator.provider_info
        measure_video_jobs(video_path, serials, estimator, jobs)

    output = {
        "schema": _SCHEMA,
        "control_usage": _CONTROL_USAGE,
        "frame_time_semantics": _FRAME_TIME_SEMANTICS,
        "vz_semantics": _VZ_SEMANTICS,
        "source": {
            "tracker_json": str(input_path),
            "rk_tracking_json": str(rk_path),
            "video": str(video_path),
        },
        "config": {
            "contact_model": "rk_ball_z_crossing_fixed_height",
            "contact_prefix_points": [6, 7, 8],
            **contact_config,
            "rk_clock_bridge": {
                "valid": bridge is not None,
                "failure_reason": bridge_failure,
                "value": bridge,
            },
            "exposure_center_offset_s": exposure_offset,
            "player_box_world_m": player_box_world_m,
            "bundle_world_volume_m": [list(values) for values in world_volume_m],
            "bundle_player_world_margin_m": margin,
            "bundle_reprojection_error_px": args.bundle_reprojection_error_px,
            "bundle_min_supported_frames": args.bundle_min_supported_frames,
            "bundle_min_fit_span_s": args.bundle_min_fit_span_s,
            "bundle_min_abs_vz_mps": args.bundle_min_abs_vz_mps,
            "bundle_max_speed_mps": args.bundle_max_speed_mps,
            "window_before_contact_s": args.window_before_contact_s,
            "window_after_contact_s": args.window_after_contact_s,
            "max_candidates_per_camera_frame": (
                args.max_candidates_per_camera_frame
            ),
            "bbox_edge_margin_px": args.bbox_edge_margin_px,
            "player_roi_padding_px": args.player_roi_padding_px,
            "model": model_config,
        },
        "racket_impact": records,
        "summary": {
            "contacts": len(records),
            "contact_anchor_accepted": sum(
                record["contact_anchor_status"] == "accepted" for record in records
            ),
            "vision_evaluated": sum(record["vision_evaluated"] for record in records),
            "accepted": sum(record["status"] == "accepted" for record in records),
            "rejected": sum(record["status"] != "accepted" for record in records),
            "processing_elapsed_s": time.perf_counter() - started,
        },
    }
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="base tracker JSON")
    parser.add_argument("--rk-tracking-json")
    parser.add_argument("--video")
    parser.add_argument("--output")
    parser.add_argument("--contact-index", type=int, action="append")
    parser.add_argument("--max-contacts", type=int)
    parser.add_argument(
        "--bbox-model",
        default=str(_ROOT / "yolo_model" / "opponent_racket_head_bbox.onnx"),
    )
    parser.add_argument("--provider", action="append")
    parser.add_argument("--bbox-confidence-min", type=float, default=0.05)
    parser.add_argument("--bbox-nms-iou", type=float, default=0.45)
    parser.add_argument("--max-candidates-per-camera-frame", type=int, default=3)
    parser.add_argument("--bbox-edge-margin-px", type=float, default=8.0)
    parser.add_argument("--player-roi-padding-px", type=int, default=40)
    parser.add_argument("--window-before-contact-s", type=float, default=0.125)
    parser.add_argument("--window-after-contact-s", type=float, default=0.035)
    parser.add_argument("--bundle-reprojection-error-px", type=float, default=8.0)
    parser.add_argument("--bundle-min-supported-frames", type=int, default=3)
    parser.add_argument("--bundle-min-fit-span-s", type=float, default=0.055)
    parser.add_argument("--bundle-min-abs-vz-mps", type=float, default=0.30)
    parser.add_argument("--bundle-max-speed-mps", type=float, default=35.0)
    parser.add_argument("--bundle-player-world-margin-m", type=float, default=0.60)
    parser.add_argument("--contact-height-m", type=float, default=1.05)
    parser.add_argument("--contact-min-approach-mps", type=float, default=3.0)
    parser.add_argument("--contact-max-gap-s", type=float, default=0.25)
    parser.add_argument("--contact-cooldown-s", type=float, default=1.5)
    parser.add_argument("--contact-max-step-speed-mps", type=float, default=40.0)
    parser.add_argument("--contact-max-prefix-spread-s", type=float, default=0.015)
    parser.add_argument("--contact-max-ball-fit-rms-m", type=float, default=0.15)
    parser.add_argument(
        "--contact-consensus-max-prefix-spread-s", type=float, default=0.060
    )
    parser.add_argument(
        "--contact-consensus-max-contact-spread-m", type=float, default=0.15
    )
    parser.add_argument(
        "--contact-consensus-max-ball-fit-rms-m", type=float, default=0.40
    )
    parser.add_argument("--contact-reach-margin-m", type=float, default=0.25)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    output = run(parse_args(argv))
    print(
        f"{len(output['racket_impact'])} contacts: "
        f"{output['summary']['accepted']} accepted, "
        f"{output['summary']['rejected']} rejected"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
