# `run_tracker_with_predict` contract

Launch with:

```powershell
.\run_tracker_with_predict.ps1 -Car v04
```

The launcher is a thin opt-in over `run_tracker.ps1`; camera, floor, ROS2 and
environment selection stay identical. The original launcher remains unchanged
unless `-EnableRkTimeAlign` is explicitly supplied.

## Per-throw time mapping

RK input is `/ball_world_topic` (`std_msgs/msg/String`) with JSON
`t/x/y/z`. The online mapping uses the two observation timestamps only:

```text
pc_exposure_t = rk_ball_world_t + pc_minus_rk
rk_timestamp  = pc_exposure_t - pc_minus_rk
```

Network receipt delay is not added to either equation. On every throw, local
arrival order is used only to prevent two different physical throws from being
paired; the fitted value itself comes exclusively from the payload and exposure
timestamps.

The release accuracy target is an independently checked absolute-offset
`P90 < 5 ms`; an independently measured error above 10 ms is an abnormal
release result. The trajectory fit cannot observe its own true clock error, so
the online protection remains fail-closed quality/identity gates rather than
pretending that fit residual is clock ground truth.

For the PC stream, the aligner also receives the existing tracker state
(`tracking_s0` / `in_landing` / `tracking_s1`) and its first-bounce estimate.
Every PC and RK point is stamped with local `perf_counter()` when it becomes
available to the alignment worker. These arrival times are identity, dirty-data
and throw-lifecycle gates only; they never enter the fitted clock offset. The
worker also checks the active throw every 100 ms, so a stopped stream can reach
its terminal result without waiting for the first packet of the next throw.
Malformed numeric payloads, including values that overflow `float`, are dropped
as one dirty packet and do not stop the worker.

Both PC and RK must independently observe one complete incoming flight:

1. the ball moves toward the car and crosses world `y=8 m`;
2. a descending pre-bounce segment is present;
3. a physical low point is followed by Stage1 evidence: at least 200 ms or six
   clean post-bounce points on each side. A continuous stream is collected for
   up to 300 ms; if valid PC/RK joint pairing stops, an already-proven clean
   prefix may be fitted at the conflict or after the 350 ms expiry gate;
4. the trimmed, bounce-centred trajectories pass the report-equivalent z(t)
   residual, coverage and ambiguity gates.

Each physical PC/RK flight pair is attempted once. A rejected or incomplete
throw leaves the previous `pc_minus_rk` unchanged. The next complete accepted
throw atomically replaces it.

One out-of-order RK packet is only discarded. Two coherent RK samples on a
clock axis more than five seconds below the previous high-water mark prove an
RK clock restart: the accepted offset and active throw buffers are cleared,
both RK-facing outputs are gated again, and a later complete throw must rebuild
the offset.

## RK-facing PC topics

`/pc_car_loc` and `/pc_world_ball_loc` use `BEST_EFFORT` QoS. In this opt-in
mode, neither topic publishes before the first accepted offset. Once alignment
exists, every payload on both topics has the same stable timestamp shape: the
original PC exposure `t` plus `rk_timestamp` for that exposure on the RK clock.
The default `run_tracker.ps1` entry remains unchanged and continues publishing
the established `/pc_car_loc` payload without online alignment.

### `/pc_car_loc`

After alignment, the existing AprilTag car pose payload keeps its PC timestamp
`t` and adds `rk_timestamp`:

```json
{
  "topic": "car_loc",
  "x": 0.1234,
  "y": 1.2345,
  "z": 0.0,
  "yaw": 0.4567,
  "yaw_valid": true,
  "t": 411987136.914,
  "rk_timestamp": 411966174.888,
  "tag_id": 0,
  "tag_ids": [0, 1]
}
```

`rk_timestamp` is the RK-axis time of the AprilTag source exposure, not the ROS
publish/receive time.

### `/pc_world_ball_loc`

PC publishes each current four-camera world-ball observation as:

```json
{
  "topic": "world_ball_loc",
  "x": 0.1234,
  "y": 7.4321,
  "z": 1.2345,
  "t": 411987136.947,
  "rk_timestamp": 411966174.921
}
```

`t` is the PC camera exposure timestamp. `rk_timestamp` is that same physical
observation mapped onto the RK clock. The PC does not publish `/predict_hit_pos`
in this mode.

### `/pc_rk_time_offset`

Every accepted per-throw alignment publishes exactly one event with `RELIABLE`,
depth-4 QoS. Rejected or incomplete throws publish nothing:

```json
{
  "topic": "pc_rk_time_offset",
  "update": 2,
  "pc_minus_rk_s": 20962.026031,
  "pc_timestamp": 411987137.004,
  "rk_timestamp": 411966174.978
}
```

`pc_timestamp` and `rk_timestamp` describe the same event-payload construction
instant on the two clock axes. The offset sign remains
`pc_timestamp = rk_timestamp + pc_minus_rk_s`.

This is a notification, not a cross-topic ordering barrier. QoS durability is
`VOLATILE`, so a subscriber that starts late waits for the next accepted throw;
`update` is session-local and restarts from 1 with the tracker process.

Every terminal alignment attempt is also written to the tracker session JSON.
Rejected terminals retain the previous offset and record a concrete `reason`
plus `features` describing the paired-arrival, 8 m, bounce, pre/post-point and
trajectory-span gates that made the decision.

The primary rejection reasons are intentionally specific:

- `joint_identity_lost`: the two live streams no longer form one arrival/spatial
  track;
- `cross_incomplete` / `bounce_before_cross`: both sides did not prove the
  strict first-bounce-after-8 m contract;
- `first_bounce_incomplete` / `bounce_invalid` / `stage1_incomplete`: the first
  landing or its rising tail was missing or physically inconsistent;
- `flight_curve_dirty` / `spatial_residual` / `bounce_mismatch`: the retained
  samples disagree spatially or around the bounce. Spatial agreement is
  translation-invariant: the median XY displacement between the two tracks is
  the car's pose belief error (reported as `xy_shift`, largest before the first
  accepted offset has unblocked `/pc_car_loc`) and never rejects a throw; only
  the de-translated residuals (`xy_res_median` / `xy_res_p90`) gate;
- `insufficient_overlap` / `insufficient_coverage`: too little bounded
  timestamp overlap remained;
- `shape_residual`, `ambiguous_rival`, `wide_profile`, `search_edge`, or
  `offset_jump`: the phase fit was inaccurate, non-unique, boundary-limited, or
  changed more than 30 ms from the last accepted offset.

`features` keeps the evidence behind the one primary reason, including joint
pair count, arrival/spatial gaps, each side's 8 m status, bounce y, pre/post
point counts, time spans, and (when fitting ran) residual/coverage/profile and
XY consistency values.
