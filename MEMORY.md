# MEMORY

项目长期记忆文件。用于记录当前默认配置、近期决策和后续协作时容易忘的约定。

## 当前 Tracker 默认事实

- Tracker 默认使用四相机 rig：
  - `src/config/camera.json`
  - `src/config/four_camera_calib.json`
- 当前默认相机序列号：
  - 主相机 `DA7403103`
  - 从相机 `DA8571029`
  - 从相机 `DA7403087`
  - 从相机 `DA8474746`
- 当前默认采集参数：
  - 全画幅 `2048x1536`
  - `35fps`
  - `3000us` 曝光
  - `23.5dB` 增益
- Tracker 的 YOLO 分片默认是：
  - `1280x1280` 切片
  - 压缩到 `640x640` 推理
- 当前默认检测 engine：
  - `yolo_model/tennis_yolo26_v2_20260203_b4_640.engine`
- 3D 球定位默认规则：
  - 至少 `2` 台相机参与三角化
  - 默认最大重投影误差 `15px`
- ROS2 输出：
  - `/pc_car_loc`
  - `/predict_hit_pos`

## 启动约定

- 启动 tracker 优先使用根目录脚本：
  - `.\run_tracker.ps1`
- 探测当前会选哪个环境：
  - `.\run_tracker.ps1 -ProbeOnly`
- `.venv_ros2` 是优先环境；有 CUDA / TensorRT 时应优先使用它。

## 近期变更

- `2026-03-23`
  - 删除旧的 `CLAUDE.md`，项目上下文以 `DEV.md` 和本文件为准。
  - `BallLocalizer` / `CarLocalizer` 默认标定切到 `four_camera_calib.json`。
  - Tracker 默认分片从 `1000x1000` 调回 `1280x1280`。
  - `DEV.md` 的 step 16 更新为当前 tracker 能力摘要。
  - 用户已人工确认：视频记录、球识别、轨迹追踪、车定位均已具备；多个网球位置 spot check 后，3D 误差整体为 cm 级。
  - 固定相机 rig 在现场放置约 1 周、包括受撞击扰动后，标定结果仍保持稳定。
  - 性能 debug 结论：
    - 四相机同步采集本身正常，`src.benchmark --duration 10` 实测 `35.1 fps`
    - 优化前，tracker `--no-video` 实测约 `22.3 fps`
    - 优化前，tracker 开启原始拼接视频保存时实测约 `13.8 fps`
    - `2026-03-24` 已新增并接入 `yolo_model/tennis_yolo26_v2_20260203_b4_640.engine`
    - `BallDetector` 已支持固定 batch engine 自动补齐/分批，默认接口也可直接使用 `b4` engine
    - Bayer 解码快路径已改成“先旋转 raw Bayer，再 demosaic”，像素结果与旧路径一致
    - 该解码快路径在 4 相机并行 benchmark 中约从 `11.9ms` 降到 `8.6ms`
    - 优化后，tracker `--no-video` 短跑约 `24.9 fps`
    - 优化后，tracker 开启原始拼接视频保存 `10s` 短跑约 `23.1 fps`
    - 当前剩余主要开销约为：`decode ~11.6ms`，`yolo ~26.7ms`，后台写视频 `~35ms`

## 协作提醒

- 如果更换相机 rig，必须同时检查：
  - `src/config/camera.json`
  - `src/config/four_camera_calib.json`
  - `src/config/tracker.json`
- `multi_calib.json` / 三目配置只保留作历史结果，不应再作为 tracker 默认入口。

## Camera API Notes

- `2026-03-24`: On live camera `DA7403103`, both `ReverseX` and `ReverseY` exist as bool nodes and can be read through the MVS universal node API.
- `ReverseX=True` + `ReverseY=True` can be used as device-side `180deg` rotation, which is more relevant to tracker performance than SDK-side image post-processing.
- `ReverseY` returned `0x80000106` (`MV_E_GC_ACCESS`) when written during grabbing, but became writable after `MV_CC_StopGrabbing()`. In practice, these nodes should be configured before `MV_CC_StartGrabbing()`.
- The SDK also exposes `MV_CC_RotateImage(...)`, but that is SDK-side rotation on acquired image data, not camera-side orientation.
- Independent probe on `2026-03-24` captured one frame without reverse and one frame with pre-grab `ReverseX=True, ReverseY=True`; the hardware-rotated frame matched the software `180deg` baseline strongly (`corr_rot180=0.958849` vs `corr_direct=-0.158286`, `mae_rot180=9.281` vs `mae_direct=84.769`).
- `src/ball_grabber.py` now supports temporary environment switches for A/B tests without changing defaults:
  - `BALL_TRACER_CAMERA_REVERSE_180=1` (or `BALL_TRACER_CAMERA_REVERSE_X/Y`)
  - `BALL_TRACER_SOFTWARE_ROTATE_180=0`
- With `BALL_TRACER_CAMERA_REVERSE_180=1` and `BALL_TRACER_SOFTWARE_ROTATE_180=0`, a real `run_tracker.ps1 -Duration 15 -NoVideo` run on `2026-03-24` reached `33.8 fps` (`519` frames / `15.4s`), close to the configured `35 fps`.
- `2026-03-24`: tracker mainline units are now meters end-to-end for ball 3D, car 3D, Curve3 state, JSON outputs, HTML, and ROS2 publish payloads.
- `src/run_tracker.py` now writes `config.distance_unit = "m"` into tracker JSON. Downstream tools should treat older JSON without that field as legacy mm data.
- `src/car_localizer.py` now applies `vehicle_reference.apriltag_center_to_car_base_offset_m = (0.06, 0.10, -0.34)` before returning `CarLoc`, so `/pc_car_loc` publishes the car base, not the AprilTag center.
- `2026-03-24`: tracker terminal/log output is forced to UTF-8 in both `run_tracker.ps1` and `src/run_tracker.py`, so redirected logs should no longer mix PowerShell UTF-8 with Python CP936 output.
- `2026-03-25`: tracker now supports `ball_detection_disabled_serials` in `src/config/tracker.json`. The current default disables camera `029` for ball YOLO and ball 3D only; capture, stitched video, saved video, JSON frame logs, and AprilTag car localization still keep all four cameras.
- `2026-03-27`: after re-checking the live image with hardware 180-degree camera reverse enabled, the current AprilTag appears in the lower part of the full image (`cy` about `1139-1413` on height `1536` for the detecting cameras). `src/car_localizer.py` therefore uses the lower 60% ROI, crop-only with native pixels and no resize.
- `2026-03-27`: HTML time axes and offline annotated-video overlay time are aligned to the same reference: the first frame's `exposure_pc` (`t=0`). `src/run_tracker.py` now records `config.first_frame_exposure_pc` and per-frame `elapsed_s`; `test_src/generate_curve3_html.py` and `test_src/annotate_video.py` both fall back to `frames[0].exposure_pc` for older JSON.
- `2026-03-27`: raw stitched tracker video now also prints the same time base as HTML: `#frame  t=...s  HH:MM:SS.mmm`, where `t` is relative to the first frame's `exposure_pc`. This is intended to make dropped-frame cases debuggable by eye.
- `2026-03-27`: added `run_tracker_terminal.ps1` for foreground tracker launches from terminal. It defaults to `ROS_DOMAIN_ID=2`, hardware 180-degree reverse on, software rotate off, and relies on `run_tracker.py`'s existing `KeyboardInterrupt` cleanup so `Ctrl+C` still flushes video/JSON cleanly.
- `2026-03-27`: added `annotate_latest_tracker.ps1` plus dual-JSON support. `test_src/annotate_video.py` now supports `--racket-json-output` to write a separate racket-only JSON (`frames[*].video_frame_idx` / `racket3d` / `racket_observations`), and `test_src/generate_curve3_html.py` now supports `--racket-json` so HTML can be generated from the base tracker JSON plus the separate racket JSON.
- `2026-03-27`: offline racket annotation was switched from the temporary tracker-style bbox/tile center logic to the ArmCalibration production logic in `src/racket_localizer.py`: `racket.onnx + racket_pose.onnx`, only keypoints `0-3` define the racket center, and one camera is accepted only when all center keypoints satisfy the configured score threshold (default `40.0`, min valid face keypoints `4`). The offline annotation path now converts the resulting racket 3D from mm to m before writing JSON/HTML artifacts.
- `2026-03-28`: tracker no longer replies to `/time_sync/ping` inside `DirectRos2Sink`. Instead, tracker startup now launches `ros2/start_time_sync.bat` as a dedicated child process and closes it on tracker exit. This keeps `time_sync` pong handling independent of tracker main-process scheduling while `/pc_car_loc` and `/predict_hit_pos` stay in-process under direct mode.
- `2026-03-28`: raw stitched tracker video is now saved as a `2x2` grid instead of a `1x4` strip, still using row-major camera order `103, 746 / 087, 029`. `test_src/annotate_video.py` was updated to annotate the same `2x2` layout, and it can still auto-detect older `1x4` recordings by video dimensions for backward compatibility.
- `2026-03-29`: `src/win_time_sync.py` now prints a 5-second summary while tracker is running, including ping receive count, receive rate, seq range/gaps, local inter-arrival interval, RK-side `t1` interval, inferred one-way delay jitter `((recv_i-recv_{i-1}) - (t1_i-t1_{i-1}))`, and local callback cost. `TimeSyncResponderProcess` in `src/run_tracker.py` no longer silences the child process, so these stats are visible in the same terminal/log stream as tracker when launched from `run_tracker_terminal.ps1`.
