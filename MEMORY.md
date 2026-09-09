# MEMORY

项目长期记忆文件。用于记录当前默认配置、近期决策和后续协作时容易忘的约定。

## 当前 Tracker 默认事实

- Tracker 默认使用四相机 rig（18F 是唯一在用的场地；16F 那套配置已于 `2026-09-06` 删除）：
  - `src/config/camera_18.json`
  - `src/config/four_camera_calib_18.json`
- 当前默认相机序列号（海康 MV-CS032-60GC，action 广播触发）：
  - 主相机 `DB0260414`
  - 从相机 `DB0260373`
  - 从相机 `DB0260405`
  - 从相机 `DB0260378`
- 当前默认采集参数：
  - `2048x1304`（`roi_height`，底部裁掉 232 行 = 15.1%；`OffsetY` 恒 0，所以保留像素的坐标与全幅逐点相同，标定不用改）
  - `40fps`（`acquisition_frame_rate=48`；单机千兆口的悬崖在 43~44 之间，实测 44 会塌到 27fps）
  - `3000us` 曝光
  - `12.78dB` 增益 + `16.5` digital shift
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

- `2026-09-07`
  - **报告端「拍心/拍速」换口径（用户定案：冻结柔度模型 + 直接替换旧列）**。原来拍心 = 刚性 `FK(q)` 的 TCP、
    拍速 = 解析 Jacobian `J(q)·q̇`；0907 黑标三场证明臂是弹性链（受载拍心落后 FK 157mm、卸载回收 66mm、
    卸载段视觉比 FK 快 +2.5m/s），所以现在：
    `p_head = FK(q_eff) + R_link6(q_eff)·tool_offset + [dx,dy,0]`，`q1_eff = q1 − c1±·τ1 + 0.00503·q̇1`、`q3/4/5_eff = q − c_j·τ_j`；
    拍速 = `p_head` 的 **±10ms 中心差分**（与 `tennis-man/rl_arm/env/swing_env.py` 评分同式），yaw 刚体项杠杆也改用 `p_head`。
    模型单源 = `tennis-man/rl_arm/assets/v04/visual_endpoint_20260907.json`（sha256 `973fbcd7…`，路径由 `TENNIS_MAN_ROOT` 解析，
    缺文件直接 SystemExit）；`arm.head_model` 在页面自报口径/指纹/窗口。**v0.3 无柔度标定，仍走刚性 TCP + Jacobian。**
    实现 `generate_curve3_html.py` 的 `_frozen_head_model` / `_head_effective_q` / `_add_face_angles` + JS `headModelNote`；
    单测 `test_report_prediction_contract.py::test_add_face_angles_v04_head_is_frozen_compliance_central_difference`
    与 `::test_v04_head_point_matches_rl_arm_frozen_endpoint`（与 rl_arm 官方实现逐值 0 差）。
    `tracker_20260907_185746` 已重生成：8 挥拍速降 15~35%、e_n 中位 0.404→0.611，**0907 之前的报告 e_n 不可与本列直接比**。
    ⚠ 差分半窗 5→20ms 会差 0.4~1.4m/s（主项是 `c·τ` 的导数，即弹性释放本身，不是数值噪声）——跨场比必须同窗；
    `fit_return_law.py` 仍是旧口径且 `RETURN_RESTITUTION=0.30` 就是在旧口径下标的，换口径要连那条律一起重标（未动）。
  - 诊断经验：一场里「后半段 RK 数据整段消失」先别怀疑视觉。`tracker_20260907_185746` 是 PC↔RK 的 WiFi 单向断
    （PC→RK 先死、RK→PC 晚 9.6s = Cyclone 参与者租约超时），RK 本机健康又跑了 90s——**RK 上 `run_arm_cpp_ready.sh`
    自己录了一份 `~/tennis-man/arm_controller/data/session_<N>/`，和 PC 侧 bag 一对照就分得清「RK 挂了」还是「链路断了」**。
  - `ros2/cyclonedds_18.xml` 的 `Peers` 只有两台 RK、没有 `127.0.0.1`，多播又是关的 ⇒ **PC 本机两个参与者互相发现不了，
    自家 bag 从来没录到过 `/pc_car_loc`/`/pc_world_ball_loc`/`/pc_rk_time_offset`**（RK 侧 bag 里有，收发本身没问题）。补 peer 即可。

- `2026-09-06`
  - 报告北极星表加「无臂回退锚」：臂没受理本抛或臂栈没跑（bag 无 `/joint_states`）时，FinalHT 回退为底盘末次 target 对应的那条
    `/predict_hit_pos`（RUN 内最后一次 target_x/y 变化按 bot.t+remaining↔ht 回配），源 `chassis_target`、格内标 `[车]`；
    只锚车移动 / PC 真值 / 车 yaw 列，TCP/拍面/目标拍速仍为空、不冒充臂目标。无臂时模式栏显示「无臂数据」。
    实现在 `generate_curve3_html.py` `[[final-ht-core]]`（`rkPredsForThrow`/`chassisTargetForThrow`），
    单测 `test_report_prediction_contract.py::test_final_ht_*chassis*`；`tracker_20260905_173206` 已重生成（7/7 [车]）。
  - 诊断经验：北极星表整表 `—` 先查 `<run>_rosbag/metadata.yaml` 有没有臂话题；173206 场臂话题一个都没有=臂栈没起，
    与 RL 版本无关（对表时板子 CST = PC run id + 15h）。
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

- `2026-07-03`: HTML 报表新增 Arm tab（仿 tennis-man arm_controller 的 session_viewer）。
  - `test_src/extract_arm_bag.py` 在 ROS2 环境（经 `ros2/run_ros2.bat`）读 `{run_id}_rosbag`，输出 `{run_id}_arm.json`：`/joint_states` 实际 + `/tennis/motor_command` 目标（首个轨迹点）+ status/arm_command/hit_pos/predict_hit_pos 事件；TCP 由本文件内置 FK 正解（逐值抄自 `arm_controller.compact_arm_kinematics`，不依赖隔壁 checkout）。
    - `2026-08-16` 起 FK **按车型分链**：`--car v03|v04`（不给就从本场 tracker JSON 的 `config.car_config_path` 推），选中的车写进 `_arm.json` 的 `car/car_source/fk_source`，报告端 `_add_face_angles` 按它复算；老 `_arm.json` 车型不符时页面就地复算 TCP。⚠ 两台车的臂不同（v0.4 肩高 +11.1cm、拍长 +5.3cm），选错车只会静默偏几厘米：0816_081524 用 v0.3 链算 v0.4，TCP 两列整场偏 (−5.4, −8.6)cm、FK 拍速低报 5%，而拍面 yaw/pitch 两车逐拍恒等（旋转链一致）——角度列看不出来。合同见 `test_src/test_arm_kinematics_cars.py`（黄金向量取自臂端导出的 `assets/<car>/test_vectors.json`）。
  - `generate_curve3_html.py` 新增 `--arm-json`（缺省自动探测 `<input>_arm.json`），Arm tab 为单 plot 四层 subplot（Position/Velocity/Effort/TCP，target 实线 vs actual 虚线，事件竖线）。
  - `run_tracker.py` post_run 链：bag 存在时先 Extract arm rosbag 再生成 HTML。
  - 性能教训（本机 Chrome 的 WebGL 是软渲染）：4 个独立 scattergl plot（4 个 WebGL context）或单 context 全量 15 万点都会把渲染器卡死几十秒；最终方案 = SVG `scatter`（lines 模式一条 trace 一条 path）+ 窗口化抽稀（hit/predict 事件 ±[-2,+4]s 内全分辨率，窗口外 2Hz），实测秒开。全量数据仍在 `_arm.json`，抽稀只发生在绘图端。
  - Arm tab 时间轴是 bag 相对时间（bag 首条消息 = 0s），与 tracker 主时间轴（首帧 exposure_pc）无对齐锚点，暂不混画。

- `2026-07-15`: HTML 报表删除 X/Y/Z Subplots tab，新增 RK Car Move tab（每抛底盘移动 ~100Hz 逐帧回放）。
  - `extract_rk_tracking_bag.py` 的 `/bot_state` 提取新增 `vx/vy/phase/steer_angle/remaining/v_target/target_active`；老 `_rk_tracking.json` 需重提取才有这些字段（HTML 端会退回 target_x 非空段分段、缺字段显示 —）。
  - 分段 = phase 离开 WAIT（RUN + BRAKE_IN_SWING + BRAKE_AFTER_SWING 为一段，天然一抛一段），前后补 0.5s；下拉框选第几次移动，2D 等比回放（播放/暂停/逐帧/倍速/滑条，快捷键 ←/→/空格），侧栏每帧显示 vx/vy 车速、yaw、IMU yaw_speed、舵轮角+旋转方向（steer_motor.velocity 最近邻 60ms）、剩余到位时间、目标位置/距离。
  - 0715 用户反馈定稿的 2D 视觉口径：目标星标与右栏**逐帧**刷新（RUN 中目标随预测更新移动，视野 bbox 包含全段目标；未激活帧沿用上一次下发值、右栏带 *）；舵轮 = 车旁短箭头（方向 yaw+steer，运动中按速度符号消歧）；车 = 圆点，yaw 不在 2D 里画（只在侧栏数值显示）。
  - 0712 bag 实测结论：`bot_state.vx/vy` 是世界系（与 dx/dt 中位差 0.02m/s）；速度方向 ≈ yaw+steer 或其 ±π（舵轮可反向驱动）；里程计跨移动连续（上段终点=下段起点），yaw 每次 WAIT 归零。
  - 技术坑：Plotly `scaleanchor` 的约束求解器会把算出的范围当"用户编辑"，`Plotly.react` 换移动段时旧 X 范围粘住不更新 → 等比范围自己算（两轴 m/px 取大者）+ react 后按 `_fullLayout` 实测轴长二次校正；`requestAnimationFrame` 在被遮挡/后台标签页被 Chrome 挂起 → 播放循环加 200ms setInterval 看门狗兜底（推进量按墙钟算，双驱动不重复计帧）。

## 协作提醒

- 如果更换相机 rig，必须同时检查：
  - `src/config/camera_18.json`
  - `src/config/four_camera_calib_18.json`
  - `src/config/tracker.json`
- `multi_calib.json` / 三目配置只保留作历史结果，不应再作为 tracker 默认入口。

## Camera API Notes

- `2026-03-24`: On live camera `DA7403103`, both `ReverseX` and `ReverseY` exist as bool nodes and can be read through the MVS universal node API.
- `ReverseX=True` + `ReverseY=True` can be used as device-side `180deg` rotation, which is more relevant to tracker performance than SDK-side image post-processing.
- `ReverseY` returned `0x80000106` (`MV_E_GC_ACCESS`) when written during grabbing, but became writable after `MV_CC_StopGrabbing()`. In practice, these nodes should be configured before `MV_CC_StartGrabbing()`.
- The SDK also exposes `MV_CC_RotateImage(...)`, but that is SDK-side rotation on acquired image data, not camera-side orientation.
- Independent probe on `2026-03-24` captured one frame without reverse and one frame with pre-grab `ReverseX=True, ReverseY=True`; the hardware-rotated frame matched the software `180deg` baseline strongly (`corr_rot180=0.958849` vs `corr_direct=-0.158286`, `mae_rot180=9.281` vs `mae_direct=84.769`).
- `BALL_TRACER_CAMERA_REVERSE_180` / `_X` / `_Y` 这组 A/B 开关已于 `2026-09-06` **删除**。
  它们的默认值还是 16F 的 `True`，而 `ReverseX/Y` 是**相机里的持久状态**：任何不经启动脚本
  直接开相机的脚本都会把四台相机翻 180° 并留在相机里，tag 照常解码（检测器旋转不变）但
  像素整体转半圈，实测车位姿飞到 `y=50m` / 重投影 `620px`。现在 `_CAMERA_REVERSE_X/Y` 在
  `src/ball_grabber.py` 里写死为 `False`（18F 正装），并在每次 `open_camera` 时无条件写入。
  `BALL_TRACER_SOFTWARE_ROTATE_180` 保留（纯软件旋转，默认 `0`，没有残留状态问题）。
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
- `2026-07-03`: tracker 用 rosbag 录制取代 pc_logger 事件接收器。
  - 新增独立 rosbag 录制边车 `src/rosbag_recorder.py`，由 `ros2/run_ros2.bat` 拉起，录制局域网内全部 ROS2 topic（`RecordOptions.all_topics`）到 `tracker_output/{run_id}_rosbag/`，与 tracker JSON 同 id；tracker JSON 的 `config.rosbag.bag_dir` 也记录该目录，便于 report viewer 配对加载。仅在保存日志时启动（`--no-log` 跳过），与 ros2_mode（发送方向，当前强制 off）无关。
  - 停止协议：`_close_sidecars` 创建 `{run_id}_rosbag.stop` 文件，子进程轮询到后 `Recorder.stop()` 写出 metadata.yaml，超时才 taskkill 兜底。jazzy 的 rosbag2_py 新 API：`record()` 非阻塞，必须先 `start_spin()`，否则本地 topic 都录不到。
  - 已删除 `src/pc_event_logger.py`、`src/pc_logger_protocol.py`，并从 `run_tracker.py` 移除 pc_logger 启动、`logger_control` 控制面（含各 Ros2Sink 的 `publish_logger_control` 与 `_publish_logger_control`）。`tracker_report_server.py` 仍能读旧 run 的 `_pc_logger.json`（找不到时优雅降级），未改动。
  - 组网（关键）：RK 端全用静态单播、`AllowMulticast=false`、`ROS_DOMAIN_ID=2`、cyclonedds。本机 PC 实际 IP 为 `192.168.50.230`（Wi-Fi），另有 Meta/VPN 虚拟网卡 `198.18.0.1`。已把 `ros2/cyclonedds.xml` 改成：`<NetworkInterface address="192.168.50.230" multicast="false"/>` 固定绑 LAN 网卡（避免选到 Meta 网卡，这是之前收不到 RK 的根因）、`AllowMulticast=false`、Peers 只留两台 RK（臂 `192.168.50.17`、底盘 `192.168.50.143`）。`src/ros2_support.py` 常量同步：`TRACKER_PC_IP=.230`、`CHASSIS_RK_IP=.143`（原为 `.68`，错的）。
  - RK 端已把 PC 配成 `.230` peer（用户完成），加 peer 前录制器要靠 learned-locator 发现远程、慢且不稳（普通 rclpy probe 能发现但 rosbag Recorder 常错过窗口）；加 peer 后发现秒级。实测：录制器 100ms 内订阅到 RK 的 `/net_test`，10s 录到 47 条，跨机录制通。
  - 防火墙：DDS 用 UDP `7400-7500`，需对该网卡的网络放行入站（本机实测已能收到，说明当前 Wi-Fi 网络已放行）。
- `2026-03-29`: `src/win_time_sync.py` now prints a 5-second summary while tracker is running, including ping receive count, receive rate, seq range/gaps, local inter-arrival interval, RK-side `t1` interval, inferred one-way delay jitter `((recv_i-recv_{i-1}) - (t1_i-t1_{i-1}))`, and local callback cost. `TimeSyncResponderProcess` in `src/run_tracker.py` no longer silences the child process, so these stats are visible in the same terminal/log stream as tracker when launched from `run_tracker_terminal.ps1`.
- `2026-07-05`: AprilTag 到车底盘中心的固定偏移按现场重新测量更新：`apriltag_center_to_car_base_offset` 从 `(60, 100, -340) mm` 改为 `(40, 160, -610) mm`（tag 中心在小车原点左方 40mm、后方 160mm、地面上方 610mm，偏移沿世界轴直接相加、不乘 yaw 的约定不变）。同步更新了 `src/config/arm_poe_racket_center*.json`（含两个 pruned15 变体）、`ArmCalibration/calibrate_poe_reprojection.py` 的导出常量、`src/config/arm_poe_racket_center.md` 与 `DEV.md` 的说明。
