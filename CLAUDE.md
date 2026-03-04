# CLAUDE.md

本文件为 Claude Code 提供项目上下文，每次打开项目自动加载。

## 项目概述

Ball Tracer PC — 网球追踪系统：双目立体相机同步拍摄 → YOLO 检测网球 → 三角定位获得 3D 坐标 → 轨迹拟合预测击球点。
基于海康 MVS SDK，Python 3.8+。

## 运行方式

```bash
# 主管线：双目采集 → YOLO → 3D定位 → curve3预测 → 视频+JSON
python -m src.run_tracker [--duration 60] [--display] [--ideal-hit-z 800]

# 离线回放：从 JSON 回放 curve3，调参验证
python test_src/test_curve3_replay.py --input tracker_output/xxx.json [--cor 0.78]

# 生成交互式 HTML 可视化
python test_src/generate_curve3_html.py --input tracker_output/xxx.json

# 标定
python -m calibration.run_calibration       # 双目标定
python -m calibration.register_ground       # 世界坐标注册

# 其他
python -m src.demo           # 4 相机同步演示
python -m src.benchmark      # 性能基准测试
```

依赖：`opencv-python`、`numpy`、`ultralytics`

## 核心管线流程

```
相机同步拍摄(SyncCapture) → 图像解码(frame_to_numpy) → YOLO检测(detect_batch)
  → 三角定位(triangulate) → 轨迹预测(Curve3Tracker.update) → 输出JSON+视频
```

### 主循环逐步说明 (run_tracker.py)

1. **相机采集** — `SyncCapture.from_config("src/config/camera.json")`
   - 硬件触发：主相机 DA8199303 自由运行 → Line1 输出 ExposureStartActive
   - 从相机 DA8199402（右）/ DA8199285（左）→ Line0 下降沿触发
   - master_min_bandwidth=true：主相机仅触发，不参与输出
   - `sync.get_frames()` → `{序列号: Frame}`（exposure_start_pc = 校准后的 perf_counter 时间）

2. **图像解码** — `frame_to_numpy(frame)` → BGR numpy 数组（~4.8ms）
   - 自动处理 Bayer 去马赛克（海康 RG↔BG 命名反转已处理）

3. **YOLO 网球检测** — `BallDetector.detect_batch([左图, 右图])`（~6.6ms TensorRT）
   - 模型自动加载优先级：.engine > .onnx > .pt（从 yolo_model/ 目录）
   - TensorRT 首次推理慢，需要 5 次预热
   - 每个相机返回 `list[BallDetection]`，需要左右各恰好 1 个检测才能三角定位

4. **双目三角定位** — `BallLocalizer.triangulate(左检测, 右检测)` → Ball3D（<1ms）
   - 流程：去畸变 → 三角测量 → 齐次坐标转 3D（单位 mm）
   - 使用 stereo_calib.json 中的 K/D/R_world/t_world

5. **轨迹追踪** — `Curve3Tracker.update(BallObservation(x,y,z,t))` → TrackerResult
   - 状态机：IDLE → TRACKING_S0 → IN_LANDING → TRACKING_S1 → DONE → 重置
   - S0（落地前）：拟合抛物线 → 预测落地 → COR 反弹 → 预测到达 ideal_hit_z 的位置
   - S1（落地后）：拟合反弹后曲线 → 在小车停留的 y 位置求 (x, z)

6. **输出** — VideoWriterThread（异步队列，MJPG 半分辨率）+ JSON 日志

### 性能指标（2026-02-27，硬件触发 30fps）
- 稳定 30fps，解码=4.8ms，YOLO=6.6ms，主线程总计 ~11ms/帧
- 延迟（曝光→结果）：平均 ~45ms

## 目录结构

```
src/                              # 生产代码包
  run_tracker.py                  # 主管线
  ball_grabber.py                 # 相机控制：SyncCapture、ImageGrabber、open_camera、Frame
  ball_detector.py                # YOLO 检测：BallDetector、BallDetection
  ball_localizer.py               # 双目定位：BallLocalizer、Ball3D
  car_localizer.py                # AprilTag 小车定位
  curve3.py                       # 轨迹追踪：Curve3Tracker、FittedCurve、PredictHitPos
  mvs_env.py                      # SDK 路径初始化
  demo.py / benchmark.py          # 演示和基准测试
  config/
    camera.json                   # 硬件配置（序列号、帧率、曝光、ROI）
    stereo_calib.json             # 标定参数（K/D/R/T，长度单位 mm）

test_src/                         # 测试与可视化工具
  test_curve3_replay.py           # 离线回放，支持参数调优
  generate_curve3_html.py         # Plotly.js 交互式三标签页 HTML
  test_ball_detector.py / test_car_localizer.py / capture_test.py

calibration/                      # 双目标定子项目（独立运行）
  stereo_calibrator.py            # 核心：blob 检测 + 双目标定 + 联合 PnP
  run_calibration.py              # 步骤 1-2：内参 + 双目外参
  register_ground.py              # 步骤 5：世界坐标注册（SQPNP）
  ground_annotator.py             # GUI 标注工具
  calibration_capture.py / ground_capture.py
  images/{序列号}/                # 标定图片

yolo_model/                       # YOLO 权重文件（.engine/.onnx/.pt）
tracker_output/                   # 管线输出（json + avi + html）
SDK_Development/                  # 海康 MVS SDK（内置）
mvs-dev/                          # 实验性高级封装（过渡状态）
```

## 核心模块接口

### ball_grabber.py — 相机控制
```python
Frame(data, width, height, pixel_type, exposure_start_pc, ...)  # 数据类
open_camera(serial, trigger_source=None, exposure_us, gain_db, roi_width, ...)
close_camera(cam)
frame_to_numpy(frame) → np.ndarray  # 自动 Bayer 去马赛克

ImageGrabber(cam)          # 后台抓取线程，自动校准设备→PC时钟偏移
  .get_frame(timeout_s=0) → Frame
  .stop()

SyncCapture(master_serial, slave_serials, fps, trigger_mode, ...)  # 上下文管理器
  .from_config(config_path="src/config/camera.json")  # 从配置文件创建
  .get_frames(timeout_s=1.0) → {序列号: Frame} | None
```

### ball_detector.py — YOLO 网球检测
```python
BallDetection(x, y, confidence, x1, y1, x2, y2)  # 像素坐标 + 置信度 + 边界框
BallDetector(model_path=None)  # 自动查找最优模型
  .detect(image) → list[BallDetection]
  .detect_batch(images) → list[list[BallDetection]]  # batch=2 双目同时推理
```

### ball_localizer.py — 双目三角定位
```python
Ball3D(x, y, z, pixel_1, pixel_2, confidence, reprojection_error)  # 世界坐标 mm
BallLocalizer(stereo_config_path=None, detector=None)
  .triangulate(det1, det2) → Ball3D | None       # 给定检测结果 → 3D 坐标
  .locate(img1, img2) → Ball3D | None             # 检测 + 三角定位一步完成
  .serial_left / .serial_right → str              # 左右相机序列号
```

### curve3.py — 轨迹追踪与击球预测
```python
BallObservation(x, y, z, t)                      # 单次 3D 观测（mm，秒）
PredictHitPos(x, y, z, stage, ct, ht)            # 预测击球位置
    # stage: 0=落地前预测, 1=落地后预测
    # ct: 给出预测时的当前时刻, ht: 预测击球时刻
TrackerState: IDLE | TRACKING_S0 | IN_LANDING | TRACKING_S1 | DONE
TrackerResult(prediction, state)                  # update() 返回值

Curve3Tracker(
    ideal_hit_z=800,        # 期望击球高度 (mm)
    cor=0.78,               # 法向恢复系数（z方向反弹，实测 ~0.787）
    cor_xy=0.42,            # 切向恢复系数（xy方向摩擦，实测 ~0.442）
    motion_window_s=0.2,    # 运动过滤：滑动窗口时长 (s)
    motion_min_y=500,       # 运动过滤：窗口内最小 |Δy| (mm)
    land_skip_time=0.05,    # 落地排除窗口 ±0.05s（避免地面帧污染拟合）
    reset_timeout=0.5,      # 帧间超时阈值 (s)
)
  .update(obs) → TrackerResult      # 核心接口：喂入观测，返回预测+状态
  .predictions → list[PredictHitPos]  # 全局累积（跨抛球不清除）
  .reset_times → list[float]          # 重置时间点列表

FittedCurve(ax, bx, ay, by, az, bz, cz, t_ref)  # 拟合抛物线参数
  .predict(t) → (x, y, z)            # 在时刻 t 的预测位置
  .velocity_at(t) → (vx, vy, vz)     # 在时刻 t 的速度
  .solve_t_for_z(target_z) → t | None  # 求解 z(t) = target_z
  .solve_t_for_y(target_y) → t | None  # 求解 y(t) = target_y
```

### Curve3 内部处理顺序（每次 update 调用）
1. **重置检查**：超时（>0.5s）或连续 3 帧速度跳变（>50 m/s）→ 重置
2. **跳帧检查**：单帧跳变 → 丢弃该帧，不重置（防止三角测量噪声打断追踪）
3. **运动过滤**：将观测缓冲到 pending，直到 0.2s 窗口内 |Δy| ≥ 500mm 确认飞行
4. **反弹检测**：z 值谷底模式（前高-低-后高，且低点 z < 200mm）
5. **曲线拟合**：S0 = 落地前观测，S1 = 落地后观测（排除落地时间 ±0.05s 区间）
6. **S0 预测**：拟合抛物线 → 求落地时间 → 应用 COR 反弹 → 求到达 ideal_hit_z 的位置
7. **S1 预测**：拟合反弹后曲线 → 求 y(t) = 小车 y 位置 → 返回该时刻的 (x, z)
8. **自动重置**：二次落地 / 超时 / 持续跳变

## 配置文件

### camera.json — 相机硬件配置
```json
{
  "master_serial": "DA8199303",
  "slave_serials": ["DA8199402", "DA8199285"],
  "fps": 30.0, "trigger_mode": "hardware",
  "exposure_us": 2340.0, "gain_db": 20.0,
  "roi_offset_y": 700, "roi_width": 2248,
  "master_min_bandwidth": true, "recalib_every": 10
}
```

### stereo_calib.json — 双目标定参数（关键字段）
- `serial_left`: "DA8199285"，`serial_right`: "DA8199402"
- `K1/K2`: 3x3 内参矩阵，`D1/D2`: 1x5 畸变系数
- `R1_world/t1_world`、`R2_world/t2_world`: 世界坐标注册（mm）
- 双目 RMS：0.303 px，基线 ~3741mm

## 标定工作流（calibration/）
1. `calibration_capture.py` — 采集标定板图片（6×6 AprilTag，间距 71.5mm）
2. `run_calibration.py` — 计算内参 + 双目外参 → stereo_calib.json
3. `ground_capture.py` — 采集地面参考图片
4. `ground_annotator.py` — 手动标注世界坐标（GUI 工具）
5. `register_ground.py` — 世界坐标注册（SQPNP 算法，共面点安全）

## 关键设计模式

- **硬件触发同步**：主相机自由运行 + ExposureStartActive 信号 → 从相机下降沿触发
- **时钟校准**：基于 perf_counter（单调时钟），ImageGrabber 每 ~0.5s 重新校准设备→PC 偏移
- **线程安全**：有界双端队列、锁保护队列、Event 停止信号
- **非阻塞视频**：异步 VideoWriterThread（队列 maxsize=30，满时丢弃最旧帧）
- **模型自动选择**：BallDetector 优先加载 .engine > .onnx > .pt
- **像素格式**：海康 BayerRG↔BG 命名反转已在 frame_to_numpy() 中处理

## 语言约定

文档和注释使用中文，代码标识符使用英文。
