# ArmCalibration

机械臂校准子项目。第 15 步已经完成，当前正式流程固定为 `15.2 -> 15.3 -> 15.4/15.5 -> 15.6`，不再使用早期的棋盘/旧 POE 原型。

## 当前结论

- 第 15 步完成时间：`2026-03-22`
- 正式数据会话：`ArmCalibration/data/004_15_2_formal_capture_domain2_03192121`
- 15.3 正式筛选规则：只看球拍关键点 `0-3`
- 15.4/15.5 POE：以 2D 重投影最小化为目标，不依赖单独构造 3D 标定点
- 15.6 实时输出：同时对比 `p_racket_rel_base_in_world(by poe)` 与 `p_racket_rel_base_in_world(by vision)`
- 人工复核结论：整体误差在几厘米量级，`z` 方向达到厘米级

## 目录说明

- `capture_poe_dataset.py`
  15.2 采集脚本。按给定时长/张数同步抓取 4 相机彩色图像，并记录拍摄时最新 `/joint_states`
- `detect_racket_keypoints.py`
  15.3 离线球拍关键点识别。读取会话目录，写回 `sample.json`，并保存标注图
- `calibrate_poe_reprojection.py`
  15.4/15.5 POE 标定脚本。用所有合格观测直接最小化 2D 重投影误差，并导出新的 POE config
- `monitor_live_state.py`
  15.6 实时脚本。以 1Hz 输出 RK 关节、POE 正解、视觉 `p_car`、视觉 `p_racket` 与两种 `p_racket_rel_base_in_world`
- `run_armcalib_ros2.ps1`
  Windows 统一启动脚本。自动优先选择支持 CUDA/TensorRT 的 `.venv_ros2`
- `src/poe_fk_reference.py`
  独立 FK 参考代码。只依赖 `arm_poe_racket_center.json` 和 `numpy`，供 RK 端或其他 agent 理解 `p_racket_rel_base_in_world(by poe)` 的计算方式
- `common.py`
  路径、JSON、会话目录等公共函数
- `data/004_15_2_formal_capture_domain2_03192121`
  当前正式 POE 标定使用的数据
- `data/live_racket_debug*`
  15.6 现场排查时保存的临时调试图片，可作为人工检查参考

## 正式流程

### 15.2 采集

作用：

- 同步抓 4 相机彩色图像
- 记录拍摄时 PC 最新 `/joint_states`
- 不做棋盘检测，不做球拍识别门控

正式采集命令：

```powershell
$env:ROS_DOMAIN_ID='2'
powershell -ExecutionPolicy Bypass -File .\ArmCalibration\run_armcalib_ros2.ps1 capture_poe_dataset.py --count 400 --duration 120
```

说明：

- 默认彩色采集：`--pixel-format BayerRG8`
- 关节时间戳不做 PC<->RK 精确时钟同步，直接使用 PC 收图时最新的 `JointState`
- 每个样本目录保存 4 张图和一个 `sample.json`
- 会话根目录保存 `session.json`

### 15.3 球拍关键点识别

作用：

- 模型：`yolo_model/racket.onnx` + `yolo_model/racket_pose.onnx`
- 几何中心：只使用关键点 `0-3`
- 关键点 `4` 只保留作调试参考，不参与中心计算

正式识别命令：

```powershell
powershell -ExecutionPolicy Bypass -File .\ArmCalibration\run_armcalib_ros2.ps1 detect_racket_keypoints.py --session ArmCalibration\data\004_15_2_formal_capture_domain2_03192121
```

正式筛选规则：

- 只看 `0-3` 号关键点分数
- `0-3` 号点都必须 `>= 40`
- 任意一个点失败，该图像直接淘汰
- 不依赖 `0-3` 的几何关系
- 至少 2 台相机合格时，样本才可用于视觉三角化

输出：

- 每个 `sample_xxxx/racket_pose/` 下保存标注图
- 结果写回同目录 `sample.json`
- 会话根目录导出 `racket_pose_summary.json`
- 会话根目录导出平铺人工检查目录 `racket_pose_accepted_flat/`

### 15.4 / 15.5 POE 标定

作用：

- 用所有合格观测最小化 2D 重投影误差
- 先做三角化与结构化初始化，再做直接 2D 优化
- 15.5 把 `base` 重新定义为 `joint_1` 轴与世界地面 `z=0` 的交点

命令：

```powershell
powershell -ExecutionPolicy Bypass -File .\ArmCalibration\run_armcalib_ros2.ps1 calibrate_poe_reprojection.py --session ArmCalibration\data\004_15_2_formal_capture_domain2_03192121
```

当前正式结果：

- 有效样本：`399`
- 合格图片：`1153`
- 样本中至少 2 台相机合格：`293`
- 全局重投影：`mean 6.189 px`
- 中位数：`4.145 px`
- `P95`：`11.764 px`

导出文件：

- 结果 JSON：`ArmCalibration/data/004_15_2_formal_capture_domain2_03192121/poe_reprojection_result.json`
- 运行配置：`src/config/arm_poe_racket_center.json`
- 配置说明：`src/config/arm_poe_racket_center.md`

### 15.6 实时监视

作用：

- 持续接收 RK 发布的 `/joint_states`
- 用当前 POE config 做正解
- AprilTag 多目三角化得到 `p_apriltag`
- 通过固定偏移换算 `p_car`
- 球拍多目三角化得到 `p_racket_world(by vision)`
- 重点输出两条最终对比量

核心约定：

- `p_car = p_apriltag + (60, 100, -340) mm`
- `car chassis center = robotic arm base`
- `p_racket_rel_base_in_world(by vision) = p_racket_world(by vision) - p_car`
- `p_racket_rel_base_in_world(by poe) = p_racket_world(by poe) - T_base_in_world.t`

持续运行：

```powershell
$env:ROS_DOMAIN_ID='2'
powershell -ExecutionPolicy Bypass -File .\ArmCalibration\run_armcalib_ros2.ps1 monitor_live_state.py --rate-hz 1 --no-ball
```

只看一轮：

```powershell
$env:ROS_DOMAIN_ID='2'
powershell -ExecutionPolicy Bypass -File .\ArmCalibration\run_armcalib_ros2.ps1 monitor_live_state.py --rate-hz 1 --warmup 0.5 --max-prints 1 --no-ball
```

终端会重点高亮：

```text
KEY  p_racket_rel_base_in_world(by poe):    (...)
KEY  p_racket_rel_base_in_world(by vision): (...)
```

其它 `joint_state / p_apriltag / p_car / racket_obs / p_ball` 都保留为 `detail:` 行，供排查时查看。

## 当前正式数据

- 正式会话：`004_15_2_formal_capture_domain2_03192121`
- 15.3 合格标注平铺目录：`racket_pose_accepted_flat/`
- 当前最新实时球拍调试图片：`ArmCalibration/data/live_racket_debug_20260322_202028`

## 环境说明

- `run_armcalib_ros2.ps1` 会自动优先选择 `.venv_ros2`
- 当前机器上 `.venv_ros2` 提供 CUDA/TensorRT，适合球拍 bbox 和关键点识别
- 需要实时接收 RK 关节时，记得设置：

```powershell
$env:ROS_DOMAIN_ID='2'
```
