# ROS2 Topics (PC 侧)

本文档只描述当前主线下，PC 侧实际会发布的 ROS2 topic。

范围说明：
- 主入口：`.\run_tracker.ps1`
- 对应实现：`src/run_tracker.py`
- 当前 ROS2 输出模式：`auto | direct | off`，默认 `direct`
- 统计口径：只算当前主线真正会发布的 topic，不把历史桥接脚本当成主线接口

## 总览

当前 PC 侧发布 2 个 topic：

| Topic | ROS 类型 | 实际载荷 | QoS | 说明 |
| --- | --- | --- | --- | --- |
| `/pc_car_loc` | `std_msgs/msg/String` | JSON 字符串 | `BEST_EFFORT`, depth=1 | 小车定位结果 |
| `/racket_vz` | `std_msgs/msg/String` | JSON 字符串 | `RELIABLE`, depth=4 | 球员拍头竖直速度，每抛一条（见 §2） |

说明：
- `/predict_hit_pos` 现由 **RK 车载 bot_center 发布**（PC 只在 rosbag 里记录它）；
  `DirectRos2Sink.publish_predict_hit` 为 no-op，`ROS2_RELIABLE_TOPICS` 中保留其
  QoS 约定仅供桥接脚本/工具使用。
- 旧主线的 `/arm_logger/control`（pc_event_logger 控制）与 `/time_sync/pong`
  （WinRKTimeSync 应答）随 newarm2 线于 2026-07-16 废弃删除。两轴对时不再有
  运行时组件：RK 全站自带 CLOCK_MONOTONIC 时间，报告端每场拟合
  `PC t = scale × RK t + bias`。

## 通用约定

- 业务 topic 统一使用 `std_msgs/msg/String`
- 真正的数据放在 `String.data` 中
- `String.data` 的内容是 JSON 字符串
- 当前主线下，坐标相关字段建议统一按“米”理解

## 1. `/pc_car_loc`

### 说明

PC 侧 AprilTag 多目定位成功后发布的小车位置结果。

### ROS 类型

`std_msgs/msg/String`

### QoS

- Reliability: `BEST_EFFORT`
- Depth: `1`

### 发布时机

- **两块车载 tag（id0、id1）都参与联合拟合成功**：发布完整位姿，`yaw` 为数值
- **只剩一块 tag 可见**（2026-08-11 起）：仍然发布位置，但 `yaw` 为 **`null`**、
  `yaw_valid` 为 `false`。位置由该 tag 中心的多视图三角化 + **冻结的**最近一次
  可信 yaw 反解；冻结 yaw 超过 0.5s 未刷新则整帧不发
- 一块 tag 都没被 ≥2 台相机看到时不发

### ⚠ 消费端契约（`yaw == null`）

收到 `yaw: null` 时**必须保持自身当前 yaw 不变**，只用 `x`/`y` 修正位置。
不要把 `null` 当 0、不要沿用上一帧的 `yaw` 数值再当成新观测喂进滤波器
（那等于把同一个历史值反复吸收，会让航向估计过度自信）。

背景：车上 id1 贴在左前立柱上，会被臂座平台自遮挡，四台相机里常年只有两台
看得见；而此前"两块 tag 都在才发布"的与门让击球瞬间成片丢定位（0811 053055 场
实测 9 个 >0.5s 的空洞、共 11.3s，18 抛里 4 抛的 PC 真值因此为空）。同批 miss
帧里 id0 在 ≥2 台相机的检出率是 15/15，所以退化路径能把绝大多数空洞填上。
位置侧代价：0.5s 陈旧 yaw ≈ 1.5°，经 id0 的 0.42m 安装杠杆 ≈ 11mm。

### JSON 格式

```json
{
  "topic": "car_loc",
  "x": 0.1234,
  "y": 1.2345,
  "z": 0.0000,
  "yaw": 0.4567,
  "yaw_valid": true,
  "t": 411987136.914000,
  "tag_id": 0,
  "tag_ids": [0, 1]
}
```

### 字段说明

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `topic` | `string` | 历史桥接链路保留下来的路由字段，当前直连模式也仍然保留，固定为 `car_loc` |
| `x` | `number` | 小车 `car_base` 参考点世界坐标 X，单位米 |
| `y` | `number` | 小车 `car_base` 参考点世界坐标 Y，单位米 |
| `z` | `number` | 小车 `car_base` 参考点世界坐标 Z，单位米（车心定义在地面，恒为 0） |
| `yaw` | `number \| null` | 本帧估计的小车绕 z 轴朝向，单位弧度。**`null` = 本帧给不出 yaw（单 tag 退化），消费端保持自身 yaw 不更新** |
| `yaw_valid` | `boolean` | 双 tag 参与拟合（~0.9m 中心基线）或单 tag 但至少 3 台相机、且四角重投影误差合格时为 `true`，RK 才使用 `yaw` 修正航向；否则本帧只修正 `x/y`。`yaw` 为 `null` 时恒为 `false` |
| `t` | `number` | 定位时间，时间轴是 Windows `perf_counter()` 秒 |
| `tag_id` | `integer` | 主 tag（拟合中相机数最多的车载 tag；并列取小 id），兼容保留 |
| `tag_ids` | `integer[]` | 参与本次联合拟合的全部车载 tag ID。`[0, 1]` = 双 tag 完整解；长度为 1 = 单 tag 退化解（此时 `yaw` 必为 `null`） |

### 备注

- 坐标参考点不是 tag 中心，而是 `car_base`（车位姿为直接优化变量）
- 车上现装两块 tag（id0 右后、id1 左前），车体系布局见
  `src/config/arm_poe_racket_center.json` 的 `vehicle_reference.apriltags`，
  由 `test_src/measure_car_tag_layout.py` 实测生成；单块可见时退化为单 tag 拟合
- 单 tag 退化帧的条数记在 session json 的 `summary.car_loc_single_tag_frames`。
  这个数长期偏高 = 有一块 tag 被长期遮挡，该挪安装位置或补第三块，而不是靠退化
  路径长期兜着（退化解不带 yaw 信息，车航向会只能靠 IMU 递推）

## 2. `/racket_vz`

### 说明

球员拍头竖直速度，**每抛一条、只发高置信**，发给 RK bot_center：其 stage0 首次算反弹
时**认领并整抛锁存**（认领新鲜度 3s），生效期间用 vz 版切向恢复系数
`cor_xy_eff = 在线锚点 + racket_vz_slope·(vz−nominal)` 取代 aMz 旋转前馈
（0813_083521 混合打法场实测：aMz 对实测 cor_xy 的相关 r=+0.001，拍头 vz r=+0.772；
11 场语料留一场交叉验证留出中位 +0.78）。

### 发布时机

`DirectRos2Sink` 订阅 RK 的 `/ball_world_topic`，`ContactSolver` 沿弹前抛物线反解出
「球员触球时刻」（回球段第 6 个点即触发，约触球后 0.3s）→ 立即回看挥拍环形缓冲，
在触球 +25ms ± 120ms 窗内拟合拍头高度斜率 → 取共识对（`consensus_serials`，当前
DB0260405 / DB0260373）均值发布。同一抛 1.2s 内只发第一条（ContactSolver 断段
重点火去重）。到达 RK 时球尚未落地（触地在触球后 ~1.2-1.5s），S0 中后段起生效。

### JSON 格式

```json
{
  "vz": 1.234,
  "n_cams": 2,
  "dv": 0.31,
  "trusted": true,
  "contact_elapsed_s": 82.0132,
  "t": 411987136.914
}
```

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `vz` | `number` | 拍头竖直速度，m/s，向上为正；共识对均值 |
| `n_cams` | `integer` | 参与均值的相机数（1 或 2） |
| `dv` | `number \| null` | 两相机读数差的绝对值；单相机时为 `null` |
| `trusted` | `boolean` | 恒为 `true`——**共识不足（单相机 / `dv > consensus_dv_max`(0.6)）整条不发**（0813 用户定：RK 端一抛只认领一条并整抛锁存，发一条可疑的比不发更糟）。字段保留作协议自描述，RK 端仍校验。被压掉的抛在 session json 的对应 entry 里记 `publish_suppressed` |
| `contact_elapsed_s` | `number` | 反解出的触球时刻（PC elapsed 轴），离线对账用 |
| `t` | `number` | 发布时刻（PC `perf_counter()` 秒） |

配置在 `src/config/tracker.json` 的 `racket_swing` 段（`consensus_serials` /
`consensus_dv_max` / `read_delay_s=-0.025` / `read_half_s=0.12`）；四台相机的完整读数
仍逐抛记录在 session json 的 `racket_swing` 列表里（含 `published` 字段 = 发出的载荷）。

## PC 侧的消费（订阅）

PC 主进程（`DirectRos2Sink`）订阅 3 条 RK topic，全部只服务内部功能、不再转发：
`/bot_state`（时钟桥采样）、`/ball_world_topic`（挥拍触球锚点反解）、
`/predict_hit_pos`（记录首条 stage0 时刻做锚点对照）。其余整个局域网的 topic
（RK 球轨迹/预测、底盘、`/joint_states`、`/tennis/*` 等）由独立的 **rosbag 录制进程**
（`src/rosbag_recorder.py`，tracker 启动时自动拉起）全量录制到
`tracker_output/{run_id}/{run_id}_rosbag/`，事后经 `test_src/extract_arm_bag.py` /
`test_src/extract_rk_tracking_bag.py` 提取供报告使用。

## 历史桥接脚本说明

仓库中还保留了以下历史桥接脚本：

- `ros2/car_loc_bridge.py`
- `ros2/predict_hit_bridge.py`

它们用于早期 UDP -> ROS2 topic 转发。当前 `src/run_tracker.py` 主线里 `auto` 和
`direct` 都走进程内直接发布，不再走 bridge fallback。

## 代码位置

- `src/run_tracker.py`
- `src/ros2_support.py`
- `src/rosbag_recorder.py`
- `src/car_localizer.py`
- `src/curve3.py`

## Time Axis（两轴制，2026-07-16 定案）

全项目只有两个时间轴：

- **PC 轴**：Windows `perf_counter()` 秒。`/pc_car_loc.t`、tracker JSON 里的曝光/
  观测时间都在这条轴上。
- **RK 轴**：RK 的 CLOCK_MONOTONIC 秒。RK 上所有 topic 自带该钟（chassis payload
  `t`、bot_center 的 `ct/ht`、`/joint_states` 与 `/tennis/motor_command` 的
  header.stamp、`/tennis/status` 文本尾缀 `t=`），不用系统钟/epoch。

两轴仅在报告端对齐：`generate_curve3_html.py` 先用 PC 发布且 RK 原样回带的小车
位姿鲁棒估计时钟 `scale`，再用逐抛球 z 形状细化 `bias`。运行时没有任何跨钟
换算或对时组件。
