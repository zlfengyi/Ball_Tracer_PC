# ROS2 Topics (PC 侧)

本文档只描述当前主线下，PC 侧实际会发布的 ROS2 topic。

范围说明：
- 主入口：`.\run_tracker.ps1`
- 对应实现：`src/run_tracker.py`
- 当前 ROS2 输出模式：`auto | direct | off`，默认 `direct`
- 统计口径：只算当前主线真正会发布的 topic，不把历史桥接脚本当成主线接口

## 总览

当前 PC 侧只发布 1 个 topic：

| Topic | ROS 类型 | 实际载荷 | QoS | 说明 |
| --- | --- | --- | --- | --- |
| `/pc_car_loc` | `std_msgs/msg/String` | JSON 字符串 | `BEST_EFFORT`, depth=1 | 小车定位结果 |

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

- **两块车载 tag（id0、id1）都参与联合拟合成功时才发布**
- 只识别到一块 tag 时不发布（单 tag 短基线 yaw + 安装杠杆位置误差，精度不足）
- 没有定位结果时不发

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
| `yaw` | `number` | 本帧估计的小车绕 z 轴朝向，单位弧度 |
| `yaw_valid` | `boolean` | 双 tag 参与拟合（~0.9m 中心基线）或单 tag 但至少 3 台相机、且四角重投影误差合格时为 `true`，RK 才使用 `yaw` 修正航向；否则本帧只修正 `x/y` |
| `t` | `number` | 定位时间，时间轴是 Windows `perf_counter()` 秒 |
| `tag_id` | `integer` | 主 tag（拟合中相机数最多的车载 tag；并列取小 id），兼容保留 |
| `tag_ids` | `integer[]` | 参与本次联合拟合的全部车载 tag ID；因单 tag 不发布，实际恒为 `[0, 1]` |

### 备注

- 坐标参考点不是 tag 中心，而是 `car_base`（车位姿为直接优化变量）
- 车上现装两块 tag（id0 右后、id1 左前），车体系布局见
  `src/config/arm_poe_racket_center.json` 的 `vehicle_reference.apriltags`，
  由 `test_src/measure_car_tag_layout.py` 实测生成；单块可见时退化为单 tag 拟合

## PC 侧的消费（订阅）

PC 主进程不再订阅任何 topic。整个局域网的 topic（RK 球轨迹/预测、底盘、
`/joint_states`、`/tennis/*` 等）由独立的 **rosbag 录制进程**（`src/rosbag_recorder.py`，
tracker 启动时自动拉起）全量录制到 `tracker_output/{run_id}_rosbag/`，事后经
`test_src/extract_arm_bag.py` / `test_src/extract_rk_tracking_bag.py` 提取供报告使用。

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
