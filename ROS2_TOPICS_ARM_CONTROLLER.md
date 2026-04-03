# ROS2 Topics From arm_controller

本文档只描述当前 `newarm2` 主线下，`arm_controller` runtime 会发布的 ROS2 topic。

范围说明：
- 主节点：`src/newarm2/newarm2/arm_controller.py`
- bundled 时间同步组件：`src/newarm2/newarm2/win_rk_time_sync.py`
- 这里只写“发布出去的 topic”，不写订阅 topic

## 总览

### arm_controller 节点直接发布

| Topic | ROS 类型 | 实际载荷 | QoS | 默认值 |
| --- | --- | --- | --- | --- |
| `/arm_controller/status` | `std_msgs/msg/String` | 纯文本字符串 | `RELIABLE`, depth=10 | 可配置 |
| `/joint_states` | `sensor_msgs/msg/JointState` | ROS 标准 JointState | `BEST_EFFORT`, depth=10 | 可配置 |
| `/arm_logger/hit_event` | `std_msgs/msg/String` | JSON 字符串 | `BEST_EFFORT`, depth=100 | 固定 |
| `/arm_logger/mit_command` | `std_msgs/msg/String` | JSON 字符串 | `BEST_EFFORT`, depth=100 | 固定 |

### bundled time sync worker 发布

仅当 `time_sync.enabled = true` 且 `time_sync.spawn_process = true` 时存在。

| Topic | ROS 类型 | 实际载荷 | QoS | 默认值 |
| --- | --- | --- | --- | --- |
| `/time_sync/ping` | `std_msgs/msg/String` | JSON 字符串 | `BEST_EFFORT`, depth=100 | 可配置 |
| `/time_sync/offset` | `std_msgs/msg/String` | JSON 字符串 | `BEST_EFFORT`, depth=100 | 可配置 |

## 1. `/arm_controller/status`

### 说明

`arm_controller` 的人类可读状态日志。

### ROS 类型

`std_msgs/msg/String`

### 载荷

`String.data` 直接是文本，不是 JSON。

### 何时发布

- 节点启动时
- 直接命令被接受或拒绝时
- 执行完成时
- 执行失败时
- IO 清理告警时
- 某些需要显式通知用户的 time sync 等待场景

## 2. `/joint_states`

### 说明

机械臂当前关节状态。

### ROS 类型

`sensor_msgs/msg/JointState`

### QoS

- Reliability: `BEST_EFFORT`
- Depth: `10`

### 何时发布

- 按 `joint_state_publish_rate_hz` 定时发布
- 前提是已有完整电机反馈，且 time sync 已准备好

### 关键字段

- `header.stamp`: 映射到 PC 时间轴的时间戳
- `name`: 当前模型关节名加 `joint_5`
- `position`
- `velocity`
- `effort`

说明：
- `header.stamp` 不是 Unix epoch；它编码的是通过 `/time_sync/offset` 映射后的 PC 时间

## 3. `/arm_logger/hit_event`

### 说明

击打状态机和执行链的关键事件。

### ROS 类型

`std_msgs/msg/String`

### 载荷

`String.data` 是 JSON object。

### 所有事件共享字段

- `event`: 事件名
- `request_id`: 请求 ID
- `source`: 触发来源，例如 `direct command` 或 `/predict_hit_pos stage=1`
- `hit_x_m`
- `hit_z_m`
- `duration_sec`
- `scheduled_hit_time_pc_ns`
- `stamp_pc_ns`

### 当前会发布的事件

| event | 触发时机 |
| --- | --- |
| `predict_hit_pending_armed` | 第一次收到可接受的 `stage=1` `/predict_hit_pos`，并把 controller 从 idle 推到 pending |
| `execution_start` | 真正开始执行主击打 |
| `execution_complete` | 主击打执行完成 |
| `ready_execution_start` | 击打后的 ready pose 开始执行 |
| `ready_execution_complete` | ready pose 执行完成 |
| `command_rejected` | 直接命令或执行入口因非法目标等原因被拒绝 |
| `command_failed` | 已进入执行链路后发生失败 |

### 事件额外字段

`extra` 字段不是统一 schema，按事件不同会附加不同内容。当前主线会附加的内容包括：

- `predict_hit_pending_armed`:
  `stage`, `world_x_m`, `world_y_m`, `world_z_m`, `ct_sec`
- `execution_start`:
  `mode`, `joint1_relative_hit_rad`
- `execution_complete`:
  `mode`, `controller_timing_sec`, `send_count`, `planning_timing_sec`, `execution_timing_sec`
- `ready_execution_start`:
  `mode`, `joint1_relative_hit_rad`
- `ready_execution_complete`:
  `mode`, `send_count`, `planning_timing_sec`, `execution_timing_sec`
- `command_rejected`:
  `detail`, `mode`, `joint1_relative_hit_rad`
- `command_failed`:
  `mode`, `error`

说明：
- `command_failed.extra.mode` 当前可能是 `hit`、`regular`、`ready`，也可能是 20.2 prepare 阶段失败时的 `predict_prepare`

## 4. `/arm_logger/mit_command`

### 说明

每一帧 MIT 下发命令的日志 topic。

### ROS 类型

`std_msgs/msg/String`

### 载荷

`String.data` 是 JSON object。

### 顶层字段

- `stamp_pc_ns`
- `request_id`
- `sequence`
- `profile_mode`
- `execution_rel_sec`
- `send_index`
- `is_final`
- `commands`

### `commands[*]` 字段

- `motor_id`
- `joint_name`
- `position_rad`
- `velocity_rad_s`
- `torque_ff_nm`
- `computed_torque_ff_nm`
- `kp`
- `kd`
- `is_hold`

说明：
- `sequence` 当前可能是 `predict_prepare`、`hit`、`regular` 或 `ready`
- `profile_mode` 当前可能是 `predict_hit_prepare`、`hit_semi_analytic` 或 `regular`
- 在 20.2 的 `predict_prepare` 阶段，活动控制关节是 2、3 号；1、4、5 号会以 hold 命令一并写进同一帧日志

## 5. `/time_sync/ping`

### 说明

由 bundled `WinRKTimeSync` worker 周期发布，用来向 Windows 侧请求时间同步应答。

### ROS 类型

`std_msgs/msg/String`

### 载荷

`String.data` 是 JSON object，当前字段为：

- `seq`
- `t1`
- `source_id`
- `tags`

### 时间语义

- `t1` 是 RK 侧 `time.perf_counter()` 秒

## 6. `/time_sync/offset`

### 说明

由 bundled `WinRKTimeSync` worker 发布的对时结果与统计摘要。

### ROS 类型

`std_msgs/msg/String`

### 载荷

`String.data` 是 JSON object。

### 当前稳定字段

- `tag`
- `tags`
- `source_id`
- `publish_reason`
- `report_period_sec`
- `generated_local_perf_sec`
- `current_offset_sec`
- `latest_accepted_offset_sec`
- `latest_offset_median_sec`
- `latest_rtt_sec`
- `latest_seq`
- `offset_window_count`
- `period_sample_count`
- `period_accepted_count`
- `period_rejected_count`
- `total_sample_count`
- `total_accepted_count`
- `total_rejected_count`
- `rtt_mean_sec`
- `rtt_median_sec`
- `rtt_p95_sec`
- `rtt_p99_sec`
- `rtt_max_sec`
- `generated_pc_sec`
- `stamp_pc_ns`
- `clock_domain`

### 时间语义

- `current_offset_sec` 的意义是：
  `pc_perf_counter_sec - rk_local_perf_counter_sec`
- `stamp_pc_ns` 与 `generated_pc_sec` 位于 PC `perf_counter()` 时间轴
- `clock_domain` 当前固定为 `pc`

## 代码位置

- `src/newarm2/newarm2/arm_controller.py`
- `src/newarm2/newarm2/arm_logger_protocol.py`
- `src/newarm2/newarm2/win_rk_time_sync.py`
- `src/newarm2/newarm2/time_sync.py`
- `src/newarm2/config/arm_controller.yaml`
