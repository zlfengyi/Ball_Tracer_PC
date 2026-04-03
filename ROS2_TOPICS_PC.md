# ROS2 Topics

本文档只描述当前主线下，PC 侧实际会发布的 ROS2 topic。

范围说明：
- 主入口：`.\run_tracker.ps1`
- 对应实现：`src/run_tracker.py`
- 当前 ROS2 输出模式：`auto | direct | off`，默认 `direct`
- 统计口径：只算当前主线真正会发布的 topic，不把历史桥接脚本当成主线接口

## 总览

当前 PC 侧会发布 4 个 topic：

| Topic | ROS 类型 | 实际载荷 | QoS | 说明 |
| --- | --- | --- | --- | --- |
| `/pc_car_loc` | `std_msgs/msg/String` | JSON 字符串 | `BEST_EFFORT`, depth=1 | 小车定位结果 |
| `/predict_hit_pos` | `std_msgs/msg/String` | JSON 字符串 | `RELIABLE`, depth=1 | 击球点预测 |
| `/arm_logger/control` | `std_msgs/msg/String` | JSON 字符串 | `RELIABLE`, depth=20 | pc logger 控制指令 |
| `/time_sync/pong` | `std_msgs/msg/String` | JSON 字符串 | `BEST_EFFORT`, depth=1 | 时间同步应答 |

## 通用约定

- 业务 topic 统一使用 `std_msgs/msg/String`
- 真正的数据放在 `String.data` 中
- `String.data` 的内容是 JSON 字符串
- 当前主线下，坐标相关字段建议统一按“米”理解
- 时间字段不是完全统一的时间轴，消费端必须按字段语义分别处理

## 1. `/pc_car_loc`

### 说明

PC 侧 AprilTag 多目定位成功后发布的小车位置结果。

### ROS 类型

`std_msgs/msg/String`

### QoS

- Reliability: `BEST_EFFORT`
- Depth: `1`

### 发布时机

- 小车定位成功时发布
- 没有定位结果时不发

### JSON 格式

```json
{
  "topic": "car_loc",
  "x": 0.1234,
  "y": 1.2345,
  "z": 0.0000,
  "yaw": 0.4567,
  "t": 411987136.914000,
  "tag_id": 5
}
```

### 字段说明

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `topic` | `string` | 历史桥接链路保留下来的路由字段，当前直连模式也仍然保留，固定为 `car_loc` |
| `x` | `number` | 小车 `car_base` 参考点世界坐标 X，单位米 |
| `y` | `number` | 小车 `car_base` 参考点世界坐标 Y，单位米 |
| `z` | `number` | 小车 `car_base` 参考点世界坐标 Z，单位米 |
| `yaw` | `number` | 小车绕 z 轴朝向，单位弧度 |
| `t` | `number` | 定位时间，时间轴是 Windows `perf_counter()` 秒 |
| `tag_id` | `integer` | 本次定位使用到的 AprilTag ID |

### 备注

- 坐标参考点不是 tag 中心，而是 `car_base`
- `car_base` 相对 AprilTag 中心的偏移来自车辆参考配置

## 2. `/predict_hit_pos`

### 说明

PC 侧轨迹拟合器给出的击球点预测。

### ROS 类型

`std_msgs/msg/String`

### QoS

- Reliability: `RELIABLE`
- Depth: `1`

### 发布时机

- 轨迹器产生预测时发布
- 每次预测都会重新发一条最新结果

### JSON 格式

```json
{
  "x": 0.4123,
  "y": 2.1056,
  "z": 0.8000,
  "stage": 0,
  "ct": 411987136.914000,
  "ht": 411987137.209500,
  "duration": 0.2955
}
```

### 字段说明

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `x` | `number` | 预测击球点 X，单位米 |
| `y` | `number` | 预测击球点 Y，单位米 |
| `z` | `number` | 预测击球点 Z，单位米 |
| `stage` | `integer` | 预测阶段，`0` 表示落地前预测，`1` 表示落地后预测 |
| `ct` | `number` | 当前给出预测的时间，时间轴为 Windows `perf_counter()` 秒 |
| `ht` | `number` | 预计击球时刻，时间轴与 `ct` 一致 |
| `duration` | `number` | 从当前预测时刻到预计击球时刻的剩余时间，等于 `ht - ct`，单位秒 |

### stage 含义

#### `stage = 0`

- 球还处于落地前阶段
- 先拟合空中轨迹，再推算落地与反弹
- 输出“反弹后下降到 `ideal_hit_z` 时”的击球点
- 此时 `z` 通常就是配置中的 `ideal_hit_z`

#### `stage = 1`

- 球已经落地并进入反弹后阶段
- 小车此时默认已经按最后一次 stage 0 的结果把 `y` 基本锁定
- 重新估计“球到达该 `y` 位置时”的 `x/z`

## 3. `/arm_logger/control`

### 说明

发给 `pc_event_logger` 的控制指令，用来切文件、立即保存、结束落盘。

### ROS 类型

`std_msgs/msg/String`

### QoS

- Reliability: `RELIABLE`
- Depth: `20`

### 发布时机

- `pc_logger` 启用时，tracker 启动和停止阶段会发

### JSON 格式

```json
{
  "schema": "pc_logger_control_v1",
  "command": "new_file",
  "command_id": "tracker_20260403_xxx-new-file",
  "source": "tracker",
  "stamp_pc_ns": 123456789012345,
  "reason": "tracker_start",
  "run_id": "tracker_20260403_xxx",
  "group_id": "tracker_20260403_xxx",
  "target_path": "C:\\path\\to\\tracker_20260403_xxx_pc_logger.json",
  "tracker_output_dir": "C:\\path\\to\\tracker_output",
  "tracker_json_path": "C:\\path\\to\\tracker_20260403_xxx.json",
  "tracker_video_path": "C:\\path\\to\\tracker_20260403_xxx.mp4"
}
```

### 字段说明

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `schema` | `string` | 固定为 `pc_logger_control_v1` |
| `command` | `string` | 控制命令 |
| `command_id` | `string` | 命令唯一 ID，供 logger 去重 |
| `source` | `string` | 指令来源，当前主线默认是 `tracker` |
| `stamp_pc_ns` | `integer` | PC 侧打点时间，`perf_counter_ns()` 时间轴 |
| `reason` | `string` | 指令原因 |
| `run_id` | `string` | 当前 tracker 运行 ID |
| `group_id` | `string` | 当前分组 ID |
| `target_path` | `string` | pc logger 目标文件路径 |
| `tracker_output_dir` | `string` | tracker 输出目录 |
| `tracker_json_path` | `string` | tracker 主 JSON 日志路径 |
| `tracker_video_path` | `string` | tracker 原始视频路径；无视频时可为空 |

### 当前主线会发的 command

| command | 发送时机 | 含义 |
| --- | --- | --- |
| `new_file` | tracker 启动时 | 通知 logger 新开一个输出文件 |
| `save_now` | tracker 结束时 | 立即把当前缓存内容落盘 |
| `shutdown` | tracker 结束时 | 落盘并请求 logger 退出 |

## 4. `/time_sync/pong`

### 说明

Windows 侧收到 RK 发来的 `/time_sync/ping` 后，立即回发时间同步应答。

### ROS 类型

`std_msgs/msg/String`

### QoS

- Reliability: `BEST_EFFORT`
- Depth: `1`

### 发布时机

- 每收到一条 `/time_sync/ping` 就立即回一条 `/time_sync/pong`

### JSON 格式

```json
{
  "seq": 12,
  "t1": 1234.567890,
  "t2": 4567.890123
}
```

### 字段说明

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `seq` | `integer` | ping 的序号，原样回显 |
| `t1` | `number` | RK 侧发 ping 时的本地计时，原样回显 |
| `t2` | `number` | Windows 侧收到 ping 时的 `perf_counter()` |

## 当前只订阅、不发布的相关 topic

下面这些 topic 当前 PC 主线会消费，但不会在 PC 侧主动发布：

| Topic | 方向 | 用途 |
| --- | --- | --- |
| `/joint_states` | 订阅 | 机械臂关节状态，供 pc logger 记录 |
| `/arm_logger/mit_command` | 订阅 | MIT 控制命令，供 pc logger 记录 |
| `/arm_logger/hit_event` | 订阅 | 击球事件，供 pc logger 记录 |
| `/time_sync/ping` | 订阅 | 时间同步请求 |

## 历史桥接脚本说明

仓库中还保留了以下历史桥接脚本：

- `ros2/car_loc_bridge.py`
- `ros2/predict_hit_bridge.py`

它们用于早期 UDP -> ROS2 topic 转发。

但当前 `src/run_tracker.py` 的主线实现里：

- `auto` 和 `direct` 都走进程内直接发布
- 不再走 bridge fallback

所以“当前主线 topic 接口”应以上文 4 个 topic 为准。

## 代码位置

- `src/run_tracker.py`
- `src/pc_logger_protocol.py`
- `src/win_time_sync.py`
- `src/ros2_support.py`
- `src/car_localizer.py`
- `src/curve3.py`

## PC Logger Subscription QoS

This section clarifies the difference between publisher QoS and the QoS used by
`pc_event_logger` when recording the topic.

## Time Axis

All tracker-facing time fields now use the Windows PC `perf_counter()` time
axis consistently. This includes ROS2 published payloads and the tracker JSON
artifacts generated from the same run.

- `/pc_car_loc.t`: `perf_counter()` seconds
- `/predict_hit_pos.ct`: `perf_counter()` seconds
- `/predict_hit_pos.ht`: `perf_counter()` seconds
- `/predict_hit_pos.duration`: seconds on the same monotonic axis, equal to `ht - ct`
- `/arm_logger/control.stamp_pc_ns`: `perf_counter_ns()`
- `/time_sync/pong.t1`: echoed back from RK, expected to already be RK `perf_counter()`
- `/time_sync/pong.t2`: Windows `perf_counter()`

`/time_sync/pong` no longer publishes `t2_epoch`, and the tracker code no
longer performs any epoch/wall-clock fallback or conversion for these fields.

### `/predict_hit_pos`

- Tracker publisher QoS: `RELIABLE`, `depth=1`
- `pc_event_logger` subscriber QoS: `RELIABLE`, `depth=100`

Code references:
- publisher: `src/run_tracker.py`
- recorder subscriber: `src/pc_event_logger.py`

Note:
- If you ask "what QoS is used for recording `/predict_hit_pos`", the answer is:
  `RELIABLE`, `depth=100`
- If you ask "what QoS is used when tracker publishes `/predict_hit_pos`", the answer is:
  `RELIABLE`, `depth=1`
