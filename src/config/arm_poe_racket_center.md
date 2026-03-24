# arm_poe_racket_center.json 说明

这份配置文件是当前机械臂“球拍中心位置”POE 标定结果，文件路径是 [arm_poe_racket_center.json](/C:/Users/zlfen/Desktop/ball_tracer_pc/src/config/arm_poe_racket_center.json)。

它不是完整 5 轴末端姿态模型，而是面向“球拍中心位置”的位置版 POE：

- 使用 `joint_1 ~ joint_4` 参与位置链
- `joint_5` 被视为球拍中心附近的姿态轴，不参与平移链
- 目标观测量是 4 相机里球拍关键点 `0-3` 的几何中心

## 1. 坐标系定义

### World

- `world` 来自四相机外参地面注册结果 [four_camera_calib.json](/C:/Users/zlfen/Desktop/ball_tracer_pc/src/config/four_camera_calib.json)
- 地面平面定义为 `z = 0`
- 单位是 `mm`

### Base

- 这次 15.5 重新定义了 `base`
- `base` 原点取 `joint_1` 轴与世界地面 `z = 0` 的交点
- `base z` 轴与 `joint_1` 轴方向一致
- `base y` 轴取 `joint_2` 轴方向在垂直于 `base z` 平面上的投影
- `base x` 轴按右手系补齐

因此，`T_base_in_world` 不是“某个零件 CAD 点”，而是一个为了表达 POE 而选定的数学参考坐标系。

在 15.6 的实时定位约定里，这个 `base` 还有一个明确的物理含义：

- `car chassis center = robotic arm base = this base origin`
- 也就是说，视觉输出里的 `p_car` 应当对应这个 `base` 原点在世界坐标系下的位置

### AprilTag 与 p_car

- 多目视觉先三角化得到的是 `p_apriltag`，它表示 **AprilTag 中心** 在世界坐标系下的位置
- 然后再按固定偏移换算为 `p_car`
- 当前项目约定直接在 **世界坐标轴** 下使用这个偏移，不再额外乘车体朝向

公式是：

```text
p_apriltag = (x, y, z)
p_car      = (x + 0.06, y + 0.10, z - 0.34)   [m]
```

也就是：

```text
apriltag_center_to_car_base_offset = (60, 100, -340)   [mm]
```

## 2. 配置字段含义

### 顶层字段

- `config_role`
  说明这是“球拍中心位置版 POE”配置
- `written_at`
  配置导出时间
- `source_session`
  本次标定使用的数据会话目录
- `source_result`
  标定原始结果文件，包含更完整的中间指标和 outlier 列表
- `camera_calibration`
  使用的四相机内外参配置

### validation_manual

- `step_15_status = completed`
  说明第 15 步已经完成
- `checked_at`
  记录现场人工复核完成时间
- `summary`
  记录手动测量、视觉结果和 POE 结果的总体一致性结论
- `z_axis_note`
  单独记录 `z` 方向的人工复核结论

### vehicle_reference

- `car_base_definition`
  说明 `p_car` 表示车底盘中心，并且这个点与机械臂 `base` 是同一个物理点
- `apriltag_center_measurement`
  说明 `p_apriltag` 是 AprilTag 中心的多目三角化结果
- `p_car_definition`
  说明 `p_car` 不是直接三角化出来的 tag 点，而是 `p_apriltag` 加上固定偏移
- `apriltag_center_to_car_base_offset_m / _mm`
  给出 AprilTag 中心到车底盘中心的固定平移
- `offset_axis_convention`
  明确这个偏移是直接按世界坐标系轴方向相加

### measurement_target

- `name = racket_center`
  说明被标定的是球拍中心，不是法兰盘、不是真正 TCP 姿态原点
- `definition = geometric center of racket keypoints 0-3`
  球拍中心定义为关键点 `0-3` 的几何中心
- `joint_5_position_invariant_assumption = true`
  位置模型假设 `joint_5` 对球拍中心位置没有影响

### T_base_in_world

- `R`
  `3x3` 旋转矩阵，表示 `base` 坐标轴在 `world` 中的方向
- `t_mm`
  `base` 原点在 `world` 中的位置，单位 `mm`

当前数值是：

```json
"T_base_in_world": {
  "R": [
    [0.9703235566, 0.2418089678, 0.0007865205],
    [-0.2418102121, 0.9703200302, 0.0026192110],
    [-0.0001298279, -0.0027316708, 0.9999962606]
  ],
  "t_mm": [-311.1930255, 2166.0068309, 0.0]
}
```

### poe_model_position_only

- `joint_count = 4`
  当前只有前 4 轴进入位置链
- `joint_angle_offsets_rad`
  每个关节角的零位偏置，单位 `rad`
- `home_point_base_mm`
  当所有“进入 POE 的关节角”取零时，球拍中心在 `base` 系中的位置，单位 `mm`

### space_axes_base

每个关节都有 4 个核心量：

- `omega`
  该关节转轴方向，单位向量，无量纲
- `q_mm`
  该转轴上任取的一点，单位 `mm`
- `v_mm`
  由 `v = -omega × q` 计算得到，单位 `mm`
- `screw_axis_base`
  把 `omega` 和 `v` 拼起来的 6 维 screw axis

也就是说，程序实际可直接使用的是：

```text
S_i = [omega_x, omega_y, omega_z, v_x, v_y, v_z]
```

再配合：

- `joint_angle_offsets_rad`
- `home_point_base_mm`

即可恢复这份位置版 POE。

## 3. 计算过程

### Step 1. 数据筛选

从正式会话 [004_15_2_formal_capture_domain2_03192121](/C:/Users/zlfen/Desktop/ball_tracer_pc/ArmCalibration/data/004_15_2_formal_capture_domain2_03192121) 中读取：

- 4 相机彩色图像
- 每张图对应的 `joint_state`
- 15.3 已通过阈值的球拍中心观测

采用的观测口径是：

- 使用所有 `0-3` 号关键点都 `>= 40` 的图片
- 必须有完整 `joint_state`
- 最终进入 15.4/15.5 的是 `1153` 张图片
- 对应 `399` 个有效样本

其中 [sample_0182/sample.json](/C:/Users/zlfen/Desktop/ball_tracer_pc/ArmCalibration/data/004_15_2_formal_capture_domain2_03192121/sample_0182/sample.json) 因早期磁盘写满丢了 `joint_state`，被自动排除。

### Step 2. 初始化

最终目标函数是直接最小化 2D 重投影误差，但为了让优化稳定收敛，初始化分成两层：

1. 只对“至少 2 台相机合格”的样本做三角化初始化
2. 用一个 4 轴结构化模型先拟合 3D 点云，拿到粗初值
3. 再把它转换成通用的 4 轴 world-space screw axis 初值

初始化统计：

- `293` 个样本拥有 `>= 2` 个相机观测
- 其中 `287` 个样本的三角化平均重投影误差 `<= 15 px`
- 这 `287` 个样本用于初始化

### Step 3. 直接 2D 重投影优化

真正优化的目标不是 3D 点，而是所有图片上的 2D 重投影误差：

```text
min_theta  Σ rho( || pi_c( p_world(q_k; theta) ) - u_obs(k, c) ||^2 )
```

其中：

- `theta` 是待优化参数
- `q_k` 是第 `k` 个样本的关节角
- `p_world(q_k; theta)` 是 POE 预测出的球拍中心世界坐标
- `pi_c(.)` 是第 `c` 台相机的投影函数，使用现有内外参
- `u_obs(k, c)` 是第 `k` 个样本在第 `c` 台相机中的观测像素
- `rho` 使用 `soft_l1` robust loss，尺度 `5 px`

### Step 4. 15.5 的 base 重定义

2D 优化收敛后，世界系下的几何已经确定。  
15.5 做的事情不是“重新拟合出另一套世界几何”，而是：

- 保持 world 中的 4 条关节轴和球拍中心模型不变
- 只把 `base` 原点改到 `joint_1` 轴与 `z=0` 地面的交点
- 然后把 world-space 的结果重新表达成新的 base-space POE

所以：

- 世界系重投影指标不变
- `T_base_in_world` 和 `space_axes_base` 会变

## 4. 当前技术指标

### 数据规模

- 有效样本数：`399`
- 合格图片数：`1153`
- 用于初始化的双目以上样本：`293`
- 初始化 inlier：`287`

### 全局重投影指标

- 平均误差：`6.189 px`
- 中位数误差：`4.145 px`
- `P95`：`11.764 px`
- 最大误差：`829.407 px`
- `> 10 px`：`99` 张
- `> 20 px`：`18` 张
- `> 50 px`：`5` 张

### 分相机指标

- `DA7403103`: mean `4.910 px`, median `3.469 px`, p95 `11.814 px`
- `DA8571029`: mean `9.299 px`, median `5.150 px`, p95 `12.680 px`
- `DA7403087`: mean `4.946 px`, median `4.279 px`, p95 `11.186 px`
- `DA8474746`: mean `5.920 px`, median `3.887 px`, p95 `12.746 px`

当前最明显的 outlier 仍然集中在少数坏图，尤其 `DA8571029`。

### 现场人工复核

- 第 15 步已经完成现场人工复核
- 综合手动测量、视觉测量和 POE 结果，整体误差在几厘米量级
- `z` 方向精度达到厘米级
- 这些人工复核结论同时记录在 [DEV.md](/C:/Users/zlfen/Desktop/ball_tracer_pc/DEV.md)

### 关节轴几何关系

这里的 `joint_i -> joint_j` 不是“关节点到关节点”的距离，而是“两条转轴直线的最短距离”。

当前结果：

- `joint_1 -> joint_2 = 8.33 mm`
- `joint_2 -> joint_3 = 388.58 mm`
- `joint_3 -> joint_4 = 4.28 mm`

这说明拟合出来的结构更接近：

- `joint_1` 与 `joint_2` 近相交
- `joint_2` 到 `joint_3` 是主要连杆跨度
- `joint_3` 与 `joint_4` 近相交

## 5. 为什么 joint_5 现在可以不进位置 POE

这次任务优化的是“球拍中心位置”。  
如果 `joint_5` 是穿过球拍中心附近的姿态轴，那么它会改变球拍姿态，但不会明显改变球拍中心位置。

当前数据上，`joint_5` 的覆盖只有：

- 范围 `[-0.0545, 0.2370] rad`
- 总宽度 `0.2914 rad`
- `0.1 rad` 离散后共 `4` 个区间

但当前忽略 `joint_5` 的位置项后：

- 全局误差仍已经收敛到 `mean 6.19 px`
- 残差绝对值与 `q5` 的相关系数只有 `-0.0274`

这说明对“球拍中心位置”这件事，`joint_5` 目前没有表现出强可观测的位置影响。

注意这不代表：

- `joint_5` 一定对姿态没有影响
- 以后做完整 TCP 姿态标定时也可以忽略它

它只代表：**在当前任务定义下，`joint_5` 对球拍中心位置不是主导项。**

## 6. 建议的下游使用方式

若 RK 端只需要预测球拍中心位置，可直接使用：

1. `T_base_in_world`
2. `joint_angle_offsets_rad`
3. `space_axes_base`
4. `home_point_base_mm`

如果 RK 端需要做 15.6 的实时对比，推荐直接对比这两个量：

- `p_racket_rel_base_in_world(by poe)`
  由 POE 正解得到的世界系相对位移，即 `p_racket_world(by poe) - T_base_in_world.t_mm`
- `p_racket_rel_base_in_world(by vision)`
  由视觉三角化得到的世界系相对位移，即 `p_racket_world(by vision) - p_car`

其中 `p_car = p_apriltag + (60, 100, -340) mm`。

如果 RK 端要做的是完整末端姿态或拍面法向控制，则还需要补：

- `joint_5` 的姿态链
- 球拍局部坐标系定义
- 姿态观测或更多 `joint_5` 激励数据

## 7. 相关文件

- 配置：[arm_poe_racket_center.json](/C:/Users/zlfen/Desktop/ball_tracer_pc/src/config/arm_poe_racket_center.json)
- 结果：[poe_reprojection_result.json](/C:/Users/zlfen/Desktop/ball_tracer_pc/ArmCalibration/data/004_15_2_formal_capture_domain2_03192121/poe_reprojection_result.json)
- 标定脚本：[calibrate_poe_reprojection.py](/C:/Users/zlfen/Desktop/ball_tracer_pc/ArmCalibration/calibrate_poe_reprojection.py)
- 四相机外参：[four_camera_calib.json](/C:/Users/zlfen/Desktop/ball_tracer_pc/src/config/four_camera_calib.json)
