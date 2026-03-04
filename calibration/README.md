# 双目相机标定子项目

双目标定完整流程：标定板图片采集 → 双目内参/外参标定 → 地面参考点标注 → 大地坐标系注册。

## 硬件

- **相机**: 2 台海康工业相机（DA8199285 左、DA8199402 右），硬件触发同步
- **标定板**: 6×6 AprilTag 网格（tag36h11），tag 间距 71.5mm，tag 边长 55mm
- **检测方式**: `SimpleBlobDetector` + `findCirclesGrid`（将 tag 视为黑色圆形 blob）

## 标定流程

### 步骤 1：采集标定板图片

手持标定板在相机视野中变换位姿（角度、距离），自动定时拍摄同步图片对。

```bash
python -m calibration.calibration_capture --count 60 --duration 180
```

输出保存在 `calibration/images/{serial}/001.png ~ 0xx.png`。

### 步骤 2：运行双目标定

检测所有图片中的 tag 角点，计算两台相机内参和相对外参。

```bash
python -m calibration.run_calibration
```

主要参数：
- `--images images` — 图片目录（相对于 calibration/）
- `--cam1 DA8199285 --cam2 DA8199402` — 相机序列号
- `--range-start 1 --range-end 51` — 标定图片编号范围
- `--output config/stereo_calib.json` — 输出路径（相对于项目根目录）
- `--annotate` — 保存检测标注图片（`*_det.png`）

输出 `config/stereo_calib.json`，包含 K1/D1/K2/D2（内参）和 R_stereo/T_stereo（相对外参）。

**诊断指标参考**：
- 单目 RMS < 0.5 px
- 双目 RMS < 0.5 px
- 有效图像对 ≥ 15

### 步骤 3：采集地面图片

在地面放置可识别的参考点，拍摄同步图片对。

```bash
python -m calibration.ground_capture --count 1 --duration 10
```

输出 `calibration/images/{serial}/ground_001.png`。

### 步骤 4：标注地面参考点

在标注工具中打开地面图片，按顺序点击已知世界坐标的参考点。

```bash
python -m calibration.ground_annotator
```

操作方式：
1. 点击"打开图片"选择 `ground_001.png`
2. 点击"开始标注"，依次点击参考点
3. 滚轮缩放，右键/中键拖动
4. 点击"保存"生成 `ground_001_annotations.json`

标注点配置在 `ground_annotator.py` 顶部的 `GROUND_POINTS` 列表中。
需要为左右两台相机分别标注。

### 步骤 5：注册大地坐标系

基于标注的地面参考点，将双目外参校准到大地坐标系（保持相对外参不变）。

```bash
python -m calibration.register_ground
```

读取 `config/stereo_calib.json` 和两台相机的 `ground_001_annotations.json`，
联合优化 cam1 的 6 DOF 位姿，更新 R1_world/t1_world/pos1_world 和 R2_world/t2_world/pos2_world。

**诊断指标参考**：
- 总 RMS < 5 px（手动标注精度约 2-4 px）

## 输出格式

`config/stereo_calib.json` 字段说明：

| 字段 | 说明 |
|------|------|
| `serial_left/right` | 左/右相机序列号 |
| `image_size` | [宽, 高] 像素 |
| `K1/K2` | 3×3 相机内参矩阵 |
| `D1/D2` | 1×5 畸变系数 |
| `R_stereo` | 3×3 双目旋转（左→右） |
| `T_stereo` | 3×1 双目平移（mm） |
| `E/F` | 本质矩阵/基础矩阵 |
| `R1_world/t1_world` | 左相机外参（世界→相机变换） |
| `R2_world/t2_world` | 右相机外参 |
| `pos1_world/pos2_world` | 相机在世界坐标系中的位置（mm） |
| `units` | 长度单位（mm） |
| `diagnostics` | 各项 RMS 指标 |
| `board` | 标定板参数 |

## 依赖

- `opencv-python` — 标定、检测
- `numpy` — 数值计算
- `scipy` — `least_squares` 联合优化
- `Pillow` — 标注工具图片显示
- `src` — 相机采集（仅采集脚本需要）
