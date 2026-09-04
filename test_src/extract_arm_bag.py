# -*- coding: utf-8 -*-
"""从 tracker rosbag 提取机械臂数据为 {run_id}_arm.json。

必须在 ROS2 环境中运行（经 ros2/run_ros2.bat 启动），依赖 rosbag2_py。
TCP 正解：v0.3 用本文件内置的 USD 链；v0.4 直接引用标准文件——tennis-man/arm_controller 的
compact_arm_kinematics + config/cars/v04.yaml（零位 offset_rad、tool_x、hit_pos_z_offset_m 都在那里，
2026-09-05 用户定：臂端 FK 与报告 FK 只认这一份真值；路径可用 TENNIS_MAN_ARM_CONTROLLER 覆盖）。

⚠ 车型（--car v03|v04）决定用哪条 FK 链，**没有默认值**：两台车的肩高、连杆与拍心标定距离
均不同，选错不会报错、只会让整场 TCP 偏几厘米。不给 --car 时从同目录的 tracker JSON
（config.car_config_path，run_tracker 按启动的 --car 写入）推断，推不出来直接失败。
选中的车型写进输出的 "car"/"car_source"/"fk_source"，报告端按它复算，绝不自己猜。

输出供 test_src/generate_curve3_html.py 的 Arm tab 使用：
  states   — /joint_states 实际关节位置/速度/力矩 + FK TCP
  commands — /tennis/motor_command 目标（首个轨迹点）+ FK TCP
  events   — status / arm_command / hit_pos / predict_hit_pos 文本事件

时间轴（全项目只有两个时间轴）：
  所有 t 一律为 RK 单调钟（CLOCK_MONOTONIC）绝对秒 —— 与 /predict_hit_pos
  的 ct/ht、rk_tracking 的 payload t 同一个钟。报告端固定用 scale=1 的 RK→PC
  常数偏移映射，臂数据没有任何独立的桥。

  新固件（damiao 驱动 + arm_controller 单调钟版）：
    /joint_states、/tennis/motor_command 的 header.stamp 就是 RK 单调钟，
    /tennis/status 文本尾缀 " t=<单调秒>"，全部原生直读、零换算。
  旧 bag 兼容（stamp 为 epoch 系统钟、status 无 t= 的场次）：
    t = stamp − median(stamp − recv) + median(bot_state.t − recv)，
    两个中位数里的传输延迟互相抵消，残差 ~ms 级；status 等无时间事件
    退回 recv + median(bot_state.t − recv)。clock_sync 里明示所用模式。

  clock_sync 自检随文件输出：RK 单调钟 vs PC 收钟漂移率（ms/min）、
  joint_states stamp 时钟域、ht 锚点残差（新调度触球≡ht 时 done−ht 中位，
  验证时间链）。漂移率若超 ~2ms/min，单 offset 假设需复查。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Iterable, NamedTuple

import numpy as np


# ── 车型运动学（v0.3 / v0.4 是两台不同的臂）────────────────────────────────────
# v03：逐值抄自 arm_controller.compact_arm_kinematics@a266857（USD tennis_arm_j5j6_7_6_world）。
# v04：**不抄**，从标准 checkout 加载 compact_arm_kinematics 并 use_car("v04")——关节表、工具轴、
# 甜点距离（yaml kinematics.tool_x 反推）全部来自 config/cars/v04.yaml 这一份真值。
# 一致性由 test_src/test_arm_kinematics_cars.py 拿臂端导出的黄金向量（assets/<car>/test_vectors.json）守着。
#
# ⚠ 拿错车算 TCP **不会报错**，只会整场偏几厘米：当前资产用 v0.3 链交叉计算 v0.4
#   黄金位形时 x 远约 2cm、z 低约 12cm，而拍面 yaw/pitch 两车逐拍恒等（旋转链一致）——
#   从角度列根本看不出来。
#   故本模块**没有默认车型**：先 use_car()（或 car_for_tracker_json 推断）再调 fk()。
SHORT_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
_V03_ROOT_LINK = "/tennis_arm_j5j6_7_6/Geometry/base_link"
ROOT_LINK = _V03_ROOT_LINK          # use_car() 重绑；留着是给老调用点的名字


def _pose(pos, quat_wxyz) -> np.ndarray:
    w, x, y, z = quat_wxyz
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    out = np.eye(4)
    out[:3, :3] = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    out[:3, 3] = pos
    return out


_AXIS = {
    "X": np.array([1.0, 0.0, 0.0]),
    "Y": np.array([0.0, 1.0, 0.0]),
    "Z": np.array([0.0, 0.0, 1.0]),
}

# ── v0.3：physics:localPos/localRot/axis copied verbatim from the USD PhysicsRevoluteJoints.
_V03_JOINTS = (
    {
        "name": "J1_joint",
        "parent": _V03_ROOT_LINK,
        "child": _V03_ROOT_LINK + "/J1_Link",
        "axis": _AXIS["Y"],
        "local0": _pose((0.0, 0.0, 0.4385), (-0.4999999, -0.5, 0.49999994, 0.5)),
        "local1": _pose((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
    },
    {
        "name": "J2_JOINT",
        "parent": _V03_ROOT_LINK + "/J1_Link",
        "child": _V03_ROOT_LINK + "/J1_Link/J2_Link",
        "axis": _AXIS["Z"],
        "local0": _pose((5e-05, 0.1719001, -0.07355), (0.0, -1.0, 0.0, 0.0)),
        "local1": _pose((0.0, 0.0, 0.0), (0.0, -1.0, 0.0, 0.0)),
    },
    {
        "name": "J3_joint",
        "parent": _V03_ROOT_LINK + "/J1_Link/J2_Link",
        "child": _V03_ROOT_LINK + "/J1_Link/J2_Link/J3_Link",
        "axis": _AXIS["Z"],
        "local0": _pose((-0.000125, 0.44997022, 0.02125), (0.0, -1.0, 0.0, 0.0)),
        "local1": _pose((0.0, 0.0, 0.0), (0.0, -1.0, 0.0, 0.0)),
    },
    {
        "name": "J4_JOINT",
        "parent": _V03_ROOT_LINK + "/J1_Link/J2_Link/J3_Link",
        "child": _V03_ROOT_LINK + "/J1_Link/J2_Link/J3_Link/J4_Link",
        "axis": _AXIS["Y"],
        "local0": _pose((0.0, 0.3, 0.08575), (0.70710677, 0.70710677, 0.0, 0.0)),
        "local1": _pose((0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)),
    },
    {
        "name": "J5_JOINT",
        "parent": _V03_ROOT_LINK + "/J1_Link/J2_Link/J3_Link/J4_Link",
        "child": _V03_ROOT_LINK + "/J1_Link/J2_Link/J3_Link/J4_Link/J5_Link",
        "axis": _AXIS["Y"],
        "local0": _pose((0.0, 0.03973, 0.095), (0.4999999, 0.5, -0.49999994, -0.5)),
        "local1": _pose((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
    },
    {
        "name": "J6_JOINT",
        "parent": _V03_ROOT_LINK + "/J1_Link/J2_Link/J3_Link/J4_Link/J5_Link",
        "child": _V03_ROOT_LINK + "/J1_Link/J2_Link/J3_Link/J4_Link/J5_Link/J6_Link",
        "axis": _AXIS["Z"],
        "local0": _pose(
            (0.00042069482, 0.047, 0.031753197),
            (1.6081226e-16, 1.6155446e-15, -1.0, -1.6653345e-15),
        ),
        "local1": _pose((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
    },
)

# USD base_link -> hit convention: rotate +90 deg about Z so -Y_base becomes +X.
BASE_ROT = np.array(
    [
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)



class CarModel(NamedTuple):
    """一台车的整条 FK 链。tcp_distance = 拍甜点沿 link6 工具轴的标定距离。"""

    car: str
    source_model: str
    root_link: str
    base_transform: np.ndarray      # 世界 ← root_link
    joints: tuple
    tool_axis_in_link6: np.ndarray  # 拍柄方向
    face_normal_in_link6: np.ndarray
    tcp_distance: float

# ── v0.4：标准运动学（唯一真值 = tennis-man/arm_controller 的 compact_arm_kinematics + config/cars/v04.yaml）
ARM_CONTROLLER_ROOT = Path(os.environ.get("TENNIS_MAN_ARM_CONTROLLER", "D:/tennis-man/arm_controller"))
_STANDARD_V04 = None


def standard_v04_kinematics():
    """加载标准 checkout 的 compact_arm_kinematics 并切到 v04（缓存）。缺 checkout 直接报错，不回退。"""
    global _STANDARD_V04
    if _STANDARD_V04 is None:
        import importlib.util

        path = ARM_CONTROLLER_ROOT / "src" / "arm_controller" / "compact_arm_kinematics.py"
        if not path.is_file():
            raise SystemExit(
                f"标准运动学不存在：{path}（设 TENNIS_MAN_ARM_CONTROLLER 指向 arm_controller checkout）")
        spec = importlib.util.spec_from_file_location("standard_compact_arm_kinematics", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.use_car("v04")
        _STANDARD_V04 = module
    return _STANDARD_V04


def _standard_v04_model() -> "CarModel":
    cak = standard_v04_kinematics()
    return CarModel(
        car="v04",
        source_model=f"{cak.URDF_PATH} + config/cars/v04.yaml",
        root_link=cak.ROOT_LINK,
        base_transform=cak.WORLD_TO_BASE,
        joints=cak.JOINTS,
        tool_axis_in_link6=cak.TOOL_AXIS_IN_LINK6,
        face_normal_in_link6=cak.FACE_NORMAL_IN_LINK6,
        tcp_distance=float(cak.TCP_DISTANCE),
    )


CAR_MODELS = {
    "v03": CarModel(
        car="v03",
        source_model="src/arm_controller/urdf/tennis_arm_j5j6_7_6_world.usd",
        root_link=_V03_ROOT_LINK,
        base_transform=BASE_ROT,
        joints=_V03_JOINTS,
        tool_axis_in_link6=np.array([0.0, 0.0, 1.0]),
        face_normal_in_link6=np.array([1.0, 0.0, 0.0]),
        tcp_distance=0.62,
    ),
    "v04": _standard_v04_model(),
}

# tracker JSON 的 config.car_config_path 文件名 → 车型（src/run_tracker.py CAR_LAYOUT_CONFIGS 的逆）。
CAR_BY_LAYOUT_CONFIG = {
    "arm_poe_racket_center.json": "v03",
    "vehicle_v04.json": "v04",
}

_ACTIVE: CarModel | None = None
# use_car() 重绑的模块级别名（老调用点按名字取，故必须整组一起换）
JOINTS: tuple = ()
TOOL_AXIS_IN_LINK6 = CAR_MODELS["v03"].tool_axis_in_link6
FACE_NORMAL_IN_LINK6 = CAR_MODELS["v03"].face_normal_in_link6


def use_car(car: str) -> CarModel:
    """选定车型；此后 fk()/JOINTS/ROOT_LINK 等全部按这台车。"""
    global _ACTIVE, JOINTS, ROOT_LINK, TOOL_AXIS_IN_LINK6, FACE_NORMAL_IN_LINK6
    if car not in CAR_MODELS:
        raise ValueError(f"未知车型 {car!r}，可选：{'/'.join(sorted(CAR_MODELS))}")
    _ACTIVE = CAR_MODELS[car]
    JOINTS = _ACTIVE.joints
    ROOT_LINK = _ACTIVE.root_link
    TOOL_AXIS_IN_LINK6 = _ACTIVE.tool_axis_in_link6
    FACE_NORMAL_IN_LINK6 = _ACTIVE.face_normal_in_link6
    return _ACTIVE


def active_car() -> CarModel:
    if _ACTIVE is None:
        raise RuntimeError(
            "还没选车型：先调 extract_arm_bag.use_car('v03'|'v04')（或 car_for_tracker_json 推断）。"
            "没有默认值是故意的——两台车的臂不同，选错只会静默偏几厘米，见文件头注释。"
        )
    return _ACTIVE


def car_for_tracker_json(tracker_json: Path | str) -> tuple[str, str]:
    """从一场 tracker JSON 推车型，返回 (car, 依据)。

    判据 = config.car_config_path 的文件名（run_tracker 启动时按 --car 写进去的）。
    2026-08-15 之前的场次没有这个字段——那时只有 v0.3 一台车，按 v03 处理并在依据里写明。
    """
    path = Path(tracker_json)
    try:
        with path.open(encoding="utf-8") as fh:
            config = json.load(fh).get("config") or {}
    except Exception as exc:
        raise RuntimeError(f"读不了 {path} 的 config，无法推车型（{exc!r}）") from exc
    raw = config.get("car_config_path")
    if not raw:
        return "v03", f"{path.name} 无 config.car_config_path（0815 之前只有 v0.3 一台车）"
    name = Path(str(raw)).name
    car = CAR_BY_LAYOUT_CONFIG.get(name)
    if car is None:
        raise RuntimeError(
            f"{path.name} 的 car_config_path={name} 认不出车型；"
            f"已知：{CAR_BY_LAYOUT_CONFIG}。新车型要同时在这里和 run_tracker.CAR_LAYOUT_CONFIGS 登记。"
        )
    return car, f"{path.name} config.car_config_path={name}"


def car_for_session(arm: dict | None = None, tracker_json: Path | str | None = None,
                    explicit: str | None = None) -> tuple[str, str]:
    """一场数据用哪台车：显式 > arm JSON 自述 > tracker JSON 推断。三条都没有就抛。

    报告/离线分析都走这一份，免得各家自己猜出不同的车。
    """
    if explicit:
        if explicit not in CAR_MODELS:
            raise ValueError(f"未知车型 {explicit!r}")
        return explicit, "显式指定"
    said = (arm or {}).get("car") if isinstance(arm, dict) else None
    if said:
        if said not in CAR_MODELS:
            raise RuntimeError(f"arm JSON 自述车型 {said!r} 不认识")
        return said, f"arm JSON 自述（{(arm or {}).get('car_source', '来源未记')}）"
    if tracker_json:
        return car_for_tracker_json(tracker_json)
    raise RuntimeError("推不出车型：arm JSON 没有 car 字段，也没给 tracker JSON")


def recompute_tcp(rows) -> int:
    """就地把 rows[].tcp 按当前车型复算（关节残缺的置 None），返回改写了几行。

    老 arm JSON 的 tcp 是当年写死的 v0.3 链算的；换车/换模型后不必重跑 rosbag 提取，
    读进来复算一遍即可——position 才是原始记录，tcp 只是派生量。
    """
    n = 0
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        q = row.get("position")
        ok = isinstance(q, list) and len(q) == 6 and all(isinstance(v, (int, float)) for v in q)
        row["tcp"] = [round(float(v), 4) for v in fk(q)["tcp"]] if ok else None
        n += 1
    return n


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c
    out = np.eye(4)
    out[:3, :3] = np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ]
    )
    return out


def _q6(q: Iterable[float]) -> np.ndarray:
    q = np.asarray(tuple(q), dtype=float)
    if q.shape == (5,):
        q = np.concatenate((q, [0.0]))
    if q.shape != (6,):
        raise ValueError(f"expected 5 or 6 joint values, got shape {q.shape}")
    return q


def fk(q: Iterable[float], *, tcp_distance: float | None = None,
       car: str | None = None) -> dict[str, np.ndarray]:
    """当前车型的精确正解，输出击球系（+X 前、+Z 上）。q 为控制器关节角。

    car 不给就用 use_car() 选定的那台；一次都没选过直接抛（没有默认车型，见文件头）。
    """
    model = CAR_MODELS[car] if car is not None else active_car()
    q = _q6(q)
    link_transforms = {model.root_link: model.base_transform.copy()}
    joint_frames = {}

    for angle, joint in zip(q, model.joints):
        joint_t = link_transforms[joint["parent"]] @ joint["local0"]
        child_t = joint_t @ _axis_rotation(joint["axis"], angle) @ np.linalg.inv(joint["local1"])
        joint_frames[joint["name"]] = joint_t
        link_transforms[joint["child"]] = child_t

    link6 = link_transforms[model.joints[-1]["child"]]
    handle_axis = link6[:3, :3] @ model.tool_axis_in_link6
    handle_axis = handle_axis / np.linalg.norm(handle_axis)
    face_normal = link6[:3, :3] @ model.face_normal_in_link6
    tcp = link6[:3, 3] + (model.tcp_distance if tcp_distance is None else tcp_distance) * handle_axis

    return {
        "q": q,
        "car": model.car,
        "tcp": tcp,
        "handle_axis": handle_axis,
        "face_normal": face_normal / np.linalg.norm(face_normal),
        "link6": link6,
        "joints": model.joints,
        "joint_frames": joint_frames,
        "link_transforms": link_transforms,
    }

EVENT_TOPICS = (
    "/tennis/status",
    "/tennis/arm_command",
    "/tennis/hit_pos",
    "/predict_hit_pos",
    "/arm_controller/status",
)

# RK 单调钟参考话题（按优先级）：payload 带发布时刻的单调钟 t，发布延迟接近 0。
# 仅旧 bag（epoch stamp / 无 t= 的 status）需要；新固件全部原生直读。
# /predict_hit_pos 的 ct 是球观测时刻（比发布早一个 RK 管线时延，0716 实测
# ~70ms），只配当保底，用到时在 clock_sync 里明示 biased。
MONO_REF_TOPICS = ("/bot_state", "/chassis_can/imu")

# 单调钟量级上限：RK 开机秒（连续运行数月也 <1e8）；epoch 秒 ~1.7e9。
MONO_MAX_SEC = 1e8

# status 文本尾缀发布时刻（arm_controller 单调钟版追加）："... t=9203.123456"
STATUS_T_RE = re.compile(r"\s+t=([0-9]+\.[0-9]+)$")

# 事件文本必须原样落盘：/predict_hit_pos 是报告端要 JSON.parse 的载荷、status 的发布时刻
# `t=` 挂在尾部，截一刀两者都静默失效。原来的 500 字上限就这么炸过一次——0809 103849 场
# RK 端加了 spin/cor 在线估计字段后 payload 越过 500，625 条 predict_hit_pos 全部 parse
# 失败 → armPreds 空 → accepted 回配 0 票 → 整张臂表全是 —，页面上没有任何报错。
# 这里只留一个防病态消息的宽上限，且一旦触发就在 stderr 告警，不再无声截断。
EVENT_TEXT_MAX = 4000


def _ordered(values: list[float], names: list[str], joint_names: tuple[str, ...]) -> list[float | None]:
    """按 joint_names 顺序重排（与 session_viewer._ordered 一致）。"""
    by_name = {name: idx for idx, name in enumerate(names)}
    ordered: list[float | None] = []
    for idx, name in enumerate(joint_names):
        src = by_name.get(name, idx if not names else None)
        ordered.append(float(values[src]) if src is not None and src < len(values) else None)
    return ordered


def _round_list(values: list[float | None], digits: int) -> list[float | None]:
    return [None if v is None else round(v, digits) for v in values]


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _thirds_drift_ms_per_min(samples: list[tuple[float, float]]) -> dict | None:
    """(recv_s, diff_s) 样本按首/末三分之一中位差估漂移率。中位数抗离群，O(n)。"""
    if len(samples) < 30:
        return None
    samples = sorted(samples)
    span = samples[-1][0] - samples[0][0]
    if span < 10.0:
        return None
    third = span / 3.0
    lo = [d for t, d in samples if t <= samples[0][0] + third]
    hi = [d for t, d in samples if t >= samples[-1][0] - third]
    t_lo = statistics.median([t for t, _ in samples if t <= samples[0][0] + third])
    t_hi = statistics.median([t for t, _ in samples if t >= samples[-1][0] - third])
    if t_hi - t_lo < 1.0:
        return None
    rate = (statistics.median(hi) - statistics.median(lo)) / (t_hi - t_lo)
    return {
        "ms_per_min": round(rate * 60_000, 3),
        "span_s": round(span, 1),
        "n": len(samples),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True, help="rosbag 目录（含 metadata.yaml）")
    parser.add_argument("--output", type=Path, required=True, help="输出 arm JSON 路径")
    parser.add_argument(
        "--car",
        choices=sorted(CAR_MODELS),
        help="车型（决定 FK 链）。不给就从同名 tracker JSON 的 config.car_config_path 推",
    )
    args = parser.parse_args()

    # 车型：显式 > tracker JSON 推断 > 失败。默认成另一台车比报错危险得多（静默偏几厘米）。
    if args.car:
        car, car_source = args.car, "--car 显式指定"
    else:
        tracker_json = args.bag.with_name(args.bag.name[:-len("_rosbag")] + ".json") \
            if args.bag.name.endswith("_rosbag") else args.output.with_name(
                args.output.name.replace("_arm.json", ".json"))
        if not tracker_json.exists():
            parser.error(
                f"没给 --car，也找不到同场 tracker JSON（{tracker_json}）来推车型。"
                f"请补 --car {'/'.join(sorted(CAR_MODELS))}。"
            )
        car, car_source = car_for_tracker_json(tracker_json)
    model = use_car(car)
    print(f"[extract_arm_bag] 车型 {car}（依据：{car_source}）；FK 源模型 {model.source_model}")

    import rosbag2_py  # noqa: E402
    from rclpy.serialization import deserialize_message  # noqa: E402
    from rosidl_runtime_py.utilities import get_message  # noqa: E402

    joint_names = tuple(SHORT_JOINT_NAMES)

    def tcp_of(positions: list[float | None]) -> list[float] | None:
        if any(v is None for v in positions):
            return None
        try:
            return [round(float(v), 4) for v in fk(positions)["tcp"]]
        except Exception:
            return None

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(args.bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )
    type_by_topic = {item.name: item.type for item in reader.get_all_topics_and_types()}
    msg_types = {}
    for topic, type_name in type_by_topic.items():
        try:
            msg_types[topic] = get_message(type_name)
        except Exception:
            pass  # 无法解析的类型只计数，不采样

    states: list[dict] = []
    commands: list[dict] = []
    events: list[dict] = []
    state_diffs: list[tuple[float, float]] = []    # (recv_s, stamp − recv) 全部有效 stamp
    command_diffs: list[tuple[float, float]] = []
    mono_diffs: dict[str, list[tuple[float, float]]] = {t: [] for t in MONO_REF_TOPICS}
    predict_ct_diffs: list[tuple[float, float]] = []
    counts: dict[str, int] = {}
    seen_state_names: list[str] = []
    seen_command_names: list[str] = []
    truncated: dict[str, int] = {}
    start_ns: int | None = None
    end_ns: int | None = None

    def _header_stamp_sec(msg) -> float:
        return msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9

    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        counts[topic] = counts.get(topic, 0) + 1
        if start_ns is None:
            start_ns = timestamp
        end_ns = timestamp
        msg_type = msg_types.get(topic)
        if msg_type is None:
            continue
        recv = timestamp / 1e9

        if topic == "/joint_states":
            msg = deserialize_message(data, msg_type)
            names = list(msg.name)
            if not seen_state_names and names:
                seen_state_names = names
            positions = _ordered(list(msg.position), names, joint_names)
            stamp = _header_stamp_sec(msg)
            if stamp > 0.0:
                state_diffs.append((recv, stamp - recv))
            states.append(
                {
                    "stamp": stamp,
                    "recv": recv,
                    "position": _round_list(positions, 5),
                    "velocity": _round_list(_ordered(list(msg.velocity), names, joint_names), 5),
                    "effort": _round_list(_ordered(list(msg.effort), names, joint_names), 5),
                    "tcp": tcp_of(positions),
                }
            )
        elif topic == "/tennis/motor_command":
            msg = deserialize_message(data, msg_type)
            if not msg.points:
                continue
            point = msg.points[0]
            names = list(msg.joint_names)
            if not seen_command_names and names:
                seen_command_names = names
            positions = _ordered(list(point.positions), names, joint_names)
            stamp = _header_stamp_sec(msg)
            if stamp > 0.0:
                command_diffs.append((recv, stamp - recv))
            commands.append(
                {
                    "stamp": stamp,
                    "recv": recv,
                    "position": _round_list(positions, 5),
                    "velocity": _round_list(_ordered(list(point.velocities), names, joint_names), 5),
                    "effort": _round_list(_ordered(list(point.effort), names, joint_names), 5),
                    "tcp": tcp_of(positions),
                }
            )
        elif topic in EVENT_TOPICS or topic in MONO_REF_TOPICS:
            msg = deserialize_message(data, msg_type)
            raw = msg.data if hasattr(msg, "data") else None
            if topic in MONO_REF_TOPICS:
                try:
                    payload = json.loads(raw)
                    t_mono = payload.get("t")
                    if isinstance(t_mono, (int, float)):
                        mono_diffs[topic].append((recv, float(t_mono) - recv))
                except Exception:
                    pass
                continue
            if raw is not None:
                text = (
                    " ".join(f"{float(v):.4g}" for v in raw)
                    if isinstance(raw, (list, tuple)) or type(raw).__name__ == "array"
                    else str(raw)
                )
            else:
                text = str(msg)
            if len(text) > EVENT_TEXT_MAX:
                truncated[topic] = truncated.get(topic, 0) + 1
                text = text[:EVENT_TEXT_MAX]
            event = {"recv": recv, "topic": topic, "text": text, "t_payload": None}
            if topic == "/predict_hit_pos":
                # payload 自带 ct（RK 单调钟，球观测时刻）——事件直接用它，
                # 与 rk_tracking 的 pred 序列同源同值。
                try:
                    ct = json.loads(text).get("ct")
                    if isinstance(ct, (int, float)):
                        predict_ct_diffs.append((recv, float(ct) - recv))
                        event["t_payload"] = float(ct)
                except Exception:
                    pass
            else:
                # 新固件 status 尾缀 " t=<单调秒>" = 发布时刻，解析后从文本剥离
                m = STATUS_T_RE.search(event["text"])
                if m:
                    event["t_payload"] = float(m.group(1))
                    event["text"] = event["text"][: m.start()]
            events.append(event)

    if start_ns is None:
        raise RuntimeError(f"bag has no messages: {args.bag}")

    if truncated:
        detail = ", ".join(f"{topic} × {n}" for topic, n in sorted(truncated.items()))
        print(
            f"[extract_arm_bag] 警告：事件文本超过 {EVENT_TEXT_MAX} 字被截断（{detail}）——"
            "JSON 载荷会 parse 失败、status 尾缀 t= 会丢，报告端相关列将整列为空",
            file=sys.stderr,
        )

    # ---- stamp 时钟域判定（按话题多数）----
    def _stamp_domain(rows: list[dict]) -> str | None:
        vals = [row["stamp"] for row in rows if row["stamp"] > 0.0]
        if not vals:
            return None
        mono_n = sum(1 for v in vals if v < MONO_MAX_SEC)
        return "rk_mono_native" if mono_n * 2 >= len(vals) else "rk_sys_converted_legacy"

    state_domain = _stamp_domain(states)
    command_domain = _stamp_domain(commands)
    # legacy = 旧固件的 epoch stamp；recv 映射 = 无自带时间的事件（PC 发的
    # hit_pos/arm_command、旧固件 status）。后者在新 bag 里也存在，不算 legacy。
    legacy_stamps = "rk_sys_converted_legacy" in (state_domain, command_domain)
    needs_c_mono = legacy_stamps or any(e["t_payload"] is None for e in events)

    # ---- RK 单调钟参考 C_mono = median(payload t − recv) ----
    mono_ref_topic = None
    mono_ref_biased = False
    c_mono = None
    if needs_c_mono:
        for topic in MONO_REF_TOPICS:
            if len(mono_diffs[topic]) >= 30:
                mono_ref_topic = topic
                c_mono = _median([d for _, d in mono_diffs[topic]])
                break
        if c_mono is None and len(predict_ct_diffs) >= 10:
            # 保底：ct 是观测时刻，比发布早一个 RK 管线时延 → 相关事件整体偏早同量
            mono_ref_topic = "/predict_hit_pos(ct, biased)"
            mono_ref_biased = True
            c_mono = _median([d for _, d in predict_ct_diffs])
        if c_mono is None:
            raise RuntimeError(
                "no RK mono reference in bag (/bot_state, /chassis_can/imu or "
                "/predict_hit_pos payload) — cannot map stamp-less/legacy data onto the RK axis"
            )

    c_sys_js = _median([d for _, d in state_diffs])
    c_sys_mc = _median([d for _, d in command_diffs])

    # ---- 统一落到 RK 单调钟绝对秒 ----
    def _finish(rows: list[dict], c_sys: float | None) -> None:
        for row in rows:
            stamp = row.pop("stamp")
            recv = row.pop("recv")
            if 0.0 < stamp < MONO_MAX_SEC:
                row["t"] = round(stamp, 5)                    # 新固件：原生单调钟直读
            elif stamp >= MONO_MAX_SEC and c_sys is not None and c_mono is not None:
                row["t"] = round(stamp - c_sys + c_mono, 5)   # 旧 bag：epoch → 单调钟
            elif c_mono is not None:
                row["t"] = round(recv + c_mono, 5)            # 缺 stamp：接收时刻映射
            else:
                row["t"] = round(recv, 5)                     # 不可达（needs_legacy 已保证 c_mono）

    _finish(states, c_sys_js)
    _finish(commands, c_sys_mc)
    status_payload_n = 0
    for e in events:
        if e["t_payload"] is not None:
            e["t"] = round(e["t_payload"], 5)
            status_payload_n += 1
        elif c_mono is not None:
            e["t"] = round(e["recv"] + c_mono, 5)
        else:
            e["t"] = round(e["recv"], 5)
    # dict 键序整理：t 放行首，便于肉眼查文件
    states = [{"t": r["t"], **{k: r[k] for k in ("position", "velocity", "effort", "tcp")}} for r in states]
    commands = [{"t": r["t"], **{k: r[k] for k in ("position", "velocity", "effort", "tcp")}} for r in commands]
    events = [{"t": e["t"], "topic": e["topic"], "text": e["text"]} for e in events]

    # ---- ht 锚点残差：新调度触球≡ht，done(=accepted.t+duration) − ht 应 ≈0 ----
    # 全链自检：stamp 域 + status 发布时刻 + predict ct 任一有偏都会体现在这里。
    predicts: list[dict] = []
    for e in events:
        if e["topic"] != "/predict_hit_pos":
            continue
        try:
            p = json.loads(e["text"])
            predicts.append({"t": e["t"], "duration": p.get("duration"),
                             "rel_x": p.get("rel_x"), "ht": p.get("ht")})
        except Exception:
            pass
    ht_residuals: list[float] = []
    acc_re = re.compile(r"^accepted hit x=([\-0-9.]+) z=[\-0-9.]+ duration=([0-9.]+)")
    for e in events:
        if e["topic"] != "/tennis/status":
            continue
        m = acc_re.match(e["text"])
        if not m:
            continue
        x, dur = float(m.group(1)), float(m.group(2))
        for p in reversed(predicts):
            if p["t"] > e["t"]:
                continue
            if e["t"] - p["t"] > 0.35:
                break
            d_ok = isinstance(p["duration"], (int, float)) and abs(p["duration"] - dur) < 2e-3
            x_ok = isinstance(p["rel_x"], (int, float)) and abs(p["rel_x"] - x) < 5e-4
            if (d_ok or x_ok) and isinstance(p["ht"], (int, float)):
                ht_residuals.append((e["t"] + dur) - p["ht"])
                break

    mono_drift_samples = (
        mono_diffs[mono_ref_topic] if mono_ref_topic in mono_diffs
        else (state_diffs if state_domain == "rk_mono_native" else predict_ct_diffs)
    )
    clock_sync = {
        "joint_states_stamp_domain": state_domain,
        "motor_command_stamp_domain": command_domain,
        "events_with_payload_t": status_payload_n,
        "events_total": len(events),
        "legacy_stamps": legacy_stamps,
        "mono_ref_topic": mono_ref_topic,
        "mono_ref_biased": mono_ref_biased,
        "mono_minus_recv_median_s": None if c_mono is None else round(c_mono, 6),
        "joint_states_stamp_minus_recv_median_s": None if c_sys_js is None else round(c_sys_js, 6),
        "motor_command_stamp_minus_recv_median_s": None if c_sys_mc is None else round(c_sys_mc, 6),
        # 漂移自检：该钟 − PC 收钟的速率（原生模式下 joint_states stamp 本身就是单调钟）
        "mono_vs_pc_drift": _thirds_drift_ms_per_min(mono_drift_samples),
        "joint_states_stamp_vs_pc_drift": _thirds_drift_ms_per_min(state_diffs),
        "ht_anchor_residual_ms": (
            None if not ht_residuals
            else round(statistics.median(ht_residuals) * 1000, 1)
        ),
        "ht_anchor_n": len(ht_residuals),
    }

    result = {
        "schema": "tracker_arm_bag_v3",
        "time_axis": "rk_mono_abs_s",
        "time_sources": {
            "states": (
                "header.stamp（RK 单调钟原生）" if state_domain == "rk_mono_native"
                else "header.stamp(RK系统钟) − median(stamp−recv) + median(bot_state.t−recv)（旧 bag 兼容）"
            ),
            "commands": (
                "header.stamp（RK 单调钟原生）" if command_domain == "rk_mono_native"
                else "同 states 换算（用 motor_command 自己的 stamp−recv 中位）"
            ),
            "events": "status 尾缀 t= / predict 的 ct（原生）；无时间的旧事件 = recv + median(bot_state.t−recv)",
            "note": "与 /predict_hit_pos ct/ht、rk_tracking payload t 同钟；报告端固定 scale=1，只用 RK→PC 常数偏移",
        },
        "clock_sync": clock_sync,
        "bag_dir": str(args.bag.resolve()),
        # 车型三件套：报告端按 car 复算 FK，不猜；car_source 留着事后查是谁定的。
        "car": car,
        "car_source": car_source,
        "kinematics_source_model": model.source_model,
        "fk_source": f"extract_arm_bag.fk({car})",
        "start_ns": start_ns,
        "duration_sec": round((end_ns - start_ns) / 1e9, 4),
        "joint_names": list(joint_names),
        "state_joint_names_raw": seen_state_names,
        "command_joint_names_raw": seen_command_names,
        "topics": [
            {"name": name, "type": type_by_topic.get(name, ""), "count": counts[name]}
            for name in sorted(counts)
        ],
        "states": states,
        "commands": commands,
        "events": events,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print(
        "arm json saved: %s (states=%d commands=%d events=%d duration=%.1fs)"
        % (args.output, len(states), len(commands), len(events), result["duration_sec"])
    )
    print("state joint names: %s" % (seen_state_names,))
    print("command joint names: %s" % (seen_command_names,))
    print("clock_sync: %s" % json.dumps(clock_sync, ensure_ascii=False))
    if legacy_stamps:
        print("NOTE: legacy epoch stamps converted (old firmware bag) — upgrade RK side for native mono stamps")
    d_mono = clock_sync["mono_vs_pc_drift"]
    if d_mono and abs(d_mono["ms_per_min"]) > 2.0:
        print("WARNING: RK mono vs PC drift %.2f ms/min > 2 — single-offset assumption needs review"
              % d_mono["ms_per_min"])
    if clock_sync["ht_anchor_residual_ms"] is not None and abs(clock_sync["ht_anchor_residual_ms"]) > 20.0:
        print("WARNING: |median(done - ht)| = %.1f ms — check clock chain or scheduler execution"
              % clock_sync["ht_anchor_residual_ms"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
