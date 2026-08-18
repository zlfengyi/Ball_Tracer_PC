# -*- coding: utf-8 -*-
"""多假设三角化：场上有第二颗球时，按几何一致性挑组合而不是整台相机丢弃。

背景（0811 两场实测）：旧规则「一台相机恰好检出 1 个球才算数」有两个失效模式——
① 冒出第 2 个框 ⇒ 整台相机废掉（击球前窗口 17.9% 的帧因此凑不齐 2 台）；
② 各相机各锁一颗**不同**的球 ⇒ 配错，重投影 >15px 被拒（另 13.6%）。
两者都源于「每台相机独立决定用哪个框」。
"""

from __future__ import annotations

import json
import math

import cv2
import numpy as np
import pytest

from src.ball_detector import BallDetection
from src.ball_localizer import BallLocalizer


K = np.array([[1000.0, 0.0, 640.0], [0.0, 1000.0, 480.0], [0.0, 0.0, 1.0]])
CAMS = {
    "cam0": np.array([-3.0, -2.0, 3.0]),
    "cam1": np.array([3.0, -2.0, 3.0]),
    "cam2": np.array([-3.0, 6.0, 3.0]),
    "cam3": np.array([3.0, 6.0, 3.0]),
}
BALL_M = np.array([0.35, 2.40, 1.20])        # 真球
GHOST_M = np.array([-1.80, 3.10, 0.05])      # 地上那颗（0811 实测的典型位置/高度）


def _look_at(camera_m: np.ndarray, target_m: np.ndarray) -> np.ndarray:
    forward = target_m - camera_m
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.stack([right, down, forward])


@pytest.fixture(scope="module")
def localizer(tmp_path_factory):
    path = tmp_path_factory.mktemp("calib") / "calib.json"
    target = np.array([0.0, 2.0, 0.5])
    cameras = {}
    for sn, pos in CAMS.items():
        R = _look_at(pos, target)
        t = -R.dot(pos * 1000.0)
        cameras[sn] = {
            "K": K.tolist(),
            "D": [0.0, 0.0, 0.0, 0.0, 0.0],
            "R_world": R.tolist(),
            "t_world": t.reshape(3, 1).tolist(),
        }
    path.write_text(json.dumps({"reference_serial": "cam0", "cameras": cameras}),
                    encoding="utf-8")
    return BallLocalizer(str(path), detector=object())


def _project(sn: str, point_m: np.ndarray) -> tuple[float, float]:
    R = _look_at(CAMS[sn], np.array([0.0, 2.0, 0.5]))
    rvec, _ = cv2.Rodrigues(R)
    tvec = -R.dot(CAMS[sn] * 1000.0)
    uv, _ = cv2.projectPoints(point_m.reshape(1, 3) * 1000.0, rvec, tvec, K, np.zeros(5))
    return float(uv[0, 0, 0]), float(uv[0, 0, 1])


def _det(sn: str, point_m: np.ndarray, conf: float = 0.8) -> BallDetection:
    u, v = _project(sn, point_m)
    return BallDetection(x=u, y=v, confidence=conf,
                         x1=u - 8, y1=v - 8, x2=u + 8, y2=v + 8)


def _err_m(result, truth=BALL_M) -> float:
    got = np.array([result.x, result.y, result.z])
    return float(np.linalg.norm(got - truth))


def test_all_single_boxes_unchanged(localizer):
    """每台只有一个框时结果与旧路径一致（组合唯一，零额外开销分支）。"""
    cands = {sn: [_det(sn, BALL_M)] for sn in CAMS}
    got = localizer.select_and_triangulate(cands)
    assert got is not None
    assert set(got.cameras_used) == set(CAMS)
    assert _err_m(got) < 0.005


def test_ghost_box_no_longer_kills_the_camera(localizer):
    """两台相机各多框到一颗地面球：旧规则只剩 2 台，新规则四台全用上。"""
    cands = {sn: [_det(sn, BALL_M)] for sn in CAMS}
    for sn in ("cam1", "cam3"):
        cands[sn].append(_det(sn, GHOST_M, conf=0.9))   # 幽灵球置信度还更高

    old = localizer.select_and_triangulate(cands, require_exactly_one=True)
    assert old is not None
    assert set(old.cameras_used) == {"cam0", "cam2"}   # 旧行为：多框的两台被丢掉

    # 默认档（每台只取最高置信度那个框）：这里两台多框相机的最高置信度框都是
    # 幽灵球 ⇒ 池子变成 2v2 无解 ⇒ 兜底回旧规则，**至少不比旧行为差**
    got = localizer.select_and_triangulate(cands)
    assert got is not None
    assert set(got.cameras_used) == {"cam0", "cam2"}
    assert _err_m(got) < 0.01

    # 放开到每台 2 个候选：四台全部用上（但生产默认不开，见 _DEFAULT_MAX_PER_CAMERA）
    full = localizer.select_and_triangulate(cands, max_per_camera=2)
    assert full is not None
    assert set(full.cameras_used) == set(CAMS)
    assert _err_m(full) < 0.005


def test_picks_the_consistent_ball_not_the_ghost(localizer):
    """每台相机都同时看到真球和幽灵球：必须整体挑出自洽的那一组。"""
    cands = {sn: [_det(sn, GHOST_M, conf=0.9), _det(sn, BALL_M, conf=0.5)]
             for sn in CAMS}
    got = localizer.select_and_triangulate(cands, max_per_camera=2)
    assert got is not None
    assert len(got.cameras_used) == 4
    # 真球和幽灵球都是自洽解；取到哪一个由重投影决定，但绝不能是两者混搭的错解
    assert min(_err_m(got, BALL_M), _err_m(got, GHOST_M)) < 0.01


def test_two_versus_two_is_refused_not_coin_flipped(localizer):
    """2v2 错配（两台锁真球、两台锁幽灵球，每台各一个框）必须拒绝。

    这种局面无解：任取一边的 2 视图组合重投影都近 0，选谁纯属抛硬币。
    宁可不发，也不能凭"误差最小"发一个一半概率是幽灵球的点。
    """
    cands = {
        "cam0": [_det("cam0", BALL_M)],
        "cam1": [_det("cam1", BALL_M)],
        "cam2": [_det("cam2", GHOST_M)],
        "cam3": [_det("cam3", GHOST_M)],
    }
    bad = localizer.triangulate({sn: d[0] for sn, d in cands.items()})
    assert bad.reprojection_error > 15.0          # 混搭确实撑爆门限
    assert localizer.select_and_triangulate(cands) is None


def test_three_versus_one_drops_the_odd_camera(localizer):
    """3v1 错配可以救：三台自洽是真证据，把跑偏的那台踢掉。"""
    cands = {
        "cam0": [_det("cam0", BALL_M)],
        "cam1": [_det("cam1", BALL_M)],
        "cam2": [_det("cam2", BALL_M)],
        "cam3": [_det("cam3", GHOST_M)],
    }
    assert localizer.triangulate({sn: d[0] for sn, d in cands.items()}
                                 ).reprojection_error > 15.0

    got = localizer.select_and_triangulate(cands)
    assert got is not None
    assert set(got.cameras_used) == {"cam0", "cam1", "cam2"}
    assert _err_m(got) < 0.005


def test_two_cameras_only_still_works(localizer):
    """本来就只有 2 台看到球：没有"挑哪两台"的问题，照旧收。"""
    cands = {sn: [_det(sn, BALL_M)] for sn in ("cam0", "cam3")}
    got = localizer.select_and_triangulate(cands)
    assert got is not None
    assert set(got.cameras_used) == {"cam0", "cam3"}
    assert _err_m(got) < 0.01


def test_more_cameras_wins_over_smaller_reprojection(localizer):
    """分层的意义：2 视图解误差恒近 0，不分层就会稳定选中错解。"""
    cands = {sn: [_det(sn, BALL_M)] for sn in CAMS}
    cands["cam0"].append(_det("cam0", GHOST_M, conf=0.95))
    got = localizer.select_and_triangulate(cands, max_per_camera=2)
    assert got is not None
    assert len(got.cameras_used) == 4              # 没有退化成 2 台的"完美"解
    assert _err_m(got) < 0.005


def test_returns_none_when_nothing_is_consistent(localizer):
    """两台相机各看一颗球、没有任何组合自洽 ⇒ 不发布（不是硬凑一个）。"""
    cands = {
        "cam0": [_det("cam0", BALL_M)],
        "cam1": [_det("cam1", np.array([2.5, 5.5, 0.9]))],
    }
    got = localizer.select_and_triangulate(cands, max_reproj_error_px=1.0)
    assert got is None


def test_min_cameras_respected(localizer):
    cands = {"cam0": [_det("cam0", BALL_M)]}
    assert localizer.select_and_triangulate(cands) is None


def test_never_worse_than_the_old_rule(localizer):
    """严格超集：旧规则能出值的局面，新规则一定也出值。

    0811 回放里"旧出/新空"只有 3 帧（另一场 2 帧），成因就是多框相机的最高
    置信度框是幽灵球、把池子污染成无解；兜底回旧规则后归零。
    """
    import itertools
    combos = [
        {sn: [_det(sn, BALL_M)] for sn in CAMS},
        {"cam0": [_det("cam0", BALL_M)], "cam1": [_det("cam1", BALL_M)]},
    ]
    # 再加几个"部分相机多框且多的那个是幽灵球"的局面
    for extra in itertools.combinations(sorted(CAMS), 2):
        c = {sn: [_det(sn, BALL_M)] for sn in CAMS}
        for sn in extra:
            c[sn].append(_det(sn, GHOST_M, conf=0.99))
        combos.append(c)

    for cands in combos:
        old = localizer.select_and_triangulate(cands, require_exactly_one=True)
        new = localizer.select_and_triangulate(cands)
        if old is not None:
            assert new is not None, "旧规则出值而新规则空 = 倒退"
