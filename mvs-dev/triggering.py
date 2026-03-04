"""触发计划（software / master-slave）计算。

目标：
- 把“CLI 级别”的参数（serials、trigger_source、master_serial、soft_trigger_fps）规整成 `open_quad_capture` 所需的触发配置。

边界：
- 本模块不做任何 SDK 操作，仅负责纯配置计算。
- 触发计划属于 MVS 采集侧的通用能力，因此放在 `mvs` 包下，供 `mvs.apps.*` 与上层业务（如 tennis3d 在线模式）复用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class TriggerPlan:
    """根据 serial/master/trigger_source 推导出的触发计划。"""

    trigger_sources: list[str]
    soft_trigger_serials: list[str]
    enable_soft_trigger_fps: float

    def mapping_str(self, serials: Sequence[str]) -> str:
        """返回形如 "serial->source" 的映射字符串，用于日志。"""

        return ", ".join([f"{s}->{src}" for s, src in zip(list(serials), self.trigger_sources)])


def build_trigger_plan(
    *,
    serials: Sequence[str],
    trigger_source: str,
    master_serial: str,
    soft_trigger_fps: float | None = 0.0,
) -> TriggerPlan:
    """根据用户参数构造触发计划。

    规则（与现有 CLI 行为保持一致，并避免多处实现漂移）：
    - 主从触发：master_serial 非空 → master 固定 Software；slaves 用 trigger_source（例如 Line0）；只对 master 下发软触发。
    - 纯软件触发：master_serial 为空 且 trigger_source=Software → 所有相机 Software；对所有相机下发软触发。
    - 其它情况：不启用软触发。

    Raises:
        ValueError: serials 为空；或 master_serial 不在 serials 中。
    """

    serials_norm = [str(s).strip() for s in (serials or []) if str(s).strip()]
    if not serials_norm:
        raise ValueError("serials is empty")

    master_serial_norm = str(master_serial or "").strip()
    trigger_source_norm = str(trigger_source or "").strip()

    if master_serial_norm:
        if master_serial_norm not in serials_norm:
            raise ValueError(f"master_serial={master_serial_norm} 不在 serials 中")
        trigger_sources = [
            ("Software" if s == master_serial_norm else trigger_source_norm) for s in serials_norm
        ]
        soft_trigger_serials = [master_serial_norm]
    else:
        trigger_sources = [trigger_source_norm] * len(serials_norm)
        soft_trigger_serials = serials_norm if trigger_source_norm.lower() == "software" else []

    # 约定：soft_trigger_fps <= 0 表示不启用软触发循环。
    # 这里允许 soft_trigger_fps=None，便于纯函数测试/上层调用方省略参数。
    soft_fps = float(soft_trigger_fps or 0.0)
    enable = soft_fps if soft_trigger_serials else 0.0

    return TriggerPlan(
        trigger_sources=list(trigger_sources),
        soft_trigger_serials=list(soft_trigger_serials),
        enable_soft_trigger_fps=float(enable),
    )
