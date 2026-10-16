"""Idle detection engine to safely scavenge idle compute nodes without disturbing active user sessions."""

from __future__ import annotations
import enum
import time
from dataclasses import dataclass
from typing import Optional

import psutil

from hybrid_gpu_mesh.config import IdleScavengerConfig
from hybrid_gpu_mesh.telemetry.apple_silicon import AppleSiliconCollector, AppleSiliconMetrics


class IdleState(str, enum.Enum):
    AVAILABLE = "AVAILABLE"          # Safe to scavenge!
    USER_ACTIVE = "USER_ACTIVE"      # User is actively typing or moving mouse
    ON_BATTERY = "ON_BATTERY"        # Unplugged from AC power
    CPU_BUSY = "CPU_BUSY"            # Local CPU load is too high (e.g. compiling)
    GPU_BUSY = "GPU_BUSY"            # Local GPU is already working


@dataclass
class IdleEvaluation:
    state: IdleState
    is_safe_to_scavenge: bool
    user_idle_seconds: float
    cpu_pct: float
    gpu_pct: float
    is_on_ac_power: bool
    reason: str
    evaluated_at: float


class IdleDetector:
    """Evaluates whether the host machine is currently idle and eligible to execute jobs."""

    def __init__(
        self,
        config: Optional[IdleScavengerConfig] = None,
        telemetry_collector: Optional[AppleSiliconCollector] = None,
    ):
        self.config = config or IdleScavengerConfig()
        self.collector = telemetry_collector or AppleSiliconCollector()
        self._last_evaluation: Optional[IdleEvaluation] = None

    def evaluate(self, force_cpu_check: Optional[float] = None) -> IdleEvaluation:
        """Run a full safety check against all thresholds."""
        metrics: AppleSiliconMetrics = self.collector.collect()

        cpu_pct = force_cpu_check if force_cpu_check is not None else psutil.cpu_percent(interval=None)
        gpu_pct = metrics.gpu_utilization_pct
        idle_secs = metrics.user_idle_seconds
        is_ac = metrics.is_on_ac_power
        now = time.time()

        # Check 1: AC Power
        if self.config.require_ac_power and not is_ac:
            eval_res = IdleEvaluation(
                state=IdleState.ON_BATTERY,
                is_safe_to_scavenge=False,
                user_idle_seconds=idle_secs,
                cpu_pct=cpu_pct,
                gpu_pct=gpu_pct,
                is_on_ac_power=is_ac,
                reason="Machine is running on battery power; scavenging paused to conserve battery.",
                evaluated_at=now,
            )
            self._last_evaluation = eval_res
            return eval_res

        # Check 2: User Keyboard/Mouse Activity
        if idle_secs < self.config.idle_threshold_sec:
            eval_res = IdleEvaluation(
                state=IdleState.USER_ACTIVE,
                is_safe_to_scavenge=False,
                user_idle_seconds=idle_secs,
                cpu_pct=cpu_pct,
                gpu_pct=gpu_pct,
                is_on_ac_power=is_ac,
                reason=f"User active (idle only {idle_secs:.1f}s < required {self.config.idle_threshold_sec:.1f}s).",
                evaluated_at=now,
            )
            self._last_evaluation = eval_res
            return eval_res

        # Check 3: CPU Threshold
        if cpu_pct > self.config.max_cpu_pct:
            eval_res = IdleEvaluation(
                state=IdleState.CPU_BUSY,
                is_safe_to_scavenge=False,
                user_idle_seconds=idle_secs,
                cpu_pct=cpu_pct,
                gpu_pct=gpu_pct,
                is_on_ac_power=is_ac,
                reason=f"CPU load ({cpu_pct:.1f}%) exceeds threshold ({self.config.max_cpu_pct:.1f}%).",
                evaluated_at=now,
            )
            self._last_evaluation = eval_res
            return eval_res

        # Check 4: GPU Threshold
        if gpu_pct > self.config.max_gpu_pct:
            eval_res = IdleEvaluation(
                state=IdleState.GPU_BUSY,
                is_safe_to_scavenge=False,
                user_idle_seconds=idle_secs,
                cpu_pct=cpu_pct,
                gpu_pct=gpu_pct,
                is_on_ac_power=is_ac,
                reason=f"GPU utilization ({gpu_pct:.1f}%) exceeds idle threshold ({self.config.max_gpu_pct:.1f}%).",
                evaluated_at=now,
            )
            self._last_evaluation = eval_res
            return eval_res

        # All safety conditions met!
        eval_res = IdleEvaluation(
            state=IdleState.AVAILABLE,
            is_safe_to_scavenge=True,
            user_idle_seconds=idle_secs,
            cpu_pct=cpu_pct,
            gpu_pct=gpu_pct,
            is_on_ac_power=is_ac,
            reason="Machine is idle, plugged into AC power, and ready for compute tasks.",
            evaluated_at=now,
        )
        self._last_evaluation = eval_res
        return eval_res

    @property
    def last_evaluation(self) -> Optional[IdleEvaluation]:
        return self._last_evaluation
