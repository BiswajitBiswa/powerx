"""Tests for IdleDetector multi-condition safety checks."""

import time
import pytest
from hybrid_gpu_mesh.config import IdleScavengerConfig
from hybrid_gpu_mesh.scavenging.idle_detector import IdleDetector, IdleState
from hybrid_gpu_mesh.telemetry.apple_silicon import AppleSiliconCollector, AppleSiliconMetrics


class MockAppleSiliconCollector(AppleSiliconCollector):
    def __init__(self, idle_sec=120.0, is_ac=True, gpu_util=5.0):
        super().__init__(mock_mode=True)
        self._mock_idle_sec = idle_sec
        self._mock_is_ac = is_ac
        self._mock_gpu_util = gpu_util

    def collect(self) -> AppleSiliconMetrics:
        return AppleSiliconMetrics(
            model="Apple M2 Ultra",
            gpu_cores=60,
            gpu_utilization_pct=self._mock_gpu_util,
            unified_memory_total_bytes=64 * 1024 * 1024 * 1024,
            unified_memory_used_bytes=10 * 1024 * 1024 * 1024,
            unified_memory_free_bytes=54 * 1024 * 1024 * 1024,
            user_idle_seconds=self._mock_idle_sec,
            is_on_ac_power=self._mock_is_ac,
            battery_pct=95.0,
            timestamp=time.time(),
        )


def test_idle_detector_safe_to_scavenge():
    cfg = IdleScavengerConfig(idle_threshold_sec=60.0, max_cpu_pct=50.0, max_gpu_pct=25.0)
    mock_coll = MockAppleSiliconCollector(idle_sec=120.0, is_ac=True, gpu_util=5.0)
    detector = IdleDetector(config=cfg, telemetry_collector=mock_coll)

    eval_res = detector.evaluate(force_cpu_check=15.0)
    assert eval_res.state == IdleState.AVAILABLE
    assert eval_res.is_safe_to_scavenge is True
    assert "ready for compute" in eval_res.reason


def test_idle_detector_user_active():
    cfg = IdleScavengerConfig(idle_threshold_sec=60.0)
    # User moved mouse 10 seconds ago
    mock_coll = MockAppleSiliconCollector(idle_sec=10.0, is_ac=True)
    detector = IdleDetector(config=cfg, telemetry_collector=mock_coll)

    eval_res = detector.evaluate(force_cpu_check=10.0)
    assert eval_res.state == IdleState.USER_ACTIVE
    assert eval_res.is_safe_to_scavenge is False
    assert "User active" in eval_res.reason


def test_idle_detector_on_battery():
    cfg = IdleScavengerConfig(require_ac_power=True)
    # Machine on battery
    mock_coll = MockAppleSiliconCollector(idle_sec=300.0, is_ac=False)
    detector = IdleDetector(config=cfg, telemetry_collector=mock_coll)

    eval_res = detector.evaluate(force_cpu_check=5.0)
    assert eval_res.state == IdleState.ON_BATTERY
    assert eval_res.is_safe_to_scavenge is False
    assert "battery power" in eval_res.reason


def test_idle_detector_cpu_busy():
    cfg = IdleScavengerConfig(max_cpu_pct=40.0)
    mock_coll = MockAppleSiliconCollector(idle_sec=300.0, is_ac=True)
    detector = IdleDetector(config=cfg, telemetry_collector=mock_coll)

    eval_res = detector.evaluate(force_cpu_check=85.0)
    assert eval_res.state == IdleState.CPU_BUSY
    assert eval_res.is_safe_to_scavenge is False
    assert "exceeds threshold" in eval_res.reason


def test_idle_detector_gpu_busy():
    cfg = IdleScavengerConfig(max_gpu_pct=20.0)
    mock_coll = MockAppleSiliconCollector(idle_sec=300.0, is_ac=True, gpu_util=75.0)
    detector = IdleDetector(config=cfg, telemetry_collector=mock_coll)

    eval_res = detector.evaluate(force_cpu_check=10.0)
    assert eval_res.state == IdleState.GPU_BUSY
    assert eval_res.is_safe_to_scavenge is False
    assert "GPU utilization" in eval_res.reason
