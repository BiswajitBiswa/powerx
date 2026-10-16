"""Tests for Apple Silicon and NVIDIA DCGM telemetry collectors."""

import time
import pytest
from hybrid_gpu_mesh.telemetry.apple_silicon import AppleSiliconCollector, AppleSiliconMetrics
from hybrid_gpu_mesh.telemetry.nvidia_dcgm import NvidiaDcgmCollector, NvidiaGpuMetrics


def test_apple_silicon_collector_mock():
    collector = AppleSiliconCollector(mock_mode=True)
    assert collector.is_apple_silicon() is True

    metrics = collector.collect()
    assert isinstance(metrics, AppleSiliconMetrics)
    assert metrics.gpu_cores == 40
    assert metrics.model == "Apple M3 Max"
    assert metrics.vram_total_gb == 16.0
    assert metrics.user_idle_seconds == 120.0
    assert metrics.is_on_ac_power is True


def test_apple_silicon_collector_native():
    collector = AppleSiliconCollector(mock_mode=False)
    # If running on macOS arm64, it should detect real hardware
    if collector.is_apple_silicon():
        metrics = collector.collect()
        assert "Apple" in metrics.model or "M" in metrics.model
        assert metrics.gpu_cores > 0
        assert metrics.unified_memory_total_bytes > 0
        assert metrics.vram_total_gb > 0


def test_nvidia_architecture_detection():
    detect = NvidiaDcgmCollector.detect_architecture
    assert detect("NVIDIA H100 80GB HBM3") == "hopper"
    assert detect("NVIDIA H200") == "hopper"
    assert detect("NVIDIA B200 NVL") == "blackwell"
    assert detect("NVIDIA Blackwell Ultra") == "blackwell"
    assert detect("NVIDIA A100-SXM4-80GB") == "ampere"
    assert detect("NVIDIA RTX 4090") == "ada_lovelace"
    assert detect("Tesla T4") == "turing"


def test_nvidia_mock_metrics():
    gpu = NvidiaDcgmCollector.create_mock(name="NVIDIA H100 80GB HBM3", util_pct=60.0)
    assert gpu.name == "NVIDIA H100 80GB HBM3"
    assert gpu.architecture == "hopper"
    assert gpu.utilization_pct == 60.0
    assert gpu.vram_total_gb == 80.0
    assert gpu.vram_free_gb == 32.0
    assert gpu.temperature_c == 58.0
    assert gpu.power_watts == 350.0
