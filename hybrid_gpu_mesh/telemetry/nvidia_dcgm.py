"""NVIDIA DCGM and NVML Telemetry Collector for Cloud/Server GPUs."""

from __future__ import annotations
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class NvidiaGpuMetrics:
    """Telemetry data captured from an NVIDIA GPU."""
    gpu_index: int
    name: str
    architecture: str
    utilization_pct: float
    memory_total_bytes: int
    memory_used_bytes: int
    memory_free_bytes: int
    temperature_c: float
    power_watts: float
    timestamp: float

    @property
    def vram_total_gb(self) -> float:
        return round(self.memory_total_bytes / (1024 ** 3), 2)

    @property
    def vram_free_gb(self) -> float:
        return round(self.memory_free_bytes / (1024 ** 3), 2)


class NvidiaDcgmCollector:
    """Reads telemetry from NVIDIA GPUs via nvidia-smi or simulated DCGM feeds."""

    def __init__(self, mock_mode: bool = False):
        self.mock_mode = mock_mode or (shutil.which("nvidia-smi") is None)

    def is_nvidia_available(self) -> bool:
        """Check if NVIDIA hardware is detected."""
        if self.mock_mode:
            return False
        return shutil.which("nvidia-smi") is not None

    @staticmethod
    def detect_architecture(gpu_name: str) -> str:
        """Categorize GPU into architecture families matching NVIDIA GPU Usage Monitor."""
        name_lower = gpu_name.lower()
        if "b200" in name_lower or "b100" in name_lower or "blackwell" in name_lower:
            return "blackwell"
        if "h100" in name_lower or "h200" in name_lower or "hopper" in name_lower:
            return "hopper"
        if "l40" in name_lower or "rtx 40" in name_lower or "ada" in name_lower:
            return "ada_lovelace"
        if "a100" in name_lower or "a10" in name_lower or "ampere" in name_lower:
            return "ampere"
        if "v100" in name_lower or "volta" in name_lower:
            return "volta"
        if "t4" in name_lower or "turing" in name_lower:
            return "turing"
        return "generic_nvidia"

    def collect(self) -> List[NvidiaGpuMetrics]:
        """Collect metrics for all local NVIDIA GPUs."""
        if self.mock_mode or not self.is_nvidia_available():
            # Return empty if real collection requested on non-NVIDIA host
            return []

        metrics: List[NvidiaGpuMetrics] = []
        try:
            query = [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.total,memory.used,memory.free,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits"
            ]
            out = subprocess.check_output(query, text=True, stderr=subprocess.DEVNULL)
            for line in out.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 8:
                    idx = int(parts[0])
                    name = parts[1]
                    util = float(parts[2])
                    total_mb = float(parts[3])
                    used_mb = float(parts[4])
                    free_mb = float(parts[5])
                    temp = float(parts[6])
                    power = float(parts[7])

                    metrics.append(
                        NvidiaGpuMetrics(
                            gpu_index=idx,
                            name=name,
                            architecture=self.detect_architecture(name),
                            utilization_pct=util,
                            memory_total_bytes=int(total_mb * 1024 * 1024),
                            memory_used_bytes=int(used_mb * 1024 * 1024),
                            memory_free_bytes=int(free_mb * 1024 * 1024),
                            temperature_c=temp,
                            power_watts=power,
                            timestamp=time.time(),
                        )
                    )
        except Exception:
            pass

        return metrics

    @classmethod
    def create_mock(cls, name: str = "NVIDIA H100 80GB HBM3", util_pct: float = 45.0) -> NvidiaGpuMetrics:
        """Utility to generate realistic DCGM metrics for tests and simulations."""
        total_bytes = 80 * 1024 * 1024 * 1024
        used_bytes = int(total_bytes * (util_pct / 100.0))
        return NvidiaGpuMetrics(
            gpu_index=0,
            name=name,
            architecture=cls.detect_architecture(name),
            utilization_pct=util_pct,
            memory_total_bytes=total_bytes,
            memory_used_bytes=used_bytes,
            memory_free_bytes=total_bytes - used_bytes,
            temperature_c=58.0,
            power_watts=350.0,
            timestamp=time.time(),
        )
