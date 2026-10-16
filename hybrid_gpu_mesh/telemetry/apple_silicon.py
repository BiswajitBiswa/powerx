"""Apple Silicon GPU, Unified Memory, and System Telemetry Collector."""

from __future__ import annotations
import platform
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class AppleSiliconMetrics:
    """Telemetry data captured from an Apple Silicon Mac."""
    model: str
    gpu_cores: int
    gpu_utilization_pct: float
    unified_memory_total_bytes: int
    unified_memory_used_bytes: int
    unified_memory_free_bytes: int
    user_idle_seconds: float
    is_on_ac_power: bool
    battery_pct: Optional[float]
    timestamp: float

    @property
    def vram_total_gb(self) -> float:
        return round(self.unified_memory_total_bytes / (1024 ** 3), 2)

    @property
    def vram_used_gb(self) -> float:
        return round(self.unified_memory_used_bytes / (1024 ** 3), 2)

    @property
    def vram_free_gb(self) -> float:
        return round(self.unified_memory_free_bytes / (1024 ** 3), 2)


class AppleSiliconCollector:
    """Reads real-time hardware telemetry on Apple Silicon macOS devices."""

    def __init__(self, mock_mode: bool = False):
        self.mock_mode = mock_mode or (platform.system() != "Darwin")
        self._cached_cores: Optional[int] = None
        self._cached_model: Optional[str] = None
        self._cached_total_mem: Optional[int] = None

    def is_apple_silicon(self) -> bool:
        """Check if current host is an Apple Silicon Mac."""
        if self.mock_mode:
            return True
        if platform.system() != "Darwin":
            return False
        return platform.machine() == "arm64"

    def get_total_unified_memory(self) -> int:
        """Fetch total unified RAM in bytes via sysctl."""
        if self._cached_total_mem is not None:
            return self._cached_total_mem
        if self.mock_mode:
            return 16 * 1024 * 1024 * 1024  # 16 GB mock default
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            self._cached_total_mem = int(out)
            return self._cached_total_mem
        except Exception:
            return 16 * 1024 * 1024 * 1024

    def get_chip_info(self) -> tuple[str, int]:
        """Fetch chip name (e.g. Apple M1 Pro) and GPU core count."""
        if self._cached_model is not None and self._cached_cores is not None:
            return self._cached_model, self._cached_cores

        if self.mock_mode:
            return "Apple M3 Max", 40

        model = "Apple Silicon"
        cores = 10
        try:
            # Query brand string
            brand = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
            if brand:
                model = brand
        except Exception:
            pass

        try:
            # Query ioreg for AGXAccelerator
            ioreg_out = subprocess.check_output(
                ["ioreg", "-r", "-c", "AGXAccelerator", "-d", "2"],
                text=True,
                stderr=subprocess.DEVNULL
            )
            core_match = re.search(r'"gpu-core-count"\s*=\s*(\d+)', ioreg_out)
            if core_match:
                cores = int(core_match.group(1))
            model_match = re.search(r'"model"\s*=\s*"([^"]+)"', ioreg_out)
            if model_match:
                model = model_match.group(1)
        except Exception:
            pass

        self._cached_model = model
        self._cached_cores = cores
        return model, cores

    def get_gpu_performance(self) -> tuple[float, int]:
        """Fetch current Device Utilization % and In-use VRAM bytes from ioreg."""
        if self.mock_mode:
            return 5.0, 2 * 1024 * 1024 * 1024

        gpu_util = 0.0
        used_bytes = 0
        try:
            ioreg_out = subprocess.check_output(
                ["ioreg", "-r", "-c", "AGXAccelerator", "-d", "2"],
                text=True,
                stderr=subprocess.DEVNULL
            )
            util_match = re.search(r'"Device Utilization %"\s*=\s*(\d+)', ioreg_out)
            if util_match:
                gpu_util = float(util_match.group(1))

            mem_match = re.search(r'"Alloc system memory"\s*=\s*(\d+)', ioreg_out)
            if mem_match:
                used_bytes = int(mem_match.group(1))
            else:
                mem_inuse = re.search(r'"In use system memory"\s*=\s*(\d+)', ioreg_out)
                if mem_inuse:
                    used_bytes = int(mem_inuse.group(1))
        except Exception:
            pass

        return gpu_util, used_bytes

    def get_user_idle_seconds(self) -> float:
        """Fetch seconds since last user keyboard/mouse activity via IOHIDSystem."""
        if self.mock_mode:
            return 120.0  # mock 2 minutes idle

        try:
            ioreg_out = subprocess.check_output(
                ["ioreg", "-c", "IOHIDSystem"],
                text=True,
                stderr=subprocess.DEVNULL
            )
            match = re.search(r'"HIDIdleTime"\s*=\s*(\d+)', ioreg_out)
            if match:
                nanoseconds = int(match.group(1))
                return max(0.0, nanoseconds / 1_000_000_000.0)
        except Exception:
            pass
        return 0.0

    def get_power_status(self) -> tuple[bool, Optional[float]]:
        """Fetch AC power status and battery percentage via pmset."""
        if self.mock_mode:
            return True, 100.0

        is_ac = True
        battery_pct = None
        try:
            pm_out = subprocess.check_output(["pmset", "-g", "batt"], text=True, stderr=subprocess.DEVNULL)
            if "Battery Power" in pm_out:
                is_ac = False
            elif "AC Power" in pm_out:
                is_ac = True

            bat_match = re.search(r'(\d+)%', pm_out)
            if bat_match:
                battery_pct = float(bat_match.group(1))
        except Exception:
            pass

        return is_ac, battery_pct

    def collect(self) -> AppleSiliconMetrics:
        """Capture complete Apple Silicon snapshot."""
        model, cores = self.get_chip_info()
        total_mem = self.get_total_unified_memory()
        gpu_util, used_vram = self.get_gpu_performance()
        idle_secs = self.get_user_idle_seconds()
        is_ac, batt_pct = self.get_power_status()

        # Ensure used doesn't exceed total
        used_vram = min(used_vram, total_mem)
        free_vram = max(0, total_mem - used_vram)

        return AppleSiliconMetrics(
            model=model,
            gpu_cores=cores,
            gpu_utilization_pct=gpu_util,
            unified_memory_total_bytes=total_mem,
            unified_memory_used_bytes=used_vram,
            unified_memory_free_bytes=free_vram,
            user_idle_seconds=idle_secs,
            is_on_ac_power=is_ac,
            battery_pct=batt_pct,
            timestamp=time.time(),
        )
