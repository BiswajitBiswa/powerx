"""Adapter to safely run Exo distributed inference on idle Macs."""

from __future__ import annotations
import logging
import shutil
import subprocess
import threading
import time
from typing import Optional

from hybrid_gpu_mesh.scavenging.idle_detector import IdleDetector
from hybrid_gpu_mesh.scavenging.preemption import PreemptionWatchdog

logger = logging.getLogger(__name__)


class ExoIdleRunner:
    """Manages an Exo worker instance, starting it ONLY when idle and evicting on user input."""

    def __init__(self, idle_detector: IdleDetector, exo_binary: str = "exo"):
        self.idle_detector = idle_detector
        self.exo_binary = exo_binary
        self._proc: Optional[subprocess.Popen] = None
        self._running = False
        self._lock = threading.Lock()

    def start_worker(self) -> bool:
        """Start local Exo peer."""
        with self._lock:
            if self._proc and self._proc.poll() is None:
                return True  # Already running

            if not shutil.which(self.exo_binary):
                logger.warning(f"Exo binary '{self.exo_binary}' not found in PATH.")
                return False

            logger.info("Mac is idle! Spawning Exo distributed inference worker...")
            self._proc = subprocess.Popen(
                [self.exo_binary],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True

    def stop_worker(self) -> None:
        """Instantly terminate Exo worker to return 100% resources to colleague."""
        with self._lock:
            if self._proc and self._proc.poll() is None:
                logger.info("User returned! Terminating Exo worker immediately (<300ms)...")
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=0.3)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                self._proc = None

    def create_watchdog(self) -> PreemptionWatchdog:
        """Create a safety watchdog configured for Exo."""
        def on_preempt(job_id: str, reason: str):
            self.stop_worker()

        watchdog = PreemptionWatchdog(
            idle_detector=self.idle_detector,
            check_interval_sec=0.3,
            on_preempt_callback=on_preempt,
        )
        return watchdog
