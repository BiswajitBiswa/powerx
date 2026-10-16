"""Adapter to dynamically join a central Ray cluster when Mac is idle."""

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


class RayOpportunisticWorker:
    """Joins a central Ray cluster when idle, gracefully disconnects when user is active."""

    def __init__(
        self,
        ray_head_address: str,
        idle_detector: IdleDetector,
        num_cpus: Optional[int] = None,
    ):
        self.ray_head_address = ray_head_address
        self.idle_detector = idle_detector
        self.num_cpus = num_cpus
        self._is_joined = False
        self._lock = threading.Lock()

    def join_cluster(self) -> bool:
        """Run `ray start --address=...` to donate compute."""
        with self._lock:
            if self._is_joined:
                return True

            if not shutil.which("ray"):
                logger.warning("Ray CLI not found in PATH.")
                return False

            cmd = ["ray", "start", f"--address={self.ray_head_address}", "--block=false"]
            if self.num_cpus:
                cmd.append(f"--num-cpus={self.num_cpus}")

            logger.info(f"Joining central Ray cluster at {self.ray_head_address}...")
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                self._is_joined = True
                return True
            logger.error(f"Failed to join Ray: {res.stderr}")
            return False

    def leave_cluster(self) -> None:
        """Run `ray stop` to instantly detach from cluster without hanging user's Mac."""
        with self._lock:
            if not self._is_joined:
                return

            logger.info("User returned! Detaching Ray worker node immediately...")
            subprocess.run(["ray", "stop", "--force"], capture_output=True)
            self._is_joined = False

    def create_watchdog(self) -> PreemptionWatchdog:
        """Returns a safety watchdog that unregisters Ray the second user returns."""
        def on_preempt(job_id: str, reason: str):
            self.leave_cluster()

        return PreemptionWatchdog(
            idle_detector=self.idle_detector,
            check_interval_sec=0.3,
            on_preempt_callback=on_preempt,
        )
