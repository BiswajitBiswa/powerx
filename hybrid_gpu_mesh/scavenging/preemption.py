"""Preemption manager to immediately vacate GPU when colleague returns to their Mac."""

from __future__ import annotations
import os
import signal
import subprocess
import threading
import time
from typing import Callable, Dict, Optional

from hybrid_gpu_mesh.scavenging.idle_detector import IdleDetector, IdleState


class PreemptionWatchdog:
    """Continuously monitors user activity while jobs are running and preempts immediately if user returns."""

    def __init__(
        self,
        idle_detector: IdleDetector,
        check_interval_sec: float = 0.5,
        on_preempt_callback: Optional[Callable[[str, str], None]] = None,
    ):
        self.idle_detector = idle_detector
        self.check_interval_sec = check_interval_sec
        self.on_preempt = on_preempt_callback
        self._running_jobs: Dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self.preemption_count = 0

    def register_job(self, job_id: str, process: subprocess.Popen) -> None:
        """Register a running subprocess to be protected by preemption watchdog."""
        with self._lock:
            self._running_jobs[job_id] = process

    def unregister_job(self, job_id: str) -> None:
        """Unregister a completed job."""
        with self._lock:
            self._running_jobs.pop(job_id, None)

    def start(self) -> None:
        """Start watchdog loop in background thread."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._loop, daemon=True)
        self._monitor_thread.start()

    def stop(self) -> None:
        """Stop watchdog loop."""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                has_jobs = len(self._running_jobs) > 0

            if has_jobs:
                evaluation = self.idle_detector.evaluate()
                if not evaluation.is_safe_to_scavenge:
                    # User returned or machine became busy! Preempt immediately!
                    self._trigger_preemption(evaluation.reason)

            time.sleep(self.check_interval_sec)

    def _trigger_preemption(self, reason: str) -> None:
        """Immediately terminate or pause running jobs and trigger migration callback."""
        jobs_to_preempt = {}
        with self._lock:
            jobs_to_preempt = dict(self._running_jobs)
            self._running_jobs.clear()

        for job_id, proc in jobs_to_preempt.items():
            self.preemption_count += 1
            try:
                # Terminate process gently first, or force kill if non-responsive
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=0.2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
            except Exception:
                pass

            if self.on_preempt:
                try:
                    self.on_preempt(job_id, reason)
                except Exception:
                    pass
