"""Tests for sub-second preemption watchdog upon user return."""

import subprocess
import sys
import time
import pytest
from hybrid_gpu_mesh.config import IdleScavengerConfig
from hybrid_gpu_mesh.scavenging.idle_detector import IdleDetector
from hybrid_gpu_mesh.scavenging.preemption import PreemptionWatchdog
from tests.test_idle_detector import MockAppleSiliconCollector


def test_preemption_watchdog_triggers_eviction():
    # Start in idle state (safe to scavenge)
    collector = MockAppleSiliconCollector(idle_sec=120.0, is_ac=True)
    cfg = IdleScavengerConfig(idle_threshold_sec=60.0, max_cpu_pct=100.0, max_gpu_pct=100.0)
    detector = IdleDetector(config=cfg, telemetry_collector=collector)

    preempted_calls = []

    def on_preempt(job_id: str, reason: str):
        preempted_calls.append((job_id, reason))

    watchdog = PreemptionWatchdog(
        idle_detector=detector,
        check_interval_sec=0.1,
        on_preempt_callback=on_preempt,
    )

    # Launch a mock long-running process (sleep 10)
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
    job_id = "test-job-99"
    watchdog.register_job(job_id, proc)

    watchdog.start()

    # Verify process is initially running
    time.sleep(0.15)
    assert proc.poll() is None
    assert len(preempted_calls) == 0

    # Simulate user moving mouse (idle time drops to 5 seconds)
    collector._mock_idle_sec = 5.0

    # Wait for watchdog to detect and preempt
    time.sleep(0.3)

    watchdog.stop()

    # Process must have been terminated immediately
    assert proc.poll() is not None
    assert len(preempted_calls) == 1
    assert preempted_calls[0][0] == job_id
    assert "User active" in preempted_calls[0][1]
    assert watchdog.preemption_count == 1
