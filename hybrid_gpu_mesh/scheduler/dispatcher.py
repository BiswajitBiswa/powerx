"""Workload dispatcher with local-first priority and cloud bursting."""

from __future__ import annotations
import logging
import subprocess
import threading
import time
from typing import Callable, List, Optional

from hybrid_gpu_mesh.config import CloudBurstConfig
from hybrid_gpu_mesh.discovery.registry import PeerNode
from hybrid_gpu_mesh.scavenging.preemption import PreemptionWatchdog
from hybrid_gpu_mesh.scheduler.cluster_state import ClusterState
from hybrid_gpu_mesh.scheduler.models import Job, JobState, TargetTier

logger = logging.getLogger(__name__)


class JobDispatcher:
    """Schedules and dispatches compute workloads across distributed edge and cloud GPUs."""

    def __init__(
        self,
        local_node_id: str,
        cluster_state: ClusterState,
        cloud_config: Optional[CloudBurstConfig] = None,
        watchdog: Optional[PreemptionWatchdog] = None,
    ):
        self.local_node_id = local_node_id
        self.cluster_state = cluster_state
        self.cloud_config = cloud_config or CloudBurstConfig()
        self.watchdog = watchdog
        self._lock = threading.Lock()
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def submit_job(self, job: Job) -> Job:
        """Submit a new job to the cluster queue."""
        job.state = JobState.QUEUED
        job.submitted_at = time.time()
        self.cluster_state.track_job(job)
        return job

    def find_placement_for_job(self, job: Job) -> tuple[Optional[PeerNode], Optional[TargetTier]]:
        """Determine whether to place on a local idle Mac or burst to Cloud."""
        all_nodes = self.cluster_state.get_all_nodes()

        # Step 1: Can we run on a free local Mac?
        # Only if job does NOT strictly require CUDA (Apple Silicon uses Metal/MPS)
        if not job.requires_cuda:
            candidate_macs = [
                n for n in all_nodes
                if n.hardware_type == "apple_silicon"
                and n.is_idle
                and n.is_on_ac_power
                and n.vram_free_gb >= job.required_vram_gb
            ]
            if candidate_macs:
                # Pick the Mac with the most free VRAM to balance memory pressure
                candidate_macs.sort(key=lambda n: n.vram_free_gb, reverse=True)
                return candidate_macs[0], TargetTier.LOCAL_MAC

        # Step 2: If no idle Mac is available or CUDA is required, check Cloud Bursting
        if self.cloud_config.enabled:
            candidate_cloud = [
                n for n in all_nodes
                if n.hardware_type in ("nvidia_cloud", "generic_cloud")
                and n.vram_free_gb >= job.required_vram_gb
            ]
            if candidate_cloud:
                candidate_cloud.sort(key=lambda n: n.vram_free_gb, reverse=True)
                return candidate_cloud[0], TargetTier.CLOUD_GPU

        return None, None

    def execute_local_job(self, job: Job) -> None:
        """Execute job locally protected by preemption watchdog."""
        job.state = JobState.RUNNING
        job.started_at = time.time()
        job.target_node_id = self.local_node_id
        job.target_tier = TargetTier.LOCAL_MAC
        self.cluster_state.track_job(job)

        try:
            cmd = job.command
            if isinstance(cmd, str):
                cmd = [cmd]

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Register with preemption watchdog to ensure instant eviction if user returns
            if self.watchdog:
                self.watchdog.register_job(job.job_id, proc)

            stdout, stderr = proc.communicate()

            if self.watchdog:
                self.watchdog.unregister_job(job.job_id)

            job.finished_at = time.time()
            job.exit_code = proc.returncode
            job.output = stdout
            job.error_message = stderr

            if proc.returncode == 0:
                job.state = JobState.COMPLETED
            elif job.state != JobState.PREEMPTED:
                job.state = JobState.FAILED

        except Exception as e:
            job.finished_at = time.time()
            job.state = JobState.FAILED
            job.error_message = str(e)
        finally:
            self.cluster_state.record_job_completion(job)

    def handle_preemption(self, job_id: str, reason: str) -> None:
        """Called when a running local job was preempted due to colleague returning."""
        job = self.cluster_state.get_job(job_id)
        if not job:
            return

        job.state = JobState.PREEMPTED
        job.preemption_count += 1
        job.error_message = f"Preempted: {reason}"
        logger.warning(f"Job {job_id} preempted ({reason}). Re-routing to cloud or peer.")

        # Re-queue the job with elevated priority
        job.state = JobState.QUEUED
        job.priority += 5

    def step_dispatch(self) -> int:
        """Evaluate pending queued jobs and dispatch what can be placed."""
        dispatched_count = 0
        queued_jobs = self.cluster_state.list_jobs(JobState.QUEUED)
        queued_jobs.sort(key=lambda j: j.priority, reverse=True)

        for job in queued_jobs:
            target_node, tier = self.find_placement_for_job(job)
            if target_node and tier:
                job.target_node_id = target_node.node_id
                job.target_tier = tier

                if target_node.node_id == self.local_node_id:
                    # Run on this machine in a worker thread
                    t = threading.Thread(target=self.execute_local_job, args=(job,), daemon=True)
                    t.start()
                    dispatched_count += 1
                else:
                    # Remote dispatch (in production: HTTP POST to peer's agent or cloud runner)
                    job.state = JobState.RUNNING
                    job.started_at = time.time()
                    dispatched_count += 1

        return dispatched_count
