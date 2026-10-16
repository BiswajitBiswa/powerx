"""Cluster state aggregation and cost metrics tracker."""

from __future__ import annotations
import threading
import time
from typing import Any, Dict, List, Optional

from hybrid_gpu_mesh.discovery.registry import PeerNode, PeerRegistry
from hybrid_gpu_mesh.scheduler.models import Job, JobState, TargetTier


class ClusterState:
    """Maintains synchronized view of all nodes, jobs, and cost savings across the mesh."""

    # Hourly baseline rate used to calculate savings from scavenging idle Macs vs cloud (e.g. AWS g5.xlarge / A10G)
    CLOUD_EQUIVALENT_HOURLY_USD = 1.25

    def __init__(self, registry: PeerRegistry):
        self.registry = registry
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._cloud_nodes: Dict[str, PeerNode] = {}
        self.total_mac_compute_seconds: float = 0.0

    def add_cloud_node(self, node: PeerNode) -> None:
        """Register a remote cloud GPU node."""
        with self._lock:
            self._cloud_nodes[node.node_id] = node

    def remove_cloud_node(self, node_id: str) -> None:
        with self._lock:
            self._cloud_nodes.pop(node_id, None)

    def track_job(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.job_id] = job

    def get_job(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self, state: Optional[JobState] = None) -> List[Job]:
        with self._lock:
            if state:
                return [j for j in self._jobs.values() if j.state == state]
            return list(self._jobs.values())

    def record_job_completion(self, job: Job) -> None:
        """Record completed compute time to estimate cost savings."""
        with self._lock:
            self._jobs[job.job_id] = job
            if job.target_tier == TargetTier.LOCAL_MAC and job.duration_seconds > 0:
                self.total_mac_compute_seconds += job.duration_seconds

    def get_all_nodes(self) -> List[PeerNode]:
        """Combine discovered Wi-Fi Macs and registered cloud GPU nodes."""
        mac_peers = self.registry.list_peers(active_only=True)
        with self._lock:
            cloud_list = list(self._cloud_nodes.values())
        return mac_peers + cloud_list

    def get_summary(self) -> Dict[str, Any]:
        """Compute top-level summary for CLI and Prometheus."""
        all_nodes = self.get_all_nodes()
        total_vram_gb = sum(n.vram_total_gb for n in all_nodes)
        free_vram_gb = sum(n.vram_free_gb for n in all_nodes)

        idle_nodes = [n for n in all_nodes if n.is_idle and n.is_on_ac_power]

        running_jobs = self.list_jobs(JobState.RUNNING)
        queued_jobs = self.list_jobs(JobState.QUEUED)

        # Dollar savings calculation
        hours_saved = self.total_mac_compute_seconds / 3600.0
        cost_saved_usd = round(hours_saved * self.CLOUD_EQUIVALENT_HOURLY_USD, 2)

        return {
            "total_nodes": len(all_nodes),
            "idle_nodes": len(idle_nodes),
            "total_vram_bytes": int(total_vram_gb * 1024 * 1024 * 1024),
            "free_vram_bytes": int(free_vram_gb * 1024 * 1024 * 1024),
            "total_vram_gb": round(total_vram_gb, 2),
            "free_vram_gb": round(free_vram_gb, 2),
            "active_jobs": len(running_jobs),
            "queued_jobs": len(queued_jobs),
            "cost_saved_usd": cost_saved_usd,
        }
