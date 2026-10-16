"""Data models for jobs, nodes, and scheduler state."""

from __future__ import annotations
import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional


class JobState(str, enum.Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    PREEMPTED = "PREEMPTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class WorkloadType(str, enum.Enum):
    TASK_BOUND = "TASK_BOUND"      # Independent batch, parameter sweep, data chunk (Ray style)
    MODEL_BOUND = "MODEL_BOUND"    # Model sharding across devices, pipeline parallelism (Exo style)
    HYBRID = "HYBRID"              # Co-scheduled: model shard pinned in VRAM, processing batch task chunks in bubbles


class TargetTier(str, enum.Enum):
    LOCAL_MAC = "LOCAL_MAC"            # Apple Silicon Unified Memory (UMA)
    CLOUD_GPU = "CLOUD_GPU"            # NVIDIA GPU VRAM
    CLOUD_TPU = "CLOUD_TPU"            # Google TPU HBM with JAX Host Offload
    TIERED_OFFLOAD = "TIERED_OFFLOAD"  # Pooled Host CPU RAM + Accelerator (DeepSpeed / Pie)


@dataclass
class Job:
    """Represents a computational task to be scheduled across the mesh."""
    job_id: str = field(default_factory=lambda: f"job-{uuid.uuid4().hex[:8]}")
    command: List[str] = field(default_factory=list)
    required_vram_gb: float = 4.0
    workload_type: WorkloadType = WorkloadType.TASK_BOUND
    requires_cuda: bool = False
    priority: int = 10
    state: JobState = JobState.QUEUED
    target_node_id: Optional[str] = None
    target_tier: Optional[TargetTier] = None
    preemption_count: int = 0
    submitted_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    exit_code: Optional[int] = None
    output: str = ""
    error_message: str = ""

    @property
    def duration_seconds(self) -> float:
        if not self.started_at:
            return 0.0
        end = self.finished_at or time.time()
        return round(end - self.started_at, 2)
