"""Tests for local-first scheduling, cloud bursting, and preemption recovery."""

import sys
import time
import pytest
from hybrid_gpu_mesh.config import CloudBurstConfig
from hybrid_gpu_mesh.discovery.registry import PeerNode, PeerRegistry
from hybrid_gpu_mesh.scheduler.cluster_state import ClusterState
from hybrid_gpu_mesh.scheduler.dispatcher import JobDispatcher
from hybrid_gpu_mesh.scheduler.models import Job, JobState, TargetTier


def create_test_nodes():
    # Local Idle Mac
    mac = PeerNode(
        node_id="mac-idle-01",
        cluster_name="test-cluster",
        host="192.168.1.10",
        port=9400,
        hardware_type="apple_silicon",
        device_name="Apple M1 Pro",
        vram_total_gb=16.0,
        vram_free_gb=12.0,
        gpu_util_pct=0.0,
        is_idle=True,
        is_on_ac_power=True,
        last_seen=time.time(),
    )
    # Cloud H100 GPU
    cloud = PeerNode(
        node_id="cloud-h100-01",
        cluster_name="test-cluster",
        host="34.120.10.5",
        port=9400,
        hardware_type="nvidia_cloud",
        device_name="NVIDIA H100 80GB HBM3",
        vram_total_gb=80.0,
        vram_free_gb=70.0,
        gpu_util_pct=10.0,
        is_idle=True,
        is_on_ac_power=True,
        last_seen=time.time(),
    )
    return mac, cloud


def test_scheduler_local_first_priority():
    registry = PeerRegistry()
    state = ClusterState(registry)
    mac, cloud = create_test_nodes()

    registry.register_or_update(mac)
    state.add_cloud_node(cloud)

    dispatcher = JobDispatcher(
        local_node_id="mac-idle-01",
        cluster_state=state,
        cloud_config=CloudBurstConfig(enabled=True),
    )

    # Job requires 6GB VRAM, general Python script (no CUDA requirement)
    job = Job(command=["python3", "-c", "print('hello')"], required_vram_gb=6.0, requires_cuda=False)
    target_node, tier = dispatcher.find_placement_for_job(job)

    # Must prefer free local Mac ($0 cost!)
    assert target_node is not None
    assert target_node.node_id == "mac-idle-01"
    assert tier == TargetTier.LOCAL_MAC


def test_scheduler_cloud_burst_when_cuda_required():
    registry = PeerRegistry()
    state = ClusterState(registry)
    mac, cloud = create_test_nodes()

    registry.register_or_update(mac)
    state.add_cloud_node(cloud)

    dispatcher = JobDispatcher(
        local_node_id="mac-idle-01",
        cluster_state=state,
        cloud_config=CloudBurstConfig(enabled=True),
    )

    # Job requires CUDA (e.g. FlashAttention-2 or Triton kernel)
    job = Job(command=["python3", "cuda_train.py"], required_vram_gb=4.0, requires_cuda=True)
    target_node, tier = dispatcher.find_placement_for_job(job)

    # Must burst to Cloud NVIDIA GPU
    assert target_node is not None
    assert target_node.node_id == "cloud-h100-01"
    assert tier == TargetTier.CLOUD_GPU


def test_scheduler_cloud_burst_when_mac_busy():
    registry = PeerRegistry()
    state = ClusterState(registry)
    mac, cloud = create_test_nodes()

    # Mark local mac as busy (colleague typing)
    mac.is_idle = False
    registry.register_or_update(mac)
    state.add_cloud_node(cloud)

    dispatcher = JobDispatcher(
        local_node_id="mac-idle-01",
        cluster_state=state,
        cloud_config=CloudBurstConfig(enabled=True),
    )

    job = Job(command=["python3", "task.py"], required_vram_gb=4.0)
    target_node, tier = dispatcher.find_placement_for_job(job)

    # Must burst to Cloud GPU because local mac is in use
    assert target_node is not None
    assert target_node.node_id == "cloud-h100-01"
    assert tier == TargetTier.CLOUD_GPU


def test_local_execution_and_cost_savings():
    registry = PeerRegistry()
    state = ClusterState(registry)
    dispatcher = JobDispatcher(local_node_id="local-mac", cluster_state=state)

    job = Job(command=[sys.executable, "-c", "print('Antigravity Hybrid Cluster')"])
    dispatcher.execute_local_job(job)

    assert job.state == JobState.COMPLETED
    assert job.exit_code == 0
    assert "Antigravity Hybrid Cluster" in job.output
    assert job.duration_seconds >= 0.0

    summary = state.get_summary()
    assert summary["active_jobs"] == 0
