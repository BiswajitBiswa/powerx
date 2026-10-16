"""End-to-End test suite simulating the complete hybrid Mac + Cloud GPU mesh."""

import sys
import time
import urllib.request
import pytest

from hybrid_gpu_mesh.config import CloudBurstConfig, IdleScavengerConfig, MeshConfig
from hybrid_gpu_mesh.discovery.registry import PeerNode, PeerRegistry
from hybrid_gpu_mesh.scavenging.idle_detector import IdleDetector
from hybrid_gpu_mesh.scavenging.preemption import PreemptionWatchdog
from hybrid_gpu_mesh.scheduler.cluster_state import ClusterState
from hybrid_gpu_mesh.scheduler.dispatcher import JobDispatcher
from hybrid_gpu_mesh.scheduler.models import Job, JobState, TargetTier
from hybrid_gpu_mesh.telemetry.exporter import MetricsRegistry, TelemetryServer
from tests.test_idle_detector import MockAppleSiliconCollector


def test_complete_e2e_hybrid_cluster_lifecycle():
    """Simulate complete lifecycle: discovery, scheduling, preemption, cloud burst, and metrics."""
    # 1. Setup Cluster Nodes
    peer_registry = PeerRegistry(timeout_sec=10.0)
    cluster_state = ClusterState(peer_registry)

    # Workstation Node 1: Primary Workstation (16 GB, IDLE on AC power)
    mac1 = PeerNode(
        node_id="node-01-primary",
        cluster_name="powerx-mesh",
        host="192.168.1.101",
        port=9400,
        hardware_type="apple_silicon",
        device_name="Unified Silicon Workstation",
        vram_total_gb=16.0,
        vram_free_gb=12.0,
        gpu_util_pct=0.0,
        is_idle=True,
        is_on_ac_power=True,
        last_seen=time.time(),
    )
    peer_registry.register_or_update(mac1)

    # Workstation Node 2: Secondary Workstation (64 GB, BUSY / interactive user typing)
    mac2 = PeerNode(
        node_id="node-02-secondary",
        cluster_name="powerx-mesh",
        host="192.168.1.102",
        port=9400,
        hardware_type="apple_silicon",
        device_name="Unified Silicon Workstation",
        vram_total_gb=64.0,
        vram_free_gb=50.0,
        gpu_util_pct=45.0,
        is_idle=False,  # Not idle!
        is_on_ac_power=False,
        last_seen=time.time(),
    )
    peer_registry.register_or_update(mac2)

    # Cloud Node 1: Remote NVIDIA H100 (80 GB)
    cloud_h100 = PeerNode(
        node_id="cloud-k8s-h100",
        cluster_name="powerx-mesh",
        host="35.200.40.10",
        port=9400,
        hardware_type="nvidia_cloud",
        device_name="NVIDIA H100 80GB HBM3",
        vram_total_gb=80.0,
        vram_free_gb=75.0,
        gpu_util_pct=5.0,
        is_idle=True,
        is_on_ac_power=True,
        last_seen=time.time(),
    )
    cluster_state.add_cloud_node(cloud_h100)

    # 2. Verify Aggregated Cluster Resources
    summary = cluster_state.get_summary()
    assert summary["total_nodes"] == 3
    assert summary["idle_nodes"] == 2  # mac1 and cloud_h100
    assert summary["total_vram_gb"] == 160.0  # 16 + 64 + 80 = 160 GB!
    assert summary["free_vram_gb"] == 137.0  # 12 + 50 + 75 = 137 GB!

    # 3. Setup Dispatcher and Preemption Watchdog on Mac Node 1
    mock_collector = MockAppleSiliconCollector(idle_sec=300.0, is_ac=True)
    idle_detector = IdleDetector(
        IdleScavengerConfig(idle_threshold_sec=60.0, max_cpu_pct=100.0, max_gpu_pct=100.0),
        mock_collector,
    )
    watchdog = PreemptionWatchdog(idle_detector, check_interval_sec=0.1)

    dispatcher = JobDispatcher(
        local_node_id="node-01-primary",
        cluster_state=cluster_state,
        cloud_config=CloudBurstConfig(enabled=True),
        watchdog=watchdog,
    )
    watchdog.on_preempt = dispatcher.handle_preemption
    watchdog.start()

    # 4. Schedule Multi-Tier Workloads
    # Job 1: 4GB ML experiment (should be scheduled locally on Node 1 for $0 cost!)
    job1 = Job(
        job_id="job-exp-local",
        command=[sys.executable, "-c", "import time; print('Training on Apple Silicon MPS'); time.sleep(0.5)"],
        required_vram_gb=4.0,
        requires_cuda=False,
    )
    node1, tier1 = dispatcher.find_placement_for_job(job1)
    assert node1.node_id == "node-01-primary"
    assert tier1 == TargetTier.LOCAL_MAC

    # Job 2: 20GB large LLM batch (exceeds 16GB M1 Pro, should automatically burst to Cloud H100)
    job2 = Job(
        job_id="job-large-llm",
        command=["python3", "serve_llm.py"],
        required_vram_gb=20.0,
        requires_cuda=False,
    )
    node2, tier2 = dispatcher.find_placement_for_job(job2)
    assert node2.node_id == "cloud-k8s-h100"
    assert tier2 == TargetTier.CLOUD_GPU

    # Job 3: FlashAttention-2 / CUDA-specific job (should burst to Cloud H100)
    job3 = Job(
        job_id="job-cuda-kernel",
        command=["python3", "triton_kernel.py"],
        required_vram_gb=8.0,
        requires_cuda=True,
    )
    node3, tier3 = dispatcher.find_placement_for_job(job3)
    assert node3.node_id == "cloud-k8s-h100"
    assert tier3 == TargetTier.CLOUD_GPU

    # 5. Execute Job 1 locally and test preemption when user returns
    # We execute a command that would sleep 5s, but user returns at 0.2s!
    job_preempt_test = Job(
        job_id="job-to-preempt",
        command=[sys.executable, "-c", "import time; time.sleep(5)"],
        required_vram_gb=2.0,
    )

    import threading
    t = threading.Thread(target=dispatcher.execute_local_job, args=(job_preempt_test,), daemon=True)
    t.start()

    time.sleep(0.2)
    assert job_preempt_test.state == JobState.RUNNING

    # Simulate colleague returning to their desk (moving mouse -> idle drops to 2 seconds)
    mock_collector._mock_idle_sec = 2.0
    time.sleep(0.3)

    # Watchdog should have triggered eviction and re-queued with elevated priority
    assert job_preempt_test.preemption_count == 1
    assert job_preempt_test.state == JobState.QUEUED

    # 6. Verify Telemetry & Prometheus Server
    metrics_registry = MetricsRegistry(node_id="node-01-primary", cluster_name="powerx-mesh")
    metrics_registry.update_mac(mock_collector.collect())
    metrics_registry.update_cluster_summary(cluster_state.get_summary())

    server = TelemetryServer(metrics_registry, host="127.0.0.1", port=19401)

    status_code, headers, body = server.handle_request("/metrics")
    assert status_code == 200
    metrics_body = body.decode("utf-8")
    assert "DCGM_FI_DEV_GPU_UTIL" in metrics_body
    assert "gpu_cluster_vram_total_bytes" in metrics_body
    assert "mac_user_idle_seconds" in metrics_body

    status_code_api, _, body_api = server.handle_request("/api/status")
    assert status_code_api == 200
    status_json = body_api.decode("utf-8")
    assert "powerx-mesh" in status_json

    watchdog.stop()
