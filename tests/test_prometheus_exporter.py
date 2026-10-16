"""Tests for Prometheus exporter conforming to NVIDIA GPU Usage Monitor conventions."""

import time
import urllib.request
import pytest
from hybrid_gpu_mesh.telemetry.apple_silicon import AppleSiliconMetrics
from hybrid_gpu_mesh.telemetry.exporter import MetricsRegistry, TelemetryServer
from hybrid_gpu_mesh.telemetry.nvidia_dcgm import NvidiaGpuMetrics


def test_metrics_registry_prometheus_text():
    registry = MetricsRegistry(node_id="test-node-01", cluster_name="test-cluster")

    # Add mock Apple Silicon
    mac_metrics = AppleSiliconMetrics(
        model="Apple M1 Pro",
        gpu_cores=14,
        gpu_utilization_pct=42.5,
        unified_memory_total_bytes=16 * 1024 * 1024 * 1024,
        unified_memory_used_bytes=6 * 1024 * 1024 * 1024,
        unified_memory_free_bytes=10 * 1024 * 1024 * 1024,
        user_idle_seconds=95.0,
        is_on_ac_power=True,
        battery_pct=100.0,
        timestamp=time.time(),
    )
    registry.update_mac(mac_metrics)

    # Add mock NVIDIA GPU
    nv_gpu = NvidiaGpuMetrics(
        gpu_index=0,
        name="NVIDIA H100 80GB HBM3",
        architecture="hopper",
        utilization_pct=88.0,
        memory_total_bytes=80 * 1024 * 1024 * 1024,
        memory_used_bytes=50 * 1024 * 1024 * 1024,
        memory_free_bytes=30 * 1024 * 1024 * 1024,
        temperature_c=65.0,
        power_watts=450.0,
        timestamp=time.time(),
    )
    registry.update_nvidia([nv_gpu])

    registry.update_cluster_summary({
        "total_nodes": 2,
        "idle_nodes": 1,
        "total_vram_bytes": 96 * 1024 * 1024 * 1024,
        "cost_saved_usd": 15.50,
    })

    text = registry.generate_prometheus_text()

    # Check NVIDIA DCGM metric signatures
    assert "# HELP DCGM_FI_DEV_GPU_UTIL" in text
    assert 'DCGM_FI_DEV_GPU_UTIL{gpu="0",device="NVIDIA H100 80GB HBM3",modelName="NVIDIA H100 80GB HBM3",architecture="hopper",node="test-node-01",cluster="test-cluster"} 88.0' in text
    # Check Apple Silicon DCGM-emulated metric
    assert 'DCGM_FI_DEV_GPU_UTIL{gpu="0",device="Apple M1 Pro",modelName="Apple M1 Pro",architecture="apple_silicon",node="test-node-01",cluster="test-cluster"} 42.5' in text
    # Check memory
    assert "DCGM_FI_DEV_FB_USED" in text
    assert "DCGM_FI_DEV_FB_FREE" in text
    # Check idle & cost metrics
    assert 'mac_user_idle_seconds{node="test-node-01",device="Apple M1 Pro"} 95.0' in text
    assert 'gpu_cluster_cost_saved_usd{cluster="test-cluster"} 15.50' in text


def test_telemetry_http_server():
    registry = MetricsRegistry(node_id="http-test-node", cluster_name="test-cluster")
    server = TelemetryServer(registry=registry, host="127.0.0.1", port=19400)

    # Test /metrics endpoint
    status, headers, body = server.handle_request("/metrics")
    assert status == 200
    assert "text/plain" in headers["Content-Type"]
    content = body.decode("utf-8")
    assert "DCGM_FI_DEV_GPU_UTIL" in content

    # Test /api/status endpoint
    status_api, headers_api, body_api = server.handle_request("/api/status")
    assert status_api == 200
    assert "application/json" in headers_api["Content-Type"]
    api_data = body_api.decode("utf-8")
    assert "http-test-node" in api_data

    # Test /worker.sh endpoint (Zero-python native shell script download)
    status_sh, headers_sh, body_sh = server.handle_request("/worker.sh")
    assert status_sh == 200
    assert "text/x-shellscript" in headers_sh["Content-Type"]
    assert b"Zero-Python Native macOS Worker" in body_sh

    # Test POST /api/heartbeat endpoint
    from hybrid_gpu_mesh.discovery.registry import PeerRegistry
    peer_reg = PeerRegistry()
    server.peer_registry = peer_reg

    heartbeat_payload = b'{"node_id": "mac-sarah-m3", "vram_total_gb": 64.0, "is_idle": true}'
    post_status, _, post_resp = server.handle_post("/api/heartbeat", heartbeat_payload, client_ip="192.168.1.180")
    assert post_status == 200
    assert b'"status": "ok"' in post_resp
    assert len(peer_reg.list_peers()) == 1
    assert peer_reg.list_peers()[0].node_id == "mac-sarah-m3"

    # Test 404
    status_404, _, _ = server.handle_request("/unknown")
    assert status_404 == 404
