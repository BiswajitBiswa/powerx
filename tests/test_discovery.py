"""Tests for peer registry and beacon discovery."""

import json
import time
import pytest
from hybrid_gpu_mesh.discovery.beacon import BeaconService
from hybrid_gpu_mesh.discovery.registry import PeerNode, PeerRegistry


def test_peer_registry_ttl():
    registry = PeerRegistry(timeout_sec=0.5)

    peer1 = PeerNode(
        node_id="mac-01",
        cluster_name="test-cluster",
        host="192.168.1.10",
        port=9400,
        hardware_type="apple_silicon",
        device_name="Apple M1 Pro",
        vram_total_gb=16.0,
        vram_free_gb=12.0,
        gpu_util_pct=5.0,
        is_idle=True,
        is_on_ac_power=True,
        last_seen=time.time(),
    )

    registry.register_or_update(peer1)
    assert len(registry.list_peers(active_only=True)) == 1
    assert registry.aggregate_cluster_vram() == (16.0, 12.0)

    # Wait for TTL to expire
    time.sleep(0.6)
    assert len(registry.list_peers(active_only=True)) == 0


def test_beacon_packet_processing():
    registry = PeerRegistry(timeout_sec=10.0)
    service = BeaconService(
        node_id="local-mac",
        cluster_name="test-cluster",
        registry=registry,
    )

    # Valid packet from a peer
    payload = {
        "node_id": "remote-m3-max",
        "cluster_name": "test-cluster",
        "host": "192.168.1.15",
        "port": 9400,
        "hardware_type": "apple_silicon",
        "device_name": "Apple M3 Max",
        "vram_total_gb": 64.0,
        "vram_free_gb": 48.0,
        "gpu_util_pct": 2.0,
        "is_idle": True,
        "is_on_ac_power": True,
    }
    data = json.dumps(payload).encode("utf-8")
    service.process_incoming_packet(data, ("192.168.1.15", 58921))

    peers = registry.list_peers()
    assert len(peers) == 1
    assert peers[0].node_id == "remote-m3-max"
    assert peers[0].vram_total_gb == 64.0
    assert peers[0].is_idle is True

    # Packet from a different cluster should be ignored
    payload_other_cluster = dict(payload)
    payload_other_cluster["node_id"] = "rogue-node"
    payload_other_cluster["cluster_name"] = "different-cluster"
    service.process_incoming_packet(json.dumps(payload_other_cluster).encode("utf-8"), ("192.168.1.99", 58921))
    assert len(registry.list_peers()) == 1  # Unchanged!
