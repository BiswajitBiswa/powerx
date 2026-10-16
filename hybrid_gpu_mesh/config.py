"""Configuration management for PowerX."""

from __future__ import annotations
import os
import socket
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class IdleScavengerConfig:
    """Settings controlling when a local node is considered idle and safe to use."""
    idle_threshold_sec: float = 60.0  # Min seconds of no keyboard/mouse before scavenging
    max_cpu_pct: float = 40.0         # Max background CPU% allowed to consider machine idle
    max_gpu_pct: float = 20.0         # Max existing GPU% allowed to consider machine idle
    require_ac_power: bool = True     # Never run heavy tasks when on battery
    preemption_latency_sec: float = 0.5  # Max time to pause/shed task when user activity detected
    check_interval_sec: float = 2.0    # Frequency of idle checks


@dataclass
class DiscoveryConfig:
    """Settings for P2P Wi-Fi/LAN discovery and peering."""
    broadcast_port: int = 58921
    beacon_interval_sec: float = 3.0
    peer_timeout_sec: float = 12.0
    bind_address: str = "0.0.0.0"
    static_peers: List[str] = field(default_factory=list)  # e.g. ["10.0.0.5:58921", "cloud.powerx.network:58921"]


@dataclass
class TelemetryConfig:
    """Settings for the Prometheus exporter and metrics collection."""
    prometheus_port: int = 9400  # Default port used by NVIDIA DCGM exporter
    api_port: int = 9401
    scrape_interval_sec: float = 2.0
    enable_dcgm_emulation: bool = True  # Format Apple Silicon metrics in DCGM format for Grafana


@dataclass
class CloudBurstConfig:
    """Settings for bursting workloads to Cloud GPUs (AWS/GCP/Lambda)."""
    enabled: bool = True
    cloud_provider: str = "generic_kubernetes"  # k8s, lambda, gcp, aws
    api_endpoint: Optional[str] = None
    api_key: Optional[str] = None
    max_cloud_gpus: int = 4
    cost_limit_usd_per_hour: float = 10.0


@dataclass
class MeshConfig:
    """Top-level configuration."""
    node_id: str = field(default_factory=lambda: f"node-{socket.gethostname()}-{uuid.uuid4().hex[:6]}")
    cluster_name: str = "powerx-mesh"
    node_role: str = "worker"  # "worker", "head", or "hybrid"
    idle: IdleScavengerConfig = field(default_factory=IdleScavengerConfig)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    cloud: CloudBurstConfig = field(default_factory=CloudBurstConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MeshConfig":
        p = Path(path)
        if not p.exists():
            return cls()
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MeshConfig":
        idle_data = data.get("idle", {})
        discovery_data = data.get("discovery", {})
        telemetry_data = data.get("telemetry", {})
        cloud_data = data.get("cloud", {})

        return cls(
            node_id=data.get("node_id", f"node-{socket.gethostname()}-{uuid.uuid4().hex[:6]}"),
            cluster_name=data.get("cluster_name", "powerx-mesh"),
            node_role=data.get("node_role", "worker"),
            idle=IdleScavengerConfig(**idle_data),
            discovery=DiscoveryConfig(**discovery_data),
            telemetry=TelemetryConfig(**telemetry_data),
            cloud=CloudBurstConfig(**cloud_data),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "cluster_name": self.cluster_name,
            "node_role": self.node_role,
            "idle": vars(self.idle),
            "discovery": vars(self.discovery),
            "telemetry": vars(self.telemetry),
            "cloud": vars(self.cloud),
        }
