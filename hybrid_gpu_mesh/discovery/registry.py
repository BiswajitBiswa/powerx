"""Registry tracking discovered cluster peer nodes and their health states."""

from __future__ import annotations
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class PeerNode:
    """Represents a discovered node in the mesh."""
    node_id: str
    cluster_name: str
    host: str
    port: int
    hardware_type: str   # "apple_silicon" or "nvidia_cloud"
    device_name: str
    vram_total_gb: float
    vram_free_gb: float
    gpu_util_pct: float
    is_idle: bool
    is_on_ac_power: bool
    last_seen: float

    @property
    def is_alive(self) -> bool:
        return (time.time() - self.last_seen) < 15.0


class PeerRegistry:
    """Thread-safe registry for cluster peers with automatic TTL eviction."""

    def __init__(self, timeout_sec: float = 12.0):
        self.timeout_sec = timeout_sec
        self._peers: Dict[str, PeerNode] = {}
        self._lock = threading.Lock()

    def register_or_update(self, peer: PeerNode) -> None:
        """Register or update a peer's heartbeat state."""
        with self._lock:
            self._peers[peer.node_id] = peer

    def remove(self, node_id: str) -> None:
        with self._lock:
            self._peers.pop(node_id, None)

    def get_peer(self, node_id: str) -> Optional[PeerNode]:
        with self._lock:
            return self._peers.get(node_id)

    def list_peers(self, active_only: bool = True) -> List[PeerNode]:
        """List current peers, pruning expired peers if active_only is True."""
        now = time.time()
        with self._lock:
            if active_only:
                # Filter out expired peers
                valid = [p for p in self._peers.values() if (now - p.last_seen) <= self.timeout_sec]
                return valid
            return list(self._peers.values())

    def get_idle_peers(self) -> List[PeerNode]:
        """Get all currently active peers that are idle and safe to scavenge."""
        return [p for p in self.list_peers(active_only=True) if p.is_idle and p.is_on_ac_power]

    def aggregate_cluster_vram(self) -> tuple[float, float]:
        """Returns total (total_vram_gb, free_vram_gb) across all active nodes."""
        peers = self.list_peers(active_only=True)
        total_vram = sum(p.vram_total_gb for p in peers)
        free_vram = sum(p.vram_free_gb for p in peers)
        return round(total_vram, 2), round(free_vram, 2)
