"""UDP Broadcast and Unicast Beacon for zero-config Wi-Fi/LAN discovery."""

from __future__ import annotations
import json
import logging
import socket
import threading
import time
from typing import Callable, List, Optional

from hybrid_gpu_mesh.discovery.registry import PeerNode, PeerRegistry

logger = logging.getLogger(__name__)


class BeaconService:
    """Discovers peer Macs on Wi-Fi and connects to remote cloud GPU nodes."""

    def __init__(
        self,
        node_id: str,
        cluster_name: str,
        registry: PeerRegistry,
        broadcast_port: int = 58921,
        beacon_interval_sec: float = 3.0,
        static_peers: Optional[List[str]] = None,
        state_provider: Optional[Callable[[], dict]] = None,
    ):
        self.node_id = node_id
        self.cluster_name = cluster_name
        self.registry = registry
        self.broadcast_port = broadcast_port
        self.beacon_interval_sec = beacon_interval_sec
        self.static_peers = static_peers or []
        self.state_provider = state_provider

        self._running = False
        self._stop_event = threading.Event()
        self._listen_thread: Optional[threading.Thread] = None
        self._broadcast_thread: Optional[threading.Thread] = None
        self._listen_sock: Optional[socket.socket] = None

    def start(self) -> None:
        """Start broadcaster and listener threads."""
        if self._running:
            return

        self._running = True
        self._stop_event.clear()

        # Start listener socket
        self._listen_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):
            try:
                self._listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except Exception:
                pass
        self._listen_sock.settimeout(1.0)
        try:
            self._listen_sock.bind(("0.0.0.0", self.broadcast_port))
        except Exception as e:
            logger.warning(f"Failed to bind broadcast port {self.broadcast_port}: {e}")

        self._listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._broadcast_thread = threading.Thread(target=self._broadcast_loop, daemon=True)

        self._listen_thread.start()
        self._broadcast_thread.start()

    def stop(self) -> None:
        """Stop beacon service."""
        self._stop_event.set()
        self._running = False
        if self._listen_sock:
            try:
                self._listen_sock.close()
            except Exception:
                pass
        if self._broadcast_thread:
            self._broadcast_thread.join(timeout=1.0)
        if self._listen_thread:
            self._listen_thread.join(timeout=1.0)

    def _get_local_ip(self) -> str:
        """Get best guess of local network IP."""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"
        finally:
            s.close()

    def create_payload(self) -> dict:
        """Build heartbeat payload."""
        payload = {
            "node_id": self.node_id,
            "cluster_name": self.cluster_name,
            "host": self._get_local_ip(),
            "port": 9400,
            "hardware_type": "apple_silicon",
            "device_name": "Apple Silicon Mac",
            "vram_total_gb": 16.0,
            "vram_free_gb": 12.0,
            "gpu_util_pct": 0.0,
            "is_idle": True,
            "is_on_ac_power": True,
            "timestamp": time.time(),
        }

        if self.state_provider:
            try:
                custom = self.state_provider()
                payload.update(custom)
            except Exception:
                pass

        return payload

    def process_incoming_packet(self, data: bytes, addr: tuple[str, int]) -> None:
        """Parse incoming heartbeat packet and update registry."""
        try:
            payload = json.loads(data.decode("utf-8"))
            if payload.get("cluster_name") != self.cluster_name:
                return  # Ignore messages from different cluster

            sender_id = payload.get("node_id")
            if not sender_id or sender_id == self.node_id:
                return  # Don't register ourselves

            host_ip = addr[0] if payload.get("host") in ("127.0.0.1", "0.0.0.0") else payload.get("host", addr[0])

            peer = PeerNode(
                node_id=sender_id,
                cluster_name=payload.get("cluster_name", self.cluster_name),
                host=host_ip,
                port=payload.get("port", 9400),
                hardware_type=payload.get("hardware_type", "unknown"),
                device_name=payload.get("device_name", "Unknown GPU"),
                vram_total_gb=float(payload.get("vram_total_gb", 0.0)),
                vram_free_gb=float(payload.get("vram_free_gb", 0.0)),
                gpu_util_pct=float(payload.get("gpu_util_pct", 0.0)),
                is_idle=bool(payload.get("is_idle", False)),
                is_on_ac_power=bool(payload.get("is_on_ac_power", True)),
                last_seen=time.time(),
            )
            self.registry.register_or_update(peer)
        except Exception:
            pass

    def _listen_loop(self) -> None:
        while not self._stop_event.is_set():
            if not self._listen_sock:
                break
            try:
                data, addr = self._listen_sock.recvfrom(4096)
                self.process_incoming_packet(data, addr)
            except socket.timeout:
                continue
            except Exception:
                break

    def _broadcast_loop(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        while not self._stop_event.is_set():
            payload = self.create_payload()
            data = json.dumps(payload).encode("utf-8")

            # 1. Local Wi-Fi / LAN Broadcast
            try:
                sock.sendto(data, ("<broadcast>", self.broadcast_port))
                # Also loopback broadcast for local multi-instance testing
                sock.sendto(data, ("127.0.0.1", self.broadcast_port))
            except Exception:
                pass

            # 2. Static Unicast Peers (e.g. Cloud GPUs or Tailscale VPNs)
            for peer in self.static_peers:
                try:
                    parts = peer.split(":")
                    peer_host = parts[0]
                    peer_port = int(parts[1]) if len(parts) > 1 else self.broadcast_port
                    sock.sendto(data, (peer_host, peer_port))
                except Exception:
                    pass

            self._stop_event.wait(self.beacon_interval_sec)

        sock.close()
