"""Prometheus Telemetry Exporter compatible with NVIDIA GPU Usage Monitor and Apple Silicon."""

from __future__ import annotations
import json
import threading
import time
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

from hybrid_gpu_mesh.telemetry.apple_silicon import AppleSiliconMetrics
from hybrid_gpu_mesh.telemetry.nvidia_dcgm import NvidiaGpuMetrics


class MetricsRegistry:
    """Stores current telemetry state and formats Prometheus exposition text."""

    def __init__(self, node_id: str, cluster_name: str = "powerx-mesh"):
        self.node_id = node_id
        self.cluster_name = cluster_name
        self.is_paused: bool = False
        self._lock = threading.Lock()
        self.mac_metrics: Optional[AppleSiliconMetrics] = None
        self.nvidia_metrics: List[NvidiaGpuMetrics] = []
        self.cluster_summary: Dict[str, Any] = {
            "total_nodes": 1,
            "idle_nodes": 1,
            "total_vram_bytes": 0,
            "free_vram_bytes": 0,
            "active_jobs": 0,
            "queued_jobs": 0,
            "cost_saved_usd": 0.0,
        }

    def update_mac(self, metrics: AppleSiliconMetrics) -> None:
        with self._lock:
            self.mac_metrics = metrics

    def update_nvidia(self, metrics: List[NvidiaGpuMetrics]) -> None:
        with self._lock:
            self.nvidia_metrics = list(metrics)

    def update_cluster_summary(self, summary: Dict[str, Any]) -> None:
        with self._lock:
            self.cluster_summary.update(summary)

    def generate_prometheus_text(self) -> str:
        """Render Prometheus 0.0.4 text format."""
        lines: List[str] = []

        with self._lock:
            # 1. NVIDIA DCGM-compatible metrics for NVIDIA GPUs
            lines.append("# HELP DCGM_FI_DEV_GPU_UTIL GPU utilization (in %).")
            lines.append("# TYPE DCGM_FI_DEV_GPU_UTIL gauge")
            for gpu in self.nvidia_metrics:
                lines.append(
                    f'DCGM_FI_DEV_GPU_UTIL{{gpu="{gpu.gpu_index}",device="{gpu.name}",modelName="{gpu.name}",architecture="{gpu.architecture}",node="{self.node_id}",cluster="{self.cluster_name}"}} {gpu.utilization_pct:.1f}'
                )

            # If Apple Silicon is present, also expose it under DCGM format so NVIDIA Grafana dashboards work out of the box!
            if self.mac_metrics:
                lines.append(
                    f'DCGM_FI_DEV_GPU_UTIL{{gpu="0",device="{self.mac_metrics.model}",modelName="{self.mac_metrics.model}",architecture="apple_silicon",node="{self.node_id}",cluster="{self.cluster_name}"}} {self.mac_metrics.gpu_utilization_pct:.1f}'
                )

            lines.append("# HELP DCGM_FI_DEV_FB_USED Framebuffer (VRAM) memory used in bytes.")
            lines.append("# TYPE DCGM_FI_DEV_FB_USED gauge")
            for gpu in self.nvidia_metrics:
                lines.append(
                    f'DCGM_FI_DEV_FB_USED{{gpu="{gpu.gpu_index}",device="{gpu.name}",architecture="{gpu.architecture}",node="{self.node_id}",cluster="{self.cluster_name}"}} {gpu.memory_used_bytes}'
                )
            if self.mac_metrics:
                lines.append(
                    f'DCGM_FI_DEV_FB_USED{{gpu="0",device="{self.mac_metrics.model}",architecture="apple_silicon",node="{self.node_id}",cluster="{self.cluster_name}"}} {self.mac_metrics.unified_memory_used_bytes}'
                )

            lines.append("# HELP DCGM_FI_DEV_FB_FREE Framebuffer (VRAM) memory free in bytes.")
            lines.append("# TYPE DCGM_FI_DEV_FB_FREE gauge")
            for gpu in self.nvidia_metrics:
                lines.append(
                    f'DCGM_FI_DEV_FB_FREE{{gpu="{gpu.gpu_index}",device="{gpu.name}",architecture="{gpu.architecture}",node="{self.node_id}",cluster="{self.cluster_name}"}} {gpu.memory_free_bytes}'
                )
            if self.mac_metrics:
                lines.append(
                    f'DCGM_FI_DEV_FB_FREE{{gpu="0",device="{self.mac_metrics.model}",architecture="apple_silicon",node="{self.node_id}",cluster="{self.cluster_name}"}} {self.mac_metrics.unified_memory_free_bytes}'
                )

            # Temperature and Power
            lines.append("# HELP DCGM_FI_DEV_GPU_TEMP GPU temperature in degrees Celsius.")
            lines.append("# TYPE DCGM_FI_DEV_GPU_TEMP gauge")
            for gpu in self.nvidia_metrics:
                lines.append(
                    f'DCGM_FI_DEV_GPU_TEMP{{gpu="{gpu.gpu_index}",device="{gpu.name}",node="{self.node_id}"}} {gpu.temperature_c:.1f}'
                )

            # 2. Apple Silicon & Scavenging specific metrics
            if self.mac_metrics:
                lines.append("# HELP mac_user_idle_seconds Seconds since last user keyboard/mouse input.")
                lines.append("# TYPE mac_user_idle_seconds gauge")
                lines.append(
                    f'mac_user_idle_seconds{{node="{self.node_id}",device="{self.mac_metrics.model}"}} {self.mac_metrics.user_idle_seconds:.1f}'
                )

                lines.append("# HELP mac_is_on_ac_power 1 if connected to wall power, 0 if running on battery.")
                lines.append("# TYPE mac_is_on_ac_power gauge")
                lines.append(
                    f'mac_is_on_ac_power{{node="{self.node_id}"}} {1 if self.mac_metrics.is_on_ac_power else 0}'
                )

            # 3. Cluster Mesh Aggregations
            lines.append("# HELP gpu_cluster_nodes_total Total nodes currently joined to the cluster.")
            lines.append("# TYPE gpu_cluster_nodes_total gauge")
            lines.append(
                f'gpu_cluster_nodes_total{{cluster="{self.cluster_name}"}} {self.cluster_summary.get("total_nodes", 1)}'
            )

            lines.append("# HELP gpu_cluster_nodes_idle Nodes ready to scavenge without disturbing users.")
            lines.append("# TYPE gpu_cluster_nodes_idle gauge")
            lines.append(
                f'gpu_cluster_nodes_idle{{cluster="{self.cluster_name}"}} {self.cluster_summary.get("idle_nodes", 0)}'
            )

            lines.append("# HELP gpu_cluster_vram_total_bytes Combined VRAM across all Macs and Cloud GPUs.")
            lines.append("# TYPE gpu_cluster_vram_total_bytes gauge")
            lines.append(
                f'gpu_cluster_vram_total_bytes{{cluster="{self.cluster_name}"}} {self.cluster_summary.get("total_vram_bytes", 0)}'
            )

            lines.append("# HELP gpu_cluster_jobs_running Active running jobs across cluster.")
            lines.append("# TYPE gpu_cluster_jobs_running gauge")
            lines.append(
                f'gpu_cluster_jobs_running{{cluster="{self.cluster_name}"}} {self.cluster_summary.get("active_jobs", 0)}'
            )

            lines.append("# HELP gpu_cluster_cost_saved_usd Estimated cloud dollar savings from scavenged Mac compute.")
            lines.append("# TYPE gpu_cluster_cost_saved_usd counter")
            lines.append(
                f'gpu_cluster_cost_saved_usd{{cluster="{self.cluster_name}"}} {self.cluster_summary.get("cost_saved_usd", 0.0):.2f}'
            )

        return "\n".join(lines) + "\n"

    def get_json_status(self) -> Dict[str, Any]:
        with self._lock:
            mac_dict = None
            if self.mac_metrics:
                mac_dict = {
                    "model": self.mac_metrics.model,
                    "gpu_cores": self.mac_metrics.gpu_cores,
                    "gpu_utilization_pct": self.mac_metrics.gpu_utilization_pct,
                    "vram_total_gb": self.mac_metrics.vram_total_gb,
                    "vram_used_gb": self.mac_metrics.vram_used_gb,
                    "vram_free_gb": self.mac_metrics.vram_free_gb,
                    "user_idle_seconds": self.mac_metrics.user_idle_seconds,
                    "is_on_ac_power": self.mac_metrics.is_on_ac_power,
                    "battery_pct": self.mac_metrics.battery_pct,
                }
            return {
                "node_id": self.node_id,
                "cluster_name": self.cluster_name,
                "is_paused": self.is_paused,
                "cluster_state": "PAUSED" if self.is_paused else "ACTIVE",
                "apple_silicon": mac_dict,
                "nvidia_gpus": [vars(g) for g in self.nvidia_metrics],
                "cluster_summary": self.cluster_summary,
            }


MAX_UPLOAD_BYTES = 1024 * 1024 * 1024


def _json(status: int, data: Any) -> tuple[int, Dict[str, str], bytes]:
    body = json.dumps(data).encode("utf-8")
    return status, {"Content-Type": "application/json", "Content-Length": str(len(body))}, body


class TelemetryServer:
    """HTTP Server providing Prometheus scraping, JSON API endpoints, zero-python worker bootstrap, the job board
    API (when a JobBoard is given) and the dashboard.

    Reading (/dashboard, /api/jobs, /api/nodes, /metrics) is open on the LAN; everything that runs code or moves
    files (submitting, claiming, reporting, bundles, results) needs the cluster token in the X-Mesh-Token header."""

    def __init__(
        self,
        registry: MetricsRegistry,
        host: str = "0.0.0.0",
        port: int = 9400,
        peer_registry: Optional[Any] = None,
        job_board: Optional[Any] = None,
        token: Optional[str] = None,
    ):
        self.registry = registry
        self.host = host
        self.port = port
        self.peer_registry = peer_registry
        self.job_board = job_board
        self.token = token
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def _authorised(self, headers: Optional[Dict[str, str]]) -> bool:
        if not self.token:
            return False                                  # no token configured: job control is off
        got = (headers or {}).get("X-Mesh-Token") or (headers or {}).get("x-mesh-token") or ""
        return hmac.compare_digest(got.encode(), self.token.encode())

    def nodes_json(self) -> List[Dict[str, Any]]:
        """This node and every peer, with the job each is running."""
        running = {}
        if self.job_board:
            for j in self.job_board.list():
                if j.state == "RUNNING":
                    running[j.node_id] = {"job_id": j.job_id, "label": j.label}
        out = []
        st = self.registry.get_json_status()
        mac = st.get("apple_silicon")
        if mac:
            out.append({"node_id": self.registry.node_id, "role": "head", "device_name": mac["model"],
                        "vram_total_gb": mac["vram_total_gb"], "vram_free_gb": mac["vram_free_gb"],
                        "gpu_util_pct": mac["gpu_utilization_pct"], "is_on_ac_power": mac["is_on_ac_power"],
                        "user_idle_seconds": mac["user_idle_seconds"], "host": "this Mac"})
        if self.peer_registry:
            for p in self.peer_registry.list_peers(active_only=True):
                if p.node_id == self.registry.node_id:
                    continue
                out.append({"node_id": p.node_id, "role": "worker", "device_name": p.device_name,
                            "vram_total_gb": p.vram_total_gb, "vram_free_gb": p.vram_free_gb,
                            "gpu_util_pct": p.gpu_util_pct, "is_on_ac_power": p.is_on_ac_power,
                            "is_idle": p.is_idle, "host": p.host, "seconds_since_seen": round(time.time() - p.last_seen, 1)})
        for n in out:
            n["job"] = running.get(n["node_id"])
        return out

    def job_metrics_text(self) -> str:
        if not self.job_board:
            return ""
        lines = ["# HELP mesh_jobs Jobs on the board by state.", "# TYPE mesh_jobs gauge"]
        for state, n in sorted(self.job_board.summary().items()):
            lines.append(f'mesh_jobs{{cluster="{self.registry.cluster_name}",state="{state}"}} {n}')
        lines += ["# HELP mesh_job_step Latest step a job reported.", "# TYPE mesh_job_step gauge",
                  "# HELP mesh_job_metric Latest value of each metric a job reported.", "# TYPE mesh_job_metric gauge"]
        for j in self.job_board.list():
            if j.state != "RUNNING" and not j.latest:
                continue
            lab = f'job="{j.job_id}",label="{j.label}",group="{j.group}",node="{j.node_id}"'
            lines.append(f"mesh_job_step{{{lab}}} {j.step}")
            for name, v in sorted(j.latest.items()):
                lines.append(f'mesh_job_metric{{{lab},metric="{name}"}} {v}')
        return "\n".join(lines) + "\n"

    def handle_request(self, path: str, headers: Optional[Dict[str, str]] = None) -> tuple[int, Dict[str, str], bytes]:
        """Process an HTTP GET path and return (status_code, headers, body_bytes)."""
        parsed = urlparse(path)
        if parsed.path in ("/dashboard", "/dashboard/"):
            from pathlib import Path
            page = (Path(__file__).resolve().parent.parent / "dashboard.html").read_bytes()
            return 200, {"Content-Type": "text/html; charset=utf-8", "Content-Length": str(len(page))}, page
        if parsed.path == "/api/nodes":
            return _json(200, {"cluster": self.registry.cluster_name, "nodes": self.nodes_json(),
                               "jobs_enabled": bool(self.job_board and self.token)})
        if parsed.path in ("/api/cluster/pause", "/api/cluster/resume", "/api/cluster/restart"):
            if parsed.path == "/api/cluster/pause":
                self.registry.is_paused = True
                if self.job_board:
                    self.job_board.is_paused = True
                return _json(200, {"ok": True, "status": "paused", "message": "All PowerX nodes paused by operator"})
            else:
                self.registry.is_paused = False
                if self.job_board:
                    self.job_board.is_paused = False
                return _json(200, {"ok": True, "status": "active", "message": "All PowerX nodes restarted & active"})
        if self.job_board and parsed.path.startswith("/api/"):
            got = self._handle_board_get(parsed.path, headers)
            if got:
                return got
        if parsed.path in ("/metrics", "/"):
            content = (self.registry.generate_prometheus_text() + self.job_metrics_text()).encode("utf-8")
            headers = {
                "Content-Type": "text/plain; version=0.0.4; charset=utf-8",
                "Content-Length": str(len(content)),
            }
            return 200, headers, content
        elif parsed.path in ("/api/status", "/api/cluster", "/health"):
            data = json.dumps(self.registry.get_json_status(), indent=2).encode("utf-8")
            headers = {
                "Content-Type": "application/json",
                "Content-Length": str(len(data)),
            }
            return 200, headers, data
        elif parsed.path in ("/worker.sh", "/join"):
            # Serve the zero-python native shell script
            from pathlib import Path
            script_path = Path(__file__).resolve().parent.parent.parent / "scripts" / "worker.sh"
            if script_path.exists():
                content = script_path.read_bytes()
                headers = {
                    "Content-Type": "text/x-shellscript",
                    "Content-Length": str(len(content)),
                }
                return 200, headers, content
            return 404, {}, b"worker.sh not found"
        return 404, {}, b"Not Found"

    def _handle_board_get(self, path: str, headers) -> Optional[tuple[int, Dict[str, str], bytes]]:
        board = self.job_board
        if path == "/api/jobs":
            return _json(200, {"jobs": [j.public() for j in board.list()], "summary": board.summary()})
        parts = path.strip("/").split("/")                 # api, jobs|bundles, id, [file]
        if len(parts) == 3 and parts[1] == "jobs":
            job = board.get(parts[2])
            return _json(200, job.public(log_lines=300)) if job else _json(404, {"error": "no such job"})
        if len(parts) == 3 and parts[1] == "bundles":
            if not self._authorised(headers):
                return _json(401, {"error": "token required"})
            p = board.bundle_path(parts[2])
            if not p:
                return _json(404, {"error": "no such bundle"})
            data = p.read_bytes()
            return 200, {"Content-Type": "application/gzip", "Content-Length": str(len(data))}, data
        if len(parts) == 4 and parts[1] == "jobs" and parts[3] in ("results", "resume"):
            if not self._authorised(headers):
                return _json(401, {"error": "token required"})
            p = board.file_path(parts[2], parts[3])
            if not p.exists():
                return _json(404, {"error": f"no {parts[3]} file"})
            data = p.read_bytes()
            return 200, {"Content-Type": "application/gzip", "Content-Length": str(len(data))}, data
        return None

    def _handle_board_post(self, path: str, body: bytes, headers) -> Optional[tuple[int, Dict[str, str], bytes]]:
        board = self.job_board
        parts = path.strip("/").split("/")
        if not (len(parts) >= 2 and parts[0] == "api" and parts[1] in ("jobs", "bundles")):
            return None
        if not self._authorised(headers):
            return _json(401, {"error": "token required"})
        if path == "/api/bundles":
            return _json(200, {"bundle": board.put_bundle(body)})
        if len(parts) == 4 and parts[3] in ("results", "resume"):
            if not board.get(parts[2]):
                return _json(404, {"error": "no such job"})
            board.file_path(parts[2], parts[3]).write_bytes(body)
            board.mark_file(parts[2], parts[3])
            return _json(200, {"ok": True})
        try:
            msg = json.loads(body.decode("utf-8") or "{}")
        except ValueError:
            return _json(400, {"error": "body is not JSON"})
        if path == "/api/jobs":
            try:
                specs = msg if isinstance(msg, list) else [msg]
                jobs = [board.submit(spec) for spec in specs]
            except (TypeError, ValueError) as e:
                return _json(400, {"error": str(e)})
            board.save(force=True)
            return _json(200, {"jobs": [j.job_id for j in jobs]})
        if path == "/api/jobs/claim":
            job = board.claim(str(msg.get("node_id", "")), float(msg.get("vram_free_gb", 0)))
            return _json(200, {"job": job.public(log_lines=0) if job else None})
        if len(parts) == 4:
            job_id, action, node = parts[2], parts[3], str(msg.get("node_id", ""))
            if action == "progress":
                return _json(200, board.progress(job_id, node, msg.get("step"), msg.get("metrics") or {}, msg.get("lines") or []))
            if action == "finish":
                ok = board.finish(job_id, node, int(msg.get("exit_code", 1)), str(msg.get("error", "")))
                board.save(force=True)
                return _json(200 if ok else 409, {"ok": ok})
            if action == "release":
                ok = board.release(job_id, node, str(msg.get("reason", "released")))
                board.save(force=True)
                return _json(200 if ok else 409, {"ok": ok})
            if action == "cancel":
                return _json(200, {"ok": board.cancel(job_id)})
            if action == "retry":
                ok = board.retry(job_id)
                board.save(force=True)
                return _json(200 if ok else 409, {"ok": ok})
        return _json(404, {"error": "unknown job endpoint"})

    def handle_post(self, path: str, body: bytes, client_ip: str,
                    headers: Optional[Dict[str, str]] = None) -> tuple[int, Dict[str, str], bytes]:
        """Handle incoming POST requests: worker heartbeats, and the job board API."""
        parsed = urlparse(path)
        if parsed.path in ("/api/cluster/pause", "/api/cluster/resume", "/api/cluster/restart"):
            if parsed.path == "/api/cluster/pause":
                self.registry.is_paused = True
                if self.job_board:
                    self.job_board.is_paused = True
                return _json(200, {"ok": True, "status": "paused", "message": "All PowerX nodes paused by operator"})
            else:
                self.registry.is_paused = False
                if self.job_board:
                    self.job_board.is_paused = False
                return _json(200, {"ok": True, "status": "active", "message": "All PowerX nodes restarted & active"})
        if self.job_board:
            got = self._handle_board_post(parsed.path, body, headers)
            if got:
                return got
        if parsed.path == "/api/heartbeat":
            try:
                payload = json.loads(body.decode("utf-8"))
                if self.peer_registry:
                    from hybrid_gpu_mesh.discovery.registry import PeerNode
                    peer = PeerNode(
                        node_id=payload.get("node_id", f"mac-{client_ip}"),
                        cluster_name=self.registry.cluster_name,
                        host=client_ip,
                        port=9400,
                        hardware_type=payload.get("hardware_type", "apple_silicon"),
                        device_name=payload.get("device_name", "Apple Silicon Mac"),
                        vram_total_gb=float(payload.get("vram_total_gb", 16.0)),
                        vram_free_gb=float(payload.get("vram_free_gb", 12.0)),
                        gpu_util_pct=float(payload.get("gpu_util_pct", 0.0)),
                        is_idle=bool(payload.get("is_idle", False)),
                        is_on_ac_power=bool(payload.get("is_on_ac_power", True)),
                        last_seen=time.time(),
                    )
                    self.peer_registry.register_or_update(peer)
                resp = json.dumps({"status": "ok", "ack": payload.get("node_id")}).encode("utf-8")
                return 200, {"Content-Type": "application/json", "Content-Length": str(len(resp))}, resp
            except Exception as e:
                err = json.dumps({"error": str(e)}).encode("utf-8")
                return 400, {"Content-Type": "application/json", "Content-Length": str(len(err))}, err
        return 404, {}, b"Not Found"

    def start(self) -> None:
        if self._running:
            return

        server_ref = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                msg = format % args
                if any(k in msg for k in ("/progress", "/api/nodes", "GET /api/jobs", "/api/heartbeat", "/api/jobs/claim")):
                    return                                # polling, heartbeats and progress reports: too chatty
                print(f"[{time.strftime('%H:%M:%S')}] HTTP {self.client_address[0]}: {msg}", flush=True)

            def do_OPTIONS(self) -> None:
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "*")
                self.end_headers()

            def do_GET(self) -> None:
                status, headers, body = server_ref.handle_request(self.path, dict(self.headers.items()))
                self.send_response(status)
                self.send_header("Access-Control-Allow-Origin", "*")
                for k, v in headers.items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:
                content_len = int(self.headers.get("Content-Length", 0))
                if content_len > MAX_UPLOAD_BYTES:
                    self.send_response(413)
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    return
                body = self.rfile.read(content_len) if content_len > 0 else b""
                client_ip = self.client_address[0]
                status, headers, resp_body = server_ref.handle_post(self.path, body, client_ip, dict(self.headers.items()))
                self.send_response(status)
                self.send_header("Access-Control-Allow-Origin", "*")
                for k, v in headers.items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(resp_body)

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._server.daemon_threads = True
        self._running = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server and self._running:
            self._running = False
            self._server.shutdown()
            self._server.server_close()
            if self._thread:
                self._thread.join(timeout=2.0)
