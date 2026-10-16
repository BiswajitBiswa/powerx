"""Unified Command-Line Interface for PowerX."""

from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time

from hybrid_gpu_mesh.config import MeshConfig
from hybrid_gpu_mesh.discovery.beacon import BeaconService
from hybrid_gpu_mesh.discovery.registry import PeerRegistry
from hybrid_gpu_mesh.scavenging.idle_detector import IdleDetector
from hybrid_gpu_mesh.scavenging.preemption import PreemptionWatchdog
from hybrid_gpu_mesh.scheduler.cluster_state import ClusterState
from hybrid_gpu_mesh.scheduler.dispatcher import JobDispatcher
from hybrid_gpu_mesh.scheduler.models import Job
from hybrid_gpu_mesh.telemetry.apple_silicon import AppleSiliconCollector
from hybrid_gpu_mesh.telemetry.exporter import MetricsRegistry, TelemetryServer
from hybrid_gpu_mesh.telemetry.nvidia_dcgm import NvidiaDcgmCollector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("hybrid_gpu_mesh")


def cmd_status(args: argparse.Namespace) -> None:
    """Show current hardware, idle status, and cluster view."""
    from pathlib import Path
    cfg_file = Path(__file__).resolve().parent.parent / "config" / "cluster.yaml"
    config = MeshConfig.from_yaml(cfg_file) if cfg_file.exists() else MeshConfig()
    mac_collector = AppleSiliconCollector()
    nv_collector = NvidiaDcgmCollector()

    print("=" * 60)
    print(" ⚡ HYBRID GPU MESH: LOCAL NODE STATUS")
    print("=" * 60)
    print(f"Node ID:       {config.node_id}")
    print(f"Cluster Name:  {config.cluster_name}")

    if mac_collector.is_apple_silicon():
        metrics = mac_collector.collect()
        detector = IdleDetector(config.idle, mac_collector)
        evaluation = detector.evaluate()

        print("\n[Apple Silicon Hardware]")
        print(f"  Chip Model:     {metrics.model}")
        print(f"  GPU Cores:      {metrics.gpu_cores}")
        print(f"  GPU Util:       {metrics.gpu_utilization_pct:.1f}%")
        print(f"  Total VRAM:     {metrics.vram_total_gb:.2f} GB (Unified Memory)")
        print(f"  Free VRAM:      {metrics.vram_free_gb:.2f} GB")
        print(f"  Power Source:   {'AC Power (Plugged in)' if metrics.is_on_ac_power else 'Battery Power'}")
        if metrics.battery_pct is not None:
            print(f"  Battery Level:  {metrics.battery_pct:.0f}%")
        print(f"  User Idle Time: {metrics.user_idle_seconds:.1f} seconds")

        print("\n[Idle Scavenging Safety Status]")
        print(f"  State:          {evaluation.state.value}")
        print(f"  Eligible:       {'YES (Ready to run compute)' if evaluation.is_safe_to_scavenge else 'NO (Colleague active or busy)'}")
        print(f"  Reason:         {evaluation.reason}")

    elif nv_collector.is_nvidia_available():
        gpus = nv_collector.collect()
        print("\n[NVIDIA Cloud / Server Hardware]")
        for g in gpus:
            print(f"  GPU #{g.gpu_index}: {g.name} ({g.architecture.upper()})")
            print(f"    Util:         {g.utilization_pct:.1f}%")
            print(f"    VRAM Total:   {g.vram_total_gb:.2f} GB")
            print(f"    VRAM Free:    {g.vram_free_gb:.2f} GB")
            print(f"    Temp:         {g.temperature_c:.1f}°C")
            print(f"    Power:        {g.power_watts:.1f}W")
    else:
        print("\n[Host Platform]")
        print(f"  System:         {sys.platform}")
        print("  Notice: No physical Apple Silicon or NVIDIA GPU found (running in generic/emulation mode).")

    print("\n" + "=" * 60)


def cluster_token() -> str:
    """The cluster token: needed to submit, claim and report jobs. Made once, kept in the state directory (0600)."""
    import secrets
    from hybrid_gpu_mesh.scheduler.board import state_dir
    p = state_dir() / "token"
    if not p.exists():
        p.write_text(secrets.token_urlsafe(24))
        p.chmod(0o600)
    return p.read_text().strip()


def _head_url(head: str) -> str:
    head = head or "localhost:9400"
    return head if head.startswith("http") else f"http://{head}"


def cmd_daemon(args: argparse.Namespace) -> None:
    """Run the complete background daemon on this machine."""
    from pathlib import Path
    default_cfg = Path(__file__).resolve().parent.parent / "config" / "cluster.yaml"
    config_path = args.config or (str(default_cfg) if default_cfg.exists() else None)
    config = MeshConfig.from_yaml(config_path) if config_path else MeshConfig()

    # If --join was specified, add to static peers
    if getattr(args, "join", None):
        config.discovery.static_peers.append(args.join)

    logger.info(f"Starting PowerX Daemon on node: {config.node_id}")

    mac_collector = AppleSiliconCollector()
    nv_collector = NvidiaDcgmCollector()
    metrics_registry = MetricsRegistry(config.node_id, config.cluster_name)
    peer_registry = PeerRegistry(timeout_sec=config.discovery.peer_timeout_sec)
    cluster_state = ClusterState(peer_registry)

    idle_detector = IdleDetector(config.idle, mac_collector)
    watchdog = PreemptionWatchdog(idle_detector, check_interval_sec=0.5)

    dispatcher = JobDispatcher(
        local_node_id=config.node_id,
        cluster_state=cluster_state,
        cloud_config=config.cloud,
        watchdog=watchdog,
    )

    watchdog.on_preempt = dispatcher.handle_preemption
    watchdog.start()

    def get_beacon_payload() -> dict:
        is_safe = False
        gpu_util = 0.0
        vram_total = 16.0
        vram_free = 12.0
        dev_name = "Apple Silicon"
        hw_type = "apple_silicon"
        is_ac = True

        if mac_collector.is_apple_silicon():
            m = mac_collector.collect()
            ev = idle_detector.evaluate()
            is_safe = ev.is_safe_to_scavenge
            gpu_util = m.gpu_utilization_pct
            vram_total = m.vram_total_gb
            vram_free = m.vram_free_gb
            dev_name = m.model
            hw_type = "apple_silicon"
            is_ac = m.is_on_ac_power
        elif nv_collector.is_nvidia_available():
            gpus = nv_collector.collect()
            if gpus:
                g = gpus[0]
                is_safe = g.utilization_pct < 20.0
                gpu_util = g.utilization_pct
                vram_total = g.vram_total_gb
                vram_free = g.vram_free_gb
                dev_name = g.name
                hw_type = "nvidia_cloud"

        return {
            "node_id": config.node_id,
            "cluster_name": config.cluster_name,
            "hardware_type": hw_type,
            "device_name": dev_name,
            "vram_total_gb": vram_total,
            "vram_free_gb": vram_free,
            "gpu_util_pct": gpu_util,
            "is_idle": is_safe,
            "is_on_ac_power": is_ac,
        }

    beacon = BeaconService(
        node_id=config.node_id,
        cluster_name=config.cluster_name,
        registry=peer_registry,
        broadcast_port=config.discovery.broadcast_port,
        beacon_interval_sec=config.discovery.beacon_interval_sec,
        static_peers=config.discovery.static_peers,
        state_provider=get_beacon_payload,
    )
    beacon.start()

    # Telemetry HTTP Server (Prometheus exporter on port 9400, zero-python worker endpoint, job board, dashboard)
    from hybrid_gpu_mesh.scheduler.board import JobBoard
    board = JobBoard()
    token = cluster_token()
    server = TelemetryServer(
        registry=metrics_registry,
        host=config.discovery.bind_address,
        port=config.telemetry.prometheus_port,
        peer_registry=peer_registry,
        job_board=board,
        token=token,
    )
    server.start()
    logger.info(f"Dashboard at http://localhost:{config.telemetry.prometheus_port}/dashboard")
    if getattr(args, "worker", False):
        import threading
        from hybrid_gpu_mesh.runner import Head, Runner
        local_runner = Runner(Head(f"http://127.0.0.1:{config.telemetry.prometheus_port}", token), config.node_id,
                              python=args.python or sys.executable, own_machine=args.own,
                              collector=mac_collector, idle_detector=idle_detector)
        threading.Thread(target=local_runner.run_forever, daemon=True).start()
    logger.info(f"Prometheus Telemetry exporter active at http://localhost:{config.telemetry.prometheus_port}/metrics")
    logger.info(f"Zero-Python worker bootstrap ready at http://localhost:{config.telemetry.prometheus_port}/worker.sh")
    logger.info("Wi-Fi P2P discovery beacon active. Listening for cluster peers...")

    try:
        while True:
            # Refresh local telemetry
            if mac_collector.is_apple_silicon():
                m = mac_collector.collect()
                metrics_registry.update_mac(m)
            if nv_collector.is_nvidia_available():
                nv = nv_collector.collect()
                metrics_registry.update_nvidia(nv)

            # Update cluster aggregates
            summary = cluster_state.get_summary()
            metrics_registry.update_cluster_summary(summary)

            # Step dispatch queue; save the job board
            dispatcher.step_dispatch()
            board.requeue_stale()
            board.save()

            time.sleep(config.telemetry.scrape_interval_sec)
    except KeyboardInterrupt:
        logger.info("Shutting down daemon...")
    finally:
        board.save(force=True)
        server.stop()
        beacon.stop()
        watchdog.stop()


def cmd_register(args: argparse.Namespace) -> None:
    """Register current device as Node #1 / Founder Head Node."""
    import socket
    import subprocess
    import yaml
    from pathlib import Path

    cluster_name = args.cluster_name or "powerx-mesh"
    node_id = args.node_id or f"node-01-{socket.gethostname().split('.')[0].lower()}"
    role = "head" if args.as_first else (args.role or "hybrid")

    mac_coll = AppleSiliconCollector()
    nv_coll = NvidiaDcgmCollector()

    dev_name = "Generic Compute Node"
    vram_gb = 16.0
    cores = 8
    hw_type = "cpu"

    if mac_coll.is_apple_silicon():
        m = mac_coll.collect()
        dev_name = m.model
        vram_gb = m.vram_total_gb
        cores = m.gpu_cores
        hw_type = "apple_silicon"
    elif nv_coll.is_nvidia_available():
        gpus = nv_coll.collect()
        if gpus:
            dev_name = gpus[0].name
            vram_gb = gpus[0].vram_total_gb
            cores = len(gpus)
            hw_type = "nvidia_cloud"

    # Get local IP (check macOS en0 Wi-Fi first)
    local_ip = "127.0.0.1"
    try:
        en0_ip = subprocess.check_output(["ipconfig", "getifaddr", "en0"], text=True, stderr=subprocess.DEVNULL).strip()
        if en0_ip:
            local_ip = en0_ip
    except Exception:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
        except Exception:
            local_ip = "127.0.0.1"

    # Load and update cluster.yaml
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "cluster.yaml"
    cfg_data = {}
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg_data = yaml.safe_load(f) or {}

    cfg_data["node_id"] = node_id
    cfg_data["cluster_name"] = cluster_name
    cfg_data["node_role"] = role

    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_data, f, default_flow_style=False)

    # Save registered nodes ledger
    reg_path = Path(__file__).resolve().parent.parent / "config" / "registered_nodes.json"
    registered_nodes = {}
    if reg_path.exists():
        try:
            with open(reg_path, "r", encoding="utf-8") as f:
                registered_nodes = json.load(f)
        except Exception:
            pass

    node_entry = {
        "node_index": 1 if args.as_first else len(registered_nodes) + 1,
        "node_id": node_id,
        "role": role,
        "hostname": socket.gethostname(),
        "ip_address": local_ip,
        "hardware_type": hw_type,
        "device_model": dev_name,
        "gpu_cores": cores,
        "vram_total_gb": vram_gb,
        "registered_at": time.time(),
        "is_head_node": (role == "head"),
    }
    registered_nodes[node_id] = node_entry

    with open(reg_path, "w", encoding="utf-8") as f:
        json.dump(registered_nodes, f, indent=2)

    print("=" * 64)
    print(" 🎉 DEVICE REGISTERED AS NODE #1 (FOUNDER HEAD NODE)")
    print("=" * 64)
    print(f"Node Position:     #01 (Primary Cluster Coordinator)")
    print(f"Node ID:           {node_id}")
    print(f"Cluster Name:      {cluster_name}")
    print(f"Role:              {role.upper()} (Schedules jobs & computes)")
    print(f"Hardware:          {dev_name} ({cores} GPU Cores)")
    print(f"Pooled VRAM:       {vram_gb:.1f} GB Unified Memory")
    print(f"Local IP Address:  {local_ip}")
    print(f"Prometheus Port:   9400 (NVIDIA DCGM format)")
    print(f"Discovery Port:    58921 (UDP Beacon)")
    print("-" * 64)
    print("🚀 TEAM JOIN COMMAND (For worker nodes on local network):")
    print(f"   python3 -m hybrid_gpu_mesh.cli daemon --join {local_ip}:58921")
    print("=" * 64)


def cmd_submit(args: argparse.Namespace) -> None:
    """Submit a job to the head node's board; with --bundle DIR, DIR is packed and shipped with it."""
    import io
    import tarfile
    from pathlib import Path
    from hybrid_gpu_mesh.runner import Head
    head = Head(_head_url(args.head), args.token or os.environ.get("MESH_TOKEN") or cluster_token())
    bundle = ""
    if args.bundle:
        root = Path(args.bundle).resolve()
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for p in sorted(root.rglob("*")):
                if p.is_file() and not any(part in (".git", "__pycache__", ".venv") for part in p.parts):
                    tar.add(p, arcname=str(p.relative_to(root)))
        bundle = head.put_bytes("/api/bundles", buf.getvalue())["bundle"]
    command = " ".join(args.command[1:] if args.command[:1] == ["--"] else args.command)
    if args.cuda:
        print("Note: CUDA jobs are not run by Mac workers; this job waits for a CUDA worker.")
    resp = head.post("/api/jobs", {"command": command, "bundle": bundle, "label": args.label or command[:60],
                                   "vram_gb": args.vram, "artifacts": args.artifacts or [], "total_steps": args.steps})
    for jid in resp.get("jobs", []):
        print(f"Submitted {jid}: {command}")
    print(f"Watch it at {_head_url(args.head)}/dashboard")


def cmd_jobs(args: argparse.Namespace) -> None:
    import urllib.request
    with urllib.request.urlopen(_head_url(args.head) + "/api/jobs", timeout=10) as r:
        data = json.loads(r.read())
    for j in data["jobs"]:
        latest = ", ".join(f"{k} {v:g}" for k, v in sorted(j["latest"].items()))
        print(f"{j['job_id']}  {j['state']:<9}  {j['node_id'] or '-':<28}  step {j['step']:>5}  {j['label'][:40]:<40}  {latest}")


def cmd_worker(args: argparse.Namespace) -> None:
    from hybrid_gpu_mesh.runner import Head, Runner, default_node_id
    from pathlib import Path
    cfg_file = Path(__file__).resolve().parent.parent / "config" / "cluster.yaml"
    config = MeshConfig.from_yaml(cfg_file) if cfg_file.exists() else MeshConfig()
    token = args.token or os.environ.get("MESH_TOKEN")
    if not token:
        raise SystemExit("a worker needs the cluster token: --token (run `cli token` on the head node)")
    mac = AppleSiliconCollector()
    runner = Runner(Head(_head_url(args.head), token), args.node_id or default_node_id(), python=args.python or sys.executable,
                    own_machine=args.own, collector=mac, idle_detector=IdleDetector(config.idle, mac))
    try:
        runner.run_forever()
    except KeyboardInterrupt:
        runner.stop()


def cmd_control(args: argparse.Namespace) -> None:
    """cancel or retry jobs by id."""
    from hybrid_gpu_mesh.runner import Head
    head = Head(_head_url(args.head), args.token or os.environ.get("MESH_TOKEN") or cluster_token())
    for jid in args.job_ids:
        try:
            ok = head.post(f"/api/jobs/{jid}/{args.command}", {}).get("ok")
        except Exception as e:                            # 409: the job is not in a state that allows it
            ok = False
            print(f"{jid}: {e}")
        print(f"{jid}: {args.command} {'done' if ok else 'not possible'}")


def cmd_token(args: argparse.Namespace) -> None:
    print(cluster_token())


def main() -> None:
    parser = argparse.ArgumentParser(prog="hybrid-gpu-mesh", description="Pools idle Macs and Cloud GPUs into a unified AI cluster.")
    subparsers = parser.add_subparsers(dest="command")

    # Status
    subparsers.add_parser("status", help="Inspect local GPU, idle status, and cluster summary.")

    # Register
    reg_parser = subparsers.add_parser("register", help="Register this machine into the mesh.")
    reg_parser.add_argument("--as-first", action="store_true", default=True, help="Register as Node #1 / Founder Head Node.")
    reg_parser.add_argument("--node-id", type=str, default=None, help="Custom Node ID.")
    reg_parser.add_argument("--cluster-name", type=str, default="powerx-mesh", help="Cluster name.")
    reg_parser.add_argument("--role", type=str, default="head", choices=["head", "worker", "hybrid"], help="Node role.")

    # Daemon
    daemon_parser = subparsers.add_parser("daemon", help="Run mesh node agent with telemetry, discovery, and safety watchdog.")
    daemon_parser.add_argument("--config", "-c", type=str, default=None, help="Path to cluster.yaml configuration file.")
    daemon_parser.add_argument("--join", type=str, default=None, help="Head node IP:PORT to join.")
    daemon_parser.add_argument("--worker", action="store_true", help="Also run jobs from the board on this machine.")
    daemon_parser.add_argument("--own", action="store_true", help="With --worker: this is your own Mac; run jobs even while you use it (still only when plugged in).")
    daemon_parser.add_argument("--python", type=str, default=None, help="Interpreter that replaces {python} in job commands.")

    # Submit
    submit_parser = subparsers.add_parser("submit", help="Submit a job to the head node's board.")
    submit_parser.add_argument("--head", type=str, default="localhost:9400", help="Head node HOST:PORT.")
    submit_parser.add_argument("--token", type=str, default=None, help="Cluster token (default: MESH_TOKEN, or this machine's).")
    submit_parser.add_argument("--vram", type=float, default=4.0, help="Required VRAM in GB.")
    submit_parser.add_argument("--cuda", action="store_true", help="Force task to run on NVIDIA CUDA hardware.")
    submit_parser.add_argument("--bundle", type=str, default=None, help="Directory packed and shipped with the job.")
    submit_parser.add_argument("--artifacts", nargs="*", default=None, help="Globs of result files to bring back.")
    submit_parser.add_argument("--label", type=str, default=None)
    submit_parser.add_argument("--steps", type=int, default=0, help="Total steps, for the progress bar.")
    submit_parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to execute ({python} = the worker's interpreter).")

    jobs_parser = subparsers.add_parser("jobs", help="List the jobs on the head node's board.")
    jobs_parser.add_argument("--head", type=str, default="localhost:9400")

    worker_parser = subparsers.add_parser("worker", help="Run the head node's jobs on this Mac.")
    worker_parser.add_argument("--head", type=str, required=True, help="Head node HOST:PORT.")
    worker_parser.add_argument("--token", type=str, default=None, help="Cluster token (or MESH_TOKEN).")
    worker_parser.add_argument("--python", type=str, default=None, help="Interpreter that replaces {python} in job commands.")
    worker_parser.add_argument("--own", action="store_true", help="Your own Mac: run jobs even while you use it (still only when plugged in).")
    worker_parser.add_argument("--node-id", type=str, default=None)

    for name, text in (("cancel", "Stop jobs (a running job stops at its next report)."), ("retry", "Queue failed or cancelled jobs again.")):
        ctl = subparsers.add_parser(name, help=text)
        ctl.add_argument("job_ids", nargs="+")
        ctl.add_argument("--head", type=str, default="localhost:9400")
        ctl.add_argument("--token", type=str, default=None)

    subparsers.add_parser("token", help="Print the cluster token (head node).")

    args = parser.parse_args()
    if args.command == "status":
        cmd_status(args)
    elif args.command == "register":
        cmd_register(args)
    elif args.command == "daemon":
        cmd_daemon(args)
    elif args.command == "submit":
        cmd_submit(args)
    elif args.command == "jobs":
        cmd_jobs(args)
    elif args.command == "worker":
        cmd_worker(args)
    elif args.command in ("cancel", "retry"):
        cmd_control(args)
    elif args.command == "token":
        cmd_token(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
