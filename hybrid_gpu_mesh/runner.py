"""Worker: runs the head node's jobs on this Mac.

  python3 -m hybrid_gpu_mesh.cli worker --head 192.168.1.144:9400 --token <cluster token> [--python PATH] [--own]

Every few seconds the worker sends a heartbeat. When the Mac is safe to use (idle, on AC power; or always with
--own, for the owner's own machine) it claims the job that best fits its free memory, then:

  1. downloads the job's bundle (and the checkpoint files of an earlier attempt, if any) into its own folder
     under ~/.hybrid-gpu-mesh/work/<job id>;
  2. runs the command there, with {python} replaced by --python (default: this interpreter);
  3. reads the command's output: lines `MESH {"step": 120, "metrics": {"train_loss": 1.2}}` report progress,
     and MLX training output ("Iter 25: Train loss 1.234 ...", "Iter 50: Val loss ...") is read as is;
  4. reports progress and log lines every 2 s; the head can answer "cancel";
  5. at the end, uploads the files matching the job's artifact globs and reports the exit code.

If the user comes back mid-job (and --own is off), or the Mac is unplugged, the worker stops the job within a
second, uploads its artifact files as the job's checkpoint and gives the job back; the next worker resumes from them.
"""

from __future__ import annotations

import fnmatch
import io
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ITER_TRAIN = re.compile(r"Iter (\d+): Train loss ([\d.]+)(?:.*?It/sec ([\d.]+))?(?:.*?Peak mem ([\d.]+) GB)?")
_ITER_VAL = re.compile(r"Iter (\d+): Val loss ([\d.]+)")


def parse_line(line: str) -> Tuple[Optional[int], Dict[str, float]]:
    """(step, metrics) from one line of job output; (None, {}) when the line reports nothing."""
    s = line.strip()
    if s.startswith("MESH "):
        try:
            d = json.loads(s[5:])
            return (int(d["step"]) if d.get("step") is not None else None), dict(d.get("metrics") or {})
        except (ValueError, TypeError, KeyError):
            return None, {}
    m = _ITER_VAL.search(s)
    if m:
        return int(m.group(1)), {"val_loss": float(m.group(2))}
    m = _ITER_TRAIN.search(s)
    if m:
        out = {"train_loss": float(m.group(2))}
        if m.group(3):
            out["it_per_sec"] = float(m.group(3))
        if m.group(4):
            out["peak_mem_gb"] = float(m.group(4))
        return int(m.group(1)), out
    return None, {}


def safe_extract(data: bytes, dest: Path) -> None:
    """Unpack a tar.gz without letting any member land outside dest."""
    dest = dest.resolve()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for m in tar.getmembers():
            target = (dest / m.name).resolve()
            if not str(target).startswith(str(dest) + os.sep) and target != dest:
                raise ValueError(f"unsafe path in bundle: {m.name}")
            if m.issym() or m.islnk():
                raise ValueError(f"links are not allowed in bundles: {m.name}")
        tar.extractall(dest)


def pack(root: Path, globs: List[str]) -> Optional[bytes]:
    """A tar.gz of the files under root matching any glob (relative paths), or None when nothing matches."""
    files = sorted({p for g in globs for p in root.glob(g) if p.is_file()})
    if not files:
        return None
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in files:
            tar.add(p, arcname=str(p.relative_to(root)))
    return buf.getvalue()


class Head:
    """The head node's HTTP API."""

    def __init__(self, url: str, token: str):
        self.url = url.rstrip("/")
        self.token = token

    def _req(self, method: str, path: str, body: Optional[bytes] = None, ctype: str = "application/json",
             timeout: float = 30.0) -> bytes:
        req = urllib.request.Request(self.url + path, data=body, method=method,
                                     headers={"X-Mesh-Token": self.token, "Content-Type": ctype})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()

    def post(self, path: str, msg: Dict[str, Any]) -> Dict[str, Any]:
        return json.loads(self._req("POST", path, json.dumps(msg).encode()) or b"{}")

    def get_bytes(self, path: str) -> bytes:
        return self._req("GET", path, timeout=600)

    def put_bytes(self, path: str, data: bytes) -> Dict[str, Any]:
        return json.loads(self._req("POST", path, data, "application/gzip", timeout=600) or b"{}")


class Runner:
    def __init__(self, head: Head, node_id: str, python: str = sys.executable, own_machine: bool = False,
                 workdir: Optional[Path] = None, poll_sec: float = 3.0, collector=None, idle_detector=None):
        self.head, self.node_id, self.python, self.own = head, node_id, python, own_machine
        self.workdir = workdir or Path(os.environ.get("MESH_STATE_DIR", Path.home() / ".hybrid-gpu-mesh")) / "work"
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.poll_sec = poll_sec
        self.collector, self.idle_detector = collector, idle_detector
        self._stop = threading.Event()

    # this Mac
    def _state(self) -> Dict[str, Any]:
        st = {"node_id": self.node_id, "hardware_type": "apple_silicon", "device_name": "Mac",
              "vram_total_gb": 16.0, "vram_free_gb": 12.0, "gpu_util_pct": 0.0, "is_idle": True, "is_on_ac_power": True}
        if self.collector and self.collector.is_apple_silicon():
            m = self.collector.collect()
            st.update(device_name=m.model, vram_total_gb=m.vram_total_gb, vram_free_gb=m.vram_free_gb,
                      gpu_util_pct=m.gpu_utilization_pct, is_on_ac_power=m.is_on_ac_power)
        st["is_idle"] = self._safe()
        return st

    def _safe(self) -> bool:
        """Whether this Mac may run a job now. Shared Macs: idle and plugged in. The owner's own Mac (--own): plugged
        in, whatever its user is doing; training on battery would drain it and the job would die with it."""
        if self.own:
            if self.collector and self.collector.is_apple_silicon():
                return self.collector.collect().is_on_ac_power
            return True
        if not self.idle_detector:
            return True
        return self.idle_detector.evaluate().is_safe_to_scavenge

    def _heartbeat(self, st: Dict[str, Any]) -> None:
        try:
            self.head.post("/api/heartbeat", st)
        except (urllib.error.URLError, OSError, ValueError):
            pass

    def stop(self) -> None:
        self._stop.set()

    def run_forever(self) -> None:
        print(f"[worker] {self.node_id} -> {self.head.url} ({'own machine: runs whenever it is plugged in' if self.own else 'runs only when idle and plugged in'})", flush=True)
        while not self._stop.is_set():
            st = self._state()
            self._heartbeat(st)
            job = None
            if st["is_idle"]:
                try:
                    job = self.head.post("/api/jobs/claim", {"node_id": self.node_id, "vram_free_gb": st["vram_free_gb"]}).get("job")
                except (urllib.error.URLError, OSError, ValueError) as e:
                    print(f"[worker] head not reachable: {e}", flush=True)
            if job:
                self.run_job(job)
            else:
                self._stop.wait(self.poll_sec)

    # one job
    def run_job(self, job: Dict[str, Any]) -> str:
        jid = job["job_id"]
        print(f"[worker] running {jid} {job.get('label') or ''}", flush=True)
        jdir = self.workdir / jid
        jdir.mkdir(parents=True, exist_ok=True)
        try:
            if job.get("bundle"):
                safe_extract(self.head.get_bytes(f"/api/bundles/{job['bundle']}"), jdir)
            if job.get("has_resume"):
                safe_extract(self.head.get_bytes(f"/api/jobs/{jid}/resume"), jdir)
        except (urllib.error.URLError, OSError, ValueError, tarfile.TarError) as e:
            self._finish(jid, 1, f"could not fetch the bundle: {e}")
            return "failed"
        env = dict(os.environ, MESH_JOB_ID=jid, MESH_PARAMS=json.dumps(job.get("params") or {}),
                   MESH_TOTAL_STEPS=str(job.get("total_steps") or 0), PYTHONUNBUFFERED="1")
        cmd = job["command"].replace("{python}", self.python)
        proc = subprocess.Popen(cmd, shell=True, cwd=jdir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, start_new_session=True)
        pending: List[str] = []                                           # log lines not yet sent
        reports: List[Tuple[Optional[int], Dict[str, float]]] = []          # (step, metrics) not yet sent
        lock = threading.Lock()

        def read() -> None:
            for line in proc.stdout:                       # type: ignore[union-attr]
                line = line.rstrip("\n")
                step, metrics = parse_line(line)
                with lock:
                    if not line.startswith("MESH "):
                        pending.append(line)
                    if step is not None or metrics:
                        reports.append((step, metrics))
        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        outcome = "done"
        last = 0.0
        while proc.poll() is None:
            if not self._safe():
                outcome = "preempted"
                self._kill(proc)
                break
            if time.time() - last >= 2.0:
                last = time.time()
                if self._send(jid, reports, pending, lock).get("cancel"):
                    outcome = "cancelled"
                    self._kill(proc)
                    break
            time.sleep(0.25)
        reader.join(timeout=5)
        self._send(jid, reports, pending, lock)
        globs = job.get("artifacts") or []
        if outcome == "preempted":
            data = pack(jdir, globs)
            if data:
                self.head.put_bytes(f"/api/jobs/{jid}/resume", data)
            why = "unplugged from power" if self.own else "this Mac's user came back or it was unplugged"
            self.head.post(f"/api/jobs/{jid}/release", {"node_id": self.node_id, "reason": why})
            print(f"[worker] {jid} released ({why})", flush=True)
            return outcome
        data = pack(jdir, globs)
        if data:
            try:
                self.head.put_bytes(f"/api/jobs/{jid}/results", data)
            except (urllib.error.URLError, OSError) as e:
                print(f"[worker] could not upload results of {jid}: {e}", flush=True)
        code = proc.returncode if proc.returncode is not None else 1
        self._finish(jid, code, "" if code == 0 else "\n".join(pending[-20:]))
        print(f"[worker] {jid} finished: {outcome}, exit {code}", flush=True)
        return outcome

    def _send(self, jid: str, reports: list, pending: List[str], lock: threading.Lock) -> Dict[str, Any]:
        with lock:
            batch, lines = list(reports), list(pending)
            reports.clear()
            pending.clear()
        try:
            resp: Dict[str, Any] = {}
            for step, metrics in batch or [(None, {})]:
                resp = self.head.post(f"/api/jobs/{jid}/progress", {"node_id": self.node_id, "step": step,
                                                                     "metrics": metrics, "lines": lines})
                lines = []
            return resp
        except (urllib.error.URLError, OSError, ValueError):
            with lock:                                      # keep them for the next report
                reports[:0] = batch
                pending[:0] = lines
            return {}

    def _finish(self, jid: str, code: int, error: str) -> None:
        try:
            self.head.post(f"/api/jobs/{jid}/finish", {"node_id": self.node_id, "exit_code": code, "error": error})
        except (urllib.error.URLError, OSError, ValueError) as e:
            print(f"[worker] could not report the end of {jid}: {e}", flush=True)

    @staticmethod
    def _kill(proc: subprocess.Popen) -> None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def default_node_id() -> str:
    return f"mac-{socket.gethostname().split('.')[0].lower()}"
