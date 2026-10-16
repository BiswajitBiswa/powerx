"""Job board on the head node: the queue that workers pull from.

A job is a shell command run inside a bundle (a tar.gz of code and data uploaded once and shared by every job
that names it). Any Mac running `cli worker` claims a job that fits its free memory, streams progress (steps,
metrics such as training loss, log lines) and uploads its result files. If the Mac's user comes back, the worker
uploads the job's checkpoint files and releases it; the next worker to claim it starts from them. A worker that
stops reporting for STALE_SECONDS loses its job back to the queue.

Jobs, bundles and results are kept under the state directory (MESH_STATE_DIR, default ~/.hybrid-gpu-mesh), so
the dashboard keeps its history across daemon restarts.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

STALE_SECONDS = 90.0
MAX_POINTS = 4000          # per metric series
MAX_LOG_LINES = 300

QUEUED, RUNNING, COMPLETED, FAILED, CANCELLED = "QUEUED", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"


def state_dir() -> Path:
    d = Path(os.environ.get("MESH_STATE_DIR", Path.home() / ".hybrid-gpu-mesh"))
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class BoardJob:
    command: str                                  # "{python}" is replaced by the worker's interpreter
    bundle: str = ""                              # sha256 of an uploaded bundle, or "" for none
    label: str = ""
    kind: str = "job"                             # e.g. "train", "eval"
    group: str = ""                               # jobs submitted together (a sweep) share a group
    params: Dict[str, Any] = field(default_factory=dict)
    vram_gb: float = 4.0
    total_steps: int = 0
    artifacts: List[str] = field(default_factory=list)    # globs, relative to the job directory
    priority: int = 10
    job_id: str = field(default_factory=lambda: f"job-{uuid.uuid4().hex[:8]}")
    state: str = QUEUED
    node_id: str = ""
    nodes: List[str] = field(default_factory=list)          # every node that ran it, in order
    submitted_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    heartbeat_at: Optional[float] = None
    attempts: int = 0
    preemptions: int = 0
    exit_code: Optional[int] = None
    error: str = ""
    step: int = 0
    metrics: Dict[str, List[List[float]]] = field(default_factory=dict)   # name -> [[step, value], ...]
    latest: Dict[str, float] = field(default_factory=dict)
    log: List[str] = field(default_factory=list)
    has_resume: bool = False
    has_result: bool = False
    cancel_requested: bool = False

    def public(self, log_lines: int = 40) -> Dict[str, Any]:
        d = asdict(self)
        d["log"] = self.log[-log_lines:] if log_lines else []
        end = self.finished_at or (time.time() if self.started_at else None)
        d["seconds"] = round(end - self.started_at, 1) if end and self.started_at else 0.0
        return d


class JobBoard:
    """Thread-safe queue of pull jobs with persistence."""

    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root) if root else state_dir()
        self.is_paused: bool = False
        for sub in ("bundles", "results", "resume"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._jobs: Dict[str, BoardJob] = {}
        self._dirty = False
        self._saved_at = 0.0
        self._load()

    # persistence
    def _path(self) -> Path:
        return self.root / "jobs.json"

    def _load(self) -> None:
        try:
            raw = json.loads(self._path().read_text())
        except (OSError, ValueError):
            return
        for d in raw:
            job = BoardJob(**{k: v for k, v in d.items() if k in BoardJob.__dataclass_fields__})
            if job.state == RUNNING:                      # the daemon restarted under it: run it again
                job.state, job.node_id = QUEUED, ""
                job.log.append(f"[mesh] {time.strftime('%H:%M:%S')} requeued after a head restart")
            self._jobs[job.job_id] = job

    def save(self, force: bool = False) -> None:
        with self._lock:
            if not (self._dirty or force) or (not force and time.time() - self._saved_at < 2.0):
                return
            data = json.dumps([asdict(j) for j in self._jobs.values()])
            self._dirty, self._saved_at = False, time.time()
        tmp = self._path().with_suffix(".tmp")
        tmp.write_text(data)
        tmp.replace(self._path())

    def _touch(self) -> None:
        self._dirty = True

    # bundles and files
    def put_bundle(self, data: bytes) -> str:
        sha = hashlib.sha256(data).hexdigest()
        p = self.root / "bundles" / f"{sha}.tar.gz"
        if not p.exists():
            p.write_bytes(data)
        return sha

    def bundle_path(self, sha: str) -> Optional[Path]:
        if not sha or not all(c in "0123456789abcdef" for c in sha):
            return None
        p = self.root / "bundles" / f"{sha}.tar.gz"
        return p if p.exists() else None

    def file_path(self, job_id: str, kind: str) -> Path:
        assert kind in ("results", "resume")
        return self.root / kind / f"{job_id}.tar.gz"

    # jobs
    def submit(self, spec: Dict[str, Any]) -> BoardJob:
        allowed = {"command", "bundle", "label", "kind", "group", "params", "vram_gb", "total_steps", "artifacts", "priority"}
        job = BoardJob(**{k: v for k, v in spec.items() if k in allowed})
        if not job.command:
            raise ValueError("a job needs a command")
        if job.bundle and not self.bundle_path(job.bundle):
            raise ValueError(f"bundle {job.bundle[:12]} has not been uploaded")
        with self._lock:
            self._jobs[job.job_id] = job
            self._touch()
        return job

    def get(self, job_id: str) -> Optional[BoardJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> List[BoardJob]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.submitted_at)

    def claim(self, node_id: str, vram_free_gb: float) -> Optional[BoardJob]:
        """The best queued job that fits this node: highest priority, then oldest."""
        if getattr(self, "is_paused", False):
            return None
        self.requeue_stale()
        with self._lock:
            fits = [j for j in self._jobs.values() if j.state == QUEUED and j.vram_gb <= vram_free_gb]
            if not fits:
                return None
            job = sorted(fits, key=lambda j: (-j.priority, j.submitted_at))[0]
            now = time.time()
            job.state, job.node_id, job.heartbeat_at = RUNNING, node_id, now
            job.started_at = job.started_at or now
            job.attempts += 1
            job.nodes.append(node_id)
            job.log.append(f"[mesh] {time.strftime('%H:%M:%S')} claimed by {node_id} (attempt {job.attempts})")
            self._touch()
            return job

    def progress(self, job_id: str, node_id: str, step: Optional[int], metrics: Dict[str, float],
                 lines: List[str]) -> Dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.node_id != node_id or job.state != RUNNING:
                return {"ok": False, "cancel": True}
            job.heartbeat_at = time.time()
            if step is not None:
                job.step = max(job.step, int(step))
            for name, value in (metrics or {}).items():
                try:
                    v = float(value)
                except (TypeError, ValueError):
                    continue
                series = job.metrics.setdefault(name, [])
                series.append([job.step if step is None else int(step), v])
                if len(series) > MAX_POINTS:
                    del series[: len(series) - MAX_POINTS]
                job.latest[name] = v
            job.log.extend(l[:500] for l in lines or [])
            if len(job.log) > MAX_LOG_LINES:
                del job.log[: len(job.log) - MAX_LOG_LINES]
            self._touch()
            return {"ok": True, "cancel": job.cancel_requested}

    def finish(self, job_id: str, node_id: str, exit_code: int, error: str = "") -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.node_id != node_id or job.state != RUNNING:
                return False
            job.exit_code, job.finished_at = int(exit_code), time.time()
            job.state = CANCELLED if job.cancel_requested else (COMPLETED if exit_code == 0 else FAILED)
            job.error = error[:2000]
            job.log.append(f"[mesh] {time.strftime('%H:%M:%S')} {job.state.lower()} on {node_id} (exit {exit_code})")
            self._touch()
            return True

    def release(self, job_id: str, node_id: str, reason: str) -> bool:
        """The worker gave the job back (its user returned); it goes back to the queue, ahead of new work."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.node_id != node_id or job.state != RUNNING:
                return False
            job.state, job.node_id = QUEUED, ""
            job.preemptions += 1
            job.priority += 5
            job.log.append(f"[mesh] {time.strftime('%H:%M:%S')} released by {node_id}: {reason}")
            self._touch()
            return True

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.state in (COMPLETED, FAILED, CANCELLED):
                return False
            if job.state == QUEUED:
                job.state, job.finished_at = CANCELLED, time.time()
            job.cancel_requested = True
            self._touch()
            return True

    def retry(self, job_id: str) -> bool:
        """Put a failed or cancelled job back in the queue. Its checkpoint files, if any, are kept, so it resumes;
        without them its curves start again."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.state not in (FAILED, CANCELLED):
                return False
            job.state, job.node_id, job.exit_code, job.error = QUEUED, "", None, ""
            job.finished_at, job.cancel_requested = None, False
            if not job.has_resume:
                job.step, job.metrics, job.latest, job.started_at = 0, {}, {}, None
            job.log.append(f"[mesh] {time.strftime('%H:%M:%S')} queued again")
            self._touch()
            return True

    def mark_file(self, job_id: str, kind: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                if kind == "resume":
                    job.has_resume = True
                else:
                    job.has_result = True
                self._touch()

    def requeue_stale(self) -> None:
        now = time.time()
        with self._lock:
            for job in self._jobs.values():
                if job.state == RUNNING and job.heartbeat_at and now - job.heartbeat_at > STALE_SECONDS:
                    job.log.append(f"[mesh] {time.strftime('%H:%M:%S')} {job.node_id} stopped reporting; requeued")
                    job.state, job.node_id = QUEUED, ""
                    self._touch()

    def summary(self) -> Dict[str, int]:
        with self._lock:
            out: Dict[str, int] = {}
            for j in self._jobs.values():
                out[j.state] = out.get(j.state, 0) + 1
            return out
