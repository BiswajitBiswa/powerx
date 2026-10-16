"""Job board, its HTTP API and the worker, end to end on this machine."""

import io
import json
import sys
import tarfile
import threading
import time

import pytest

from hybrid_gpu_mesh.runner import Head, Runner, pack, parse_line, safe_extract
from hybrid_gpu_mesh.scheduler import board as board_mod
from hybrid_gpu_mesh.scheduler.board import JobBoard
from hybrid_gpu_mesh.telemetry.exporter import MetricsRegistry, TelemetryServer

TOKEN = "test-token"


def _tar(files: dict) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_claim_fits_memory_and_priority(tmp_path):
    b = JobBoard(tmp_path)
    small = b.submit({"command": "true", "vram_gb": 2, "label": "small"})
    big = b.submit({"command": "true", "vram_gb": 12, "label": "big", "priority": 50})
    assert b.claim("mac-a", vram_free_gb=4).job_id == small.job_id      # the big one does not fit
    assert b.claim("mac-b", vram_free_gb=16).job_id == big.job_id
    assert b.claim("mac-c", vram_free_gb=16) is None


def test_progress_release_and_resume_order(tmp_path):
    b = JobBoard(tmp_path)
    j = b.submit({"command": "true"})
    later = b.submit({"command": "true"})
    b.claim("mac-a", 16)
    assert b.progress(j.job_id, "mac-a", 10, {"train_loss": 2.5}, ["Iter 10"])["ok"]
    assert b.progress(j.job_id, "mac-z", 11, {}, [])["cancel"]               # not the job's node
    assert b.release(j.job_id, "mac-a", "user came back")
    again = b.claim("mac-b", 16)
    assert again.job_id == j.job_id and again.preemptions == 1 and again.nodes == ["mac-a", "mac-b"]
    assert b.get(j.job_id).metrics["train_loss"] == [[10, 2.5]]
    assert b.get(later.job_id).state == "QUEUED"


def test_a_silent_worker_loses_its_job_and_state_survives_restart(tmp_path, monkeypatch):
    b = JobBoard(tmp_path)
    j = b.submit({"command": "true", "label": "x"})
    b.claim("mac-a", 16)
    monkeypatch.setattr(board_mod, "STALE_SECONDS", 0.0)
    time.sleep(0.01)
    b.requeue_stale()
    assert b.get(j.job_id).state == "QUEUED"
    b.claim("mac-b", 16)
    b.save(force=True)
    again = JobBoard(tmp_path)                                               # the daemon restarted
    assert again.get(j.job_id).state == "QUEUED" and again.get(j.job_id).label == "x"


def test_job_control_needs_the_token_but_reading_does_not(tmp_path):
    server = TelemetryServer(MetricsRegistry("head"), job_board=JobBoard(tmp_path), token=TOKEN)
    body = json.dumps({"command": "true"}).encode()
    assert server.handle_post("/api/jobs", body, "10.0.0.2", {})[0] == 401
    assert server.handle_post("/api/jobs", body, "10.0.0.2", {"X-Mesh-Token": "wrong"})[0] == 401
    status, _, resp = server.handle_post("/api/jobs", body, "10.0.0.2", {"X-Mesh-Token": TOKEN})
    assert status == 200 and len(json.loads(resp)["jobs"]) == 1
    assert server.handle_request("/api/jobs")[0] == 200
    assert server.handle_request("/dashboard")[0] == 200
    sha = server.job_board.put_bundle(_tar({"a.txt": "x"}))
    assert server.handle_request(f"/api/bundles/{sha}", {})[0] == 401
    assert server.handle_request(f"/api/bundles/{sha}", {"X-Mesh-Token": TOKEN})[0] == 200
    assert b"mesh_jobs" in server.handle_request("/metrics")[2]


def test_without_a_token_job_control_is_off(tmp_path):
    server = TelemetryServer(MetricsRegistry("head"), job_board=JobBoard(tmp_path), token=None)
    assert server.handle_post("/api/jobs", b"{}", "10.0.0.2", {"X-Mesh-Token": ""})[0] == 401


@pytest.mark.parametrize("line, step, metrics", [
    ('MESH {"step": 40, "metrics": {"eval_exact": 0.8}}', 40, {"eval_exact": 0.8}),
    ("Iter 25: Train loss 1.234, Learning Rate 1.000e-04, It/sec 0.512, Tokens/sec 310.1, Trained Tokens 9000, Peak mem 9.120 GB",
     25, {"train_loss": 1.234, "it_per_sec": 0.512, "peak_mem_gb": 9.12}),
    ("Iter 50: Val loss 0.876, Val took 12.3s", 50, {"val_loss": 0.876}),
    ("Loading pretrained model", None, {}),
])
def test_output_lines_become_progress(line, step, metrics):
    assert parse_line(line) == (step, metrics)


def test_bundles_cannot_write_outside_their_folder(tmp_path):
    with pytest.raises(ValueError):
        safe_extract(_tar({"../escape.txt": "x"}), tmp_path / "job")
    safe_extract(_tar({"ok/inside.txt": "x"}), tmp_path / "job")
    assert (tmp_path / "job" / "ok" / "inside.txt").read_text() == "x"
    assert pack(tmp_path / "job", ["nothing/*"]) is None


def test_a_worker_runs_a_job_from_bundle_to_results(tmp_path, monkeypatch):
    monkeypatch.setenv("MESH_STATE_DIR", str(tmp_path / "state"))
    board = JobBoard(tmp_path / "board")
    server = TelemetryServer(MetricsRegistry("head"), host="127.0.0.1", port=0, job_board=board, token=TOKEN)
    # port 0: let the OS choose, then read it back
    from http.server import ThreadingHTTPServer
    server.start()
    port = server._server.server_address[1]
    try:
        script = ("import json, pathlib\n"
                  "for s in (1, 2, 3):\n"
                  "    print('MESH ' + json.dumps({'step': s, 'metrics': {'train_loss': 3.0 / s}}), flush=True)\n"
                  "pathlib.Path('out').mkdir()\n"
                  "pathlib.Path('out/result.json').write_text(json.dumps({'ok': True}))\n")
        head = Head(f"http://127.0.0.1:{port}", TOKEN)
        sha = head.put_bytes("/api/bundles", _tar({"job.py": script}))["bundle"]
        jid = head.post("/api/jobs", {"command": "{python} job.py", "bundle": sha, "artifacts": ["out/*.json"],
                                      "total_steps": 3, "group": "demo"})["jobs"][0]
        runner = Runner(head, "mac-test", python=sys.executable, own_machine=True, workdir=tmp_path / "work", poll_sec=0.2)
        job = head.post("/api/jobs/claim", {"node_id": "mac-test", "vram_free_gb": 16})["job"]
        assert runner.run_job(job) == "done"
        done = board.get(jid)
        assert done.state == "COMPLETED" and done.step == 3 and done.latest["train_loss"] == 1.0
        assert [p[0] for p in done.metrics["train_loss"]] == [1, 2, 3]
        with tarfile.open(board.file_path(jid, "results")) as tar:
            assert json.load(tar.extractfile("out/result.json")) == {"ok": True}
    finally:
        server.stop()


def test_a_failed_job_can_be_queued_again(tmp_path):
    b = JobBoard(tmp_path)
    j = b.submit({"command": "false"})
    b.claim("mac-a", 16)
    b.progress(j.job_id, "mac-a", 3, {"train_loss": 2.0}, [])
    b.finish(j.job_id, "mac-a", 1, "out of memory")
    assert not b.retry(b.submit({"command": "true"}).job_id)            # a queued job has nothing to retry
    assert b.retry(j.job_id)
    again = b.get(j.job_id)
    assert again.state == "QUEUED" and again.step == 0 and not again.metrics and again.error == ""
