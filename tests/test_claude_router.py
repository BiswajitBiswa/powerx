"""Unit tests for Claude LLM Training Mesh Router."""

import pytest
from hybrid_gpu_mesh.integrations.claude_router import ClaudeMeshRouter, MeshNodeStatus


def test_claude_router_selection_mock():
    router = ClaudeMeshRouter(head_url="http://127.0.0.1:9999")  # Unreachable port to test fallback
    status = router.select_training_device(required_vram_gb=4.0)

    assert isinstance(status, MeshNodeStatus)
    # If on Apple Silicon locally, it detects local Mac; if not, falls back to cloud/cuda
    assert status.recommended_device in ("mps", "cuda", "cpu")
    assert status.vram_free_gb >= 0.0


def test_claude_router_local_detection():
    router = ClaudeMeshRouter()
    res = router.check_mesh_availability()
    assert isinstance(res, dict)
    assert "connected" in res
