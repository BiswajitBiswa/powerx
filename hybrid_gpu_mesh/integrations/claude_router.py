"""Claude AI Training Router for PowerX.

Allows Claude (Claude Code, Claude Projects, or Python training scripts) to automatically
detect available idle compute nodes across the mesh, route fine-tuning
or training workloads to MPS or local accelerators ($0 cost), and fall back to Cloud GPUs when nodes are busy.
"""

from __future__ import annotations
import json
import logging
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class MeshNodeStatus:
    available: bool
    node_id: str
    hardware_type: str       # "apple_silicon" or "nvidia_cloud"
    device_model: str        # e.g. "Apple M1 Pro"
    vram_free_gb: float
    recommended_device: str  # "mps", "cuda", or "cpu"
    engine: str              # "task_bound" or "model_bound"


class ClaudeMeshRouter:
    """Routes LLM training jobs to idle mesh nodes or Cloud GPUs."""

    def __init__(self, head_url: str = "http://192.168.1.144:9400"):
        self.head_url = head_url.rstrip("/")

    def check_mesh_availability(self, timeout_sec: float = 2.0) -> Dict[str, Any]:
        """Query the mesh head node to inspect available idle compute."""
        try:
            req = urllib.request.Request(
                f"{self.head_url}/api/status",
                headers={"User-Agent": "Claude-LLM-Trainer/1.0"},
            )
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    return {"connected": True, "data": data}
        except Exception as e:
            logger.debug(f"Could not reach mesh head over HTTP: {e}")

        # Local fallback if running on the Head Node itself
        try:
            from hybrid_gpu_mesh.telemetry.apple_silicon import AppleSiliconCollector
            from hybrid_gpu_mesh.scavenging.idle_detector import IdleConfig, IdleDetector
            collector = AppleSiliconCollector()
            if collector.is_apple_silicon():
                metrics = collector.collect()
                detector = IdleDetector(IdleConfig(), collector)
                eval_res = detector.evaluate()
                return {
                    "connected": True,
                    "data": {
                        "node_id": "local-head",
                        "apple_silicon": {
                            "model": metrics.model,
                            "gpu_cores": metrics.gpu_cores,
                            "vram_total_gb": metrics.vram_total_gb,
                            "vram_free_gb": metrics.vram_free_gb,
                            "is_safe_to_scavenge": eval_res.is_safe_to_scavenge,
                            "is_on_ac_power": metrics.is_on_ac_power,
                        },
                    },
                }
        except Exception:
            pass

        return {"connected": False, "data": {}}

    def select_training_device(
        self,
        required_vram_gb: float = 4.0,
        prefer_local_zero_cost: bool = True,
    ) -> MeshNodeStatus:
        """Select best execution device for Claude's LLM training task."""
        status = self.check_mesh_availability()
        
        if status.get("connected"):
            data = status.get("data", {})
            mac = data.get("apple_silicon", {})
            vram_free = float(mac.get("vram_free_gb", 0.0))
            is_safe = bool(mac.get("is_safe_to_scavenge", True))

            if prefer_local_zero_cost and is_safe and vram_free >= required_vram_gb:
                # Idle node ready for training!
                engine = "model_bound" if required_vram_gb > 16.0 else "task_bound"
                return MeshNodeStatus(
                    available=True,
                    node_id=data.get("node_id", "node-01-primary"),
                    hardware_type="apple_silicon",
                    device_model=mac.get("model", "Unified Memory Node"),
                    vram_free_gb=vram_free,
                    recommended_device="mps",
                    engine=engine,
                )

        # Cloud fallback (CUDA)
        return MeshNodeStatus(
            available=False,
            node_id="cloud-burst-reserve",
            hardware_type="nvidia_cloud",
            device_model="NVIDIA Cloud GPU",
            vram_free_gb=80.0,
            recommended_device="cuda",
            engine="cloud_burst",
        )


def get_mesh_training_config(required_vram_gb: float = 4.0) -> Dict[str, Any]:
    """Convenience helper for Claude generated PyTorch / HuggingFace scripts."""
    router = ClaudeMeshRouter()
    status = router.select_training_device(required_vram_gb=required_vram_gb)

    import torch
    if status.recommended_device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        use_mesh = True
    elif torch.cuda.is_available():
        device = "cuda"
        use_mesh = False
    else:
        device = "cpu"
        use_mesh = False

    return {
        "device": device,
        "is_mesh_node": use_mesh,
        "node_id": status.node_id,
        "hardware": status.device_model,
        "vram_free_gb": status.vram_free_gb,
        "engine": status.engine,
        # Optimal training hyperparameters for Unified Memory
        "recommended_batch_size": 4 if device == "mps" else 16,
        "use_gradient_checkpointing": True if device == "mps" else False,
        "checkpoint_interval_steps": 50,  # Save frequently for instant keystroke safety
        "save_dir": "./checkpoints_mesh",
    }
