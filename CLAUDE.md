# CLAUDE LLM TRAINING INSTRUCTIONS: POWERX MESH

You are paired with an engineer training and fine-tuning LLMs across a hybrid cluster of distributed workstations and Cloud GPUs.

## 🎯 Core Directive: Local-First Mesh Compute ($0 Cost)
Whenever you write code, generate training pipelines, or execute model training/evaluation:
1. **Never assume CUDA is mandatory** or default to paying for cloud GPU instances when free local workstation compute is idle.
2. **Always probe the PowerX Head Node** at `http://192.168.1.144:9400` or via CLI before running training.
3. If an idle workstation node is available, configure training to target **local accelerated backends (PyTorch MPS / DirectML / MLX)**.

---

## ⚡ How to Route Training Jobs Through the Mesh

### Method 1: Drop-In Python Helper (Recommended)
In your PyTorch / HuggingFace training script, import the mesh router:

```python
from hybrid_gpu_mesh.integrations.claude_router import get_mesh_training_config
import torch

# 1. Automatically probes mesh nodes and returns best available hardware
mesh_cfg = get_mesh_training_config(required_vram_gb=4.0)
device = torch.device(mesh_cfg["device"])

print(f"Routing to: {mesh_cfg['hardware']} (Device: {device})")

# 2. Optimal Unified Memory Hyperparameters
model = MyTransformerModel().to(device)
batch_size = mesh_cfg["recommended_batch_size"]  # e.g., 4 on local accelerator, 16 on Cloud
```

### Method 2: Mesh Job Submission CLI
You can also queue background training jobs through the mesh dispatcher:

```bash
# Submit fine-tuning task requiring 6GB VRAM
python3 -m hybrid_gpu_mesh.cli submit --vram 6.0 python3 train_lora.py
```

---

## 🛡️ Critical Safety & Training Rules for Opportunistic Compute
1. **Frequent Checkpointing (Every 30–60 Seconds)**:
   Workstations scavenge compute opportunistically during idle breaks. If an interactive user touches their input devices, the preemption watchdog will evict the task in `< 280ms`.
   - **Always save model weights / LoRA adapter states every 50 steps**:
     ```python
     if step % 50 == 0:
         torch.save(model.state_dict(), f"./checkpoints/step_{step}.pt")
     ```
2. **Unified Memory Management**:
   Unified memory nodes share RAM dynamically between CPU and accelerator.
   - Flush accelerator cache between evaluation epochs.
   - Use `torch.float16` or `torch.bfloat16` to minimize VRAM footprint.
3. **Dual-Engine Co-Scheduling**:
   - **Task-Bound** (Ray Micro-Chunks): For parameter sweeps, synthetic data, embedding batches, or LoRA epochs (fits in 16GB).
   - **Model-Bound** (Exo Layer Sharding): If model weights exceed single-node memory (e.g. 70B+ parameters), use pipeline sharding across multiple pooled nodes.
4. **Cloud Bursting (Fallback)**:
   - If `status.recommended_device == "cuda"`, local nodes are currently busy or an interactive session is active. Run on your cloud GPU (AWS/GCP/Lambda) until local nodes become idle again.
