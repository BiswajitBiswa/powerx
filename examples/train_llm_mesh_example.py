"""Example: Fine-tuning / Training an LLM using PowerX with Claude.

Claude uses this pattern to:
1. Probe the mesh for idle compute nodes ($0 cost).
2. Train on PyTorch MPS with gradient checkpointing.
3. Automatically checkpoint every 50 steps so user keystroke preemption never loses progress.
4. Fall back to Cloud CUDA if mesh nodes are busy.
"""

import os
import sys
from pathlib import Path

# Ensure hybrid_gpu_mesh package is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Import the Claude Mesh Router
from hybrid_gpu_mesh.integrations.claude_router import get_mesh_training_config


def train_model():
    print("=" * 64)
    print(" ⚡ CLAUDE LLM TRAINING ROUTER (PowerX)")
    print("=" * 64)

    # 1. Ask Mesh Router for optimal training target
    mesh_cfg = get_mesh_training_config(required_vram_gb=4.0)
    device = torch.device(mesh_cfg["device"])

    print(f"Target Device:     {device.type.upper()}")
    print(f"Hardware Model:    {mesh_cfg['hardware']}")
    print(f"Mesh Local Node:   {mesh_cfg['is_mesh_node']} (Node: {mesh_cfg['node_id']})")
    print(f"Free VRAM:         {mesh_cfg['vram_free_gb']:.1f} GB")
    print(f"Batch Size:        {mesh_cfg['recommended_batch_size']}")
    print(f"Checkpoint Steps:  Every {mesh_cfg['checkpoint_interval_steps']} steps")
    print("-" * 64)

    # 2. Setup a dummy transformer-like layer for demonstration
    model = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # Synthetic training batch
    x_data = torch.randn(128, 512)
    y_data = torch.randn(128, 512)
    dataset = TensorDataset(x_data, y_data)
    loader = DataLoader(dataset, batch_size=mesh_cfg["recommended_batch_size"], shuffle=True)

    os.makedirs(mesh_cfg["save_dir"], exist_ok=True)

    print("🚀 Training in progress on opportunistic idle hardware...")
    for epoch in range(2):
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print(f"  [Epoch {epoch+1}] Step {step:02d} | Loss: {loss.item():.4f} | Device: {device}")

            # Checkpoint for keystroke safety
            if step > 0 and step % mesh_cfg["checkpoint_interval_steps"] == 0:
                ckpt_path = os.path.join(mesh_cfg["save_dir"], f"checkpoint_step_{step}.pt")
                torch.save(model.state_dict(), ckpt_path)
                print(f"  💾 Safety Checkpoint saved at {ckpt_path}")

    final_path = os.path.join(mesh_cfg["save_dir"], "final_model.pt")
    torch.save(model.state_dict(), final_path)
    print("=" * 64)
    print(f"✅ Training completed successfully! Model saved to {final_path}")
    print("=" * 64)


if __name__ == "__main__":
    train_model()
