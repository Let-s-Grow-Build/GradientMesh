"""
train_laptop1_rank0.py

Run this file on LAPTOP 1. This laptop acts as "rank 0" (the coordinator).

WHAT TO CHANGE BEFORE RUNNING:
  MASTER_ADDR -> set this to Laptop 1's own Tailscale IP (run `tailscale ip -4` on Laptop 1)

HOW TO RUN (on Laptop 1):
  python train_laptop1_rank0.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ----------------------------------------------------------------------
# STEP 1: Network settings - EDIT THESE TWO LINES FOR YOUR SETUP
# ----------------------------------------------------------------------
MASTER_ADDR = "100.101.102.103"   # <-- Laptop 1's Tailscale IP (run: tailscale ip -4)
MASTER_PORT = "29500"             # any free port, must match on both laptops

RANK = 0            # this laptop is rank 0 (the coordinator)
WORLD_SIZE = 2       # total number of laptops taking part


def setup_distributed():
    import os
    os.environ["MASTER_ADDR"] = MASTER_ADDR
    os.environ["MASTER_PORT"] = MASTER_PORT

    # gloo backend works across CPU/GPU and over a normal network (like Tailscale)
    dist.init_process_group(
        backend="gloo",
        rank=RANK,
        world_size=WORLD_SIZE,
    )
    print(f"[Rank {RANK}] Connected. World size = {WORLD_SIZE}")


class TinyANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.net(x)


def main():
    setup_distributed()

    # Use GPU if this laptop has one, otherwise CPU - DDP works with either
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Rank {RANK}] Using device: {device}")

    # Every rank must build the model with the SAME random seed so all
    # copies start identical, then DDP keeps them in sync after that.
    torch.manual_seed(42)
    model = TinyANN().to(device)
    model = DDP(model)   # DDP handles the gradient averaging (all-reduce) automatically

    # ------------------------------------------------------------------
    # STEP 2: This laptop's OWN slice of data (each laptop uses different data)
    # ------------------------------------------------------------------
    torch.manual_seed(RANK)  # different seed per rank -> different data slice
    num_samples = 128
    X = torch.randn(num_samples, 4).to(device)
    true_weights = torch.tensor([2.0, -1.0, 0.5, 3.0]).to(device)
    y = (X @ true_weights).unsqueeze(1) + 0.1 * torch.randn(num_samples, 1).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    epochs = 100
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(X)
        loss = criterion(predictions, y)
        loss.backward()          # DDP automatically all-reduces gradients across BOTH laptops here
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"[Rank {RANK}] Epoch {epoch+1:3d}/{epochs} | Loss: {loss.item():.4f}")

    print(f"[Rank {RANK}] Training finished.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
