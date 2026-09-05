"""
train_laptop2_rank1.py

Run this file on LAPTOP 2. This laptop acts as "rank 1" (a worker).

WHAT TO CHANGE BEFORE RUNNING:
  MASTER_ADDR -> set this to LAPTOP 1's Tailscale IP (the SAME value used in
                 train_laptop1_rank0.py, NOT this laptop's own IP)

HOW TO RUN (on Laptop 2):
  python train_laptop2_rank1.py

NOTE: Start train_laptop1_rank0.py FIRST (or within a few seconds of this one) -
both scripts wait for each other to connect before training starts.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ----------------------------------------------------------------------
# STEP 1: Network settings - EDIT THESE TWO LINES FOR YOUR SETUP
# ----------------------------------------------------------------------
MASTER_ADDR = "100.101.102.103"   # <-- LAPTOP 1's Tailscale IP (same as in the rank 0 file)
MASTER_PORT = "29500"             # must match the port used in the rank 0 file

RANK = 1            # this laptop is rank 1 (a worker)
WORLD_SIZE = 2       # total number of laptops taking part


def setup_distributed():
    import os
    os.environ["MASTER_ADDR"] = MASTER_ADDR
    os.environ["MASTER_PORT"] = MASTER_PORT

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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Rank {RANK}] Using device: {device}")

    # Same seed as rank 0 -> both laptops start with an IDENTICAL model copy
    torch.manual_seed(42)
    model = TinyANN().to(device)
    model = DDP(model)

    # ------------------------------------------------------------------
    # STEP 2: This laptop's OWN slice of data (different from Laptop 1's slice)
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
        loss.backward()          # gradients sync with Laptop 1 automatically over Tailscale
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"[Rank {RANK}] Epoch {epoch+1:3d}/{epochs} | Loss: {loss.item():.4f}")

    print(f"[Rank {RANK}] Training finished.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
