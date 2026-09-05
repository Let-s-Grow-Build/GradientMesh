"""
train_mp_laptop2_rank1.py

MODEL PARALLELISM across two laptops.
Run this file on LAPTOP 2. This laptop holds LAYER 2 of the model (rank 1).

This laptop receives Layer 1's output (the activation) and the labels from
Laptop 1, computes Layer 2's forward pass, computes the loss, runs its OWN
backward pass, and then sends the gradient (w.r.t. the received activation)
back to Laptop 1 so it can finish its half of the backward pass too.

WHAT TO CHANGE BEFORE RUNNING:
  MASTER_ADDR -> LAPTOP 1's Tailscale IP (same value as in the rank 0 file,
                 NOT this laptop's own IP)

HOW TO RUN (on Laptop 2):
  python train_mp_laptop2_rank1.py

NOTE: Start train_mp_laptop1_rank0.py FIRST (or within a few seconds).
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

# ----------------------------------------------------------------------
# STEP 1: Network settings - EDIT THESE FOR YOUR SETUP
# ----------------------------------------------------------------------
MASTER_ADDR = "100.101.102.103"   # <-- LAPTOP 1's Tailscale IP (same as rank 0 file)
MASTER_PORT = "29501"             # must match the rank 0 file exactly

RANK = 1
WORLD_SIZE = 2

BATCH_SIZE = 128   # must match exactly on both laptops
HIDDEN_SIZE = 8    # input width of Layer 2 = output width of Layer 1, must match both laptops


def setup_distributed():
    os.environ["MASTER_ADDR"] = MASTER_ADDR
    os.environ["MASTER_PORT"] = MASTER_PORT
    dist.init_process_group(backend="gloo", rank=RANK, world_size=WORLD_SIZE)
    print(f"[Rank {RANK}] Connected. World size = {WORLD_SIZE}")


class Layer2(nn.Module):
    """Stage 2 of the model: lives entirely on Laptop 2."""
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        return self.net(x)


def main():
    setup_distributed()

    device = torch.device("cpu")  # gloo send/recv works most reliably on CPU tensors
    print(f"[Rank {RANK}] Running Layer 2 on: {device}")

    torch.manual_seed(43)  # deliberately a different seed - Layer 2 is a different module
    layer2 = Layer2().to(device)
    optimizer2 = optim.SGD(layer2.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    epochs = 100
    for epoch in range(epochs):
        optimizer2.zero_grad()

        # ---- Receive Layer 1's output + labels from Laptop 1 ----
        activation = torch.zeros(BATCH_SIZE, HIDDEN_SIZE)
        dist.recv(activation, src=0)
        activation.requires_grad_(True)  # we need gradients w.r.t. this received tensor

        y = torch.zeros(BATCH_SIZE, 1)
        dist.recv(y, src=0)

        # ---- Forward pass: compute Layer 2's output and the loss ----
        output = layer2(activation)
        loss = criterion(output, y)

        # ---- Backward pass: this also computes activation.grad ----
        loss.backward()
        optimizer2.step()

        # ---- Send the gradient w.r.t. the activation back to Laptop 1 ----
        dist.send(activation.grad, dst=0)

        # ---- Send the loss value back just for logging on Laptop 1 ----
        dist.send(torch.tensor([loss.item()]), dst=0)

        if (epoch + 1) % 10 == 0:
            print(f"[Rank {RANK}] Epoch {epoch+1:3d}/{epochs} | Loss: {loss.item():.4f}")

    print(f"[Rank {RANK}] Training finished.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
