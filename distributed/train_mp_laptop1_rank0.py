"""
train_mp_laptop1_rank0.py

MODEL PARALLELISM across two laptops.
Run this file on LAPTOP 1. This laptop holds LAYER 1 of the model (rank 0).

Unlike Data Parallelism, here the two laptops hold DIFFERENT PARTS of the
SAME model. Laptop 1 computes Layer 1, sends its output (the "activation")
to Laptop 2 over the network. Laptop 2 computes Layer 2, computes the loss,
then sends the GRADIENT back to Laptop 1, so Laptop 1 can finish its own
backward pass. This manual send/receive of activations and gradients is
necessary because autograd cannot automatically cross a process/network
boundary the way it can cross devices in the same process (see the
CPU+GPU model-parallel notebook, which uses regular .to(device) instead).

WHAT TO CHANGE BEFORE RUNNING:
  MASTER_ADDR -> Laptop 1's own Tailscale IP (run: tailscale ip -4)

HOW TO RUN (on Laptop 1):
  python train_mp_laptop1_rank0.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

# ----------------------------------------------------------------------
# STEP 1: Network settings - EDIT THESE FOR YOUR SETUP
# ----------------------------------------------------------------------
MASTER_ADDR = "100.101.102.103"   # <-- Laptop 1's Tailscale IP
MASTER_PORT = "29501"             # different port from the data-parallel scripts is fine

RANK = 0
WORLD_SIZE = 2

BATCH_SIZE = 128   # must match exactly on both laptops
HIDDEN_SIZE = 8    # output width of Layer 1 = input width of Layer 2, must match both laptops


def setup_distributed():
    os.environ["MASTER_ADDR"] = MASTER_ADDR
    os.environ["MASTER_PORT"] = MASTER_PORT
    dist.init_process_group(backend="gloo", rank=RANK, world_size=WORLD_SIZE)
    print(f"[Rank {RANK}] Connected. World size = {WORLD_SIZE}")


class Layer1(nn.Module):
    """Stage 1 of the model: lives entirely on Laptop 1."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, HIDDEN_SIZE),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


def main():
    setup_distributed()

    device = torch.device("cpu")  # gloo send/recv works most reliably on CPU tensors
    print(f"[Rank {RANK}] Running Layer 1 on: {device}")

    torch.manual_seed(42)
    layer1 = Layer1().to(device)
    optimizer1 = optim.SGD(layer1.parameters(), lr=0.01)

    # This laptop is where the raw input data lives (Stage 1 of the pipeline)
    torch.manual_seed(0)
    X = torch.randn(BATCH_SIZE, 4)
    true_weights = torch.tensor([2.0, -1.0, 0.5, 3.0])
    y = (X @ true_weights).unsqueeze(1) + 0.1 * torch.randn(BATCH_SIZE, 1)

    epochs = 100
    for epoch in range(epochs):
        optimizer1.zero_grad()

        # ---- Forward pass: compute Layer 1's output ----
        activation = layer1(X)

        # ---- Send activation + labels to Laptop 2 (rank 1) ----
        dist.send(activation.detach(), dst=1)
        dist.send(y, dst=1)

        # ---- Receive the gradient of the loss w.r.t. our activation ----
        grad_from_rank1 = torch.zeros_like(activation)
        dist.recv(grad_from_rank1, src=1)

        # ---- Finish OUR backward pass using that received gradient ----
        activation.backward(grad_from_rank1)
        optimizer1.step()

        # ---- Receive the loss value just for logging ----
        loss_tensor = torch.zeros(1)
        dist.recv(loss_tensor, src=1)

        if (epoch + 1) % 10 == 0:
            print(f"[Rank {RANK}] Epoch {epoch+1:3d}/{epochs} | Loss: {loss_tensor.item():.4f}")

    print(f"[Rank {RANK}] Training finished.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
