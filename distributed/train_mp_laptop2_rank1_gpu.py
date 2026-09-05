"""
train_mp_laptop2_rank1_gpu.py

MODEL PARALLELISM across two laptops -- GPU VERSION.
Run this file on LAPTOP 2. This laptop holds LAYER 2 of the model (rank 1).

This laptop receives Layer 1's output (the activation) and the labels from
Laptop 1, computes Layer 2's forward pass, computes the loss, runs its OWN
backward pass, and then sends the gradient (w.r.t. the received activation)
back to Laptop 1 so it can finish its half of the backward pass too.

GPU NOTE (synced with the Laptop 1 script):
  The `gloo` backend's point-to-point `send`/`recv` calls are only reliable
  with CPU tensors -- even when both machines have CUDA GPUs. So the pattern
  used here is:
    1. Compute everything (forward/backward) on the GPU as normal.
    2. Right before a `dist.send`, move the tensor to CPU.
    3. Right after a `dist.recv` into a CPU buffer, move the tensor to GPU
       before using it in further computation.
  This gives you GPU speed for the actual math while keeping the network
  transport working correctly.

WHAT TO CHANGE BEFORE RUNNING:
  MASTER_ADDR -> LAPTOP 1's Tailscale IP (same value as in the rank 0 file,
                 NOT this laptop's own IP)

HOW TO RUN (on Laptop 2):
  python train_mp_laptop2_rank1_gpu.py

NOTE: Start train_mp_laptop1_rank0_gpu.py FIRST (or within a few seconds).
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

# ----------------------------------------------------------------------
# STEP 1: Network settings - EDIT THESE FOR YOUR SETUP
# ----------------------------------------------------------------------
MASTER_ADDR = "100.99.155.78"   # <-- LAPTOP 1's Tailscale IP (same as rank 0 file)
MASTER_PORT = "29501"             # must match the rank 0 file exactly

RANK = 1
WORLD_SIZE = 2

BATCH_SIZE = 128   # must match exactly on both laptops
HIDDEN_SIZE = 8    # input width of Layer 2 = output width of Layer 1, must match both laptops


def setup_distributed():
    os.environ["GLOO_SOCKET_IFNAME"] = "Tailscale"
    os.environ["USE_LIBUV"] = "0"
    os.environ["MASTER_ADDR"] = MASTER_ADDR
    os.environ["MASTER_PORT"] = MASTER_PORT
    dist.init_process_group(backend="gloo", rank=RANK, world_size=WORLD_SIZE)
    print(f"[Rank {RANK}] Connected. World size = {WORLD_SIZE}")


class Layer2(nn.Module):
    """Stage 2 of the model: lives entirely on Laptop 2.

    4 layers deep, synced with Layer 1's depth on Laptop 1.
    Input width stays HIDDEN_SIZE (must match Layer 1's output width).
    The 3 interior layers are HIDDEN_SIZE -> HIDDEN_SIZE, and the final
    layer projects down to a single scalar prediction (no ReLU on the
    output layer, since this is a regression head feeding MSELoss).
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 1),
        )

    def forward(self, x):
        return self.net(x)


def main():
    setup_distributed()

    # ---- GPU device selection ----
    # All actual model math happens here. Network transport (send/recv)
    # still stages through CPU -- see the module docstring above.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Rank {RANK}] Running Layer 2 on: {device}")
    if device.type == "cpu":
        print(f"[Rank {RANK}] WARNING: CUDA not available, falling back to CPU.")

    torch.manual_seed(43)  # deliberately a different seed - Layer 2 is a different module
    layer2 = Layer2().to(device)
    optimizer2 = optim.SGD(layer2.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    epochs = 300
    for epoch in range(epochs):
        optimizer2.zero_grad()

        # ---- Receive Layer 1's output + labels from Laptop 1 ----
        # Receive buffers must be CPU tensors: gloo send/recv needs CPU.
        activation_cpu = torch.zeros(BATCH_SIZE, HIDDEN_SIZE)
        dist.recv(activation_cpu, src=0)
        # Move to GPU and mark as requiring grad *after* the transfer,
        # since it needs to be a leaf tensor for autograd on this rank.
        activation = activation_cpu.to(device)
        activation.requires_grad_(True)

        y_cpu = torch.zeros(BATCH_SIZE, 1)
        dist.recv(y_cpu, src=0)
        y = y_cpu.to(device)

        # ---- Forward pass: compute Layer 2's output and the loss (GPU) ----
        output = layer2(activation)
        loss = criterion(output, y)

        # ---- Backward pass: this also computes activation.grad (GPU) ----
        loss.backward()
        optimizer2.step()

        # ---- Send the gradient w.r.t. the activation back to Laptop 1 ----
        # Must move to CPU first: gloo send/recv needs CPU tensors.
        dist.send(activation.grad.cpu(), dst=0)

        # ---- Send the loss value back just for logging on Laptop 1 ----
        dist.send(torch.tensor([loss.item()]), dst=0)  # scalar, stays on CPU

        if (epoch + 1) % 10 == 0:
            print(f"[Rank {RANK}] Epoch {epoch+1:3d}/{epochs} | Loss: {loss.item():.4f}")

    print(f"[Rank {RANK}] Training finished.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
