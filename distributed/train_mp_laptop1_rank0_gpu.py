"""
train_mp_laptop1_rank0_gpu.py

MODEL PARALLELISM across two laptops -- GPU VERSION.
Run this file on LAPTOP 1. This laptop holds LAYER 1 of the model (rank 0).

Unlike Data Parallelism, here the two laptops hold DIFFERENT PARTS of the
SAME model. Laptop 1 computes Layer 1, sends its output (the "activation")
to Laptop 2 over the network. Laptop 2 computes Layer 2, computes the loss,
then sends the GRADIENT back to Laptop 1, so Laptop 1 can finish its own
backward pass.

GPU NOTE:
  The `gloo` backend's point-to-point `send`/`recv` calls are only reliable
  with CPU tensors -- even when both machines have CUDA GPUs. So the pattern
  used here is:
    1. Compute everything (forward/backward) on the GPU as normal.
    2. Right before a `dist.send`/`dist.recv`, move the tensor to CPU.
    3. Right after receiving, move the tensor back to GPU before using it
       in further computation (e.g. `.backward()`).
  This gives you GPU speed for the actual math while keeping the network
  transport working correctly.

  If you want true GPU-to-GPU transport (no CPU staging), you'd need the
  `nccl` backend instead of `gloo` -- but `nccl` requires NVIDIA GPUs on
  BOTH machines plus proper multi-node NCCL setup, and doesn't support
  Windows. Given this is a Tailscale-linked laptop setup, `gloo` + CPU
  staging is the practical/robust choice.

WHAT TO CHANGE BEFORE RUNNING:
  MASTER_ADDR -> Laptop 1's own Tailscale IP (run: tailscale ip -4)

HOW TO RUN (on Laptop 1):
  python train_mp_laptop1_rank0_gpu.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

# ----------------------------------------------------------------------
# STEP 1: Network settings - EDIT THESE FOR YOUR SETUP
# ----------------------------------------------------------------------
MASTER_ADDR = "100.99.155.78"   # <-- Laptop 1's Tailscale IP
MASTER_PORT = "29501"             # different port from the data-parallel scripts is fine

RANK = 0
WORLD_SIZE = 2

BATCH_SIZE = 128   # must match exactly on both laptops
HIDDEN_SIZE = 8    # output width of Layer 1 = input width of Layer 2, must match both laptops


def setup_distributed():
    os.environ["GLOO_SOCKET_IFNAME"] = "Tailscale"
    os.environ["USE_LIBUV"] = "0"
    os.environ["MASTER_ADDR"] = MASTER_ADDR
    os.environ["MASTER_PORT"] = MASTER_PORT
    dist.init_process_group(backend="gloo", rank=RANK, world_size=WORLD_SIZE)
    print(f"[Rank {RANK}] Connected. World size = {WORLD_SIZE}")


class Layer1(nn.Module):
    """Stage 1 of the model: lives entirely on Laptop 1.

    2 layers deep. Input width stays 4 (raw features), output width
    stays HIDDEN_SIZE (must still match Layer 2's input width on Laptop 2).
    (Reduced from 4 layers -> 2 layers: the 4-layer version was plateauing
    near the target variance, i.e. barely better than predicting the mean,
    due to dead ReLUs / vanishing gradients with plain SGD at this depth.)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


def main():
    setup_distributed()

    # ---- GPU device selection ----
    # All actual model math happens here. Network transport (send/recv)
    # still stages through CPU -- see the module docstring above.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Rank {RANK}] Running Layer 1 on: {device}")
    if device.type == "cpu":
        print(f"[Rank {RANK}] WARNING: CUDA not available, falling back to CPU.")

    torch.manual_seed(42)
    layer1 = Layer1().to(device)
    optimizer1 = optim.Adam(layer1.parameters(), lr=0.01)

    # This laptop is where the raw input data lives (Stage 1 of the pipeline)
    torch.manual_seed(0)
    X = torch.randn(BATCH_SIZE, 4).to(device)
    true_weights = torch.tensor([2.0, -1.0, 0.5, 3.0])
    y = (X.cpu() @ true_weights).unsqueeze(1) + 0.1 * torch.randn(BATCH_SIZE, 1)
    y = y.to(device)

    epochs = 500
    for epoch in range(epochs):
        optimizer1.zero_grad()

        # ---- Forward pass: compute Layer 1's output (on GPU) ----
        activation = layer1(X)

        # ---- Send activation + labels to Laptop 2 (rank 1) ----
        # Must move to CPU first: gloo send/recv needs CPU tensors.
        dist.send(activation.detach().cpu(), dst=1)
        dist.send(y.cpu(), dst=1)

        # ---- Receive the gradient of the loss w.r.t. our activation ----
        # Receive buffer must be a CPU tensor too.
        grad_from_rank1_cpu = torch.zeros_like(activation, device="cpu")
        dist.recv(grad_from_rank1_cpu, src=1)
        grad_from_rank1 = grad_from_rank1_cpu.to(device)

        # ---- Finish OUR backward pass using that received gradient ----
        activation.backward(grad_from_rank1)
        optimizer1.step()

        # ---- Receive the loss value just for logging ----
        loss_tensor = torch.zeros(1)  # stays on CPU, it's just a scalar for printing
        dist.recv(loss_tensor, src=1)

        if (epoch + 1) % 50 == 0:
            print(f"[Rank {RANK}] Epoch {epoch+1:3d}/{epochs} | Loss: {loss_tensor.item():.4f}")

    print(f"[Rank {RANK}] Training finished.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
