# GradientMesh

**Splitting intelligence across GPUs** — a hands-on guide to data, tensor, model, and pipeline parallelism, from one GPU to a full cluster.

This repo has two parts:

- **`docs/`** — 12 plain-English markdown files explaining every parallelism technique used in distributed training, from the basics up to full 3D hybrid setups.
- **`notebooks/`** — runnable PyTorch notebooks that actually implement the core techniques on real hardware (CPU + a single CUDA GPU), so you can see the mechanics work, not just read about them.

Every doc file is written in simple, plain English. Any hard word is explained at the top of the file, in a **Keywords** section, before it's used in the text.

---

## Folder Structure

```
GradientMesh/
│   README.md
│
├───docs/
│       01-introduction.md
│       02-data-parallelism.md
│       03-model-parallelism.md
│       04-tensor-parallelism.md
│       05-pipeline-parallelism.md
│       06-hybrid-model-pipeline.md
│       07-hybrid-tensor-pipeline.md
│       08-hybrid-tensor-model-pipeline.md
│       09-hybrid-full-3d-with-data.md
│       10-all-reduce-and-network.md
│       11-fault-tolerance.md
│       12-choosing-strategy.md
│
└───notebooks/
        01-data-parallel-cpu-gpu-ann.ipynb
        02-model-parallel-cpu-gpu-ann.ipynb
        03-tensor-parallel-cpu-gpu-ann.ipynb
```

---

## `docs/` — Concepts, In Order

### The Building Blocks

| File | Covers |
|---|---|
| [01-introduction.md](./docs/01-introduction.md) | Why we split training across GPUs at all — memory limits, time limits, and why this matters for inference too, not just training. |
| [02-data-parallelism.md](./docs/02-data-parallelism.md) | Copy the whole model onto every GPU, split the data, average the gradients. The simplest and most common technique. |
| [03-model-parallelism.md](./docs/03-model-parallelism.md) | Cut the model itself into pieces by layers, one piece per GPU. Includes how to decide *where* to cut layers, and how to handle a mix of weak and strong GPUs. |
| [04-tensor-parallelism.md](./docs/04-tensor-parallelism.md) | Split a single layer's weight matrix across GPUs, for layers too big or too heavy for one GPU alone. |
| [05-pipeline-parallelism.md](./docs/05-pipeline-parallelism.md) | Stream small micro-batches through model stages like an assembly line, to keep GPUs busy instead of idle. |

### Hybrid Parallelism — Simple to Complex

| File | Covers |
|---|---|
| [06-hybrid-model-pipeline.md](./docs/06-hybrid-model-pipeline.md) | The simplest hybrid: layer-splitting done the smart way, with micro-batching added to reduce idle GPU time. |
| [07-hybrid-tensor-pipeline.md](./docs/07-hybrid-tensor-pipeline.md) | Adds single-layer splitting (tensor parallelism) inside each pipeline stage, using fast links inside a machine and normal links between machines. |
| [08-hybrid-tensor-model-pipeline.md](./docs/08-hybrid-tensor-model-pipeline.md) | Adds uneven, hardware-aware stage sizing on top — for real models where some sections are much heavier than others. |
| [09-hybrid-full-3d-with-data.md](./docs/09-hybrid-full-3d-with-data.md) | The most complete setup: everything above, duplicated with Data Parallelism for real training speed. This is what large real-world clusters actually run (3D Parallelism). |

### Making It Work in Practice

| File | Covers |
|---|---|
| [10-all-reduce-and-network.md](./docs/10-all-reduce-and-network.md) | How GPUs actually exchange and combine data — Ring All-Reduce, NCCL, and bandwidth vs. latency. |
| [11-fault-tolerance.md](./docs/11-fault-tolerance.md) | How node failures are detected, and what recovery looks like differently in data-parallel vs. model/pipeline/tensor-parallel setups. |
| [12-choosing-strategy.md](./docs/12-choosing-strategy.md) | A practical decision guide and checklist for picking the right technique(s) for your own model and hardware. |

**Recommended reading order:** 1 → 12. The hybrid files (6-9) are ordered from simplest to most complex on purpose. If you already know the basics, jump straight to whichever file covers your current problem.

---

## `notebooks/` — Hands-On PyTorch Implementations

These notebooks build a tiny 2-layer ANN (4 → 8 → 1) and implement each parallelism technique **for real**, using PyTorch. They're written to run on a normal laptop setup: **one CPU + one CUDA GPU** — no multi-GPU machine required.

| Notebook | Technique | What It Does |
|---|---|---|
| [01-data-parallel-cpu-gpu-ann.ipynb](./notebooks/01-data-parallel-cpu-gpu-ann.ipynb) | Data Parallelism | Two full copies of the model — one on CPU, one on GPU. Each trains on its own slice of the batch, then a hand-written `average_gradients()` function averages gradients across both (a mini All-Reduce), keeping both copies in sync. |
| [02-model-parallel-cpu-gpu-ann.ipynb](./notebooks/02-model-parallel-cpu-gpu-ann.ipynb) | Model Parallelism | Layer 1 lives and runs on CPU, Layer 2 lives and runs on GPU. Activations flow CPU → GPU on the forward pass, gradients flow GPU → CPU on the backward pass. |
| [03-tensor-parallel-cpu-gpu-ann.ipynb](./notebooks/03-tensor-parallel-cpu-gpu-ann.ipynb) | Tensor Parallelism | Layer 1's weight matrix is cut into two column-shards — one on CPU, one on GPU. Both shards see the same input, compute their own piece, and the results are concatenated. Includes a proof step showing this is mathematically identical to one un-split layer. |

### Why CPU + GPU Instead of Two GPUs?

These notebooks were built for a machine with an **Intel i5 (CPU)** and one **NVIDIA GTX 1650 (1 CUDA GPU)** — a very common setup for learning, without access to a multi-GPU machine. The CPU stands in for a "second GPU" so you can see the exact same mechanics (data splitting, layer splitting, matrix splitting, gradient averaging) that would normally run across `cuda:0` and `cuda:1`.

**If you get a second CUDA GPU later**, every notebook only needs a one-line change:

```python
# Instead of this (CPU + GPU):
device_a = torch.device("cpu")
device_b = torch.device("cuda:0")

# Just change to (GPU + GPU):
device_a = torch.device("cuda:0")
device_b = torch.device("cuda:1")
```

Nothing else in any notebook needs to change — the model, training loop, and splitting logic are all already written to work with whatever two devices you assign.

### Requirements

```bash
pip install torch
```

That's the only dependency needed to run any of the three notebooks. CUDA support is optional — every notebook automatically falls back to running both "devices" on CPU if no GPU is found, so they still run (just without a real device split) on a CPU-only machine.

---

## How to Use This Repo

1. **New to distributed training?** Start at `docs/01-introduction.md` and read straight through to `docs/12-choosing-strategy.md`.
2. **Want to see the code work?** Open the notebooks in order: `01` (data) → `02` (model) → `03` (tensor). Each one pairs directly with its matching doc file.
3. **Already know the basics, have a specific question?** Jump straight to the relevant doc — e.g. `docs/11-fault-tolerance.md` for failure handling, or `docs/12-choosing-strategy.md` for a decision checklist.

## Source / Further Reading

This repo was put together while studying distributed training in depth, including this article: [Distributed Deep Learning: How I Scale Training From One GPU to a Cluster](https://thelinuxcode.com/distributed-deep-learning-how-i-scale-training-from-one-gpu-to-a-cluster-2026/), along with other sources on data, model, tensor, and pipeline parallelism.