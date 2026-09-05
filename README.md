# ShardedIntelligence

A simple, plain-English guide to every way large models are split across GPUs — for training and for inference.

## What This Covers

This repo explains **Data Parallelism, Model Parallelism, Tensor Parallelism, Pipeline Parallelism**, and then **four hybrid combinations, from simplest to most complex** — how GPUs actually talk to each other (All-Reduce and networking), what happens when a GPU or machine fails, and how to decide which technique to use for your own setup.

Every file is written in easy, simple English. Any hard word is explained at the top of the file, in a **Keywords** section, before it's used.

## Index

### The Building Blocks
1. **[Introduction](./01-introduction.md)** — Why we split training across GPUs at all, and why this matters for inference too, not just training.
2. **[Data Parallelism](./02-data-parallelism.md)** — Copy the model, split the data, average the gradients.
3. **[Model Parallelism](./03-model-parallelism.md)** — Split the model by layers across GPUs; includes how to decide where to cut layers, and handling weak/strong GPUs.
4. **[Tensor Parallelism](./04-tensor-parallelism.md)** — Split a single layer's matrix math across GPUs, for layers too big to handle alone.
5. **[Pipeline Parallelism](./05-pipeline-parallelism.md)** — Stream micro-batches through model stages like an assembly line, to keep GPUs busy.

### Hybrid Parallelism: Simple to Complex
6. **[Hybrid #1: Model + Pipeline](./06-hybrid-model-pipeline.md)** — The simplest combo: layer-splitting done the smart way, with micro-batching to reduce idle GPU time.
7. **[Hybrid #2: Tensor + Pipeline](./07-hybrid-tensor-pipeline.md)** — Adds single-layer splitting on top, using fast links inside a machine and normal links between machines.
8. **[Hybrid #3: Tensor + Model + Pipeline](./08-hybrid-tensor-model-pipeline.md)** — Adds uneven, hardware-aware layer/stage sizing on top, for models with heavy and light sections.
9. **[Hybrid #4: Full 3D (Tensor + Pipeline + Data)](./09-hybrid-full-3d-with-data.md)** — The most complete setup: everything above, duplicated with Data Parallelism for real training speed. This is what large real-world clusters run.

### Making It Work in Practice
10. **[All-Reduce and Network Communication](./10-all-reduce-and-network.md)** — How GPUs actually exchange and combine data, including Ring All-Reduce, and bandwidth vs. latency.
11. **[Fault Tolerance](./11-fault-tolerance.md)** — How failures are detected, and what happens differently in data-parallel vs. model/pipeline/tensor-parallel setups.
12. **[Choosing a Strategy](./12-choosing-strategy.md)** — A practical decision guide and checklist for picking the right technique(s) for your model and hardware.

## How To Read This

Read in order, 1 through 12 — each file builds on the previous one. The hybrid files (6-9) are ordered from simplest to most complex on purpose — read them in order the first time. If you already know the basics, jump straight to whichever file covers your current problem (e.g., go to file 11 if you just want to understand failure handling).

## Source / Further Reading

This repo was put together while studying distributed training in depth, including this article: [Distributed Deep Learning: How I Scale Training From One GPU to a Cluster](https://thelinuxcode.com/distributed-deep-learning-how-i-scale-training-from-one-gpu-to-a-cluster-2026/), along with other sources on data, model, tensor, and pipeline parallelism.
