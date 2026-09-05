# 01. Why Study Distributed Training?

**Keywords used in this file:** GPU (a chip good at doing many small math operations at once), Model (the file that holds all the learned numbers, called weights), Batch (a group of data samples processed together), Inference (using a trained model to get an answer, not training it).

---

## The Problem in One Line

Modern models are too big or too slow to train (or even run) on a single GPU, so we split the work across many GPUs.

## Why a Single GPU Is Not Enough

A GPU has two limited things:

1. **Memory** — GPUs have 24GB, 40GB, or 80GB of memory. A large model's weights, plus the extra numbers needed for training (gradients, optimizer states), can easily need 10x more memory than the weights alone.
2. **Time** — Even if a model fits, training on one GPU can take weeks or months. Businesses and researchers can't wait that long.

So we spread the work across multiple GPUs, on one machine or across many machines (a **cluster**).

## It's Not Just About Training

People often think "distributed" only means training. That's wrong. Two situations need splitting across GPUs:

- **Training** — updating the model's weights using data, again and again.
- **Inference** — using an already-trained model to answer questions or generate output.

A model like a large language model can be so big that even just *loading* it for inference needs more memory than one GPU has. So the same splitting ideas (mainly model parallelism and tensor parallelism) are used at inference time too — just without the extra memory needed for gradients and optimizer states, since we are not updating weights anymore.

## The Two Things That Get Split

When we say "distributed training," we are really splitting one (or both) of these:

| What gets split | Called | Simple idea |
|---|---|---|
| The **data** | Data Parallelism | Same full model copied on every GPU, each GPU sees a different slice of data |
| The **model** | Model / Tensor / Pipeline Parallelism | Model is cut into pieces, each GPU holds only a piece |

This repo covers both, one file at a time, then shows how real systems mix them together (**hybrid parallelism**).

## What This Repo Covers

1. Data Parallelism — copy the model, split the data
2. Model Parallelism — split the model by layers, across GPUs
3. Tensor Parallelism — split individual layers/matrices across GPUs
4. Pipeline Parallelism — split the model by stages, and stream batches through like an assembly line
5. Hybrid Parallelism — combining the above for real, large clusters
6. All-Reduce and Network Communication — how GPUs actually talk to each other and stay in sync
7. Fault Tolerance — what happens when a GPU or a whole machine dies
8. Choosing a Strategy — how to decide what to use, based on your GPUs, model size, and network

## A Simple Mental Model

Think of training a giant model like building a car in a factory:

- **Data Parallelism** = many identical factories, each building the same car design, using different batches of parts, then comparing notes at the end of each shift.
- **Model Parallelism** = one factory, but the assembly line is so long it's split across multiple buildings; a half-built car physically moves from Building A to Building B.
- **Tensor Parallelism** = one workstation on the assembly line is too big for one team, so two teams share that single workstation, each doing half the work on the same car part, at the same time.
- **Pipeline Parallelism** = the assembly line has stages (paint, engine, wheels), and while stage 2 works on car #5, stage 1 has already started car #6, so no stage sits idle.

Keep this mental picture — every diagram in this repo is a more detailed version of it.

---

**Next:** [02. Data Parallelism](./02-data-parallelism.md)
