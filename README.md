WarSim Pro – Production‑Grade 2‑D Multi‑Agent Battle Simulator

Branch: main

Overview

WarSim Pro is a research‑grade rewrite of a 2‑D multi‑agent combat simulator. It emphasises determinism, modularity and high performance. The simulator cleanly separates the pipeline into perception → ego‑frame → policy → mask → sampling → engine → grid and imposes explicit ABI contracts for actions, directions, observations and masks. Hot paths are vectorised and GPU‑friendly so thousands of agents can run at ≥60 ticks per second on a 128×128 grid
raw.githubusercontent.com
.

Design Tenets

Contracts over code: All directional and action layouts are defined by rigid schemas, making rotation and masking easy to reason about
raw.githubusercontent.com
.

Strict modularity: Each layer is independently swappable; tests pin invariants at the seams
raw.githubusercontent.com
.

Performance discipline: Struct‑of‑Arrays (SoA) tensors and batched inference minimise Python overheads
raw.githubusercontent.com
.

Observability & testability: Built‑in stats, logs and property tests catch regressions quickly
raw.githubusercontent.com
.

Determinism knobs: Global seeds and fixed initialisation ensure reproducibility
raw.githubusercontent.com
.

System Architecture

At a high level the runtime consists of a tick engine that updates health, movement, collisions and scoring; an agents registry that stores SoA data and brains; an ego‑frame runtime that rotates observations into an ego‑centric frame and unrotates logits; a bucketer that groups agents by brain topology; a mask builder that enforces legal actions; and a sampler that produces actions. These components interact as shown in the architecture diagram
raw.githubusercontent.com
.

Repository Layout

The codebase is organised under final_war_sim/ (or war_simulation/ on this branch) with the following key modules
raw.githubusercontent.com
:

Directory/File	Description
agent/	Contains the per‑agent brain implementation (brain.py) and the supporting encoders (e.g. RayEncoder)
raw.githubusercontent.com
.
engine/	Core simulation engine: tick logic, grid representation, raycasting and map generation
raw.githubusercontent.com
.
config.py	Centralised configuration; all knobs for grid size, action space, unit stats, raycasting and respawn settings live here
raw.githubusercontent.com
.
rl/ and scripts/	Reinforcement‑learning algorithms (e.g. PPO) and training/launch scripts.
tests/	Unit and property tests that pin ABI contracts and invariants
raw.githubusercontent.com
.
Installation & Quick Start

Prerequisites: Windows 11, Python 3.10, CUDA 12.x, a GPU such as an RTX 3060 and PyTorch ≥ 2.1
raw.githubusercontent.com
.

Clone and install dependencies

git clone https://github.com/grumpyCat179/war_simulation.git
cd war_simulation
python -m venv .venv && source .venv/bin/activate  # optional
pip install -r requirements.txt
pip install -e .


Run a headless simulation

python -m war_simulation.main --ticks 2000 --grid 128 128 --agents 1000


The program will run without a UI, logging per‑tick stats to the console or a results file.

Configuration

All hyper‑parameters and runtime knobs live in config.py. Examples include:

GRID_WIDTH/GRID_HEIGHT (default 128) – world dimensions
raw.githubusercontent.com
.

NUM_ACTIONS – 17 for melee‐only or 41 for ranged actions
raw.githubusercontent.com
.

RAY_PE_DIM and RAY_ATTN_DIM – control positional encoding and tiny attention in the ray encoder
raw.githubusercontent.com
.

MAX_AGENTS – maximum number of agents to allocate in the registry
raw.githubusercontent.com
.

RESPAWN_COOLDOWN_TICKS, RESPAWN_BATCH_PER_TEAM etc. – tune respawn frequency and diversity
raw.githubusercontent.com
.

Change these constants or set environment variables (FWS_…) before importing modules. Do not modify config dynamically inside hot loops
raw.githubusercontent.com
.

Agent Brain

The default per‑agent brain is a tiny actor‑critic network with:

A RayEncoder that maps the first 64‑dimensional ray features into a 32‑dimensional context via linear projection, optional ring positional encoding and a light 1‑head self‑attention
raw.githubusercontent.com
.

A small MLP trunk with three SiLU layers and two output heads: an actor head producing logits over actions and a critic head estimating the value
raw.githubusercontent.com
.

Factorised directional heads keep the action space organised into 8‑wide groups for moves, melee and ranged attacks
raw.githubusercontent.com
.

This design prioritises throughput while still capturing directional context
raw.githubusercontent.com
.

Tick Loop & Bucketing

During each tick, the engine performs the following high‑level steps
raw.githubusercontent.com
:

Observation: raycast the environment around each alive agent and collect rich self/environment features.

Ego rotation: rotate the ray block so index 0 aligns with the agent’s heading
raw.githubusercontent.com
.

Bucketing: group agents by identical brain topology to enable batched inference
raw.githubusercontent.com
.

Policy forward: run the actor‑critic brain(s) per bucket and unrotate logits back to global coordinates
raw.githubusercontent.com
.

Mask & sampling: apply legal action masks (idle, move, melee, ranged)
raw.githubusercontent.com
, then sample actions and step the engine.

Mutation (optional): mutate brains gently by adding noise, widening or pruning parameters
raw.githubusercontent.com
.

Performance & Testing

Refer to the performance playbook for optimisation tips such as disabling attention (RAY_ATTN_DIM=0) and keeping batch sizes ≥64
raw.githubusercontent.com
. Unit and property tests are provided under tests/ and can be run with pytest -q
raw.githubusercontent.com
.

Branch‑Specific Notes

The main branch represents the baseline WarSim Pro implementation. It prioritises clarity, modularity and performance. Other branches may introduce experimental features or trained brains; see their READMEs for details.
