WarSim Pro ⚔️

Production-Grade 2-D Multi-Agent Battle Simulator
Branch: main

Overview

WarSim Pro is a research-grade rewrite of a 2-D multi-agent combat simulator, built with a focus on determinism, modularity, and high performance.

The pipeline is cleanly decomposed into:
perception → ego-frame → policy → mask → sampling → engine → grid

Explicit ABI contracts define actions, directions, observations, and masks. Performance-critical paths are vectorised and GPU-friendly, enabling thousands of agents to run at ≥60 ticks/second on a 128×128 grid 🚀.

Design Tenets 🧩

Contracts over code – All directional/action layouts follow rigid schemas, simplifying rotation and masking.

Strict modularity – Each layer is independently swappable; seam tests pin down invariants.

Performance discipline – Struct-of-Arrays (SoA) tensors and batched inference minimise Python overhead.

Observability & testability – Built-in stats, logs, and property tests catch regressions early.

Determinism knobs – Seeds and fixed initialisation guarantee reproducibility.

System Architecture 📊

The runtime consists of:

Tick Engine – Updates health, movement, collisions, and scoring.

Agents Registry – Manages SoA data and brain assignments.

Ego-Frame Runtime – Rotates observations into an ego-centric frame, then unrotates logits.

Bucketer – Groups agents by brain topology for efficient batched inference.

Mask Builder – Enforces legal action constraints.

Sampler – Draws final actions from masked logits.

All components interlock under a deterministic, test-driven runtime.

Repository Layout
war_simulation/
├── agent/       # Agent brains (actor-critic, encoders)
├── engine/      # Core engine: ticks, grid, raycasting, mapgen
├── rl/          # PPO and other RL algorithms
├── scripts/     # Training & launch scripts
├── tests/       # Unit & property tests pinning ABI contracts
├── config.py    # Centralised knobs & hyper-parameters

Installation & Quick Start
Prerequisites

Windows 11

Python 3.10

CUDA 12.x

PyTorch ≥ 2.1

GPU: RTX 3060 (6 GB VRAM recommended)

Setup
git clone https://github.com/grumpyCat179/war_simulation.git
cd war_simulation
python -m venv .venv && source .venv/bin/activate  # optional
pip install -r requirements.txt
pip install -e .

Run a Headless Simulation
python -m war_simulation.main --ticks 2000 --grid 128 128 --agents 1000


Per-tick stats are logged to console or file.

Configuration

All runtime knobs live in config.py (or via FWS_* environment variables).
Key options include:

GRID_WIDTH, GRID_HEIGHT – world size (default 128×128)

NUM_ACTIONS – 17 (melee) or 41 (ranged)

RAY_PE_DIM, RAY_ATTN_DIM – ray encoder capacity

MAX_AGENTS – maximum allocated agents

RESPAWN_COOLDOWN_TICKS, RESPAWN_BATCH_PER_TEAM – respawn controls

⚠️ Avoid modifying config dynamically inside hot loops.

Agent Brain

Default per-agent brain = tiny actor-critic:

RayEncoder – projects 64 ray features into 32-D context, with optional ring positional encoding and light self-attention.

MLP Trunk – three SiLU layers.

Output Heads – actor (logits over actions) + critic (value).

Directional Factorisation – actions grouped into 8-wide move/melee/ranged heads for clarity.

This balances throughput with directional context capture.

Tick Loop

Each tick performs:

Observation – raycast & gather environment features.

Ego Rotation – align rays with agent heading.

Bucketing – group by brain topology.

Policy Forward – actor-critic inference, logits unrotated.

Mask & Sampling – legal mask applied, actions sampled.

Engine Update – apply movement, combat, scoring.

(Optional) Mutation – inject small parameter noise or structural variation.

Performance & Testing

Optimisation tips in the Performance Playbook (e.g. set RAY_ATTN_DIM=0 to disable attention).

Batch sizes ≥64 recommended.

Tests under tests/ enforce invariants:

pytest -q

Branch Notes

This main branch is the baseline WarSim Pro:

Emphasises clarity, modularity, and performance.

Other branches may contain experimental features or trained policies; see their respective READMEs.
