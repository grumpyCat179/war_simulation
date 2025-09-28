WarSim Pro – Respawning Noise Branch

Branch: respawning_noise

Overview

This branch builds upon the baseline WarSim Pro simulator. All architectural principles from the main branch apply (modular pipeline, deterministic contracts, SoA data layout, etc.)
raw.githubusercontent.com
. The key change is how defeated agents are brought back into the world: respawn events now include controlled randomness. The goal is to prevent spawn camping and encourage more dynamic engagements.

What’s Different?

In the baseline, agents respawn in batches at fixed interior tiles. In respawning_noise we introduce positional jitter and bias to the respawn algorithm:

Jitter radius: Agents re‑enter the game at random positions within a configurable radius (RESPAWN_JITTER_RADIUS) around their team’s spawn area
raw.githubusercontent.com
. This reduces predictable spawn locations.

Interior bias: A tunable bias (RESPAWN_INTERIOR_BIAS) nudges respawns away from the border and towards safer interior cells
raw.githubusercontent.com
.

Mixed unit composition: You can adjust the fraction of archers in each respawn batch via RESPAWN_ARCHER_SHARE
raw.githubusercontent.com
.

All other systems—perception, ego‑frame, bucketing, masking and mutation—remain unchanged. You can still run the simulation headless or with a viewer, train using PPO or evolve brains.

Configuration

Respawn parameters live in config.py and can be overridden via environment variables:

Knob	Description	Default
RESPAWN_COOLDOWN_TICKS	Minimum ticks before a team can respawn a new batch.	 1000
RESPAWN_BATCH_PER_TEAM	Number of agents respawned per team at each event.	 2
RESPAWN_ARCHER_SHARE	Fraction of archers in the respawn batch
raw.githubusercontent.com
.	 0.50
RESPAWN_INTERIOR_BIAS	Probability of choosing interior cells over border cells
raw.githubusercontent.com
.	 0.75
RESPAWN_JITTER_RADIUS	Radius of positional noise added to spawn points
raw.githubusercontent.com
.	 5

These knobs allow you to explore how stochastic respawns influence learning and emergent behaviours. For example, increasing RESPAWN_JITTER_RADIUS forces agents to re‑orient quickly after revival, while lowering RESPAWN_INTERIOR_BIAS makes spawn locations more exposed.

Getting Started

Follow the installation and quick‑start instructions in the main README. To experiment with respawn noise:

# override respawn settings via environment variables
export FWS_RESPAWN_JITTER=10        # larger jitter radius
export FWS_RESPAWN_INTERIOR_BIAS=0.5 # more spawn diversity
python -m war_simulation.main --ticks 5000 --grid 128 128 --agents 1000


Monitor game dynamics (e.g., kill/death ratios and positional heatmaps) to measure the impact of randomized respawns.

Use Cases

Curriculum learning: Stochastic respawns force agents to adapt to fresh surroundings rather than memorising fixed spawn points.

Anti‑camping experiments: Evaluate whether respawn noise reduces the dominance of spawn‑camping strategies.

Game design: Tune bias and jitter to achieve desired difficulty curves and fairness.

Further Reading

For details on the underlying architecture, agent brains, tick loop and performance tips, see the README on the main branch.
