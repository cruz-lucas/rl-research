# rl-research

This repo is a ongoing research repository made public now only to be linked in PhD application, sorry about the mess.

`rl-research` is a lightweight reinforcement-learning playground for tabular agents built on JAX,
Hydra, and MLflow. Every result in the repository (and in the accompanying PhD application) can be
reproduced by composing YAML configs and launching the single CLI entry point `rl-run`.

## Highlights

- Declarative configuration tree under `rl_research/conf/` with clear separation between environments,
  agents, experiment schedules, presets, and sweeps.
- Tabular agent zoo (R-MAX variants, MBIE-EB, Q-learning, RMAX+MCTS) that shares a functional base
  class so new planners only need to implement their update rule.
- Sweep-aware CLI with SLURM-array support. Grid and random searches are described entirely in YAML
  and logged consistently via MLflow.
- Rich logging story: every run creates a Hydra output directory, MLflow experiment, and zipped `.npz`
  artifacts for raw trajectories and agent state snapshots.
- Research-ready documentation in `docs/` so the codebase can be referenced in applications, papers,
  or collaboration hand-offs.

## Installation

```bash
uv sync            # installs dependencies
source .venv/bin/activate
```

`uv` pins the exact versions of Hydra, JAX, MLflow, and the custom environment packages referenced
in the configs. Using `pip install -e .[dev]` works as well if you prefer stock virtualenvs.

## Quickstart

Default run (Decision-Time RMAX on Double GoRight):

```bash
uv run rl-run
```

Reproduce additional presets (see the [run catalog](docs/run_catalog.md) for the complete list):

```bash
uv run rl-run run=doublegoright_mcts
uv run rl-run run=riverswim_rmax_dtp
uv run rl-run run=sixarms_mbieeb
```

Standard Hydra overrides apply to any field:

```bash
# swap the agent while keeping the rest of the preset intact
uv run rl-run run=riverswim_rmax agent=rmax_mcts/empirical_riverswim

# adjust training schedule on the fly
uv run rl-run run=sixarms_rmax experiment.total_train_episodes=50
```

## Configuration layout

- `conf/env/` – environment builders plus optional expectation models for model-based agents.
- `conf/agent/` – agent templates with `_target_` definitions and optional `autofill` hints for
  tabular cardinalities (`tabular_num_states`, `tabular_num_actions`) or dynamics models.
- `conf/experiment/` – reusable training schedules (`default`, `goright`, `riverswim`, `sixarms`).
- `conf/run/` – curated presets that mirror classic tabular benchmarks; see
  `docs/run_catalog.md` for a searchable reference.
- `conf/sweep/` – Cartesian and random search definitions. Attach any sweep to any preset with
  `rl-run run=<preset> sweep=<sweep_name>`.

Example agent override (`conf/agent/dt_rmax_nstep/doublegoright.yaml`):

```yaml
defaults:
  - base

name: rmax_dt_rollout_m2000_behaviorpolicylr01

params:
  discount: 0.9
  learning_rate: 0.1
  behavior_learning_rate: 0.1
  horizon: 10
  m: 2000
  r_max: 6.0
```

Hydra composes the `base` file (which defines `_target_`, builder kwargs, and autofill directives)
with this environment-specific override at runtime, so each YAML stays tiny and readable.

## Hyperparameter sweeps

Attach a sweep to any preset:

```bash
uv run rl-run run=doublegoright_fullyobs_mcts_rmax_empirical sweep=rmax_mcts
```

`conf/sweep/rmax_mcts.yaml` (grid):

```yaml
mode: grid
parameters:
  agent.params.m: [500, 1000, 2000]
  agent.params.max_depth: [5, 10, 20]
name_template: m{agent.params.m}_depth{agent.params.max_depth}
```

Random sweeps (`conf/sweep/rmax_mcts_random.yaml`) deterministically sample one configuration per
SLURM array index or sequentially on a single machine:

```bash
# local: run all random draws sequentially
uv run rl-run run=doublegoright_rmax_mcts sweep=rmax_mcts_random

# cluster: submit an array job so each task evaluates its own sample
sbatch --array=0-63 scripts/job.sh sweep=rmax_mcts_random
```

The CLI now emits structured logs (see `rl_research/cli.py`) so you always know which array index or
trial is running, and MLflow run names automatically include the sweep suffix.

## Scripts & automation

Operational helpers live in [`scripts/`](scripts/README.md):

- `job.sh` – SLURM launcher that requests sensible defaults, prints diagnostics, and executes
  `uv run --offline --active rl-run ...`. Works interactively thanks to guarded environment
  variables.
- `fetch_mlruns.sh` – syncs the `mlruns/` directory from a remote machine via SSH, keeping
  application figures and dashboards in lockstep.

## Documentation map

- [`docs/architecture.md`](docs/architecture.md) – explains the end-to-end flow and core modules.
- [`docs/run_catalog.md`](docs/run_catalog.md) – authoritative list of every `run=` preset.
- [`scripts/README.md`](scripts/README.md) – operational notes for SLURM + artifact syncing.

Use these references in proposals or thesis appendices to highlight the software engineering story.

## Logging & outputs

MLflow logging is enabled by default (`mlflow.enabled=true`). Each run logs:

- parent parameters (agent/env/experiment config),
- nested seed metrics (`train/*`, `eval/*` curves), and
- zipped `.npz` artifacts for raw trajectories and agent states.

Hydra retains a copy of the resolved configuration and stdout under `outputs/<timestamp>/`, which,
combined with MLflow, gives complete reproducibility for future audits or publications.
