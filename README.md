## Overview

`rl-research` is a lightweight reinforcement-learning playground built around JAX tabular agents. The codebase now centres on declarative configuration: every environment, agent, experiment schedule, and hyperparameter sweep is described in YAML and composed at runtime through Hydra. Key features:

- Structured configs under `rl_research/conf/` for environments, agents, training loops, pre-defined runs, and sweeps.
- Single entry-point CLI (`rl-run`) that instantiates experiments, launches training, and logs results to MLflow.
- Auto-populated tabular dimensions—configs can request `num_states`, `num_actions`, or model dynamics directly from the environment wrapper.
- Optional grid-search sweeps described entirely in YAML; each combination is executed sequentially with consistent logging metadata.
- Examples for GoRight, RiverSwim, and SixArms reproduced as `run=` presets so experiments stay reproducible without copy/pasting code.

## Installation

```bash
uv sync  # or pip install -e .
```

This installs the project together with Hydra, MLflow, JAX, and the companion environment packages referenced in the configs.

## Quickstart

Run the default Decision-Time RMAX planner on the double GoRight task:

```bash
uv run rl-run
```

The command loads `rl_research/conf/experiment.yaml`, which by default composes:

- `env: doublegoright`
- `agent: dt_rmax_nstep/doublegoright`
- `experiment: default`

To reproduce other examples, pick the corresponding preset from `conf/run`:

```bash
uv run rl-run run=doublegoright_mcts
uv run rl-run run=riverswim_rmax_dtp
uv run rl-run run=sixarms_mbieeb
```

Overrides follow standard Hydra syntax. For instance, reuse the RiverSwim setup but swap in vanilla RMAX:

```bash
uv run rl-run run=riverswim_rmax
```

Fine-grained overrides work on any field, e.g. shorter training:

```bash
uv run rl-run run=riverswim_rmax experiment.total_train_episodes=10
```

## Config Structure

All YAML configs live under `rl_research/conf`:

- `env/` — environment constructors plus optional expectation models for model-based agents.
- `agent/` — agent templates. Each entry declares the builder target, immutable parameter struct, and optional `autofill` hints (e.g. `tabular_num_states`).
- `experiment/` — training schedules (`ExperimentParams`).
- `run/` — convenience compositions that mirror the original Python examples.
- `sweep/` — grid definitions for hyper-parameter searches.

Example (`conf/agent/dt_rmax_nstep/doublegoright.yaml`):

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

The shared `base` file sets the `_target_` references and `autofill` behaviour so environment-specific files only override the interesting pieces.

## Hyperparameter Sweeps

Attach a sweep config to any run by overriding `sweep=`. Each sweep lists the keys to vary and an optional `name_template` used for MLflow run naming:

```bash
uv run rl-run run=doublegoright_rmax_dtp sweep=doublegoright_dt_rmax_grid
```

`conf/sweep/doublegoright_dt_rmax_grid.yaml`:

```yaml
mode: grid
parameters:
  agent.params.learning_rate: [0.05, 0.1, 0.2]
  agent.params.m: [500, 1000, 2000]
name_template: lr{agent.params.learning_rate}_m{agent.params.m}
```

Each combination is executed sequentially; MLflow receives both the original parameters and the sweep-specific overrides.

## Extending the Library

1. **New environment** — create `conf/env/<name>.yaml` with a `builder` target and optional `params`/`expectation_model`.
2. **New agent configuration** — add `conf/agent/<agent_family>/<variant>.yaml`. Use `autofill` options to inherit tabular dimensions or the environment’s expectation model.
3. **Training schedule** — drop a file under `conf/experiment/` or override values directly via the CLI.
4. **Preset run** — compose the pieces in `conf/run/<experiment>.yaml` so collaborators can reproduce results with `rl-run run=<experiment>`.

The CLI automatically propagates environment sizes into agent configs when `autofill` requests `tabular_num_states` or `tabular_num_actions`, so you rarely need to hard-code them.

## Logging & Outputs

All experiments use MLflow (enabled by default). Metrics are logged per seed, and sweep overrides are recorded as additional parameters. Disable logging with `mlflow.enabled=false`. Episode traces are saved as `.npz` artifacts for later inspection.

Outputs (including Hydra’s working directory) remain under the project `outputs/` folder unless you override `hydra.run.dir`.

## Legacy Examples

The original Python scripts under `examples/` now serve as references only. Every scenario has a matching `run=` config, so you can delete the duplicated boilerplate and rely on YAML composition instead.
