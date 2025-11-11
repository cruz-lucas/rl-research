# Architecture

The project is intentionally small and declarative: every experiment is described as a Hydra
composition and the Python surface area is limited to a handful of focused modules. This document
captures how the pieces fit together so that it is easy to extend the library or justify its design
in a research application.

## End-to-end flow

```
┌──────────┐    ┌───────────────┐    ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│ hydra    │    │ rl_research/  │    │ agents/     │    │ experiment/  │    │ mlflow +     │
│ configs  ├───►│ cli.py        ├───►│ *.py        ├───►│ experiment.py├───►│ artifacts    │
└──────────┘    └───────────────┘    └─────────────┘    └──────────────┘    └──────────────┘
      defaults.yaml          instantiate()           JAX update loop          metrics + npz
```

1. **Configuration** – `rl_research/conf` holds Hydra defaults for environments (`env/`), agents
   (`agent/`), training schedules (`experiment/`), bundled presets (`run/`), and sweeps
   (`sweep/`). Each YAML file only describes parameters; there is no imperative logic.
2. **Composition** – Invoking `uv run rl-run ...` triggers `rl_research/cli.py`. Hydra composes the
   requested config tree, `_prepare_environment()` materialises the environment builder, and
   `_prepare_agent()` wires agent parameters (including optional `autofill` hooks for tabular sizes).
3. **Execution** – The composed experiment is handled by `rl_research/experiment/experiment.py`
   (look at `run_experiment()` and the JIT-ed `rollout()` helper). Agents inherit from
   `rl_research/agents/base.py`, so the training loop stays environment agnostic.
4. **Logging** – Results flow into MLflow through `log_experiment()`, which stores aggregated runs,
   per-seed metrics, and raw `.npz` trajectories under `mlruns/` (or the remote server you point
   MLflow at). Hydra’s working directories live under `outputs/` unless overridden.

## Key modules

| Module | Summary |
| --- | --- |
| `rl_research/cli.py` | Hydra entry-point, sweep orchestration (grid + random), MLflow run naming. |
| `rl_research/experiment/experiment.py` | Pure JAX rollout + training loop shared by all agents. |
| `rl_research/agents/` | R-MAX, MBIE-EB, tabular Q-learning, and the RMAX+MCTS planner used in the PhD work. |
| `rl_research/models/tabular.py` | Light-weight empirical dynamics models (counts + expectation updates). |
| `rl_research/policies/` | Action-selection strategies (greedy, ε-greedy, UCB-style). |
| `scripts/` | Operational helpers: SLURM launcher and MLflow artefact sync. |

## Configuration patterns

- **Base/variant split** – Every agent family (e.g. `conf/agent/rmax_mcts`) provides a `base.yaml`
  with `_target_` definitions and sane defaults. Environment-specific files only override
  interesting parameters.
- **Autofill** – Agent configs can request `autofill.num_states: tabular_num_states` (see
  `rl_research/cli.py`). The CLI inspects the instantiated environment and injects tabular
  dimensions at runtime, eliminating duplicated constants.
- **Runs vs sweeps** – `conf/run/<name>.yaml` always mirrors a reproducible setting from past
  experiments. `conf/sweep/<name>.yaml` contains Cartesian grids or random sampling instructions.
  You can launch a sweep for any preset via `rl-run run=<preset> sweep=<sweep_name>`.

## Training loop internals

- `rollout()` is `jax.jit`-compiled and only depends on the functional environment and tabular
  agent interface; the agent decides how to update itself, and whether it is in training or eval
  mode.
- `run_experiment()` iterates over seeds, alternates training and evaluation episodes according to
  `ExperimentParams`, and stacks the per-episode trajectories into PyTrees so they can be persisted
  without Python loops.
- `log_experiment()` owns everything related to MLflow: parent runs map to agent presets, nested
  runs capture seed-level metrics, and raw arrays are attached as artifacts for post-hoc analysis.

## Outputs and reproducibility

- Hydra stores the concrete config that was executed inside `outputs/<timestamp>/`. Combined with
  the MLflow logs, this gives a complete provenance trail for every experiment that appears in the
  repository or your application material.
- Local, cloud, or SLURM runs share the exact same CLI surface; the only difference is which sweep
  mode (`grid` or `random`) you pass and whether the SLURM array injects `SLURM_ARRAY_TASK_ID`.

Use this file (together with `docs/run_catalog.md`) whenever you need to justify the software
engineering story behind the experiments in papers, theses, or cover letters.
