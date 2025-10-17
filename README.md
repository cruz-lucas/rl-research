## Overview

`rl-research` is a lightweight reinforcement-learning research framework focused on rapid experimentation with JAX-based agents and environments. It provides:

- Consistent abstractions for agents, environments, and experiment loops.
- Hydra-driven configuration to compose experiments and sweeps.
- Built-in MLflow integration for experiment tracking.
- Utilities to inspect state-action visitation counts and Q-values after every episode.
- Dispatch helpers for local execution and Compute Canada SLURM clusters.
- Hyperparameter sweep utilities that keep search and evaluation phases separate.

The repository is designed to interoperate with agents and environments defined in separate projects (e.g., `goright`, `riverswim`, or other JAX-based codebases) through a small registry or dynamic-import mechanism.

## Installation

```bash
uv sync  # or pip install -e .
```

## Quickstart

Run an example experiment with the built-in random policy agent on the tabular `LineWorld` environment:

```bash
python -m rl_research.cli
```

The command uses Hydra to load `src/rl_research/config/config.yaml`, which composes agent, environment, experiment, and tracking sub-configs. Override any value via standard Hydra syntax, for example:

```bash
python -m rl_research.cli agent.name=random env.params.length=10 experiment.total_episodes=50 tracking.type=mlflow tracking.params.tracking_uri=/tmp/mlruns
```

After each episode, the CLI prints a sample of visitation counts and Q-values (where available) to support rapid inspection.

## Agents and Environments

- Implement new agents by subclassing `rl_research.agents.base.Agent` and registering them with `rl_research.agents.AGENTS`.
- Implement new environments by subclassing `rl_research.envs.base.Environment` and registering via `rl_research.envs.ENVIRONMENTS`.
- Alternatively, specify a fully-qualified class path in the Hydra config (e.g., `agent.target=goright.agents.GoRightAgent`) to instantiate external implementations without touching the registry.

`ExperimentRunner` (see `src/rl_research/core/experiment.py`) coordinates the interaction loop, collects per-episode statistics, and routes metrics to the tracker. `EpisodeStatsCollector` (`src/rl_research/core/stats.py`) aggregates visitation counts and optional Q-values derived from the agent.

### Example: RiverSwim + Minimal Q-Learning

Install the external packages in the same environment (editable installs keep your local changes live):

```bash
pip install -e ../riverswim
pip install -e ../minimal_agents
```

Run a RiverSwim experiment using the minimal-agents Q-learning implementation:

```bash
python -m rl_research.cli \
  agent=minimal_q_learning \
  env=riverswim \
  experiment.total_episodes=200 \
  experiment.training_seeds='[0,1,2]' \
  tracking.type=mlflow tracking.params.tracking_uri=/tmp/mlruns
```

- `agent=minimal_q_learning` activates the adapter around `minimal_agents.agents.q_learning.QLearningAgent`. The agent config reuses the RiverSwim `n_states` via Hydra interpolation, so no extra overrides are needed.
- `env=riverswim` selects the wrapper around `riverswim.river_swim_env.RiverSwimEnv`.
- Adjust seeds/episodes or pass extra overrides (e.g., `agent.params.epsilon=0.05`, `env.params.p_right=0.2`) to sweep hyperparameters or alter the task.

## Experiment Tracking

- Use `tracking.type=mlflow` to enable MLflow logging. Override `tracking.params.tracking_uri` to point at a remote or local server.
- Set `tracking.type=noop` (the default) for quick local iterations without persistence.
- Metrics emitted from the agent’s `update` and `on_episode_end` hooks are logged automatically.

## Hyperparameter Searches

`rl_research.dispatch.sweeps.HyperparameterSearch` generates Hydra override combinations for grid searches:

```python
from pathlib import Path
from rl_research.dispatch.base import ExperimentJob
from rl_research.dispatch.sweeps import HyperparameterSearch

base = ExperimentJob(
    name="line_world_sweep",
    overrides=("agent.name=random",),
    working_dir=str(Path.cwd()),
)

search = HyperparameterSearch(
    name="dqn_grid",
    base_job=base,
    grid={
        "agent.params.learning_rate": [3e-4, 1e-3],
        "agent.params.epsilon": [0.01, 0.1],
    },
)

for job in search.iter_jobs():
    print(job.hydra_command())
```

Use `EvaluationPlan` to re-run the best hyperparameters on a fresh set of seeds once the search completes.

## Dispatching Jobs

- `rl_research.dispatch.local.LocalDispatcher` runs jobs sequentially on the local machine.
- `rl_research.dispatch.compute_canada.ComputeCanadaDispatcher` generates (and optionally submits) SLURM scripts configured for Compute Canada clusters. Configure SLURM resources, modules, environment variables, and pre-run commands in one place.

Example:

```python
from pathlib import Path
from rl_research.dispatch.base import ExperimentJob
from rl_research.dispatch.compute_canada import ComputeCanadaDispatcher

job = ExperimentJob(
    name="line_world_random_seed0",
    overrides=("experiment.training_seeds=[0]",),
    working_dir=str(Path.cwd()),
)

dispatcher = ComputeCanadaDispatcher(
    output_dir=Path("jobs"),
    submit=False,  # set True to call sbatch automatically
)

dispatcher.dispatch(job)  # writes jobs/line_world_random_seed0.sbatch
```

## State-Action Diagnostics

`EpisodeStatsCollector` converts observations and actions into hashable keys and tracks visitation counts. If the agent implements `estimate_action_values`, the collector also snapshots Q-values per visited state-action pair. These diagnostics are surfaced in the CLI and can be persisted via the tracker or custom hooks.
