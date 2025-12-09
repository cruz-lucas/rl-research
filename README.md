# RL Research (WIP)

Tabular reinforcement-learning sandbox for ongoing research on exploration and planning. Experiments are expressed with Gin configs, run with JAX/Flax-based agents, and tracked with MLflow (including tabular agent-state artifacts for inspection).

## Highlights
- JAX-first training loop with Gin-configurable agents, buffers, and environments.
- Tabular agents (Q-learning, optimistic Monte Carlo and Q-learning, RMax, MCTS) plus optional empirical/hand-coded models.
- Functional GoRight environment with precomputed transitions and optional `pygame` rendering.
- MLflow logging backed by SQLite (`mlruns.db`) with agent-state artifacts per run.
- Reproducible dependency management via `uv` and the committed `uv.lock`.

## Repository Layout
- `rl_research/main.py` – Tyro CLI that wires configs, runs multi-seed experiments, and logs to MLflow.
- `rl_research/agents/` – Implementations of tabular agents (e.g., QLearning, RMax, MCTS, optimistic variants).
- `rl_research/environments/` – Functional JAX GoRight environment and base protocol.
- `rl_research/models/` – Tabular dynamics models (empirical and static expectations).
- `rl_research/configs/` – Gin configs for common setups; tweak via `--binding` overrides.
- `scripts/` – Slurm job wrapper, hyperparameter sweeps, MLflow migration, plotting utilities, and GoRight visualizer.
- Generated: `mlruns/`, `mlruns.db`, `outputs/`, `tmp/` (left out of version control).

## Setup (uv)
Requirements: Python 3.11+ and [`uv`](https://github.com/astral-sh/uv). If `uv` is missing, install with `pip install uv` or follow the official instructions.

```bash
cd /Users/lucascruz/Documents/GitHub/rl-research
uv sync                       # Base runtime deps
uv sync --extra pygame        # Add optional rendering support
uv sync --group dev           # Tooling (ruff, pytest, ipykernel)
# Optional: source .venv/bin/activate  # uv run will auto-use the venv
```

`uv` will honor `uv.lock` for reproducibility and uses the custom Gymnasium source declared in `pyproject.toml`.

## Running Experiments
Gin controls nearly everything. Pick a config from `rl_research/configs/` and optionally override parameters with `--binding`:

```bash
uv run python -m rl_research.main \
  --config rl_research/configs/double_goright/qlearning_randomwalk.gin \
  --binding run_loop.train_episodes=100 \
  --binding OptimisticQLearningAgent.step_size=0.05
```

Notes:
- Bindings apply to any Gin-configured symbol (agents, buffers, environment params, or training loop settings).

### MLflow Tracking
- Default tracking URI: `sqlite:///mlruns.db` (created alongside `mlruns/` artifacts).
- Launch the UI locally (after runs exist):

  ```bash
  uv run mlflow ui \
    --backend-store-uri sqlite:///mlruns.db \
    --default-artifact-root "$(pwd)/mlruns"
  ```

- Agent state tensors (e.g., Q-values, visit counts) are logged as `artifacts/agent_states.npz`.

### Utilities
- `scripts/plot_agent_states.py` – Visualize tabular agent tensors from an MLflow run (supports PDF export).
- `scripts/play_goright.py` – Interactive GoRight viewer (requires `--extra pygame`).
- `scripts/submit_sweep.py` – Sample hyperparameters and submit Slurm arrays (wraps `scripts/job.sh`).
- `scripts/migrate_mlflow_store.py` – Migrate file-backed MLflow runs into SQLite.
- `scripts/job.sh` – Slurm job wrapper that executes `uv run python -m rl_research.main ...`.

## Development
- Lint/format: `uv run ruff check .`
- Tests: `uv run pytest` (add tests alongside changes; none are checked in yet).
- Type checking: `uv run pyright` (basic mode enabled in `pyproject.toml`).

## Status & License
- Work-in-progress research code; APIs and configs may change without notice.
- Licensed under the MIT License (see `LICENSE`).
