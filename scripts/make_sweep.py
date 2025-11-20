#!/usr/bin/env python3
"""
Utility to sample hyperparameter configurations and emit per-job commands.

Search space format (Python dict JSON-compatible):
{
  "OptimisticQLearningAgent": {
    "step_size": {"type": "log_uniform", "min": 1e-3, "max": 5e-1},
    "discount": {"type": "choice", "values": [0.9, 0.95, 0.99]},
    "known_threshold": {"type": "int", "min": 5, "max": 50}
  },
  "QLearningAgent": { ... }
}

Supported spec types:
  - choice: pick uniformly from the provided values
  - uniform: continuous sample in [min, max]
  - log_uniform: continuous sample in [min, max] on a log10 scale
  - int: integer sample in [min, max] inclusive
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

# Default space is a starting point; override with --space-file.
DEFAULT_SPACE: Dict[str, Dict[str, Dict[str, Any]]] = {
    "OptimisticQLearningAgent": {
        "step_size": {"type": "log_uniform", "min": 1e-3, "max": 9e-1},
        "known_threshold": {"type": "int", "min": 5, "max": 50},
        "convergence_threshold": {"type": "log_uniform", "min": 1e-4, "max": 1.0},
    },
    "QLearningAgent": {
        "step_size": {"type": "log_uniform", "min": 1e-3, "max": 1.0},
    },
}


def _format_value(val: Any) -> str:
    if isinstance(val, str):
        return f'"{val}"'
    if isinstance(val, bool):
        return "True" if val else "False"
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        return f"{val:.6g}"
    raise TypeError(f"Unsupported value type: {type(val)}")


def sample_value(spec: Dict[str, Any], rng: random.Random) -> Any:
    stype = spec["type"]
    if stype == "choice":
        return rng.choice(spec["values"])
    if stype == "uniform":
        return rng.uniform(spec["min"], spec["max"])
    if stype == "log_uniform":
        low, high = math.log10(spec["min"]), math.log10(spec["max"])
        return 10 ** rng.uniform(low, high)
    if stype == "int":
        return rng.randint(int(spec["min"]), int(spec["max"]))
    raise ValueError(f"Unknown spec type: {stype}")


def sample_bindings(
    space: Dict[str, Dict[str, Any]],
    algorithm: str,
    num_samples: int,
    rng: random.Random,
) -> List[List[str]]:
    if algorithm not in space:
        raise KeyError(f"{algorithm} not in search space keys {list(space.keys())}")
    algo_space = space[algorithm]
    bindings: List[List[str]] = []
    for _ in range(num_samples):
        combo = []
        for name, spec in algo_space.items():
            val = sample_value(spec, rng)
            combo.append(f"{algorithm}.{name}={_format_value(val)}")
        bindings.append(combo)
    return bindings


def render_commands(
    bindings: Sequence[Sequence[str]],
    seeds: Sequence[int],
    config: str,
    group_prefix: str,
) -> List[str]:
    commands: List[str] = []
    for combo_idx, combo in enumerate(bindings):
        group_name = f"{group_prefix}_c{combo_idx}"
        for seed in seeds:
            binding_flags = " ".join(f"--binding {b}" for b in combo)
            binding_flags += (
                f" --binding setup_mlflow.experiment_group={_format_value(group_name)}"
            )
            cmd = (
                f"python -m rl_research.main "
                f"--config {config} "
                f"--seed {seed} "
                f"{binding_flags}"
            )
            commands.append(cmd.strip())
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample hyperparameters and emit runnable commands."
    )
    parser.add_argument(
        "--algorithm",
        required=True,
        help="Agent class name used in gin bindings (e.g., OptimisticQLearningAgent).",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the gin config file passed to rl_research.main.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of hyperparameter combinations to sample.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="Number of seeds per combination (0..seeds-1).",
    )
    parser.add_argument(
        "--space-file",
        type=Path,
        default=None,
        help="Optional JSON file defining the search space. Falls back to built-in defaults.",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=0,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--group-prefix",
        default="sweep",
        help="Prefix for MLflow experiment_group binding.",
    )
    args = parser.parse_args()

    rng = random.Random(args.rng_seed)

    if args.space_file:
        with args.space_file.open() as f:
            space = json.load(f)
    else:
        space = DEFAULT_SPACE

    bindings = sample_bindings(space, args.algorithm, args.samples, rng)
    seeds = list(range(args.seeds))
    commands = render_commands(bindings, seeds, args.config, args.group_prefix)

    print(f"# Sampled {len(bindings)} combos x {len(seeds)} seeds = {len(commands)} jobs")
    print("# Paste the following into a bash array in your SLURM script:")
    print("COMMANDS=(")
    for cmd in commands:
        print(f'  "{cmd}"')
    print(")")
    print('echo "Running job ${SLURM_ARRAY_TASK_ID} / ${#COMMANDS[@]}"')
    print('eval "${COMMANDS[$SLURM_ARRAY_TASK_ID]}"')


if __name__ == "__main__":
    main()
