#!/usr/bin/env python3
"""
Sample hyperparameters and submit one sbatch array per combination.

Each array runs over seeds (0..seeds-1); bindings and config are forwarded to
scripts/job.sh, which forwards them to rl_research.main.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
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


def _sample_value(spec: Dict[str, Any], rng: random.Random) -> Any:
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
            val = _sample_value(spec, rng)
            combo.append(f"{algorithm}.{name}={_format_value(val)}")
        bindings.append(combo)
    return bindings


def submit_combo(
    combo_idx: int,
    bindings: Sequence[str],
    seeds: int,
    config: str,
    group_prefix: str,
    job_script: str,
    sbatch_opts: Sequence[str],
    dry_run: bool,
) -> None:
    group_name = f"{group_prefix}_c{combo_idx}"
    array_flag = f"--array=0-{seeds-1}"
    cmd: List[str] = ["sbatch", array_flag]
    cmd.extend(sbatch_opts)
    cmd.append(job_script)
    cmd.append(config)
    cmd.extend(bindings)
    cmd.append(f"setup_mlflow.experiment_group={_format_value(group_name)}")

    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit sbatch arrays for sampled hyperparameter combos."
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
        default=10,
        help="Number of hyperparameter combinations to sample.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="Number of seeds per combination (drives --array size).",
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
    parser.add_argument(
        "--job-script",
        default="scripts/job.sh",
        help="Path to the sbatch script that calls rl_research.main.",
    )
    parser.add_argument(
        "--sbatch-opt",
        action="append",
        default=[],
        help="Extra sbatch options (e.g., --partition=gpu). Can be repeated.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sbatch commands without submitting.",
    )
    args = parser.parse_args()

    rng = random.Random(args.rng_seed)

    if args.space_file:
        with args.space_file.open() as f:
            space = json.load(f)
    else:
        space = DEFAULT_SPACE

    if args.seeds < 1:
        raise ValueError("--seeds must be >= 1")

    combos = sample_bindings(space, args.algorithm, args.samples, rng)

    print(f"# Submitting {len(combos)} combos with {args.seeds} seeds each")
    for idx, combo in enumerate(combos):
        submit_combo(
            combo_idx=idx,
            bindings=combo,
            seeds=args.seeds,
            config=args.config,
            group_prefix=args.group_prefix,
            job_script=args.job_script,
            sbatch_opts=args.sbatch_opt,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
