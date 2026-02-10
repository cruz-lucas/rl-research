#!/usr/bin/env python3
"""
Sample hyperparameters and submit one sbatch array per combination.

Each array runs over seeds (0..seeds-1); bindings and config are forwarded to
scripts/job.sh, which forwards them to rl_research.main.
"""

from __future__ import annotations

import json
import math
import random
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Dict, List, Sequence

import gin
import tyro

from rl_research.main import run_single_seed, setup_mlflow


# Default space is a starting point; override with --space-file.
DEFAULT_SPACE: Dict[str, Dict[str, Dict[str, Any]]] = {
    "BMFMBIEEBAgent": {
        "step_size": {"type": "log_uniform", "min": 1e-3, "max": 1},
        "beta": {"type": "uniform", "min": 0, "max": 100},
    },
    "BMFRmaxAgent": {
        "step_size": {"type": "log_uniform", "min": 1e-3, "max": 1},
        "known_threshold": {"type": "int", "min": 1, "max": 100},
    },
    "OptimisticMonteCarloAgent": {
        "step_size": {"type": "log_uniform", "min": 1e-3, "max": 1},
        "known_threshold": {"type": "int", "min": 1, "max": 50},
    },
    "QLearningAgent": {
        "step_size": {"type": "log_uniform", "min": 1e-4, "max": 1.0},
        "initial_epsilon": {"type": "uniform", "min": 0.3, "max": 1.0},
        "final_epsilon": {"type": "log_uniform", "min": 1e-2, "max": 0.3},
        "anneal_steps": {"type": "int", "min": 1_000, "max": 500_000},
    },
    "DQNAgent": {
        "learning_rate": {"type": "log_uniform", "min": 1e-6, "max": 0.8},
        "eps_start": {"type": "uniform", "min": 0.3, "max": 1.0},
        "eps_end": {"type": "log_uniform", "min": 1e-2, "max": 0.3},
        "eps_decay_steps": {"type": "int", "min": 1_000, "max": 500_000},
        "target_update_freq": {"type": "choice", "values": list([2**i for i in range(6, 16)])},
        "max_grad_norm": {"type": "uniform", "min": 0.1, "max": 15.0},
    },
    "DRMAgent": {
        "known_threshold": {"type": "int", "min": 1, "max": 500},
        "num_update_epochs": {"type": "int", "min": 1, "max": 1},
        "learning_rate": {"type": "log_uniform", "min": 1e-6, "max": 0.8},
        "target_update_freq": {"type": "choice", "values": list([2**i for i in range(6, 16)])},
        "max_grad_norm": {"type": "uniform", "min": 0.1, "max": 15.0},
    },
    "MCTSAgent": {
        "num_simulations": {"type": "int", "min": 1, "max": 500},
        "rollout_depth": {"type": "int", "min": 1, "max": 50},
        "ucb_c": {"type": "uniform", "min": 0, "max": 100},
    },
    "RMaxAgent": {
        "known_threshold": {"type": "int", "min": 1, "max": 1_000},
        "convergence_threshold": {"type": "log_uniform", "min": 1e-9, "max": 1e-4},
    },
    "DelayedQLearningAgent": {
        "update_threshold": {"type": "int", "min": 1, "max": 500},
        "epsilon": {"type": "log_uniform", "min": 1e-5, "max": 20},
    },
    "params": {
        # "ReplayBuffer.buffer_size": {"type": "choice", "values": list([2**i for i in range(6, 12)])},
        # "FlatteningReplayBuffer.buffer_size": {"type": "choice", "values": list([2**i for i in range(10, 18)])},
        # "run_loop.minibatch_size": {"type": "choice", "values": list([2**i for i in range(4, 12)])},
        # "run_loop.update_frequency": {"type": "int", "min": 1, "max": 50},
        # "run_loop.num_minibatches": {"type": "int", "min": 1, "max": 1},
        # "run_loop.warmup_steps": {"type": "choice", "values": list([2**i for i in range(1, 14)])},
    },
}


@dataclass
class Args:
    """Arguments for submitting sbatch arrays for sampled hyperparameter combos."""

    config: Annotated[
        Path,
        tyro.conf.arg(help="Path to the gin config file passed to rl_research.main."),
    ]
    samples: Annotated[
        int, tyro.conf.arg(help="Number of hyperparameter combinations to sample.")
    ] = 100
    seeds: Annotated[
        int,
        tyro.conf.arg(help="Number of seeds per combination (drives --array size)."),
    ] = 10
    space_file: Annotated[
        Path | None,
        tyro.conf.arg(
            help="Optional JSON file defining the search space. Falls back to defaults."
        ),
    ] = None
    rng_seed: Annotated[
        int, tyro.conf.arg(help="Random seed for reproducible sampling.")
    ] = 0
    group_prefix: Annotated[
        str, tyro.conf.arg(help="Prefix for MLflow experiment_group binding.")
    ] = "sweep"
    job_script: Annotated[
        Path,
        tyro.conf.arg(help="Path to the sbatch script that calls rl_research.main."),
    ] = Path("scripts/single_seed_job.sh")
    sbatch_opt: Annotated[
        List[str], tyro.conf.arg(help="Extra sbatch options (repeatable).")
    ] = field(default_factory=list)
    dry_run: Annotated[
        bool, tyro.conf.arg(help="Print sbatch commands without submitting.")
    ] = False


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
    param_space = space.get("params", {})
    bindings: List[List[str]] = []
    for _ in range(num_samples):
        combo: List[str] = []
        for name, spec in algo_space.items():
            val = _sample_value(spec, rng)
            combo.append(f"{algorithm}.{name}={_format_value(val)}")

        for name, spec in param_space.items():
            val = _sample_value(spec, rng)
            combo.append(f"{name}={_format_value(val)}")

        bindings.append(combo.copy())
    return bindings


def submit_combo(
    combo_idx: int,
    bindings: Sequence[str],
    seeds: int,
    config: Path,
    group_prefix: str,
    job_script: Path,
    sbatch_opts: Sequence[str],
    dry_run: bool,
) -> None:
    group_name = f"{group_prefix}_c{combo_idx}"
    array_flag = f"--array=0-{seeds - 1}"
    cmd: List[str] = ["sbatch", array_flag]
    cmd.extend(sbatch_opts)
    cmd.append(str(job_script))
    cmd.append(str(config))
    cmd.extend(bindings)
    cmd.append(f"setup_mlflow.experiment_group={_format_value(group_name)}")

    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def infer_algorithm_from_config(config_path: Path) -> str:
    """Return the agent class name configured in the gin file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    gin.clear_config()
    gin.parse_config_files_and_bindings(
        [str(config_path)],
        bindings=None,
        skip_unknown=True,
    )
    run_bindings = gin.get_bindings("run_single_seed")
    agent_cls = run_bindings.get("agent_cls")
    gin.clear_config()

    if agent_cls is None:
        raise ValueError(f"run_single_seed.agent_cls not set in config {config_path}")

    return getattr(agent_cls, "__name__", str(agent_cls))


def main(args: Args) -> None:
    rng = random.Random(args.rng_seed)

    if args.space_file:
        with args.space_file.open() as f:
            space = json.load(f)
    else:
        space = DEFAULT_SPACE

    algorithm = infer_algorithm_from_config(args.config)
    if algorithm not in space:
        raise KeyError(
            f"Algorithm {algorithm} not in search space keys {list(space.keys())}"
        )

    if args.seeds < 1:
        raise ValueError("--seeds must be >= 1")

    combos = sample_bindings(space, algorithm, args.samples, rng)

    print(
        f"# Submitting {len(combos)} combos for {algorithm} with "
        f"{args.seeds} seeds each"
    )
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
    main(tyro.cli(Args))
