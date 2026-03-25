#!/usr/bin/env python3
"""
Sample hyperparameters and either submit Slurm arrays or generate local commands.

Each combination runs over seeds (0..seeds-1). In `sbatch` mode, bindings and
config are forwarded to `scripts/single_seed_job.sh`. In `shell` mode, the
script writes a bash launcher you can run inside an interactive allocation.
"""

from __future__ import annotations

import json
import math
import random
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Sequence

import gin
import tyro


TARGET_UPDATE_CHOICES = [2**i for i in range(6, 16)]


# Default space is a starting point; override with --space-file.
DEFAULT_SPACE: Dict[str, Dict[str, Dict[str, Any]]] = {
    "ReplaybasedMBIEEB": {
        "step_size": {"type": "log_uniform", "min": 1e-3, "max": 1},
        "beta": {"type": "uniform", "min": 0, "max": 100},
    },
    "ReplaybasedRmax": {
        "step_size": {"type": "log_uniform", "min": 1e-3, "max": 1},
        "known_threshold": {"type": "int", "min": 1, "max": 100},
    },
    "QLearningAgent": {
        "step_size": {"type": "log_uniform", "min": 1e-4, "max": 5e-1},
        # "initial_epsilon": {"type": "uniform", "min": 0.3, "max": 1.0},
        # "final_epsilon": {"type": "log_uniform", "min": 1e-2, "max": 0.3},
        # "anneal_steps": {"type": "int", "min": 0, "max": 3_000},
        "reward_bonus": {"type": "int", "min": 0, "max": 10_000},
    },
    "DQNAgent": {
        "learning_rate": {"type": "log_uniform", "min": 1e-6, "max": 0.5},
        "eps_start": {"type": "uniform", "min": 0.3, "max": 1.0},
        "eps_end": {"type": "log_uniform", "min": 1e-2, "max": 0.3},
        "eps_decay_steps": {"type": "int", "min": 1_000, "max": 500_000},
        "target_update_freq": {
            "type": "choice",
            "values": TARGET_UPDATE_CHOICES,
        },
        "max_grad_norm": {"type": "uniform", "min": 0.1, "max": 20.0},
    },
    "DQNRNDAgent": {
        "learning_rate": {"type": "log_uniform", "min": 1e-6, "max": 0.5},
        "rnd_learning_rate": {"type": "log_uniform", "min": 1e-6, "max": 0.5},
        "eps_start": {"type": "uniform", "min": 0.3, "max": 1.0},
        "eps_end": {"type": "log_uniform", "min": 1e-2, "max": 0.3},
        "eps_decay_steps": {"type": "int", "min": 1_000, "max": 500_000},
        "target_update_freq": {
            "type": "choice",
            "values": TARGET_UPDATE_CHOICES,
        },
        "max_grad_norm": {"type": "uniform", "min": 0.1, "max": 20.0},
        "intrinsic_reward_scale": {
            "type": "log_uniform",
            "min": 1e-3,
            "max": 10.0,
        },
        "intrinsic_reward_clip": {
            "type": "choice",
            "values": [1.0, 5.0, 10.0, 20.0],
        },
    },
    "NFQAgent": {
        "learning_rate": {"type": "log_uniform", "min": 1e-6, "max": 0.5},
        "eps_start": {"type": "uniform", "min": 0.3, "max": 1.0},
        "eps_end": {"type": "log_uniform", "min": 1e-2, "max": 0.3},
        "eps_decay_steps": {"type": "int", "min": 1_000, "max": 500_000},
        "max_grad_norm": {"type": "uniform", "min": 0.5, "max": 20.0},
        "num_iters": {"type": "int", "min": 1, "max": 300},
    },
    "RMaxNFQAgent": {
        "learning_rate": {"type": "log_uniform", "min": 1e-6, "max": 0.5},
        "min_visits": {"type": "int", "min": 0, "max": 500},
        "max_grad_norm": {"type": "uniform", "min": 0.5, "max": 20.0},
        "num_iters": {"type": "int", "min": 1, "max": 300},
    },
    "DQNRmaxAgent": {
        "known_threshold": {"type": "int", "min": 1, "max": 500},
        "learning_rate": {"type": "log_uniform", "min": 1e-6, "max": 0.5},
        "target_update_freq": {
            "type": "choice",
            "values": TARGET_UPDATE_CHOICES,
        },
        "max_grad_norm": {"type": "uniform", "min": 0.1, "max": 20.0},
    },
    "RMaxAgent": {
        "known_threshold": {"type": "int", "min": 1, "max": 1_000},
        "convergence_threshold": {"type": "log_uniform", "min": 1e-9, "max": 1e-4},
    },
    # "DelayedQLearningAgent": {
    #     "update_threshold": {"type": "int", "min": 1, "max": 500},
    #     "epsilon": {"type": "log_uniform", "min": 1e-5, "max": 20},
    # },
    "params": {
        # Example overrides:
        # "ReplayBuffer.buffer_size": {
        #     "type": "choice",
        #     "values": [2**i for i in range(12, 18)],
        # },
        # "TrainingConfig.minibatch_size": {
        #     "type": "choice",
        #     "values": [2**i for i in range(0, 12)],
        # },
        # "TrainingConfig.update_frequency": {
        #     "type": "choice",
        #     "values": [2**i for i in range(0, 5)],
        # },
        # "TrainingConfig.num_minibatches": {
        #     "type": "choice",
        #     "values": [2**i for i in range(0, 5)],
        # },
        # "TrainingConfig.warmup_steps": {
        #     "type": "choice",
        #     "values": [2**i for i in range(0, 14)],
        # },
    },
}


@dataclass
class Args:
    """Arguments for sampling hyperparameter combinations."""

    config: Annotated[
        Path,
        tyro.conf.arg(help="Path to the gin config file passed to rl_research.main."),
    ]
    samples: Annotated[
        int, tyro.conf.arg(help="Number of hyperparameter combinations to sample.")
    ] = 50
    seeds: Annotated[
        int,
        tyro.conf.arg(help="Number of seeds per combination."),
    ] = 20
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
    mode: Annotated[
        Literal["sbatch", "shell"],
        tyro.conf.arg(
            help=(
                "Execution mode: submit Slurm arrays with `sbatch`, or generate a "
                "bash launcher for an interactive allocation."
            )
        ),
    ] = "sbatch"
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
    shell_script: Annotated[
        Path | None,
        tyro.conf.arg(
            help=(
                "Output path for the generated launcher when --mode shell. "
                "Defaults to scripts/generated_sweep.sh."
            )
        ),
    ] = None
    shell_parallelism: Annotated[
        int,
        tyro.conf.arg(
            help="Max concurrent runs in the generated launcher when --mode shell."
        ),
    ] = 1


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


def build_group_name(group_prefix: str, combo_idx: int) -> str:
    return f"{group_prefix}_c{combo_idx}"


def build_sbatch_cmd(
    combo_idx: int,
    bindings: Sequence[str],
    seeds: int,
    config: Path,
    group_prefix: str,
    job_script: Path,
    sbatch_opts: Sequence[str],
) -> List[str]:
    group_name = build_group_name(group_prefix, combo_idx)
    array_flag = f"--array=0-{seeds - 1}"
    cmd: List[str] = ["sbatch", array_flag]
    cmd.extend(sbatch_opts)
    cmd.append(str(job_script))
    cmd.append(str(config))
    cmd.extend(bindings)
    cmd.append(f"setup_mlflow.experiment_group={_format_value(group_name)}")
    return cmd


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
    cmd = build_sbatch_cmd(
        combo_idx=combo_idx,
        bindings=bindings,
        seeds=seeds,
        config=config,
        group_prefix=group_prefix,
        job_script=job_script,
        sbatch_opts=sbatch_opts,
    )
    print(shlex.join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def build_local_cmd(
    combo_idx: int,
    seed: int,
    bindings: Sequence[str],
    config: Path,
    group_prefix: str,
) -> List[str]:
    group_name = build_group_name(group_prefix, combo_idx)
    cmd: List[str] = [
        "uv",
        "run",
        "--active",
        "--offline",
        "python",
        "-m",
        "rl_research.main",
        "--config",
        str(config),
        "--seed",
        str(seed),
    ]

    cmd.extend(
        ["--binding", f"setup_mlflow.experiment_group={_format_value(group_name)}"]
    )
    for binding in bindings:
        cmd.extend([binding])
    
    return cmd


def write_shell_script(
    shell_script: Path,
    commands: Sequence[Sequence[str]],
    parallelism: int,
) -> None:
    if parallelism < 1:
        raise ValueError("--shell-parallelism must be >= 1")

    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        "",
        f"cd {shlex.quote(str(Path.cwd().resolve()))}",
        "",
        f"parallelism={parallelism}",
        "",
    ]

    if parallelism == 1:
        for cmd in commands:
            lines.append(shlex.join(list(cmd)))
    else:
        lines.extend(["active_jobs=0", ""])
        for cmd in commands:
            lines.append(f"{shlex.join(list(cmd))} &")
            lines.extend(
                [
                    "((active_jobs+=1))",
                    "if (( active_jobs >= parallelism )); then",
                    "  wait -n",
                    "  ((active_jobs-=1))",
                    "fi",
                    "",
                ]
            )
        lines.append("wait")

    shell_script.parent.mkdir(parents=True, exist_ok=True)
    shell_script.write_text("\n".join(lines) + "\n")
    shell_script.chmod(0o755)


def infer_algorithm_from_config(config_path: Path) -> str:
    """Return the agent class name configured in the gin file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Import the experiment entrypoint so Gin registers `run_loop` and friends.
    import rl_research.main  # noqa: F401

    gin.clear_config()
    gin.parse_config_files_and_bindings(
        [str(config_path)],
        bindings=None,
        skip_unknown=True,
    )
    run_bindings = gin.get_bindings("run_loop")
    agent_cls = run_bindings.get("agent_cls")
    gin.clear_config()

    if agent_cls is None:
        raise ValueError(f"run_loop.agent_cls not set in config {config_path}")

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

    if args.mode == "sbatch":
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
        return

    shell_script = args.shell_script or Path("scripts/generated_sweep.sh")
    local_commands: List[List[str]] = []
    for idx, combo in enumerate(combos):
        for seed in range(args.seeds):
            local_commands.append(
                build_local_cmd(
                    combo_idx=idx,
                    seed=seed,
                    bindings=combo,
                    config=args.config,
                    group_prefix=args.group_prefix,
                )
            )

    write_shell_script(
        shell_script=shell_script,
        commands=local_commands,
        parallelism=args.shell_parallelism,
    )
    print(
        f"# Wrote {len(local_commands)} commands for {algorithm} "
        f"to {shell_script} (parallelism={args.shell_parallelism})"
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
