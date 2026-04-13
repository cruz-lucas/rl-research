#!/usr/bin/env python3
"""
Sample hyperparameters and submit Slurm sweeps.

Modes:
- `sbatch`: submit one Slurm array per sampled hyperparameter combination.
- `shell`: generate a local bash launcher for all sampled runs.
- `packed_sbatch`: pack many runs into each Slurm job using manifest files.
- `packed_resubmit`: resubmit only incomplete packed jobs from an existing batch.
"""

from __future__ import annotations

import json
import math
import random
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Sequence

import gin
import tyro


TARGET_UPDATE_CHOICES = [2**i for i in range(6, 16)]
DQN_HIDDEN_DIMS_CHOICES = [
    [64, 64],
    [128, 128],
    [256, 256],
    [256, 256, 256],
    [512, 256],
    [512, 512, 256],
]
RND_HIDDEN_DIMS_CHOICES = [
    [64, 64],
    [128, 128],
    [256, 128],
    [256, 256],
    [512, 256],
]
ACTIVATION_CHOICES = ["relu", "gelu", "silu"]
NORMALIZATION_CHOICES = ["none", "last", "all"]
OPTIMIZER_CHOICES = ["adam", "adamw", "rmsprop"]
DQN_BUFFER_SIZE_CHOICES = [2**i for i in range(15, 21)]
DQN_MINIBATCH_SIZE_CHOICES = [2**i for i in range(6, 12)]
DQN_WARMUP_CHOICES = [0] + [2**i for i in range(8, 16)]
DQN_NUM_MINIBATCH_CHOICES = [1, 2, 4, 8]


# Default space is a starting point; override with --space-file.
DEFAULT_SPACE: Dict[str, Any] = {
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
        "hidden_dims": {"type": "choice", "values": DQN_HIDDEN_DIMS_CHOICES},
        "activation": {"type": "choice", "values": ACTIVATION_CHOICES},
        "normalization": {"type": "choice", "values": NORMALIZATION_CHOICES},
        "optimizer": {"type": "choice", "values": OPTIMIZER_CHOICES},
        "optimizer_weight_decay": {
            "type": "choice",
            "values": [0.0, 1e-6, 1e-5, 1e-4, 1e-3],
        },
        "optimizer_momentum": {"type": "choice", "values": [0.0, 0.9, 0.95]},
        "optimizer_decay": {"type": "choice", "values": [0.9, 0.95, 0.99]},
        "optimizer_centered": {"type": "choice", "values": [False, True]},
        "learning_rate": {"type": "log_uniform", "min": 1e-6, "max": 1e-2},
        "discount": {"type": "choice", "values": [0.99, 0.995, 0.999]},
        "eps_start": {"type": "uniform", "min": 0.5, "max": 1.0},
        "eps_end": {"type": "log_uniform", "min": 1e-3, "max": 0.3},
        "eps_decay_steps": {"type": "int", "min": 10_000, "max": 1_000_000},
        "target_update_freq": {
            "type": "choice",
            "values": TARGET_UPDATE_CHOICES,
        },
        "max_grad_norm": {"type": "choice", "values": [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]},
        "loss_type": {"type": "choice", "values": ["mse", "huber"]},
        "huber_delta": {"type": "choice", "values": [0.5, 1.0, 2.0, 5.0]},
        "double_q": {"type": "choice", "values": [False]},
        "normalize_observations": {"type": "choice", "values": [False, True]},
        "obs_normalization_clip": {"type": "choice", "values": [3.0, 5.0, 7.0, 10.0]},
    },
    "DQNRNDAgent": {
        "hidden_dims": {"type": "choice", "values": DQN_HIDDEN_DIMS_CHOICES},
        "activation": {"type": "choice", "values": ACTIVATION_CHOICES},
        "normalization": {"type": "choice", "values": NORMALIZATION_CHOICES},
        "optimizer": {"type": "choice", "values": OPTIMIZER_CHOICES},
        "optimizer_weight_decay": {
            "type": "choice",
            "values": [0.0, 1e-6, 1e-5, 1e-4, 1e-3],
        },
        "optimizer_momentum": {"type": "choice", "values": [0.0, 0.9, 0.95]},
        "optimizer_decay": {"type": "choice", "values": [0.9, 0.95, 0.99]},
        "optimizer_centered": {"type": "choice", "values": [False, True]},
        "learning_rate": {"type": "log_uniform", "min": 1e-6, "max": 1e-2},
        "rnd_learning_rate": {"type": "log_uniform", "min": 1e-6, "max": 1e-2},
        "discount": {"type": "choice", "values": [0.99, 0.995, 0.999]},
        "eps_start": {"type": "uniform", "min": 0.5, "max": 1.0},
        "eps_end": {"type": "log_uniform", "min": 1e-3, "max": 0.3},
        "eps_decay_steps": {"type": "int", "min": 10_000, "max": 1_000_000},
        "target_update_freq": {
            "type": "choice",
            "values": TARGET_UPDATE_CHOICES,
        },
        "max_grad_norm": {"type": "choice", "values": [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]},
        "loss_type": {"type": "choice", "values": ["mse", "huber"]},
        "huber_delta": {"type": "choice", "values": [0.5, 1.0, 2.0, 5.0]},
        "double_q": {"type": "choice", "values": [False]},
        "normalize_observations": {"type": "choice", "values": [False, True]},
        "obs_normalization_clip": {"type": "choice", "values": [3.0, 5.0, 7.0, 10.0]},
        "rnd_hidden_dims": {"type": "choice", "values": RND_HIDDEN_DIMS_CHOICES},
        "rnd_activation": {"type": "choice", "values": ACTIVATION_CHOICES},
        "rnd_normalization": {"type": "choice", "values": NORMALIZATION_CHOICES},
        "rnd_optimizer": {"type": "choice", "values": OPTIMIZER_CHOICES},
        "rnd_output_dim": {"type": "choice", "values": [16, 32, 64, 128, 256]},
        "rnd_action_conditioning": {
            "type": "choice",
            "values": ["none", "input", "output"],
        },
        "intrinsic_reward_scale": {
            "type": "log_uniform",
            "min": 1e-3,
            "max": 10.0,
        },
        "intrinsic_stats_decay": {
            "type": "choice",
            "values": [0.95, 0.99, 0.995, 0.999],
        },
        "intrinsic_reward_clip": {
            "type": "choice",
            "values": [1.0, 5.0, 10.0, 20.0, 50.0],
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
        #     "values": [1],  # [2**i for i in range(0, 5)],
        # },
        # "TrainingConfig.warmup_steps": {
        #     "type": "choice",
        #     "values": [2**i for i in range(0, 14)],
        # },
    },
    "params_by_algorithm": {
        "DQNAgent": {
            "ReplayBuffer.buffer_size": {
                "type": "choice",
                "values": DQN_BUFFER_SIZE_CHOICES,
            },
            "TrainingConfig.minibatch_size": {
                "type": "choice",
                "values": DQN_MINIBATCH_SIZE_CHOICES,
            },
            "TrainingConfig.update_frequency": {
                "type": "choice",
                "values": [1, 2, 4, 8, 16],
            },
            "TrainingConfig.num_minibatches": {
                "type": "choice",
                "values": [1],
            },
            "TrainingConfig.warmup_steps": {
                "type": "choice",
                "values": DQN_WARMUP_CHOICES,
            },
        },
        "DQNRNDAgent": {
            "ReplayBuffer.buffer_size": {
                "type": "choice",
                "values": DQN_BUFFER_SIZE_CHOICES,
            },
            "TrainingConfig.minibatch_size": {
                "type": "choice",
                "values": DQN_MINIBATCH_SIZE_CHOICES,
            },
            "TrainingConfig.update_frequency": {
                "type": "choice",
                "values": [1, 2, 4, 8, 16],
            },
            "TrainingConfig.num_minibatches": {
                "type": "choice",
                "values": [1],
            },
            "TrainingConfig.warmup_steps": {
                "type": "choice",
                "values": DQN_WARMUP_CHOICES,
            },
        },
    },
}


@dataclass
class Args:
    """Arguments for sampling hyperparameter combinations."""

    config: Annotated[
        tyro.conf.Positional[Path | None],
        tyro.conf.arg(
            help=(
                "Path to the gin config file passed to rl_research.main. "
                "Not required for --mode packed_resubmit."
            )
        ),
    ] = None
    samples: Annotated[
        int, tyro.conf.arg(help="Number of hyperparameter combinations to sample.")
    ] = 1000
    seeds: Annotated[
        int,
        tyro.conf.arg(help="Number of seeds per combination."),
    ] = 5
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
        Literal["sbatch", "shell", "packed_sbatch", "packed_resubmit"],
        tyro.conf.arg(
            help=(
                "Execution mode: submit Slurm arrays, generate a local launcher, "
                "submit packed Slurm jobs, or resubmit unfinished packed jobs."
            )
        ),
    ] = "sbatch"
    job_script: Annotated[
        Path,
        tyro.conf.arg(help="Path to the sbatch script used by --mode sbatch."),
    ] = Path("scripts/single_seed_job.sh")
    sbatch_opt: Annotated[
        List[str], tyro.conf.arg(help="Extra sbatch options (repeatable).")
    ] = field(default_factory=list)
    dry_run: Annotated[
        bool, tyro.conf.arg(help="Print submission commands without submitting.")
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
    batch_name: Annotated[
        str | None,
        tyro.conf.arg(
            help=(
                "Name for a packed batch under --packed-root. "
                "Defaults to --group-prefix."
            )
        ),
    ] = None
    packed_root: Annotated[
        Path,
        tyro.conf.arg(help="Root directory used to store packed batch artifacts."),
    ] = Path("outputs/packed_runs")
    packed_runner_script: Annotated[
        Path,
        tyro.conf.arg(
            help="Base runner script invoked by generated packed sbatch scripts."
        ),
    ] = Path("scripts/packed_runs_job.sh")
    estimated_run_minutes: Annotated[
        float | None,
        tyro.conf.arg(
            help=(
                "Estimated runtime per single run. Required for --mode packed_sbatch."
            )
        ),
    ] = None
    time_limit_minutes: Annotated[
        int,
        tyro.conf.arg(help="Requested wall-clock limit per packed job, in minutes."),
    ] = 180
    time_buffer_minutes: Annotated[
        int,
        tyro.conf.arg(
            help="Minutes held back from the wall-clock limit when packing runs."
        ),
    ] = 10
    packing_slack: Annotated[
        float,
        tyro.conf.arg(
            help=(
                "Multiplier applied to usable time before packing, to absorb runtime "
                "variance."
            )
        ),
    ] = 0.85
    resume_batch_dir: Annotated[
        Path | None,
        tyro.conf.arg(
            help=(
                "Existing packed batch directory to inspect when "
                "--mode packed_resubmit."
            )
        ),
    ] = None


def _format_value(val: Any) -> str:
    if val is None:
        return "None"
    if isinstance(val, str):
        return json.dumps(val)
    if isinstance(val, bool):
        return "True" if val else "False"
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        return f"{val:.6g}"
    if isinstance(val, tuple):
        inner = ", ".join(_format_value(item) for item in val)
        if len(val) == 1:
            inner += ","
        return f"({inner})"
    if isinstance(val, list):
        inner = ", ".join(_format_value(item) for item in val)
        return f"[{inner}]"
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
    space: Dict[str, Any],
    algorithm: str,
    num_samples: int,
    rng: random.Random,
) -> List[List[str]]:
    if algorithm not in space:
        raise KeyError(f"{algorithm} not in search space keys {list(space.keys())}")
    algo_space = space[algorithm]
    param_space = {
        **space.get("params", {}),
        **space.get("params_by_algorithm", {}).get(algorithm, {}),
    }
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


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_space(space_file: Path | None) -> Dict[str, Any]:
    if space_file is None:
        return DEFAULT_SPACE

    with space_file.open() as f:
        return json.load(f)


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return slug or "run"


def format_sbatch_time(total_minutes: int) -> str:
    if total_minutes < 1:
        raise ValueError("--time-limit-minutes must be >= 1")
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours}:{minutes:02d}:00"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def compute_runs_per_job(
    estimated_run_minutes: float,
    time_limit_minutes: int,
    time_buffer_minutes: int,
    packing_slack: float,
) -> int:
    if estimated_run_minutes <= 0:
        raise ValueError("--estimated-run-minutes must be > 0")
    if time_buffer_minutes < 0:
        raise ValueError("--time-buffer-minutes must be >= 0")
    if not 0 < packing_slack <= 1:
        raise ValueError("--packing-slack must be in the interval (0, 1]")

    usable_minutes = (time_limit_minutes - time_buffer_minutes) * packing_slack
    if usable_minutes <= 0:
        raise ValueError(
            "No usable time remains after applying --time-buffer-minutes and "
            "--packing-slack"
        )
    return max(1, math.floor(usable_minutes / estimated_run_minutes))


def chunk_runs(
    tasks: Sequence[dict[str, Any]],
    chunk_size: int,
) -> List[List[dict[str, Any]]]:
    return [list(tasks[i : i + chunk_size]) for i in range(0, len(tasks), chunk_size)]


def validate_packed_runner_script(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Packed runner script not found: {resolved}")
    return resolved


def build_packed_tasks(
    combos: Sequence[Sequence[str]],
    seeds: int,
    config: Path,
    group_prefix: str,
) -> List[dict[str, Any]]:
    tasks: List[dict[str, Any]] = []
    resolved_config = config.resolve()
    for combo_idx, combo in enumerate(combos):
        group_name = build_group_name(group_prefix, combo_idx)
        bindings = [
            *combo,
            f"setup_mlflow.experiment_group={_format_value(group_name)}",
        ]
        for seed in range(seeds):
            run_id = f"{slugify(group_name)}-s{seed:03d}"
            tasks.append(
                {
                    "run_id": run_id,
                    "combo_idx": combo_idx,
                    "group_name": group_name,
                    "config_path": str(resolved_config),
                    "seed": seed,
                    "bindings": list(bindings),
                }
            )
    return tasks


def build_packed_sbatch_script(
    *,
    repo_root: Path,
    runner_script: Path,
    manifest_path: Path,
    progress_file: Path,
    log_out: Path,
    log_err: Path,
    mlruns_dir: Path,
    batch_name: str,
    job_index: int,
    time_limit_minutes: int,
) -> str:
    job_name = f"{slugify(batch_name)[:40]}-j{job_index:03d}"

    db_path_str = shlex.quote(str(mlruns_dir / f"{job_name}.db"))

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        "#SBATCH --account=def-machado",
        f"#SBATCH --time={format_sbatch_time(time_limit_minutes)}",
        "#SBATCH --cpus-per-task=2",
        "#SBATCH --mem=16G",
        f"#SBATCH --output={log_out}",
        f"#SBATCH --error={log_err}",
        "",
        "set -euo pipefail",
        "",
        f'export MLFLOW_TRACKING_URI="sqlite:////{db_path_str}"',
        "",
        f"cd {shlex.quote(str(repo_root))}",
        "",
        "exec /bin/bash \\",
        f"  {shlex.quote(str(runner_script))} \\",
        f"  {shlex.quote(str(manifest_path))} \\",
        f"  {shlex.quote(str(progress_file))}",
        "",
    ]
    return "\n".join(lines)


def build_archive_sbatch_script(
    *,
    batch_dir: Path,
    log_out: Path,
    log_err: Path,
    batch_name: str,
    time_limit_minutes: int = 60,
) -> str:
    archive_job_name = f"archive-{slugify(batch_name)[:32]}"
    source_parent = batch_dir.parent.resolve()
    source_name = batch_dir.name
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={archive_job_name}",
        "#SBATCH --account=def-machado",
        f"#SBATCH --time={format_sbatch_time(time_limit_minutes)}",
        "#SBATCH --cpus-per-task=1",
        "#SBATCH --mem=4G",
        f"#SBATCH --output={log_out}",
        f"#SBATCH --error={log_err}",
        "",
        "set -euo pipefail",
        "",
        f"JOB_NAME={shlex.quote(source_name)}",
        'ARCHIVE_PATH="$HOME/${JOB_NAME}.tar.gz"',
        f"SOURCE_PARENT={shlex.quote(str(source_parent))}",
        "",
        'echo "Job ID: ${SLURM_JOB_ID:-local}"',
        'echo "Job Name: ${SLURM_JOB_NAME:-$JOB_NAME}"',
        'echo "Start Time: $(date)"',
        'echo "Archive Path: ${ARCHIVE_PATH}"',
        'echo "Source Directory: ${SOURCE_PARENT}/${JOB_NAME}"',
        "",
        'tar -czvf "$ARCHIVE_PATH" -C "$SOURCE_PARENT" "$JOB_NAME"',
        "",
    ]
    return "\n".join(lines)


def write_submit_all_script(
    path: Path,
    job_scripts: Sequence[Path],
    sbatch_opts: Sequence[str],
) -> None:
    sbatch_array = " ".join(shlex.quote(opt) for opt in sbatch_opts)
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        "",
        f"cd {shlex.quote(str(Path.cwd().resolve()))}",
        "",
        f"sbatch_opts=({sbatch_array})" if sbatch_array else "sbatch_opts=()",
        "",
    ]
    for job_script in job_scripts:
        lines.append(
            f'sbatch "${{sbatch_opts[@]}}" "$@" {shlex.quote(str(job_script.resolve()))}'
        )
    path.write_text("\n".join(lines) + "\n")
    path.chmod(0o755)


def write_submit_job_script(
    path: Path,
    job_script: Path,
    sbatch_opts: Sequence[str],
) -> None:
    sbatch_array = " ".join(shlex.quote(opt) for opt in sbatch_opts)
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        "",
        f"cd {shlex.quote(str(Path.cwd().resolve()))}",
        "",
        f"sbatch_opts=({sbatch_array})" if sbatch_array else "sbatch_opts=()",
        "",
        f'sbatch "${{sbatch_opts[@]}}" "$@" {shlex.quote(str(job_script.resolve()))}',
        "",
    ]
    path.write_text("\n".join(lines) + "\n")
    path.chmod(0o755)


def write_submit_remaining_script(
    path: Path,
    batch_dir: Path,
    sbatch_opts: Sequence[str],
) -> None:
    cmd = [
        "uv",
        "run",
        "--active",
        "--offline",
        "python",
        "scripts/submit_sweep.py",
        "--mode",
        "packed_resubmit",
        "--resume-batch-dir",
        str(batch_dir.resolve()),
    ]
    for opt in sbatch_opts:
        cmd.extend(["--sbatch-opt", opt])

    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        "",
        f"cd {shlex.quote(str(Path.cwd().resolve()))}",
        "",
        f'{shlex.join(cmd)} "$@"',
        "",
    ]
    path.write_text("\n".join(lines))
    path.chmod(0o755)


def create_packed_batch(
    *,
    args: Args,
    config: Path,
    algorithm: str,
    combos: Sequence[Sequence[str]],
) -> Path:
    batch_name = args.batch_name or args.group_prefix
    batch_dir = (args.packed_root / batch_name).resolve()
    if batch_dir.exists():
        raise FileExistsError(
            f"Packed batch directory already exists: {batch_dir}. "
            "Choose a new --batch-name or remove the old batch first."
        )

    runner_script = validate_packed_runner_script(args.packed_runner_script)
    manifests_dir = batch_dir / "manifests"
    jobs_dir = batch_dir / "jobs"
    progress_dir = batch_dir / "progress"
    logs_dir = batch_dir / "logs"
    mlruns_dir = batch_dir / "mlruns"
    archive_job_script = batch_dir / "archive_batch.sbatch"
    archive_submit_script = batch_dir / "submit_archive.sh"
    archive_log_out = logs_dir / "archive-%j.out"
    archive_log_err = logs_dir / "archive-%j.err"
    batch_dir.mkdir(parents=True, exist_ok=False)
    manifests_dir.mkdir()
    jobs_dir.mkdir()
    progress_dir.mkdir()
    logs_dir.mkdir()
    mlruns_dir.mkdir()

    if args.estimated_run_minutes is None:
        raise ValueError("--estimated-run-minutes is required for --mode packed_sbatch")

    runs_per_job = compute_runs_per_job(
        estimated_run_minutes=args.estimated_run_minutes,
        time_limit_minutes=args.time_limit_minutes,
        time_buffer_minutes=args.time_buffer_minutes,
        packing_slack=args.packing_slack,
    )

    tasks = build_packed_tasks(
        combos=combos,
        seeds=args.seeds,
        config=config,
        group_prefix=args.group_prefix,
    )
    packed_jobs: List[dict[str, Any]] = []
    created_at = now_utc()

    for job_index, job_tasks in enumerate(chunk_runs(tasks, runs_per_job)):
        manifest_path = manifests_dir / f"job_{job_index:03d}.json"
        progress_file = progress_dir / f"job_{job_index:03d}.progress.jsonl"
        log_out = logs_dir / f"job_{job_index:03d}-%j.out"
        log_err = logs_dir / f"job_{job_index:03d}-%j.err"
        job_script = jobs_dir / f"job_{job_index:03d}.sbatch"

        manifest = {
            "batch_name": batch_name,
            "group_prefix": args.group_prefix,
            "job_index": job_index,
            "created_at": created_at,
            "tasks": job_tasks,
        }
        write_json(manifest_path, manifest)

        job_script.write_text(
            build_packed_sbatch_script(
                repo_root=Path.cwd().resolve(),
                runner_script=runner_script,
                manifest_path=manifest_path,
                progress_file=progress_file,
                log_out=log_out,
                log_err=log_err,
                mlruns_dir=mlruns_dir,
                batch_name=batch_name,
                job_index=job_index,
                time_limit_minutes=args.time_limit_minutes,
            )
        )
        job_script.chmod(0o755)

        packed_jobs.append(
            {
                "job_index": job_index,
                "manifest": str(manifest_path),
                "sbatch_script": str(job_script),
                "progress_file": str(progress_file),
                "log_out": str(log_out),
                "log_err": str(log_err),
                "run_count": len(job_tasks),
            }
        )

    archive_job_script.write_text(
        build_archive_sbatch_script(
            batch_dir=batch_dir,
            log_out=archive_log_out,
            log_err=archive_log_err,
            batch_name=batch_name,
        )
    )
    archive_job_script.chmod(0o755)

    plan_path = batch_dir / "plan.json"
    plan = {
        "batch_name": batch_name,
        "group_prefix": args.group_prefix,
        "batch_dir": str(batch_dir),
        "jobs_dir": str(jobs_dir),
        "manifests_dir": str(manifests_dir),
        "progress_dir": str(progress_dir),
        "logs_dir": str(logs_dir),
        "created_at": created_at,
        "config": str(config.resolve()),
        "algorithm": algorithm,
        "space_file": str(args.space_file.resolve()) if args.space_file else None,
        "sample_count": len(combos),
        "seeds": args.seeds,
        "run_count": len(tasks),
        "estimated_run_minutes": args.estimated_run_minutes,
        "time_limit_minutes": args.time_limit_minutes,
        "time_buffer_minutes": args.time_buffer_minutes,
        "packing_slack": args.packing_slack,
        "runs_per_job": runs_per_job,
        "job_count": len(packed_jobs),
        "packed_runner_script": str(runner_script),
        "submit_all_script": str((batch_dir / "submit_all.sh").resolve()),
        "submit_remaining_script": str((batch_dir / "submit_remaining.sh").resolve()),
        "archive_job_script": str(archive_job_script.resolve()),
        "submit_archive_script": str(archive_submit_script.resolve()),
        "archive_log_out": str(archive_log_out),
        "archive_log_err": str(archive_log_err),
        "jobs": packed_jobs,
    }
    write_json(plan_path, plan)

    job_scripts = [Path(job["sbatch_script"]) for job in packed_jobs]
    write_submit_all_script(
        batch_dir / "submit_all.sh",
        job_scripts=job_scripts,
        sbatch_opts=args.sbatch_opt,
    )
    write_submit_remaining_script(
        batch_dir / "submit_remaining.sh",
        batch_dir=batch_dir,
        sbatch_opts=args.sbatch_opt,
    )
    write_submit_job_script(
        archive_submit_script,
        job_script=archive_job_script,
        sbatch_opts=args.sbatch_opt,
    )
    return plan_path


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def load_successful_runs(progress_file: Path) -> set[str]:
    successful: set[str] = set()
    if not progress_file.exists():
        return successful

    with progress_file.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") == "success" and "run_id" in record:
                successful.add(record["run_id"])
    return successful


def job_completion_summary(job_entry: dict[str, Any]) -> tuple[int, int]:
    manifest = load_json(Path(job_entry["manifest"]))
    successful = load_successful_runs(Path(job_entry["progress_file"]))
    total = len(manifest["tasks"])
    completed = sum(1 for task in manifest["tasks"] if task["run_id"] in successful)
    return completed, total


def submit_packed_job(
    job_script: Path,
    sbatch_opts: Sequence[str],
    dry_run: bool,
) -> None:
    cmd = ["sbatch", *sbatch_opts, str(job_script)]
    print(shlex.join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def submit_packed_batch(
    plan_path: Path,
    sbatch_opts: Sequence[str],
    dry_run: bool,
) -> None:
    plan = load_json(plan_path)
    print(
        f"# Submitting packed batch {plan['batch_name']} "
        f"({plan['run_count']} runs across {plan['job_count']} jobs)"
    )
    for job in plan["jobs"]:
        submit_packed_job(
            Path(job["sbatch_script"]),
            sbatch_opts=sbatch_opts,
            dry_run=dry_run,
        )


def resubmit_incomplete_packed_jobs(
    batch_dir: Path,
    sbatch_opts: Sequence[str],
    dry_run: bool,
) -> None:
    plan_path = batch_dir / "plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(f"Packed batch plan not found: {plan_path}")

    plan = load_json(plan_path)
    pending_jobs: List[dict[str, Any]] = []
    completed_runs = 0
    total_runs = 0

    for job in plan["jobs"]:
        completed, total = job_completion_summary(job)
        completed_runs += completed
        total_runs += total
        if completed < total:
            pending_jobs.append(job)

    print(
        f"# Packed batch {plan['batch_name']}: "
        f"{completed_runs}/{total_runs} runs completed, "
        f"{len(pending_jobs)}/{len(plan['jobs'])} jobs still incomplete"
    )

    for job in pending_jobs:
        submit_packed_job(
            Path(job["sbatch_script"]),
            sbatch_opts=sbatch_opts,
            dry_run=dry_run,
        )


def require_config(args: Args) -> Path:
    if args.config is None:
        raise ValueError(f"--mode {args.mode} requires CONFIG.gin")
    return args.config


def main(args: Args) -> None:
    if args.mode == "packed_resubmit":
        if args.resume_batch_dir is None:
            raise ValueError(
                "--resume-batch-dir is required for --mode packed_resubmit"
            )
        resubmit_incomplete_packed_jobs(
            batch_dir=args.resume_batch_dir.resolve(),
            sbatch_opts=args.sbatch_opt,
            dry_run=args.dry_run,
        )
        return

    config = require_config(args).resolve()
    rng = random.Random(args.rng_seed)
    space = load_space(args.space_file)
    algorithm = infer_algorithm_from_config(config)
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
                config=config,
                group_prefix=args.group_prefix,
                job_script=args.job_script,
                sbatch_opts=args.sbatch_opt,
                dry_run=args.dry_run,
            )
        return

    if args.mode == "shell":
        shell_script = args.shell_script or Path("scripts/generated_sweep.sh")
        local_commands: List[List[str]] = []
        for idx, combo in enumerate(combos):
            for seed in range(args.seeds):
                local_commands.append(
                    build_local_cmd(
                        combo_idx=idx,
                        seed=seed,
                        bindings=combo,
                        config=config,
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
        return

    if args.mode == "packed_sbatch":
        plan_path = create_packed_batch(
            args=args,
            config=config,
            algorithm=algorithm,
            combos=combos,
        )
        plan = load_json(plan_path)
        print(
            f"# Created packed batch {plan['batch_name']} at {plan_path.parent} "
            f"({plan['run_count']} runs, {plan['job_count']} jobs, "
            f"{plan['runs_per_job']} runs/job)"
        )
        print(f"# Archive helper: {plan['submit_archive_script']}")
        submit_packed_batch(
            plan_path=plan_path,
            sbatch_opts=args.sbatch_opt,
            dry_run=args.dry_run,
        )
        return

    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main(tyro.cli(Args))
