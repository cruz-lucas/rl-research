#!/usr/bin/env python3
"""Pack many short RL runs into a small number of longer-lived jobs.

This script is intentionally separate from the existing submission scripts.
It builds a manifest of atomic runs and can then:

1. plan a workload from a list of configs and/or config folders,
2. expand each config into either:
   - a plain "run this config as-is" workload, or
   - a grid/random sweep described in JSON,
3. pack the resulting runs into one or more Slurm jobs, and
4. execute a manifest with resume support via a progress file.

Examples
--------
Run every config in a folder as-is:

    uv run --active --offline python scripts/submit_packed_runs.py plan \
      --config-dir rl_research/configs/riverswim \
      --seeds 20 \
      --estimated-run-minutes 2 \
      --time-limit-minutes 180 \
      --batch-name riverswim-all

Submit the same workload to Slurm:

    uv run --active --offline python scripts/submit_packed_runs.py submit \
      --config-dir rl_research/configs/riverswim \
      --seeds 20 \
      --estimated-run-minutes 2 \
      --time-limit-minutes 180 \
      --account aip-machado \
      --job-name tabular-pack \
      --batch-name riverswim-all

Sweep a list of configs with a space file:

    uv run --active --offline python scripts/submit_packed_runs.py submit \
      --config rl_research/configs/riverswim/qlearning_intrinsic_reward.gin \
      --config rl_research/configs/sixarms/qlearning_intrinsic_reward.gin \
      --space-file sweep_spaces.json \
      --seeds 20 \
      --estimated-run-minutes 2 \
      --time-limit-minutes 180 \
      --batch-name qlearning-sweep

Space file formats
------------------
Simple one-space-for-all-configs file:

{
  "search": "grid",
  "params": {
    "QLearningAgent.step_size": {"values": [1e-4, 1e-3, 1e-2]},
    "QLearningAgent.reward_bonus": {"values": [0, 100, 1000]}
  }
}

Algorithm-aware file:

{
  "defaults": {
    "params": {
      "TrainingConfig.train_steps": {"values": [5000]}
    }
  },
  "algorithms": {
    "QLearningAgent": {
      "search": "random",
      "samples": 32,
      "params": {
        "QLearningAgent.step_size": {
          "distribution": "log_uniform",
          "min": 1e-4,
          "max": 5e-1
        },
        "QLearningAgent.reward_bonus": {
          "distribution": "int_uniform",
          "min": 0,
          "max": 10000
        }
      }
    },
    "RMaxAgent": {
      "search": "grid",
      "params": {
        "RMaxAgent.known_threshold": {"values": [1, 2, 5, 10, 20]}
      }
    }
  },
  "configs": {
    "rl_research/configs/riverswim/qlearning_intrinsic_reward.gin": {
      "params": {
        "TrainingConfig.max_episode_steps": {"values": [5000]}
      }
    }
  }
}

The existing `submit_sweep.py` style is also accepted:

{
  "QLearningAgent": {
    "step_size": {"type": "log_uniform", "min": 1e-4, "max": 5e-1},
    "reward_bonus": {"type": "int", "min": 0, "max": 10000}
  },
  "params": {
    "TrainingConfig.num_minibatches": {"type": "choice", "values": [1, 5, 10]}
  }
}
"""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import math
import random
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import gin

import rl_research.main  # noqa: F401


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "packed_runs"
DEFAULT_MODULES = ("python/3.11", "gcc", "arrow")
DEFAULT_ACTIVATE = "$HOME/.venv/bin/activate"


@dataclass(frozen=True)
class Variant:
    config_path: str
    config_key: str
    config_slug: str
    algorithm: str
    variant_id: str
    group_name: str
    bindings: list[str]
    combo: dict[str, Any]
    search_mode: str


@dataclass(frozen=True)
class RunTask:
    run_id: str
    config_path: str
    config_key: str
    config_slug: str
    algorithm: str
    variant_id: str
    group_name: str
    seed: int
    bindings: list[str]
    combo: dict[str, Any]


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slugify(value: str) -> str:
    chars: list[str] = []
    prev_dash = False
    for ch in value:
        if ch.isalnum():
            chars.append(ch.lower())
            prev_dash = False
            continue
        if not prev_dash:
            chars.append("-")
            prev_dash = True
    slug = "".join(chars).strip("-")
    return slug or "item"


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def format_gin_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Cannot format non-finite float value: {value}")
        return f"{value:.12g}"
    if isinstance(value, str):
        return json.dumps(value)
    raise TypeError(f"Unsupported gin value type: {type(value)}")


def discover_configs(
    config_paths: Iterable[Path],
    config_dirs: Iterable[Path],
    glob_pattern: str,
    recursive: bool,
) -> list[Path]:
    found: dict[str, Path] = {}

    for config_path in config_paths:
        resolved = config_path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if resolved.suffix != ".gin":
            raise ValueError(f"Expected a .gin config file, got: {config_path}")
        found[str(resolved)] = resolved

    for config_dir in config_dirs:
        resolved_dir = config_dir.resolve()
        if not resolved_dir.exists():
            raise FileNotFoundError(f"Config directory not found: {config_dir}")
        if not resolved_dir.is_dir():
            raise NotADirectoryError(f"Expected a directory: {config_dir}")
        iterator = resolved_dir.rglob(glob_pattern) if recursive else resolved_dir.glob(
            glob_pattern
        )
        for config_path in iterator:
            if config_path.is_file() and config_path.suffix == ".gin":
                found[str(config_path.resolve())] = config_path.resolve()

    configs = sorted(found.values())
    if not configs:
        raise ValueError(
            "No config files were found. Provide --config and/or --config-dir."
        )
    return configs


def infer_algorithm_from_config(config_path: Path) -> str:
    gin.clear_config()
    try:
        gin.parse_config_files_and_bindings(
            [str(config_path.resolve())],
            bindings=None,
            skip_unknown=True,
        )
        run_bindings = gin.get_bindings("run_loop")
        agent_cls = run_bindings.get("agent_cls")
    finally:
        gin.clear_config()

    if agent_cls is None:
        raise ValueError(f"run_loop.agent_cls not set in config {config_path}")
    return getattr(agent_cls, "__name__", str(agent_cls))


def load_space_bundle(space_file: Path | None) -> dict[str, Any]:
    if space_file is None:
        return {
            "defaults": {},
            "algorithms": {},
            "configs": {},
        }

    with space_file.open() as handle:
        raw = json.load(handle)

    if not isinstance(raw, dict):
        raise ValueError("Space file must contain a JSON object at the top level.")

    if any(key in raw for key in ("defaults", "algorithms", "configs")):
        return {
            "defaults": _normalize_scope(raw.get("defaults", {})),
            "algorithms": {
                key: _normalize_scope(value)
                for key, value in raw.get("algorithms", {}).items()
            },
            "configs": {
                key: _normalize_scope(value)
                for key, value in raw.get("configs", {}).items()
            },
        }

    if any(key in raw for key in ("params", "search", "samples")):
        return {
            "defaults": _normalize_scope(raw),
            "algorithms": {},
            "configs": {},
        }

    defaults = {"params": raw.get("params", {})} if "params" in raw else {}
    algorithms = {}
    for key, value in raw.items():
        if key == "params":
            continue
        algorithms[key] = _normalize_scope({"params": value})
    return {
        "defaults": _normalize_scope(defaults),
        "algorithms": algorithms,
        "configs": {},
    }


def _normalize_scope(scope: Any) -> dict[str, Any]:
    if not scope:
        return {}
    if not isinstance(scope, dict):
        raise ValueError(
            f"Search-space scope must be a JSON object, got: {type(scope)}"
        )

    if any(key in scope for key in ("params", "search", "samples")):
        normalized = dict(scope)
        params = normalized.get("params", {})
    else:
        normalized = {"params": dict(scope)}
        params = normalized["params"]

    if not isinstance(params, dict):
        raise ValueError("'params' in a search-space scope must be an object.")

    normalized["params"] = {str(name): dict(spec) for name, spec in params.items()}
    return normalized


def resolve_space_scope(
    config_path: Path,
    algorithm: str,
    bundle: dict[str, Any],
) -> dict[str, Any]:
    merged: dict[str, Any] = {"params": {}}

    scopes = (
        bundle.get("defaults", {}),
        bundle.get("algorithms", {}).get(algorithm, {}),
    )
    for scope in scopes:
        if not scope:
            continue
        if "search" in scope:
            merged["search"] = scope["search"]
        if "samples" in scope:
            merged["samples"] = scope["samples"]
        merged["params"].update(scope.get("params", {}))

    config_scopes = bundle.get("configs", {})
    config_override = None
    for key in _config_match_candidates(config_path):
        if key in config_scopes:
            config_override = config_scopes[key]
            break
    if config_override:
        if "search" in config_override:
            merged["search"] = config_override["search"]
        if "samples" in config_override:
            merged["samples"] = config_override["samples"]
        merged["params"].update(config_override.get("params", {}))

    return merged


def _config_match_candidates(config_path: Path) -> list[str]:
    resolved = config_path.resolve()
    candidates = [
        repo_relative(resolved),
        resolved.as_posix(),
        resolved.name,
        resolved.stem,
    ]
    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        if candidate not in seen:
            unique.append(candidate)
            seen.add(candidate)
    return unique


def infer_search_mode(params: dict[str, Any], explicit_mode: str | None) -> str:
    if explicit_mode:
        return explicit_mode
    if not params:
        return "none"
    if all("values" in spec for spec in params.values()):
        return "grid"
    return "random"


def expand_grid(
    params: dict[str, Any],
    max_grid_combinations: int,
) -> list[dict[str, Any]]:
    if not params:
        return [{}]

    choices: list[tuple[str, list[Any]]] = []
    total = 1
    for name, spec in sorted(params.items()):
        values = spec.get("values")
        if values is None:
            raise ValueError(
                f"Grid search requires discrete 'values' for parameter {name}."
            )
        if not isinstance(values, list) or not values:
            raise ValueError(
                f"Parameter {name} must provide a non-empty 'values' list."
            )
        total *= len(values)
        if total > max_grid_combinations:
            raise ValueError(
                "Grid search expands to "
                f"{total} combinations, which exceeds --max-grid-combinations="
                f"{max_grid_combinations}."
            )
        choices.append((name, values))

    combos = []
    for values_tuple in itertools.product(*(values for _, values in choices)):
        combos.append(
            {
                name: value
                for (name, _), value in zip(choices, values_tuple, strict=True)
            }
        )
    return combos


def expand_random(
    params: dict[str, Any],
    samples: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if samples < 1:
        raise ValueError("Random search requires --samples >= 1 in the space file.")
    if not params:
        return [{}]

    combos: list[dict[str, Any]] = []
    seen: set[str] = set()
    max_attempts = max(samples * 20, 100)
    attempts = 0

    while len(combos) < samples and attempts < max_attempts:
        attempts += 1
        combo = {
            name: sample_param_value(spec, rng)
            for name, spec in sorted(params.items())
        }
        key = json.dumps(combo, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        combos.append(combo)

    if len(combos) < samples:
        raise ValueError(
            f"Could only draw {len(combos)} unique random samples out of {samples}. "
            "The search space may be too small or too discrete."
        )
    return combos


def sample_param_value(spec: dict[str, Any], rng: random.Random) -> Any:
    if "values" in spec:
        values = spec["values"]
        if not isinstance(values, list) or not values:
            raise ValueError("A parameter with 'values' must use a non-empty list.")
        return rng.choice(values)

    dist = spec.get("distribution", spec.get("type"))
    if dist == "uniform":
        return rng.uniform(spec["min"], spec["max"])
    if dist in ("log_uniform", "loguniform"):
        low = math.log10(spec["min"])
        high = math.log10(spec["max"])
        return 10 ** rng.uniform(low, high)
    if dist in ("int_uniform", "int"):
        return rng.randint(int(spec["min"]), int(spec["max"]))
    if dist == "choice":
        values = spec.get("values")
        if not isinstance(values, list) or not values:
            raise ValueError(
                "'choice' parameters must provide a non-empty 'values' list."
            )
        return rng.choice(values)

    raise ValueError(
        "Unknown parameter distribution. Use 'values', 'uniform', "
        "'log_uniform', 'int_uniform', or legacy 'type'."
    )


def build_variants(
    config_path: Path,
    algorithm: str,
    bundle: dict[str, Any],
    base_bindings: list[str],
    group_prefix: str,
    search_rng: random.Random,
    max_grid_combinations: int,
) -> list[Variant]:
    config_key = repo_relative(config_path)
    config_slug = slugify(config_key.removesuffix(".gin"))
    scope = resolve_space_scope(config_path, algorithm, bundle)
    params = scope.get("params", {})
    search_mode = infer_search_mode(params, scope.get("search"))

    if search_mode == "none":
        combos = [{}]
    elif search_mode == "grid":
        combos = expand_grid(params, max_grid_combinations)
    elif search_mode == "random":
        combos = expand_random(params, int(scope.get("samples", 0)), search_rng)
    else:
        raise ValueError(f"Unsupported search mode for {config_key}: {search_mode}")

    variants: list[Variant] = []
    single_empty_combo = len(combos) == 1 and not combos[0]
    for idx, combo in enumerate(combos):
        variant_id = "base" if single_empty_combo else f"v{idx:04d}"
        group_name = (
            f"{group_prefix}-{config_slug}"
            if variant_id == "base"
            else f"{group_prefix}-{config_slug}-{variant_id}"
        )
        bindings = list(base_bindings)
        for name, value in sorted(combo.items()):
            bindings.append(f"{name}={format_gin_value(value)}")
        bindings.append(f"setup_mlflow.experiment_group={format_gin_value(group_name)}")
        variants.append(
            Variant(
                config_path=str(config_path.resolve()),
                config_key=config_key,
                config_slug=config_slug,
                algorithm=algorithm,
                variant_id=variant_id,
                group_name=group_name,
                bindings=bindings,
                combo=combo,
                search_mode=search_mode,
            )
        )
    return variants


def build_tasks(variants: list[Variant], seeds: int) -> list[RunTask]:
    if seeds < 1:
        raise ValueError("--seeds must be >= 1")

    seed_width = max(3, len(str(seeds - 1)))
    tasks: list[RunTask] = []
    for variant in variants:
        for seed in range(seeds):
            run_id = (
                f"{variant.config_slug}-{variant.variant_id}-"
                f"s{seed:0{seed_width}d}"
            )
            tasks.append(
                RunTask(
                    run_id=run_id,
                    config_path=variant.config_path,
                    config_key=variant.config_key,
                    config_slug=variant.config_slug,
                    algorithm=variant.algorithm,
                    variant_id=variant.variant_id,
                    group_name=variant.group_name,
                    seed=seed,
                    bindings=list(variant.bindings),
                    combo=dict(variant.combo),
                )
            )
    return tasks


def infer_runs_per_job(
    runs_per_job: int,
    estimated_run_minutes: float,
    time_limit_minutes: int,
    time_buffer_minutes: int,
    parallelism: int,
    packing_slack: float,
) -> int:
    if runs_per_job > 0:
        return runs_per_job

    if estimated_run_minutes <= 0:
        raise ValueError("--estimated-run-minutes must be > 0 when auto-packing.")
    if time_limit_minutes <= time_buffer_minutes:
        raise ValueError(
            "--time-limit-minutes must be greater than --time-buffer-minutes."
        )
    if parallelism < 1:
        raise ValueError("--parallelism must be >= 1")
    if not (0 < packing_slack <= 1.0):
        raise ValueError("--packing-slack must be in the range (0, 1].")

    usable_minutes = time_limit_minutes - time_buffer_minutes
    capacity = int(
        (usable_minutes * parallelism * packing_slack) // estimated_run_minutes
    )
    return max(1, capacity)


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


def build_batch_plan(args: argparse.Namespace) -> dict[str, Any]:
    configs = discover_configs(
        config_paths=args.config,
        config_dirs=args.config_dir,
        glob_pattern=args.glob,
        recursive=not args.no_recursive,
    )

    if args.shuffle_configs:
        rng = random.Random(args.search_seed)
        rng.shuffle(configs)

    if args.config_limit is not None:
        configs = configs[: args.config_limit]

    batch_name = args.batch_name or datetime.now().strftime("packed-%Y%m%d-%H%M%S")
    group_prefix = args.group_prefix or batch_name
    search_rng = random.Random(args.search_seed)
    bundle = load_space_bundle(args.space_file)

    variants: list[Variant] = []
    configs_summary: list[dict[str, Any]] = []
    algorithms_count: dict[str, int] = {}

    for config_path in configs:
        algorithm = infer_algorithm_from_config(config_path)
        algorithms_count[algorithm] = algorithms_count.get(algorithm, 0) + 1
        config_variants = build_variants(
            config_path=config_path,
            algorithm=algorithm,
            bundle=bundle,
            base_bindings=list(args.base_binding),
            group_prefix=group_prefix,
            search_rng=search_rng,
            max_grid_combinations=args.max_grid_combinations,
        )
        variants.extend(config_variants)
        configs_summary.append(
            {
                "config": repo_relative(config_path),
                "algorithm": algorithm,
                "variants": len(config_variants),
                "search_mode": (
                    config_variants[0].search_mode if config_variants else "none"
                ),
            }
        )

    tasks = build_tasks(variants, args.seeds)
    runs_per_job = infer_runs_per_job(
        runs_per_job=args.runs_per_job,
        estimated_run_minutes=args.estimated_run_minutes,
        time_limit_minutes=args.time_limit_minutes,
        time_buffer_minutes=args.time_buffer_minutes,
        parallelism=args.parallelism,
        packing_slack=args.packing_slack,
    )
    job_chunks = chunked([asdict(task) for task in tasks], runs_per_job)

    batch_dir = args.out_dir.resolve() / batch_name
    manifests_dir = batch_dir / "manifests"
    progress_dir = batch_dir / "progress"
    logs_dir = batch_dir / "logs"

    return {
        "batch_name": batch_name,
        "group_prefix": group_prefix,
        "batch_dir": str(batch_dir),
        "manifests_dir": str(manifests_dir),
        "progress_dir": str(progress_dir),
        "logs_dir": str(logs_dir),
        "created_at": now_utc(),
        "space_file": str(args.space_file.resolve()) if args.space_file else None,
        "config_count": len(configs),
        "variant_count": len(variants),
        "run_count": len(tasks),
        "seeds": args.seeds,
        "parallelism": args.parallelism,
        "estimated_run_minutes": args.estimated_run_minutes,
        "time_limit_minutes": args.time_limit_minutes,
        "time_buffer_minutes": args.time_buffer_minutes,
        "packing_slack": args.packing_slack,
        "runs_per_job": runs_per_job,
        "job_count": len(job_chunks),
        "algorithms": algorithms_count,
        "configs": configs_summary,
        "jobs": [
            {
                "job_index": idx,
                "manifest": str((manifests_dir / f"job_{idx:03d}.json").resolve()),
                "progress_file": str(
                    (progress_dir / f"job_{idx:03d}.progress.jsonl").resolve()
                ),
                "log_out": str((logs_dir / f"job_{idx:03d}-%j.out").resolve()),
                "log_err": str((logs_dir / f"job_{idx:03d}-%j.err").resolve()),
                "run_count": len(chunk),
                "tasks": chunk,
            }
            for idx, chunk in enumerate(job_chunks)
        ],
    }


def write_batch_plan(plan: dict[str, Any]) -> None:
    batch_dir = Path(plan["batch_dir"])
    if batch_dir.exists():
        raise FileExistsError(
            f"Batch directory already exists: {batch_dir}. "
            "Choose a different --batch-name or move the existing batch."
        )

    manifests_dir = Path(plan["manifests_dir"])
    progress_dir = Path(plan["progress_dir"])
    logs_dir = Path(plan["logs_dir"])

    manifests_dir.mkdir(parents=True, exist_ok=False)
    progress_dir.mkdir(parents=True, exist_ok=False)
    logs_dir.mkdir(parents=True, exist_ok=False)

    for job in plan["jobs"]:
        manifest_path = Path(job["manifest"])
        payload = {
            "batch_name": plan["batch_name"],
            "group_prefix": plan["group_prefix"],
            "job_index": job["job_index"],
            "created_at": plan["created_at"],
            "tasks": job["tasks"],
        }
        manifest_path.write_text(json.dumps(payload, indent=2) + "\n")

    plan_path = batch_dir / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")


def minutes_to_slurm_time(total_minutes: int) -> str:
    hours, minutes = divmod(total_minutes, 60)
    days, hours = divmod(hours, 24)
    if days:
        return f"{days}-{hours:02d}:{minutes:02d}:00"
    return f"{hours:02d}:{minutes:02d}:00"


def quote_command(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def build_wrap_command(
    manifest_path: Path,
    progress_file: Path,
    args: argparse.Namespace,
) -> str:
    runner = [
        "uv",
        "run",
        "--active",
        "--offline",
        "python",
        str((REPO_ROOT / "scripts" / "submit_packed_runs.py").resolve()),
        "run-manifest",
        "--manifest",
        str(manifest_path.resolve()),
        "--progress-file",
        str(progress_file.resolve()),
        "--parallelism",
        str(args.parallelism),
        "--time-limit-minutes",
        str(args.time_limit_minutes),
        "--time-buffer-minutes",
        str(args.time_buffer_minutes),
    ]
    if args.continue_on_error:
        runner.append("--continue-on-error")

    shell_lines = [
        "set -euo pipefail",
        "export PYTHONUNBUFFERED=1",
        f"mkdir -p {shlex.quote(str(Path(args.logs_dir).resolve()))}",
    ]
    modules = args.module or list(DEFAULT_MODULES)
    if modules:
        quoted_modules = " ".join(shlex.quote(module) for module in modules)
        shell_lines.append(f"module load {quoted_modules}")
    if args.activate:
        shell_lines.append(f"source {args.activate}")
    shell_lines.extend(
        [
            f"cd {shlex.quote(str(REPO_ROOT))}",
            'echo "Job ID: ${SLURM_JOB_ID:-local}"',
            'echo "Job Name: ${SLURM_JOB_NAME:-interactive}"',
            'echo "Node: ${SLURMD_NODENAME:-localhost}"',
            'echo "Start Time: $(date)"',
            'echo "CPUs: ${SLURM_CPUS_PER_TASK:-1}"',
            quote_command(runner),
        ]
    )
    script = "; ".join(shell_lines)
    return f"bash -lc {shlex.quote(script)}"


def build_sbatch_command(job: dict[str, Any], args: argparse.Namespace) -> list[str]:
    manifest_path = Path(job["manifest"])
    progress_file = Path(job["progress_file"])
    wrap_command = build_wrap_command(manifest_path, progress_file, args)

    command = [
        "sbatch",
        "--parsable",
        "--job-name",
        f"{args.job_name}-{job['job_index']:03d}",
        "--time",
        minutes_to_slurm_time(args.time_limit_minutes),
        "--cpus-per-task",
        str(args.cpus_per_task),
        "--mem",
        f"{args.mem_gb}G",
        "--output",
        job["log_out"],
        "--error",
        job["log_err"],
        "--signal",
        f"TERM@{args.signal_buffer_seconds}",
    ]
    if args.account:
        command.extend(["--account", args.account])
    if args.partition:
        command.extend(["--partition", args.partition])
    if args.qos:
        command.extend(["--qos", args.qos])
    if args.constraint:
        command.extend(["--constraint", args.constraint])
    if args.requeue:
        command.append("--requeue")
    for option in args.extra_sbatch_option:
        command.append(option)
    command.extend(["--wrap", wrap_command])
    return command


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def load_completed_ids(progress_file: Path) -> set[str]:
    completed: set[str] = set()
    if not progress_file.exists():
        return completed
    with progress_file.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("status") == "done":
                completed.add(record["run_id"])
    return completed


def run_one_task(task: dict[str, Any]) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "rl_research.main",
        "--config",
        task["config_path"],
        "--seed",
        str(task["seed"]),
    ]
    for binding in task["bindings"]:
        command.extend(["--binding", binding])

    started = time.time()
    process = subprocess.run(command, cwd=REPO_ROOT, check=False)
    return {
        "run_id": task["run_id"],
        "returncode": process.returncode,
        "duration_seconds": round(time.time() - started, 3),
        "command": command,
    }


def execute_manifest(args: argparse.Namespace) -> int:
    manifest_path = args.manifest.resolve()
    with manifest_path.open() as handle:
        manifest = json.load(handle)

    progress_file = (
        args.progress_file.resolve()
        if args.progress_file
        else manifest_path.with_suffix(".progress.jsonl")
    )

    tasks = manifest["tasks"]
    completed = load_completed_ids(progress_file)
    pending = [task for task in tasks if task["run_id"] not in completed]

    print(
        f"Manifest {manifest_path.name}: {len(completed)} already complete, "
        f"{len(pending)} pending, parallelism={args.parallelism}",
        flush=True,
    )
    if not pending:
        return 0

    stop_event = threading.Event()

    def _handle_signal(signum: int, _frame: Any) -> None:
        print(
            f"Received signal {signum}; no new runs will be started. "
            "In-flight runs may still finish before Slurm stops the job.",
            flush=True,
        )
        stop_event.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    deadline = None
    if args.time_limit_minutes > 0:
        deadline = (
            time.monotonic()
            + (args.time_limit_minutes * 60)
            - (args.time_buffer_minutes * 60)
        )

    failures: list[str] = []
    pending_iter = iter(pending)
    futures: dict[concurrent.futures.Future[dict[str, Any]], dict[str, Any]] = {}

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.parallelism
    ) as executor:
        while True:
            while len(futures) < args.parallelism and not stop_event.is_set():
                if deadline is not None and time.monotonic() >= deadline:
                    print(
                        "Reached the no-new-runs deadline; waiting for in-flight runs.",
                        flush=True,
                    )
                    stop_event.set()
                    break

                try:
                    task = next(pending_iter)
                except StopIteration:
                    break

                print(
                    f"[start] {task['run_id']} "
                    f"config={task['config_key']} seed={task['seed']}",
                    flush=True,
                )
                futures[executor.submit(run_one_task, task)] = task

            if not futures:
                break

            done, _ = concurrent.futures.wait(
                futures,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                task = futures.pop(future)
                finished_at = now_utc()
                try:
                    result = future.result()
                    returncode = int(result["returncode"])
                    status = "done" if returncode == 0 else "failed"
                    append_jsonl(
                        progress_file,
                        {
                            "run_id": task["run_id"],
                            "status": status,
                            "finished_at": finished_at,
                            "duration_seconds": result["duration_seconds"],
                            "returncode": returncode,
                        },
                    )
                    print(
                        f"[{status}] {task['run_id']} "
                        f"duration={result['duration_seconds']}s "
                        f"returncode={returncode}",
                        flush=True,
                    )
                    if returncode != 0:
                        failures.append(task["run_id"])
                        if not args.continue_on_error:
                            stop_event.set()
                except BaseException as exc:  # noqa: BLE001
                    append_jsonl(
                        progress_file,
                        {
                            "run_id": task["run_id"],
                            "status": "failed",
                            "finished_at": finished_at,
                            "exception": repr(exc),
                        },
                    )
                    print(f"[failed] {task['run_id']} exception={exc!r}", flush=True)
                    failures.append(task["run_id"])
                    if not args.continue_on_error:
                        stop_event.set()

        if futures:
            print(f"Waiting for {len(futures)} in-flight runs to finish.", flush=True)
            for future in concurrent.futures.as_completed(list(futures)):
                task = futures[future]
                finished_at = now_utc()
                try:
                    result = future.result()
                    returncode = int(result["returncode"])
                    status = "done" if returncode == 0 else "failed"
                    append_jsonl(
                        progress_file,
                        {
                            "run_id": task["run_id"],
                            "status": status,
                            "finished_at": finished_at,
                            "duration_seconds": result["duration_seconds"],
                            "returncode": returncode,
                        },
                    )
                    print(
                        f"[{status}] {task['run_id']} "
                        f"duration={result['duration_seconds']}s "
                        f"returncode={returncode}",
                        flush=True,
                    )
                    if returncode != 0:
                        failures.append(task["run_id"])
                except BaseException as exc:  # noqa: BLE001
                    append_jsonl(
                        progress_file,
                        {
                            "run_id": task["run_id"],
                            "status": "failed",
                            "finished_at": finished_at,
                            "exception": repr(exc),
                        },
                    )
                    print(f"[failed] {task['run_id']} exception={exc!r}", flush=True)
                    failures.append(task["run_id"])

    completed_after = load_completed_ids(progress_file)
    remaining = len(tasks) - len(completed_after)
    if failures:
        print(f"{len(failures)} run(s) failed.", flush=True)
        return 1
    if remaining > 0:
        print(
            f"Manifest finished with {remaining} run(s) still incomplete. "
            "Re-run the same manifest to resume.",
            flush=True,
        )
        return 2
    return 0


def print_plan_summary(plan: dict[str, Any]) -> None:
    print(
        f"Batch {plan['batch_name']}: "
        f"{plan['config_count']} configs, "
        f"{plan['variant_count']} variants, "
        f"{plan['run_count']} runs, "
        f"{plan['job_count']} jobs, "
        f"{plan['runs_per_job']} runs/job",
        flush=True,
    )
    print(f"Batch dir: {plan['batch_dir']}", flush=True)
    for algorithm, count in sorted(plan["algorithms"].items()):
        print(f"  {algorithm}: {count} config(s)", flush=True)


def plan_command(args: argparse.Namespace) -> int:
    plan = build_batch_plan(args)
    write_batch_plan(plan)
    print_plan_summary(plan)
    print(f"Plan written to {Path(plan['batch_dir']) / 'plan.json'}", flush=True)
    return 0


def submit_command(args: argparse.Namespace) -> int:
    if args.parallelism > args.cpus_per_task:
        raise ValueError(
            "--parallelism cannot exceed --cpus-per-task for submitted Slurm jobs."
        )

    plan = build_batch_plan(args)
    write_batch_plan(plan)
    args.logs_dir = plan["logs_dir"]

    print_plan_summary(plan)

    submitted_jobs: list[dict[str, Any]] = []
    for job in plan["jobs"]:
        command = build_sbatch_command(job, args)
        print(quote_command(command), flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True,
        )
        job_id = result.stdout.strip()
        submitted_jobs.append(
            {
                "job_index": job["job_index"],
                "job_id": job_id,
                "manifest": job["manifest"],
            }
        )
        print(f"submitted job_index={job['job_index']} job_id={job_id}", flush=True)

    submissions_path = Path(plan["batch_dir"]) / "submitted_jobs.json"
    submissions_path.write_text(json.dumps(submitted_jobs, indent=2) + "\n")
    print(f"Submission record: {submissions_path}", flush=True)
    return 0


def add_workload_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        action="append",
        default=[],
        help="Path to a gin config file. Repeatable.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        action="append",
        default=[],
        help="Directory to scan for gin config files. Repeatable.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.gin",
        help="Glob used when scanning config directories.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Disable recursive directory scanning for --config-dir.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=20,
        help="Number of seeds to run for every config/sweep variant.",
    )
    parser.add_argument(
        "--base-binding",
        action="append",
        default=[],
        help="Gin binding applied to every run. Repeatable.",
    )
    parser.add_argument(
        "--space-file",
        type=Path,
        help="Optional JSON file describing grid/random sweeps.",
    )
    parser.add_argument(
        "--search-seed",
        type=int,
        default=0,
        help="Random seed used for config shuffling and random search.",
    )
    parser.add_argument(
        "--batch-name",
        type=str,
        help="Name for the batch output directory. Defaults to a timestamped name.",
    )
    parser.add_argument(
        "--group-prefix",
        type=str,
        help="Prefix for setup_mlflow.experiment_group. Defaults to --batch-name.",
    )
    parser.add_argument(
        "--config-limit",
        type=int,
        help="Only use the first N discovered configs after sorting/shuffling.",
    )
    parser.add_argument(
        "--shuffle-configs",
        action="store_true",
        help="Shuffle configs before applying --config-limit.",
    )
    parser.add_argument(
        "--max-grid-combinations",
        type=int,
        default=5000,
        help="Fail if grid expansion exceeds this many variants for one config.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Base directory used to store manifests, logs, and progress files.",
    )


def add_packing_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--parallelism",
        type=int,
        default=1,
        help="How many runs each job should execute in parallel.",
    )
    parser.add_argument(
        "--estimated-run-minutes",
        type=float,
        default=2.0,
        help="Typical runtime per seed, used for packing when --runs-per-job=0.",
    )
    parser.add_argument(
        "--runs-per-job",
        type=int,
        default=0,
        help="Explicitly set runs per job. Zero enables automatic packing.",
    )
    parser.add_argument(
        "--time-limit-minutes",
        type=int,
        default=180,
        help="Job walltime used both for packing and for the Slurm request.",
    )
    parser.add_argument(
        "--time-buffer-minutes",
        type=int,
        default=10,
        help="Reserve this buffer so jobs stop launching new runs before walltime.",
    )
    parser.add_argument(
        "--packing-slack",
        type=float,
        default=0.85,
        help="Fraction of theoretical capacity to actually pack into each job.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running other tasks even if one task fails.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser(
        "plan",
        help="Build manifests and write a batch plan without submitting jobs.",
    )
    add_workload_args(plan_parser)
    add_packing_args(plan_parser)

    submit_parser = subparsers.add_parser(
        "submit",
        help="Build manifests and submit one Slurm job per manifest.",
    )
    add_workload_args(submit_parser)
    add_packing_args(submit_parser)
    submit_parser.add_argument(
        "--job-name",
        type=str,
        default="rl-packed",
        help="Base Slurm job name.",
    )
    submit_parser.add_argument(
        "--account",
        type=str,
        help="Slurm account to charge.",
    )
    submit_parser.add_argument(
        "--partition",
        type=str,
        help="Slurm partition.",
    )
    submit_parser.add_argument(
        "--qos",
        type=str,
        help="Slurm QoS. Use this if your cluster exposes a preemptible QoS.",
    )
    submit_parser.add_argument(
        "--constraint",
        type=str,
        help="Optional Slurm node constraint.",
    )
    submit_parser.add_argument(
        "--cpus-per-task",
        type=int,
        default=1,
        help="Requested CPUs per Slurm job.",
    )
    submit_parser.add_argument(
        "--mem-gb",
        type=int,
        default=8,
        help="Requested memory per Slurm job in GB.",
    )
    submit_parser.add_argument(
        "--module",
        action="append",
        default=[],
        help=(
            "Module to load in the job wrapper. Repeatable. "
            "Defaults to python/3.11 gcc arrow."
        ),
    )
    submit_parser.add_argument(
        "--activate",
        type=str,
        default=DEFAULT_ACTIVATE,
        help="Shell snippet used to activate the Python environment inside the job.",
    )
    submit_parser.add_argument(
        "--signal-buffer-seconds",
        type=int,
        default=90,
        help="Seconds before Slurm timeout/preemption to send SIGTERM.",
    )
    submit_parser.add_argument(
        "--extra-sbatch-option",
        action="append",
        default=[],
        help=(
            "Raw extra sbatch option, for flags not covered by the script. "
            "Repeatable."
        ),
    )
    submit_parser.add_argument(
        "--requeue",
        action="store_true",
        help="Add --requeue to sbatch. Useful when running under a preemptible QoS.",
    )
    submit_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sbatch commands without submitting them.",
    )

    run_parser = subparsers.add_parser(
        "run-manifest",
        help="Execute a manifest locally or inside a Slurm allocation.",
    )
    run_parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to a manifest JSON file produced by plan/submit.",
    )
    run_parser.add_argument(
        "--progress-file",
        type=Path,
        help="JSONL file used to record completed runs and support resume.",
    )
    run_parser.add_argument(
        "--parallelism",
        type=int,
        default=1,
        help="How many runs to execute in parallel.",
    )
    run_parser.add_argument(
        "--time-limit-minutes",
        type=int,
        default=0,
        help="If > 0, stop launching new runs near this walltime budget.",
    )
    run_parser.add_argument(
        "--time-buffer-minutes",
        type=int,
        default=10,
        help="Buffer used together with --time-limit-minutes.",
    )
    run_parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep executing pending tasks after a failure.",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "plan":
        return plan_command(args)
    if args.command == "submit":
        return submit_command(args)
    if args.command == "run-manifest":
        return execute_manifest(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
