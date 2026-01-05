"""Run multiple seeds locally, then migrate MLflow runs into SQLite."""

from __future__ import annotations

import concurrent.futures
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, List

import tyro

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.migrate_mlflow_store import Args as MigrateArgs, migrate


def _sqlite_uri(db_path: Path) -> str:
    return f"sqlite:///{db_path.absolute()}"


def _run_seed_subprocess(seed: int, args: "Args") -> None:
    cmd: List[str] = [
        sys.executable,
        "-m",
        "rl_research.main",
        "--config",
        str(args.config.resolve()),
        "--seed",
        str(seed),
    ]

    for binding in args.binding:
        cmd.extend(["--binding", binding])

    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = str(args.mlruns_path)

    subprocess.run(
        cmd,
        check=True,
        cwd=REPO_ROOT,
        env=env,
    )


@dataclass
class Args:
    """Arguments for running seeds locally and migrating MLflow runs."""

    config: Annotated[
        Path,
        tyro.conf.arg(help="Path to the gin config file passed to rl_research.main."),
    ]
    seeds: Annotated[
        int,
        tyro.conf.arg(help="Number of seeds to run (0-indexed)."),
    ] = 10
    max_workers: Annotated[
        int,
        tyro.conf.arg(
            help="Parallel workers (0 defaults to min(seeds, CPU count)).",
        ),
    ] = 0
    binding: Annotated[
        List[str],
        tyro.conf.arg(help="Optional gin binding overrides (repeatable)."),
    ] = field(default_factory=list)
    mlruns_path: Annotated[
        Path,
        tyro.conf.arg(help="File-backed MLflow tracking directory."),
    ] = Path("./mlruns")
    sqlite_db: Annotated[
        Path,
        tyro.conf.arg(help="Destination SQLite backend store file."),
    ] = Path("mlruns.db")
    dest_artifact_root: Annotated[
        Path,
        tyro.conf.arg(help="Artifact root used for migrated SQLite runs."),
    ] = Path("./mlruns_sqlite_artifacts")
    skip_migration: Annotated[
        bool,
        tyro.conf.arg(help="Only run seeds; skip SQLite sync step."),
    ] = False


def run_seeds(args: Args) -> None:
    if args.seeds < 1:
        raise ValueError("--seeds must be >= 1")

    workers = args.max_workers or min(args.seeds, os.cpu_count() or 1)
    args.mlruns_path.mkdir(parents=True, exist_ok=True)

    print(
        f"Running {args.seeds} seeds with {workers} workers; "
        f"file-backed store: {args.mlruns_path}"
    )

    futures = {}
    errors: list[tuple[int, BaseException]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for seed in range(args.seeds):
            futures[executor.submit(_run_seed_subprocess, seed, args)] = seed

        for future in concurrent.futures.as_completed(futures):
            seed = futures[future]
            try:
                future.result()
                print(f"Seed {seed} finished.")
            except BaseException as exc:  # noqa: BLE001
                errors.append((seed, exc))
                print(f"Seed {seed} failed: {exc}")

    if errors:
        failed = ", ".join(str(seed) for seed, _ in errors)
        raise RuntimeError(f"Seed failures: {failed}") from errors[0][1]


def migrate_runs(args: Args) -> None:
    migrate(
        MigrateArgs(
            source_uri=str(args.mlruns_path),
            dest_uri=_sqlite_uri(args.sqlite_db),
            dest_artifact_root=args.dest_artifact_root,
        )
    )


def main(args: Args) -> None:
    run_seeds(args)
    if args.skip_migration:
        print("Skipping migration step (--skip-migration set).")
        return

    print(
        f"Migrating runs from {args.mlruns_path} "
        f"to SQLite DB at {args.sqlite_db.resolve()}"
    )
    migrate_runs(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
