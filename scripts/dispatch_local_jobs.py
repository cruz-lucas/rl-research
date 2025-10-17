"""Helper script to launch a batch of local experiments via Hydra overrides.

The combinations are:
    agents:            q_learning_epsgreedy, q_learning_ucb
    environments:      riverswim, sixarms
    episode settings:  (episodes=30, max_steps=50), (episodes=15, max_steps=100)

Usage:
    uv run python scripts/dispatch_local_jobs.py
    uv run python scripts/dispatch_local_jobs.py --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

from rl_research.dispatch.base import ExperimentJob
from rl_research.dispatch.local import LocalDispatcher


AGENTS: Sequence[str] = ("rmax", "mbie", "mbie_eb")
ENVIRONMENTS: Sequence[str] = ("riverswim", "sixarms")
SETTINGS: Sequence[str] = ["riverswim_sixarms"]

REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_jobs() -> Iterable[ExperimentJob]:
    for agent in AGENTS:
        for environment in ENVIRONMENTS:
            # for experiment in SETTINGS:
                # job_name = f"{environment}_{agent}_{experiment}"
            job_name = f"{environment}_{agent}"
            overrides = (
                f"agent={agent}",
                f"environment={environment}",
                # f"experiment={experiment}",
                f"experiment.name={job_name}",
                "tracking.type=mlflow",
            )

            yield ExperimentJob(
                name=job_name,
                overrides=overrides,
                working_dir=str(REPO_ROOT),
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Dispatch a batch of local RL jobs.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands instead of executing them.",
    )
    parser.add_argument(
        "--capture-output",
        action="store_true",
        help="Capture subprocess stdout/stderr for each job.",
    )
    args = parser.parse_args()

    dispatcher = LocalDispatcher(dry_run=args.dry_run, capture_output=args.capture_output)
    for job in _build_jobs():
        dispatcher.dispatch(job)


if __name__ == "__main__":
    main()
