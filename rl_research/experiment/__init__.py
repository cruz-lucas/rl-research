"""User-facing entry-points for the experiment package."""

from rl_research.experiment.experiment import run_experiment, log_experiment, ExperimentParams

__all__ = [
    "run_experiment",
    "log_experiment",
    "ExperimentParams"
]
