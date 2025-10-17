"""Hyperparameter search utilities integrated with Hydra overrides."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Iterable, Mapping, Sequence

from rl_research.dispatch.base import ExperimentJob


def _format_override(name: str, value: object) -> str:
    if isinstance(value, bool):
        value_str = "true" if value else "false"
    else:
        value_str = str(value)
    return f"{name}={value_str}"


@dataclass(slots=True)
class HyperparameterSearch:
    name: str
    base_job: ExperimentJob
    grid: Mapping[str, Sequence[object]]
    seeds: Sequence[int] = (0,)
    additional_overrides: Sequence[str] = field(default_factory=tuple)

    def iter_jobs(self) -> Iterable[ExperimentJob]:
        keys = list(self.grid.keys())
        values_product = list(product(*[self.grid[k] for k in keys]))
        for idx, combo in enumerate(values_product):
            overrides = list(self.base_job.overrides)
            overrides.extend(self.additional_overrides)
            overrides.extend(_format_override(k, v) for k, v in zip(keys, combo))

            job_name = f"{self.name}_trial{idx}"
            yield ExperimentJob(
                name=job_name,
                overrides=tuple(overrides),
                config_path=self.base_job.config_path,
                config_name=self.base_job.config_name,
                working_dir=self.base_job.working_dir,
                python_executable=self.base_job.python_executable,
                module=self.base_job.module,
                env=self.base_job.env,
            )


@dataclass(slots=True)
class EvaluationPlan:
    base_job: ExperimentJob
    best_overrides: Sequence[str]
    eval_seeds: Sequence[int]

    def iter_jobs(self) -> Iterable[ExperimentJob]:
        for seed in self.eval_seeds:
            overrides = list(self.base_job.overrides)
            overrides.extend(self.best_overrides)
            overrides.append(f"experiment.training_seeds=[{seed}]")
            name = f"{self.base_job.name}_eval_seed{seed}"
            yield ExperimentJob(
                name=name,
                overrides=tuple(overrides),
                config_path=self.base_job.config_path,
                config_name=self.base_job.config_name,
                working_dir=self.base_job.working_dir,
                python_executable=self.base_job.python_executable,
                module=self.base_job.module,
                env=self.base_job.env,
            )

