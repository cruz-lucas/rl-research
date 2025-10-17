"""Base classes for describing and dispatching experiment jobs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Mapping, Sequence


@dataclass(slots=True)
class ExperimentJob:
    """Description of a single experiment invocation."""

    name: str
    overrides: Sequence[str] = field(default_factory=tuple)
    config_path: str = "config"
    config_name: str = "config"
    working_dir: str = "."
    python_executable: str = "python"
    module: str = "rl_research.cli"
    env: Mapping[str, str] = field(default_factory=dict)

    def hydra_command(self) -> Sequence[str]:
        command = [
            self.python_executable,
            "-m",
            self.module,
            f"--config-path={self.config_path}",
            f"--config-name={self.config_name}",
        ]
        command.extend(self.overrides)
        return command


class Dispatcher(ABC):
    """Abstract dispatcher capable of launching experiment jobs."""

    @abstractmethod
    def dispatch(self, job: ExperimentJob) -> None:
        ...

