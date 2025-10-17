"""Base environment interfaces used by the rl-research framework."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Mapping, MutableMapping

import jax.numpy as jnp

from rl_research.core.types import ActionSpec, EnvironmentSpec, ObservationSpec, PRNGKey


EnvState = Any


@dataclass(slots=True)
class StepResult:
    next_state: EnvState
    observation: jnp.ndarray
    reward: float
    discount: float
    done: bool
    info: Mapping[str, Any]


class Environment(ABC):
    """Abstract base class describing the interface environments must implement."""

    @abstractmethod
    def observation_spec(self) -> ObservationSpec:
        ...

    @abstractmethod
    def action_spec(self) -> ActionSpec:
        ...

    @abstractmethod
    def reset(self, rng: PRNGKey) -> tuple[EnvState, jnp.ndarray]:
        ...

    @abstractmethod
    def step(self, rng: PRNGKey, state: EnvState, action: jnp.ndarray) -> StepResult:
        ...


EnvFactory = Callable[[Mapping[str, Any]], Environment]


class EnvironmentRegistry(MutableMapping[str, EnvFactory]):
    """Registry mapping string identifiers to environment factories."""

    def __init__(self) -> None:
        self._builders: dict[str, EnvFactory] = {}

    def register(self, name: str, factory: EnvFactory, *, overwrite: bool = False) -> None:
        if not overwrite and name in self._builders:
            raise ValueError(f"Environment '{name}' is already registered.")
        self._builders[name] = factory

    def __getitem__(self, key: str) -> EnvFactory:
        return self._builders[key]

    def __setitem__(self, key: str, value: EnvFactory) -> None:
        self.register(key, value, overwrite=True)

    def __delitem__(self, key: str) -> None:
        del self._builders[key]

    def __iter__(self):
        return iter(self._builders)

    def __len__(self) -> int:
        return len(self._builders)


ENVIRONMENTS = EnvironmentRegistry()

