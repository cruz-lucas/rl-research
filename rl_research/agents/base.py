"""Base interfaces for JAX-based reinforcement learning agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, MutableMapping, Protocol

import jax.numpy as jnp

from rl_research.core.types import EnvironmentSpec, PRNGKey, Transition


@dataclass(slots=True)
class AgentState:
    """Generic container for agent parameters and mutable training state."""

    params: Any
    mutable_state: Any | None = None
    optimizer_state: Any | None = None

    def replace(self, **updates: Any) -> "AgentState":
        return replace(self, **updates)


class Agent(ABC):
    """Abstract base class describing the minimum capabilities of an agent."""
    pass
    # def __init__(self, config: Mapping[str, Any]) -> None:
    #     self.config = dict(config)

    # @abstractmethod
    # def init(self, rng: PRNGKey, spec: EnvironmentSpec) -> AgentState:
    #     """Initialise learnable parameters and internal state."""

    # @abstractmethod
    # def select_action(
    #     self, rng: PRNGKey, agent_state: AgentState, observation: jnp.ndarray
    # ) -> tuple[jnp.ndarray, AgentState, Mapping[str, Any]]:
    #     """Select an action for the provided observation."""

    # @abstractmethod
    # def update(
    #     self, rng: PRNGKey, agent_state: AgentState, transition: Transition
    # ) -> tuple[AgentState, Mapping[str, Any]]:
    #     """Update the agent given a single transition."""

    # def on_episode_end(
    #     self,
    #     *,
    #     agent_state: AgentState,
    #     episode_return: float,
    #     episode_length: int,
    # ) -> Mapping[str, float]:
    #     """Hook executed when an episode finishes to emit additional metrics."""
    #     return {}

    # def estimate_action_values(self, observations: jnp.ndarray) -> jnp.ndarray:
    #     """Compute Q-values for a batch of observations.

    #     Agents that do not operate on Q-values should override this with an informative error.
    #     """
    #     raise NotImplementedError(
    #         f"{self.__class__.__name__} does not implement estimate_action_values."
    #     )


AgentFactory = Callable[[Mapping[str, Any]], Agent]


class AgentRegistry(MutableMapping[str, AgentFactory]):
    """Simple registry that maps string identifiers to agent factories."""

    def __init__(self) -> None:
        self._builders: dict[str, AgentFactory] = {}

    def register(self, name: str, factory: AgentFactory, *, overwrite: bool = False) -> None:
        if not overwrite and name in self._builders:
            raise ValueError(f"Agent '{name}' is already registered.")
        self._builders[name] = factory

    def __getitem__(self, key: str) -> AgentFactory:
        return self._builders[key]

    def __setitem__(self, key: str, value: AgentFactory) -> None:
        self.register(key, value, overwrite=True)

    def __delitem__(self, key: str) -> None:
        del self._builders[key]

    def __iter__(self):
        return iter(self._builders)

    def __len__(self) -> int:
        return len(self._builders)


AGENTS = AgentRegistry()

