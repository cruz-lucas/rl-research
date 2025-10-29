"""Functional abstractions for JAX-friendly tabular agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, Optional, Tuple, TypeVar

import jax
import jax.numpy as jnp
import jax.random as jrandom
from flax import struct

from rl_research.policies import ActionSelectionPolicy

PRNGKey = jax.Array


def _canonical_seed(seed: Optional[int]) -> int:
    """Converts an optional seed into a 32-bit unsigned integer."""
    if seed is None:
        import secrets

        return secrets.randbits(32)

    if not isinstance(seed, int):
        raise TypeError("Seed must be an integer or None.")
    return seed & 0xFFFFFFFF


@struct.dataclass
class AgentState:
    """State container shared across tabular agents."""

    q_values: jax.Array
    rng: PRNGKey


@struct.dataclass
class AgentParams:
    """State container shared across tabular agents."""

    num_states: int
    num_actions: int
    discount: float


AgentStateT = TypeVar("AgentStateT", bound=AgentState)
AgentParamsT = TypeVar("AgentParamsT", bound=AgentParams)


@dataclass(slots=True)
class UpdateResult:
    """Standard container for returning diagnostic information from updates."""

    td_error: Optional[float] = None
    info: Dict[str, Any] | None = None

    def as_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representation for convenience."""
        payload: Dict[str, Any] = {}
        if self.td_error is not None:
            payload["td_error"] = self.td_error
        if self.info:
            payload.update(self.info)
        return payload


class TabularAgent(ABC, Generic[AgentStateT, AgentParamsT]):
    """Base class for JAX-friendly tabular agents."""

    def __init__(
        self,
        params: AgentParamsT,
        seed: int | None = None,
        policy: ActionSelectionPolicy | None = None,
    ) -> None:
        self.num_states = params.num_states
        self.num_actions = params.num_actions
        self.discount = params.discount

        self._seed = _canonical_seed(seed)
        self._policy: ActionSelectionPolicy = policy or self._default_policy()

    @abstractmethod
    def _default_policy(self) -> ActionSelectionPolicy:
        """Provides a sensible default policy for the agent."""

    @abstractmethod
    def _initial_state(self, key: PRNGKey) -> AgentStateT:
        """Constructs the initial agent state for the given PRNG key."""

    def init(self, key: PRNGKey | None = None) -> AgentStateT:
        """Initialises a fresh agent state."""
        if key is None:
            key = jrandom.key(self._seed)
        return self._initial_state(key)

    def select_action(
        self, state: AgentStateT, obs: jax.Array
    ) -> Tuple[jax.Array, AgentStateT, Dict[str, float]]:
        """Selects an action using the configured policy."""
        obs_idx = jnp.asarray(obs, dtype=jnp.int32)
        q_row = state.q_values[obs_idx]

        extras = self._policy_extras(state, obs_idx)
        action, new_rng, info = self._policy.select(state.rng, q_row, extras)
        state = state.replace(rng=new_rng)
        return action, state, info

    def _policy_extras(
        self, state: AgentStateT, obs: jax.Array
    ) -> Dict[str, jnp.ndarray]:
        """Optional hook for subclasses to provide additional policy context."""
        del state, obs
        return {}

    @abstractmethod
    def update(
        self,
        agent_state: AgentStateT,
        obs: jax.Array,
        action: jax.Array,
        reward: jax.Array,
        next_obs: jax.Array,
        terminated: jax.Array | bool = False,
    ) -> Tuple[AgentStateT, UpdateResult]:
        """Updates the agent with a transition and returns diagnostic info."""

    @abstractmethod
    def train(self, state: AgentStateT) -> AgentStateT:
        """Configures agent for training."""
    
    @abstractmethod
    def eval(self, state: AgentStateT) -> AgentStateT:
        """Configures agent for evaluation."""
    
    def set_policy(self, policy: ActionSelectionPolicy) -> None:
        """Configures the action-selection policy."""
        self._policy = policy

    def policy(self) -> ActionSelectionPolicy:
        """Returns the currently configured policy."""
        return self._policy

    def q_values(self, state: AgentStateT) -> jax.Array:
        """Returns the current Q-value table."""
        return state.q_values
