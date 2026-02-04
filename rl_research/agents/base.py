from abc import abstractmethod
from typing import Protocol

import jax
import jax.numpy as jnp
from flax import struct

from rl_research.buffers import Transition


@struct.dataclass
class AgentState(Protocol):
    """State for Q-learning agent."""

    ...


class BaseAgent:
    @abstractmethod
    def __init__(self, num_states, num_actions): ...

    @abstractmethod
    def initial_state(self) -> AgentState: ...

    @abstractmethod
    def select_action(
        self, state: AgentState, obs: jnp.ndarray, key: jax.Array, is_training: bool
    ) -> jnp.ndarray: ...

    @abstractmethod
    def update(
        self,
        state: AgentState,
        batch: Transition,
    ) -> tuple[AgentState, jax.Array]: ...

    @abstractmethod
    def bootstrap_value(
        self, state: AgentState, next_observation: jnp.ndarray
    ) -> jax.Array: ...
