import jax
import jax.numpy as jnp
from typing import Literal, Protocol
from flax import struct
from rl_research.policies import _select_greedy, _select_random, _select_epsilon_greedy, _select_ucb
from rl_research.experiment import Transition
from abc import abstractmethod

@struct.dataclass
class AgentState(Protocol):
    """State for Q-learning agent."""
    ...

class BaseAgent:

    @abstractmethod
    def __init__(self, num_states, num_actions):
        ...
    
    @abstractmethod
    def initial_state(self) -> AgentState:
        ...
    
    @abstractmethod
    def select_action(
        self,
        state: AgentState,
        obs: jnp.ndarray,
        key: jax.Array,
        is_training: bool
    ) -> jnp.ndarray:
        ...
    
    @abstractmethod
    def update(
        self,
        state: AgentState,
        batch: Transition,
        batch_mask: jnp.ndarray | None = None
    ) -> tuple[AgentState, jax.Array]:
        ...
