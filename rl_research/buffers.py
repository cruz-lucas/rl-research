from typing import NamedTuple, Protocol

import gin
import jax
import jax.numpy as jnp
from flax import struct


class Transition(struct.PyTreeNode):
    """Single transition tuple."""

    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    next_observation: jnp.ndarray
    terminal: jnp.ndarray
    mask: jnp.ndarray


class BufferState(struct.PyTreeNode):
    """Jittable buffer state for storing transitions."""

    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    discounts: jnp.ndarray
    next_observations: jnp.ndarray
    terminals: jnp.ndarray
    position: int
    size: int

    def is_ready(self, batch_size: int) -> bool:
        return self.size >= batch_size

    def push(
            self,
            observation: jax.Array,
            action: jax.Array,
            reward: jax.Array,
            next_observation: jax.Array,
            discount: jax.Array,
            terminal: jax.Array,
            bootstrap_value: jax.Array | None = None,
        ) -> "BufferState":
        """Add transition to buffer (circular)."""
        max_size = self.observations.shape[0]
        idx = self.position % max_size

        flat_obs = observation.reshape(-1)
        flat_next_obs = next_observation.reshape(-1)

        observations = self.observations.at[idx].set(flat_obs)
        actions = self.actions.at[idx].set(action)
        rewards = self.rewards.at[idx].set(reward)
        discounts = self.discounts.at[idx].set(discount)
        next_observations = self.next_observations.at[idx].set(
            flat_next_obs
        )
        terminals = self.terminals.at[idx].set(terminal)

        return self.replace(
            observations=observations,
            actions=actions,
            rewards=rewards,
            discounts=discounts,
            next_observations=next_observations,
            terminals=terminals,
            position=self.position + 1,
            size=jnp.minimum(self.size + 1, max_size),
        )

    def sample(self, key: jax.Array, batch_size: int) -> Transition:
        """Sample batch from buffer. Returns transitions."""
        safe_size = jnp.maximum(self.size, 1)
        indices = jax.random.randint(key, (batch_size,), 0, safe_size) # with replacement
        transition = Transition(
            observation=self.observations[indices],
            action=self.actions[indices],
            reward=self.rewards[indices],
            discount=self.discounts[indices],
            next_observation=self.next_observations[indices],
            terminal=self.terminals[indices],
            mask=jnp.ones_like(indices)
        )
        
        return transition


class EntireBufferState(BufferState):
    def sample(self, key: jax.Array, batch_size: int) -> Transition:
        """Sample entire buffer. Returns transitions."""
        max_size = self.observations.shape[0]
        mask = jnp.arange(max_size) < self.size

        transition = Transition(
            observation=self.observations,
            action=self.actions,
            reward=self.rewards,
            discount=self.discounts,
            next_observation=self.next_observations,
            terminal=self.terminals,
            mask=mask,
        )
        return transition


class BaseBuffer(Protocol):
    """Protocol for configurable buffers."""

    def initial_state(self) -> BufferState: ...


@gin.configurable
class ReplayBuffer(BaseBuffer):
    """Uniform replay buffer."""

    def __init__(
        self,
        buffer_size: int = 1,
        obs_shape: tuple[int, ...] = (1,),
        obs_dtype=jnp.int32,
        action_dtype=jnp.int32,
    ):
        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.obs_dim = int(jnp.prod(jnp.array(obs_shape)))
        self.obs_dtype = obs_dtype
        self.action_dtype = action_dtype

    def initial_state(self) -> BufferState:
        return BufferState(
            observations=jnp.zeros((self.buffer_size, self.obs_dim), dtype=self.obs_dtype),
            actions=jnp.zeros((self.buffer_size,), dtype=self.action_dtype),
            rewards=jnp.zeros((self.buffer_size,), dtype=jnp.float32),
            discounts=jnp.zeros((self.buffer_size,), dtype=jnp.float32),
            next_observations=jnp.zeros((self.buffer_size, self.obs_dim), dtype=self.obs_dtype),
            terminals=jnp.zeros((self.buffer_size,), dtype=jnp.bool_),
            position=0,
            size=0,
        )
    

@gin.configurable
class EntireReplayBuffer(ReplayBuffer):
    """Uniform replay the entire buffer."""

    def initial_state(self) -> EntireBufferState:
        return EntireBufferState(
            observations=jnp.zeros((self.buffer_size, self.obs_dim), dtype=self.obs_dtype),
            actions=jnp.zeros((self.buffer_size,), dtype=self.action_dtype),
            rewards=jnp.zeros((self.buffer_size,), dtype=jnp.float32),
            discounts=jnp.zeros((self.buffer_size,), dtype=jnp.float32),
            next_observations=jnp.zeros((self.buffer_size, self.obs_dim), dtype=self.obs_dtype),
            terminals=jnp.zeros((self.buffer_size,), dtype=jnp.bool_),
            position=0,
            size=0,
        )