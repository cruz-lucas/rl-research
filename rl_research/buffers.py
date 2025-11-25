import jax
import jax.numpy as jnp
from flax import struct
from typing import NamedTuple, Protocol
import gin


class Transition(NamedTuple):
    """Single transition tuple."""
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    next_observation: jnp.ndarray


@struct.dataclass
class BufferState:
    """Jittable buffer state for storing transitions."""
    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    discounts: jnp.ndarray
    next_observations: jnp.ndarray
    position: int
    size: int
    deduplicate: bool = False
    train_on_full_buffer: bool = False
    
    def is_ready(self, batch_size: int) -> bool:
        return jax.lax.cond(
            jnp.asarray(self.train_on_full_buffer),
            lambda _: self.size > 0,
            lambda _: self.size >= batch_size,
            operand=None
        )
    
    def push(self, transition: Transition) -> 'BufferState':
        """Add transition to buffer (circular)."""
        max_size = self.observations.shape[0]
        idx = self.position % max_size

        def _maybe_reduce(arr, value):
            """Check equality on the first dimension only (handles vector observations)."""
            eq = jnp.equal(arr, value)
            return jnp.all(eq, axis=tuple(range(1, eq.ndim))) if eq.ndim > 1 else eq

        def insert(_):
            observations = self.observations.at[idx].set(transition.observation)
            actions = self.actions.at[idx].set(transition.action)
            rewards = self.rewards.at[idx].set(transition.reward)
            discounts = self.discounts.at[idx].set(transition.discount)
            next_observations = self.next_observations.at[idx].set(transition.next_observation)

            return self.replace(
                observations=observations,
                actions=actions,
                rewards=rewards,
                discounts=discounts,
                next_observations=next_observations,
                position=self.position + 1,
                size=jnp.minimum(self.size + 1, max_size)
            )

        def insert_if_unique(_):
            valid_mask = jnp.arange(max_size) < self.size
            duplicate_mask = (
                _maybe_reduce(self.observations, transition.observation) &
                _maybe_reduce(self.actions, transition.action) &
                _maybe_reduce(self.rewards, transition.reward) &
                _maybe_reduce(self.discounts, transition.discount) &
                _maybe_reduce(self.next_observations, transition.next_observation) &
                valid_mask
            )
            is_duplicate = jnp.any(duplicate_mask)
            return jax.lax.cond(is_duplicate, lambda __: self, insert, operand=None)
        
        return jax.lax.cond(
            jnp.asarray(self.deduplicate),
            insert_if_unique,
            insert,
            operand=None
        )
    
    def sample(self, key: jax.Array, batch_size: int) -> tuple[Transition, jnp.ndarray]:
        """Sample batch from buffer. Returns transition and a mask for valid entries."""
        max_size = self.observations.shape[0]

        def full_buffer(_):
            idx = jnp.arange(max_size)
            mask = idx < self.size
            return idx, mask

        def random_batch(_):
            safe_size = jnp.maximum(self.size, 1)
            idx = jax.random.randint(key, (batch_size,), 0, safe_size)
            mask = jnp.ones((batch_size,), dtype=bool)
            return idx, mask

        indices, mask = jax.lax.cond(
            jnp.asarray(self.train_on_full_buffer),
            full_buffer,
            random_batch,
            operand=None
        )

        transition = Transition(
            observation=self.observations[indices],
            action=self.actions[indices],
            reward=self.rewards[indices],
            discount=self.discounts[indices],
            next_observation=self.next_observations[indices]
        )
        return transition, mask


class BaseBuffer(Protocol):
    """Protocol for configurable buffers."""

    def initial_state(self) -> BufferState:
        ...


@gin.configurable
class ReplayBuffer(BaseBuffer):
    """Uniform replay buffer with optional deduplication and full-buffer training."""

    def __init__(
        self,
        buffer_size: int = 1,
        deduplicate: bool = True,
        train_on_full_buffer: bool = False,
    ):
        self.buffer_size = buffer_size
        self.deduplicate = deduplicate
        self.train_on_full_buffer = train_on_full_buffer

    def initial_state(self) -> BufferState:
        zeros = lambda: jnp.zeros((self.buffer_size,))
        return BufferState(
            observations=zeros(),
            actions=zeros(),
            rewards=zeros(),
            discounts=zeros(),
            next_observations=zeros(),
            position=0,
            size=0,
            deduplicate=self.deduplicate,
            train_on_full_buffer=self.train_on_full_buffer,
        )


@gin.configurable
class FullBuffer(ReplayBuffer):
    """Train on the full buffer each step without deduplication."""

    def __init__(self, buffer_size: int = 1):
        super().__init__(
            buffer_size=buffer_size,
            deduplicate=False,
            train_on_full_buffer=True,
        )


@gin.configurable
class DeduplicatingBuffer(ReplayBuffer):
    """Deduplicate samples while training from random batches."""

    def __init__(self, buffer_size: int = 1):
        super().__init__(
            buffer_size=buffer_size,
            deduplicate=True,
            train_on_full_buffer=False,
        )


@gin.configurable
class DeduplicatingFullBuffer(ReplayBuffer):
    """Deduplicate samples and train on the full buffer every update."""

    def __init__(self, buffer_size: int = 1):
        super().__init__(
            buffer_size=buffer_size,
            deduplicate=True,
            train_on_full_buffer=True,
        )
