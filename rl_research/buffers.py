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
        target_size = max(max_size, batch_size)

        def _pad(arr, value):
            pad_width = target_size - arr.shape[0]
            if pad_width == 0:
                return arr
            return jnp.pad(arr, (0, pad_width), constant_values=value)

        def full_buffer(_):
            idx = _pad(jnp.arange(max_size), 0)
            mask = _pad(jnp.arange(max_size) < self.size, False)
            return idx, mask

        def random_batch(_):
            safe_size = jnp.maximum(self.size, 1)
            idx = _pad(jax.random.randint(key, (batch_size,), 0, safe_size), 0)
            mask = _pad(jnp.ones((batch_size,), dtype=bool), False)
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


@struct.dataclass
class MonteCarloBufferState:
    """Buffer state that accumulates full episodes and stores Monte Carlo returns."""
    observations: jnp.ndarray
    actions: jnp.ndarray
    returns: jnp.ndarray
    discounts: jnp.ndarray
    next_observations: jnp.ndarray
    position: int
    size: int
    episode_observations: jnp.ndarray
    episode_actions: jnp.ndarray
    episode_rewards: jnp.ndarray
    episode_next_observations: jnp.ndarray
    episode_steps: int
    discount: float
    train_on_full_buffer: bool = False

    def is_ready(self, batch_size: int) -> bool:
        return jax.lax.cond(
            jnp.asarray(self.train_on_full_buffer),
            lambda _: self.size > 0,
            lambda _: self.size >= batch_size,
            operand=None
        )

    def _compute_returns(self, rewards: jnp.ndarray) -> jnp.ndarray:
        """Compute discounted returns for a full episode."""
        def scan_fn(carry, r):
            new_carry = r + self.discount * carry
            return new_carry, new_carry

        _, reversed_returns = jax.lax.scan(
            scan_fn,
            0.0,
            rewards[::-1]
        )
        return reversed_returns[::-1]

    def _write_episode(self) -> 'MonteCarloBufferState':
        """Materialize the collected episode into the main buffer with MC returns."""
        max_size = self.observations.shape[0]
        episode_length = self.episode_rewards.shape[0]
        returns = self._compute_returns(self.episode_rewards)
        discounts = jnp.full_like(returns, self.discount)

        def body(i, state):
            idx = state.position % max_size
            observations = state.observations.at[idx].set(self.episode_observations[i])
            actions = state.actions.at[idx].set(self.episode_actions[i])
            mc_returns = state.returns.at[idx].set(returns[i])
            discounts_arr = state.discounts.at[idx].set(discounts[i])
            next_obs = state.next_observations.at[idx].set(self.episode_next_observations[i])

            return state.replace(
                observations=observations,
                actions=actions,
                returns=mc_returns,
                discounts=discounts_arr,
                next_observations=next_obs,
                position=state.position + 1,
                size=jnp.minimum(state.size + 1, max_size)
            )

        new_state = jax.lax.fori_loop(0, episode_length, body, self)

        zeros_like_obs = jnp.zeros_like(self.episode_observations)
        zeros_like_actions = jnp.zeros_like(self.episode_actions)
        zeros_like_rewards = jnp.zeros_like(self.episode_rewards)
        zeros_like_next_obs = jnp.zeros_like(self.episode_next_observations)

        return new_state.replace(
            episode_observations=zeros_like_obs,
            episode_actions=zeros_like_actions,
            episode_rewards=zeros_like_rewards,
            episode_next_observations=zeros_like_next_obs,
            episode_steps=0
        )

    def push(self, transition: Transition) -> 'MonteCarloBufferState':
        """Accumulate transitions until an episode is complete, then store MC returns."""
        idx = self.episode_steps
        episode_observations = self.episode_observations.at[idx].set(transition.observation)
        episode_actions = self.episode_actions.at[idx].set(transition.action)
        episode_rewards = self.episode_rewards.at[idx].set(transition.reward)
        episode_next_observations = self.episode_next_observations.at[idx].set(transition.next_observation)

        updated_state = self.replace(
            episode_observations=episode_observations,
            episode_actions=episode_actions,
            episode_rewards=episode_rewards,
            episode_next_observations=episode_next_observations,
            episode_steps=self.episode_steps + 1
        )

        def finalize(_):
            return updated_state._write_episode()

        def keep_gathering(_):
            return updated_state

        episode_complete = updated_state.episode_steps >= updated_state.episode_rewards.shape[0]
        return jax.lax.cond(episode_complete, finalize, keep_gathering, operand=None)

    def sample(self, key: jax.Array, batch_size: int) -> tuple[Transition, jnp.ndarray]:
        """Sample Monte Carlo returns; optionally return the full buffer for training."""
        max_size = self.observations.shape[0]
        target_size = max(max_size, batch_size)

        def _pad(arr, value):
            pad_width = target_size - arr.shape[0]
            if pad_width == 0:
                return arr
            return jnp.pad(arr, (0, pad_width), constant_values=value)

        def full_buffer(_):
            idx = _pad(jnp.arange(max_size), 0)
            mask = _pad(jnp.arange(max_size) < self.size, False)
            return idx, mask

        def random_batch(_):
            safe_size = jnp.maximum(self.size, 1)
            idx = _pad(jax.random.randint(key, (batch_size,), 0, safe_size), 0)
            mask = _pad(jnp.ones((batch_size,), dtype=bool), False)
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
            reward=self.returns[indices],
            discount=self.discounts[indices],
            next_observation=self.next_observations[indices]
        )
        return transition, mask


@struct.dataclass
class RMaxMonteCarloBufferState:
    """Monte Carlo buffer that applies optimistic R-Max returns to unknown pairs."""
    observations: jnp.ndarray
    actions: jnp.ndarray
    returns: jnp.ndarray
    discounts: jnp.ndarray
    next_observations: jnp.ndarray
    position: int
    size: int
    episode_observations: jnp.ndarray
    episode_actions: jnp.ndarray
    episode_rewards: jnp.ndarray
    episode_next_observations: jnp.ndarray
    episode_steps: int
    discount: float
    r_max: float
    known_threshold: int
    self_loop_unknown: bool
    visit_counts: jnp.ndarray
    train_on_full_buffer: bool = False

    def is_ready(self, batch_size: int) -> bool:
        return jax.lax.cond(
            jnp.asarray(self.train_on_full_buffer),
            lambda _: self.size > 0,
            lambda _: self.size >= batch_size,
            operand=None
        )

    def _compute_unknown_mask(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return per-step unknown mask and updated visit counts."""
        def body(counts, oa):
            obs, act = oa
            s = obs.astype(jnp.int32).squeeze()
            a = act.astype(jnp.int32).squeeze()
            is_unknown = counts[s, a] < self.known_threshold
            counts = counts.at[s, a].add(1)
            return counts, is_unknown

        return jax.lax.scan(body, self.visit_counts, (self.episode_observations, self.episode_actions))

    def _compute_returns(self, rewards: jnp.ndarray, unknown_mask: jnp.ndarray) -> jnp.ndarray:
        """Compute discounted returns with optimistic handling of unknown pairs."""
        optimistic_value = self.r_max / (1.0 - self.discount)

        def scan_fn(next_return, inputs):
            reward, is_unknown = inputs
            optimistic_return = jax.lax.cond(
                jnp.asarray(self.self_loop_unknown),
                lambda _: optimistic_value,
                lambda _: self.r_max + self.discount * next_return,
                operand=None
            )
            regular_return = reward + self.discount * next_return
            mc_return = jax.lax.select(is_unknown, optimistic_return, regular_return)
            return mc_return, mc_return

        _, reversed_returns = jax.lax.scan(
            scan_fn,
            0.0,
            (rewards[::-1], unknown_mask[::-1])
        )
        return reversed_returns[::-1]

    def _write_episode(self) -> 'RMaxMonteCarloBufferState':
        """Materialize the collected episode into the main buffer with MC returns."""
        max_size = self.observations.shape[0]
        episode_length = self.episode_rewards.shape[0]
        updated_counts, unknown_mask = self._compute_unknown_mask()
        returns = self._compute_returns(self.episode_rewards, unknown_mask)
        discounts = jnp.full_like(returns, self.discount)

        def body(i, state):
            idx = state.position % max_size
            observations = state.observations.at[idx].set(self.episode_observations[i])
            actions = state.actions.at[idx].set(self.episode_actions[i])
            mc_returns = state.returns.at[idx].set(returns[i])
            discounts_arr = state.discounts.at[idx].set(discounts[i])
            next_obs = state.next_observations.at[idx].set(self.episode_next_observations[i])

            return state.replace(
                observations=observations,
                actions=actions,
                returns=mc_returns,
                discounts=discounts_arr,
                next_observations=next_obs,
                position=state.position + 1,
                size=jnp.minimum(state.size + 1, max_size)
            )

        new_state = jax.lax.fori_loop(0, episode_length, body, self)

        zeros_like_obs = jnp.zeros_like(self.episode_observations)
        zeros_like_actions = jnp.zeros_like(self.episode_actions)
        zeros_like_rewards = jnp.zeros_like(self.episode_rewards)
        zeros_like_next_obs = jnp.zeros_like(self.episode_next_observations)

        return new_state.replace(
            episode_observations=zeros_like_obs,
            episode_actions=zeros_like_actions,
            episode_rewards=zeros_like_rewards,
            episode_next_observations=zeros_like_next_obs,
            episode_steps=0,
            visit_counts=updated_counts
        )

    def push(self, transition: Transition) -> 'RMaxMonteCarloBufferState':
        """Accumulate transitions until an episode is complete, then store MC returns."""
        idx = self.episode_steps
        episode_observations = self.episode_observations.at[idx].set(transition.observation)
        episode_actions = self.episode_actions.at[idx].set(transition.action)
        episode_rewards = self.episode_rewards.at[idx].set(transition.reward)
        episode_next_observations = self.episode_next_observations.at[idx].set(transition.next_observation)

        updated_state = self.replace(
            episode_observations=episode_observations,
            episode_actions=episode_actions,
            episode_rewards=episode_rewards,
            episode_next_observations=episode_next_observations,
            episode_steps=self.episode_steps + 1
        )

        def finalize(_):
            return updated_state._write_episode()

        def keep_gathering(_):
            return updated_state

        episode_complete = updated_state.episode_steps >= updated_state.episode_rewards.shape[0]
        return jax.lax.cond(episode_complete, finalize, keep_gathering, operand=None)

    def sample(self, key: jax.Array, batch_size: int) -> tuple[Transition, jnp.ndarray]:
        """Sample Monte Carlo returns; optionally return the full buffer for training."""
        max_size = self.observations.shape[0]
        target_size = max(max_size, batch_size)

        def _pad(arr, value):
            pad_width = target_size - arr.shape[0]
            if pad_width == 0:
                return arr
            return jnp.pad(arr, (0, pad_width), constant_values=value)

        def full_buffer(_):
            idx = _pad(jnp.arange(max_size), 0)
            mask = _pad(jnp.arange(max_size) < self.size, False)
            return idx, mask

        def random_batch(_):
            safe_size = jnp.maximum(self.size, 1)
            idx = _pad(jax.random.randint(key, (batch_size,), 0, safe_size), 0)
            mask = _pad(jnp.ones((batch_size,), dtype=bool), False)
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
            reward=self.returns[indices],
            discount=self.discounts[indices],
            next_observation=self.next_observations[indices]
        )
        return transition, mask


class BaseBuffer(Protocol):
    """Protocol for configurable buffers."""

    def initial_state(self) -> BufferState | MonteCarloBufferState | RMaxMonteCarloBufferState:
        ...


@gin.configurable
class ReplayBuffer(BaseBuffer):
    """Uniform replay buffer with optional deduplication and full-buffer training."""

    def __init__(
        self,
        buffer_size: int = 1,
        deduplicate: bool = False,
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


@gin.configurable
class MonteCarloBuffer(BaseBuffer):
    """Buffer that stores per-episode Monte Carlo returns before sampling."""

    def __init__(
        self,
        buffer_size: int = 1,
        episode_length: int = 1,
        discount: float = 0.9,
        train_on_full_buffer: bool = False,
    ):
        self.buffer_size = buffer_size
        self.episode_length = episode_length
        self.discount = discount
        self.train_on_full_buffer = train_on_full_buffer

    def initial_state(self) -> MonteCarloBufferState:
        zeros = lambda: jnp.zeros((self.buffer_size,))
        episode_zeros = lambda: jnp.zeros((self.episode_length,))
        return MonteCarloBufferState(
            observations=zeros(),
            actions=zeros(),
            returns=zeros(),
            discounts=zeros(),
            next_observations=zeros(),
            position=0,
            size=0,
            episode_observations=episode_zeros(),
            episode_actions=episode_zeros(),
            episode_rewards=episode_zeros(),
            episode_next_observations=episode_zeros(),
            episode_steps=0,
            discount=self.discount,
            train_on_full_buffer=self.train_on_full_buffer,
        )


@gin.configurable
class RMaxMonteCarloBuffer(BaseBuffer):
    """Monte Carlo buffer that applies R-Max optimism to unknown state-actions."""

    def __init__(
        self,
        num_states: int | None = None,
        num_actions: int | None = None,
        buffer_size: int = 1,
        episode_length: int = 1,
        discount: float = 0.9,
        r_max: float = 1.0,
        known_threshold: int = 1,
        self_loop_unknown: bool = True,
        train_on_full_buffer: bool = False,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.buffer_size = buffer_size
        self.episode_length = episode_length
        self.discount = discount
        self.r_max = r_max
        self.known_threshold = known_threshold
        self.self_loop_unknown = self_loop_unknown
        self.train_on_full_buffer = train_on_full_buffer

    def initial_state(self) -> RMaxMonteCarloBufferState:
        if self.num_states is None or self.num_actions is None:
            raise ValueError("RMaxMonteCarloBuffer requires num_states and num_actions.")

        zeros = lambda: jnp.zeros((self.buffer_size,))
        episode_zeros = lambda: jnp.zeros((self.episode_length,))
        return RMaxMonteCarloBufferState(
            observations=zeros(),
            actions=zeros(),
            returns=zeros(),
            discounts=zeros(),
            next_observations=zeros(),
            position=0,
            size=0,
            episode_observations=episode_zeros(),
            episode_actions=episode_zeros(),
            episode_rewards=episode_zeros(),
            episode_next_observations=episode_zeros(),
            episode_steps=0,
            discount=self.discount,
            r_max=self.r_max,
            known_threshold=self.known_threshold,
            self_loop_unknown=self.self_loop_unknown,
            visit_counts=jnp.zeros((self.num_states, self.num_actions), dtype=jnp.int32),
            train_on_full_buffer=self.train_on_full_buffer,
        )
