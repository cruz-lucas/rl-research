from typing import NamedTuple, Protocol

import gin
import jax
import jax.numpy as jnp
from flax import struct


class Transition(NamedTuple):
    """Single transition tuple."""

    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    next_observation: jnp.ndarray
    terminal: jnp.ndarray


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

    def push(self, transition: Transition, bootstrap_value: float = 0) -> "BufferState":
        """Add transition to buffer (circular)."""
        max_size = self.observations.shape[0]
        idx = self.position % max_size

        flat_obs = transition.observation.reshape(-1)
        flat_next_obs = transition.next_observation.reshape(-1)

        observations = self.observations.at[idx].set(flat_obs)
        actions = self.actions.at[idx].set(transition.action)
        rewards = self.rewards.at[idx].set(transition.reward)
        discounts = self.discounts.at[idx].set(transition.discount)
        next_observations = self.next_observations.at[idx].set(
            flat_next_obs
        )
        terminals = self.terminals.at[idx].set(transition.terminal)

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
        indices = jax.random.randint(key, (batch_size,), 0, safe_size)

        transition = Transition(
            observation=self.observations[indices],
            action=self.actions[indices],
            reward=self.rewards[indices],
            discount=self.discounts[indices],
            next_observation=self.next_observations[indices],
            terminal=self.terminals[indices],
        )
        return transition


@struct.dataclass
class MonteCarloBufferState:
    """Buffer state that accumulates full episodes and stores Monte Carlo returns."""

    observations: jnp.ndarray
    actions: jnp.ndarray
    returns: jnp.ndarray
    discounts: jnp.ndarray
    next_observations: jnp.ndarray
    terminals: jnp.ndarray
    position: int
    size: int
    episode_observations: jnp.ndarray
    episode_actions: jnp.ndarray
    episode_rewards: jnp.ndarray
    episode_next_observations: jnp.ndarray
    episode_terminals: jnp.ndarray
    episode_steps: int
    discount: float

    def is_ready(self, batch_size: int) -> bool:
        return self.size >= batch_size

    def _compute_returns(
        self, rewards: jnp.ndarray, bootstrap_value: float
    ) -> jnp.ndarray:
        """Compute discounted returns for a full episode, bootstrapping the tail."""

        def scan_fn(carry, r):
            new_carry = r + self.discount * carry
            return new_carry, new_carry

        _, reversed_returns = jax.lax.scan(scan_fn, bootstrap_value, rewards[::-1])
        return reversed_returns[::-1]

    def _write_episode(self, bootstrap_value: float) -> "MonteCarloBufferState":
        """Materialize the collected episode into the main buffer with MC returns."""
        max_size = self.observations.shape[0]
        episode_length = self.episode_rewards.shape[0]
        returns = self._compute_returns(self.episode_rewards, bootstrap_value)
        discounts = jnp.full_like(returns, self.discount)
        episode_terminals = self.episode_terminals

        def body(i, state):
            idx = state.position % max_size
            observations = state.observations.at[idx].set(self.episode_observations[i])
            actions = state.actions.at[idx].set(self.episode_actions[i])
            mc_returns = state.returns.at[idx].set(returns[i])
            discounts_arr = state.discounts.at[idx].set(discounts[i])
            next_obs = state.next_observations.at[idx].set(
                self.episode_next_observations[i]
            )
            terminals_arr = state.terminals.at[idx].set(episode_terminals[i])

            return state.replace(
                observations=observations,
                actions=actions,
                returns=mc_returns,
                discounts=discounts_arr,
                next_observations=next_obs,
                terminals=terminals_arr,
                position=state.position + 1,
                size=jnp.minimum(state.size + 1, max_size),
            )

        new_state = jax.lax.fori_loop(0, episode_length, body, self)

        zeros_like_obs = jnp.zeros_like(self.episode_observations)
        zeros_like_actions = jnp.zeros_like(self.episode_actions)
        zeros_like_rewards = jnp.zeros_like(self.episode_rewards)
        zeros_like_next_obs = jnp.zeros_like(self.episode_next_observations)
        zeros_like_terminals = jnp.zeros_like(self.episode_terminals, dtype=bool)

        return new_state.replace(
            episode_observations=zeros_like_obs,
            episode_actions=zeros_like_actions,
            episode_rewards=zeros_like_rewards,
            episode_next_observations=zeros_like_next_obs,
            episode_terminals=zeros_like_terminals,
            episode_steps=0,
        )

    def push(
        self, transition: Transition, bootstrap_value: float = 0
    ) -> "MonteCarloBufferState":
        """Accumulate transitions until an episode is complete, then store MC returns."""
        idx = self.episode_steps
        episode_observations = self.episode_observations.at[idx].set(
            transition.observation
        )
        episode_actions = self.episode_actions.at[idx].set(transition.action)
        episode_rewards = self.episode_rewards.at[idx].set(transition.reward)
        episode_next_observations = self.episode_next_observations.at[idx].set(
            transition.next_observation
        )
        episode_terminals = self.episode_terminals.at[idx].set(transition.terminal)

        updated_state = self.replace(
            episode_observations=episode_observations,
            episode_actions=episode_actions,
            episode_rewards=episode_rewards,
            episode_next_observations=episode_next_observations,
            episode_terminals=episode_terminals,
            episode_steps=self.episode_steps + 1,
        )

        def finalize(_):
            return updated_state._write_episode(bootstrap_value)

        def keep_gathering(_):
            return updated_state

        episode_complete = jnp.logical_or(
            updated_state.episode_steps >= updated_state.episode_rewards.shape[0],
            transition.terminal,
        )
        return jax.lax.cond(episode_complete, finalize, keep_gathering, operand=None)

    def sample(self, key: jax.Array, batch_size: int) -> Transition:
        """Sample Monte Carlo returns."""
        safe_size = jnp.maximum(self.size, 1)
        indices = jax.random.randint(key, (batch_size,), 0, safe_size)

        transition = Transition(
            observation=self.observations[indices],
            action=self.actions[indices],
            reward=self.returns[indices],
            discount=self.discounts[indices],
            next_observation=self.next_observations[indices],
            terminal=self.terminals[indices],
        )
        return transition


class BaseBuffer(Protocol):
    """Protocol for configurable buffers."""

    def initial_state(self) -> BufferState | MonteCarloBufferState: ...


@gin.configurable
class ReplayBuffer(BaseBuffer):
    """Uniform replay buffer with optional deduplication and full-buffer training."""

    def __init__(
        self,
        buffer_size: int = 1,
    ):
        self.buffer_size = buffer_size

    def initial_state(self) -> BufferState:
        zeros = lambda: jnp.zeros((self.buffer_size, 1))
        return BufferState(
            observations=zeros(),
            actions=zeros(),
            rewards=zeros(),
            discounts=zeros(),
            next_observations=zeros(),
            terminals=jnp.zeros((self.buffer_size,), dtype=bool),
            position=0,
            size=0,
        )


@gin.configurable
class MonteCarloBuffer(BaseBuffer):
    """Buffer that stores per-episode Monte Carlo returns before sampling."""

    def __init__(
        self,
        buffer_size: int = 1,
        episode_length: int = 1,
        discount: float = 0.9,
    ):
        self.buffer_size = buffer_size
        self.episode_length = episode_length
        self.discount = discount

    def initial_state(self) -> MonteCarloBufferState:
        zeros = lambda: jnp.zeros((self.buffer_size,))
        episode_zeros = lambda: jnp.zeros((self.episode_length,))
        episode_terminal_zeros = lambda: jnp.zeros((self.episode_length,), dtype=bool)
        return MonteCarloBufferState(
            observations=zeros(),
            actions=zeros(),
            returns=zeros(),
            discounts=zeros(),
            next_observations=zeros(),
            terminals=jnp.zeros((self.buffer_size,), dtype=bool),
            position=0,
            size=0,
            episode_observations=episode_zeros(),
            episode_actions=episode_zeros(),
            episode_rewards=episode_zeros(),
            episode_next_observations=episode_zeros(),
            episode_terminals=episode_terminal_zeros(),
            episode_steps=0,
            discount=self.discount,
        )


@gin.configurable
class FlatteningReplayBuffer(BaseBuffer):
    """Replay buffer that flattens observations before storing."""

    def __init__(
        self,
        buffer_size: int,
        obs_shape: tuple[int, ...],
        dtype=jnp.float32,
    ):
        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.obs_dim = int(jnp.prod(jnp.array(obs_shape)))
        self.dtype = dtype

    def initial_state(self) -> BufferState:
        obs = jnp.zeros((self.buffer_size, self.obs_dim), dtype=self.dtype)
        return BufferState(
            observations=obs,
            actions=jnp.zeros((self.buffer_size,), dtype=jnp.int32),
            rewards=jnp.zeros((self.buffer_size,), dtype=jnp.float32),
            discounts=jnp.zeros((self.buffer_size,), dtype=jnp.float32),
            next_observations=obs,
            terminals=jnp.zeros((self.buffer_size,), dtype=bool),
            position=0,
            size=0,
        )
    
    
