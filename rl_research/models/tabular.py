"""Tabular dynamics model abstractions shared across planners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np


class TabularDynamicsModel(ABC):
    """Interface for tabular environment models."""

    def __init__(self, *, num_states: int, num_actions: int) -> None:
        if num_states <= 0:
            raise ValueError("`num_states` must be positive.")
        if num_actions <= 0:
            raise ValueError("`num_actions` must be positive.")
        self._num_states = int(num_states)
        self._num_actions = int(num_actions)

    @property
    def num_states(self) -> int:
        return self._num_states

    @property
    def num_actions(self) -> int:
        return self._num_actions

    @abstractmethod
    def query(self, state: jax.Array, action: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Returns the model's prediction for `(next_state, reward)`."""

    @abstractmethod
    def update(
        self,
        state: jax.Array,
        action: jax.Array,
        reward: jax.Array,
        next_state: jax.Array,
        terminated: jax.Array | bool = False,
    ) -> None:
        """Incorporates a new transition into the model."""


class StaticTabularModel(TabularDynamicsModel):
    """Wraps a static expectation table to match the dynamics model API."""

    def __init__(self, expectation_table: np.ndarray | jnp.ndarray) -> None:
        array = jnp.asarray(expectation_table)
        if (
            array.ndim != 3
            or array.shape[2] < 2
        ):
            raise ValueError(
                "StaticTabularModel expects an array shaped "
                "(num_states, num_actions, >=2) storing next-state and reward expectations."
            )

        num_states, num_actions = array.shape[:2]
        super().__init__(num_states=num_states, num_actions=num_actions)

        next_states = array[..., 0]
        rewards = array[..., 1]

        max_state = jnp.asarray(num_states - 1, dtype=jnp.int32)
        min_state = jnp.asarray(0, dtype=jnp.int32)

        rounded_next = jnp.rint(next_states).astype(jnp.int32)
        self._next_states = jnp.clip(rounded_next, min_state, max_state)
        self._rewards = jnp.asarray(rewards, dtype=jnp.float32)

    def query(self, state: jax.Array, action: jax.Array) -> Tuple[jax.Array, jax.Array]:
        return self._next_states[state, action], self._rewards[state, action]

    def update(
        self,
        state: jax.Array,
        action: jax.Array,
        reward: jax.Array,
        next_state: jax.Array,
        terminated: jax.Array | bool = False,
    ) -> None:
        del state, action, reward, next_state, terminated
        # Static models are immutable; nothing to update.


class EmpiricalTabularModel(TabularDynamicsModel):
    """Count-based empirical model of tabular dynamics and rewards."""

    def __init__(
        self,
        *,
        num_states: int,
        num_actions: int,
        default_reward: float = 0.0,
    ) -> None:
        super().__init__(num_states=num_states, num_actions=num_actions)
        self._default_reward = jnp.asarray(default_reward, dtype=jnp.float32)
        self._state_values = jnp.arange(self.num_states, dtype=jnp.float32)
        self._transition_counts = jnp.zeros(
            (self.num_states, self.num_actions, self.num_states),
            dtype=jnp.float32,
        )
        self._reward_sums = jnp.zeros(
            (self.num_states, self.num_actions), dtype=jnp.float32
        )
        self._sa_counts = jnp.zeros(
            (self.num_states, self.num_actions), dtype=jnp.float32
        )

    def query(self, state: jax.Array, action: jax.Array) -> Tuple[jax.Array, jax.Array]:
        counts = self._transition_counts[state, action]
        total = jnp.sum(counts)

        expected_next = jnp.where(
            total > 0.0,
            jnp.sum(counts * self._state_values) / total,
            state.astype(jnp.float32),
        )

        expected_next = jnp.clip(
            jnp.rint(expected_next).astype(jnp.int32),
            0,
            self.num_states - 1,
        )

        visit_count = self._sa_counts[state, action]
        mean_reward = jnp.where(
            visit_count > 0.0,
            self._reward_sums[state, action] / visit_count,
            self._default_reward,
        )

        return expected_next, mean_reward.astype(jnp.float32)

    def update(
        self,
        state: jax.Array,
        action: jax.Array,
        reward: jax.Array,
        next_state: jax.Array,
        terminated: jax.Array | bool = False,
    ) -> None:
        del terminated

        state_idx = jnp.asarray(state, dtype=jnp.int32)
        action_idx = jnp.asarray(action, dtype=jnp.int32)
        reward_val = jnp.asarray(reward, dtype=jnp.float32)
        next_state_idx = jnp.asarray(next_state, dtype=jnp.int32)

        self._transition_counts = self._transition_counts.at[
            state_idx, action_idx, next_state_idx
        ].add(1.0)
        self._reward_sums = self._reward_sums.at[state_idx, action_idx].add(reward_val)
        self._sa_counts = self._sa_counts.at[state_idx, action_idx].add(1.0)
