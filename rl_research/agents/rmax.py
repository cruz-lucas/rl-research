"""Implementation of the R-MAX algorithm with JAX."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax import struct

from rl_research.agents.base import AgentState, AgentParams, TabularAgent, UpdateResult
from rl_research.policies import ActionSelectionPolicy, GreedyPolicy


@struct.dataclass
class RMaxState(AgentState):
    """Container for the mutable quantities tracked by R-MAX."""

    sa_counts: jax.Array
    reward_sums: jax.Array
    trans_counts: jax.Array


@struct.dataclass
class RMaxParams(AgentParams):
    """Static params for Rmax."""

    threshold: float
    r_max: float
    m: int


class RMaxAgent(TabularAgent[RMaxState, RMaxParams]):
    """Model-based R-MAX agent implemented with pure JAX transformations."""

    def __init__(
        self,
        params: RMaxParams,
        seed: int | None = None,
        policy: ActionSelectionPolicy | None = None,
    ) -> None:
        self.threshold = params.threshold
        self.r_max = params.r_max
        self.m = params.m

        self._optimistic_value = self.r_max / (1.0 - params.discount)

        if self.threshold <= 0.0 or self.threshold >= 1.0:
            max_iter = 1
        else:
            log_arg = max(self.threshold * (1.0 - params.discount), 1e-8)
            max_iter = max(
                1,
                math.ceil(math.log(1.0 / log_arg) / max(1e-6, (1.0 - params.discount))),
            )
        self._max_value_iterations = max_iter
        self._threshold_array = jnp.asarray(self.threshold, dtype=jnp.float32)

        super().__init__(
            params=params,
            seed=seed,
            policy=policy,
        )

    def _default_policy(self) -> ActionSelectionPolicy:
        return GreedyPolicy()

    def _initial_state(self, key: jax.Array) -> RMaxState:
        q_values = jnp.full(
            (self.num_states, self.num_actions),
            self._optimistic_value,
            dtype=jnp.float32,
        )

        sa_counts = jnp.zeros_like(q_values, dtype=jnp.int32)
        reward_sums = jnp.zeros_like(q_values, dtype=jnp.float32)
        trans_counts = jnp.zeros(
            (self.num_states, self.num_actions, self.num_states),
            dtype=jnp.int32,
        )

        return RMaxState(
            q_values=q_values,
            rng=key,
            sa_counts=sa_counts,
            reward_sums=reward_sums,
            trans_counts=trans_counts,
        )

    def _policy_extras(
        self, state: RMaxState, obs: jax.Array
    ) -> Dict[str, jnp.ndarray]:
        counts_row = state.sa_counts[obs].astype(jnp.float32)
        total = jnp.sum(counts_row) + 1.0
        return {"counts": counts_row, "total": total}

    def update(
        self,
        agent_state: RMaxState,
        obs: jax.Array,
        action: jax.Array,
        reward: jax.Array,
        next_obs: jax.Array,
        terminated: jax.Array | bool = False,  # Unused but kept for API parity.
    ) -> Tuple[RMaxState, UpdateResult]:
        del terminated

        obs_idx = jnp.asarray(obs, dtype=jnp.int32)
        action_idx = jnp.asarray(action, dtype=jnp.int32)
        next_obs_idx = jnp.asarray(next_obs, dtype=jnp.int32)
        reward_val = jnp.asarray(reward, dtype=jnp.float32)

        count_before = agent_state.sa_counts[obs_idx, action_idx]
        known_before = count_before >= self.m

        def _add_transition(curr_state: RMaxState) -> RMaxState:
            sa_counts = curr_state.sa_counts.at[obs_idx, action_idx].add(1)
            reward_sums = curr_state.reward_sums.at[obs_idx, action_idx].add(
                reward_val
            )
            trans_counts = curr_state.trans_counts.at[
                obs_idx, action_idx, next_obs_idx
            ].add(1)
            return curr_state.replace(
                sa_counts=sa_counts,
                reward_sums=reward_sums,
                trans_counts=trans_counts,
            )

        agent_state = jax.lax.cond(known_before, lambda s: s, _add_transition, agent_state)

        count_after = agent_state.sa_counts[obs_idx, action_idx]
        became_known = jnp.logical_and(count_after >= self.m, jnp.logical_not(known_before))

        agent_state = jax.lax.cond(
            became_known, self._value_iteration, lambda s: s, agent_state
        )
        return agent_state, UpdateResult()

    def _value_iteration(self, agent_state: RMaxState) -> RMaxState:
        counts = agent_state.sa_counts.astype(jnp.float32)
        known_mask = agent_state.sa_counts >= self.m

        safe_counts = jnp.maximum(counts, 1.0)
        reward_estimates = agent_state.reward_sums / safe_counts
        transitions = agent_state.trans_counts.astype(jnp.float32) / safe_counts[..., None]

        def cond_fun(carry):
            _, delta, iteration = carry
            continue_delta = delta >= self._threshold_array
            continue_iter = iteration < self._max_value_iterations
            return jnp.logical_and(continue_delta, continue_iter)

        def body_fun(carry):
            q_values, _, iteration = carry
            V = jnp.max(q_values, axis=1)
            expected = reward_estimates + self.discount * jnp.sum(
                transitions * V[None, None, :], axis=-1
            )
            updated_q = jnp.where(known_mask, expected, self._optimistic_value)
            delta = jnp.max(jnp.abs(updated_q - q_values))
            return updated_q, delta, iteration + 1

        init_carry = (
            agent_state.q_values,
            jnp.asarray(jnp.inf, dtype=jnp.float32),
            jnp.array(0, dtype=jnp.int32),
        )
        new_q_values, _, _ = jax.lax.while_loop(cond_fun, body_fun, init_carry)
        return agent_state.replace(q_values=new_q_values)

    def train(self, state: AgentState) -> AgentState:
        return state
    
    def eval(self, state: AgentState) -> AgentState:
        return state