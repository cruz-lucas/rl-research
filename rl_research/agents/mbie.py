"""Model-Based Interval Estimation with Exploration Bonus (MBIE-EB)."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax import struct

from rl_research.agents.base import AgentState, TabularAgent, UpdateResult, AgentParams
from rl_research.policies import ActionSelectionPolicy, GreedyPolicy


@struct.dataclass
class MBIEAgentState(AgentState):
    """State container for the MBIE agent."""

    sa_counts: jax.Array
    reward_sums: jax.Array
    trans_counts: jax.Array


@struct.dataclass
class MBIEParams(AgentParams):
    """Static params for Rmax."""

    threshold: float
    r_max: float
    epsilon_r_coeff: float      # A
    epsilon_t_coeff: float      # B
    exploration_coeff: float    # C
    m: int | None = None
    use_exploration_bonus: bool = True


class MBIEAgent(TabularAgent[MBIEAgentState, MBIEParams]):
    """Implementation of MBIE and MBIE-EB with pure JAX updates."""

    def __init__(
        self,
        params: MBIEParams,
        seed: int | None = None,
        policy: ActionSelectionPolicy | None = None,
    ) -> None:
        self.r_max = float(params.r_max)
        self.discount = float(params.discount)
        self.epsilon_r_coeff = float(params.epsilon_r_coeff)
        self.epsilon_t_coeff = float(params.epsilon_t_coeff)
        self.exploration_coeff = float(params.exploration_coeff)
        self.threshold = float(params.threshold)
        self.m_limit = math.inf if params.m is None else int(params.m)
        self.use_exploration_bonus = bool(params.use_exploration_bonus)

        self.beta = self.exploration_coeff * self.r_max
        self._optimistic_value = self.r_max / (1.0 - params.discount)

        if self.threshold <= 0.0 or self.threshold >= 1.0:
            max_iter = 1
        else:
            log_arg = max(self.threshold * (1.0 - self.discount), 1e-8)
            max_iter = max(
                1,
                math.ceil(math.log(1.0 / log_arg) / max(1e-6, (1.0 - self.discount))),
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

    def _initial_state(self, key: jax.Array) -> MBIEAgentState:
        q_values = jnp.full(
            (self.num_states, self.num_actions),
            self._optimistic_value,
            dtype=jnp.float32,
        )
        sa_counts = jnp.zeros((self.num_states, self.num_actions), dtype=jnp.int32)
        reward_sums = jnp.zeros_like(q_values)
        trans_counts = jnp.zeros(
            (self.num_states, self.num_actions, self.num_states),
            dtype=jnp.int32,
        )
        return MBIEAgentState(
            q_values=q_values,
            rng=key,
            sa_counts=sa_counts,
            reward_sums=reward_sums,
            trans_counts=trans_counts,
        )

    def _policy_extras(
        self, state: MBIEAgentState, obs: jax.Array
    ) -> Dict[str, jnp.ndarray]:
        counts_row = state.sa_counts[obs].astype(jnp.float32)
        total = jnp.sum(counts_row) + 1.0
        return {"counts": counts_row, "total": total}

    def update(
        self,
        agent_state: MBIEAgentState,
        obs: jax.Array,
        action: jax.Array,
        reward: jax.Array,
        next_obs: jax.Array,
        terminated: jax.Array | bool = False,
    ) -> Tuple[MBIEAgentState, UpdateResult]:
        del terminated

        obs_idx = jnp.asarray(obs, dtype=jnp.int32)
        action_idx = jnp.asarray(action, dtype=jnp.int32)
        next_obs_idx = jnp.asarray(next_obs, dtype=jnp.int32)
        reward_val = jnp.asarray(reward, dtype=jnp.float32)

        count_before = agent_state.sa_counts[obs_idx, action_idx]
        should_update = count_before < self.m_limit

        def _perform_update(curr_state: MBIEAgentState) -> MBIEAgentState:
            sa_counts = curr_state.sa_counts.at[obs_idx, action_idx].add(1)
            reward_sums = curr_state.reward_sums.at[obs_idx, action_idx].add(
                reward_val
            )
            trans_counts = curr_state.trans_counts.at[
                obs_idx, action_idx, next_obs_idx
            ].add(1)

            curr_state = curr_state.replace(
                sa_counts=sa_counts,
                reward_sums=reward_sums,
                trans_counts=trans_counts,
            )
            return self._value_iteration(curr_state)

        agent_state = jax.lax.cond(should_update, _perform_update, lambda s: s, agent_state)
        return agent_state, UpdateResult()

    def _value_iteration(self, state: MBIEAgentState) -> MBIEAgentState:
        counts = state.sa_counts.astype(jnp.float32)
        nonzero_mask = counts > 0.0
        safe_counts = jnp.maximum(counts, 1.0)

        reward_estimates = jnp.where(
            nonzero_mask,
            state.reward_sums / safe_counts,
            self.r_max,
        )
        reward_conf = jnp.where(
            nonzero_mask,
            self.epsilon_r_coeff * self.r_max / jnp.sqrt(safe_counts),
            jnp.inf,
        )

        transitions = state.trans_counts.astype(jnp.float32) / safe_counts[..., None]
        transition_conf = jnp.where(
            nonzero_mask,
            self.epsilon_t_coeff / jnp.sqrt(safe_counts),
            0.0,
        )

        bonuses = jnp.where(
            nonzero_mask,
            self.beta / jnp.sqrt(safe_counts),
            0.0,
        )

        def cond_fun(carry):
            _, delta, iteration = carry
            continue_delta = delta >= self._threshold_array
            continue_iter = iteration < self._max_value_iterations
            return jnp.logical_and(continue_delta, continue_iter)

        def body_fun(carry):
            q_values, _, iteration = carry
            V = jnp.max(q_values, axis=1)
            expected = jnp.sum(
                transitions * V[None, None, :], axis=-1
            )

            if self.use_exploration_bonus:
                target_known = reward_estimates + self.discount * expected + bonuses
            else:
                opt_vals = self._optimistic_transition_values(
                    transitions, V, transition_conf
                )
                target_known = (
                    reward_estimates
                    + self.discount * opt_vals
                    + reward_conf
                )

            optimistic_table = jnp.full_like(q_values, self._optimistic_value)
            updated_q = jnp.where(nonzero_mask, target_known, optimistic_table)
            delta = jnp.max(jnp.abs(updated_q - q_values))
            return updated_q, delta, iteration + 1

        init_carry = (
            state.q_values,
            jnp.asarray(jnp.inf, dtype=jnp.float32),
            jnp.array(0, dtype=jnp.int32),
        )
        new_q_values, _, _ = jax.lax.while_loop(cond_fun, body_fun, init_carry)
        return state.replace(q_values=new_q_values)

    def _optimistic_transition_values(
        self,
        transitions: jax.Array,
        values: jax.Array,
        epsilon_t: jax.Array,
    ) -> jax.Array:
        flat_trans = transitions.reshape(-1, self.num_states)
        flat_eps = epsilon_t.reshape(-1)

        def compute(prob, eps):
            return self._optimistic_transition_value(prob, values, eps)

        results = jax.vmap(compute)(flat_trans, flat_eps)
        return results.reshape(self.num_states, self.num_actions)

    def _optimistic_transition_value(
        self,
        probs: jax.Array,
        values: jax.Array,
        epsilon_t: jax.Array,
    ) -> jax.Array:
        probs = jnp.asarray(probs, dtype=jnp.float32)
        epsilon = jnp.asarray(epsilon_t, dtype=jnp.float32)

        extra = epsilon / 2.0
        s_star = jnp.argmax(values)

        T_tilde = probs.at[s_star].add(extra)
        remaining = extra
        order = jnp.argsort(values)

        def body(i, carry):
            T_curr, remaining_curr = carry
            idx = order[i]

            def remove(payload):
                T_local, rem = payload
                removable = jnp.minimum(T_local[idx], rem)
                T_local = T_local.at[idx].add(-removable)
                return T_local, rem - removable

            condition = jnp.logical_and(remaining_curr > 0.0, idx != s_star)
            return jax.lax.cond(
                condition,
                remove,
                lambda payload: payload,
                operand=(T_curr, remaining_curr),
            )

        T_tilde, _ = jax.lax.fori_loop(0, self.num_states, body, (T_tilde, remaining))
        T_tilde = jnp.clip(T_tilde, a_min=0.0)
        denom = jnp.maximum(jnp.sum(T_tilde), 1e-8)
        T_tilde = T_tilde / denom
        return jnp.dot(T_tilde, values)

    def train(self, state: AgentState) -> AgentState:
        return state
    
    def eval(self, state: AgentState) -> AgentState:
        return state