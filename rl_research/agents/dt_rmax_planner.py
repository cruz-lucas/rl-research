"""Decision-time planning agent with optimistic R-MAX rollouts."""

from __future__ import annotations

from typing import Dict, Tuple

import jax
from jax import lax
from flax import struct
import jax.numpy as jnp

from rl_research.agents.base import AgentState, AgentParams, TabularAgent, UpdateResult
from rl_research.policies import ActionSelectionPolicy, GreedyPolicy


@struct.dataclass
class DTRMaxNStepState(AgentState):
    """Mutable state tracked by the R-MAX decision-time planner."""

    sa_counts: jax.Array
    eval: bool = False


@struct.dataclass
class DTRMaxNStepParams(AgentParams):
    """Static configuration for the R-MAX n-step decision-time planning agent."""

    learning_rate: float
    initial_value: float
    horizon: int
    m: int
    r_max: float
    dynamics_model: jax.Array


class DTRMaxNStepAgent(TabularAgent[DTRMaxNStepState, DTRMaxNStepParams]):
    """Decision-time planner using R-MAX optimistic rollouts for n-step targets."""

    def __init__(
        self,
        params: DTRMaxNStepParams,
        seed: int | None = None,
        policy: ActionSelectionPolicy | None = None,
    ) -> None:
        self.learning_rate = params.learning_rate
        self.initial_value = params.initial_value

        self.horizon = int(params.horizon)
        if self.horizon <= 0:
            raise ValueError("`horizon` must be a positive integer.")

        self._discount = jnp.asarray(params.discount, dtype=jnp.float32)

        self.m = int(params.m)
        if self.m <= 0:
            raise ValueError("`m` must be a positive integer.")
        self._m_threshold = jnp.asarray(self.m, dtype=jnp.int32)

        self.r_max = float(params.r_max)
        self._optimistic_value = jnp.asarray(
            self.r_max / (1.0 - params.discount), dtype=jnp.float32
        )

        if params.dynamics_model is None:
            raise ValueError("`dynamics_model` must be provided for DTRMaxNStepAgent.")

        dynamics = jnp.asarray(params.dynamics_model)
        if (
            dynamics.ndim != 3
            or dynamics.shape[0] != params.num_states
            or dynamics.shape[1] != params.num_actions
            or dynamics.shape[2] < 2
        ):
            raise ValueError(
                "Expected `dynamics_model` with shape "
                f"(num_states={params.num_states}, num_actions={params.num_actions}, >=2). "
                f"Received shape {dynamics.shape}."
            )

        next_obs = dynamics[..., 0]
        rewards = dynamics[..., 1]

        max_state = jnp.asarray(params.num_states - 1, dtype=jnp.int32)
        min_state = jnp.asarray(0, dtype=jnp.int32)
        rounded_next_obs = jnp.rint(next_obs).astype(jnp.int32)
        self._model_next_obs = jnp.clip(rounded_next_obs, min_state, max_state)
        self._model_rewards = jnp.asarray(rewards, dtype=jnp.float32)

        self._td_errors: list[float] = []

        super().__init__(
            params=params,
            seed=seed,
            policy=policy,
        )

    def _default_policy(self) -> ActionSelectionPolicy:
        return GreedyPolicy()

    def _policy_extras(
        self, state: DTRMaxNStepState, obs: int
    ) -> Dict[str, jnp.ndarray]:
        counts_row = state.sa_counts[obs].astype(jnp.float32)
        total = jnp.sum(counts_row) + 1.0
        return {"counts": counts_row, "total": total}

    def select_action(
        self, state: DTRMaxNStepState, obs: jax.Array
    ) -> Tuple[jax.Array, DTRMaxNStepState, Dict[str, float]]:
        """Selects an action using the configured policy."""
        obs_idx = jnp.asarray(obs, dtype=jnp.int32)
        plan_values = self._plan_targets(state.q_values, state.sa_counts, obs_idx)
        q_values = state.q_values[obs_idx]

        extras = self._policy_extras(state, obs_idx)

        action_values = state.eval * q_values + (1 - state.eval) * plan_values
        action, new_rng, info = self._policy.select(state.rng, action_values, extras)
        info = {
            **info,
            "plan_values": plan_values,
            "q_row": state.q_values[obs_idx],
        }
        state = state.replace(rng=new_rng)
        return action, state, info

    def _initial_state(self, key: jax.Array) -> DTRMaxNStepState:
        q_values = jnp.full(
            (self.num_states, self.num_actions),
            self.initial_value,
            dtype=jnp.float32,
        )
        self._td_errors.clear()

        sa_counts = jnp.zeros_like(q_values, dtype=jnp.int32)
        return DTRMaxNStepState(
            q_values=q_values,
            rng=key,
            sa_counts=sa_counts,
            eval=False,
        )

    def update(
        self,
        agent_state: DTRMaxNStepState,
        obs: int,
        action: int,
        reward: float,
        next_obs: int,
        terminated: bool = False,
    ) -> Tuple[DTRMaxNStepState, UpdateResult]:
        obs_idx = jnp.asarray(obs, dtype=jnp.int32)
        action_idx = jnp.asarray(action, dtype=jnp.int32)
        next_obs_idx = jnp.asarray(next_obs, dtype=jnp.int32)
        reward_val = jnp.asarray(reward, dtype=jnp.float32)

        terminated_mask = jnp.asarray(terminated, dtype=jnp.float32)

        next_value = jnp.max(agent_state.q_values[next_obs_idx])
        target = reward_val + self.discount * (1.0 - terminated_mask) * next_value
        td_error = target - agent_state.q_values[obs_idx, action_idx]

        q_values = agent_state.q_values.at[obs_idx, action_idx].add(
            self.learning_rate * td_error
        )
        sa_counts = agent_state.sa_counts.at[obs_idx, action_idx].add(1)

        self._td_errors.append(td_error)
        return (
            DTRMaxNStepState(
                q_values=q_values,
                rng=agent_state.rng,
                sa_counts=sa_counts,
                eval=agent_state.eval,
            ),
            UpdateResult(td_error=td_error),
        )

    @property
    def td_errors(self) -> list[float]:
        return self._td_errors

    def train(self, state: DTRMaxNStepState) -> AgentState:
        return state.replace(eval=False)

    def eval(self, state: DTRMaxNStepState) -> AgentState:
        return state.replace(eval=True)

    @staticmethod
    def _greedy_action(q_values: jax.Array, state_idx: jax.Array) -> jax.Array:
        """Return the greedy action index for `state_idx`."""
        return jnp.asarray(jnp.argmax(q_values[state_idx]), dtype=jnp.int32)

    def _plan_targets(
        self, q_values: jax.Array, sa_counts: jax.Array, obs_idx: jax.Array
    ) -> jax.Array:
        """Compute n-step return targets under the optimistic R-MAX model."""
        actions = jnp.arange(self.num_actions, dtype=jnp.int32)

        def rollout_target(action_idx: jax.Array) -> jax.Array:
            return self._rmax_n_step_return(
                q_values, sa_counts, obs_idx, action_idx
            )

        return jax.vmap(rollout_target)(actions)

    def _rmax_n_step_return(
        self,
        q_values: jax.Array,
        sa_counts: jax.Array,
        start_state: jax.Array,
        first_action: jax.Array,
    ) -> jax.Array:
        """Roll out `first_action` and compute the n-step return with R-MAX optimism."""

        def body(step: jax.Array, carry):
            del step
            state_idx, action_idx, done, acc_return, discount = carry

            visited = sa_counts[state_idx, action_idx]
            known = visited >= self._m_threshold

            optimistic = acc_return + discount * self._optimistic_value

            reward = self._model_rewards[state_idx, action_idx]
            next_state = self._model_next_obs[state_idx, action_idx]
            next_action = self._greedy_action(q_values, next_state)

            acc_known = acc_return + discount * reward
            discount_known = discount * self._discount

            updated_state = jax.lax.select(known, next_state, state_idx)
            updated_action = jax.lax.select(known, next_action, action_idx)
            updated_discount = jax.lax.select(known, discount_known, discount)
            updated_acc = jax.lax.select(known, acc_known, optimistic)

            updated_state = jax.lax.select(done, state_idx, updated_state)
            updated_action = jax.lax.select(done, action_idx, updated_action)
            updated_discount = jax.lax.select(done, discount, updated_discount)
            updated_acc = jax.lax.select(done, acc_return, updated_acc)

            updated_done = jnp.logical_or(done, jnp.logical_not(known))

            return (
                updated_state,
                updated_action,
                updated_done,
                updated_acc,
                updated_discount,
            )

        init_carry = (
            start_state,
            first_action,
            jnp.asarray(False),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(1.0, dtype=jnp.float32),
        )
        final_state, _, done, acc_return, discount = lax.fori_loop(
            0, self.horizon, body, init_carry
        )

        bootstrap = jnp.where(
            done,
            jnp.asarray(0.0, dtype=jnp.float32),
            discount * jnp.max(q_values[final_state]),
        )
        return acc_return + bootstrap
