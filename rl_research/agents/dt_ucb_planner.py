"""Decision-time planning agent with UCB-style exploration bonuses."""

from __future__ import annotations

from typing import Dict, Tuple

import jax
from flax import struct
from jax import lax
import jax.numpy as jnp

from rl_research.agents.base import AgentParams, AgentState, TabularAgent, UpdateResult
from rl_research.policies import ActionSelectionPolicy, GreedyPolicy


@struct.dataclass
class DTUCBState(AgentState):
    """Mutable quantities tracked by the UCB planner."""

    visit_counts: jax.Array
    timestep: jax.Array


@struct.dataclass
class DTUCBParams(AgentParams):
    """Static configuration for the decision-time UCB planner."""

    learning_rate: float
    initial_value: float
    horizon: int
    beta: float
    dynamics_model: jax.Array
    use_time_bonus: bool = False


class DTUCBPlanner(TabularAgent[DTUCBState, DTUCBParams]):
    """Decision-time planner that accumulates UCB-style exploration bonuses."""

    def __init__(
        self,
        params: DTUCBParams,
        seed: int | None = None,
        policy: ActionSelectionPolicy | None = None,
    ) -> None:
        self.learning_rate = params.learning_rate
        self.initial_value = params.initial_value
        self.horizon = int(params.horizon)
        if self.horizon <= 0:
            raise ValueError("`horizon` must be a positive integer.")

        beta = float(params.beta)
        if beta < 0.0:
            raise ValueError("`beta` must be non-negative.")
        self._beta = jnp.asarray(beta, dtype=jnp.float32)
        self._discount = jnp.asarray(params.discount, dtype=jnp.float32)
        self._use_time_bonus = bool(params.use_time_bonus)

        if params.dynamics_model is None:
            raise ValueError("`dynamics_model` must be provided for DTUCBPlanner.")
        dynamics = jnp.asarray(params.dynamics_model)
        if (
            dynamics.ndim != 3
            or dynamics.shape[0] != params.num_states
            or dynamics.shape[1] != params.num_actions
            or dynamics.shape[2] < 1
        ):
            raise ValueError(
                "Expected `dynamics_model` with shape "
                f"(num_states={params.num_states}, num_actions={params.num_actions}, >=1). "
                f"Received shape {dynamics.shape}."
            )

        next_obs = dynamics[..., 0]

        max_state = jnp.asarray(params.num_states - 1, dtype=jnp.int32)
        min_state = jnp.asarray(0, dtype=jnp.int32)
        rounded_next_obs = jnp.rint(next_obs).astype(jnp.int32)
        self._model_next_obs = jnp.clip(rounded_next_obs, min_state, max_state)

        self._td_errors: list[float] = []

        super().__init__(
            params=params,
            seed=seed,
            policy=policy,
        )

    def _default_policy(self) -> ActionSelectionPolicy:
        return GreedyPolicy()

    def _policy_extras(self, state: DTUCBState, obs: int) -> Dict[str, jnp.ndarray]:
        return dict({})

    def select_action(
        self, state: DTUCBState, obs: jax.Array
    ) -> Tuple[jax.Array, DTUCBState, Dict[str, float]]:
        obs_idx = jnp.asarray(obs, dtype=jnp.int32)
        bonuses = self._plan_bonuses(
            state.q_values, state.visit_counts, obs_idx, state.timestep
        )
        plan_values = state.q_values[obs_idx] + self._beta * bonuses

        extras = self._policy_extras(state, obs_idx)

        action, new_rng, info = self._policy.select(state.rng, plan_values, extras)
        info = {
            **info,
            "plan_values": plan_values,
            "q_row": state.q_values[obs_idx],
            "bonuses": bonuses,
        }
        state = state.replace(rng=new_rng)
        return action, state, info

    def _initial_state(self, key: jax.Array) -> DTUCBState:
        q_values = jnp.full(
            (self.num_states, self.num_actions),
            self.initial_value,
            dtype=jnp.float32,
        )
        visit_counts = jnp.zeros_like(q_values)
        timestep = jnp.asarray(1.0, dtype=jnp.float32)
        self._td_errors.clear()

        return DTUCBState(
            q_values=q_values,
            visit_counts=visit_counts,
            timestep=timestep,
            rng=key,
        )

    def update(
        self,
        agent_state: DTUCBState,
        obs: int,
        action: int,
        reward: float,
        next_obs: int,
        terminated: bool = False,
    ) -> Tuple[DTUCBState, UpdateResult]:
        obs_idx = jnp.asarray(obs, dtype=jnp.int32)
        action_idx = jnp.asarray(action, dtype=jnp.int32)
        next_obs_idx = jnp.asarray(next_obs, dtype=jnp.int32)
        reward_val = jnp.asarray(reward, dtype=jnp.float32)

        terminated_mask = jnp.asarray(terminated, dtype=jnp.float32)

        next_value = jnp.max(agent_state.q_values[next_obs_idx])
        target = reward_val + self._discount * (1.0 - terminated_mask) * next_value
        td_error = target - agent_state.q_values[obs_idx, action_idx]

        q_values = agent_state.q_values.at[obs_idx, action_idx].add(
            self.learning_rate * td_error
        )
        visit_counts = agent_state.visit_counts.at[obs_idx, action_idx].add(1.0)
        timestep = agent_state.timestep + jnp.asarray(1.0, dtype=jnp.float32)

        self._td_errors.append(td_error)
        return (
            DTUCBState(
                q_values=q_values,
                visit_counts=visit_counts,
                timestep=timestep,
                rng=agent_state.rng,
            ),
            UpdateResult(td_error=td_error),
        )

    @property
    def td_errors(self) -> list[float]:
        return self._td_errors

    def train(self, state: DTUCBState) -> AgentState:
        return state

    def eval(self, state: DTUCBState) -> AgentState:
        return state

    @staticmethod
    def _greedy_action(q_values: jax.Array, state_idx: jax.Array) -> jax.Array:
        return jnp.asarray(jnp.argmax(q_values[state_idx]), dtype=jnp.int32)

    def _plan_bonuses(
        self,
        q_values: jax.Array,
        visit_counts: jax.Array,
        obs_idx: jax.Array,
        timestep: jax.Array,
    ) -> jax.Array:
        actions = jnp.arange(self.num_actions, dtype=jnp.int32)

        def rollout_bonus(action_idx: jax.Array) -> jax.Array:
            return self._trajectory_bonus(
                q_values, visit_counts, obs_idx, action_idx, timestep
            )

        return jax.vmap(rollout_bonus)(actions)

    def _trajectory_bonus(
        self,
        q_values: jax.Array,
        visit_counts: jax.Array,
        start_state: jax.Array,
        first_action: jax.Array,
        timestep: jax.Array,
    ) -> jax.Array:
        def step(carry, _):
            state_idx, action_idx = carry
            bonus = self._bonus_value(visit_counts, state_idx, action_idx, timestep)
            next_state = self._model_next_obs[state_idx, action_idx]
            next_action = self._greedy_action(q_values, next_state)
            return (next_state, next_action), bonus

        init_carry = (start_state, first_action)
        (_, _), bonuses = lax.scan(
            step,
            init_carry,
            xs=None,
            length=self.horizon,
        )

        return jnp.sum(bonuses)

    def _bonus_value(
        self,
        visit_counts: jax.Array,
        state_idx: jax.Array,
        action_idx: jax.Array,
        timestep: jax.Array,
    ) -> jax.Array:
        counts = visit_counts[state_idx, action_idx]
        safe_counts = jnp.maximum(counts, jnp.asarray(1.0, dtype=jnp.float32))
        scale = jnp.asarray(1.0, dtype=jnp.float32)
        if self._use_time_bonus:
            min_time = jnp.asarray(2.0, dtype=jnp.float32)
            scale = jnp.log(jnp.maximum(timestep, min_time))
        return jnp.sqrt(scale / safe_counts)
