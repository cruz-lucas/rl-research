"""Decision-time planning agent with model-based n-step rollouts."""

from __future__ import annotations

from typing import Dict, Tuple

import jax
from jax import lax
from flax import struct
import jax.numpy as jnp

from rl_research.agents.base import AgentState, TabularAgent, UpdateResult, AgentParams
from rl_research.policies import ActionSelectionPolicy, GreedyPolicy


@struct.dataclass
class DTNStepPState(AgentState):
    """Mutable state tracked by the DTNStep planner."""

    eval: bool = False


@struct.dataclass
class DTNStepPParams(AgentParams):
    """Static configuration for the n-step decision-time planning agent."""

    learning_rate: float
    initial_value: float
    horizon: int
    dynamics_model: jax.Array


class DTNStepPAgent(TabularAgent[DTNStepPState, DTNStepPParams]):
    """Decision-time planner using model rollouts and n-step returns."""

    def __init__(
        self,
        params: DTNStepPParams,
        seed: int | None = None,
        policy: ActionSelectionPolicy | None = None,
    ) -> None:
        self.learning_rate = params.learning_rate
        self.initial_value = params.initial_value

        self.horizon = int(params.horizon)
        if self.horizon <= 0:
            raise ValueError("`horizon` must be a positive integer.")

        self._discount = jnp.asarray(params.discount, dtype=jnp.float32)

        if params.dynamics_model is None:
            raise ValueError("`dynamics_model` must be provided for DTNStepPAgent.")

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

    def _policy_extras(self, state: DTNStepPState, obs: int) -> Dict[str, jnp.ndarray]:
        return dict({})

    def select_action(
        self, state: DTNStepPState, obs: jax.Array
    ) -> Tuple[jax.Array, DTNStepPState, Dict[str, float]]:
        """Selects an action using the configured policy."""
        obs_idx = jnp.asarray(obs, dtype=jnp.int32)
        plan_values = self._plan_targets(state.q_values, obs_idx)
        q_values = state.q_values[obs_idx]

        extras = self._policy_extras(state, obs_idx)

        action_values = state.eval * q_values + (1 - state.eval) * plan_values
        action, new_rng, info = self._policy.select(state.rng, action_values, extras)
        info = {**info, "plan_values": plan_values, "q_row": state.q_values[obs_idx]}
        state = state.replace(rng=new_rng)
        return action, state, info

    def _initial_state(self, key: jax.Array) -> DTNStepPState:
        q_values = jnp.full(
            (self.num_states, self.num_actions),
            self.initial_value,
            dtype=jnp.float32,
        )
        self._td_errors.clear()

        return DTNStepPState(
            q_values=q_values,
            rng=key,
            eval=False,
        )

    def update(
        self,
        agent_state: DTNStepPState,
        obs: int,
        action: int,
        reward: float,
        next_obs: int,
        terminated: bool = False,
    ) -> Tuple[DTNStepPState, UpdateResult]:
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

        self._td_errors.append(td_error)
        return (
            DTNStepPState(
                q_values=q_values,
                rng=agent_state.rng,
                eval=agent_state.eval,
            ),
            UpdateResult(td_error=td_error),
        )

    @property
    def td_errors(self) -> list[float]:
        return self._td_errors

    def train(self, state: DTNStepPState) -> AgentState:
        return state.replace(eval=False)

    def eval(self, state: DTNStepPState) -> AgentState:
        return state.replace(eval=True)

    @staticmethod
    def _greedy_action(q_values: jax.Array, state_idx: jax.Array) -> jax.Array:
        """Return the greedy action index for `state_idx`."""
        return jnp.asarray(jnp.argmax(q_values[state_idx]), dtype=jnp.int32)

    def _plan_targets(self, q_values: jax.Array, obs_idx: jax.Array) -> jax.Array:
        """Compute n-step return targets for all actions from `obs_idx`."""
        actions = jnp.arange(self.num_actions, dtype=jnp.int32)

        def rollout_target(action_idx: jax.Array) -> jax.Array:
            return self._n_step_return(q_values, obs_idx, action_idx)

        return jax.vmap(rollout_target)(actions)

    def _n_step_return(
        self,
        q_values: jax.Array,
        start_state: jax.Array,
        first_action: jax.Array,
    ) -> jax.Array:
        """Roll out `first_action` and compute the n-step return for the root."""

        def step(carry, _):
            state_idx, action_idx = carry
            reward = self._model_rewards[state_idx, action_idx]
            next_state = self._model_next_obs[state_idx, action_idx]
            next_action = self._greedy_action(q_values, next_state)
            return (next_state, next_action), (reward, next_state)

        init_carry = (start_state, first_action)
        (final_state, _), (rewards, _) = lax.scan(
            step,
            init_carry,
            xs=None,
            length=self.horizon,
        )

        steps = jnp.arange(self.horizon, dtype=jnp.float32)
        discounts = jnp.power(self._discount, steps)
        return_sum = jnp.sum(rewards * discounts)

        bootstrap_discount = jnp.power(
            self._discount, jnp.asarray(self.horizon, dtype=jnp.float32)
        )
        bootstrap_value = bootstrap_discount * jnp.max(q_values[final_state])

        return return_sum + bootstrap_value
