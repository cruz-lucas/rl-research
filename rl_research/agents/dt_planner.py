"""Decision-time planning agent with model-based rollouts."""

from __future__ import annotations

from typing import Dict, Tuple

import jax
from jax import lax
from flax import struct
import jax.numpy as jnp

from rl_research.agents.base import AgentState, TabularAgent, UpdateResult, AgentParams
from rl_research.policies import ActionSelectionPolicy, GreedyPolicy

@struct.dataclass
class DTPState(AgentState):
    """Mutable state tracked by the DTP agent."""
    pass


@struct.dataclass
class DTPParams(AgentParams):
    """Static configuration for the decision-time planning agent."""

    learning_rate: float
    initial_value: float
    horizon: int
    lambda_: float
    dynamics_model: jax.Array

class DTPAgent(TabularAgent[DTPState, DTPParams]):
    """Decision-time planner using model rollouts and lambda-returns."""

    def __init__(
        self,
        params: DTPParams,
        seed: int | None = None,
        policy: ActionSelectionPolicy | None = None,
    ) -> None:
        self.learning_rate = params.learning_rate
        self.initial_value = params.initial_value
        self.horizon = int(params.horizon)
        if self.horizon <= 0:
            raise ValueError("`horizon` must be a positive integer.")

        lambda_val = float(params.lambda_)
        if not 0.0 <= lambda_val <= 1.0:
            raise ValueError("`lambda_` must lie within [0, 1].")
        self._lambda = jnp.asarray(lambda_val, dtype=jnp.float32)
        self._discount = jnp.asarray(params.discount, dtype=jnp.float32)

        if params.dynamics_model is None:
            raise ValueError("`dynamics_model` must be provided for DTPAgent.")
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

    def _policy_extras(self, state: DTPState, obs: int) -> Dict[str, jnp.ndarray]:
        return dict({})
    
    def select_action(
        self, state: DTPState, obs: jax.Array
    ) -> Tuple[jax.Array, DTPState, Dict[str, float]]:
        """Selects an action using the configured policy."""
        obs_idx = jnp.asarray(obs, dtype=jnp.int32)
        plan_values = self._plan_targets(state.q_values, obs_idx)

        extras = self._policy_extras(state, obs_idx)

        action, new_rng, info = self._policy.select(state.rng, plan_values, extras)
        info = {**info, "plan_values": plan_values, "q_row": state.q_values[obs_idx]}
        state = state.replace(rng=new_rng)
        return action, state, info

    def _initial_state(self, key: jax.Array) -> DTPState:
        q_values = jnp.full(
            (self.num_states, self.num_actions),
            self.initial_value,
            dtype=jnp.float32,
        )
        self._td_errors.clear()

        return DTPState(
            q_values=q_values,
            rng=key,
        )

    def update(
        self,
        agent_state: DTPState,
        obs: int,
        action: int,
        reward: float,
        next_obs: int,
        terminated: bool = False,
    ) -> Tuple[DTPState, UpdateResult]:
        
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
            DTPState(
                q_values=q_values,
                rng=agent_state.rng,
            ),
            UpdateResult(td_error=td_error)
        )

    @property
    def td_errors(self) -> list[float]:
        return self._td_errors
    
    def train(self, state: DTPState) -> AgentState:
        return state
    
    def eval(self, state: DTPState) -> AgentState:
        return state

    @staticmethod
    def _greedy_action(q_values: jax.Array, state_idx: jax.Array) -> jax.Array:
        """Return the greedy action index for `state_idx`."""
        return jnp.asarray(jnp.argmax(q_values[state_idx]), dtype=jnp.int32)

    def _plan_targets(self, q_values: jax.Array, obs_idx: jax.Array) -> jax.Array:
        """Compute lambda-return targets for all actions from `obs_idx`."""
        actions = jnp.arange(self.num_actions, dtype=jnp.int32)

        def rollout_target(action_idx: jax.Array) -> jax.Array:
            return self._lambda_return(q_values, obs_idx, action_idx)

        return jax.vmap(rollout_target)(actions)

    def _lambda_return(
        self,
        q_values: jax.Array,
        start_state: jax.Array,
        first_action: jax.Array,
    ) -> jax.Array:
        """Roll out `first_action` and compute the lambda return for the root."""

        def step(carry, _):
            state_idx, action_idx = carry
            reward = self._model_rewards[state_idx, action_idx]
            next_state = self._model_next_obs[state_idx, action_idx]
            next_action = self._greedy_action(q_values, next_state)
            return (next_state, next_action), (reward, next_state)

        init_carry = (start_state, first_action)
        (final_state, _), (rewards, next_states) = lax.scan(
            step,
            init_carry,
            xs=None,
            length=self.horizon,
        )

        bootstrap_value = jnp.max(q_values[final_state])

        def backward(i, ret):
            idx = self.horizon - 1 - i
            reward = rewards[idx]
            next_state = next_states[idx]
            next_value = jnp.max(q_values[next_state])
            blended = (1.0 - self._lambda) * next_value + self._lambda * ret
            return reward + self._discount * blended

        return lax.fori_loop(0, self.horizon, backward, bootstrap_value)
