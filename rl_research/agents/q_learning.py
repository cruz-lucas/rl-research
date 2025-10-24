"""Classical tabular Q-learning implemented with JAX."""

from __future__ import annotations

from typing import Dict, Tuple

import jax
from flax import struct
import jax.numpy as jnp

from rl_research.agents.base import AgentState, TabularAgent, UpdateResult, AgentParams
from rl_research.policies import ActionSelectionPolicy, EpsilonGreedyPolicy

@struct.dataclass
class QlearningState(AgentState):
    """Container for the mutable quantities tracked by R-MAX."""

    epsilon: jax.Array


@struct.dataclass
class QlearningParams(AgentParams):
    """Static params for Rmax."""

    initial_epsilon: float
    learning_rate: float
    initial_value: float

class QLearningAgent(TabularAgent[QlearningState, QlearningParams]):
    """Standard off-policy Q-learning agent."""

    def __init__(
        self,
        params: QlearningParams,
        seed: int | None = None,
        policy: ActionSelectionPolicy | None = None,
    ) -> None:
        self.learning_rate = params.learning_rate
        self.initial_epsilon = jnp.array(params.initial_epsilon, dtype=float)
        self.initial_value = params.initial_value

        self._td_errors: list[float] = []

        super().__init__(
            params=params,
            seed=seed,
            policy=policy,
        )

    def _default_policy(self) -> ActionSelectionPolicy:
        return EpsilonGreedyPolicy()

    def _initial_state(self, key: jax.Array) -> QlearningState:
        q_values = jnp.full(
            (self.num_states, self.num_actions),
            self.initial_value,
            dtype=jnp.float32,
        )
        self._td_errors.clear()

        return QlearningState(
            q_values=q_values,
            rng=key,
            epsilon=self.initial_epsilon,
        )

    def _policy_extras(self, state: QlearningState, obs: int) -> Dict[str, jnp.ndarray]:
        return dict({
            'epsilon': state.epsilon
        })

    def update(
        self,
        agent_state: QlearningState,
        obs: int,
        action: int,
        reward: float,
        next_obs: int,
        terminated: bool = False,
    ) -> Tuple[QlearningState, UpdateResult]:
        
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
            QlearningState(
                q_values=q_values,
                rng=agent_state.rng,
                epsilon=agent_state.epsilon
            ),
            UpdateResult(td_error=td_error)
        )

    @property
    def td_errors(self) -> list[float]:
        return self._td_errors
    
    def train(self, state: QlearningState) -> AgentState:
        return QlearningState(
            q_values=state.q_values,
            rng=state.rng,
            epsilon=self.initial_epsilon,
        )
    
    def eval(self, state: QlearningState) -> AgentState:
        return QlearningState(
            q_values=state.q_values,
            rng=state.rng,
            epsilon=jnp.array(0.0, dtype=float),
        )
