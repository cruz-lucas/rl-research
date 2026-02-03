from typing import Literal

import gin
import jax
import jax.numpy as jnp
from flax import struct

from rl_research.buffers import Transition
from rl_research.policies import (
    _select_epsilon_greedy,
    _select_greedy,
    _select_random,
    _select_ucb,
)


class QLearningState(struct.PyTreeNode):
    """State for Q-learning agent."""

    q_table: jnp.ndarray
    visit_counts: jnp.ndarray
    eps_greedy_epsilon: float
    step: int


@gin.configurable
class QLearningAgent:
    """Q-Learning agent with epsilon-greedy, random, or UCB policies."""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        discount: float,
        initial_epsilon: float,
        step_size: float,
        initial_q_value: float,
        ucb_c: float = 0.0,
        policy: Literal["epsilon_greedy", "random", "ucb"] = "random",
        use_scheduler: bool = False,
        anneal_steps: int = 1_000_000,
        final_epsilon: float = 0.01,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.discount = discount
        self.step_size = step_size
        self.initial_q_value = initial_q_value

        self.initial_epsilon = initial_epsilon
        self.ucb_c = ucb_c
        self.policy = policy
        self.anneal_steps = anneal_steps
        self.final_epsilon = final_epsilon
        self.use_scheduler = use_scheduler

    def initial_state(self) -> QLearningState:
        """Initialize Q-table and visit counts."""
        return QLearningState(
            q_table=jnp.full((self.num_states, self.num_actions), self.initial_q_value),
            visit_counts=jnp.zeros((self.num_states, self.num_actions)),
            eps_greedy_epsilon=self.initial_epsilon,
            step=0,
        )

    def select_action(
        self, state: QLearningState, obs: jnp.ndarray, key: jax.Array, is_training: bool
    ) -> jnp.ndarray:
        """Select action based on policy."""
        q_values = state.q_table[obs]

        def eval_action():
            return _select_greedy(q_values, key)

        def train_action():
            def eps_greedy():
                return _select_epsilon_greedy(q_values, state.eps_greedy_epsilon, key)

            def ucb():
                visit_counts = state.visit_counts[obs]
                total_visits = jnp.sum(state.visit_counts[obs])
                return _select_ucb(
                    q_values, visit_counts, total_visits, self.ucb_c, key
                )

            def random():
                return _select_random(q_values, key)

            return jax.lax.cond(
                self.policy == "epsilon_greedy",
                eps_greedy,
                lambda: jax.lax.cond(self.policy == "ucb", ucb, random),
            )

        return jax.lax.cond(is_training, train_action, eval_action)

    def update(
        self,
        state: QLearningState,
        batch: Transition,
    ) -> tuple[QLearningState, jax.Array]:
        """Update Q-table."""
        s = batch.observation.astype(jnp.int32)
        a = batch.action.astype(jnp.int32)
        r = batch.reward
        s_next = batch.next_observation.astype(jnp.int32)
        terminal = batch.terminal.astype(jnp.bool)

        q_current = state.q_table[s, a]
        q_next_max = jnp.max(state.q_table[s_next])
        target = r + self.discount * q_next_max * (
            1.0 - terminal.astype(jnp.float32)
        )
        td_error = target - q_current
        new_q = q_current + self.step_size * td_error

        next_state = state.replace(
            q_table=state.q_table.at[s, a].set(new_q),
            visit_counts=state.visit_counts.at[s, a].add(1),
            step=state.step + 1,
        )

        frac = jnp.where(self.use_scheduler, jnp.clip(next_state.step / self.anneal_steps, 0.0, 1.0), 0.0)
        new_eps = self.initial_epsilon + frac * (self.final_epsilon - self.initial_epsilon)
        next_state = next_state.replace(eps_greedy_epsilon=new_eps)

        return next_state, td_error

    def bootstrap_value(
        self, state: QLearningState, next_observation: jnp.ndarray
    ) -> jax.Array:
        s_next = next_observation.astype(jnp.int32).squeeze()
        return jnp.max(state.q_table[s_next])
