import gin
import jax
import jax.numpy as jnp
from flax import struct
from typing import Tuple
import distrax

from rl_research.buffers import Transition


class QLearningState(struct.PyTreeNode):
    """State for Q-learning agent."""

    q_table: jnp.ndarray
    visit_counts: jnp.ndarray
    step: int


@gin.configurable
class QLearningAgent:
    """Q-Learning agent with epsilon-greedy."""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        discount: float,
        step_size: float,
        initial_q_value: float = 0.0,
        use_scheduler: bool = False,
        anneal_steps: int = 1_000_000,
        initial_epsilon: float = 1.0,
        final_epsilon: float = 0.01,
        reward_bonus: float = 0.0,
        stability_coef: float = 0.02,
        num_epochs: int = 1
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.discount = discount
        self.step_size = step_size
        self.initial_q_value = initial_q_value

        self.initial_epsilon = initial_epsilon
        self.anneal_steps = anneal_steps
        self.final_epsilon = final_epsilon
        self.use_scheduler = use_scheduler

        self.reward_bonus = reward_bonus
        self.stability_coef = stability_coef

        self.num_epochs = num_epochs

    def initial_state(self) -> QLearningState:
        """Initialize Q-table and visit counts."""
        return QLearningState(
            q_table=jnp.full((self.num_states, self.num_actions), self.initial_q_value, dtype=jnp.float32),
            visit_counts=jnp.zeros((self.num_states, self.num_actions)),
            step=0,
        )

    def select_action(
        self, state: QLearningState, obs: jnp.ndarray, key: jax.Array, is_training: bool
    ) -> Tuple[QLearningState, jnp.ndarray]:
        """Select action based on policy."""
        q_values = state.q_table[obs]

        if is_training:
            if self.use_scheduler:
                frac = jnp.clip(state.step / max(1, self.anneal_steps), 0.0, 1.0)
                eps = self.initial_epsilon + frac * (self.final_epsilon - self.initial_epsilon)
            else:
                eps = self.initial_epsilon

            action = distrax.EpsilonGreedy(q_values, epsilon=eps).sample(seed=key)
            state = state.replace(
                step=state.step + 1,
                visit_counts=state.visit_counts.at[obs, action].add(1),
            )
        else:
            action = distrax.Greedy(q_values).sample(seed=key)

        return state, action


    def update(
        self,
        state: QLearningState,
        batch: Transition,
    ) -> tuple[QLearningState, jax.Array]:
        """Update Q-table."""
        batch_size = batch.observation.shape[0]

        s = batch.observation.astype(jnp.int32).reshape(-1)
        a = batch.action.astype(jnp.int32).reshape(-1)
        r = batch.reward
        s_next = batch.next_observation.astype(jnp.int32).reshape(-1)
        terminal = batch.terminal
        mask = batch.mask

        reward_bonus = self.reward_bonus / (jnp.sqrt(state.visit_counts[s, a]) + self.stability_coef)
        reward = r + reward_bonus * mask

        def update_body(q_table, idx):
            q_next_max = jnp.max(q_table[s_next[idx]], axis=-1)
            target = reward[idx] + self.discount * q_next_max * (
                1.0 - terminal[idx]
            )
            masked_td_error = (target - q_table[s[idx], a[idx]]) * mask[idx]

            q_table = q_table.at[s[idx], a[idx]].add(self.step_size * masked_td_error)
            return q_table, jnp.abs(masked_td_error)
        
        # TODO: shuffle batch per epoch or do it in the buffer
        new_q_table, losses = jax.lax.scan(
            update_body, state.q_table, jnp.tile(jnp.arange(batch_size), self.num_epochs)
        )

        next_state = state.replace(
            q_table=new_q_table,
        )

        return next_state, jnp.sum(losses[-batch_size:]) / jnp.maximum(mask.sum(), 1.0)

    def bootstrap_value(
        self, state: QLearningState, next_observation: jnp.ndarray
    ) -> jax.Array:
        s_next = next_observation.astype(jnp.int32).reshape(-1)
        return jnp.max(state.q_table[s_next], axis=-1)
