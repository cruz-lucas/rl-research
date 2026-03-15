import gin
import jax
import jax.numpy as jnp
from typing import Tuple
import distrax

from rl_research.buffers import Transition
from rl_research.agents.tabular.qlearning import QLearningAgent, QLearningState


@gin.configurable
class ReplaybasedMBIEEB(QLearningAgent):
    """Model-free MBIE-EB variant."""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        r_max: float,
        discount: float,
        step_size: float,
        exploration_bonus: float,
        num_epochs: int,
        initial_q_value: float,
        stability_coeff: float = 0.02,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.r_max = r_max
        self.discount = discount
        self.step_size = step_size
        self.exploration_bonus = exploration_bonus
        self.optimistic_value = r_max / (1 - discount)
        self.stability_coeff = stability_coeff
        self.num_epochs = num_epochs
        self.initial_q_value = initial_q_value


    def select_action(
        self,
        state: QLearningState,
        obs: jnp.ndarray,
        key: jax.Array,
        is_training: bool,
    ) -> Tuple[QLearningState, jnp.ndarray]:
        """Select greedy action with random tie-breaking."""
        q_values = state.q_table[obs]

        values = q_values + self.exploration_bonus / (jnp.sqrt(state.visit_counts[obs]) + self.stability_coeff)
        action = distrax.Greedy(values).sample(seed=key)

        if is_training:
            state = state.replace(
                step=state.step + 1,
                visit_counts=state.visit_counts.at[obs, action].add(1)
            )

        return state, action

    def update(
        self,
        state: QLearningState,
        batch: Transition,
    ) -> tuple[QLearningState, jax.Array]:
        """Single-step optimistic Q-learning updates for each transition."""
        batch_size = batch.observation.shape[0]

        s = batch.observation.astype(jnp.int32).reshape(-1)
        a = batch.action.astype(jnp.int32).reshape(-1)
        r = batch.reward
        s_next = batch.next_observation.astype(jnp.int32).reshape(-1)
        terminal = batch.terminal
        mask = batch.mask

        exploration_bonus = self.exploration_bonus / (jnp.sqrt(state.visit_counts[s, a]) + self.stability_coeff)
        next_state_bonus = self.exploration_bonus / (jnp.sqrt(state.visit_counts[s_next]) + self.stability_coeff)

        def update_single(q_table, idx):
            q_current = q_table[s[idx], a[idx]]
            q_next_max = jnp.max(q_table[s_next[idx]] + next_state_bonus[idx], axis=-1)
            
            target = r[idx] + (exploration_bonus[idx] + self.discount * q_next_max) * (1-terminal[idx])

            td_error = (target - q_current) * mask[idx]
            new_q = q_current + self.step_size * td_error

            loss_val = jnp.abs(td_error)
            return q_table.at[s[idx], a[idx]].set(new_q), loss_val

        new_q_table, losses = jax.lax.scan(
            update_single, state.q_table, jnp.tile(jnp.arange(batch_size), self.num_epochs)
        )

        mean_loss = jnp.mean(losses[-batch_size:])

        new_state = state.replace(
            q_table=new_q_table,
        )

        return new_state, mean_loss
