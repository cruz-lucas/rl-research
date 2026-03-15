import gin
import jax
import jax.numpy as jnp
from typing import Tuple
import distrax

from rl_research.buffers import Transition
from rl_research.agents.tabular.qlearning import QLearningAgent, QLearningState

@gin.configurable
class ReplaybasedRmax(QLearningAgent):
    """Model-free R-Max variant."""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        r_max: float,
        discount: float,
        step_size: float,
        known_threshold: int,
        initial_q_value: float,
        num_epochs: int,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.r_max = r_max
        self.discount = discount
        self.step_size = step_size
        self.known_threshold = known_threshold
        self.optimistic_value = r_max / (1 - discount)

        self.initial_q_value = initial_q_value
        self.num_epochs = num_epochs


    def select_action(
        self,
        state: QLearningState,
        obs: jnp.ndarray,
        key: jax.Array,
        is_training: bool,
    ) -> Tuple[QLearningState, jnp.ndarray]:
        """Select greedy action with random tie-breaking."""
        q_values = state.q_table[obs]

        is_known = state.visit_counts[obs] >= self.known_threshold
        values = jnp.where(is_known, q_values, self.optimistic_value)

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

        is_unknown = state.visit_counts[s, a] < self.known_threshold
        is_next_unknown = jnp.any(state.visit_counts[s_next] < self.known_threshold, axis=-1)

        def update_single(q_table, idx):
            q_current = q_table[s[idx], a[idx]]
            q_next_max = jax.lax.cond(
                terminal[idx],
                lambda _: 0.0,
                lambda _: jnp.where(
                    is_next_unknown[idx], self.optimistic_value, jnp.max(q_table[s_next[idx]])
                ),
                operand=None,
            )
            target = r[idx] + self.discount * q_next_max

            td_error = (target - q_current) * mask[idx]
            updated_q = q_current + self.step_size * td_error
            new_q = jnp.where(is_unknown[idx], q_current, updated_q)

            loss_val = jnp.abs(td_error)
            return q_table.at[s[idx], a[idx]].set(new_q), loss_val

        new_q_table, losses = jax.lax.scan(
            update_single, state.q_table, jnp.tile(jnp.arange(batch_size), self.num_epochs)
        )

        new_state = state.replace(
            q_table=new_q_table,
        )

        return new_state, jnp.sum(losses[-batch_size:]) / jnp.maximum(mask.sum(), 1.0)
