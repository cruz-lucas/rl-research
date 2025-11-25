import jax
import jax.numpy as jnp
from flax import struct
from rl_research.policies import _select_greedy
from rl_research.buffers import Transition
import gin


@struct.dataclass
class DelayedQLearningState:
    """State for delayed Q-learning."""
    q_table: jnp.ndarray
    visit_counts: jnp.ndarray
    pending_counts: jnp.ndarray
    pending_returns: jnp.ndarray
    step: int


@gin.configurable
class DelayedQLearningAgent:
    """Delayed Q-Learning with optimistic initialization."""
    
    def __init__(
        self,
        num_states: int,
        num_actions: int,
        r_max: float = 6.0,
        discount: float = 0.9,
        epsilon: float = 0.05,
        update_threshold: int = 20,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.r_max = r_max
        self.discount = discount
        self.epsilon = epsilon
        self.update_threshold = update_threshold
        self.optimistic_value = r_max / (1 - discount)
    
    def initial_state(self) -> DelayedQLearningState:
        """Initialize Q-values optimistically and zero counters."""
        return DelayedQLearningState(
            q_table=jnp.full((self.num_states, self.num_actions), self.optimistic_value),
            visit_counts=jnp.zeros((self.num_states, self.num_actions)),
            pending_counts=jnp.zeros((self.num_states, self.num_actions)),
            pending_returns=jnp.zeros((self.num_states, self.num_actions)),
            step=0
        )
    
    def select_action(
        self,
        state: DelayedQLearningState,
        obs: jnp.ndarray,
        key: jax.Array,
        is_training: bool
    ) -> jnp.ndarray:
        """Greedy selection with optimistic Q-values."""
        q_values = state.q_table[obs]
        return _select_greedy(q_values, key)
    
    def update(
        self,
        state: DelayedQLearningState,
        batch: Transition,
        batch_mask: jnp.ndarray | None = None
    ) -> tuple[DelayedQLearningState, jax.Array]:
        """Accumulate samples per (s,a) and update after enough evidence."""
        batch_size = batch.observation.shape[0]
        if batch_mask is None:
            batch_mask = jnp.ones((batch_size,), dtype=bool)
        batch_mask = batch_mask.astype(jnp.bool_)

        q_table = state.q_table
        visit_counts = state.visit_counts
        pending_counts = state.pending_counts
        pending_returns = state.pending_returns

        total_change = jnp.array(0.0)
        total_updates = jnp.array(0.0)

        for i in range(batch_size):
            valid = batch_mask[i]
            s = batch.observation[i].astype(jnp.int32).squeeze()
            a = batch.action[i].astype(jnp.int32).squeeze()
            r = batch.reward[i]
            s_next = batch.next_observation[i].astype(jnp.int32).squeeze()

            q_next_max = jnp.max(q_table[s_next])
            target = r + self.discount * q_next_max

            visit_counts = visit_counts.at[s, a].add(jnp.where(valid, 1, 0))
            pending_counts = pending_counts.at[s, a].add(jnp.where(valid, 1, 0))
            pending_returns = pending_returns.at[s, a].add(jnp.where(valid, target, 0.0))

            count = pending_counts[s, a]
            total_return = pending_returns[s, a]
            ready = jnp.logical_and(valid, count >= self.update_threshold)

            def apply_update(_):
                q_current = q_table[s, a]
                mean_est = total_return / jnp.maximum(count, 1)
                decrease = q_current - mean_est >= 2 * self.epsilon
                proposed_q = mean_est + self.epsilon
                new_q = jnp.where(decrease, proposed_q, q_current)

                updated_table = q_table.at[s, a].set(new_q)
                updated_counts = pending_counts.at[s, a].set(0)
                updated_returns = pending_returns.at[s, a].set(0.0)

                change = jnp.where(decrease, jnp.abs(q_current - new_q), 0.0)
                update_flag = jnp.where(decrease, 1.0, 0.0)
                return updated_table, updated_counts, updated_returns, change, update_flag

            def skip_update(_):
                return q_table, pending_counts, pending_returns, 0.0, 0.0

            q_table, pending_counts, pending_returns, change, update_flag = jax.lax.cond(
                ready,
                apply_update,
                skip_update,
                operand=None
            )

            total_change = total_change + change
            total_updates = total_updates + update_flag
        
        new_state = state.replace(
            q_table=q_table,
            visit_counts=visit_counts,
            pending_counts=pending_counts,
            pending_returns=pending_returns,
            step=state.step + jnp.sum(batch_mask)
        )

        loss = total_change / jnp.maximum(total_updates, 1.0)
        return new_state, loss
