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
    learn_flags: jnp.ndarray
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
            learn_flags=jnp.ones((self.num_states, self.num_actions), dtype=bool),
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
        learn_flags = state.learn_flags

        def update_body(i, carry):
            q_table, visit_counts, pending_counts, pending_returns, learn_flags, total_change, total_updates, success_any = carry
            s = batch.observation[i].astype(jnp.int32).squeeze()
            a = batch.action[i].astype(jnp.int32).squeeze()
            r = batch.reward[i]
            s_next = batch.next_observation[i].astype(jnp.int32).squeeze()
            terminal = batch.terminal[i]

            q_next_max = jnp.max(q_table[s_next])
            target = r + self.discount * q_next_max * (1.0 - terminal.astype(jnp.float32))

            valid = batch_mask[i]
            visit_counts = visit_counts.at[s, a].add(jnp.where(valid, 1, 0))

            can_learn = jnp.logical_and(valid, learn_flags[s, a])
            pending_counts = pending_counts.at[s, a].add(jnp.where(can_learn, 1, 0))
            pending_returns = pending_returns.at[s, a].add(jnp.where(can_learn, target, 0.0))

            count = pending_counts[s, a]
            total_return = pending_returns[s, a]
            ready = jnp.logical_and(can_learn, count >= self.update_threshold)

            def apply_update(_):
                q_current = q_table[s, a]
                mean_est = total_return / jnp.maximum(count, 1)
                decrease = q_current - mean_est >= 2 * self.epsilon
                proposed_q = mean_est + self.epsilon
                new_q = jnp.where(decrease, proposed_q, q_current)

                updated_table = q_table.at[s, a].set(new_q)
                updated_counts = pending_counts.at[s, a].set(0)
                updated_returns = pending_returns.at[s, a].set(0.0)
                updated_flags = jax.lax.cond(
                    decrease,
                    lambda f: f,
                    lambda f: f.at[s, a].set(False),
                    learn_flags
                )

                change = jnp.where(decrease, jnp.abs(q_current - new_q), 0.0)
                update_flag = jnp.where(decrease, 1.0, 0.0)
                return updated_table, updated_counts, updated_returns, updated_flags, change, update_flag

            def skip_update(_):
                return q_table, pending_counts, pending_returns, learn_flags, 0.0, 0.0

            q_table, pending_counts, pending_returns, learn_flags, change, update_flag = jax.lax.cond(
                ready,
                apply_update,
                skip_update,
                operand=None
            )

            total_change = total_change + change
            total_updates = total_updates + update_flag
            success_any = jnp.logical_or(success_any, update_flag > 0)
            return q_table, visit_counts, pending_counts, pending_returns, learn_flags, total_change, total_updates, success_any

        init_carry = (q_table, visit_counts, pending_counts, pending_returns, learn_flags, 0.0, 0.0, False)
        q_table, visit_counts, pending_counts, pending_returns, learn_flags, total_change, total_updates, success_any = jax.lax.fori_loop(
            0, batch_size, update_body, init_carry
        )

        learn_flags = jax.lax.cond(
            success_any,
            lambda _: jnp.ones_like(learn_flags, dtype=bool),
            lambda f: f,
            learn_flags
        )
        
        new_state = state.replace(
            q_table=q_table,
            visit_counts=visit_counts,
            pending_counts=pending_counts,
            pending_returns=pending_returns,
            learn_flags=learn_flags,
            step=state.step + jnp.sum(batch_mask)
        )

        loss = total_change / jnp.maximum(total_updates, 1.0)
        return new_state, loss

    def bootstrap_value(self, state: DelayedQLearningState, next_observation: jnp.ndarray) -> jax.Array:
        s_next = next_observation.astype(jnp.int32).squeeze()
        return jnp.max(state.q_table[s_next])
