import gin
import jax
import jax.numpy as jnp
from flax import struct

from rl_research.buffers import Transition
from rl_research.policies import _select_greedy


@struct.dataclass
class BMFMBIEEBState:
    """State for Batch Model-free MBIE-EB."""

    q_table: jnp.ndarray
    visit_counts: jnp.ndarray
    step: int


@gin.configurable
class BMFMBIEEBAgent:
    """Model-free MBIE-EB variant."""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        r_max: float,
        discount: float,
        step_size: float,
        beta: float,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.r_max = r_max
        self.discount = discount
        self.step_size = step_size
        self.beta = beta
        self.optimistic_value = r_max / (1 - discount)

    def initial_state(self) -> BMFMBIEEBState:
        """Initialize with optimistic Q-values."""
        return BMFMBIEEBState(
            q_table=jnp.full(
                (self.num_states, self.num_actions), self.optimistic_value # 0
            ),
            visit_counts=jnp.zeros((self.num_states, self.num_actions)),
            step=0,
        )

    def select_action(
        self,
        state: BMFMBIEEBState,
        obs: jnp.ndarray,
        key: jax.Array,
        is_training: bool,
    ) -> jnp.ndarray:
        """Select greedy action with random tie-breaking."""
        q_values = state.q_table[obs]
        values = q_values# + self.beta / (jnp.sqrt(state.visit_counts[obs]) + 0.02)

        return _select_greedy(values, key)

    def update(
        self,
        state: BMFMBIEEBState,
        batch: Transition,
    ) -> tuple[BMFMBIEEBState, jax.Array]:
        """Single-step optimistic Q-learning updates for each transition."""
        batch_size = batch.observation.shape[0]

        # new_visit_counts = state.visit_counts.at[
        #     batch.observation.astype(jnp.int32), batch.action.astype(jnp.int32)].add(1)
        new_visit_counts = state.visit_counts

        def update_single(carry, i):
            q_table = carry

            s = batch.observation[i].astype(jnp.int32).squeeze()
            a = batch.action[i].astype(jnp.int32).squeeze()
            r = batch.reward[i]
            s_next = batch.next_observation[i].astype(jnp.int32).squeeze()
            terminal = batch.terminal[i].astype(jnp.int32).squeeze()

            exploration_bonus = self.beta / (jnp.sqrt(new_visit_counts[s, a]) + 0.02)
            # next_state_bonus = self.beta / (jnp.sqrt(new_visit_counts[s_next]) + 0.02)

            q_current = q_table[s, a]
            # q_next_max = jnp.max(q_table[s_next] + next_state_bonus)
            q_next_max = jnp.max(q_table[s_next])
            
            target = r + self.discount * q_next_max * (1-terminal) + exploration_bonus

            td_error = target - q_current
            new_q = q_current + self.step_size * td_error

            loss_val = jnp.abs(td_error)
            return q_table.at[s, a].set(new_q[0]), loss_val

        epochs = 200
        new_q_table, losses = jax.lax.scan(
            update_single, state.q_table, jnp.tile(jnp.arange(batch_size), epochs)
        )

        mean_loss = jnp.mean(losses[-batch_size:])

        new_state = state.replace(
            q_table=new_q_table,
            # visit_counts=new_visit_counts
        )

        return new_state, mean_loss

    def bootstrap_value(
        self, state: BMFMBIEEBState, next_observation: jnp.ndarray
    ) -> jax.Array:
        s_next = next_observation.astype(jnp.int32).squeeze()
        return jnp.max(state.q_table[s_next])
