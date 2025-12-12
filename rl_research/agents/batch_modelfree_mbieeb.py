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
                (self.num_states, self.num_actions), 0.0#self.optimistic_value
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
        values = q_values + self.beta / (jnp.sqrt(state.visit_counts[obs]) + 0.001)

        return _select_greedy(values, key)

    def update(
        self,
        state: BMFMBIEEBState,
        batch: Transition,
    ) -> tuple[BMFMBIEEBState, jax.Array]:
        """Single-step optimistic Q-learning updates for each transition."""
        batch_size = batch.observation.shape[0]

        def update_single(carry, i):
            q_table, visit_counts = carry

            s = batch.observation[i].astype(jnp.int32).squeeze()
            a = batch.action[i].astype(jnp.int32).squeeze()
            r = batch.reward[i]
            s_next = batch.next_observation[i].astype(jnp.int32).squeeze()
            terminal = batch.terminal[i]

            visit_counts = visit_counts.at[s, a].add(1)

            exploration_bonus = self.beta / jnp.sqrt(visit_counts[s, a])

            q_current = q_table[s, a]
            q_next_max = jnp.max(q_table[s_next])

            # TODO: include termination
            target = r + self.discount * q_next_max + exploration_bonus

            td_error = target - q_current
            new_q = q_current + self.step_size * td_error

            loss_val = jnp.abs(td_error)
            return (q_table.at[s, a].set(new_q), visit_counts), loss_val

        (new_q_table, new_visit_counts), losses = jax.lax.scan(
            update_single, (state.q_table, state.visit_counts), jnp.arange(batch_size)
        )

        mean_loss = jnp.sum(losses) / batch_size

        new_state = state.replace(
            q_table=new_q_table,
            visit_counts=new_visit_counts,
            step=state.step + jnp.sum(batch_size),
        )

        return new_state, mean_loss

    def bootstrap_value(
        self, state: BMFMBIEEBState, next_observation: jnp.ndarray
    ) -> jax.Array:
        s_next = next_observation.astype(jnp.int32).squeeze()
        return jnp.max(state.q_table[s_next])
