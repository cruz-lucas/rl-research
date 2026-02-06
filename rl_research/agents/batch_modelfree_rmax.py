import gin
import jax
import jax.numpy as jnp
from flax import struct

from rl_research.buffers import Transition
from rl_research.policies import _select_greedy


@struct.dataclass
class BMFRmaxState:
    """State for optimistic Q-learning agent."""

    q_table: jnp.ndarray
    visit_counts: jnp.ndarray
    step: int


@gin.configurable
class BMFRmaxAgent:
    """Model-free R-Max variant using optimistic initialization."""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        r_max: float,
        discount: float,
        step_size: float,
        known_threshold: int,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.r_max = r_max
        self.discount = discount
        self.step_size = step_size
        self.known_threshold = known_threshold
        self.optimistic_value = r_max / (1 - discount)

    def initial_state(self) -> BMFRmaxState:
        """Initialize with optimistic Q-values."""
        return BMFRmaxState(
            q_table=jnp.full(
                (self.num_states, self.num_actions), 0.0#self.optimistic_value
            ),
            visit_counts=jnp.zeros((self.num_states, self.num_actions)),
            step=0,
        )

    def select_action(
        self,
        state: BMFRmaxState,
        obs: jnp.ndarray,
        key: jax.Array,
        is_training: bool,
    ) -> jnp.ndarray:
        """Select greedy action with random tie-breaking."""
        q_values = state.q_table[obs]

        is_known = state.visit_counts[obs] > self.known_threshold
        values = jnp.where(is_known, q_values, self.optimistic_value)

        return _select_greedy(values, key)
        # return _select_greedy(q_values, key)

    def update(
        self,
        state: BMFRmaxState,
        batch: Transition,
    ) -> tuple[BMFRmaxState, jax.Array]:
        """Single-step optimistic Q-learning updates for each transition."""
        batch_size = batch.observation.shape[0]

        def update_single(carry, i):
            q_table, visit_counts = carry

            s = batch.observation[i].astype(jnp.int32).squeeze()
            a = batch.action[i].astype(jnp.int32).squeeze()
            r = batch.reward[i]
            s_next = batch.next_observation[i].astype(jnp.int32).squeeze()
            terminal = batch.terminal[i]

            # The version 2 of this algo increments visit counts after checking knownness. This is version 3.
            visit_counts = visit_counts.at[s, a].add(1)

            is_unknown = visit_counts[s, a] < self.known_threshold
            is_next_unknown = jnp.any(visit_counts[s_next] < self.known_threshold)

            q_current = q_table[s, a]
            q_next_max = jax.lax.cond(
                terminal,
                lambda _: 0.0,
                lambda _: jnp.where(
                    is_next_unknown, self.optimistic_value, jnp.max(q_table[s_next])
                ),
                operand=None,
            )
            target = r + self.discount * q_next_max

            td_error = target - q_current
            # new_q = q_current + self.step_size * td_error
            updated_q = q_current + self.step_size * td_error
            # new_q = jnp.where(is_unknown, self.optimistic_value, updated_q)
            new_q = jnp.array(jnp.where(is_unknown, q_current, updated_q))[0]

            loss_val = jnp.abs(td_error)
            return (q_table.at[s, a].set(new_q), visit_counts), loss_val

        (new_q_table, new_visit_counts), losses = jax.lax.scan(
            update_single, (state.q_table, state.visit_counts), jnp.arange(batch_size)
        )

        mean_loss = jnp.sum(losses) / batch_size

        new_state = state.replace(
            q_table=new_q_table,
            visit_counts=new_visit_counts,
        )

        return new_state, mean_loss

    def bootstrap_value(
        self, state: BMFRmaxState, next_observation: jnp.ndarray
    ) -> jax.Array:
        s_next = next_observation.astype(jnp.int32).squeeze()
        return jnp.max(state.q_table[s_next])
