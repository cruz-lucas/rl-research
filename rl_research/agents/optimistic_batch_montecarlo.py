import jax
import jax.numpy as jnp
from flax import struct
from rl_research.policies import _select_greedy
from rl_research.buffers import Transition
import gin


@struct.dataclass
class OptimisticQLearningState:
    """State for optimistic Q-learning agent."""
    q_table: jnp.ndarray
    visit_counts: jnp.ndarray
    step: int
    converged: bool


@gin.configurable
class OptimisticMonteCarloAgent:
    """Optimistic Q-learning variant that learns from Monte Carlo returns."""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        r_max: float,
        discount: float,
        step_size: float,
        known_threshold: int,
        convergence_threshold: float
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.r_max = r_max
        self.discount = discount
        self.step_size = step_size
        self.known_threshold = known_threshold
        self.optimistic_value = r_max / (1 - discount)
        self.convergence_threshold = convergence_threshold

    def initial_state(self) -> OptimisticQLearningState:
        """Initialize with optimistic Q-values and zero counts."""
        return OptimisticQLearningState(
            q_table=jnp.full((self.num_states, self.num_actions), self.optimistic_value),
            visit_counts=jnp.zeros((self.num_states, self.num_actions)),
            converged=False,
            step=0,
        )

    def select_action(
        self,
        state: OptimisticQLearningState,
        obs: jnp.ndarray,
        key: jax.Array,
        is_training: bool,
    ) -> jnp.ndarray:
        """Greedy action selection with optimistic initialization."""
        q_values = state.q_table[obs]
        return _select_greedy(q_values, key)

    def update(
        self,
        state: OptimisticQLearningState,
        batch: Transition,
        batch_mask: jnp.ndarray | None = None
    ) -> tuple[OptimisticQLearningState, jax.Array]:
        """Single-step optimistic updates using Monte Carlo returns."""
        batch_size = batch.observation.shape[0]
        if batch_mask is None:
            batch_mask = jnp.ones((batch_size,), dtype=bool)
        batch_mask = batch_mask.astype(jnp.bool_)
        
        def update_single(carry, i):
            q_table, visit_counts = carry
            valid = batch_mask[i]

            s = batch.observation[i].astype(jnp.int32).squeeze()
            a = batch.action[i].astype(jnp.int32).squeeze()
            mc_return = batch.reward[i]
            s_next = batch.next_observation[i].astype(jnp.int32).squeeze()

            visit_counts = visit_counts.at[s, a].add(jnp.where(valid, 1, 0))
            is_unknown = visit_counts[s, a] < self.known_threshold
            is_next_unknown = jnp.any(visit_counts[s_next] < self.known_threshold)

            q_current = q_table[s, a]
            target = jnp.where(is_next_unknown, self.optimistic_value, mc_return)

            td_error = target - q_current
            updated_q = q_current + self.step_size * td_error
            new_q = jnp.where(is_unknown, self.optimistic_value, updated_q)
            new_q = jnp.where(valid, new_q, q_current)

            q_table = jax.lax.cond(
                valid,
                lambda tab: tab.at[s, a].set(new_q),
                lambda tab: tab,
                q_table
            )

            loss_val = jnp.where(valid, jnp.abs(td_error), 0.0)
            return (q_table, visit_counts), loss_val

        (new_q_table, new_visit_counts), losses = jax.lax.scan(
            update_single,
            (state.q_table, state.visit_counts),
            jnp.arange(batch_size)
        )

        total_valid = jnp.maximum(jnp.sum(batch_mask), 1)
        mean_loss = jnp.sum(losses) / total_valid

        new_state = state.replace(
            q_table=new_q_table,
            visit_counts=new_visit_counts,
            step=state.step + jnp.sum(batch_mask)
        )
        
        return new_state, mean_loss
