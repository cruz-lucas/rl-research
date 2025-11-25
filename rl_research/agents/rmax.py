import jax
import jax.numpy as jnp
from flax import struct
from rl_research.buffers import Transition
from rl_research.policies import _select_greedy
import gin

@struct.dataclass
class RMaxState:
    """State for R-Max agent."""
    q_table: jnp.ndarray
    transition_counts: jnp.ndarray
    reward_sums: jnp.ndarray
    visit_counts: jnp.ndarray
    step: int

@gin.configurable
class RMaxAgent:
    """R-Max agent with known model."""
    
    def __init__(
        self,
        num_states: int,
        num_actions: int,
        r_max: float = 6.0,
        discount: float = 0.9,
        known_threshold: int = 1,
        convergence_threshold: float = 1e-2,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.r_max = r_max
        self.discount = discount
        self.known_threshold = known_threshold
        self.convergence_threshold = convergence_threshold
        self.vi_iterations = int(
            jnp.ceil(jnp.log(1 / (convergence_threshold * (1 - discount))) / (1 - discount))
        )
        self.optimistic_value = r_max / (1 - discount)
    
    def initial_state(self) -> RMaxState:
        """Initialize with optimistic Q-values."""
        return RMaxState(
            q_table=jnp.full((self.num_states, self.num_actions), self.optimistic_value),
            transition_counts=jnp.zeros((self.num_states, self.num_actions, self.num_states)),
            reward_sums=jnp.zeros((self.num_states, self.num_actions)),
            visit_counts=jnp.zeros((self.num_states, self.num_actions)),
            step=0
        )
    
    def select_action(
        self,
        state: RMaxState,
        obs: jnp.ndarray,
        key: jax.Array,
        is_training: bool
    ) -> jnp.ndarray:
        """Select greedy action with random tie-breaking."""
        q_values = state.q_table[obs]
                
        return _select_greedy(q_values, key)
    
    def _value_iteration_step(self, q_table, is_known, rewards, transitions):
        """Single step of value iteration."""
        v_table = jnp.max(q_table, axis=1)
        
        def compute_q(s, a):
            expected_next_value = jnp.sum(transitions[s, a] * v_table)
            q_val = rewards[s, a] + self.discount * expected_next_value
            
            q_val = jnp.where(is_known[s, a], q_val, self.optimistic_value)
            return q_val
        
        new_q = jax.vmap(
            lambda s: jax.vmap(lambda a: compute_q(s, a))(jnp.arange(self.num_actions))
        )(jnp.arange(self.num_states))
        
        return new_q
    
    def update(
        self,
        state: RMaxState,
        batch: Transition,
        batch_mask: jnp.ndarray | None = None
    ) -> tuple[RMaxState, jax.Array]:
        """Update model and recompute Q-values."""
        batch_size = batch.observation.shape[0]
        if batch_mask is None:
            batch_mask = jnp.ones((batch_size,), dtype=bool)
        batch_mask = batch_mask.astype(jnp.bool_)
        
        new_transition_counts = state.transition_counts
        new_reward_sums = state.reward_sums
        new_visit_counts = state.visit_counts
        prev_is_known = state.visit_counts >= self.known_threshold
        
        for i in range(batch_size):
            s = batch.observation[i].astype(jnp.int32).squeeze()
            a = batch.action[i].astype(jnp.int32).squeeze()
            r = batch.reward[i]
            s_next = batch.next_observation[i].astype(jnp.int32).squeeze()
            
            valid_unknown = jnp.logical_and(batch_mask[i], new_visit_counts[s, a] < self.known_threshold)
            inc = jnp.where(valid_unknown, 1, 0)

            new_transition_counts = new_transition_counts.at[s, a, s_next].add(inc)
            new_reward_sums = new_reward_sums.at[s, a].add(jnp.where(valid_unknown, r, 0.0))
            new_visit_counts = new_visit_counts.at[s, a].add(inc)
        
        q_table = state.q_table
        new_is_known = new_visit_counts >= self.known_threshold
        newly_known = jnp.logical_and(jnp.logical_not(prev_is_known), new_is_known)
        newly_known_any = jnp.any(newly_known)

        def run_value_iteration(q_init):
            safe_counts = jnp.maximum(new_visit_counts, 1)
            avg_rewards = new_reward_sums / safe_counts
            transition_probs = new_transition_counts / safe_counts[:, :, None]

            def body(_, carry):
                q_prev, converged = carry
                new_q = self._value_iteration_step(q_prev, new_is_known, avg_rewards, transition_probs)
                max_change = jnp.max(jnp.abs(new_q - q_prev))
                converged_now = max_change < self.convergence_threshold
                converged = jnp.logical_or(converged, converged_now)
                q_next = jnp.where(converged, q_prev, new_q)
                return q_next, converged

            q_final, _ = jax.lax.fori_loop(0, self.vi_iterations, body, (q_init, False))
            return q_final

        q_table = jax.lax.cond(newly_known_any, run_value_iteration, lambda q: q, q_table)
        
        new_state = state.replace(
            q_table=q_table,
            transition_counts=new_transition_counts,
            reward_sums=new_reward_sums,
            visit_counts=new_visit_counts,
            step=state.step + jnp.sum(batch_mask)
        )
        
        loss = jnp.mean(jnp.abs(q_table - state.q_table))
        return new_state, loss
