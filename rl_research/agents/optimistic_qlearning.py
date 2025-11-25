import jax
import jax.numpy as jnp
from typing import Literal
from flax import struct
from rl_research.policies import _select_greedy, _select_random, _select_epsilon_greedy, _select_ucb
from rl_research.experiment import Transition
import gin


@struct.dataclass
class OptimisticQLearningState:
    """State for optimistic Q-learning agent."""
    q_table: jnp.ndarray
    visit_counts: jnp.ndarray
    step: int
    converged: bool


@gin.configurable
class OptimisticQLearningAgent:
    """Model-free R-Max variant using optimistic initialization."""
    
    def __init__(
        self,
        num_states: int,
        num_actions: int,
        r_max: float,
        discount: float,
        step_size: float,
        known_threshold: int,
        convergence_threshold: float = 1e-3
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.r_max = r_max
        self.discount = discount
        self.step_size = step_size
        self.known_threshold = known_threshold
        self.convergence_iterations = jnp.log(1/(convergence_threshold * (1-discount)))/(1-discount)
        self.convergence_threshold = convergence_threshold
        self.optimistic_value = r_max / (1 - discount)
    
    def initial_state(self) -> OptimisticQLearningState:
        """Initialize with optimistic Q-values."""
        return OptimisticQLearningState(
            q_table=jnp.full((self.num_states, self.num_actions), 0.0),
            visit_counts=jnp.zeros((self.num_states, self.num_actions)),
            step=0,
            converged=False
        )
    
    def select_action(
        self,
        state: OptimisticQLearningState,
        obs: jnp.ndarray,
        key: jax.Array,
        is_training: bool
    ) -> jnp.ndarray:
        """Select greedy action with random tie-breaking."""
        q_values = state.q_table[obs]
                
        return _select_greedy(q_values, key)
    
    def _update_to_convergence(self, q_table, visit_counts, batch):
        """Update Q-values using batch until convergence."""
        batch_size = batch.observation.shape[0]
        
        def iteration_step(q_tab, _):
            def update_single(i):
                s = batch.observation[i].astype(jnp.int32).squeeze()
                a = batch.action[i].astype(jnp.int32).squeeze()
                r = batch.reward[i]
                d = batch.discount[i]
                s_next = batch.next_observation[i].astype(jnp.int32).squeeze()
                
                is_unknown = visit_counts[s, a] < self.known_threshold
                is_next_unknown = jnp.any(visit_counts[s_next] < self.known_threshold)
                
                q_current = q_tab[s, a]
                q_next_max = jnp.where(is_next_unknown, self.optimistic_value, jnp.max(q_tab[s_next]))

                target = r + d * self.discount * q_next_max
                
                td_error = target - q_current
                new_q = jnp.where(is_unknown, self.optimistic_value, q_current + self.step_size * td_error)
                
                return s, a, new_q, jnp.abs(q_current - new_q)
            
            states, actions, new_qs, errors = jax.vmap(update_single)(jnp.arange(batch_size))
            
            new_q_tab = q_tab
            for i in range(batch_size):
                new_q_tab = new_q_tab.at[states[i], actions[i]].set(new_qs[i])
            
            max_change = jnp.max(errors)
            return new_q_tab, max_change
        
        final_q_table, changes = jax.lax.scan(
            iteration_step,
            q_table,
            None,
            length=self.convergence_iterations
        )
        
        return final_q_table, jnp.mean(changes)
    
    def update(
        self,
        state: OptimisticQLearningState,
        batch: Transition
    ) -> tuple[OptimisticQLearningState, jax.Array]:
        """Update Q-table with convergence iterations."""
        batch_size = batch.observation.shape[0]
        
        new_visit_counts = state.visit_counts
        for i in range(batch_size):
            s = batch.observation[i].astype(jnp.int32).squeeze()
            a = batch.action[i].astype(jnp.int32).squeeze()
            new_visit_counts = new_visit_counts.at[s, a].add(1)
        
        new_q_table, loss = self._update_to_convergence(
            state.q_table,
            new_visit_counts,
            batch
        )
        
        new_state = state.replace(
            q_table=new_q_table,
            visit_counts=new_visit_counts,
            step=state.step + batch_size
        )
        
        return new_state, loss