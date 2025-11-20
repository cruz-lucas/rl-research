@struct.dataclass
class OptimisticQLearningState:
    """State for optimistic Q-learning agent."""
    q_table: jnp.ndarray  # [num_states, num_actions]
    visit_counts: jnp.ndarray  # [num_states, num_actions]
    step: int
    converged: bool


class OptimisticQLearningAgent:
    """Model-free R-Max variant using optimistic initialization."""
    
    def __init__(
        self,
        num_states: int,
        num_actions: int,
        r_max: float = 1.0,
        discount: float = 0.99,
        learning_rate: float = 0.1,
        known_threshold: int = 1,
        convergence_iterations: int = 100,
        convergence_threshold: float = 1e-4
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.r_max = r_max
        self.discount = discount
        self.learning_rate = learning_rate
        self.known_threshold = known_threshold
        self.convergence_iterations = convergence_iterations
        self.convergence_threshold = convergence_threshold
        self.optimistic_value = r_max / (1 - discount)
    
    def initial_state(self) -> OptimisticQLearningState:
        """Initialize with optimistic Q-values."""
        return OptimisticQLearningState(
            q_table=jnp.full((self.num_states, self.num_actions), self.optimistic_value),
            visit_counts=jnp.zeros((self.num_states, self.num_actions)),
            step=0,
            converged=False
        )
    
    def _break_ties_randomly(self, values: jnp.ndarray, key: jax.Array) -> int:
        """Break ties randomly among maximum values."""
        max_val = jnp.max(values)
        is_max = values == max_val
        noise = jax.random.uniform(key, shape=values.shape) * 1e-8
        perturbed = jnp.where(is_max, values + noise, -jnp.inf)
        return jnp.argmax(perturbed)
    
    def select_action(
        self,
        state: OptimisticQLearningState,
        obs: jnp.ndarray,
        key: jax.Array,
        is_training: bool
    ) -> jnp.ndarray:
        """Select greedy action with random tie-breaking."""
        state_idx = obs.astype(jnp.int32).squeeze()
        
        is_batched = obs.ndim > 1
        
        if is_batched:
            batch_size = obs.shape[0]
            keys = jax.random.split(key, batch_size)
            
            def select_single(i):
                s_idx = obs[i].astype(jnp.int32).squeeze()
                q_vals = state.q_table[s_idx]
                return self._break_ties_randomly(q_vals, keys[i])
            
            return jax.vmap(select_single)(jnp.arange(batch_size))
        else:
            q_values = state.q_table[state_idx]
            action = self._break_ties_randomly(q_values, key)
            return jnp.array([action])
    
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
                
                # Check if this state-action is unknown
                is_unknown = visit_counts[s, a] < self.known_threshold
                
                # Q-learning update
                q_current = q_tab[s, a]
                q_next_max = jnp.max(q_tab[s_next])
                target = r + d * self.discount * q_next_max
                
                # If unknown, push toward optimistic value
                target = jnp.where(is_unknown, self.optimistic_value, target)
                
                td_error = target - q_current
                new_q = q_current + self.learning_rate * td_error
                
                return s, a, new_q, jnp.abs(td_error)
            
            # Update all transitions in batch
            states, actions, new_qs, errors = jax.vmap(update_single)(jnp.arange(batch_size))
            
            # Apply updates
            new_q_tab = q_tab
            for i in range(batch_size):
                new_q_tab = new_q_tab.at[states[i], actions[i]].set(new_qs[i])
            
            max_change = jnp.max(errors)
            return new_q_tab, max_change
        
        # Run multiple iterations
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
    ) -> tuple[OptimisticQLearningState, float]:
        """Update Q-table with convergence iterations."""
        batch_size = batch.observation.shape[0]
        
        # Update visit counts
        new_visit_counts = state.visit_counts
        for i in range(batch_size):
            s = batch.observation[i].astype(jnp.int32).squeeze()
            a = batch.action[i].astype(jnp.int32).squeeze()
            new_visit_counts = new_visit_counts.at[s, a].add(1)
        
        # Update Q-table with convergence
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