@struct.dataclass
class RMaxState:
    """State for R-Max agent."""
    q_table: jnp.ndarray  # [num_states, num_actions]
    transition_counts: jnp.ndarray  # [num_states, num_actions, num_states]
    reward_sums: jnp.ndarray  # [num_states, num_actions]
    visit_counts: jnp.ndarray  # [num_states, num_actions]
    step: int


class RMaxAgent:
    """R-Max agent with known model."""
    
    def __init__(
        self,
        num_states: int,
        num_actions: int,
        r_max: float = 1.0,
        discount: float = 0.99,
        known_threshold: int = 1,
        vi_iterations: int = 100
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.r_max = r_max
        self.discount = discount
        self.known_threshold = known_threshold
        self.vi_iterations = vi_iterations
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
    
    def _break_ties_randomly(self, values: jnp.ndarray, key: jax.Array) -> int:
        """Break ties randomly among maximum values."""
        max_val = jnp.max(values)
        is_max = values == max_val
        noise = jax.random.uniform(key, shape=values.shape) * 1e-8
        perturbed = jnp.where(is_max, values + noise, -jnp.inf)
        return jnp.argmax(perturbed)
    
    def select_action(
        self,
        state: RMaxState,
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
    
    def _value_iteration_step(self, q_table, is_known, rewards, transitions):
        """Single step of value iteration."""
        # For each state-action, compute expected value
        v_table = jnp.max(q_table, axis=1)  # [num_states]
        
        def compute_q(s, a):
            # If known, use model
            expected_next_value = jnp.sum(transitions[s, a] * v_table)
            q_val = rewards[s, a] + self.discount * expected_next_value
            
            # If unknown, use optimistic value
            q_val = jnp.where(is_known[s, a], q_val, self.optimistic_value)
            return q_val
        
        # Vectorized Q-value computation
        new_q = jax.vmap(
            lambda s: jax.vmap(lambda a: compute_q(s, a))(jnp.arange(self.num_actions))
        )(jnp.arange(self.num_states))
        
        return new_q
    
    def update(
        self,
        state: RMaxState,
        batch: Transition
    ) -> tuple[RMaxState, float]:
        """Update model and recompute Q-values."""
        batch_size = batch.observation.shape[0]
        
        # Update model statistics
        new_transition_counts = state.transition_counts
        new_reward_sums = state.reward_sums
        new_visit_counts = state.visit_counts
        
        for i in range(batch_size):
            s = batch.observation[i].astype(jnp.int32).squeeze()
            a = batch.action[i].astype(jnp.int32).squeeze()
            r = batch.reward[i]
            s_next = batch.next_observation[i].astype(jnp.int32).squeeze()
            
            new_transition_counts = new_transition_counts.at[s, a, s_next].add(1)
            new_reward_sums = new_reward_sums.at[s, a].add(r)
            new_visit_counts = new_visit_counts.at[s, a].add(1)
        
        # Determine which state-actions are known
        is_known = new_visit_counts >= self.known_threshold
        
        # Compute model (average rewards and transition probabilities)
        safe_counts = jnp.maximum(new_visit_counts, 1)
        avg_rewards = new_reward_sums / safe_counts
        transition_probs = new_transition_counts / safe_counts[:, :, None]
        
        # Run value iteration
        q_table = state.q_table
        for _ in range(self.vi_iterations):
            q_table = self._value_iteration_step(q_table, is_known, avg_rewards, transition_probs)
        
        new_state = state.replace(
            q_table=q_table,
            transition_counts=new_transition_counts,
            reward_sums=new_reward_sums,
            visit_counts=new_visit_counts,
            step=state.step + batch_size
        )
        
        # Loss is change in Q-values
        loss = jnp.mean(jnp.abs(q_table - state.q_table))
        return new_state, loss
