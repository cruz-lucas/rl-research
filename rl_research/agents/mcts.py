import jax
import jax.numpy as jnp
import jax.random as jrandom
from flax import struct
from rl_research.policies import _select_greedy
from rl_research.experiment import Transition
import gin


@struct.dataclass
class MCTSState:
    """State for MCTS agent."""
    transition_counts: jnp.ndarray
    reward_sums: jnp.ndarray
    visit_counts: jnp.ndarray
    step: int


@gin.configurable
class MCTSAgent:
    """Model-based Monte Carlo Tree Search with UCB tree policy and random rollouts."""
    
    def __init__(
        self,
        num_states: int,
        num_actions: int,
        discount: float = 0.9,
        num_simulations: int = 50,
        rollout_depth: int = 20,
        ucb_c: float = 1.4,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.discount = discount
        self.num_simulations = num_simulations
        self.rollout_depth = rollout_depth
        self.ucb_c = ucb_c
    
    def initial_state(self) -> MCTSState:
        """Initialize empty model statistics."""
        return MCTSState(
            transition_counts=jnp.zeros((self.num_states, self.num_actions, self.num_states), dtype=jnp.float32),
            reward_sums=jnp.zeros((self.num_states, self.num_actions), dtype=jnp.float32),
            visit_counts=jnp.zeros((self.num_states, self.num_actions), dtype=jnp.float32),
            step=0
        )

    def _sample_model(self, state: MCTSState, s: jnp.ndarray, a: jnp.ndarray, key: jax.Array):
        """Sample next state and reward from learned empirical model."""
        counts = state.transition_counts[s, a]
        total = jnp.sum(counts)
        safe_total = jnp.maximum(total, 1.0)
        
        uniform_probs = jnp.ones(self.num_states, dtype=jnp.float32) / self.num_states
        probs = jnp.where(total > 0, counts / safe_total, uniform_probs)
        
        next_state = jrandom.choice(key, self.num_states, p=probs)
        
        visits = state.visit_counts[s, a]
        avg_reward = state.reward_sums[s, a] / jnp.maximum(visits, 1)
        reward = jnp.where(visits > 0, avg_reward, 0.0)
        return next_state.astype(jnp.int32), reward

    def _run_rollout(self, state: MCTSState, start_state: jnp.ndarray, depth: jnp.ndarray, key: jax.Array):
        """Random rollout from start_state for remaining depth."""
        def body_fn(i, carry):
            s, k, ret, disc = carry
            
            def do_roll(carry_in):
                s_in, k_in, ret_in, disc_in = carry_in
                k_in, k_action, k_next = jrandom.split(k_in, 3)
                action = jrandom.randint(k_action, (), 0, self.num_actions)
                next_state, reward = self._sample_model(state, s_in, action, k_next)
                new_ret = ret_in + disc_in * reward
                new_disc = disc_in * self.discount
                return next_state, k_in, new_ret, new_disc
            
            return jax.lax.cond(
                i < depth,
                do_roll,
                lambda carry_in: carry_in,
                (s, k, ret, disc)
            )
        
        _, key_out, rollout_return, _ = jax.lax.fori_loop(
            0, self.rollout_depth, body_fn, (start_state, key, 0.0, 1.0)
        )
        return rollout_return, key_out

    def select_action(
        self,
        state: MCTSState,
        obs: jnp.ndarray,
        key: jax.Array,
        is_training: bool
    ) -> jnp.ndarray:
        """Plan with MCTS from current observation and pick highest-value action."""
        root_state = obs.astype(jnp.int32).squeeze()
        
        tree_visits = jnp.zeros((self.num_states, self.num_actions), dtype=jnp.float32)
        tree_values = jnp.zeros((self.num_states, self.num_actions), dtype=jnp.float32)
        
        def simulation(carry, _):
            tree_v, tree_val, key_sim = carry
            key_sim, key_sel, key_roll = jrandom.split(key_sim, 3)
            
            path_states = jnp.zeros((self.rollout_depth,), dtype=jnp.int32)
            path_actions = jnp.zeros((self.rollout_depth,), dtype=jnp.int32)
            path_rewards = jnp.zeros((self.rollout_depth,), dtype=jnp.float32)
            path_len = jnp.asarray(0, dtype=jnp.int32)
            expanded = jnp.asarray(False)
            current_state = root_state
            
            def selection_step(i, sel_carry):
                s, k_sel, p_states, p_actions, p_rewards, p_len, exp_flag = sel_carry
                
                def do_step(carry_in):
                    s_in, k_in, p_states_in, p_actions_in, p_rewards_in, p_len_in, exp_flag_in = carry_in
                    k_in, k_action, k_next = jrandom.split(k_in, 3)
                    
                    sa_visits = tree_v[s_in]
                    total_visits = jnp.sum(sa_visits)
                    mean_values = tree_val[s_in] / jnp.maximum(sa_visits, 1)
                    exploration = self.ucb_c * jnp.sqrt(jnp.log(total_visits + 1) / jnp.maximum(sa_visits, 1))
                    exploration = jnp.where(sa_visits == 0, jnp.inf, exploration)
                    ucb_scores = mean_values + exploration
                    
                    action = _select_greedy(ucb_scores, k_action)
                    next_state, reward = self._sample_model(state, s_in, action, k_next)
                    
                    p_states_in = p_states_in.at[p_len_in].set(s_in)
                    p_actions_in = p_actions_in.at[p_len_in].set(action)
                    p_rewards_in = p_rewards_in.at[p_len_in].set(reward)
                    
                    new_len = p_len_in + 1
                    new_expanded = jnp.logical_or(exp_flag_in, sa_visits[action] == 0)
                    return (
                        next_state,
                        k_in,
                        p_states_in,
                        p_actions_in,
                        p_rewards_in,
                        new_len,
                        new_expanded
                    )
                
                return jax.lax.cond(
                    jnp.logical_and(jnp.logical_not(exp_flag), p_len < self.rollout_depth),
                    do_step,
                    lambda carry_in: carry_in,
                    (s, k_sel, p_states, p_actions, p_rewards, p_len, exp_flag)
                )
            
            current_state, key_sel, path_states, path_actions, path_rewards, path_len, expanded = jax.lax.fori_loop(
                0, self.rollout_depth, selection_step,
                (current_state, key_sel, path_states, path_actions, path_rewards, path_len, expanded)
            )
            
            path_len = jnp.minimum(path_len, self.rollout_depth)
            remaining_depth = self.rollout_depth - path_len
            
            rollout_return, key_roll = self._run_rollout(state, current_state, remaining_depth, key_roll)
            
            def backup_step(i, backup_carry):
                value, t_visits, t_values = backup_carry
                
                def do_backup(carry_in):
                    value_in, t_visits_in, t_values_in = carry_in
                    idx = path_len - 1 - i
                    reward = path_rewards[idx]
                    new_value = reward + self.discount * value_in
                    s_idx = path_states[idx]
                    a_idx = path_actions[idx]
                    t_visits_in = t_visits_in.at[s_idx, a_idx].add(1)
                    t_values_in = t_values_in.at[s_idx, a_idx].add(new_value)
                    return new_value, t_visits_in, t_values_in
                
                return jax.lax.cond(
                    i < path_len,
                    do_backup,
                    lambda carry_in: carry_in,
                    (value, t_visits, t_values)
                )
            
            _, tree_v, tree_val = jax.lax.fori_loop(
                0, self.rollout_depth, backup_step, (rollout_return, tree_v, tree_val)
            )
            
            return (tree_v, tree_val, key_sim), None
        
        tree_visits, tree_values, key = jax.lax.scan(
            simulation,
            (tree_visits, tree_values, key),
            None,
            length=self.num_simulations
        )[0]
        
        action_values = tree_values[root_state] / jnp.maximum(tree_visits[root_state], 1)
        action = _select_greedy(action_values, key)
        return jnp.array([action])

    def update(
        self,
        state: MCTSState,
        batch: Transition
    ) -> tuple[MCTSState, jax.Array]:
        """Update empirical model with collected transitions."""
        batch_size = batch.observation.shape[0]
        
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
        
        new_state = state.replace(
            transition_counts=new_transition_counts,
            reward_sums=new_reward_sums,
            visit_counts=new_visit_counts,
            step=state.step + batch_size
        )
        
        loss = jnp.array(0.0)
        return new_state, loss
