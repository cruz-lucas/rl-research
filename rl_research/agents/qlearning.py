import jax
import jax.numpy as jnp
from typing import Literal
from flax import struct
from rl_research.policies import _select_greedy, _select_random, _select_epsilon_greedy, _select_ucb
from rl_research.buffers import Transition
import gin

@struct.dataclass
class QLearningState:
    """State for Q-learning agent."""
    q_table: jnp.ndarray
    visit_counts: jnp.ndarray
    eps_greedy_epsilon: float
    step: int

@gin.configurable
class QLearningAgent:
    """Q-Learning agent with epsilon-greedy, random, or UCB policies."""
    
    def __init__(
        self,
        num_states: int,
        num_actions: int,
        discount: float,
        initial_epsilon: float,
        step_size: float,
        initial_q_value: float,
        ucb_c: float,
        policy: Literal["epsilon_greedy", "random", "ucb"],
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.discount = discount
        self.step_size = step_size
        self.initial_q_value = initial_q_value

        self.initial_epsilon = initial_epsilon
        self.ucb_c = ucb_c
        self.policy = policy
    
    def initial_state(self) -> QLearningState:
        """Initialize Q-table and visit counts."""
        return QLearningState(
            q_table=jnp.full((self.num_states, self.num_actions), self.initial_q_value),
            visit_counts=jnp.zeros((self.num_states, self.num_actions)),
            eps_greedy_epsilon=self.initial_epsilon,
            step=0
        )
    
    def select_action(
        self,
        state: QLearningState,
        obs: jnp.ndarray,
        key: jax.Array,
        is_training: bool
    ) -> jnp.ndarray:
        """Select action based on policy."""
        q_values = state.q_table[obs]
                
        def eval_action():
            return _select_greedy(q_values, key)
    
        def train_action():
            def eps_greedy():
                return _select_epsilon_greedy(q_values, state.eps_greedy_epsilon, key)
            def ucb():
                visit_counts = state.visit_counts[obs]
                total_visits = jnp.sum(state.visit_counts[obs])
                return _select_ucb(q_values, visit_counts, total_visits, self.ucb_c, key)
            def random():
                return _select_random(q_values, key)
        
            return jax.lax.cond(
                self.policy == "epsilon_greedy",
                eps_greedy,
                lambda: jax.lax.cond(
                    self.policy == "ucb",
                    ucb,
                    random
                )
            )
    
        return jax.lax.cond(is_training, train_action, eval_action)
    
    def update(
        self,
        state: QLearningState,
        batch: Transition,
        batch_mask: jnp.ndarray | None = None
    ) -> tuple[QLearningState, jax.Array]:
        """Update Q-table with batch of transitions."""
        batch_size = batch.observation.shape[0]
        if batch_mask is None:
            batch_mask = jnp.ones((batch_size,), dtype=bool)
        batch_mask = batch_mask.astype(jnp.bool_)

        # jax.debug.print("step {t}: {s} - {a} - {r} - {sp}", s=batch.observation, a=batch.action, r=batch.reward, sp=batch.next_observation, t=state.step)
        
        def update_single(i):
            valid = batch_mask[i]
            s = batch.observation[i].astype(jnp.int32).squeeze()
            a = batch.action[i].astype(jnp.int32).squeeze()
            r = batch.reward[i]
            s_next = batch.next_observation[i].astype(jnp.int32).squeeze()
            
            q_current = state.q_table[s, a]
            q_next_max = jnp.max(state.q_table[s_next])
            target = r + self.discount * q_next_max
            td_error = target - q_current
            new_q = q_current + self.step_size * td_error
            
            masked_q = jnp.where(valid, new_q, q_current)
            masked_td = jnp.where(valid, td_error, 0.0)
            visit_inc = jnp.where(valid, 1, 0)
            return s, a, masked_q, masked_td, visit_inc
        
        states, actions, new_qs, td_errors, visit_incs = jax.vmap(update_single)(jnp.arange(batch_size))

        def apply_q_updates(table, i):
            return jax.lax.cond(
                visit_incs[i] > 0,
                lambda t: t.at[states[i], actions[i]].set(new_qs[i]),
                lambda t: t,
                table
            )

        new_q_table = jax.lax.fori_loop(0, batch_size, apply_q_updates, state.q_table)
        
        new_visit_counts = state.visit_counts
        for i in range(batch_size):
            new_visit_counts = new_visit_counts.at[states[i], actions[i]].add(visit_incs[i])
        
        new_state = state.replace(
            q_table=new_q_table,
            visit_counts=new_visit_counts,
            step=state.step + jnp.sum(visit_incs)
        )
        
        total_valid = jnp.maximum(jnp.sum(visit_incs), 1)
        loss = jnp.sum(jnp.abs(td_errors)) / total_valid
        return new_state, loss
