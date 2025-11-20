import jax
import jax.numpy as jnp
from typing import Literal
from flax import struct
from rl_research.policies import _select_greedy, _select_random, _select_epsilon_greedy, _select_ucb
from rl_research.experiment import Transition
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
        batch: Transition
    ) -> tuple[QLearningState, jax.Array]:
        """Update Q-table with batch of transitions."""
        batch_size = batch.observation.shape[0]

        # jax.debug.print("step {t}: {s} - {a} - {r} - {sp}", s=batch.observation, a=batch.action, r=batch.reward, sp=batch.next_observation, t=state.step)
        
        def update_single(i):
            s = batch.observation[i].astype(jnp.int32).squeeze()
            a = batch.action[i].astype(jnp.int32).squeeze()
            r = batch.reward[i]
            s_next = batch.next_observation[i].astype(jnp.int32).squeeze()
            
            q_current = state.q_table[s, a]
            q_next_max = jnp.max(state.q_table[s_next])
            target = r + self.discount * q_next_max
            td_error = target - q_current
            new_q = q_current + self.step_size * td_error
            
            return s, a, new_q, td_error
        
        states, actions, new_qs, td_errors = jax.vmap(update_single)(jnp.arange(batch_size))

        new_q_table = state.q_table
        for i in range(batch_size):
            new_q_table = new_q_table.at[states[i], actions[i]].set(new_qs[i])
        
        new_visit_counts = state.visit_counts
        for i in range(batch_size):
            new_visit_counts = new_visit_counts.at[states[i], actions[i]].add(1)
        
        new_state = state.replace(
            q_table=new_q_table,
            visit_counts=new_visit_counts,
            step=state.step + batch_size
        )
        
        loss = jnp.mean(jnp.abs(td_errors))
        return new_state, loss

