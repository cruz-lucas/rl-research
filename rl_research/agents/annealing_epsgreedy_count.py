import gin
import jax
import jax.numpy as jnp
from flax import struct
from rl_research.buffers import Transition
from rl_research.agents.base import BaseAgent
import distrax

class AnnealingEpsGreedyCountState(struct.PyTreeNode):
    q_table: jnp.ndarray
    visit_counts: jnp.ndarray
    step: int

@gin.configurable
class AnnealingEpsGreedyCountAgent(BaseAgent):
    def __init__(
            self,
            num_states,
            num_actions,
            discount=0.99,
            initial_epsilon=1.0,
            final_epsilon=0.01,
            anneal_steps=100000,
            step_size=0.1,
            intrinsic_reward_scale=0.1,
            initial_q_value=0.0
            ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.discount = discount
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.anneal_steps = anneal_steps
        self.step_size = step_size
        self.intrinsic_reward_scale = intrinsic_reward_scale
        self.initial_q_value = initial_q_value

    def initial_state(self):
        return AnnealingEpsGreedyCountState(
            q_table=jnp.full((self.num_states, self.num_actions), self.initial_q_value),
            visit_counts=jnp.zeros((self.num_states, self.num_actions)),
            step=0
        )

    def select_action(self, state: AnnealingEpsGreedyCountState, obs: jax.Array, key: jax.Array, is_training: bool=True):
        q_values = state.q_table[obs]

        def greedy():
            action_dist = distrax.Greedy(q_values)
            return action_dist.sample(seed=key)

        def eps_greedy():
            frac = jnp.clip(state.step / max(1, self.anneal_steps), 0.0, 1.0)
            eps = self.initial_epsilon + frac * (self.final_epsilon - self.initial_epsilon)

            action_dist = distrax.EpsilonGreedy(q_values, epsilon=eps)
            return action_dist.sample(seed=key)

        return jax.lax.cond(is_training, eps_greedy, greedy)

    def update(self, state: AnnealingEpsGreedyCountState, batch: Transition):
        batch_size = batch.observation.shape[0]

        # new_visit_counts = state.visit_counts.at[
        #     batch.observation.astype(jnp.int32), batch.action.astype(jnp.int32)].add(1)
        new_visit_counts = state.visit_counts

        def update_single(carry, i):
            q_table = carry

            s = batch.observation[i].astype(jnp.int32).squeeze()
            a = batch.action[i].astype(jnp.int32).squeeze()
            r = batch.reward[i]
            s_next = batch.next_observation[i].astype(jnp.int32).squeeze()
            terminal = batch.terminal[i]

            intrinsic_reward = self.intrinsic_reward_scale / jnp.sqrt(new_visit_counts[s, a])
            total_reward = r + intrinsic_reward

            q_next = jnp.max(q_table[s_next], axis=-1)
            target = total_reward + self.discount * q_next * (1 - terminal)
            q_old = q_table[s, a]
            td_error = (target - q_old)
            q_new = q_old + self.step_size * td_error
            new_q_table = state.q_table.at[s, a].set(q_new[0])

            loss_val = jnp.abs(td_error)
            return new_q_table, loss_val

        epochs = 200
        new_q_table, losses = jax.lax.scan(
            update_single, state.q_table, jnp.tile(jnp.arange(batch_size), epochs)
        )

        mean_loss = jnp.mean(losses[-batch_size:])

        new_state = state.replace(
            q_table=new_q_table,
            # visit_counts=new_visit_counts,
            # step=step
        )
        return new_state, mean_loss

    def bootstrap_value(self, state: AnnealingEpsGreedyCountState, next_observation: jax.Array):
        s_next = next_observation.astype(jnp.int32).squeeze()
        return jnp.max(state.q_table[s_next])
