import jax
import jax.numpy as jnp
from typing import Tuple, Any
from flax import struct
import gin
from rl_research.buffers import BufferState, Transition


@struct.dataclass
class TrainingState:
    """Complete training state."""
    agent_state: Any
    buffer_state: BufferState
    env_state: Any
    key: jax.Array
    step: int
    episode: int


@struct.dataclass
class History:
    """Training history for logging."""
    train_returns: jnp.ndarray
    train_discounted_returns: jnp.ndarray
    train_lengths: jnp.ndarray
    eval_returns: jnp.ndarray
    eval_discounted_returns: jnp.ndarray
    eval_lengths: jnp.ndarray
    train_losses: jnp.ndarray


def run_episode(
    agent_state: Any,
    env_state: Any,
    buffer_state: BufferState,
    key: jax.Array,
    agent,
    environment,
    batch_size: int,
    max_episode_steps: int,
    is_training: bool
) -> Tuple[Any, Any, BufferState, float, float, int, jnp.ndarray]:
    """Run a single episode."""
    
    def step_fn(carry, _):
        agent_st, env_st, buff_st, k, ep_return, ep_disc_return, discount_factor, ep_length, losses = carry
        
        k, k_action, k_update = jax.random.split(k, 3)
        
        obs = environment.env.observation(env_st)
        action = agent.select_action(agent_st, obs, k_action, is_training)
        next_env_st, next_obs, reward, terminal, _, _ = environment.step(env_st, action)
        
        def update_buffer_and_train(buff_st, agent_st):
            transition = Transition(
                observation=obs,
                action=action,
                reward=reward,
                discount=agent.discount,
                next_observation=next_obs
            )

            # new_buff_st = jax.lax.cond(buff_st.is_ready(batch_size), lambda: buff_st, lambda: buff_st.push(transition))  # test without pushing new samples into buffer
            new_buff_st = buff_st.push(transition)
            
            def train_step(states):
                b_st, a_st = states
                batch, batch_mask = b_st.sample(k_update, batch_size)
                new_a_st, loss = agent.update(a_st, batch, batch_mask)
                return b_st, new_a_st, loss
            
            def no_train(states):
                b_st, a_st = states
                return b_st, a_st, 0.0
            
            new_buff_st, new_agent_st, loss = jax.lax.cond(
                new_buff_st.is_ready(batch_size),
                train_step,
                no_train,
                (new_buff_st, agent_st)
            )
            return new_buff_st, new_agent_st, loss
        
        def no_update(buff_st, agent_st):
            return buff_st, agent_st, 0.0

        new_buff_st, new_agent_st, loss = jax.lax.cond(
            is_training,
            update_buffer_and_train,
            no_update,
            buff_st,
            agent_st
        )
        
        new_ep_return = ep_return + reward
        new_ep_disc_return = ep_disc_return + (discount_factor * reward)
        new_discount_factor = discount_factor * agent.discount
        new_ep_length = ep_length + 1
        new_losses = losses.at[ep_length].set(loss)
        
        new_carry = (
            new_agent_st,
            next_env_st,
            new_buff_st,
            k,
            new_ep_return,
            new_ep_disc_return,
            new_discount_factor,
            new_ep_length,
            new_losses
        )
        
        return new_carry, terminal
    
    ep_return = 0.0
    ep_disc_return = 0.0
    discount_factor = 1.0
    ep_length = 0
    losses = jnp.zeros(max_episode_steps)
    
    init_carry = (agent_state, env_state, buffer_state, key, ep_return, ep_disc_return, discount_factor, ep_length, losses)
    
    final_carry, dones = jax.lax.scan(step_fn, init_carry, None, length=max_episode_steps)
    
    agent_st, env_st, buff_st, _, ep_return, ep_disc_return, _, ep_length, losses = final_carry
    
    return agent_st, env_st, buff_st, ep_return, ep_disc_return, ep_length, losses

@gin.configurable
# @partial(jax.jit, static_argnames=['agent', 'environment', 'batch_size', 'max_episode_steps',
#                                      'train_episodes', 'evaluate_every', 'eval_episodes'])
def run_loop(
    agent,
    environment,
    buffer_state: BufferState,
    agent_state: Any,
    seed: int,
    batch_size: int,
    max_episode_steps: int,
    train_episodes: int,
    evaluate_every: int,
    eval_episodes: int
) -> Tuple[History, Any]:
    """
    Main training loop - fully jitted.
    
    Args:
        agent: Agent with methods: select_action(state, obs, key, train), update(state, batch)
        environment: Environment with methods: reset(key), step(state, action), 
                    get_observation(state), max_episode_steps
        buffer_state: Initial buffer state
        agent_state: Initial agent state
        seed: Random seed
        batch_size: Batch size for training
        train_episodes: Total number of training episodes
        evaluate_every: Evaluate every N episodes
        eval_episodes: Number of evaluation episodes
    
    Returns:
        Tuple of:
            History with training and evaluation metrics
            Agent states captured after each training episode
    """
    
    key = jax.random.PRNGKey(seed)
    
    def episode_fn(carry, episode_idx):
        agent_st, buff_st, k = carry
        
        k, k_reset, k_episode, k_eval = jax.random.split(k, 4)
        
        env_st, _ = environment.reset(k_reset)
        agent_st, env_st, buff_st, train_rets, train_disc_rets, train_lens, losses = run_episode(
            agent_st, env_st, buff_st, k_episode,
            agent, environment, batch_size, max_episode_steps, is_training=True
        )
        
        train_loss = jnp.mean(losses)
        
        def run_eval(k_eval_in):
            def eval_episode_fn(eval_carry, eval_idx):
                k_e = eval_carry
                k_e, k_e_reset, k_e_episode = jax.random.split(k_e, 3)
                
                eval_env_st, _ = environment.reset(k_e_reset)
                _, _, _, eval_return, eval_disc_return, eval_length, _ = run_episode(
                    agent_st, eval_env_st, buff_st, k_e_episode,
                    agent, environment, batch_size, max_episode_steps, is_training=False
                )
                return k_e, (eval_return, eval_disc_return, eval_length)
            
            _, (eval_rets, eval_disc_rets, eval_lens) = jax.lax.scan(
                eval_episode_fn, k_eval_in, None, length=eval_episodes
            )
            return eval_rets, eval_disc_rets, eval_lens
        
        def no_eval(k_eval_in):
            return (jnp.zeros(eval_episodes), 
                    jnp.zeros(eval_episodes), 
                    jnp.zeros(eval_episodes, dtype=jnp.int32))
        
        should_eval = (episode_idx + 1) % evaluate_every == 0
        eval_rets, eval_disc_rets, eval_lens = jax.lax.cond(
            should_eval,
            run_eval,
            no_eval,
            k_eval
        )
        
        eval_returns_new = jnp.mean(eval_rets)
        eval_discounted_returns_new = jnp.mean(eval_disc_rets)
        eval_lengths_new = jnp.mean(eval_lens)
        
        new_carry = (agent_st, buff_st, k)
        out = (train_rets, train_disc_rets, train_lens, train_loss, 
               eval_returns_new, eval_discounted_returns_new, eval_lengths_new, agent_st)
        
        return new_carry, out
    
    init_carry = (agent_state, buffer_state, key)
    (final_agent_state, _, _), (train_rets, train_disc_rets, train_lens, train_losses, 
        eval_rets, eval_disc_rets, eval_lens, agent_states) = jax.lax.scan(
        episode_fn, init_carry, jnp.arange(train_episodes)
    )

    # jax.debug.print("{x}", x=final_agent_state.q_table)
    
    history = History(
        train_returns=train_rets,
        train_discounted_returns=train_disc_rets,
        train_lengths=train_lens,
        eval_returns=eval_rets,
        eval_discounted_returns=eval_disc_rets,
        eval_lengths=eval_lens,
        train_losses=train_losses
    )
    
    return history, agent_states
