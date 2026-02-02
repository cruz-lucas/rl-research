from functools import partial
from typing import Any, Tuple

import gin
import jax
import jax.numpy as jnp
from flax import struct

from rl_research.buffers import BufferState, Transition


class TrainingState(struct.PyTreeNode):
    agent_state: Any
    env_state: Any
    env_obs: jax.Array
    buffer_state: BufferState
    key: jax.Array
    episode_return: jax.Array
    episode_discounted_return: jax.Array
    discount_factor: jax.Array
    episode_length: jax.Array
    loss: jax.Array
    global_step: jax.Array
    episode_idx: jax.Array
    done: jax.Array


class TrainingConfig(struct.PyTreeNode):
    minibatch_size: int
    num_minibatches: int
    max_episode_steps: int
    update_frequency: int
    warmup_steps: int


class History(struct.PyTreeNode):
    train_returns: jax.Array
    train_discounted_returns: jax.Array
    train_lengths: jax.Array
    train_losses: jax.Array
    episode_idx: jax.Array
    global_steps: jax.Array
    dones: jax.Array


@gin.configurable
@partial(
    jax.jit,
    static_argnames=[
        "agent",
        "environment",
        "minibatch_size",
        "max_episode_steps",
        "train_episodes",
        "train_steps",
        "evaluate_every",
        "eval_episodes",
        "update_frequency",
        "num_minibatches",
        "warmup_steps",
        "use_steps",
    ],
)
def run_loop(
    agent,
    environment,
    buffer_state: BufferState,
    agent_state: Any,
    seed: int,
    minibatch_size: int,
    max_episode_steps: int,
    train_episodes: int,
    evaluate_every: int,
    eval_episodes: int,
    update_frequency: int,
    num_minibatches: int,
    warmup_steps: int,
    train_steps: int = 0,
    use_steps: bool = False,
    is_training: bool = True,
) -> History:

    config = TrainingConfig(
        minibatch_size=minibatch_size,
        num_minibatches=num_minibatches,
        max_episode_steps=max_episode_steps,
        update_frequency=update_frequency,
        warmup_steps=warmup_steps,
    )

    num_iters = train_steps if use_steps else train_episodes * max_episode_steps

    key = jax.random.PRNGKey(seed)
    key, k_reset = jax.random.split(key)
    env_state, env_obs = environment.reset(k_reset)

    init_train_state = TrainingState(
        agent_state=agent_state,
        env_state=env_state,
        env_obs=env_obs,
        buffer_state=buffer_state,
        key=key,
        episode_return=jnp.asarray(0.0, jnp.float32),
        episode_discounted_return=jnp.asarray(0.0, jnp.float32),
        discount_factor=jnp.asarray(1.0, jnp.float32),
        episode_length=jnp.asarray(0, jnp.int32),
        loss=jnp.asarray(0.0, jnp.float32),
        global_step=jnp.asarray(0, jnp.int32),
        episode_idx=jnp.asarray(0, jnp.int32),
        done=jnp.asarray(False),
    )

    def run_episodes(train_state: TrainingState, _) -> Tuple[TrainingState, History]:
        next_key, action_key = jax.random.split(train_state.key)
        obs = train_state.env_obs

        action = agent.select_action(train_state.agent_state, obs, action_key, is_training)
        next_env_st, next_obs, reward, terminal, truncation, info = environment.step(train_state.env_state, action)

        transition = Transition(
            observation=obs,
            action=action,
            reward=reward,
            discount=agent.discount,
            next_observation=next_obs,
            terminal=terminal,
        )

        bootstrap_value = jnp.where(
            transition.terminal, 
            jnp.asarray(0.0, dtype=jnp.float32), 
            agent.bootstrap_value(train_state.agent_state, transition.next_observation)
        )
        new_buff_st = train_state.buffer_state.push(transition, bootstrap_value=bootstrap_value)

        train_state = train_state.replace(
            buffer_state=new_buff_st,
            key=next_key,
            env_state=next_env_st,
            env_obs=next_obs,
            episode_return=train_state.episode_return + reward,
            episode_discounted_return=train_state.episode_discounted_return + (train_state.discount_factor * reward),
            discount_factor=train_state.discount_factor * agent.discount,
            episode_length=train_state.episode_length + 1,
            global_step=train_state.global_step + 1,
            done=jnp.logical_or(terminal, truncation)
        )

        def update_agent(ts: TrainingState):
            def single_update(carry, _):
                agent_carry, key_carry = carry
                key_carry, subkey = jax.random.split(key_carry)
                batch = ts.buffer_state.sample(subkey, config.minibatch_size)
                new_agent_state, loss_val = agent.update(agent_carry, batch)
                return (new_agent_state, key_carry), loss_val

            (agent_st, new_key), losses = jax.lax.scan(
                single_update,
                (ts.agent_state, ts.key),
                None,
                length=config.num_minibatches,
            )

            return ts.replace(
                agent_state=agent_st,
                key=new_key,
                loss=jnp.mean(losses)
            )

        has_updates = train_state.global_step % config.update_frequency == 0
        should_train = jnp.logical_and(
            train_state.buffer_state.is_ready(config.minibatch_size), train_state.global_step >= config.warmup_steps
        )
        should_train = jnp.logical_and(should_train, jnp.asarray(has_updates))
        must_train = jnp.logical_and(is_training, should_train)

        train_state = jax.lax.cond(
            must_train,
            lambda ts: update_agent(ts),
            lambda ts: ts,
            train_state,
        )

        def reset_fn(ts: TrainingState):
            key, k_reset = jax.random.split(ts.key)
            env_state, env_obs = environment.reset(k_reset)

            return ts.replace(
                key=key,
                env_state=env_state,
                env_obs=env_obs,
                episode_return=0.0,
                episode_discounted_return=0.0,
                discount_factor=1.0,
                episode_length=0,
                loss=0.0,
                done=False,
                episode_idx=ts.episode_idx+1
            )

        must_reset = jnp.logical_or(train_state.episode_length >= config.max_episode_steps, train_state.done)
        output = History(
            train_returns=train_state.episode_return,
            train_discounted_returns=train_state.episode_discounted_return,
            train_lengths=train_state.episode_length,
            train_losses=train_state.loss,
            episode_idx=train_state.episode_idx,
            global_steps=train_state.global_step,
            dones=train_state.done
        )

        return jax.lax.cond(
            must_reset,
            lambda ts: reset_fn(ts),
            lambda ts: ts,
            train_state
        ), output


    final_carry, history = jax.lax.scan(run_episodes, init_train_state, None, length=num_iters)
    return history


@partial(
    jax.jit,
    static_argnames=("agent", "environment", "max_episode_steps"),
)
def run_eval(
    agent,
    environment,
    agent_state,
    key,
    max_episode_steps,
    num_episodes,
):
    def run_one_episode(carry, _):
        key = carry
        key, k_reset = jax.random.split(key)
        env_state, obs = environment.reset(k_reset)

        def step_fn(step_carry, _):
            env_state, obs, ret, done = step_carry
            action = agent.select_action(agent_state, obs, None, is_training=False)
            env_state, obs, reward, terminal, trunc, _ = environment.step(env_state, action)
            done = jnp.logical_or(terminal, trunc)
            ret = ret + reward
            return (env_state, obs, ret, done), None

        (env_state, obs, ret, done), _ = jax.lax.scan(
            step_fn,
            (env_state, obs, 0.0, False),
            None,
            length=max_episode_steps,
        )

        return key, ret

    key, returns = jax.lax.scan(
        run_one_episode,
        key,
        None,
        length=num_episodes,
    )

    return returns

