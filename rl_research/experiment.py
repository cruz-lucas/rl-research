from functools import partial
from typing import Any, Tuple, Type
import gin
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import struct, nnx
import numpy as np

from rl_research.agents import *
from rl_research.buffers import BaseBuffer, ReplayBuffer, Transition, BufferState
from rl_research.environments import *
from rl_research.utils import RecordWriter
import orbax.checkpoint as ocp

# TODO: include the path in the config and make it more general to support different agents and checkpoints
path = Path("./tmp/ckpt/").resolve()
# path = ocp.test_utils.erase_and_create_empty(path)
checkpointer = ocp.StandardCheckpointer()

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


@gin.configurable
class TrainingConfig(struct.PyTreeNode):
    minibatch_size: int
    num_minibatches: int
    max_episode_steps: int
    update_frequency: int
    warmup_steps: int
    train_episodes: int
    evaluate_every: int
    eval_episodes: int
    train_steps: int = 0
    use_steps: bool = False
    checkpoint_freq: int | None = None


class History(struct.PyTreeNode):
    train_returns: jax.Array
    train_discounted_returns: jax.Array
    train_lengths: jax.Array
    train_losses: jax.Array
    episode_idx: jax.Array
    global_steps: jax.Array
    dones: jax.Array

@gin.configurable
def run_loop(
    agent_cls: Type[BaseAgent],
    buffer_cls: Type[BaseBuffer],
    env_cls: Type[BaseJaxEnv],
    seed: int,
) -> None:
    config = TrainingConfig()
    environment = env_cls()

    obs_shape = environment.env.observation_space.shape
    n_states = environment.env.observation_space.n if obs_shape in [(), (1,)] else int(np.prod(np.array(obs_shape)))
    n_actions = environment.env.action_space.n
    agent = agent_cls(
        num_states=n_states,
        num_actions=n_actions,
    )

    buffer_init_kwargs = {}
    buffer = buffer_cls(**buffer_init_kwargs)

    fresh_agent_state = agent.initial_state()
    empty_buffer_state = buffer.initial_state()

    assert config.use_steps or config.train_episodes > 0, "Must set train_episodes or use_steps=True"
    num_iters = config.train_steps if config.use_steps else config.train_episodes * config.max_episode_steps

    key = jax.random.PRNGKey(seed)
    key, k_reset = jax.random.split(key)
    env_state, env_obs = environment.reset(k_reset)

    train_state = TrainingState(
        agent_state=fresh_agent_state,
        env_state=env_state,
        env_obs=env_obs,
        buffer_state=empty_buffer_state,
        key=key,
        episode_return=jnp.asarray(0.0, jnp.float32),
        episode_discounted_return=jnp.asarray(0.0, jnp.float32),
        discount_factor=jnp.asarray(1.0, jnp.float32),
        episode_length=jnp.asarray(0, jnp.int32),
        loss=jnp.asarray(0.0, jnp.float32),
        global_step=jnp.asarray(0, jnp.int32),
        episode_idx=jnp.asarray(0, jnp.int32),
        done=jnp.asarray(False)
    )

    # @nnx.jit
    def train_step(train_state: TrainingState) -> Tuple[TrainingState, History]:
        next_key, action_key = jax.random.split(train_state.key)
        obs = train_state.env_obs

        new_agent_state, action = agent.select_action(train_state.agent_state, obs, action_key, is_training=True)
        next_env_st, next_obs, reward, terminal, truncation, info = environment.step(train_state.env_state, action)

        # TODO: fix this later, not using MC returns now. Also, wouldn't it be better to query this with the current target net?
        bootstrap_value = 0.0
        # bootstrap_value = jax.lax.cond(
        #     terminal, 
        #     lambda: jnp.asarray(0.0, dtype=jnp.float32), 
        #     lambda: agent.bootstrap_value(train_state.agent_state, next_obs)
        # )
        new_buff_st = train_state.buffer_state.push(
            observation=obs,
            action=action,
            reward=reward,
            discount=agent.discount,
            next_observation=next_obs,
            terminal=terminal,
            bootstrap_value=bootstrap_value
        )

        train_state = train_state.replace(
            agent_state=new_agent_state,
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
            def update_minibatch(carry):
                agent_carry, key_carry = carry
                key_carry, subkey = jax.random.split(key_carry)
                batch = ts.buffer_state.sample(subkey, config.minibatch_size)
                new_agent_state, loss_val = agent.update(agent_carry, batch)
                return (new_agent_state, key_carry), loss_val

            (agent_st, new_key), losses = nnx.scan(
                update_minibatch,
                in_axes=nnx.Carry,
                out_axes=(nnx.Carry, 0),
                length=config.num_minibatches,
            )((ts.agent_state, ts.key))

            return ts.replace(
                agent_state=agent_st,
                key=new_key,
                loss=jnp.mean(losses)
            )

        has_updates = train_state.global_step % config.update_frequency == 0
        should_train = jnp.logical_and(
            train_state.buffer_state.is_ready(config.minibatch_size),
            train_state.global_step >= config.warmup_steps
        )
        must_train = jnp.logical_and(should_train, jnp.asarray(has_updates))

        train_state = nnx.cond(
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
                episode_return=jnp.asarray(0.0, jnp.float32),
                episode_discounted_return=jnp.asarray(0.0, jnp.float32),
                discount_factor=jnp.asarray(1.0, jnp.float32),
                episode_length=jnp.asarray(0, jnp.int32),
                loss=jnp.asarray(0.0, jnp.float32),
                done=jnp.asarray(False),
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
            dones=must_reset,
        )

        return nnx.cond(
            must_reset,
            lambda ts: reset_fn(ts),
            lambda ts: ts,
            train_state
        ), output

    record_writer = RecordWriter()
    run_bindings = gin.get_bindings("run_loop")
    agent_cls = run_bindings.get("agent_cls", BaseAgent)
    agent_cls_name = agent_cls.__name__
    
    train_state, history = nnx.scan(
        train_step,
        in_axes=nnx.Carry,
        out_axes=(nnx.Carry, 0),
        length=num_iters
    )(train_state)

    done_indices = jnp.where(history.dones)[0]
    for idx in done_indices:
        record_writer({"step": history.global_steps[idx]-1, "metrics": jax.tree.map(lambda x: x[idx], history)})
    
    # TODO: solve checkpointing
    # for step in range(num_iters):
    #     train_state, history = train_step(train_state)

    #     if history.dones:
    #         record_writer({"step": step, "metrics": history})

    #     if (config.checkpoint_freq is not None) and (step % config.checkpoint_freq == 0):
    #         if hasattr(train_state.agent_state, 'online_network'):
    #             _, state = nnx.split(train_state.agent_state.online_network)
    #             checkpointer.save(path / agent_cls_name / f"checkpoint_{step}", state)

    record_writer.flush_summary()


