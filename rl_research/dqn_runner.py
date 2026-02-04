import gin
import jax
import jax.numpy as jnp
from typing import Tuple, List

from rl_research.buffers import Transition
from rl_research.experiment import History


@gin.configurable
def run_dqn_training(agent, environment, buffer, seed: int):
    """Non-jitted training loop for DQN to avoid putting nnx Modules inside jitted code.

    This returns a `History` object compatible with the rest of the codebase.
    """
    bindings = gin.get_bindings("run_loop")
    minibatch_size = bindings["minibatch_size"]
    num_minibatches = bindings["num_minibatches"]
    max_episode_steps = bindings["max_episode_steps"]
    update_frequency = bindings["update_frequency"]
    warmup_steps = bindings["warmup_steps"]
    train_episodes = bindings.get("train_episodes", 0)
    train_steps = bindings.get("train_steps", 0)
    use_steps = bindings.get("use_steps", False)

    agent_state = agent.initial_state()
    buffer_state = buffer.initial_state()

    num_iters = train_steps if use_steps else train_episodes * max_episode_steps

    key = jax.random.PRNGKey(seed)
    key, k_reset = jax.random.split(key)
    env_state, env_obs = environment.reset(k_reset)

    # running state
    episode_return = 0.0
    episode_discounted_return = 0.0
    discount_factor = 1.0
    episode_length = 0
    loss_val = 0.0
    global_step = 0
    episode_idx = 0
    done = False

    # history buffers
    train_returns: List[float] = []
    train_discounted_returns: List[float] = []
    train_lengths: List[int] = []
    train_losses: List[float] = []
    episode_idxs: List[int] = []
    global_steps: List[int] = []
    dones: List[bool] = []

    for _ in range(int(num_iters)):
        key, action_key = jax.random.split(key)

        action = agent.select_action(agent_state, env_obs, action_key, is_training=True)
        next_env_st, next_obs, reward, terminal, truncation, info = environment.step(env_state, action)

        transition = Transition(
            observation=env_obs,
            action=action,
            reward=reward,
            discount=agent.discount,
            next_observation=next_obs,
            terminal=terminal,
        )

        if transition.terminal:
            bootstrap_value = jnp.asarray(0.0, dtype=jnp.float32)
        else:
            bootstrap_value = agent.bootstrap_value(agent_state, transition.next_observation)

        buffer_state = buffer_state.push(transition, bootstrap_value=bootstrap_value)

        episode_return += float(reward)
        episode_discounted_return += float(discount_factor * reward)
        discount_factor *= float(agent.discount)
        episode_length += 1
        global_step += 1
        done = bool(terminal or truncation)

        # possibly update
        should_train = buffer_state.is_ready(minibatch_size) and (global_step >= warmup_steps)
        has_updates = global_step % update_frequency == 0
        must_train = should_train and has_updates

        if must_train:
            losses = []
            for _ in range(int(num_minibatches)):
                key, subkey = jax.random.split(key)
                batch = buffer_state.sample(subkey, minibatch_size)
                agent_state, l = agent.update(agent_state, batch)
                losses.append(float(l))
            loss_val = float(jnp.mean(jnp.array(losses)))
        else:
            loss_val = 0.0

        # append metrics
        train_returns.append(episode_return)
        train_discounted_returns.append(episode_discounted_return)
        train_lengths.append(episode_length)
        train_losses.append(loss_val)
        episode_idxs.append(episode_idx)
        global_steps.append(global_step)
        dones.append(done)

        # reset if episode ends or truncation or max steps reached
        if episode_length >= max_episode_steps or done:
            key, k_reset = jax.random.split(key)
            env_state, env_obs = environment.reset(k_reset)

            episode_return = 0.0
            episode_discounted_return = 0.0
            discount_factor = 1.0
            episode_length = 0
            loss_val = 0.0
            done = False
            episode_idx += 1
        else:
            env_state = next_env_st
            env_obs = next_obs

    history = History(
        train_returns=jnp.array(train_returns),
        train_discounted_returns=jnp.array(train_discounted_returns),
        train_lengths=jnp.array(train_lengths),
        train_losses=jnp.array(train_losses),
        episode_idx=jnp.array(episode_idxs),
        global_steps=jnp.array(global_steps),
        dones=jnp.array(dones),
    )

    return history
