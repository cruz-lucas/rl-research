"""Pure JAX training loop."""

from __future__ import annotations

from functools import partial
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
import jax.random as jrng
import jax.tree_util as jtu
from gymnasium.envs.functional_jax_env import FunctionalJaxEnv

from rl_research.agents.base import AgentState, TabularAgent, AgentParams


class EpisodeStep(NamedTuple):
    """Structure to carry out the outcome of a step."""

    observation: jax.Array
    action: jax.Array
    next_observation: jax.Array
    reward: jax.Array
    terminal: jax.Array


class Episode(NamedTuple):
    """Structure to carry out the outcome of an episode."""

    observations: jax.Array
    actions: jax.Array
    next_observations: jax.Array
    rewards: jax.Array
    terminals: jax.Array


@partial(jax.jit, static_argnames=("env", "agent", "num_steps"))
def rollout(
    env: FunctionalJaxEnv,
    agent: TabularAgent,
    agent_state: AgentState,
    num_steps: int,
    rng: jax.Array,
    train: bool = True,
) -> Tuple[Episode, AgentState]:
    env_state, obs = env.reset(rng=rng)

    def step_fn(carry, _):
        env_state, obs, agent_state = carry
        action, agent_state, _ = agent.select_action(agent_state, obs)
        next_env_state, next_obs, reward, terminal, _, _ = env.step(state=env_state, action=action)

        def do_update(agent_state: AgentState) -> AgentState:
            new_state, _ = agent.update(
                agent_state=agent_state,
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                terminated=terminal,
            )
            return new_state

        agent_state = jax.lax.cond(train, do_update, lambda s: s, agent_state)
        return (next_env_state, next_obs, agent_state), EpisodeStep(
            obs, action, next_obs, reward, terminal
        )

    (_, _, final_agent_state), traj = jax.lax.scan(
        step_fn, (env_state, obs, agent_state), jnp.arange(num_steps, dtype=jnp.int32)
    )

    episode = Episode(
        observations=traj.observation,
        actions=traj.action,
        next_observations=traj.next_observation,
        rewards=traj.reward,
        terminals=traj.terminal,
    )
    return episode, final_agent_state

def run_experiment(
    env: FunctionalJaxEnv,
    agent: TabularAgent,
    agent_params: AgentParams,
    rng: jax.Array,
    num_seeds: int,
    total_train_episodes: int,
    episode_length: int,
    eval_every: int,
    num_eval_episodes: int,
):
    """Trains for `rng`, running `num_eval_episodes` eval eps every `eval_every`."""
    keys = jrng.split(rng, num_seeds)

    def single_seed(key):
        next_rng, agent_key = jrng.split(key, 2)
        agent_state = agent.init(agent_key)

        agent_states: list[AgentState] = []
        training_episodes: list[Episode] = []
        eval_episodes: list[Episode] = []

        for ep in range(1, total_train_episodes + 1):
            next_rng, rollout_key = jrng.split(next_rng, 2)
            agent_state = agent.train(agent_state)
            train_ep, agent_state = rollout(
                env, agent, agent_state, episode_length, rng=rollout_key, train=True
            )
            training_episodes.append(train_ep)

            if eval_every > 0 and ep % eval_every == 0 and num_eval_episodes > 0:
                agent_state = agent.eval(agent_state)
                for _ in range(num_eval_episodes):
                    next_rng, rollout_key = jrng.split(next_rng, 2)
                    eval_ep, _ = rollout(
                        env, agent, agent_state, episode_length, rng=rollout_key, train=False
                    )
                    eval_episodes.append(eval_ep)

        def _stack_episodes(items):
            return Episode(*jtu.tree_map(lambda *xs: jnp.stack(xs), *items))

        train_stack = _stack_episodes(training_episodes) if training_episodes else None
        eval_stack = _stack_episodes(eval_episodes) if eval_episodes else None

        return train_stack, eval_stack, agent_state
    return jax.vmap(single_seed)(keys)

    # train_stack, eval_stack, agent_state = single_seed(rng)
    # print(agent_state.q_values)
    # print(agent_state.sa_counts)
    # return Episode(*jtu.tree_map(lambda *xs: jnp.stack(xs), *[train_stack])), None, agent_state
