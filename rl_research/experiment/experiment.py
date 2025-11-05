"""Pure JAX training loop."""

from __future__ import annotations

import tempfile
from pathlib import Path
from functools import partial
from typing import Any, NamedTuple, Tuple

import mlflow

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
    undiscounted_return: jax.Array
    discounted_return: jax.Array
    length: jax.Array


class SeedResult(NamedTuple):
    train_episodes: Episode
    agent_states: AgentState
    eval_episodes: Episode


class ExperimentParams(NamedTuple):
    num_seeds: int
    total_train_episodes: int
    episode_length: int
    eval_every: int
    num_eval_episodes: int


def _zeros_with_leading_axis(value: Any) -> Any:
    if value is None:
        return None
    array = jnp.asarray(value)
    return jnp.zeros((0,) + array.shape, dtype=array.dtype)


def _stack_pytrees(items: list[Any], template: Any) -> Any:
    if items:
        return jtu.tree_map(lambda *xs: jnp.stack(xs, axis=0), *items)
    if template is None:
        raise ValueError("Cannot create an empty stack without a template value.")
    return jtu.tree_map(_zeros_with_leading_axis, template)


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

    discounts = agent.discount ** jnp.arange(num_steps)
    discounted_return = jnp.dot(traj.reward, discounts)

    episode = Episode(
        observations=traj.observation,
        actions=traj.action,
        next_observations=traj.next_observation,
        rewards=traj.reward,
        terminals=traj.terminal,
        undiscounted_return=jnp.sum(traj.reward),
        discounted_return=discounted_return,
        # TODO: # if using env with terminal state, the number of steps change accordingly
        length=jnp.array(num_steps, dtype=int),
    )
    return episode, final_agent_state

def run_experiment(
    env: FunctionalJaxEnv,
    agent: TabularAgent,
    rng: jax.Array,
    params: ExperimentParams
):
    """Trains for `rng`, running `num_eval_episodes` eval eps every `eval_every`."""
    num_seeds = params.num_seeds
    total_train_episodes = params.total_train_episodes
    episode_length = params.episode_length
    eval_every = params.eval_every
    num_eval_episodes = params.num_eval_episodes

    keys = jrng.split(rng, num_seeds)

    def single_seed(key):
        next_rng, agent_key = jrng.split(key, 2)
        agent_state = agent.init(agent_key)
        agent_state_template = agent_state
        episode_template: Episode | None = None

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
            agent_states.append(agent_state)
            if episode_template is None:
                episode_template = train_ep

            if eval_every > 0 and ep % eval_every == 0 and num_eval_episodes > 0:
                agent_state = agent.eval(agent_state)
                for _ in range(num_eval_episodes):
                    next_rng, rollout_key = jrng.split(next_rng, 2)
                    eval_ep, _ = rollout(
                        env, agent, agent_state, episode_length, rng=rollout_key, train=False
                    )
                    eval_episodes.append(eval_ep)
                    if episode_template is None:
                        episode_template = eval_ep

        train_batch = _stack_pytrees(training_episodes, episode_template)
        eval_batch = _stack_pytrees(eval_episodes, episode_template)
        agent_state_batch = _stack_pytrees(agent_states, agent_state_template)

        return SeedResult(train_batch, agent_state_batch, eval_batch)
    return jax.vmap(single_seed)(keys)


def log_experiment(
    experiment_name: str,
    parent_run_name: str,
    agent_name: str,
    agent_params: AgentParams,
    experiment_params: ExperimentParams,
    env_params: Any,
    experiment_results: SeedResult,
    extra_params: dict[str, Any] | None = None,
):
    train_eps = experiment_results.train_episodes
    agent_states = experiment_results.agent_states
    eval_eps = experiment_results.eval_episodes

    num_seeds = experiment_params.num_seeds
    num_train_episodes = experiment_params.total_train_episodes
    num_eval_episodes = (
        (experiment_params.total_train_episodes // experiment_params.eval_every) * experiment_params.num_eval_episodes
        )

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=parent_run_name, nested=False):
        def _to_dict(payload: Any) -> dict[str, Any]:
            if payload is None:
                return {}
            if hasattr(payload, "_asdict"):
                return payload._asdict()
            if hasattr(payload, "__dict__"):
                return {
                    key: value if isinstance(value, (int, float, str, bool)) else str(value)
                    for key, value in payload.__dict__.items()
                }
            return {}

        base_params: dict[str, Any] = {
            "agent_name": agent_name,
            **_to_dict(agent_params),
            **_to_dict(env_params),
            **experiment_params._asdict(),
        }
        if extra_params:
            base_params.update(extra_params)

        mlflow.log_params(base_params)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            training_artifacts = tmp_dir / 'training_episodes.npz'
            eval_artifacts = tmp_dir / 'eval_episodes.npz'
            agent_artifacts = tmp_dir / 'agent_states.npz'

            jax.numpy.savez(
                training_artifacts,
                **train_eps._asdict()
            )

            jax.numpy.savez(
                eval_artifacts,
                **eval_eps._asdict()
            )

            jax.numpy.savez(
                agent_artifacts,
                **agent_states.__dict__
            )
            
            mlflow.log_artifacts(
                str(tmp_dir),
                artifact_path="artifacts"
            )

        for seed in range(num_seeds):
            with mlflow.start_run(run_name=f"seed_{seed}", nested=True, tags={"parent": parent_run_name}):
                for ep_index in range(num_train_episodes):
                    step = int((ep_index+1)*int(train_eps.length[seed, ep_index]))
                    mlflow.log_metric(f"train/discounted_return", float(train_eps.discounted_return[seed, ep_index]), step=step)
                    mlflow.log_metric(f"train/undiscounted_return", float(train_eps.undiscounted_return[seed, ep_index]), step=step)              

                for ep_index in range(num_eval_episodes):
                    step = int((ep_index+1)*int(eval_eps.length[seed, ep_index]))
                    mlflow.log_metric(f"eval/discounted_return", float(eval_eps.discounted_return[seed, ep_index]), step=step)
                    mlflow.log_metric(f"eval/undiscounted_return", float(eval_eps.undiscounted_return[seed, ep_index]), step=step)
