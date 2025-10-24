"""Shared helpers for running and logging tabular experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import jax
import jax.numpy as jnp
from gymnasium.envs.functional_jax_env import FunctionalJaxEnv

from rl_research.agents.base import AgentParams, TabularAgent
from rl_research.core.experiment import Episode, run_experiment
from rl_research.tracking.base import Tracker
from rl_research.tracking.mlflow import MLFlowTracker


@dataclass(slots=True)
class ExperimentConfig:
    """Configuration describing how to roll out training and evaluation episodes."""

    num_seeds: int
    total_train_episodes: int
    episode_length: int
    eval_every: int = 0
    num_eval_episodes: int = 0


@dataclass(slots=True)
class TrackingConfig:
    """Configuration for MLflow trackers created by the helper runner."""

    experiment_name: str
    agent_name: str
    parent_run_name: str | None = None
    seed_run_name_template: str = "{agent}_seed_{seed:03d}"
    parent_tags: Mapping[str, str] | None = None
    seed_tags: Mapping[str, str] | None = None


TrackerFactory = Callable[..., Tracker]


def discounted_returns_from_rewards(rewards: jax.Array, discount: float) -> jax.Array:
    """Computes discounted returns for a batch of per-episode rewards."""
    rewards = jnp.asarray(rewards)
    if rewards.ndim == 1:
        rewards = rewards[None, :]
    timesteps = rewards.shape[-1]
    discounts = discount ** jnp.arange(timesteps, dtype=rewards.dtype)
    return jnp.sum(rewards * discounts, axis=-1)


def log_seed_episodes(tracker: Tracker, prefix: str, episodes: Episode, discount: float) -> None:
    """Logs undiscounted and discounted returns for every episode in a batch."""
    rewards = getattr(episodes, "rewards", None)
    if rewards is None:
        return
    rewards = jnp.asarray(rewards)
    if rewards.size == 0:
        return
    if rewards.ndim == 1:
        rewards = rewards[None, :]

    ep_returns = jnp.sum(rewards, axis=-1)
    ep_discounted = discounted_returns_from_rewards(rewards, discount)

    ep_returns = jax.device_get(ep_returns)
    ep_discounted = jax.device_get(ep_discounted)

    horizon = rewards.shape[-1]
    for idx, (ret_val, disc_val) in enumerate(zip(ep_returns, ep_discounted, strict=False)):
        tracker.log_metrics(
            {
                f"{prefix}/episode_reward": float(ret_val),
                f"{prefix}/discounted_return": float(disc_val),
            },
            step=int((idx + 1) * horizon),
        )


def log_q_values_after_each_episode(
    tracker: Tracker,
    q_values_sequence: jax.Array,
    prefix: str = "q_values",
) -> None:
    """Logs Q-value snapshots after each episode for a single seed."""
    q_values = jax.device_get(q_values_sequence)
    if q_values.ndim == 4:  # strip leading dimension if present (e.g. metrics history)
        q_values = q_values[-1]
    if q_values.ndim == 3 and q_values.shape[0] >= 2 and q_values.shape[0] == q_values.shape[1] + 1:
        q_values = q_values[1:]
    if q_values.ndim != 3:
        raise ValueError(
            "Expected Q-values shaped as [num_episodes, num_states, num_actions]; "
            f"received array with shape {q_values.shape}."
        )

    num_states = q_values.shape[1]
    num_actions = q_values.shape[2]
    for episode_idx, q_snapshot in enumerate(q_values, start=1):
        metrics = {
            f"{prefix}/s{state}_a{action}": float(q_snapshot[state, action])
            for state in range(num_states)
            for action in range(num_actions)
        }
        tracker.log_metrics(metrics, step=episode_idx)


def run_tabular_mlflow_example(
    *,
    env: FunctionalJaxEnv,
    agent: TabularAgent,
    agent_params: AgentParams,
    rng: jax.Array,
    run_config: ExperimentConfig,
    tracking: TrackingConfig,
    tracker_factory: TrackerFactory = MLFlowTracker,
    tracker_kwargs: Mapping[str, Any] | None = None,
    seed_tracker_kwargs: Mapping[str, Any] | None = None,
    discount: float | None = None,
) -> tuple[Episode | None, Episode | None, Any]:
    """
    Runs the shared training loop and logs metrics to MLflow for every seed.

    Returns the batched training episodes, evaluation episodes (if any), and the
    final agent states produced by :func:`rl_research.core.experiment.run_experiment`.
    """

    tracker_kwargs = dict(tracker_kwargs or {})
    seed_tracker_kwargs = dict(seed_tracker_kwargs or {})
    parent_tags = {"role": "parent", "agent": tracking.agent_name, **(tracking.parent_tags or {})}

    parent_tracker = tracker_factory(
        tracking.experiment_name,
        tags=parent_tags,
        nested=False,
        **tracker_kwargs,
    )

    parent_params = _params_to_dict(agent_params)
    parent_run_name = tracking.parent_run_name or tracking.agent_name

    parent_tracker.start_run(run_name=parent_run_name, params=parent_params)

    try:
        train_eps, eval_eps, agent_states = run_experiment(
            env=env,
            agent=agent,
            agent_params=agent_params,
            rng=rng,
            num_seeds=run_config.num_seeds,
            total_train_episodes=run_config.total_train_episodes,
            episode_length=run_config.episode_length,
            eval_every=run_config.eval_every,
            num_eval_episodes=run_config.num_eval_episodes,
        )

        discount_to_use = discount if discount is not None else getattr(agent, "discount", 1.0)

        for seed in range(run_config.num_seeds):
            seed_tags = {"role": "seed", "agent": tracking.agent_name, **(tracking.seed_tags or {})}
            seed_tags["seed"] = str(seed)
            seed_tracker = tracker_factory(
                tracking.experiment_name,
                tags=seed_tags,
                nested=True,
                **seed_tracker_kwargs,
            )

            seed_params = dict(parent_params)
            seed_params["seed"] = seed
            seed_run_name = tracking.seed_run_name_template.format(
                agent=tracking.agent_name, seed=seed
            )

            seed_tracker.start_run(run_name=seed_run_name, params=seed_params)
            try:
                seed_train_eps = _select_seed_episode(train_eps, seed)
                if _episode_has_rewards(seed_train_eps):
                    log_seed_episodes(
                        seed_tracker,
                        "train",
                        seed_train_eps,  # type: ignore[arg-type]
                        discount=discount_to_use,
                    )

                if run_config.eval_every > 0 and run_config.num_eval_episodes > 0:
                    seed_eval_eps = _select_seed_episode(eval_eps, seed)
                    if _episode_has_rewards(seed_eval_eps):
                        log_seed_episodes(
                            seed_tracker,
                            "eval",
                            seed_eval_eps,  # type: ignore[arg-type]
                            discount=discount_to_use,
                        )
            finally:
                seed_tracker.end_run(status="FINISHED")

        return train_eps, eval_eps, agent_states
    finally:
        parent_tracker.flush()
        parent_tracker.end_run(status="FINISHED")


def _episode_has_rewards(episode: Episode | None) -> bool:
    if episode is None:
        return False
    rewards = getattr(episode, "rewards", None)
    if rewards is None:
        return False
    rewards = jnp.asarray(rewards)
    return rewards.size > 0


def _select_seed_episode(episodes: Episode | None, seed: int) -> Episode | None:
    if episodes is None:
        return None

    def _pick(field: Any) -> Any:
        if field is None:
            return None
        array = jnp.asarray(field)
        if array.ndim == 0:
            return array
        return array[seed]

    rewards = _pick(episodes.rewards)
    if rewards is not None and rewards.ndim == 1:
        rewards = rewards[None, :]

    return Episode(
        observations=_pick(episodes.observations),
        actions=_pick(episodes.actions),
        next_observations=_pick(episodes.next_observations),
        rewards=rewards,
        terminals=_pick(episodes.terminals),
    )


def _params_to_dict(params: AgentParams) -> dict[str, Any]:
    if hasattr(params, "as_dict"):
        return dict(params.as_dict())  # type: ignore[attr-defined]
    if hasattr(params, "to_dict"):
        return dict(params.to_dict())  # type: ignore[attr-defined]
    if hasattr(params, "__dict__"):
        return {k: v for k, v in params.__dict__.items() if not k.startswith("_")}
    return dict(vars(params))
