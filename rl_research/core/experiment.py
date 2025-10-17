"""Experiment orchestration utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Mapping, Sequence

import jax
import jax.numpy as jnp

from rl_research.agents.base import Agent, AgentState
from rl_research.core.stats import EpisodeStatsCollector
from rl_research.core.types import (
    ActionSpec,
    EpisodeLog,
    EpisodeStats,
    EnvironmentSpec,
    ObservationSpec,
    PRNGKey,
    Transition,
)
from rl_research.envs.base import Environment, StepResult
from rl_research.tracking.base import NullTracker, Tracker


@dataclass(slots=True)
class ExperimentConfig:
    """Configuration controlling how an experiment is executed."""

    name: str = "experiment"
    n_training_seeds: int = 30
    total_episodes: int = 1
    max_steps_per_episode: int = 256
    discount: float = 1.0
    log_every: int = 1
    enable_q_value_logging: bool = True

class ExperimentRunner:
    """Coordinates interaction between an agent and environment across episodes."""

    def __init__(
        self,
        agent: Agent,
        environment: Environment,
        config: ExperimentConfig,
        *,
        agent_metadata: Mapping[str, Any] | None = None,
        environment_metadata: Mapping[str, Any] | None = None,
        experiment_metadata: Mapping[str, Any] | None = None,
        tracker: Tracker | None = None,
        stats_collector: EpisodeStatsCollector | None = None,
    ) -> None:
        self.agent = agent
        self.environment = environment
        self.config = config
        self.tracker = tracker or NullTracker()
        self.stats_collector = stats_collector or EpisodeStatsCollector()
        self._run_tracking_params = self._prepare_tracking_params(
            agent_metadata=agent_metadata,
            environment_metadata=environment_metadata,
            experiment_metadata=experiment_metadata,
        )

    def run(self) -> list[EpisodeStats]:
        results: list[EpisodeStats] = []

        for seed in range(self.config.n_training_seeds):
            run_name = f"{self.config.name}_seed{seed}"
            run_params = {
                "seed": seed,
                "episodes": self.config.total_episodes,
                "max_steps_per_episode": self.config.max_steps_per_episode,
            }
            run_params.update(self._run_tracking_params)

            self.tracker.start_run(run_name=run_name, params=run_params)
            rng = jax.random.PRNGKey(seed)
            rng, agent_rng = jax.random.split(rng)
            # agent_state = self.agent.init(agent_rng, self._spec)
            # agent_state = self.agent.init()

            for episode in range(self.config.total_episodes):
                rng, episode_rng = jax.random.split(rng)
                episode_seed = int(jax.random.randint(episode_rng, shape=(), minval=0, maxval=2**31 - 1))
                episode_stats = self._run_single_episode(
                    rng=episode_rng,
                    # agent_state=agent_state,
                    episode_seed=episode_seed,
                )
                # agent_state = episode_stats["agent_state"]
                stats: EpisodeStats = episode_stats["stats"]
                results.append(stats)

                metrics = dict(stats.metrics)
                metrics["episode"] = episode
                metrics["seed"] = seed
                self.tracker.log_metrics(metrics, step=episode)

                if (episode + 1) % self.config.log_every == 0:
                    self.tracker.flush()

            self.tracker.end_run(status="FINISHED")

        return results

    def _run_single_episode(
        self,
        rng: PRNGKey,
        # agent_state: AgentState,
        episode_seed: int,
    ) -> Mapping[str, EpisodeStats | AgentState]:
        observation, info = self.environment.reset(seed=episode_seed)
        transitions: list[Transition] = []
        total_reward = 0.0

        for step in range(self.config.max_steps_per_episode):
            # rng, actor_rng, update_rng = jax.random.split(rng, 4)
            # action, agent_state, policy_info = self.agent.select_action(
            #     actor_rng, agent_state, observation
            # )
            action = self.agent.select_action(observation)
            # step_result = self.environment.step(action)
            next_obs, reward, terminal, truncated, info = self.environment.step(action)
            transitions.append(
                Transition(
                    observation=observation,
                    action=action,
                    reward=reward,
                    terminal=terminal,
                    truncation=truncated,
                    discount=1.0,
                    next_observation=next_obs,
                    # extras={"policy_info": policy_info, "env_info": info},
                    extras={"env_info": info},
                )
            )
        
            # agent_state, update_metrics = self.agent.update(update_rng, agent_state, transitions[-1])
            update_metrics = self.agent.update(
                obs=observation,
                action=action,
                reward=reward,
                next_obs=next_obs
            )

            total_reward += reward
            observation = next_obs

            if update_metrics:
                self.tracker.log_metrics(update_metrics.as_dict(), step=step)

            if terminal or truncated:
                break

        episode_log = EpisodeLog(
            transitions=tuple(transitions),
            total_reward=total_reward,
            length=len(transitions),
            seed=episode_seed,
        )

        # agent_metrics = self.agent.on_episode_end(
        #     # agent_state=agent_state,
        #     episode_return=total_reward,
        #     episode_length=len(transitions),
        # )

        stats = self.stats_collector.collect(
            episode=episode_log,
            agent=self.agent if self.config.enable_q_value_logging else None,
        )
        # if agent_metrics:
        #     stats.metrics.update(agent_metrics)

        # return {"agent_state": agent_state, "stats": stats}
        return {"agent_state": None, "stats": stats}

    def _prepare_tracking_params(
        self,
        *,
        agent_metadata: Mapping[str, Any] | None,
        environment_metadata: Mapping[str, Any] | None,
        experiment_metadata: Mapping[str, Any] | None,
    ) -> dict[str, str]:
        params: dict[str, str] = {}

        params["agent.python_class"] = f"{self.agent.__module__}.{self.agent.__class__.__qualname__}"
        if agent_metadata:
            self._flatten_metadata("agent", agent_metadata, params)

        params["environment.python_class"] = f"{self.environment.__module__}.{self.environment.__class__.__qualname__}"
        if environment_metadata:
            self._flatten_metadata("environment", environment_metadata, params)

        if experiment_metadata:
            self._flatten_metadata("experiment", experiment_metadata, params)

        return params

    @staticmethod
    def _flatten_metadata(prefix: str, value: Any, output: dict[str, str]) -> None:
        from collections.abc import Mapping as ABCMapping, Sequence as ABCSequence

        if isinstance(value, ABCMapping):
            for key, nested in value.items():
                key_str = f"{prefix}.{key}" if prefix else str(key)
                ExperimentRunner._flatten_metadata(key_str, nested, output)
            return

        if isinstance(value, ABCSequence) and not isinstance(value, (str, bytes)):
            for idx, item in enumerate(value):
                key_str = f"{prefix}[{idx}]"
                ExperimentRunner._flatten_metadata(key_str, item, output)
            return

        output[prefix] = ExperimentRunner._stringify_for_param(value)

    @staticmethod
    def _stringify_for_param(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, (int, float, bool)):
            return str(value)
        if value is None:
            return "None"
        try:
            return json.dumps(value)
        except TypeError:
            return repr(value)
