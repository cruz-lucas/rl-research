"""Command line entrypoint for orchestrating experiments with Hydra."""

from __future__ import annotations

from typing import Any, Mapping

import hydra
from omegaconf import DictConfig, OmegaConf

from rl_research.agents import AGENTS, Agent
from rl_research.core.experiment import ExperimentConfig, ExperimentRunner
from rl_research.core.stats import EpisodeStatsCollector
from rl_research.envs import ENVIRONMENTS, Environment
from rl_research.tracking.base import NullTracker, Tracker
from rl_research.utils.importing import load_attr


def _to_container(cfg: Any) -> Mapping[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True) if cfg is not None else {}


def _instantiate_agent(cfg: DictConfig) -> Agent:
    params = cfg.agent.get(cfg.environment.name, cfg.agent.base_params)
    name = cfg.agent.get("name")
    if name:
        if name not in AGENTS:
            raise ValueError(f"Agent '{name}' is not registered. Available: {list(AGENTS.keys())}")
        return AGENTS[name](params)
    raise ValueError("Agent configuration must define either 'target' or 'name'.")


def _instantiate_environment(cfg: DictConfig) -> Environment:
    params = _to_container(cfg.params)
    target = cfg.get("target")
    name = cfg.get("name")
    if target:
        cls = load_attr(target)
        return cls(params)
    if name:
        if name not in ENVIRONMENTS:
            raise ValueError(f"Environment '{name}' is not registered. Available: {list(ENVIRONMENTS.keys())}")
        return ENVIRONMENTS[name](params)
    raise ValueError("Environment configuration must define either 'target' or 'name'.")


def _build_tracker(cfg: DictConfig) -> Tracker:
    if cfg is None:
        return NullTracker()

    tracking_type = cfg.get("type", "noop")
    params = _to_container(cfg.params)

    if tracking_type == "noop":
        return NullTracker()
    if tracking_type == "mlflow":
        from rl_research.tracking.mlflow import MLFlowTracker

        return MLFlowTracker(**params)

    raise ValueError(f"Unsupported tracking type '{tracking_type}'.")


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    agent = _instantiate_agent(cfg)
    environment = _instantiate_environment(cfg.environment)
    agent_metadata = _to_container(cfg.agent)
    environment_metadata = _to_container(cfg.environment)
    experiment_metadata = _to_container(cfg.experiment)

    experiment_cfg = ExperimentConfig(**experiment_metadata)
    tracker = _build_tracker(cfg.tracking)
    stats_collector = EpisodeStatsCollector()

    runner = ExperimentRunner(
        agent=agent,
        environment=environment,
        config=experiment_cfg,
        agent_metadata=agent_metadata,
        environment_metadata=environment_metadata,
        experiment_metadata=experiment_metadata,
        tracker=tracker,
        stats_collector=stats_collector,
    )

    episode_stats = runner.run()

    # for idx, stats in enumerate(episode_stats):
    #     print(f"Episode {idx}: return={stats.metrics.get('episode_return', 'n/a')} length={stats.metrics.get('episode_length', 'n/a')}")
    #     if stats.visitation_counts:
    #         sample_items = list(stats.visitation_counts.items())[:5]
    #         print("  visitation sample:", sample_items)
    #     if stats.q_values:
    #         sample_q = list(stats.q_values.items())[:5]
    #         print("  q-value sample:", sample_q)


if __name__ == "__main__":
    main()
