from __future__ import annotations

from typing import Any, Dict, List

from rl_research.core.experiment import ExperimentConfig, ExperimentRunner
from rl_research.core.stats import EpisodeStatsCollector


class DummyAgent:
    def __init__(self, actions: List[int] | None = None) -> None:
        self._actions = actions or []
        self._idx = 0

    def select_action(self, observation: Any) -> int:
        if self._idx < len(self._actions):
            action = self._actions[self._idx]
            self._idx += 1
            return action
        return 0

    def update(self, *, obs: Any, action: int, reward: float, next_obs: Any):
        return None


class DummyEnvironment:
    def __init__(self, rewards: List[float]) -> None:
        self._rewards = rewards
        self._step = 0

    def reset(self, seed: int | None = None):
        self._step = 0
        return 0, {}

    def step(self, action: int):
        reward = self._rewards[self._step]
        self._step += 1
        terminal = self._step >= len(self._rewards)
        return 0, reward, terminal, False, {}


class RecordingTracker:
    def __init__(self) -> None:
        self.started: List[tuple[str, Dict[str, Any]]] = []
        self.logged_metrics: List[tuple[Dict[str, Any], int]] = []
        self.flush_calls = 0
        self.ended: List[str] = []

    def start_run(self, run_name: str, params: Dict[str, Any]) -> None:
        self.started.append((run_name, dict(params)))

    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        self.logged_metrics.append((dict(metrics), step))

    def log_params(self, params: Dict[str, Any]) -> None:  # pragma: no cover - not used
        return None

    def log_artifact(self, path: str, artifact_path: str | None = None) -> None:  # pragma: no cover - not used
        return None

    def flush(self) -> None:
        self.flush_calls += 1

    def end_run(self, status: str) -> None:
        self.ended.append(status)


def test_experiment_runner_collects_episode() -> None:
    agent = DummyAgent(actions=[0, 0, 0])
    env = DummyEnvironment(rewards=[1.0, 1.0, 0.0])
    config = ExperimentConfig(
        name="dummy_experiment",
        n_training_seeds=1,
        total_episodes=2,
        max_steps_per_episode=3,
        log_every=1,
    )
    tracker = RecordingTracker()
    runner = ExperimentRunner(
        agent=agent,
        environment=env,
        config=config,
        tracker=tracker,
        stats_collector=EpisodeStatsCollector(),
    )

    results = runner.run()

    assert len(results) == config.total_episodes
    assert tracker.started[0][0] == "dummy_experiment_seed0"
    assert tracker.ended == ["FINISHED"]
    for stats in results:
        assert stats.episode.length > 0
        assert "episode_return" in stats.metrics


def test_agent_metadata_logged_for_tracking() -> None:
    agent = DummyAgent()
    env = DummyEnvironment(rewards=[1.0])
    config = ExperimentConfig(
        name="metadata_experiment",
        n_training_seeds=1,
        total_episodes=1,
        max_steps_per_episode=1,
    )
    tracker = RecordingTracker()
    agent_metadata = {
        "name": "test_agent",
        "params": {
            "alpha": 0.1,
            "schedule": [1, 2],
            "seed": None,
        },
    }
    environment_metadata = {
        "name": "test_env",
        "params": {"num_states": 7, "num_actions": 6},
    }
    experiment_metadata = {
        "name": "metadata_experiment",
        "n_training_seeds": 1,
        "total_episodes": 1,
        "max_steps_per_episode": 1,
    }

    runner = ExperimentRunner(
        agent=agent,
        environment=env,
        config=config,
        agent_metadata=agent_metadata,
        environment_metadata=environment_metadata,
        experiment_metadata=experiment_metadata,
        tracker=tracker,
        stats_collector=EpisodeStatsCollector(),
    )

    runner.run()

    assert tracker.started, "Tracker did not record any runs."
    _, params = tracker.started[0]
    assert params["agent.name"] == "test_agent"
    assert params["agent.params.alpha"] == "0.1"
    assert params["agent.params.schedule[1]"] == "2"
    assert params["agent.params.seed"] == "None"
    assert params["agent.python_class"].endswith("DummyAgent")
    assert params["environment.name"] == "test_env"
    assert params["environment.params.num_states"] == "7"
    assert params["environment.python_class"].endswith("DummyEnvironment")
    assert params["experiment.name"] == "metadata_experiment"
    assert params["experiment.max_steps_per_episode"] == "1"
