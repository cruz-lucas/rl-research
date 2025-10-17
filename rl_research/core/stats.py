"""Utilities for collecting and aggregating statistics from agent-environment interaction."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Mapping, MutableMapping, Protocol, Sequence, Tuple

import jax.numpy as jnp
import numpy as np

from rl_research.core.types import EpisodeLog, EpisodeStats, StateActionKey, Transition


class SupportsQValues(Protocol):
    """Protocol for agents that can compute action-values for a batch of observations."""

    def estimate_action_values(
        self, observations: jnp.ndarray
    ) -> jnp.ndarray:
        """Return Q-values shaped [batch, num_actions] for the provided observations."""


StateTransform = Callable[[jnp.ndarray], np.ndarray]
ActionTransform = Callable[[jnp.ndarray], np.ndarray]


def _default_transform(x: jnp.ndarray) -> np.ndarray:
    array = np.asarray(x)
    if array.ndim == 0:
        return array.reshape(1)
    return array


def _hashable_key(array: np.ndarray) -> Tuple:
    return tuple(array.tolist())


@dataclass(slots=True)
class EpisodeStatsCollector:
    """Collects visitation counts and Q-values for every episode."""

    state_transform: StateTransform = _default_transform
    action_transform: ActionTransform = _default_transform
    aggregate_metrics: bool = True
    custom_metrics: Mapping[str, Callable[[Sequence[Transition]], float]] = field(default_factory=dict)

    def collect(
        self,
        episode: EpisodeLog,
        agent: SupportsQValues | None = None,
    ) -> EpisodeStats:
        """Aggregate visitation counts and Q-values for a finished episode."""

        visitation = defaultdict(int)  # type: MutableMapping[StateActionKey, int]

        for transition in episode.transitions:
            state_key = _hashable_key(self.state_transform(transition.observation))
            action_key = _hashable_key(self.action_transform(transition.action))
            visitation[(state_key, action_key)] += 1

        q_values: MutableMapping[StateActionKey, float] = {}
        if agent is not None and visitation:
            unique_states = {
                _hashable_key(self.state_transform(t.observation)): t.observation
                for t in episode.transitions
            }
            batched_states = jnp.stack([obs for obs in unique_states.values()], axis=0)
            # try:
            #     q_batch = agent.estimate_action_values(batched_states)
            # except NotImplementedError:
            q_batch = None

            if q_batch is not None:
                if q_batch.shape[0] != len(unique_states):
                    raise ValueError("Agent returned Q-values with unexpected batch dimension.")

                for idx, (state_key, _) in enumerate(unique_states.items()):
                    q_values.update(
                        {
                            (state_key, (a,)): float(q_val)
                            for a, q_val in enumerate(np.asarray(q_batch[idx]))
                        }
                    )

        metrics: Dict[str, float] = {}
        if self.aggregate_metrics:
            metrics["episode_return"] = episode.total_reward
            metrics["episode_length"] = episode.length
            metrics["unique_state_action_pairs"] = len(visitation)

        for name, fn in self.custom_metrics.items():
            metrics[name] = float(fn(episode.transitions))

        return EpisodeStats(
            episode=episode,
            visitation_counts=visitation,
            q_values=q_values,
            metrics=metrics,
        )
