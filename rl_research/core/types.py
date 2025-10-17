"""Shared type definitions and dataclasses used across the framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, Iterable, Mapping, MutableMapping, Sequence

import jax.numpy as jnp

try:  # pragma: no cover - backwards compatibility for older JAX versions
    from jax import Array  # type: ignore
except ImportError:  # pragma: no cover
    Array = jnp.ndarray  # type: ignore


PRNGKey = Array


@dataclass(slots=True)
class ObservationSpec:
    """Metadata describing the shape and dtype of observations emitted by an environment."""

    shape: Sequence[int]
    dtype: Any


@dataclass(slots=True)
class ActionSpec:
    """Metadata describing the structure of actions consumed by an environment."""

    shape: Sequence[int]
    dtype: Any
    num_values: int | None = None


@dataclass(slots=True)
class EnvironmentSpec:
    """Bundle describing the interfaces an agent needs to interact with an environment."""

    observation: ObservationSpec
    action: ActionSpec


@dataclass(slots=True)
class Transition:
    """Single transition collected during interaction with the environment."""

    observation: Array
    action: Array
    reward: float
    discount: float
    next_observation: Array
    truncation: bool
    terminal: bool
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EpisodeLog:
    """Container summarising an episode trajectory."""

    transitions: Sequence[Transition]
    total_reward: float
    length: int
    seed: int


StateActionKey = tuple[Hashable, Hashable]


@dataclass(slots=True)
class EpisodeStats:
    """Aggregated statistics for an episode useful for analysis and diagnostics."""

    episode: EpisodeLog
    visitation_counts: MutableMapping[StateActionKey, int]
    q_values: MutableMapping[StateActionKey, float]
    metrics: Mapping[str, Any] = field(default_factory=dict)

