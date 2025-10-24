"""Tracking interfaces used by the experiment runner."""

from __future__ import annotations

from typing import Mapping, Protocol


class Tracker(Protocol):
    def start_run(self, run_name: str, params: Mapping[str, object]) -> None:
        ...

    def log_metrics(self, metrics: Mapping[str, float | int | bool], step: int) -> None:
        ...

    def log_params(self, params: Mapping[str, object]) -> None:
        ...

    def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
        ...

    def flush(self) -> None:
        ...

    def end_run(self, status: str) -> None:
        ...


class NullTracker:
    """No-op tracker used when experiment tracking is disabled."""

    def start_run(self, run_name: str, params: Mapping[str, object]) -> None:
        return None

    def log_metrics(self, metrics: Mapping[str, float | int | bool], step: int) -> None:
        return None

    def log_params(self, params: Mapping[str, object]) -> None:
        return None

    def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
        return None

    def flush(self) -> None:
        return None

    def end_run(self, status: str) -> None:
        return None
