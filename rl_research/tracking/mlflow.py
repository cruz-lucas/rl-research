"""MLflow-backed implementation of the Tracker interface."""

from __future__ import annotations

from typing import Mapping

try:
    import mlflow
except ImportError as exc:  # pragma: no cover - import guarded for optional dependency
    raise RuntimeError(
        "MLflow is not installed. Install rl-research with the 'tracking' extra or add mlflow to dependencies."
    ) from exc


from rl_research.tracking.base import Tracker


class MLFlowTracker(Tracker):
    def __init__(
        self,
        experiment_name: str,
        *,
        tracking_uri: str | None = None,
        tags: Mapping[str, str] | None = None,
        nested: bool = False,
    ) -> None:
        self.tags = dict(tags or {})
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._run = None
        self._nested = nested

    def start_run(self, run_name: str, params: Mapping[str, object]) -> None:
        if self._run is not None:
            raise RuntimeError("An MLflow run is already active.")
        self._run = mlflow.start_run(run_name=run_name, tags=self.tags, nested=self._nested)
        mlflow.log_params(params)

    def log_metrics(self, metrics: Mapping[str, float | int | bool], step: int) -> None:
        if self._run is None:
            raise RuntimeError("Cannot log metrics without an active MLflow run.")
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)

    def log_params(self, params: Mapping[str, object]) -> None:
        if self._run is None:
            raise RuntimeError("Cannot log params without an active MLflow run.")
        mlflow.log_params(params)

    def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
        if self._run is None:
            raise RuntimeError("Cannot log artifacts without an active MLflow run.")
        mlflow.log_artifact(path, artifact_path)

    def flush(self) -> None:
        if hasattr(mlflow, "flush"):
            mlflow.flush()  # type: ignore[attr-defined]

    def end_run(self, status: str) -> None:
        if self._run is None:
            return
        mlflow.end_run(status=status)
        self._run = None

