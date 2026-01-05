"""Migrate MLflow runs from a file-backed store into a SQLite backend store."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import tyro
from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient


@dataclass
class Args:
    """Arguments for migrating MLflow runs between tracking stores."""

    source_uri: str = "./mlruns"
    dest_uri: str = "sqlite:///mlruns.db"
    dest_artifact_root: Path = Path("./mlruns_sqlite_artifacts")
    artifact_uri_prefix_to_replace: str | None = None
    artifact_uri_prefix_replacement: Path | None = None
    source_runs_root: Path | None = None


def _path_from_uriish(value: str | Path | None) -> Path | None:
    """Best-effort conversion of a path or file:// URI into a local Path."""
    if value is None:
        return None
    if isinstance(value, Path):
        return value

    parsed = urlparse(str(value))
    if parsed.scheme == "file":
        netloc = f"/{parsed.netloc}" if parsed.netloc else ""
        return Path(f"{netloc}{parsed.path}")
    if parsed.scheme == "":
        return Path(str(value))
    return None


def resolve_local_artifacts_dir(
    src_run,
    source_runs_root: Path | None,
    artifact_uri_prefix_to_replace: Path | None,
    artifact_uri_prefix_replacement: Path | None,
) -> Path | None:
    """Find a local artifacts directory even if the stored artifact_uri is stale."""
    candidates: list[Path] = []

    artifact_uri_path = _path_from_uriish(src_run.info.artifact_uri)
    if artifact_uri_path is not None:
        candidates.append(artifact_uri_path)

    if (
        artifact_uri_prefix_to_replace is not None
        and artifact_uri_prefix_replacement is not None
        and artifact_uri_path is not None
    ):
        try:
            relative = artifact_uri_path.relative_to(artifact_uri_prefix_to_replace)
            candidates.append(artifact_uri_prefix_replacement / relative)
        except ValueError:
            # Prefix does not match; ignore.
            pass

    if source_runs_root is not None:
        candidates.append(
            source_runs_root
            / src_run.info.experiment_id
            / src_run.info.run_id
            / "artifacts"
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def ensure_dest_artifact_root(path: Path) -> str:
    path.mkdir(parents=True, exist_ok=True)
    return path.absolute().as_uri()


def migrate_single_run(
    src_client: MlflowClient,
    dest_client: MlflowClient,
    src_run,
    dest_experiment_id: str,
    source_runs_root: Path | None,
    artifact_uri_prefix_to_replace: Path | None,
    artifact_uri_prefix_replacement: Path | None,
) -> None:
    src_run_id = src_run.info.run_id
    tags = dict(src_run.data.tags) if src_run.data.tags is not None else {}
    tags["migrated_from_run_id"] = src_run_id
    run_name = tags.get("mlflow.runName")

    dest_run = dest_client.create_run(
        experiment_id=dest_experiment_id,
        start_time=src_run.info.start_time,
        tags=tags,
        run_name=run_name,
    )
    dest_run_id = dest_run.info.run_id

    for key, value in src_run.data.params.items():
        dest_client.log_param(dest_run_id, key, value)

    for metric_key in src_run.data.metrics.keys():
        history = src_client.get_metric_history(src_run_id, metric_key)
        for metric in history:
            dest_client.log_metric(
                run_id=dest_run_id,
                key=metric_key,
                value=metric.value,
                step=metric.step,
                timestamp=metric.timestamp,
            )

    local_artifacts_dir = resolve_local_artifacts_dir(
        src_run=src_run,
        source_runs_root=source_runs_root,
        artifact_uri_prefix_to_replace=artifact_uri_prefix_to_replace,
        artifact_uri_prefix_replacement=artifact_uri_prefix_replacement,
    )
    copy_artifacts_between_runs(
        src_client,
        dest_client,
        src_run_id,
        dest_run_id,
        local_artifacts_dir=local_artifacts_dir,
    )

    dest_client.set_terminated(
        run_id=dest_run_id,
        status=src_run.info.status,
        end_time=src_run.info.end_time,
    )


def copy_artifacts_between_runs(
    src_client: MlflowClient,
    dest_client: MlflowClient,
    src_run_id: str,
    dest_run_id: str,
    local_artifacts_dir: Path | None,
) -> None:
    if local_artifacts_dir is not None:
        dest_client.log_artifacts(
            run_id=dest_run_id,
            local_dir=str(local_artifacts_dir),
        )
        return

    with tempfile.TemporaryDirectory(prefix="mlflow_migrate_") as tmpdir:
        try:
            local_artifact_path = src_client.download_artifacts(
                run_id=src_run_id,
                path="",
                dst_path=tmpdir,
            )
        except MlflowException as exc:
            raise MlflowException(
                f"Failed to download artifacts for run {src_run_id}. "
                "Provide --artifact-uri-prefix-to-replace/--artifact-uri-prefix-replacement "
                "or --source-runs-root to point to the local artifacts directory."
            ) from exc
        dest_client.log_artifacts(
            run_id=dest_run_id,
            local_dir=local_artifact_path,
        )


def migrate_experiment_runs(
    src_client: MlflowClient,
    dest_client: MlflowClient,
    src_experiment_id: str,
    dest_experiment_id: str,
    source_runs_root: Path | None,
    artifact_uri_prefix_to_replace: Path | None,
    artifact_uri_prefix_replacement: Path | None,
) -> None:
    print(f"Migrating runs for experiment {src_experiment_id} -> {dest_experiment_id}")
    page_token = None
    total_runs = 0

    while True:
        runs_page = src_client.search_runs(
            experiment_ids=[src_experiment_id],
            filter_string="",
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=1000,
            page_token=page_token,
        )

        if not runs_page:
            break

        for src_run in runs_page:
            total_runs += 1
            print(f"  Migrating run {src_run.info.run_id}")
            migrate_single_run(
                src_client=src_client,
                dest_client=dest_client,
                src_run=src_run,
                dest_experiment_id=dest_experiment_id,
                source_runs_root=source_runs_root,
                artifact_uri_prefix_to_replace=artifact_uri_prefix_to_replace,
                artifact_uri_prefix_replacement=artifact_uri_prefix_replacement,
            )

        page_token = runs_page.token
        if page_token is None:
            break

    print(f"  Migrated {total_runs} runs from experiment {src_experiment_id}")


def migrate(args: Args) -> None:
    print(f"Connecting to source tracking URI: {args.source_uri}")
    src_client = MlflowClient(tracking_uri=args.source_uri)

    print(f"Connecting to destination backend store: {args.dest_uri}")
    dest_client = MlflowClient(tracking_uri=args.dest_uri)

    if (args.artifact_uri_prefix_to_replace is None) != (
        args.artifact_uri_prefix_replacement is None
    ):
        raise ValueError(
            "Provide both --artifact-uri-prefix-to-replace and "
            "--artifact-uri-prefix-replacement, or neither."
        )

    artifact_uri_prefix_to_replace = _path_from_uriish(
        args.artifact_uri_prefix_to_replace
    )
    if artifact_uri_prefix_to_replace is not None:
        artifact_uri_prefix_to_replace = artifact_uri_prefix_to_replace.resolve()
    artifact_uri_prefix_replacement = (
        args.artifact_uri_prefix_replacement.resolve()
        if args.artifact_uri_prefix_replacement is not None
        else None
    )

    source_runs_root = (
        args.source_runs_root.resolve() if args.source_runs_root is not None else None
    )
    if source_runs_root is None:
        inferred_root = _path_from_uriish(args.source_uri)
        if inferred_root is not None and inferred_root.exists():
            source_runs_root = inferred_root.resolve()

    dest_artifact_root_uri = ensure_dest_artifact_root(args.dest_artifact_root)
    print(f"Destination artifact root: {dest_artifact_root_uri}")

    print("Listing source experiments...")
    src_experiments = src_client.search_experiments(view_type=ViewType.ACTIVE_ONLY)

    for src_exp in src_experiments:
        print(f"\nExperiment: '{src_exp.name}' (id={src_exp.experiment_id})")
        dest_exp = dest_client.get_experiment_by_name(src_exp.name)
        if dest_exp is None:
            dest_exp_id = dest_client.create_experiment(
                name=src_exp.name,
                artifact_location=f"{dest_artifact_root_uri.rstrip('/')}/{src_exp.name}",
            )
            print(f"  Created destination experiment id={dest_exp_id}")
        else:
            dest_exp_id = dest_exp.experiment_id
            print(f"  Using existing destination experiment id={dest_exp_id}")

        migrate_experiment_runs(
            src_client=src_client,
            dest_client=dest_client,
            src_experiment_id=src_exp.experiment_id,
            dest_experiment_id=dest_exp_id,
            source_runs_root=source_runs_root,
            artifact_uri_prefix_to_replace=artifact_uri_prefix_to_replace,
            artifact_uri_prefix_replacement=artifact_uri_prefix_replacement,
        )


def main() -> None:
    migrate(tyro.cli(Args))


if __name__ == "__main__":
    main()
