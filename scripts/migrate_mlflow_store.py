"""Migrate MLflow runs from a file-backed store into a SQLite backend store."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import tyro
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient


@dataclass
class Args:
    """Arguments for migrating MLflow runs between tracking stores."""

    source_uri: str = "./mlruns"
    dest_uri: str = "sqlite:///mlruns.db"
    dest_artifact_root: Path = Path("./mlruns_sqlite_artifacts")


def ensure_dest_artifact_root(path: Path) -> str:
    path.mkdir(parents=True, exist_ok=True)
    return path.absolute().as_uri()


def migrate_single_run(
    src_client: MlflowClient,
    dest_client: MlflowClient,
    src_run,
    dest_experiment_id: str,
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

    copy_artifacts_between_runs(src_client, dest_client, src_run_id, dest_run_id)

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
) -> None:
    with tempfile.TemporaryDirectory(prefix="mlflow_migrate_") as tmpdir:
        local_artifact_path = src_client.download_artifacts(
            run_id=src_run_id,
            path="",
            dst_path=tmpdir,
        )
        dest_client.log_artifacts(
            run_id=dest_run_id,
            local_dir=local_artifact_path,
        )


def migrate_experiment_runs(
    src_client: MlflowClient,
    dest_client: MlflowClient,
    src_experiment_id: str,
    dest_experiment_id: str,
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
        )


def main() -> None:
    migrate(tyro.cli(Args))


if __name__ == "__main__":
    main()
