#!/usr/bin/env python3
"""
Parse MLflow runs directly from either:
  - file-based MLflow stores (mlruns experiment/run directories), or
  - SQLite-backed MLflow stores (.db / .sqlite files),
and export them as a single pandas DataFrame.
"""

import sqlite3
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


SQLITE_SUFFIXES = {".db", ".sqlite", ".sqlite3"}


def _is_sqlite(path: Path) -> bool:
    """Return True if *path* points to an SQLite database file."""
    if not path.is_file():
        return False

    try:
        with open(path, "rb") as fh:
            header = fh.read(16)
        return header.startswith(b"SQLite format 3")
    except OSError:
        return path.suffix.lower() in SQLITE_SUFFIXES


def _safe_float(value: Any) -> Any:
    """Convert numeric metric values to float when possible."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _normalise_metric_key(metric_key: str) -> str:
    """Map MLflow metric keys to the column layout used by existing analysis."""
    if metric_key.startswith("summary/"):
        return f"metric_summary_{metric_key.removeprefix('summary/')}"
    if metric_key.startswith("train/"):
        return f"metric_train_{metric_key.removeprefix('train/')}"
    return f"metric_{metric_key.replace('/', '_')}"


def _looks_like_file_store(mlruns_dir: Path) -> bool:
    """Return True if the directory looks like a file-based MLflow store root."""
    if not mlruns_dir.is_dir():
        return False

    for child in mlruns_dir.iterdir():
        if child.is_dir() and child.name != ".trash" and (child / "meta.yaml").exists():
            return True
    return False


def _normalise_runs_df(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder columns to keep run metadata first, then params/tags/metrics."""
    basic_cols = [
        "experiment_id",
        "experiment_name",
        "run_id",
        "run_name",
        "status",
        "start_time",
        "end_time",
        "lifecycle_stage",
    ]
    basic_cols = [col for col in basic_cols if col in df.columns]

    param_cols = sorted(col for col in df.columns if col.startswith("param_"))
    tag_cols = sorted(col for col in df.columns if col.startswith("tag_"))
    metric_cols = sorted(col for col in df.columns if col.startswith("metric_"))

    other_cols = [
        col
        for col in df.columns
        if col not in basic_cols + param_cols + tag_cols + metric_cols
    ]
    return df[basic_cols + param_cols + tag_cols + metric_cols + other_cols]


def parse_mlflow_run_file(run_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Parse a single MLflow run directory from a file-based store.

    Args:
        run_dir: Path to a run directory, e.g. ``mlruns/<experiment>/<run_id>/``.

    Returns:
        Dictionary containing run info, params, tags, and summary metrics.
    """
    run_data: Dict[str, Any] = {}

    meta_file = run_dir / "meta.yaml"
    if meta_file.exists():
        import yaml

        with open(meta_file, "r") as fh:
            meta = yaml.safe_load(fh)
            run_data["run_id"] = meta.get("run_id", "")
            run_data["run_name"] = meta.get("run_name", "")
            run_data["experiment_id"] = meta.get("experiment_id", "")
            run_data["status"] = meta.get("status", "")
            run_data["start_time"] = meta.get("start_time", "")
            run_data["end_time"] = meta.get("end_time", "")
            run_data["artifact_uri"] = meta.get("artifact_uri", "")
            run_data["lifecycle_stage"] = meta.get("lifecycle_stage", "")

    if run_data.get("lifecycle_stage") == "deleted":
        return None

    params_dir = run_dir / "params"
    if params_dir.exists():
        for param_file in params_dir.iterdir():
            if param_file.is_file():
                run_data[f"param_{param_file.name}"] = param_file.read_text().strip()

    tags_dir = run_dir / "tags"
    if tags_dir.exists():
        for tag_file in tags_dir.iterdir():
            if tag_file.is_file():
                run_data[f"tag_{tag_file.name}"] = tag_file.read_text().strip()

    for metric_group in ("summary", "train"):
        metrics_dir = run_dir / "metrics" / metric_group
        if not metrics_dir.exists():
            continue

        for metric_file in metrics_dir.iterdir():
            if not metric_file.is_file():
                continue

            lines = metric_file.read_text().splitlines()
            if not lines:
                continue

            parts = lines[-1].strip().split()
            if len(parts) >= 2:
                run_data[f"metric_{metric_group}_{metric_file.name}"] = _safe_float(
                    parts[1]
                )

    return run_data


def parse_mlflow_experiment_file(mlruns_path: str = "mlruns") -> pd.DataFrame:
    """Parse all experiments and runs from a file-based MLflow store."""
    mlruns_dir = Path(mlruns_path)

    if not mlruns_dir.exists():
        raise FileNotFoundError(f"MLflow directory not found: {mlruns_path}")

    all_runs: List[Dict[str, Any]] = []

    for exp_dir in mlruns_dir.iterdir():
        if not exp_dir.is_dir() or exp_dir.name == ".trash":
            continue

        exp_meta_file = exp_dir / "meta.yaml"
        if not exp_meta_file.exists():
            continue

        experiment_name = exp_dir.name
        import yaml

        with open(exp_meta_file, "r") as fh:
            exp_meta = yaml.safe_load(fh)
            experiment_name = exp_meta.get("name", exp_dir.name)

        for run_dir in exp_dir.iterdir():
            if not run_dir.is_dir():
                continue

            try:
                run_data = parse_mlflow_run_file(run_dir)
                if run_data is None:
                    continue
                run_data["experiment_name"] = experiment_name
                run_data["mlflow_backend"] = "file"
                run_data["mlflow_source"] = str(mlruns_dir.resolve())
                all_runs.append(run_data)
            except Exception as exc:
                print(f"Error parsing run {run_dir}: {exc}")

    if not all_runs:
        return pd.DataFrame()

    return _normalise_runs_df(pd.DataFrame(all_runs))


def _table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    """Return True if a table exists in the SQLite database."""
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _fetch_sqlite_metric_rows(
    connection: sqlite3.Connection,
) -> Sequence[sqlite3.Row]:
    """Fetch one latest metric row per (run_uuid, key)."""
    if _table_exists(connection, "latest_metrics"):
        return connection.execute(
            "SELECT run_uuid, key, value FROM latest_metrics"
        ).fetchall()

    if _table_exists(connection, "metrics"):
        return connection.execute(
            """
            SELECT run_uuid, key, value
            FROM (
                SELECT
                    run_uuid,
                    key,
                    value,
                    ROW_NUMBER() OVER (
                        PARTITION BY run_uuid, key
                        ORDER BY step DESC, timestamp DESC, rowid DESC
                    ) AS rn
                FROM metrics
            )
            WHERE rn = 1
            """
        ).fetchall()

    return []


def parse_mlflow_experiment_sqlite(db_path: str) -> pd.DataFrame:
    """Parse all experiments and runs from a single MLflow SQLite database."""
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"SQLite database not found: {db_path}")

    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row

    try:
        required_tables = {"experiments", "runs"}
        missing_tables = [
            table for table in required_tables if not _table_exists(connection, table)
        ]
        if missing_tables:
            raise ValueError(
                f"Not an MLflow tracking database, missing tables: {', '.join(missing_tables)}"
            )

        experiment_rows = connection.execute(
            """
            SELECT experiment_id, name
            FROM experiments
            WHERE COALESCE(lifecycle_stage, 'active') != 'deleted'
            """
        ).fetchall()
        experiment_map = {
            str(row["experiment_id"]): row["name"] for row in experiment_rows
        }

        run_rows = connection.execute(
            """
            SELECT
                run_uuid,
                experiment_id,
                name,
                status,
                start_time,
                end_time,
                lifecycle_stage,
                artifact_uri
            FROM runs
            WHERE COALESCE(lifecycle_stage, 'active') != 'deleted'
            """
        ).fetchall()

        if not run_rows:
            return pd.DataFrame()

        all_runs: List[Dict[str, Any]] = []
        for row in run_rows:
            experiment_id = str(row["experiment_id"])
            all_runs.append(
                {
                    "run_id": row["run_uuid"],
                    "run_name": row["name"] or "",
                    "experiment_id": experiment_id,
                    "experiment_name": experiment_map.get(experiment_id, experiment_id),
                    "status": row["status"] or "",
                    "start_time": row["start_time"],
                    "end_time": row["end_time"],
                    "lifecycle_stage": row["lifecycle_stage"] or "",
                    "artifact_uri": row["artifact_uri"] or "",
                    "mlflow_backend": "sqlite",
                    "mlflow_source": str(path.resolve()),
                }
            )

        run_index = {run["run_id"]: run for run in all_runs}

        if _table_exists(connection, "params"):
            for row in connection.execute("SELECT run_uuid, key, value FROM params"):
                run = run_index.get(row["run_uuid"])
                if run is not None:
                    run[f"param_{row['key']}"] = row["value"]

        if _table_exists(connection, "tags"):
            for row in connection.execute("SELECT run_uuid, key, value FROM tags"):
                run = run_index.get(row["run_uuid"])
                if run is not None:
                    run[f"tag_{row['key']}"] = row["value"]

        for row in _fetch_sqlite_metric_rows(connection):
            run = run_index.get(row["run_uuid"])
            if run is not None:
                run[_normalise_metric_key(row["key"])] = _safe_float(row["value"])

    finally:
        connection.close()

    return _normalise_runs_df(pd.DataFrame(list(run_index.values())))


def _discover_mlflow_sources(path: Path) -> Tuple[List[Path], List[Path]]:
    """
    Discover MLflow sources under *path*.

    Returns:
        Tuple of (file_store_roots, sqlite_db_files).
    """
    if path.is_file():
        if _is_sqlite(path):
            return [], [path]
        raise ValueError(f"Unsupported file path: {path}")

    file_roots: List[Path] = []
    sqlite_files: List[Path] = []

    seen_roots = set()
    seen_dbs = set()

    if _looks_like_file_store(path):
        resolved = path.resolve()
        seen_roots.add(resolved)
        file_roots.append(path)

    for candidate in path.rglob("mlruns"):
        if candidate.is_dir() and _looks_like_file_store(candidate):
            resolved = candidate.resolve()
            if resolved not in seen_roots:
                seen_roots.add(resolved)
                file_roots.append(candidate)

    for suffix in SQLITE_SUFFIXES:
        for candidate in path.rglob(f"*{suffix}"):
            if candidate.is_file() and _is_sqlite(candidate):
                resolved = candidate.resolve()
                if resolved not in seen_dbs:
                    seen_dbs.add(resolved)
                    sqlite_files.append(candidate)

    return sorted(file_roots, key=str), sorted(sqlite_files, key=str)


def _concat_dataframes(dataframes: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate non-empty dataframes using the shared column layout."""
    non_empty = [df for df in dataframes if df is not None and not df.empty]
    if not non_empty:
        return pd.DataFrame()
    if len(non_empty) == 1:
        return _normalise_runs_df(non_empty[0].copy())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        combined = pd.concat(non_empty, ignore_index=True, sort=False)
    return _normalise_runs_df(combined)


def parse_mlflow_experiment(mlruns_path: str = "mlruns") -> pd.DataFrame:
    """
    Parse MLflow runs from:
      - a single SQLite database,
      - a single file-based MLflow store,
      - or a directory tree containing multiple MLflow stores.
    """
    path = Path(mlruns_path)
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {mlruns_path}")

    if path.is_file():
        if _is_sqlite(path):
            print(f"[backend] SQLite database detected: {path}")
            return parse_mlflow_experiment_sqlite(str(path))
        raise ValueError(f"Unsupported file path: {path}")

    file_roots, sqlite_files = _discover_mlflow_sources(path)
    if not file_roots and not sqlite_files:
        raise ValueError(f"No MLflow stores found under: {mlruns_path}")

    print(
        f"[backend] Detected {len(file_roots)} file store(s) and "
        f"{len(sqlite_files)} SQLite database(s) under {path}"
    )

    frames: List[pd.DataFrame] = []

    for file_root in file_roots:
        try:
            frames.append(parse_mlflow_experiment_file(str(file_root)))
        except Exception as exc:
            print(f"Error parsing file-based store {file_root}: {exc}")

    for db_file in sqlite_files:
        try:
            frames.append(parse_mlflow_experiment_sqlite(str(db_file)))
        except Exception as exc:
            print(f"Error parsing SQLite store {db_file}: {exc}")

    df = _concat_dataframes(frames)
    if df.empty:
        print("No runs found in the provided MLflow path.")
    return df


def main():
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse MLflow stores into a pandas DataFrame"
    )
    parser.add_argument(
        "--mlruns-path",
        default="/home/lcruz1/scratch/hpo_navix",
        help="Path to an MLflow store, SQLite file, or a directory containing them",
    )
    parser.add_argument("--output", "-o", help="Output CSV file path (optional)")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Print the DataFrame preview and info to console",
    )

    args = parser.parse_args()

    print(f"Parsing MLflow data from: {args.mlruns_path}")
    df = parse_mlflow_experiment(args.mlruns_path)

    if df.empty:
        print("\nFound 0 runs across 0 experiments")
        print("Columns: 0")
    else:
        print(f"\nFound {len(df)} runs across {df['experiment_name'].nunique()} experiments")
        print(f"Columns: {len(df.columns)}")

    if args.show and not df.empty:
        print("\nDataFrame preview:")
        print(df.head())
        print("\nDataFrame info:")
        print(df.info())

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nSaved to: {args.output}")

    return df


if __name__ == "__main__":
    df = main()
