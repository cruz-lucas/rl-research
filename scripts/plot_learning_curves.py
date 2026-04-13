#!/usr/bin/env python3
"""
Plot learning curves with bootstrap confidence intervals.
Handles unaligned episode data and produces publication-ready figures.

Supports two MLflow storage backends, detected automatically:
  - File-based store  (a directory, e.g. ./mlruns)
  - SQLite database   (a .db / .sqlite file, e.g. ./mlflow.db)
"""

import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from scipy import interpolate

# Publication-ready plot settings
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'lines.linewidth': 2.0,
    'patch.linewidth': 0.5,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.minor.width': 0.8,
    'ytick.minor.width': 0.8,
    'mathtext.default': 'regular',
})

# Default configurations (can be overridden)
DEFAULT_GROUP_COLORS = {
    "qlearning_epsgreedy": "#ff7f0e",
    "mbieeb": "#9467bd",
    "rmax": "#2ca02c",
    "replaybased_rmax": "#1f77b4",
    "replaybased_rmax_uniquevisitationincrement": "#1f77b4",
    "replaybased_rmax_refact": "#1f77b4",
    "replaybased_mbieeb": "#d62728",
    "epsgreedy_intrinsicreward": "#999999",
    "intrinsicreward": "#999999",
    "epsgreedy_intrinsicreward_optinit": "#27d6a7",
    "intrinsicreward_optinit": "#27d6a7",
    "intrinsicreward_optinit_2": "#27d6a7",

    "rmax_nfq_onehot_linear": "#d62728",
    "rmax_dqn_onehot_linear": "#ff7f0e",
    "nfq_onehot": "#1f77b4",
    "dqn_onehot_linear": "#9467bd",

    "dqn": "#d62728",
    "dqn_rnd": "#ff7f0e",
    "dqn_rnd_action_input": "#9467bd",
    "dqn_rnd_action_output": "#9467bd",
    "rmax_dqn": "#9467bd",
    "nfq_originalobs": "#ff7f0e",
    "rmax_nfq_originalobs": "#1f77b4",
}

DEFAULT_GROUP_LABELS = {
    "qlearning_epsgreedy": "Q-learning (Epsilon-Greedy)",
    "mbieeb": "MBIE-EB",
    "replaybased_mbieeb": "Replay-based MBIE-EB",
    "rmax": "R-Max",
    "replaybased_rmax": "Replay-based R-Max",
    "replaybased_rmax_uniquevisitationincrement": "Replay-based R-Max",
    "replaybased_rmax_refact": "Replay-based R-Max",
    "epsgreedy_intrinsicreward": "Count-based Intrinsic Reward",
    "intrinsicreward": "Count-based Intrinsic Reward",
    "epsgreedy_intrinsicreward_optinit": "Count-based Intrinsic Reward + Opt. Init.",
    "intrinsicreward_optinit": "Count-based Intrinsic Reward + Opt. Init.",
    "intrinsicreward_optinit_2": "Count-based Intrinsic Reward + Opt. Init.",

    "rmax_nfq_onehot_linear": "NFQ + R-max",
    "rmax_dqn_onehot_linear": "DQN + R-max",
    "nfq_onehot": "NFQ",
    "dqn_onehot_linear": "DQN",

    "dqn": "DQN",
    "dqn_rnd": "DQN + RND",
    "dqn_rnd_action_input": "DQN + RND (w/ Action)",
    "dqn_rnd_action_output": "DQN + RND (w/ Action)",
    "rmax_dqn": "DQN + R-max",
    "nfq_originalobs": "NFQ",
    "rmax_nfq_originalobs": "NFQ + R-max",
}

DEFAULT_GROUP_ORDER = [
    # "qlearning_epsgreedy",
    "rmax",
    # "replaybased_rmax",
    "replaybased_rmax_refact",
    # "replaybased_rmax_uniquevisitationincrement",
    "mbieeb",
    "replaybased_mbieeb",
    # "epsgreedy_intrinsicreward",
    "intrinsicreward",
    # "epsgreedy_intrinsicreward_optinit",
    "intrinsicreward_optinit",
    # "intrinsicreward_optinit_2",

    # "dqn_onehot_linear",
    # "nfq_onehot",
    # "rmax_nfq_onehot_linear",
    # "rmax_dqn_onehot_linear",

    "dqn",
    "dqn_rnd",
    "dqn_rnd_action_input",
    "dqn_rnd_action_output",
    "rmax_dqn",
    # "nfq_originalobs",
    # "rmax_nfq_originalobs",
]


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def _is_sqlite(path: str) -> bool:
    """
    Return True if *path* points to an SQLite database file.

    Detection order:
      1. Path is an existing file whose first 16 bytes match the SQLite magic
         header (most reliable).
      2. Path ends with a common SQLite extension (.db, .sqlite, .sqlite3) as
         a fallback for files that don't exist yet or can't be read.
    """
    p = Path(path)
    if p.is_file():
        try:
            with open(p, "rb") as fh:
                header = fh.read(16)
            return header.startswith(b"SQLite format 3")
        except OSError:
            pass
    return p.suffix.lower() in {".db", ".sqlite", ".sqlite3"}


# ---------------------------------------------------------------------------
# File-based backend
# ---------------------------------------------------------------------------

def _parse_mlflow_run_file(run_dir: Path) -> Optional[dict]:
    """Parse a single MLflow run directory (file-based store)."""
    run_data = {}

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
        for pf in params_dir.iterdir():
            if pf.is_file():
                run_data[f"param_{pf.name}"] = pf.read_text().strip()

    tags_dir = run_dir / "tags"
    if tags_dir.exists():
        for tf in tags_dir.iterdir():
            if tf.is_file():
                run_data[f"tag_{tf.name}"] = tf.read_text().strip()

    metrics_dir = run_dir / "metrics" / "summary"
    if metrics_dir.exists():
        for mf in metrics_dir.iterdir():
            if mf.is_file():
                lines = mf.read_text().splitlines()
                if lines:
                    parts = lines[-1].strip().split()
                    if len(parts) >= 2:
                        run_data[f"metric_summary_{mf.name}"] = float(parts[1])
    
    metrics_dir = run_dir / "metrics" / "train"
    if metrics_dir.exists():
        for mf in metrics_dir.iterdir():
            if mf.is_file():
                lines = mf.read_text().splitlines()
                if lines:
                    parts = lines[-1].strip().split()
                    if len(parts) >= 2:
                        run_data[f"metric_train_{mf.name}"] = float(parts[1])

    return run_data


def parse_mlflow_experiment_file(mlruns_path: str = "mlruns") -> pd.DataFrame:
    """Parse all MLflow experiments from a file-based store directory."""
    mlruns_dir = Path(mlruns_path)

    if not mlruns_dir.exists():
        raise FileNotFoundError(f"MLflow directory not found: {mlruns_path}")

    all_runs = []

    for exp_dir in mlruns_dir.iterdir():
        if not exp_dir.is_dir() or exp_dir.name == ".trash":
            continue

        experiment_name = exp_dir.name
        exp_meta_file = exp_dir / "meta.yaml"
        if exp_meta_file.exists():
            import yaml
            with open(exp_meta_file, "r") as fh:
                exp_meta = yaml.safe_load(fh)
                experiment_name = exp_meta.get("name", exp_dir.name)

        for run_dir in exp_dir.iterdir():
            if not run_dir.is_dir():
                continue
            try:
                run_data = _parse_mlflow_run_file(run_dir)
                if run_data is not None:
                    run_data["experiment_name"] = experiment_name
                    all_runs.append(run_data)
            except Exception as exc:
                print(f"Error parsing run {run_dir}: {exc}")

    if not all_runs:
        print("No runs found in the MLflow directory.")
        return pd.DataFrame()

    return _normalise_runs_df(pd.DataFrame(all_runs))


# ---------------------------------------------------------------------------
# SQLite backend
# ---------------------------------------------------------------------------

def parse_mlflow_experiment_sqlite(db_path: str) -> pd.DataFrame:
    """
    Parse all MLflow experiments and runs from an MLflow SQLite database.

    MLflow's SQLite schema (created via --backend-store-uri sqlite:///...):

      experiments : experiment_id, name, lifecycle_stage, ...
      runs        : run_uuid (= run_id), experiment_id, name, status,
                    start_time, end_time, lifecycle_stage, artifact_uri
      params      : run_uuid, key, value
      tags        : run_uuid, key, value
      metrics     : run_uuid, key, value, timestamp, step

    Summary metrics (used for the 'metric_*' columns) are derived by taking
    the value at the highest recorded step for each (run, metric) pair.
    """
    p = Path(db_path)
    if not p.exists():
        raise FileNotFoundError(f"SQLite database not found: {db_path}")

    con = sqlite3.connect(str(p))
    con.row_factory = sqlite3.Row

    try:
        # ---- experiments ----
        exp_rows = con.execute(
            "SELECT experiment_id, name FROM experiments "
            "WHERE lifecycle_stage != 'deleted'"
        ).fetchall()
        exp_map = {str(r["experiment_id"]): r["name"] for r in exp_rows}

        # ---- runs ----
        run_rows = con.execute(
            "SELECT run_uuid, experiment_id, name, status, "
            "       start_time, end_time, lifecycle_stage, artifact_uri "
            "FROM runs "
            "WHERE lifecycle_stage != 'deleted'"
        ).fetchall()

        if not run_rows:
            print("No runs found in the SQLite database.")
            return pd.DataFrame()

        all_runs = []
        for r in run_rows:
            run_id = r["run_uuid"]
            exp_id = str(r["experiment_id"])
            all_runs.append({
                "run_id": run_id,
                "run_name": r["name"] or "",
                "experiment_id": exp_id,
                "experiment_name": exp_map.get(exp_id, exp_id),
                "status": r["status"] or "",
                "start_time": r["start_time"],
                "end_time": r["end_time"],
                "lifecycle_stage": r["lifecycle_stage"] or "",
                "artifact_uri": r["artifact_uri"] or "",
            })

        run_index = {r["run_id"]: r for r in all_runs}

        # ---- params ----
        for row in con.execute("SELECT run_uuid, key, value FROM params"):
            if row["run_uuid"] in run_index:
                run_index[row["run_uuid"]][f"param_{row['key']}"] = row["value"]

        # ---- tags ----
        for row in con.execute("SELECT run_uuid, key, value FROM tags"):
            if row["run_uuid"] in run_index:
                run_index[row["run_uuid"]][f"tag_{row['key']}"] = row["value"]

        # ---- summary metrics: value at highest step per (run, metric) ----
        # summary_rows = con.execute(
        #     "SELECT m.run_uuid, m.key, m.value "
        #     "FROM metrics m "
        #     "INNER JOIN ("
        #     "    SELECT run_uuid, key, MAX(step) AS max_step "
        #     "    FROM metrics GROUP BY run_uuid, key"
        #     ") last ON m.run_uuid = last.run_uuid "
        #     "       AND m.key = last.key "
        #     "       AND m.step = last.max_step"
        # ).fetchall()

        # print(summary_rows)
        # for row in summary_rows:
        #     if row["run_uuid"] in run_index:
        #         run_index[row["run_uuid"]][f"metric_{row['key']}"] = float(row["value"])

    finally:
        con.close()

    return _normalise_runs_df(pd.DataFrame(list(run_index.values())))


# ---------------------------------------------------------------------------
# Shared normalisation
# ---------------------------------------------------------------------------

def _normalise_runs_df(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder columns into the canonical layout expected by the plotter."""
    basic_cols = [
        "experiment_id", "experiment_name", "run_id", "run_name",
        "status", "start_time", "end_time", "lifecycle_stage",
    ]
    basic_cols = [c for c in basic_cols if c in df.columns]
    param_cols = sorted(c for c in df.columns if c.startswith("param_"))
    tag_cols = sorted(c for c in df.columns if c.startswith("tag_"))
    metric_cols = sorted(c for c in df.columns if c.startswith("metric_"))
    other_cols = [
        c for c in df.columns
        if c not in basic_cols + param_cols + tag_cols + metric_cols
    ]
    return df[basic_cols + param_cols + tag_cols + metric_cols + other_cols]


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def parse_mlflow_experiment(mlruns_path: str = "mlruns") -> pd.DataFrame:
    """
    Parse MLflow experiment data from either a file-based store or an SQLite DB.

    The backend is detected automatically:
      - If *mlruns_path* is (or looks like) an SQLite file  → SQLite backend
      - Otherwise                                            → file-based backend
    """
    if _is_sqlite(mlruns_path):
        print(f"[backend] SQLite database detected: {mlruns_path}")
        return parse_mlflow_experiment_sqlite(mlruns_path)
    else:
        print(f"[backend] File-based store detected: {mlruns_path}")
        return parse_mlflow_experiment_file(mlruns_path)


# ---------------------------------------------------------------------------
# LearningCurvePlotter
# ---------------------------------------------------------------------------

class LearningCurvePlotter:
    """Plot learning curves with bootstrap confidence intervals."""

    def __init__(self,
                 mlruns_path: str = "mlruns",
                 group_colors: Optional[Dict] = None,
                 group_labels: Optional[Dict] = None,
                 group_order: Optional[List] = None):
        """
        Initialize plotter.

        Args:
            mlruns_path: Path to an mlruns directory *or* an SQLite .db file.
                         The backend is chosen automatically.
            group_colors: Dict mapping group names to colours.
            group_labels: Dict mapping group names to display labels.
            group_order:  List of group names in desired order.
        """
        self._is_sqlite = _is_sqlite(mlruns_path)
        self.mlruns_path = Path(mlruns_path)

        self.df = parse_mlflow_experiment(mlruns_path)

        self.group_colors = group_colors or DEFAULT_GROUP_COLORS
        self.group_labels = group_labels or DEFAULT_GROUP_LABELS
        self.group_order = group_order or DEFAULT_GROUP_ORDER

    # ------------------------------------------------------------------
    # Internal helpers — one per backend
    # ------------------------------------------------------------------

    def _load_metric_series_file(self,
                                  run_id: str,
                                  experiment_id: str,
                                  metric_name: str
                                  ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Load a full metric time series from the file-based store."""
        metric_file = (
            self.mlruns_path / experiment_id / run_id / "metrics" / metric_name
        )
        if not metric_file.exists():
            return None

        steps, values = [], []
        with open(metric_file, "r") as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) >= 3:
                    # format: timestamp  value  step
                    values.append(float(parts[1]))
                    steps.append(float(parts[2]))
                elif len(parts) >= 2:
                    values.append(float(parts[1]))
                    steps.append(float(len(steps)))
                    print(f"Warning: step missing in {metric_file}, using index.")

        return (np.array(steps), np.array(values)) if values else None

    def _load_metric_series_sqlite(self,
                                    run_id: str,
                                    metric_name: str
                                    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Load a full metric time series from the SQLite database."""
        con = sqlite3.connect(str(self.mlruns_path))
        try:
            rows = con.execute(
                "SELECT value, step FROM metrics "
                "WHERE run_uuid = ? AND key = ? "
                "ORDER BY step ASC, timestamp ASC",
                (run_id, metric_name),
            ).fetchall()
        finally:
            con.close()

        if not rows:
            return None

        steps = np.array([r[1] for r in rows], dtype=float)
        values = np.array([r[0] for r in rows], dtype=float)
        return steps, values

    # ------------------------------------------------------------------
    # Public API — backend-agnostic
    # ------------------------------------------------------------------

    def load_returns_from_run(self,
                               run_id: str,
                               experiment_id: str,
                               metric_name: str = "episode_return"
                               ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Load episode returns and steps from a single run.

        Dispatches to the correct backend automatically.

        Args:
            run_id:        MLflow run ID.
            experiment_id: Experiment ID (only used by the file backend).
            metric_name:   Name of the metric to load.

        Returns:
            Tuple of (steps, values) arrays, or None if not found.
        """
        if self._is_sqlite:
            return self._load_metric_series_sqlite(run_id, metric_name)
        else:
            return self._load_metric_series_file(run_id, experiment_id, metric_name)

    def load_group_data(self,
                        experiment_name: str,
                        group_name: str,
                        metric_name: str = "episode_return"
                        ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Load returns and steps for all seeds in a group.

        Returns:
            List of (steps, values) tuples, one per seed.
        """
        group_df = self.df[
            (self.df["experiment_name"] == experiment_name) &
            (self.df["tag_group"] == group_name)
        ]

        if len(group_df) == 0:
            print(f"Warning: No runs found for group '{group_name}'")
            return []

        all_data = []
        for _, row in group_df.iterrows():
            run_id = row["run_id"]
            exp_id = str(row["experiment_id"])
            data = self.load_returns_from_run(run_id, exp_id, metric_name)
            if data is not None:
                steps, values = data
                if len(steps) > 0 and len(values) > 0:
                    all_data.append((steps, values))

        return all_data

    def load_group_final_values(self,
                                 experiment_name: str,
                                 group_name: str,
                                 metric_name: str = "episode_return"
                                 ) -> np.ndarray:
        """
        Load the final (last) metric value for each seed in a group.

        Useful for single-episode environments (e.g. RiverSwim) where only
        the terminal value is meaningful.

        Returns:
            1-D array of final values, one per seed.
        """
        curves_data = self.load_group_data(experiment_name, group_name, metric_name)
        return np.array([v[-1] for _, v in curves_data if len(v) > 0])

    # ------------------------------------------------------------------
    # Statistics helpers
    # ------------------------------------------------------------------

    def align_curves(self,
                     curves_data: List[Tuple[np.ndarray, np.ndarray]],
                     num_points: int = 1000
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """Align curves of different lengths by interpolating to a common step axis."""
        if not curves_data:
            return np.array([]), np.array([])

        min_step = min(s[0] for s, _ in curves_data)
        max_step = max(s[-1] for s, _ in curves_data)
        step_points = np.linspace(min_step, max_step, num_points)

        aligned = []
        for steps, values in curves_data:
            if len(steps) == 0 or len(values) == 0:
                continue
            f = interpolate.interp1d(
                steps, values,
                kind="linear",
                bounds_error=False,
                fill_value=(values[0], values[-1]),
            )
            aligned.append(f(step_points))

        return step_points, np.array(aligned).T  # (num_points, num_seeds)

    def bootstrap_ci(self,
                     data: np.ndarray,
                     n_bootstrap: int = 10000,
                     ci: float = 0.95,
                     random_seed: int = 42
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate bootstrap confidence intervals.

        Args:
            data: Array of shape (num_points, num_seeds).

        Returns:
            mean, lower_bound, upper_bound
        """
        np.random.seed(random_seed)
        num_points, num_seeds = data.shape
        mean = np.mean(data, axis=1)

        alpha = 1 - ci
        bootstrap_means = np.zeros((n_bootstrap, num_points))
        for i in range(n_bootstrap):
            idx = np.random.randint(0, num_seeds, size=num_seeds)
            bootstrap_means[i] = np.mean(data[:, idx], axis=1)

        lower = np.percentile(bootstrap_means, (alpha / 2) * 100, axis=0)
        upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100, axis=0)
        return mean, lower, upper

    # ------------------------------------------------------------------
    # Plot: learning curves
    # ------------------------------------------------------------------

    def plot_learning_curves(self,
                              experiment_name: str,
                              groups: Optional[List[str]] = None,
                              metric_name: str = "episode_return",
                              num_points: int = 1000,
                              n_bootstrap: int = 10000,
                              ci: float = 0.95,
                              xlabel: str = "Steps",
                              ylabel: str = "Discounted Return",
                              title: Optional[str] = None,
                              figsize: Tuple[int, int] = (10, 6),
                              grid: bool = True,
                              legend_loc: str = "best",
                              save_path: Optional[str] = None,
                              dpi: int = 300,
                              show_seeds: bool = False,
                              seed_alpha: float = 0.1) -> plt.Figure:
        """Plot learning curves with bootstrap confidence intervals."""
        if groups is None:
            groups = self.group_order

        fig, ax = plt.subplots(figsize=figsize)

        for group in groups:
            curves_data = self.load_group_data(experiment_name, group, metric_name)
            if not curves_data:
                print(f"Skipping group '{group}' - no data")
                continue
            print(f"Group '{group}': {len(curves_data)} seeds")

            steps, aligned_data = self.align_curves(curves_data, num_points)
            if aligned_data.size == 0:
                continue

            mean, lower, upper = self.bootstrap_ci(aligned_data, n_bootstrap, ci)
            color = self.group_colors.get(group)
            label = self.group_labels.get(group, group)

            if show_seeds:
                for i in range(aligned_data.shape[1]):
                    ax.plot(steps, aligned_data[:, i],
                            color=color, alpha=seed_alpha, linewidth=0.8, zorder=1)

            ax.plot(steps, mean, color=color, label=label, linewidth=2.5, zorder=3)
            ax.fill_between(steps, lower, upper,
                            color=color, alpha=0.2, linewidth=0, zorder=2)

        ax.set_xlabel(xlabel, fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        if title:
            ax.set_title(title, fontweight="bold")
        if grid:
            ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)
        ax.legend(loc=legend_loc, framealpha=0.95, edgecolor="black", fancybox=True)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
            print(f"Saved figure to: {save_path}")

        return fig

    # ------------------------------------------------------------------
    # Plot: box plot of final values
    # ------------------------------------------------------------------

    def plot_boxplot(self,
                     experiment_name: str,
                     groups: Optional[List[str]] = None,
                     metric_name: str = "episode_return",
                     ylabel: str = "Discounted Return",
                     title: Optional[str] = None,
                     figsize: Tuple[int, int] = (8, 6),
                     grid: bool = True,
                     save_path: Optional[str] = None,
                     dpi: int = 300,
                     show_points: bool = True,
                     point_alpha: float = 0.6,
                     point_jitter: float = 0.05) -> plt.Figure:
        """
        Plot a box plot of final metric values across seeds per group.

        Intended for single-episode environments (e.g. RiverSwim) where only
        the terminal value is meaningful.
        """
        if groups is None:
            groups = self.group_order

        fig, ax = plt.subplots(figsize=figsize)

        group_values: List[np.ndarray] = []
        x_positions: List[int] = []
        x_labels: List[str] = []
        colors: List[str] = []

        for i, group in enumerate(groups):
            final_vals = self.load_group_final_values(
                experiment_name, group, metric_name
            )
            if len(final_vals) == 0:
                print(f"Skipping group '{group}' - no data")
                continue
            print(
                f"Group '{group}': {len(final_vals)} seeds, "
                f"mean={final_vals.mean():.4f}, std={final_vals.std():.4f}"
            )
            group_values.append(final_vals)
            x_positions.append(i)
            x_labels.append(self.group_labels.get(group, group))
            colors.append(self.group_colors.get(group, "#333333"))

        if not group_values:
            print("No data found for any group.")
            return fig

        bp = ax.boxplot(
            group_values,
            positions=x_positions,
            widths=0.5,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2.0),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker="o", markersize=4, alpha=0.5),
            zorder=2,
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
            patch.set_linewidth(1.2)

        # Draw mean as a horizontal line across each box and annotate value
        box_width = 0.5
        for vals, x, color in zip(group_values, x_positions, colors):
            mean_val = vals.mean()
            # ax.hlines(
            #     mean_val,
            #     x - box_width / 2, x + box_width / 2,
            #     colors=color, linewidths=2.0, linestyles="--",
            #     zorder=4, label="_nolegend_",
            # )
            ax.text(
                x - 0.1, 3.65e7, #3.82e6, #3.7e7,
                f"{mean_val:.2e}",
                va="center", ha="left",
                fontsize=9, color=color,
                zorder=5,
            )

        if show_points:
            rng = np.random.default_rng(42)
            for vals, x, color in zip(group_values, x_positions, colors):
                jitter = rng.uniform(-point_jitter, point_jitter, size=len(vals))
                ax.scatter(
                    x + jitter, vals,
                    color=color, alpha=point_alpha, s=30, zorder=3,
                    edgecolors="white", linewidths=0.5,
                )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=15, ha="right")
        ax.set_ylabel(ylabel, fontweight="bold")
        if title:
            ax.set_title(title, fontweight="bold")
        if grid:
            ax.yaxis.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)
            ax.set_axisbelow(True)
        ax.set_yscale('log')
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
            print(f"Saved figure to: {save_path}")

        return fig

    # ------------------------------------------------------------------
    # Convenience: multiple metrics
    # ------------------------------------------------------------------

    def plot_multiple_metrics(self,
                               experiment_name: str,
                               metrics: List[Tuple[str, str]],
                               groups: Optional[List[str]] = None,
                               **plot_kwargs) -> List[plt.Figure]:
        """
        Plot multiple metrics in separate figures.

        Args:
            metrics: List of (metric_name, ylabel) tuples.
        """
        figures = []
        for metric_name, ylabel in metrics:
            print(f"\nPlotting {metric_name}...")
            fig = self.plot_learning_curves(
                experiment_name=experiment_name,
                groups=groups,
                metric_name=metric_name,
                ylabel=ylabel,
                **plot_kwargs,
            )
            figures.append(fig)
        return figures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot learning curves with bootstrap confidence intervals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # File-based store (directory) — detected automatically
  python plot_learning_curves.py "my_experiment" --mlruns-path ./mlruns

  # SQLite database — detected automatically
  python plot_learning_curves.py "my_experiment" --mlruns-path ./mlflow.db

  # Box plot of final values (e.g. for RiverSwim)
  python plot_learning_curves.py "my_experiment" --boxplot --output boxplot.pdf

  # Customise groups and bootstrap
  python plot_learning_curves.py "my_experiment" \\
      --metric train/discounted_return \\
      --groups dqn drm rmax \\
      --n-bootstrap 5000 \\
      --output learning_curves.png

  # Show individual seeds
  python plot_learning_curves.py "my_experiment" --show-seeds --seed-alpha 0.15
        """,
    )

    parser.add_argument("experiment_name", help="Experiment name to plot")
    parser.add_argument(
        "--mlruns-path", default="./mlruns",
        help=(
            "Path to the MLflow store. Accepts either a directory (file-based "
            "store, e.g. ./mlruns) or an SQLite database file (e.g. ./mlflow.db). "
            "The backend is detected automatically. (default: ./mlruns)"
        ),
    )
    parser.add_argument(
        "--metric", default="train/discounted_return",
        help="Metric name to plot (default: train/discounted_return)",
    )
    parser.add_argument("--groups", nargs="+",
                        help="Groups to plot (default: all in GROUP_ORDER)")
    parser.add_argument("--num-points", type=int, default=500,
                        help="Number of interpolation points (default: 500)")
    parser.add_argument("--n-bootstrap", type=int, default=10000,
                        help="Number of bootstrap samples (default: 10000)")
    parser.add_argument("--ci", type=float, default=0.95,
                        help="Confidence interval (default: 0.95)")
    parser.add_argument("--xlabel", default="Steps", help="X-axis label")
    parser.add_argument("--ylabel", default="Discounted Return", help="Y-axis label")
    parser.add_argument("--title", help="Plot title")
    parser.add_argument("--figsize", nargs=2, type=int, default=[10, 6],
                        help="Figure size: width height (default: 10 6)")
    parser.add_argument("--no-grid", action="store_true", help="Disable grid")
    parser.add_argument("--legend-loc", default="best",
                        help="Legend location (default: best)")
    parser.add_argument("--output", "-o", default="plot.pdf",
                        help="Output file path (default: plot.pdf)")
    parser.add_argument("--dpi", type=int, default=300,
                        help="DPI for saved figure (default: 300)")
    parser.add_argument(
        "--show-seeds", action="store_true",
        help="Overlay individual seed trajectories (learning curve mode only)",
    )
    parser.add_argument("--seed-alpha", type=float, default=0.1,
                        help="Alpha for seed trajectories (default: 0.1)")
    parser.add_argument("--show", action="store_true",
                        help="Display plot interactively")

    # Box plot mode
    parser.add_argument(
        "--boxplot", action="store_true",
        help=(
            "Plot a box plot of final metric values instead of a learning curve. "
            "Useful for single-episode environments such as RiverSwim."
        ),
    )
    parser.add_argument("--no-points", action="store_true",
                        help="Hide individual seed points in box plot mode")
    parser.add_argument("--point-alpha", type=float, default=0.6,
                        help="Alpha for individual points in box plot (default: 0.6)")
    parser.add_argument("--point-jitter", type=float, default=0.05,
                        help="Horizontal jitter for individual points in box plot (default: 0.05)")

    args = parser.parse_args()

    plotter = LearningCurvePlotter(mlruns_path=args.mlruns_path)

    if args.boxplot:
        fig = plotter.plot_boxplot(
            experiment_name=args.experiment_name,
            groups=args.groups,
            metric_name=args.metric,
            ylabel=args.ylabel,
            title=args.title,
            figsize=tuple(args.figsize),
            grid=not args.no_grid,
            save_path=args.output,
            dpi=args.dpi,
            show_points=not args.no_points,
            point_alpha=args.point_alpha,
            point_jitter=args.point_jitter,
        )
    else:
        fig = plotter.plot_learning_curves(
            experiment_name=args.experiment_name,
            groups=args.groups,
            metric_name=args.metric,
            num_points=args.num_points,
            n_bootstrap=args.n_bootstrap,
            ci=args.ci,
            xlabel=args.xlabel,
            ylabel=args.ylabel,
            title=args.title,
            figsize=tuple(args.figsize),
            grid=not args.no_grid,
            legend_loc=args.legend_loc,
            save_path=args.output,
            dpi=args.dpi,
            show_seeds=args.show_seeds,
            seed_alpha=args.seed_alpha,
        )

    if args.show:
        plt.show()
    elif not args.output:
        print("\nUse --output to save or --show to display the plot")


if __name__ == "__main__":
    main()