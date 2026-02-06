
import argparse
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from dataclasses import dataclass
from typing import Annotated, List, Dict, Tuple

# =============================
# User-defined plot appearance
# =============================

GROUP_COLORS = {
    "dqn": "#1f77b4",
    "qlearning_epsgreedy": "#ff7f0e",
    "rmax": "#2ca02c",
    "replaybased_rmax": "#d62728",
}

GROUP_LABELS = {
    "dqn": "DQN",
    "qlearning_epsgreedy": "Q-learning (Epsilon-Greedy)",
    "rmax": "R-Max",
    "replaybased_rmax": "Replay-based R-Max",
}

# Order in legend (optional but recommended)
GROUP_ORDER = [
    "dqn",
    "qlearning_epsgreedy",
    "rmax",
    "replaybased_rmax",
]

# =============================
# Experiment settings
# =============================
# EXPERIMENT_NAME = "navix_5x5_layout1",
MLFLOW_TRACKING_URI = "sqlite:///mlruns.db"
METRIC_NAME = "train/discounted_return"
FILTER_STRING = None
GROUP_TAG = "group"

# =============================
# Configuration
# =============================
N_GRID = 200
N_BOOTSTRAP = 2000
CI_ALPHA = 0.05

FIGSIZE = (6.5, 4.2)      # ICML single-column
LINEWIDTH = 2.0
ALPHA_FILL = 0.2
RANDOM_SEED = 0

np.random.seed(RANDOM_SEED)


# =============================
# MLFlow data fetching
# =============================
def fetch_data(experiment_name, mlflow_tracking_uri="./mlruns.db", metric_name="train/discounted_return", filter_string=None, group_tag="group") -> pd.DataFrame:
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()

    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"Experiment '{experiment_name}' not found.")
            return
        experiment_id = experiment.experiment_id
    except Exception as e:
        print(f"Error getting experiment '{experiment_name}': {e}")
        return

    runs = mlflow.search_runs(experiment_ids=[experiment_id], filter_string=filter_string)
    
    if runs.empty:
        print("No runs found for the given criteria.")
        return

    if f"tags.{group_tag}" not in runs.columns:
        print(f"Tag '{group_tag}' not found in runs. Available tags are: {', '.join([c for c in runs.columns if c.startswith('tags.')])}")
        return

    grouped_runs = runs.groupby(f"tags.{group_tag}")

    all_group_metric_histories = []

    for group_name, group_data in grouped_runs:
        print(f"Processing group: {group_name}")
        # if group_name not in ["rmax", "factored_rmax", "batch_modelfree_rmax"]:
        #     continue
        # print(f"Processing group: {group_name}")
        # group_name = "R-max" if (group_name == "rmax") else "Misspecified R-max" if (group_name == "factored_rmax") else "Replay-based R-max (Ours)"
        
        
        group_metric_histories = []

        for run_id in group_data["run_id"]:
            try:
                metric_history = client.get_metric_history(run_id, metric_name)
                group_metric_histories.extend([{"group": group_name, "step": step.step, "value": step.value, "run_id": run_id} for step in metric_history])

                # if metric_history is not None:
                #     min_length = min(min_length, len(metric_history))
            except Exception as e:
                print(f"Could not retrieve metric '{metric_name}' for run '{run_id}': {e}")

        if not group_metric_histories:
            print(f"No metric data for '{metric_name}' in group '{group_name}'")
            continue

        all_group_metric_histories.append(group_metric_histories)

    return pd.DataFrame([d for sublist in all_group_metric_histories for d in sublist])

# =============================
# Interpolation
# =============================
def interpolate_run(
    run_df: pd.DataFrame,
    step_grid: np.ndarray,
) -> np.ndarray:
    run_df = run_df.sort_values("step")
    return np.interp(
        step_grid,
        run_df["step"].to_numpy(),
        run_df["value"].to_numpy(),
        left=np.nan,
        right=np.nan,
    )


def interpolate_all_runs(
    df: pd.DataFrame,
    step_grid: np.ndarray,
) -> Dict[str, np.ndarray]:
    curves = {}

    for group, gdf in df.groupby("group"):
        run_curves = []
        for _, rdf in gdf.groupby("run_id"):
            run_curves.append(interpolate_run(rdf, step_grid))
        curves[group] = np.stack(run_curves, axis=0)

    return curves


# =============================
# Bootstrap statistics
# =============================
def bootstrap_mean_ci(
    x: np.ndarray,
    n_boot: int,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_runs, n_steps = x.shape

    mean = np.nanmean(x, axis=0)

    boot_means = np.empty((n_boot, n_steps))
    for i in range(n_boot):
        idx = np.random.choice(n_runs, size=n_runs, replace=True)
        boot_means[i] = np.nanmean(x[idx], axis=0)

    low = np.percentile(boot_means, 100 * alpha / 2, axis=0)
    high = np.percentile(boot_means, 100 * (1 - alpha / 2), axis=0)

    return mean, low, high


# =============================
# Main plotting function
# =============================
def plot_learning_curves(df: pd.DataFrame):
    # Step grid
    step_grid = np.linspace(df.step.min(), df.step.max(), N_GRID)

    # Interpolate
    curves = interpolate_all_runs(df, step_grid)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for group in GROUP_ORDER:
        if group not in curves:
            continue

        x = curves[group]
        mean, low, high = bootstrap_mean_ci(
            x,
            n_boot=N_BOOTSTRAP,
            alpha=CI_ALPHA,
        )

        color = GROUP_COLORS.get(group, None)
        label = GROUP_LABELS.get(group, group)

        ax.plot(
            step_grid,
            mean,
            color=color,
            linewidth=LINEWIDTH,
            label=label,
        )
        ax.fill_between(
            step_grid,
            low,
            high,
            color=color,
            alpha=ALPHA_FILL,
        )

    # Axes styling
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Discounted Episodic Return")
    ax.grid(True, alpha=0.3)

    # Legend ABOVE the plot
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(GROUP_ORDER),
        frameon=False,
        columnspacing=1.5,
        handlelength=2.5,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig, ax

if __name__ == "__main__":
    for EXPERIMENT_NAME in [
        "navix_5x5_layout1",
        # "navix_5x5_layout2",
        # "navix_5x5_layout3",
        # "navix_16x16_layout1",
        # "navix_16x16_layout2",
        # "navix_16x16_layout3",
    ]:
        df = fetch_data(
            EXPERIMENT_NAME,
            MLFLOW_TRACKING_URI,
            METRIC_NAME,
            FILTER_STRING,
            group_tag=GROUP_TAG,
        )
        fig, ax = plot_learning_curves(df)
        plt.show()
        fig.savefig(f"{EXPERIMENT_NAME}_learning_curves.pdf")