
import argparse
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from dataclasses import dataclass
from typing import Annotated, List

import tyro

def plot_metrics(experiment_name, mlflow_tracking_uri="./mlruns.db", metric_name="train/discounted_return", filter_string=None, group_tag="group"):
    """
    Fetches runs from an mlflow experiment, groups them, and plots a metric
    with 95% CI.

    Args:
        experiment_name: The name of the mlflow experiment.
        mlflow_tracking_uri: The path to the mlflow registry.
        metric_name: The name of the metric to plot.
        filter_string: A string to filter runs.
        group_tag: The tag to group runs by.
    """
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

    plt.figure(figsize=(10, 6))
    
    all_metrics_data = []

    for group_name, group_data in grouped_runs:
        print(f"Processing group: {group_name}")
        
        group_metric_histories = []
        min_length = float('inf')

        for run_id in group_data["run_id"]:
            try:
                metric_history = client.get_metric_history(run_id, metric_name)
                group_metric_histories.append([step.value for step in metric_history])

                if metric_history is not None:
                    min_length = min(min_length, len(metric_history))
            except Exception as e:
                print(f"Could not retrieve metric '{metric_name}' for run '{run_id}': {e}")

        if not group_metric_histories:
            print(f"No metric data for '{metric_name}' in group '{group_name}'")
            continue
        
        # Truncate all histories to the minimum common length
        truncated_histories = [hist[:min_length] for hist in group_metric_histories]
        
        if not truncated_histories:
            print(f"No valid metric data to plot for group '{group_name}'.")
            continue

        metric_matrix = np.array(truncated_histories)

        # Normality Test (Shapiro-Wilk)
        # We test for normality at a few key points (e.g., start, middle, end)
        # because testing every single step could be overwhelming.
        print(f"\nNormality Test (Shapiro-Wilk) for group '{group_name}':")
        for step in [0, min_length // 2, min_length - 1]:
            if step < metric_matrix.shape[1]:
                stat, p_value = stats.shapiro(metric_matrix[:, step])
                print(f"  Step {step}: p-value={p_value:.3f}", end="")
                if p_value > 0.05:
                    print(" (Data appears normal)")
                else:
                    print(" (Data does not appear normal)")
        print("-" * 20)

        # Prepare data for Seaborn
        # We need a long-form DataFrame for sns.lineplot
        steps = np.arange(min_length)
        for history in truncated_histories:
            for step, value in zip(steps, history):
                all_metrics_data.append({"group": group_name, "step": step, "value": value})

    if not all_metrics_data:
        print("No data to plot.")
        return

    metrics_df = pd.DataFrame(all_metrics_data)

    # Plotting with Seaborn
    sns.set_theme(style="whitegrid")
    sns.lineplot(
        data=metrics_df,
        x="step",
        y="value",
        hue="group",
        errorbar=("ci", 95), #This calculates the 95% CI
    )

    plt.title(f"'{metric_name}' for Experiment '{experiment_name}'")
    plt.xlabel("Step")
    plt.ylabel(metric_name)
    plt.legend(title=group_tag)
    
    # Save the plot
    output_path = Path("outputs")
    output_path.mkdir(exist_ok=True)
    metric_name_safe = metric_name.replace("/", "_")
    filename = f"{experiment_name}_{metric_name_safe}.png"
    plt.savefig(output_path / filename, bbox_inches="tight", dpi=300)
    print(f"\nPlot saved to {output_path / filename}")
    
    plt.show()

@dataclass
class Args:
    """Arguments for running seeds locally and migrating MLflow runs."""

    experiment_name: Annotated[
        str,
        tyro.conf.arg(help="The name of the MLflow experiment."),
    ]
    metric_name: Annotated[
        str,
        tyro.conf.arg(help="The name of the metric to plot."),
    ] = "train/discounted_return"
    mlflow_tracking_uri: Annotated[
        str,
        tyro.conf.arg(help="The MLflow tracking URI."),
    ] = "sqlite:///mlruns.db"
    filter: Annotated[
        str | None,
        tyro.conf.arg(help="An MLflow filter string (e.g., \"params.learning_rate='0.01'\")."),
    ] = None
    group_tag: Annotated[
        str,
        tyro.conf.arg(help="The tag to group runs by (default: 'group')."),
    ] = "group"


if __name__ == "__main__":
    args = tyro.cli(Args)

    plot_metrics(
        args.experiment_name,
        args.mlflow_tracking_uri,
        args.metric_name,
        filter_string=args.filter,
        group_tag=args.group_tag,
    )
