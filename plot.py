import argparse
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class SeriesStats:
    auc: float
    last_n_mean: float
    full_mean: float


def trapezoid_auc(steps: np.ndarray, values: np.ndarray) -> float:
    """Compute area under curve using the trapezoid rule over (step, value)."""
    if len(steps) == 0:
        return np.nan
    # Ensure increasing by step
    order = np.argsort(steps)
    s = steps[order].astype(float)
    v = values[order].astype(float)
    # Remove duplicate steps, keep the last occurrence
    uniq_s, uniq_idx = np.unique(s, return_index=True)
    s = s[np.sort(uniq_idx)]
    v = v[np.sort(uniq_idx)]
    if len(s) == 1:
        return 0.0
    return float(np.trapezoid(v, s))


def last_n_mean(values: np.ndarray, n: int) -> float:
    if len(values) == 0:
        return np.nan
    n = min(n, len(values))
    return float(np.mean(values[-n:]))


def series_stats(steps: List[int], values: List[float], last_n: int) -> SeriesStats:
    arr_steps = np.asarray(steps, dtype=float)
    arr_vals = np.asarray(values, dtype=float)
    return SeriesStats(
        auc=trapezoid_auc(arr_steps, arr_vals),
        last_n_mean=last_n_mean(arr_vals, last_n),
        full_mean=float(np.mean(arr_vals)) if len(arr_vals) else np.nan,
    )


def standard_error(vals: List[float]) -> float:
    arr = np.asarray(vals, dtype=float)
    if len(arr) <= 1:
        return float("nan")
    return float(np.std(arr, ddof=1) / math.sqrt(len(arr)))


def fetch_metric_history(client: MlflowClient, run_id: str, metric_key: str) -> Tuple[List[int], List[float]]:
    """Fetch full metric history (step, value) for a metric_key from a run."""
    history = client.get_metric_history(run_id, metric_key)
    steps = [m.step for m in history]
    values = [m.value for m in history]
    return steps, values


def load_all_runs(experiment_id: str, filter_str: Optional[str], client: MlflowClient, max_results: int = 50000):
    """Paginate through mlflow.search_runs to get all runs for an experiment."""
    all_runs = []
    page_token = None
    while True:
        df = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string=filter_str,
            max_results=max_results,
            # page_token=page_token,
            output_format="pandas",
        )
        if df is None or df.empty:
            break
        all_runs.append(df)
        try:
            page_token = df._info_dict.get("next_page_token")  # type: ignore[attr-defined]
        except Exception:
            page_token = None
        if not page_token:
            break
    if not all_runs:
        return pd.DataFrame()
    return pd.concat(all_runs, ignore_index=True)


def infer_parent_groups(runs_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Group runs by parent (config) using tag.mlflow.parentRunId.
    Returns: dict parent_run_id -> list of child run ids.
    If a run has no parent, we treat the run itself as its own parent group.
    """
    parent_tag_col = "tags.mlflow.parentRunId"
    groups = defaultdict(list)
    for _, row in runs_df.iterrows():
        rid = row["run_id"]
        parent = row.get(parent_tag_col, None)
        if isinstance(parent, float) and np.isnan(parent):
            parent = None
        if parent and str(parent).strip():
            groups[str(parent)].append(rid)
        else:
            groups[rid].append(rid)
    return groups


def get_parent_name_and_params(client: MlflowClient, parent_id: str) -> Tuple[str, Dict[str, str]]:
    """Fetch parent run name and params for context in the table."""
    try:
        pr = client.get_run(parent_id)
        name = pr.data.tags.get("mlflow.runName", parent_id)
        params = dict(pr.data.params)
        return name, params
    except Exception:
        return parent_id, {}  # Fallback


def compute_group_summary(
    client: MlflowClient,
    parent_id: str,
    child_run_ids: List[str],
    metric_keys: Tuple[str, str],
    last_n: int,
) -> Dict:
    eval_key, train_key = metric_keys
    # Collect per-seed summaries
    per_seed = []
    for rid in child_run_ids:
        eval_steps, eval_vals = fetch_metric_history(client, rid, eval_key)
        train_steps, train_vals = fetch_metric_history(client, rid, train_key) if train_key else ([], [])
        per_seed.append({
            "run_id": rid,
            "eval": series_stats(eval_steps, eval_vals, last_n) if eval_vals else SeriesStats(np.nan, np.nan, np.nan),
            "train": series_stats(train_steps, train_vals, last_n) if train_vals else SeriesStats(np.nan, np.nan, np.nan),
        })

    # Aggregate across seeds
    def agg(which: str) -> Dict[str, float]:
        aucs = [getattr(s[which], "auc") for s in per_seed if not math.isnan(s[which].auc)]
        last_means = [getattr(s[which], "last_n_mean") for s in per_seed if not math.isnan(s[which].last_n_mean)]
        full_means = [getattr(s[which], "full_mean") for s in per_seed if not math.isnan(s[which].full_mean)]
        return {
            f"{which}_auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
            f"{which}_auc_se": standard_error(aucs) if aucs else float("nan"),
            f"{which}_last{last_n}_mean": float(np.mean(last_means)) if last_means else float("nan"),
            f"{which}_last{last_n}_se": standard_error(last_means) if last_means else float("nan"),
            f"{which}_full_mean": float(np.mean(full_means)) if full_means else float("nan"),
            f"{which}_full_se": standard_error(full_means) if full_means else float("nan"),
            f"{which}_n_seeds": len(per_seed),
        }

    parent_name, parent_params = get_parent_name_and_params(client, parent_id)
    summary = {
        "parent_run_id": parent_id,
        "parent_name": parent_name,
    }
    summary.update(agg("eval"))
    summary.update(agg("train"))
    # Flatten a subset of params for readability
    for k, v in parent_params.items():
        if len(str(v)) <= 80:
            summary[f"param.{k}"] = v
    return summary


def plot_top_runs(df: pd.DataFrame, top_ids: List[str], client: MlflowClient, metric_key: str, out_path: str):
    """Plot metric curves for the top groups (mean across seeds per parent)."""
    plt.figure(figsize=(10, 6))

    for parent_id in top_ids:
        # Build a step -> list of values map across child seeds
        seed_series: Dict[int, List[float]] = defaultdict(list)
        try:
            child_df = mlflow.search_runs(
                experiment_ids=[df.attrs["experiment_id"]],
                filter_string=f"tags.mlflow.parentRunId = '{parent_id}' OR run_id = '{parent_id}'",
                max_results=100000,
                output_format="pandas",
            )
        except Exception:
            child_df = pd.DataFrame(columns=["run_id"])

        child_ids = child_df["run_id"].tolist() if not child_df.empty else [parent_id]
        for rid in child_ids:
            steps, vals = fetch_metric_history(client, rid, metric_key)
            for s, v in zip(steps, vals):
                seed_series[int(s)].append(float(v))

        if not seed_series:
            continue

        steps_sorted = sorted(seed_series.keys())
        means = [float(np.mean(seed_series[s])) for s in steps_sorted]
        plt.plot(steps_sorted, means, label=df.loc[df["parent_run_id"] == parent_id, "parent_name"].values[0])

    plt.xlabel("Step")
    plt.ylabel(metric_key)
    plt.title(f"Top {len(top_ids)} by {metric_key} (mean across seeds)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Summarize MLflow runs across seeds by parent config.")
    parser.add_argument("--tracking-uri", type=str, default=None, help="MLflow tracking URI. If omitted, use env/default.")
    parser.add_argument("--experiment-name", type=str, default="double_goright_ucb_dtp_sweep", help="Experiment name.")
    parser.add_argument("--experiment-id", type=str, default=None, help="Experiment ID (overrides name if provided).")
    parser.add_argument("--eval-metric", type=str, default="eval/discounted_return", help="Eval metric key.")
    parser.add_argument("--train-metric", type=str, default="train/discounted_return", help="Train metric key.")
    parser.add_argument("--last-n", type=int, default=100, help="Last-N episodes to average.")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K groups to plot.")
    parser.add_argument("--out-csv", type=str, default="mlflow_summary.csv")
    parser.add_argument("--out-plot", type=str, default="top10_eval_curve.png")
    parser.add_argument("--filter", type=str, default=None, help="Additional MLflow filter string (e.g., status = 'FINISHED').")

    args = parser.parse_args()

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    client = MlflowClient()

    # Resolve experiment id
    if args.experiment_id:
        exp_id = args.experiment_id
    elif args.experiment_name:
        exp = client.get_experiment_by_name(args.experiment_name)
        if exp is None:
            raise RuntimeError(f"Experiment named '{args.experiment_name}' not found.")
        exp_id = exp.experiment_id
    else:
        raise RuntimeError("You must provide --experiment-id or --experiment-name.")

    # Load ALL runs for the experiment
    runs_df = load_all_runs(exp_id, args.filter, client)
    if runs_df.empty:
        print("No runs found for the given experiment/filter.")
        return

    # Group by parent
    groups = infer_parent_groups(runs_df)

    # Compute summaries
    summaries = []
    for parent_id, child_ids in groups.items():
        try:
            summary = compute_group_summary(
                client=client,
                parent_id=parent_id,
                child_run_ids=child_ids,
                metric_keys=(args.eval_metric, args.train_metric),
                last_n=args.last_n,
            )
            summaries.append(summary)
        except Exception as e:
            print(f"Warning: failed to summarize parent {parent_id}: {e}")

    if not summaries:
        print("No summaries computed.")
        return

    df = pd.DataFrame(summaries)

    # Ranking: by eval last-N mean (desc), then by eval AUC (desc)
    last_col = f"eval_last{args.last_n}_mean"
    auc_col = "eval_auc_mean"
    sort_cols = [c for c in [last_col, auc_col] if c in df.columns]
    df_sorted = df.sort_values(by=sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)

    # Attach experiment_id for plotting helper (used in plot_top_runs)
    df_sorted.attrs["experiment_id"] = exp_id

    # Save CSV
    df_sorted.to_csv(args.out_csv, index=False)

    # Print a condensed view
    display_cols = [
        "parent_name", "parent_run_id",
        last_col, f"eval_last{args.last_n}_se", auc_col, "eval_auc_se",
        "eval_full_mean", "eval_full_se",
        "train_full_mean", "train_full_se",
        "eval_n_seeds", "train_n_seeds",
    ]
    display_cols = [c for c in display_cols if c in df_sorted.columns]
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(df_sorted[display_cols].head(50).to_string(index=False))

    # # Plot top-K
    # top_k = min(args.top_k, len(df_sorted))
    # top_ids = df_sorted["parent_run_id"].head(top_k).tolist()
    # try:
    #     plot_top_runs(df_sorted, top_ids, client, args.eval_metric, args.out_plot)
    #     print(f"\\nSaved plot to: {args.out_plot}")
    # except Exception as e:
    #     print(f"Warning: failed to plot top {top_k}: {e}")

    print(f"Saved summary CSV to: {args.out_csv}")

if __name__ == "__main__":
    main()