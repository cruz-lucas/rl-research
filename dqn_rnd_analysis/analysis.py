import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _as_float(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        raw = row.get(key, "")
        if raw in {"", None}:
            values.append(np.nan)
        elif isinstance(raw, str) and raw.lower() in {"true", "false"}:
            values.append(1.0 if raw.lower() == "true" else 0.0)
        else:
            values.append(float(raw))
    return np.asarray(values, dtype=np.float32)


def _state_title(state_key: str, state_payload: dict[str, Any]) -> str:
    metadata = state_payload.get("metadata", {})
    label = metadata.get("label")
    if label:
        return f"state {state_key}: {label}"
    return f"state {state_key}"


def plot_intrinsic_ratio_over_time(output_dir: Path) -> None:
    rows = _load_csv_rows(output_dir / "batch_summary.csv")
    if not rows:
        return

    x = _as_float(rows, "env_step")
    ratio = _as_float(rows, "ratio_mean")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, ratio, linewidth=1.5)
    ax.set_title("Intrinsic / Extrinsic Contribution Ratio")
    ax.set_xlabel("Environment Step")
    ax.set_ylabel("Mean Ratio")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "ratio_over_time.png", dpi=160)
    plt.close(fig)


def plot_gradient_norms(output_dir: Path) -> None:
    rows = _load_csv_rows(output_dir / "gradient_stats.csv")
    if not rows:
        return

    x = _as_float(rows, "env_step")
    g_full = _as_float(rows, "g_full_norm")
    g_no_intrinsic = _as_float(rows, "g_no_intrinsic_norm")
    g_intrinsic = _as_float(rows, "g_intrinsic_only_norm")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, g_full, label="||g_full||")
    ax.plot(x, g_intrinsic, label="||g_int||")
    ax.plot(x, g_no_intrinsic, label="||g_no_int||")
    ax.set_title("Gradient Attribution Norms")
    ax.set_xlabel("Environment Step")
    ax.set_ylabel("Norm")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "gradient_norms.png", dpi=160)
    plt.close(fig)


def plot_clipping_frequency(output_dir: Path, window: int = 50) -> None:
    rows = _load_csv_rows(output_dir / "optimizer_stats.csv")
    if not rows:
        return

    x = _as_float(rows, "env_step")
    clipped = _as_float(rows, "was_clipped")
    if clipped.size == 0:
        return

    effective_window = min(max(1, int(window)), int(clipped.size))
    kernel = np.ones(effective_window, dtype=np.float32) / effective_window
    rolling = np.convolve(clipped, kernel, mode="same")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, rolling, linewidth=1.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Gradient Clipping Frequency")
    ax.set_xlabel("Environment Step")
    ax.set_ylabel(f"Rolling Clip Rate (window={window})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "clipping_frequency.png", dpi=160)
    plt.close(fig)


def plot_state_histories(output_dir: Path, top_k_states: int = 6) -> None:
    state_stats = _load_json(output_dir / "state_stats.json")
    if not state_stats:
        return

    ranked_states = sorted(
        state_stats.items(),
        key=lambda item: int(item[1].get("count", 0)),
        reverse=True,
    )
    selected = [item for item in ranked_states[:top_k_states] if item[1]["count"] > 0]
    if not selected:
        return

    fig_intrinsic, intrinsic_axes = plt.subplots(
        len(selected),
        1,
        figsize=(12, 3 * len(selected)),
        squeeze=False,
    )
    for axis, (state_key, payload) in zip(intrinsic_axes[:, 0], selected):
        history = payload.get("r_int_history", [])
        steps = [item["step"] for item in history]
        values = [item["value"] for item in history]
        axis.plot(steps, values, linewidth=1.25)
        axis.set_title(_state_title(state_key, payload))
        axis.set_xlabel("Environment Step")
        axis.set_ylabel("Intrinsic Reward")
        axis.grid(alpha=0.3)

    fig_intrinsic.tight_layout()
    fig_intrinsic.savefig(output_dir / "state_intrinsic_histories.png", dpi=160)
    plt.close(fig_intrinsic)

    fig_q, q_axes = plt.subplots(
        len(selected),
        1,
        figsize=(12, 3.5 * len(selected)),
        squeeze=False,
    )
    for axis, (state_key, payload) in zip(q_axes[:, 0], selected):
        history = payload.get("q_history", [])
        if not history:
            continue
        steps = np.asarray([item["step"] for item in history], dtype=np.float32)
        q_matrix = np.asarray([item["q_values"] for item in history], dtype=np.float32)
        q_mean = np.asarray([item["q_mean"] for item in history], dtype=np.float32)

        for action_index in range(q_matrix.shape[1]):
            axis.plot(
                steps,
                q_matrix[:, action_index],
                alpha=0.4,
                linewidth=0.9,
                label=f"a{action_index}",
            )
        axis.plot(steps, q_mean, color="black", linewidth=2.0, label="mean Q")
        axis.set_title(_state_title(state_key, payload))
        axis.set_xlabel("Environment Step")
        axis.set_ylabel("Q-value")
        axis.grid(alpha=0.3)
        axis.legend(ncol=min(4, q_matrix.shape[1] + 1), fontsize=8)

    fig_q.tight_layout()
    fig_q.savefig(output_dir / "state_qvalue_histories.png", dpi=160)
    plt.close(fig_q)


def plot_replay_drift_vs_age(output_dir: Path) -> None:
    rows = _load_csv_rows(output_dir / "replay_diagnostics.csv")
    if not rows:
        return

    age = _as_float(rows, "age")
    drift = _as_float(rows, "drift")
    if age.size == 0:
        return

    order = np.argsort(age)
    age_sorted = age[order]
    drift_sorted = drift[order]
    if np.allclose(np.min(age_sorted), np.max(age_sorted)):
        return

    num_bins = min(40, max(5, int(np.sqrt(len(age_sorted)))))
    bins = np.linspace(np.min(age_sorted), np.max(age_sorted), num_bins + 1)
    binned_age = []
    binned_drift = []
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (age_sorted >= left) & (age_sorted <= right)
        if not np.any(mask):
            continue
        binned_age.append(float(np.mean(age_sorted[mask])))
        binned_drift.append(float(np.mean(drift_sorted[mask])))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(age, drift, s=8, alpha=0.12, label="samples")
    if binned_age:
        ax.plot(
            binned_age,
            binned_drift,
            color="black",
            linewidth=2,
            label="binned mean",
        )
    ax.set_title("Replay Intrinsic Reward Drift vs Transition Age")
    ax.set_xlabel("Transition Age")
    ax.set_ylabel("|r_int_current - r_int_stored|")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "replay_drift_vs_age.png", dpi=160)
    plt.close(fig)


def plot_q_correlations(output_dir: Path) -> None:
    rows = _load_csv_rows(output_dir / "correlation_stats.csv")
    if not rows:
        return

    x = _as_float(rows, "env_step")
    visit_corr = _as_float(rows, "q_visit_corr_visited_states")
    intrinsic_corr = _as_float(rows, "q_intrinsic_corr_visited_states")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, visit_corr, label="corr(mean Q, visits)")
    ax.plot(x, intrinsic_corr, label="corr(mean Q, cumulative intrinsic)")
    ax.set_title("State-Level Q Correlations")
    ax.set_xlabel("Environment Step")
    ax.set_ylabel("Pearson Correlation")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "q_correlations.png", dpi=160)
    plt.close(fig)


def generate_all_plots(output_dir: Path, top_k_states: int = 6) -> None:
    output_dir = Path(output_dir)
    plot_intrinsic_ratio_over_time(output_dir)
    plot_gradient_norms(output_dir)
    plot_clipping_frequency(output_dir)
    plot_state_histories(output_dir, top_k_states=top_k_states)
    plot_replay_drift_vs_age(output_dir)
    plot_q_correlations(output_dir)
