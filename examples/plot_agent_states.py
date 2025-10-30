"""Visualise per-episode Q-values from an MLflow experiment run."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.widgets import Slider
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot tabular agent Q-values recorded as MLflow artifacts."
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Name of the MLflow experiment containing the run.",
    )
    parser.add_argument(
        "--parent-run",
        required=True,
        help="Display name of the parent (non-nested) run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed index to visualise (default: 0).",
    )
    parser.add_argument(
        "--state-shape",
        type=int,
        nargs="+",
        default=[21, 3, 2, 2],
        help="Reshape for the state index prior to plotting (default: 21 3 2 2).",
    )
    parser.add_argument(
        "--tracking-uri",
        help="Override the MLflow tracking URI (defaults to local ./mlruns).",
    )
    parser.add_argument(
        "--artifact-path",
        default="artifacts/agent_states.npz",
        help="Relative artifact path for agent states (default: artifacts/agent_states.npz).",
    )
    parser.add_argument("--metric", choices=["q_values", "sa_counts"], default="q_values", help="Which agent state tensor to visualise (default: q_values).")
    parser.add_argument("--start-episode", type=int, default=0, help="Episode index to show on startup (default: 0).")
    parser.add_argument("--m", type=int, default=None, help="Number of visits to consider state-action known.")
    parser.add_argument("--save-pdf", help="Optional path to export every episode as a PDF slideshow.")
    parser.add_argument("--no-gui", action="store_true", help="Skip the interactive window (useful when only exporting).")
    parser.add_argument(
        "--action-labels",
        help="Comma-separated action labels (default uses action indices).",
    )
    parser.add_argument(
        "--dimension-labels",
        help="Comma-separated labels for each state dimension (default Dim 0, Dim 1, ...).",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=2000,
        help="Maximum number of runs to inspect when resolving the parent run.",
    )
    return parser.parse_args()


def _escape_for_filter(value: str) -> str:
    return value.replace("'", "''")


def _resolve_experiment(client: MlflowClient, name: str):
    experiment = client.get_experiment_by_name(name)
    if experiment is None:
        raise RuntimeError(f"Experiment '{name}' was not found.")
    return experiment


def _resolve_parent_run(
    client: MlflowClient, experiment_id: str, parent_run_name: str, max_runs: int
):
    escaped_name = _escape_for_filter(parent_run_name)
    filter_string = f"attribute.run_name = '{escaped_name}'"
    runs = client.search_runs(
        [experiment_id],
        filter_string=filter_string,
        max_results=max_runs,
        order_by=["attribute.start_time DESC"],
    )
    if not runs:
        raise RuntimeError(
            f"No runs named '{parent_run_name}' found in experiment id {experiment_id}."
        )
    def is_parent(run) -> bool:
        parent_tag = run.data.tags.get("parent")
        mlflow_parent = run.data.tags.get("mlflow.parentRunId")
        return not parent_tag and not mlflow_parent
    for run in runs:
        if is_parent(run):
            return run
    return runs[0]


def _download_agent_states(run_id: str, artifact_path: str) -> Path:
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=artifact_path
    )
    resolved = Path(local_path)
    if resolved.is_dir():
        candidate = resolved / "agent_states.npz"
        if candidate.exists():
            resolved = candidate
    if not resolved.exists():
        raise RuntimeError(f"Could not locate agent_states artifact at '{resolved}'.")
    return resolved


def _load_tensor(npz_path: Path, key: str) -> np.ndarray:
    with np.load(npz_path) as data:
        if key not in data:
            raise RuntimeError(f"'{key}' array missing from {npz_path}.")
        tensor = np.asarray(data[key])
    if tensor.ndim < 3:
        raise RuntimeError(
            f"Expected {key} with at least 3 dims (found {tensor.ndim})."
        )
    if tensor.ndim == 3:
        tensor = np.expand_dims(tensor, axis=0)
    return tensor


def _parse_labels(raw: str | None, expected: int, prefix: str) -> List[str]:
    if raw is None:
        return [f"{prefix}{idx}" for idx in range(expected)]
    labels = [part.strip() for part in raw.split(",") if part.strip()]
    if len(labels) != expected:
        raise RuntimeError(
            f"Expected {expected} {prefix.lower()} labels but received {len(labels)}."
        )
    return labels


def _parse_dimension_labels(
    raw: str | None, state_shape: Sequence[int]
) -> List[str]:
    if raw is None:
        labels: List[str] = []
        for idx in range(len(state_shape)):
            if idx == 0:
                labels.append("Length")
            elif idx == 1:
                labels.append("Status")
            else:
                labels.append(f"Dim {idx}")
        return labels

    labels = [part.strip() for part in raw.split(",") if part.strip()]
    if len(labels) != len(state_shape):
        raise RuntimeError(
            f"Expected {len(state_shape)} dimension labels, received {len(labels)}."
        )
    return labels


def _other_dimension_combos(
    state_shape: Sequence[int], dimension_labels: Sequence[str]
) -> tuple[List[tuple[int, ...]], List[str]]:
    other_dims = state_shape[2:]
    if not other_dims:
        return [()], ["All"]

    combos = list(np.ndindex(*other_dims))
    if len(dimension_labels) > 2:
        other_names = dimension_labels[2:]
    else:
        other_names = [f"Dim {idx}" for idx in range(2, 2 + len(other_dims))]

    if len(other_names) != len(other_dims):
        other_names = [f"Dim {idx}" for idx in range(2, 2 + len(other_dims))]

    all_binary = all(size == 2 for size in other_dims)
    combo_labels: List[str] = []
    for combo in combos:
        if all_binary:
            combo_labels.append("".join(str(int(val)) for val in combo))
        else:
            parts = [
                f"{other_names[i]}={int(value)}" for i, value in enumerate(combo)
            ]
            combo_labels.append(", ".join(parts))
    return combos, combo_labels


def _iter_grid_cells(grid: np.ndarray):
    rows, cols = grid.shape[:2]
    for r in range(rows):
        for c in range(cols):
            yield grid[r, c]


def _length_grid_values(
    metric_values: np.ndarray,
    state_shape: Sequence[int],
    combos: Sequence[tuple[int, ...]],
) -> np.ndarray:
    reshaped = np.reshape(metric_values, tuple(state_shape) + (metric_values.shape[-1],))
    length_dim = state_shape[0]
    status_dim = state_shape[1] if len(state_shape) > 1 else 1
    num_actions = metric_values.shape[-1]

    if not combos:
        combos = [()]

    grid = np.zeros(
        (len(combos), status_dim, num_actions, length_dim), dtype=reshaped.dtype
    )
    for row_idx, combo in enumerate(combos):
        for status_idx in range(status_dim):
            indices: List[object] = [slice(None)]
            if len(state_shape) > 1:
                indices.append(status_idx)
            indices.extend(combo)
            indices.append(slice(None))
            matrix = reshaped[tuple(indices)]
            matrix = np.reshape(matrix, (length_dim, num_actions))
            grid[row_idx, status_idx] = matrix.T
    return grid


def _create_length_grid_figure(
    grid_values: np.ndarray,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    action_labels: Sequence[str],
    length_label: str,
    vmin: float,
    vmax: float,
    suptitle: str | None = None,
    colorbar_label: str = "Value",
    cmap_name: str = "viridis",
) -> tuple[plt.Figure, np.ndarray, List[plt.AxesImage]]:
    rows, cols, num_actions, length_dim = grid_values.shape
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4.2 * max(cols, 1), 2.6 * max(rows, 1)),
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.12,
        right=0.92,
        bottom=0.12,
        top=0.9,
        hspace=0.35,
        wspace=0.3,
    )
    images: List[plt.AxesImage] = []
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            image = ax.imshow(
                grid_values[r, c],
                cmap=cmap_name,
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_xticks(np.arange(length_dim + 1) - 0.5, minor=True)
            ax.set_yticks(np.arange(num_actions + 1) - 0.5, minor=True)
            ax.grid(which="minor", color="white", linewidth=0.5)
            ax.tick_params(which="minor", bottom=False, left=False)
            if r == 0:
                ax.set_title(column_labels[c])
            ax.set_yticks(np.arange(len(action_labels)))
            if c == 0:
                ax.set_yticklabels(action_labels)
                ax.set_ylabel(row_labels[r], rotation=90, labelpad=20)
            else:
                ax.set_yticklabels([])
            if r == rows - 1:
                tick_positions = (
                    np.arange(length_dim)
                    if length_dim <= 25
                    else np.linspace(0, length_dim - 1, num=min(10, length_dim), dtype=int)
                )
                ax.set_xticks(tick_positions)
                ax.set_xlabel(length_label)
            else:
                ax.set_xticks([])
            images.append(image)

    cbar = fig.colorbar(images[0], ax=axes.ravel().tolist(), fraction=0.045, pad=0.02)
    cbar.set_label(colorbar_label)
    if suptitle:
        fig.suptitle(suptitle, y=0.97)
    return fig, axes, images


def _build_static_figure(
    metric_values: np.ndarray,
    episode_index: int,
    state_shape: Sequence[int],
    combos: Sequence[tuple[int, ...]],
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    action_labels: Sequence[str],
    length_label: str,
    vmin: float,
    vmax: float,
    suptitle: str,
    colorbar_label: str,
    cmap_name: str
) -> plt.Figure:
    grid = _length_grid_values(metric_values, state_shape, combos)
    fig, _, _ = _create_length_grid_figure(
        grid,
        row_labels,
        column_labels,
        action_labels,
        length_label,
        vmin,
        vmax,
        f"{suptitle} — episode {episode_index}",
        colorbar_label,
        cmap_name=cmap_name
    )
    return fig


def _interactive_plot(
    metric_series: np.ndarray,
    state_shape: Sequence[int],
    combos: Sequence[tuple[int, ...]],
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    action_labels: Sequence[str],
    length_label: str,
    vmin: float,
    vmax: float,
    suptitle: str,
    start_episode: int,
    colorbar_label: str,
    cmap_name: str
) -> None:
    num_episodes = metric_series.shape[0]
    grid = _length_grid_values(metric_series[start_episode], state_shape, combos)
    fig, axes, images = _create_length_grid_figure(
        grid,
        row_labels,
        column_labels,
        action_labels,
        length_label,
        vmin,
        vmax,
        f"{suptitle} — episode {start_episode}",
        colorbar_label,
        cmap_name
    )
    fig.subplots_adjust(bottom=0.24, right=0.86)

    slider_ax = fig.add_axes([0.12, 0.06, 0.76, 0.04])
    slider = Slider(
        slider_ax,
        "Episode",
        valmin=0,
        valmax=num_episodes - 1,
        valinit=start_episode,
        valstep=1,
    )

    def _on_change(val: float) -> None:
        idx = int(slider.val)
        updated = _length_grid_values(metric_series[idx], state_shape, combos)
        for image, data in zip(images, _iter_grid_cells(updated)):
            image.set_data(data)
        fig.suptitle(f"{suptitle} — episode {idx}")
        fig.canvas.draw_idle()

    slider.on_changed(_on_change)
    plt.show()


def main() -> None:
    args = _parse_args()

    state_shape = tuple(args.state_shape)
    if not state_shape:
        raise RuntimeError("state-shape must contain at least one dimension.")

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    client = MlflowClient()
    experiment = _resolve_experiment(client, args.experiment)
    parent_run = _resolve_parent_run(
        client, experiment.experiment_id, args.parent_run, args.max_runs
    )
    run_id = parent_run.info.run_id

    artifact_path = _download_agent_states(run_id, args.artifact_path)
    metric_tensor = _load_tensor(artifact_path, args.metric)

    num_seeds, num_episodes, num_states, num_actions = metric_tensor.shape
    if np.prod(state_shape) != num_states:
        raise RuntimeError(
            f"state-shape {state_shape} does not match stored num_states={num_states}."
        )
    if args.seed < 0 or args.seed >= num_seeds:
        raise RuntimeError(
            f"Seed index {args.seed} out of bounds for {num_seeds} seeds."
        )
    selected = metric_tensor[args.seed]
    if num_episodes == 0:
        raise RuntimeError("No agent states were logged for this run.")
    start_episode = int(np.clip(args.start_episode, 0, num_episodes - 1))

    action_labels = _parse_labels(args.action_labels, num_actions, "A")
    dimension_labels = _parse_dimension_labels(args.dimension_labels, state_shape)
    status_dim = state_shape[1] if len(state_shape) > 1 else 1
    if len(state_shape) > 1:
        status_label = dimension_labels[1]
        column_labels = [f"{status_label}={idx}" for idx in range(status_dim)]
    else:
        column_labels = ["All"]
    combos, row_labels = _other_dimension_combos(state_shape, dimension_labels)
    length_label = dimension_labels[0]

    label = f"{args.experiment} · {args.parent_run} · seed {args.seed}"
    
    if args.metric == "q_values":
        largest = np.maximum(np.abs(float(np.max(selected))), np.abs(float(np.min(selected))))
        vmin = -largest
        vmax = largest
        colorbar_label = "Q-value"
        cmap_name = "RdBu"
    else:
        if args.m is not None:
            colorbar_label = "Is known"
            selected = np.array(selected >= args.m, dtype=int)
        else:
            colorbar_label = "State-Action Count"

        vmin = float(np.min(selected))
        vmax = float(np.max(selected))
        cmap_name = "viridis"

    if args.save_pdf:
        out_path = Path(args.save_pdf)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with PdfPages(out_path) as pdf:
            for episode_idx in range(num_episodes):
                fig = _build_static_figure(
                    selected[episode_idx],
                    episode_idx,
                    state_shape,
                    combos,
                    row_labels,
                    column_labels,
                    action_labels,
                    length_label,
                    vmin,
                    vmax,
                    label,
                    colorbar_label,
                    cmap_name=cmap_name
                )
                pdf.savefig(fig)
                plt.close(fig)
        print(f"Saved PDF with {num_episodes} pages to {out_path}")

    if not args.no_gui:
        _interactive_plot(
            selected,
            state_shape,
            combos,
            row_labels,
            column_labels,
            action_labels,
            length_label,
            vmin,
            vmax,
            label,
            start_episode,
            colorbar_label,
            cmap_name=cmap_name
        )


if __name__ == "__main__":
    main()
