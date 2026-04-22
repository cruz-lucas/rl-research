from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.collections import PolyCollection

from rl_research.analysis.navix_knownness_collection import (
    POLICY_NAMES,
    PolicyRollout,
    compute_policy_summary,
    load_collection_metadata,
    load_policy_rollout,
    save_policy_summary,
)


@dataclass(frozen=True)
class PlotThresholds:
    visitation_threshold: int
    bonus_threshold: float


def _triangles_for_cell(row: int, col: int) -> tuple[np.ndarray, ...]:
    center = np.asarray([col + 0.5, row + 0.5], dtype=np.float32)
    top_left = np.asarray([col, row], dtype=np.float32)
    top_right = np.asarray([col + 1.0, row], dtype=np.float32)
    bottom_left = np.asarray([col, row + 1.0], dtype=np.float32)
    bottom_right = np.asarray([col + 1.0, row + 1.0], dtype=np.float32)
    return (
        np.stack([top_left, top_right, center]),
        np.stack([bottom_left, bottom_right, center]),
        np.stack([top_left, bottom_left, center]),
        np.stack([top_right, bottom_right, center]),
    )


def _flatten_triangles(values: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    height, width, _ = values.shape
    polygons: list[np.ndarray] = []
    flattened_values = np.zeros(height * width * values.shape[-1], dtype=np.float32)
    idx = 0
    for row in range(height):
        for col in range(width):
            for action_vertices, action_value in zip(
                _triangles_for_cell(row, col),
                values[row, col],
            ):
                polygons.append(action_vertices)
                flattened_values[idx] = float(action_value)
                idx += 1
    return polygons, flattened_values


def _flatten_mask(mask: np.ndarray) -> np.ndarray:
    return mask.reshape(-1).astype(bool)


def _configure_axis(ax: plt.Axes, height: int, width: int, title: str) -> None:
    ax.set_title(title, fontsize=12)
    ax.set_xlim(0.0, width)
    ax.set_ylim(height, 0.0)
    ax.set_aspect("equal")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xticks(np.arange(width) + 0.5)
    ax.set_yticks(np.arange(height) + 0.5)
    ax.set_xticklabels(np.arange(width))
    ax.set_yticklabels(np.arange(height))
    ax.tick_params(labelsize=8)
    ax.set_facecolor("#ffffff")
    for row in range(height + 1):
        ax.axhline(row, color="#d0d0d0", linewidth=0.4, zorder=0)
    for col in range(width + 1):
        ax.axvline(col, color="#d0d0d0", linewidth=0.4, zorder=0)


def _draw_triangular_heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    valid_mask: np.ndarray,
    cmap: matplotlib.colors.Colormap,
    norm: matplotlib.colors.Normalize,
    colorbar_label: str,
    title: str,
    annotate_counts: bool = False,
) -> PolyCollection:
    polygons, flat_values = _flatten_triangles(values)
    flat_mask = _flatten_mask(valid_mask)
    masked_values = np.ma.array(flat_values, mask=~flat_mask)

    cmap = cmap.copy()
    cmap.set_bad("#ffffff")

    collection = PolyCollection(
        polygons,
        array=masked_values,
        cmap=cmap,
        norm=norm,
        edgecolors="#4a4a4a",
        linewidths=0.25,
    )
    ax.add_collection(collection)
    _configure_axis(ax, values.shape[0], values.shape[1], title)

    if annotate_counts:
        _annotate_triangular_counts(
            ax,
            values,
            valid_mask=valid_mask,
            cmap=cmap,
            norm=norm,
        )

    colorbar = plt.colorbar(collection, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label(colorbar_label)
    colorbar.ax.tick_params(labelsize=8)
    return collection


def _annotation_position(row: int, col: int, action_idx: int) -> tuple[float, float]:
    if action_idx == 0:
        return col + 0.5, row + 0.22
    if action_idx == 1:
        return col + 0.5, row + 0.78
    if action_idx == 2:
        return col + 0.22, row + 0.5
    return col + 0.78, row + 0.5


def _annotation_color(
    value: float,
    *,
    cmap: matplotlib.colors.Colormap,
    norm: matplotlib.colors.Normalize,
) -> str:
    red, green, blue, _ = cmap(norm(value))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "black" if luminance > 0.58 else "white"


def _annotate_triangular_counts(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    valid_mask: np.ndarray,
    cmap: matplotlib.colors.Colormap,
    norm: matplotlib.colors.Normalize,
) -> None:
    height, width, num_actions = values.shape
    for row in range(height):
        for col in range(width):
            for action_idx in range(num_actions):
                if not valid_mask[row, col, action_idx]:
                    continue
                value = int(values[row, col, action_idx])
                if value == 0:
                    continue
                x_pos, y_pos = _annotation_position(row, col, action_idx)
                ax.text(
                    x_pos,
                    y_pos,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=3.8,
                    color=_annotation_color(value, cmap=cmap, norm=norm),
                )


def _binary_colorbar(ax: plt.Axes, collection: PolyCollection, *, label: str) -> None:
    colorbar = plt.colorbar(collection, ax=ax, fraction=0.046, pad=0.03, ticks=[0, 1])
    colorbar.ax.set_yticklabels(["Unknown", "Known"])
    colorbar.set_label(label)
    colorbar.ax.tick_params(labelsize=8)


def _draw_binary_knownness(
    ax: plt.Axes,
    knownness: np.ndarray,
    *,
    valid_mask: np.ndarray,
    title: str,
    label: str,
) -> None:
    polygons, flat_values = _flatten_triangles(knownness)
    flat_mask = _flatten_mask(valid_mask)
    masked_values = np.ma.array(flat_values, mask=~flat_mask)

    cmap = colors.ListedColormap(["#d73027", "#1a9850"])
    cmap.set_bad("#ffffff")
    norm = colors.BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    collection = PolyCollection(
        polygons,
        array=masked_values,
        cmap=cmap,
        norm=norm,
        edgecolors="#4a4a4a",
        linewidths=0.25,
    )
    ax.add_collection(collection)
    _configure_axis(ax, knownness.shape[0], knownness.shape[1], title)
    _binary_colorbar(ax, collection, label=label)


def _count_norm(
    values: np.ndarray,
    valid_mask: np.ndarray,
) -> matplotlib.colors.Normalize:
    if not np.any(valid_mask):
        return colors.Normalize(vmin=0.0, vmax=1.0)
    max_value = float(np.max(values[valid_mask]))
    if max_value <= 0.0:
        max_value = 1.0
    return colors.PowerNorm(gamma=0.5, vmin=0.0, vmax=max_value)


def _bonus_norm(
    values: np.ndarray,
    valid_mask: np.ndarray,
) -> matplotlib.colors.Normalize:
    if not np.any(valid_mask):
        return colors.Normalize(vmin=0.0, vmax=1.0)
    max_value = float(np.max(values[valid_mask]))
    min_value = float(np.min(values[valid_mask]))
    if np.isclose(max_value, min_value):
        max_value = min_value + 1e-6
    return colors.Normalize(vmin=min_value, vmax=max_value)


def _policy_title(policy_name: str) -> str:
    return "Random Policy" if policy_name == "random_policy" else "Agent Policy"


def _render_policy_figure(
    rollout: PolicyRollout,
    *,
    metadata: dict[str, Any],
    thresholds: PlotThresholds,
    output_dir: Path,
) -> dict[str, Any]:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    visited_cell_mask = rollout.state_visit_counts > 0
    triangle_valid_mask = np.repeat(
        visited_cell_mask[..., None],
        rollout.visitation_counts.shape[-1],
        axis=-1,
    )

    visitation_known = (rollout.visitation_counts.astype(np.float32) >= float(thresholds.visitation_threshold)).astype(np.float32)
    bonus_known = (rollout.final_bonus_mean.astype(np.float32) <= float(thresholds.bonus_threshold)).astype(np.float32)

    fig, axes = plt.subplots(2, 2, figsize=(17, 16), constrained_layout=True)
    fig.suptitle(
        f"{_policy_title(rollout.policy_name)} on {metadata['env_id']}",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.985,
        (
            "Triangles encode actions: up=top, down=bottom, left=left, right=right. "
            "White cells were never visited."
        ),
        ha="center",
        va="top",
        fontsize=10,
    )

    count_cmap = plt.get_cmap("viridis")
    count_norm = _count_norm(rollout.visitation_counts, triangle_valid_mask)
    _draw_triangular_heatmap(
        axes[0, 0],
        rollout.visitation_counts.astype(np.float32),
        valid_mask=triangle_valid_mask,
        cmap=count_cmap,
        norm=count_norm,
        colorbar_label="Count",
        title="Visitation Count",
        annotate_counts=True,
    )

    bonus_cmap = plt.get_cmap("magma")
    bonus_norm = _bonus_norm(rollout.final_bonus_mean, triangle_valid_mask)
    _draw_triangular_heatmap(
        axes[0, 1],
        rollout.final_bonus_mean.astype(np.float32),
        valid_mask=triangle_valid_mask,
        cmap=bonus_cmap,
        norm=bonus_norm,
        colorbar_label="RND Bonus",
        title="Final Queried RND Bonus",
    )

    _draw_binary_knownness(
        axes[1, 0],
        visitation_known,
        valid_mask=triangle_valid_mask,
        title=(
            "Known by Visitation Count "
            f"(threshold >= {thresholds.visitation_threshold})"
        ),
        label="Knownness",
    )
    _draw_binary_knownness(
        axes[1, 1],
        bonus_known,
        valid_mask=triangle_valid_mask,
        title=f"Known by RND Bonus (threshold <= {thresholds.bonus_threshold:g})",
        label="Knownness",
    )

    figure_base = figures_dir / rollout.policy_name
    fig.savefig(figure_base.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(figure_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    summary = compute_policy_summary(
        rollout,
        visitation_threshold=thresholds.visitation_threshold,
        bonus_threshold=thresholds.bonus_threshold,
    )
    save_policy_summary(output_dir, summary, policy_name=rollout.policy_name)
    return summary


def plot_saved_collection(
    output_dir: Path,
    *,
    visitation_threshold: int,
    bonus_threshold: float,
) -> dict[str, dict[str, Any]]:
    metadata = load_collection_metadata(output_dir)
    thresholds = PlotThresholds(
        visitation_threshold=visitation_threshold,
        bonus_threshold=bonus_threshold,
    )

    summaries: dict[str, dict[str, Any]] = {}
    for policy_name in POLICY_NAMES:
        rollout = load_policy_rollout(output_dir, policy_name)
        summaries[policy_name] = _render_policy_figure(
            rollout,
            metadata=metadata,
            thresholds=thresholds,
            output_dir=output_dir,
        )
    return summaries
