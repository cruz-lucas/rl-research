#!/usr/bin/env python3
"""Export a paper-ready Navix environment layout image.

Examples:
  uv run python scripts/export_navix_layout.py --preset empty_16x16
  uv run python scripts/export_navix_layout.py --preset doorkey_16x16_layout1
  uv run python scripts/export_navix_layout.py \
    --env-id GridDoorKey-5x5-layout2-v0 \
    --agent-position 3 1 \
    --output docs/images/navix/layouts/doorkey_5x5_layout2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from navix.observations import rgb

from rl_research.environments import NavixWrapper


PRESETS = {
    "empty_16x16": "Navix-Empty-16x16-v0",
    "empty_16x16_cardinal": "Navix-Empty-16x16-Cardinal-v0",
    "doorkey_5x5_layout1": "GridDoorKey-5x5-layout1-v0",
    "doorkey_5x5_layout2": "GridDoorKey-5x5-layout2-v0",
    "doorkey_5x5_layout3": "GridDoorKey-5x5-layout3-v0",
    "doorkey_16x16_layout1": "GridDoorKey-16x16-layout1-v0",
    "doorkey_16x16_layout2": "GridDoorKey-16x16-layout2-v0",
    "doorkey_16x16_layout3": "GridDoorKey-16x16-layout3-v0",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a high-resolution PNG/PDF of a Navix environment layout."
    )
    parser.add_argument(
        "--preset",
        choices=tuple(PRESETS),
        default="empty_16x16",
        help="Named environment used in this project.",
    )
    parser.add_argument(
        "--env-id",
        default=None,
        help="Override the preset with an explicit Navix environment id.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output path without extension, or a .png/.pdf path. Defaults to "
            "docs/images/navix/layouts/<preset>."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Reset seed. Fixed-grid layouts are deterministic.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1024,
        help="Environment horizon used only to instantiate the environment.",
    )
    parser.add_argument(
        "--agent-position",
        "--player-position",
        nargs=2,
        type=int,
        metavar=("ROW", "COL"),
        default=None,
        help=(
            "Optional grid position for the rendered agent/player, using "
            "zero-indexed (row, col) coordinates."
        ),
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=24,
        help="Nearest-neighbor pixel upscale factor for --renderer rgb.",
    )
    parser.add_argument(
        "--renderer",
        choices=("vector", "rgb"),
        default="vector",
        help=(
            "Renderer to use. `vector` draws a publication-style layout from "
            "state entities; `rgb` exports Navix's exact pixel-art renderer."
        ),
    )
    parser.add_argument(
        "--size-px",
        type=int,
        default=3072,
        help="Output width/height in pixels for --renderer vector PNG/PDF raster size.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="DPI metadata for PNG and rasterized PDF export.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf"),
        default=("png", "pdf"),
        help="File formats to write.",
    )
    return parser


def _resolve_output_base(args: argparse.Namespace) -> Path:
    if args.output is None:
        return Path("docs/images/navix/layouts") / str(args.preset)

    output = args.output
    if output.suffix.lower() in {".png", ".pdf"}:
        return output.with_suffix("")
    return output


def _set_agent_position(timestep, row: int, col: int):
    height, width = timestep.state.grid.shape
    if not (0 <= row < height and 0 <= col < width):
        raise ValueError(
            f"Agent position ({row}, {col}) is outside the {height}x{width} grid."
        )

    player = timestep.state.get_player(idx=0)
    player = player.replace(position=jnp.asarray((row, col), dtype=jnp.int32))
    return timestep.replace(state=timestep.state.set_player(player))


def _load_timestep(
    env_id: str,
    seed: int,
    max_steps: int,
    agent_position: tuple[int, int] | None,
):
    environment = NavixWrapper(
        env_id=env_id,
        max_steps=max_steps,
        observation_mode="symbolic",
    )
    timestep, _ = environment.reset(jax.random.PRNGKey(seed))
    if agent_position is not None:
        timestep = _set_agent_position(timestep, *agent_position)
    return timestep


def _render_rgb_image(timestep) -> np.ndarray:
    return np.asarray(jax.device_get(rgb(timestep.state)), dtype=np.uint8)


def _upscale_image(image: np.ndarray, scale: int) -> np.ndarray:
    if scale < 1:
        raise ValueError("--scale must be at least 1.")
    return np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)


def _save_png(image: np.ndarray, output_path: Path, dpi: int) -> None:
    plt.imsave(output_path, image, dpi=dpi)


def _entity_positions(state, entity_name: str) -> np.ndarray:
    entity = state.entities.get(entity_name)
    if entity is None:
        return np.empty((0, 2), dtype=np.int32)
    return np.asarray(jax.device_get(entity.position), dtype=np.int32).reshape(-1, 2)


def _draw_cell(
    ax,
    row: int,
    col: int,
    *,
    facecolor: str,
    edgecolor: str = "none",
    linewidth: float = 0.0,
    inset: float = 0.0,
) -> None:
    ax.add_patch(
        patches.Rectangle(
            (col + inset, row + inset),
            1.0 - 2.0 * inset,
            1.0 - 2.0 * inset,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
        )
    )


def _draw_vector_layout(ax, state) -> None:
    grid = np.asarray(jax.device_get(state.grid))
    height, width = grid.shape

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_facecolor("#f8f7f2")

    for row in range(height):
        for col in range(width):
            if grid[row, col] < 0:
                _draw_cell(ax, row, col, facecolor="#2f3437")
            else:
                _draw_cell(
                    ax,
                    row,
                    col,
                    facecolor="#fbfaf5",
                    edgecolor="#d8d2c4",
                    linewidth=0.35,
                )

    for row, col in _entity_positions(state, "wall"):
        _draw_cell(ax, int(row), int(col), facecolor="#2f3437")

    for row, col in _entity_positions(state, "door"):
        _draw_cell(
            ax,
            int(row),
            int(col),
            facecolor="#d7a928",
            edgecolor="#7a5a0a",
            linewidth=1.0,
            inset=0.08,
        )
        ax.plot(
            [col + 0.28, col + 0.28],
            [row + 0.18, row + 0.82],
            color="#7a5a0a",
            linewidth=1.2,
        )

    for row, col in _entity_positions(state, "goal"):
        ax.add_patch(
            patches.Circle(
                (col + 0.5, row + 0.5),
                radius=0.32,
                facecolor="#55a868",
                edgecolor="#245d38",
                linewidth=1.0,
            )
        )

    for row, col in _entity_positions(state, "key"):
        ax.add_patch(
            patches.Circle(
                (col + 0.42, row + 0.44),
                radius=0.14,
                facecolor="#e0b43b",
                edgecolor="#8b6a12",
                linewidth=0.9,
            )
        )
        ax.plot(
            [col + 0.53, col + 0.78],
            [row + 0.55, row + 0.72],
            color="#8b6a12",
            linewidth=1.4,
        )
        ax.plot(
            [col + 0.68, col + 0.72],
            [row + 0.65, row + 0.56],
            color="#8b6a12",
            linewidth=1.1,
        )

    player = state.get_player(idx=0)
    player_row, player_col = np.asarray(jax.device_get(player.position), dtype=float)
    direction = int(np.asarray(jax.device_get(player.direction)))
    center = np.asarray((player_col + 0.5, player_row + 0.5), dtype=float)
    direction_vectors = {
        0: np.asarray((1.0, 0.0)),
        1: np.asarray((0.0, 1.0)),
        2: np.asarray((-1.0, 0.0)),
        3: np.asarray((0.0, -1.0)),
    }
    forward = direction_vectors.get(direction, direction_vectors[0])
    side = np.asarray((-forward[1], forward[0]))
    triangle = np.stack(
        [
            center + 0.34 * forward,
            center - 0.26 * forward + 0.24 * side,
            center - 0.26 * forward - 0.24 * side,
        ]
    )
    ax.add_patch(
        patches.Polygon(
            triangle,
            closed=True,
            facecolor="#4c72b0",
            edgecolor="#1f355f",
            linewidth=1.0,
            joinstyle="round",
        )
    )


def _save_vector(state, output_path: Path, dpi: int, size_px: int) -> None:
    if size_px < 1:
        raise ValueError("--size-px must be at least 1.")
    fig = plt.figure(figsize=(size_px / dpi, size_px / dpi), dpi=dpi, frameon=False)
    ax = fig.add_axes((0, 0, 1, 1))
    _draw_vector_layout(ax, state)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _save_pdf(image: np.ndarray, output_path: Path, dpi: int) -> None:
    height, width = image.shape[:2]
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, frameon=False)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(image, interpolation="nearest")
    ax.set_axis_off()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main() -> None:
    args = _build_parser().parse_args()
    env_id = args.env_id or PRESETS[str(args.preset)]
    output_base = _resolve_output_base(args)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    timestep = _load_timestep(
        env_id=env_id,
        seed=args.seed,
        max_steps=args.max_steps,
        agent_position=(
            tuple(args.agent_position) if args.agent_position is not None else None
        ),
    )

    written: list[Path] = []
    if args.renderer == "vector":
        for file_format in args.formats:
            path = output_base.with_suffix(f".{file_format}")
            _save_vector(timestep.state, path, dpi=args.dpi, size_px=args.size_px)
            written.append(path)
    else:
        image = _upscale_image(_render_rgb_image(timestep), scale=args.scale)
        if "png" in args.formats:
            png_path = output_base.with_suffix(".png")
            _save_png(image, png_path, dpi=args.dpi)
            written.append(png_path)
        if "pdf" in args.formats:
            pdf_path = output_base.with_suffix(".pdf")
            _save_pdf(image, pdf_path, dpi=args.dpi)
            written.append(pdf_path)

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
