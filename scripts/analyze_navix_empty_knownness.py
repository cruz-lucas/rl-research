#!/usr/bin/env python3
"""Analyse state-action knownness on Navix-Empty-16x16-v0.

Usage:
  Collect rollouts, save `.npz`/`.json` artifacts, and render figures:
    uv run python scripts/analyze_navix_empty_knownness.py collect \
      --config rl_research/configs/navix/doorkey/16x16_layout1/dqn_rmax_rnd.gin \
      --checkpoint /absolute/path/to/checkpoint \
      --output-dir outputs/navix_empty_knownness \
      --episodes 100 \
      --train-rnd-after-each-episode \
      --rnd-train-epochs-per-episode 1 \
      --bonus-threshold 1.0 \
      --visitation-threshold 5

  Change thresholds and regenerate plots from saved artifacts only:
    uv run python scripts/analyze_navix_empty_knownness.py plot \
      --output-dir outputs/navix_empty_knownness \
      --bonus-threshold 0.5 \
      --visitation-threshold 10

Notes:
  - The script always uses `Navix-Empty-16x16-v0` with a 4-action wrapper:
    `up`, `down`, `left`, `right`.
  - For meaningful RND heatmaps, point `--config` and `--checkpoint` at the
    trained 4-action DQN + RND + RMax agent you want to analyse.
  - When `--train-rnd-after-each-episode` is enabled, only the RND predictor is
    updated after each episode; the DQN/Q-network is left unchanged.
  - The saved RND heatmap is always a final post-collection query over every
    visited trajectory state and all four actions using the final RND predictor.
  - If `--checkpoint` is omitted, the analysis still runs with a randomly
    initialised agent, but the RND/agent-policy plots will not reflect a trained
    policy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rl_research.analysis import (
    CollectionSettings,
    collect_knownness_rollouts,
    compute_policy_summary,
    plot_saved_collection,
    save_collection_metadata,
    save_policy_rollout,
    save_policy_summary,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect and plot state-action knownness heatmaps for "
            "Navix-Empty-16x16-v0."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser(
        "collect",
        help="Run random and agent-policy rollouts, save artifacts, and render plots.",
    )
    collect_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory that will receive `.npz`, `.json`, and figure outputs.",
    )
    collect_parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes to collect for each policy.",
    )
    collect_parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed for environment resets and the random policy rollout.",
    )
    collect_parser.add_argument(
        "--max-steps",
        type=int,
        default=1024,
        help="Optional override for the environment max-step horizon.",
    )
    collect_parser.add_argument(
        "--bonus-threshold",
        type=float,
        default=1.0,
        help="Default RND knownness threshold to store in metadata and summaries.",
    )
    collect_parser.add_argument(
        "--visitation-threshold",
        type=int,
        default=1,
        help=(
            "Default visitation knownness threshold to store in metadata and "
            "summaries."
        ),
    )
    collect_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional gin config that defines the DQN+RND+RMax agent architecture.",
    )
    collect_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint for the agent policy and RND bonus estimator.",
    )
    collect_parser.add_argument(
        "--train-rnd-after-each-episode",
        action="store_true",
        help=(
            "After each episode, fit the RND predictor on that episode's visited "
            "state-action pairs before continuing collection."
        ),
    )
    collect_parser.add_argument(
        "--rnd-train-epochs-per-episode",
        type=int,
        default=1,
        help=(
            "Number of RND-only optimizer passes to run on each episode when "
            "`--train-rnd-after-each-episode` is enabled."
        ),
    )
    collect_parser.add_argument(
        "--binding",
        action="append",
        default=[],
        help="Optional gin binding override. Repeat the flag to pass multiple values.",
    )
    collect_parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Collect and save artifacts without rendering figures immediately.",
    )

    plot_parser = subparsers.add_parser(
        "plot",
        help="Regenerate plots and threshold-dependent summaries from saved artifacts.",
    )
    plot_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory containing the saved `.npz`/`.json` artifacts.",
    )
    plot_parser.add_argument(
        "--bonus-threshold",
        type=float,
        required=True,
        help="RND knownness threshold to apply during plotting.",
    )
    plot_parser.add_argument(
        "--visitation-threshold",
        type=int,
        required=True,
        help="Visitation knownness threshold to apply during plotting.",
    )

    return parser


def _run_collect(args: argparse.Namespace) -> None:
    settings = CollectionSettings(
        output_dir=args.output_dir.resolve(),
        episodes=args.episodes,
        seed=args.seed,
        max_steps=args.max_steps,
        bonus_threshold=args.bonus_threshold,
        visitation_threshold=args.visitation_threshold,
        config_path=args.config.resolve() if args.config is not None else None,
        checkpoint_path=(
            args.checkpoint.resolve() if args.checkpoint is not None else None
        ),
        gin_bindings=tuple(args.binding),
        train_rnd_after_each_episode=bool(args.train_rnd_after_each_episode),
        rnd_train_epochs_per_episode=int(args.rnd_train_epochs_per_episode),
    )

    rollouts, metadata = collect_knownness_rollouts(settings)
    save_collection_metadata(settings.output_dir, metadata)

    for policy_name, rollout in rollouts.items():
        save_policy_rollout(settings.output_dir, rollout)
        summary = compute_policy_summary(
            rollout,
            visitation_threshold=settings.visitation_threshold,
            bonus_threshold=settings.bonus_threshold,
        )
        save_policy_summary(settings.output_dir, summary, policy_name=policy_name)

    if not args.skip_plot:
        plot_saved_collection(
            settings.output_dir,
            visitation_threshold=settings.visitation_threshold,
            bonus_threshold=settings.bonus_threshold,
        )


def _run_plot(args: argparse.Namespace) -> None:
    plot_saved_collection(
        args.output_dir.resolve(),
        visitation_threshold=args.visitation_threshold,
        bonus_threshold=args.bonus_threshold,
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "collect":
        _run_collect(args)
        return
    if args.command == "plot":
        _run_plot(args)
        return
    raise RuntimeError(f"Unsupported command '{args.command}'.")


if __name__ == "__main__":
    main()
