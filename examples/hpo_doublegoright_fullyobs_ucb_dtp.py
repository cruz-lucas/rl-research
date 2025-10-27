"""Hyperparameter sweep runner for the DoubleGoRight UCB decision-time planner."""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from typing import Iterable, Sequence

import jax
from goright.jax.env import EnvParams, GoRightJaxEnv

from rl_research.agents import DTUCBPlanner, DTUCBParams, goright_expectation_model
from rl_research.examples import ExperimentConfig, TrackingConfig, run_tabular_mlflow_example


DEFAULT_ENV_PARAMS = EnvParams(
    length=21,
    num_indicators=2,
    num_actions=2,
    first_checkpoint=10,
    first_reward=3.0,
    second_checkpoint=20,
    second_reward=6.0,
    is_partially_obs=False,
    mapping="default",
)


@dataclass(frozen=True)
class SweepPoint:
    """Container describing a single sweep configuration."""

    id: str
    label: str
    overrides: dict[str, float | int | bool]


def _str_to_bool(value: str) -> bool:
    value = value.lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot interpret '{value}' as boolean.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an MLflow-tracked hyperparameter sweep for the DoubleGoRight "
            "UCB decision-time planner."
        )
    )
    parser.add_argument("--experiment-name", default="double_goright_fully_obs_ucb_dtp_sweep")
    parser.add_argument("--agent-name", default="ucb_dtp_planner")
    parser.add_argument("--base-seed", type=int, default=0, help="Seed used to branch RNG streams.")
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--total-train-episodes", type=int, default=600)
    parser.add_argument("--episode-length", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--num-eval-episodes", type=int, default=1)
    parser.add_argument("--discount", type=float, default=0.9)

    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0],
        help="Learning rates to sweep over.",
    )
    parser.add_argument(
        "--betas",
        type=float,
        nargs="+",
        default=[5, 10.0, 20.0, 40.0, 50, 100],
        help="UCB beta bonuses to sweep over.",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6, 7],
        help="Planning horizons to sweep over.",
    )
    parser.add_argument(
        "--time-bonus-options",
        type=_str_to_bool,
        nargs="+",
        default=[False, True],
        help="Whether to enable time-dependent bonuses (one value per option).",
    )

    parser.add_argument(
        "--configs",
        nargs="+",
        help="Optional list of config IDs or labels to run (default runs all).",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip the first N configs after filtering.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run at most this many configs after applying offset.",
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List the generated configurations and exit.",
    )
    return parser.parse_args()


def build_environment() -> GoRightJaxEnv:
    """Constructs the partially-observed DoubleGoRight environment."""
    return GoRightJaxEnv(DEFAULT_ENV_PARAMS)


def build_sweep_points(args: argparse.Namespace) -> list[SweepPoint]:
    """Creates all combinations defined by the search space."""
    points: list[SweepPoint] = []
    grid: Iterable[tuple[float, float, int, bool]] = itertools.product(
        args.learning_rates,
        args.betas,
        args.horizons,
        args.time_bonus_options,
    )

    for idx, (lr, beta, horizon, use_time_bonus) in enumerate(grid, start=1):
        label = _make_label(lr, beta, horizon, use_time_bonus)
        overrides = {
            "learning_rate": lr,
            "beta": beta,
            "horizon": horizon,
            "use_time_bonus": use_time_bonus,
        }
        points.append(
            SweepPoint(
                id=f"cfg_{idx:03d}",
                label=label,
                overrides=overrides,
            )
        )
    return points


def _make_label(lr: float, beta: float, horizon: int, use_time_bonus: bool) -> str:
    def _fmt(value: float | int) -> str:
        text = f"{value}"
        return text.replace(".", "p").replace("-", "m")

    time_label = "timebonus" if use_time_bonus else "static"
    return f"lr{_fmt(lr)}_beta{_fmt(beta)}_hz{_fmt(horizon)}_{time_label}"


def select_points(points: Sequence[SweepPoint], args: argparse.Namespace) -> list[SweepPoint]:
    """Applies CLI filters (configs, offset, limit) to the sweep points."""
    selected: list[SweepPoint]
    if args.configs:
        lookup = {p.id.lower(): p for p in points}
        lookup |= {p.label.lower(): p for p in points}
        selected = []
        for token in args.configs:
            key = token.lower()
            point = lookup.get(key)
            if point is None:
                raise SystemExit(f"Unknown config identifier '{token}'.")
            selected.append(point)
    else:
        selected = list(points)

    if args.offset:
        selected = selected[args.offset :]
    if args.limit is not None:
        selected = selected[: args.limit]
    return selected


def list_configs(points: Sequence[SweepPoint]) -> None:
    """Prints the available sweep configurations."""
    for point in points:
        overrides = ", ".join(f"{k}={v}" for k, v in point.overrides.items())
        print(f"{point.id:>7}  {point.label:<35}  {overrides}")


def main() -> None:
    args = parse_args()
    env = build_environment()
    sweep_points = build_sweep_points(args)

    if args.list_configs:
        list_configs(sweep_points)
        return

    selected = select_points(sweep_points, args)
    if not selected:
        raise SystemExit("No configurations selected. Adjust filters or sweep space.")

    run_config = ExperimentConfig(
        num_seeds=args.num_seeds,
        total_train_episodes=args.total_train_episodes,
        episode_length=args.episode_length,
        eval_every=args.eval_every,
        num_eval_episodes=args.num_eval_episodes,
    )

    base_agent_kwargs = {
        "num_states": env.env.observation_space.n,
        "num_actions": env.env.action_space.n,
        "discount": args.discount,
        "learning_rate": args.learning_rates[0],
        "initial_value": 0.0,
        "horizon": args.horizons[0],
        "beta": args.betas[0],
        "dynamics_model": goright_expectation_model(
            length=DEFAULT_ENV_PARAMS.length,
            first_checkpoint=DEFAULT_ENV_PARAMS.first_checkpoint,
            first_reward=DEFAULT_ENV_PARAMS.first_reward,
            second_checkpoint=DEFAULT_ENV_PARAMS.second_checkpoint,
            second_reward=DEFAULT_ENV_PARAMS.second_reward,
            num_indicators=DEFAULT_ENV_PARAMS.num_indicators,
            is_partially_obs=DEFAULT_ENV_PARAMS.is_partially_obs,
        ),
        "use_time_bonus": args.time_bonus_options[0],
    }

    sweep_rng = jax.random.PRNGKey(args.base_seed)
    for point in selected:
        sweep_rng, run_rng = jax.random.split(sweep_rng)
        agent_params = DTUCBParams(**base_agent_kwargs | point.overrides)
        agent = DTUCBPlanner(params=agent_params)

        parent_tags = {
            "config_id": point.id,
            "config_label": point.label,
        }
        seed_tags = parent_tags | {}

        print(f"Running {point.id} ({point.label}) with overrides: {point.overrides}")
        run_tabular_mlflow_example(
            env=env,
            agent=agent,
            agent_params=agent_params,
            rng=run_rng,
            run_config=run_config,
            tracking=TrackingConfig(
                experiment_name=args.experiment_name,
                agent_name=args.agent_name,
                parent_run_name=f"{args.agent_name}_{point.label}",
                parent_tags=parent_tags,
                seed_tags=seed_tags,
            ),
        )


if __name__ == "__main__":
    main()
