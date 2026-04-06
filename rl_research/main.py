import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import gin
import mlflow
import tyro

from rl_research.agents import BaseAgent
from rl_research.buffers import ReplayBuffer
from rl_research.experiment import run_loop
from rl_research.utils import setup_mlflow


def _serialize_param(value: object) -> object:
    if value is None:
        return "None"
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, type):
        return value.__name__
    if isinstance(value, (list, tuple, set)):
        return repr(tuple(value) if not isinstance(value, tuple) else value)
    if isinstance(value, dict):
        return repr(value)
    return str(value)


def _serialize_params(params: dict[str, object]) -> dict[str, object]:
    return {key: _serialize_param(value) for key, value in params.items()}


@dataclass
class Args:
    """Arguments for running an RL experiment."""

    config: Annotated[Path, tyro.conf.arg(help="Path to the gin config file.")]
    seed: Annotated[
        int | None,
        tyro.conf.arg(help="Random seed (uses SLURM_ARRAY_TASK_ID when omitted)."),
    ] = None
    binding: Annotated[
        list[str],
        tyro.conf.arg(
            help=(
                "Optional gin binding overrides (repeatable). "
                "Example: --binding OptimisticQLearningAgent.step_size=0.05"
            ),
        ),
    ] = field(default_factory=list)


def main(args: Args) -> None:
    gin.parse_config_files_and_bindings([str(args.config)], args.binding)

    if args.seed is None:
        seed = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
    else:
        seed = args.seed


    with setup_mlflow(seed=seed):
        run_bindings = gin.get_bindings("run_loop")
        agent_cls = run_bindings.get("agent_cls", BaseAgent)
        agent_cls_name = agent_cls.__name__
        agent_params = gin.get_bindings(agent_cls_name)
        buffer_cls = run_bindings.get("buffer_cls", ReplayBuffer)
        buffer_cls_name = buffer_cls.__name__
        buffer_params = gin.get_bindings(buffer_cls_name)
        train_params = gin.get_bindings("TrainingConfig")

        mlflow.log_params(
            {
                "seed": seed,
                "agent_class": agent_cls_name,
                "buffer_class": buffer_cls_name,
                **{f"agent_{k}": v for k, v in _serialize_params(agent_params).items()},
                **{
                    f"buffer_{k}": v
                    for k, v in _serialize_params(buffer_params).items()
                },
                **_serialize_params(train_params),
            }
        )

        run_loop(seed=seed)


if __name__ == "__main__":
    main(tyro.cli(Args))
