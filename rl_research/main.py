import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import gin
import mlflow
import tyro

from rl_research.agents import BaseAgent
from rl_research.buffers import ReplayBuffer
from rl_research.environments import *
from rl_research.experiment import run_loop
from rl_research.utils import setup_mlflow


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

    mlflow.set_tracking_uri("./mlruns_test/")
    # mlflow.set_tracking_uri("./mlruns/")
    # mlflow.set_tracking_uri("sqlite:///mlruns.db")

    with setup_mlflow(seed=seed) as run:
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
                **{f"agent_{k}": v for k, v in agent_params.items()},
                **{f"buffer_{k}": v for k, v in buffer_params.items()},
                **{k: v for k, v in train_params.items()},
            }
        )

        run_loop(seed=seed)


if __name__ == "__main__":
    main(tyro.cli(Args))
