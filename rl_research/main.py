import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Type, Tuple

import gin
import jax
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import Metric
import numpy as np
import tyro

from rl_research.agents import *
from rl_research.buffers import BaseBuffer, ReplayBuffer
from rl_research.environments import *
from rl_research.experiment import History, run_loop


@gin.configurable
def setup_mlflow(
    seed: int,
    experiment_name: str = "placeholder",
    experiment_group: str = "placeholder",
):
    """Setup MLflow experiment and run."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    max_tries = 10
    tries = 0
    while experiment is None:
        try:
            tries += 1
            experiment_id = mlflow.create_experiment(experiment_name)
        except mlflow.MlflowException:
            experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if tries >= max_tries:
            break
    # TODO: better handle exception, experiment shouldn't be allowed to be none

    experiment_id = experiment.experiment_id

    run_name = f"{experiment_group}_seed_{seed}"
    return mlflow.start_run(
        run_name=run_name,
        experiment_id=experiment_id,
        tags={
            "group": experiment_group,
        },
    )


def log_history_to_mlflow(history: History):
    """Log training history to MLflow."""
    client = MlflowClient()
    run_id = mlflow.active_run().info.run_id

    dones = history.dones
    steps = history.global_steps[dones]

    metrics = []
    timestamp = int(time.time() * 1000)

    for i, step in enumerate(steps):
        metrics.extend([
            Metric(
                key="train/return",
                value=float(history.train_returns[dones][i]),
                step=int(step),
                timestamp=timestamp,
            ),
            Metric(
                key="train/discounted_return",
                value=float(history.train_discounted_returns[dones][i]),
                step=int(step),
                timestamp=timestamp,
            ),
            Metric(
                key="train/loss",
                value=float(history.train_losses[dones][i]),
                step=int(step),
                timestamp=timestamp,
            ),
        ])

    # Log in chunks to avoid memory blow-up
    BATCH_SIZE = 100_000
    for i in range(0, len(metrics), BATCH_SIZE):
        client.log_batch(
            run_id=run_id,
            metrics=metrics[i:i + BATCH_SIZE],
        )

    mlflow.log_metric(
        "last_100/train_disc_return_mean",
        float(np.mean(history.train_discounted_returns[dones][-100:])),
    )

    evaluate_every = gin.get_bindings("run_loop")["evaluate_every"]
    eval_episodes = gin.get_bindings("run_loop")["eval_episodes"]

    # if evaluate_every != 0:
    #     for episode in range(0, num_episodes, evaluate_every):
    #         mlflow.log_metrics(
    #             {
    #                 "eval/mean_return": float(history.eval_returns[episode]),
    #                 "eval/mean_discounted_return": float(
    #                     history.eval_discounted_returns[episode]
    #                 ),
    #             },
    #             step=int(history.global_steps[episode]),
    #         )

    final_train_window = 100
    mlflow.log_metrics(
        {
            "last_100/train_disc_return_mean": float(
                np.mean(history.train_discounted_returns[history.dones][-final_train_window:])
            ),
            # "last_100/eval_disc_return_mean": float(
            #     np.mean(history.eval_discounted_returns[-final_train_window:])
            # ),
        }
    )


@gin.configurable
def run_single_seed(
    seed: int,
    buffer_cls: Type[BaseBuffer] = ReplayBuffer,
    env_cls: Type[BaseJaxEnv] = BaseJaxEnv,
    agent_cls: Type[BaseAgent] = BaseAgent,
):
    """Run training for a single seed."""
    env = env_cls()

    n_states = env.env.observation_space.n
    n_actions = env.env.action_space.n

    agent = agent_cls(
        num_states=n_states,
        num_actions=n_actions,
    )
    agent_state = agent.initial_state()

    buffer_init_kwargs = {}
    buffer = buffer_cls(**buffer_init_kwargs)
    buffer_state = buffer.initial_state()

    history = run_loop(
        agent=agent,
        environment=env,
        buffer_state=buffer_state,
        agent_state=agent_state,
        seed=seed,
    )

    return history


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

    # local_root = Path(os.environ.get("SLURM_TMPDIR", "./local_runs")) / "mlruns"
    # local_root.mkdir(parents=True, exist_ok=True)

    # shared_root = Path("~/mlruns")
    # shared_root.mkdir(parents=True, exist_ok=True)

    # mlflow.set_tracking_uri(shared_root)
    mlflow.set_tracking_uri("sqlite:///mlruns.db")

    with setup_mlflow(seed=seed) as run:
        run_bindings = gin.get_bindings("run_single_seed")
        agent_cls = run_bindings.get("agent_cls", BaseAgent)
        agent_cls_name = agent_cls.__name__
        agent_params = gin.get_bindings(agent_cls_name)
        buffer_cls = run_bindings.get("buffer_cls", ReplayBuffer)
        buffer_cls_name = buffer_cls.__name__
        buffer_params = gin.get_bindings(buffer_cls_name)
        train_params = gin.get_bindings("run_loop")

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

        history = run_single_seed(seed=seed)
        history = jax.device_get(history)

        log_history_to_mlflow(history)


if __name__ == "__main__":
    main(tyro.cli(Args))
