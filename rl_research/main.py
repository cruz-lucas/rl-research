import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Type

import gin
import jax
import mlflow
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
    train_episodes = gin.get_bindings("run_loop")["train_episodes"]
    max_episode_steps = gin.get_bindings("run_loop")["max_episode_steps"]
    evaluate_every = gin.get_bindings("run_loop")["evaluate_every"]
    eval_episodes = gin.get_bindings("run_loop")["eval_episodes"]

    for episode in range(train_episodes):
        mlflow.log_metrics(
            {
                "train/return": float(history.train_returns[episode]),
                "train/discounted_return": float(
                    history.train_discounted_returns[episode]
                ),
                "train/loss": float(history.train_losses[episode]),
            },
            step=episode * max_episode_steps,
        )

    num_evals = train_episodes // evaluate_every
    for episode in range(num_evals):
        mlflow.log_metrics(
            {
                "eval/mean_return": float(history.eval_returns[episode]),
                "eval/mean_discounted_return": float(
                    history.eval_discounted_returns[episode]
                ),
            },
            step=episode * max_episode_steps,
        )

    final_train_window = 100
    mlflow.log_metrics(
        {
            "last_100/train_disc_return_mean": float(
                np.mean(history.train_discounted_returns[-final_train_window:])
            ),
            "last_100/eval_disc_return_mean": float(
                np.mean(history.eval_discounted_returns[-final_train_window:])
            ),
        }
    )


def log_agent_states_to_mlflow(agent_states):
    """Save per-episode agent state tensors as an MLflow artifact."""
    agent_states_np = jax.tree_util.tree_map(np.array, agent_states)

    payload = {}
    if hasattr(agent_states_np, "q_table"):
        payload["q_values"] = agent_states_np.q_table
    if hasattr(agent_states_np, "visit_counts"):
        payload["sa_counts"] = agent_states_np.visit_counts
    if hasattr(agent_states_np, "behavior_q_values"):
        payload["behavior_q_values"] = agent_states_np.behavior_q_values

    if not payload:
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "agent_states.npz")
        np.savez(save_path, **payload)
        mlflow.log_artifact(save_path, artifact_path="artifacts")


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

    history, agent_states = run_loop(
        agent=agent,
        environment=env,
        buffer_state=buffer_state,
        agent_state=agent_state,
        seed=seed,
    )

    history = jax.tree_util.tree_map(np.array, history)

    return history, agent_states


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

    shared_root = Path("./mlruns")
    shared_root.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(shared_root)
    # mlflow.set_tracking_uri("sqlite:///mlruns.db")

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

        history, agent_states = run_single_seed(seed=seed)
        log_history_to_mlflow(history)
        # log_agent_states_to_mlflow(agent_states)

    # subprocess.run(
    #     ["rsync", "-av", str(local_root) + "/", str(shared_root) + "/"], check=True
    # )


if __name__ == "__main__":
    main(tyro.cli(Args))
