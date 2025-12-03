import os
import jax
import numpy as np
import mlflow
import argparse
import inspect
import tempfile
from typing import Type

import gin
from rl_research.experiment import run_loop, History
from rl_research.buffers import ReplayBuffer, BaseBuffer
from rl_research.agents import *
from rl_research.environments import *


@gin.configurable
def setup_mlflow(seed: int, experiment_name: str = 'placeholder', experiment_group: str = 'placeholder'):
    """Setup MLflow experiment and run."""

    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    
    run_name = f"{experiment_group}_seed_{seed}"
    mlflow.start_run(run_name=run_name, experiment_id=experiment_id, tags={
        "group": experiment_group,
    })

    run_bindings = gin.get_bindings('run_single_seed')
    agent_cls = run_bindings.get('agent_cls', BaseAgent)
    agent_cls_name = agent_cls.__name__
    agent_params = gin.get_bindings(agent_cls_name)
    buffer_cls = run_bindings.get('buffer_cls', ReplayBuffer)
    buffer_cls_name = buffer_cls.__name__
    buffer_params = gin.get_bindings(buffer_cls_name)
    
    mlflow.log_params({
        "seed": seed,
        "agent_class": agent_cls_name,
        "buffer_class": buffer_cls_name,
        **{f"agent_{k}": v for k, v in agent_params.items()},
        **{f"buffer_{k}": v for k, v in buffer_params.items()},
    })


def log_history_to_mlflow(history: History):
    """Log training history to MLflow."""
    train_episodes = gin.get_bindings('run_loop')['train_episodes']
    max_episode_steps = gin.get_bindings('run_loop')['max_episode_steps']
    evaluate_every = gin.get_bindings('run_loop')['evaluate_every']
    eval_episodes = gin.get_bindings('run_loop')['eval_episodes']

    for episode in range(train_episodes):
        mlflow.log_metrics({
            "train/return": float(history.train_returns[episode]),
            "train/discounted_return": float(history.train_discounted_returns[episode]),
            "train/loss": float(history.train_losses[episode]),
        }, step=episode * max_episode_steps)
    
    num_evals = train_episodes // evaluate_every
    for episode in range(num_evals):
        mlflow.log_metrics({
            "eval/mean_return": float(history.eval_returns[episode]),
            "eval/mean_discounted_return": float(history.eval_discounted_returns[episode]),
        }, step=episode * max_episode_steps)
    
    final_train_window = 100
    mlflow.log_metrics({
        "last_100/train_disc_return_mean": float(np.mean(history.train_discounted_returns[-final_train_window:])),
        "last_100/eval_disc_return_mean": float(np.mean(history.eval_discounted_returns[-final_train_window:])),
    })

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
def run_single_seed(seed: int, buffer_cls: Type[BaseBuffer] = ReplayBuffer, env_cls: Type[BaseJaxEnv] = BaseJaxEnv, agent_cls: Type[BaseAgent] = BaseAgent):
    """Run training for a single seed."""
    try:
        env = env_cls()    

        n_states = env.env.observation_space.n
        n_actions = env.env.action_space.n

        agent = agent_cls(
            num_states=n_states,
            num_actions=n_actions,
        )
        agent_state = agent.initial_state()

        buffer_init_kwargs = {}
        buffer_sig = inspect.signature(buffer_cls)
        if 'num_states' in buffer_sig.parameters:
            buffer_init_kwargs['num_states'] = n_states
        if 'num_actions' in buffer_sig.parameters:
            buffer_init_kwargs['num_actions'] = n_actions

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
        log_history_to_mlflow(history)
        log_agent_states_to_mlflow(agent_states)

    finally:
        mlflow.end_run()


def main():
    parser = argparse.ArgumentParser(description="Run RL experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (will use SLURM_ARRAY_TASK_ID if not provided)")
    parser.add_argument(
        "--binding",
        action="append",
        default=[],
        help="Optional gin binding overrides (can be passed multiple times). Example: --binding OptimisticQLearningAgent.step_size=0.05",
    )
    args = parser.parse_args()
    
    gin.parse_config_files_and_bindings([args.config], args.binding)
    
    if args.seed is None:
        seed = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
    else:
        seed = args.seed

    # for seed in range(5):
    setup_mlflow(seed=seed)
    run_single_seed(seed=seed)


if __name__ == "__main__":
    main()
