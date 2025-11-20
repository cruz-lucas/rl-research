import os
import yaml
import jax
import jax.numpy as jnp
import mlflow
import argparse
from pathlib import Path
from typing import Dict, Any, Type
import numpy as np

import gin
from rl_research.experiment import run_loop, BufferState, History
from rl_research.agents import *
from rl_research.environments import *


@gin.configurable
def setup_mlflow(seed: int, experiment_name: str = 'placeholder', experiment_group: str = 'placeholder'):
    """Setup MLflow experiment and run."""

    # mlflow_dir = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    # mlflow_dir = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    # mlflow.set_tracking_uri(f"file://{mlflow_dir}")
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

    agent_cls = gin.get_bindings('run_single_seed')['agent_cls']
    agent_cls_name = agent_cls.__name__
    agent_params = gin.get_bindings(agent_cls_name)
    
    mlflow.log_params({
        "seed": seed,
        "agent_class": agent_cls_name,
        **{f"agent_{k}": v for k, v in agent_params.items()},
    })


def log_history_to_mlflow(history: History):
    """Log training history to MLflow."""
    train_episodes = gin.get_bindings('run_loop')['train_episodes']
    max_episode_steps = gin.get_bindings('run_loop')['max_episode_steps']
    evaluate_every = gin.get_bindings('run_loop')['evaluate_every']
    eval_episodes = gin.get_bindings('run_loop')['eval_episodes']

    for episode in range(train_episodes):
        mlflow.log_metrics({
            # "train/return": float(history.train_returns[episode]),
            "train/discounted_return": float(history.train_discounted_returns[episode]),
            # "train/length": int(history.train_lengths[episode]),
            # "train/loss": float(history.train_losses[episode]),
        }, step=episode * max_episode_steps)
    
    num_evals = train_episodes // evaluate_every
    for episode in range(num_evals):
        # eval_start = eval_idx * eval_episodes
        # eval_end = eval_start + eval_episodes
        
        # eval_returns = history.eval_returns[eval_start:eval_end]
        # eval_disc_returns = history.eval_discounted_returns[eval_start:eval_end]
        # eval_lengths = history.eval_lengths[eval_start:eval_end]
        
        mlflow.log_metrics({
            # "eval/mean_return": float(jnp.mean(eval_returns)),
            # "eval/std_return": float(jnp.std(eval_returns)),
            "eval/mean_discounted_return": float(history.eval_discounted_returns[episode]),
            # "eval/mean_length": float(jnp.mean(eval_lengths)),
        }, step=episode * max_episode_steps)
    
    # final_train_window = 100
    # mlflow.log_metrics({
    #     "final/train_return_mean": float(jnp.mean(history.train_returns[-final_train_window:])),
    #     "final/train_return_std": float(jnp.std(history.train_returns[-final_train_window:])),
    #     "final/eval_return_mean": float(jnp.mean(history.eval_returns)),
    #     "final/eval_return_std": float(jnp.std(history.eval_returns)),
    # })


# def save_agent(agent_state: Any, config: Dict[str, Any], seed: int):
#     """Save final agent state."""
#     if config['logging'].get('save_final_agent', False):
#         save_path = f"agent_seed_{seed}.npz"
#         # Convert agent state to numpy and save
#         state_dict = {
#             'q_table': np.array(agent_state.q_table),
#             'visit_counts': np.array(agent_state.visit_counts),
#             'step': int(agent_state.step),
#         }
#         np.savez(save_path, **state_dict)
#         mlflow.log_artifact(save_path)
#         os.remove(save_path)  # Clean up local file

@gin.configurable
def run_single_seed(seed: int, buffer_size: int = 1, env_cls: Type[BaseJaxEnv] = BaseJaxEnv, agent_cls: Type[BaseAgent] = BaseAgent):
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
    
        buffer_state = BufferState(
            observations=jnp.zeros(buffer_size),
            actions=jnp.zeros(buffer_size),
            rewards=jnp.zeros(buffer_size),
            discounts=jnp.zeros(buffer_size),
            next_observations=jnp.zeros(buffer_size),
            position=0,
            size=0
        )
        
        history = run_loop(
            agent=agent,
            environment=env,
            buffer_state=buffer_state,
            agent_state=agent_state,
            seed=seed,
            
        )
        
        log_history_to_mlflow(history)
        # save_agent(final_agent_state, config, seed)

    finally:
        mlflow.end_run()


def main():
    parser = argparse.ArgumentParser(description="Run RL experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (will use SLURM_ARRAY_TASK_ID if not provided)")
    args = parser.parse_args()
    
    gin.parse_config_file(args.config)
    
    if args.seed is None:
        seed = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
    else:
        seed = args.seed

    # for seed in range(30):
    setup_mlflow(seed=seed)
    run_single_seed(seed=seed)


if __name__ == "__main__":
    main()
