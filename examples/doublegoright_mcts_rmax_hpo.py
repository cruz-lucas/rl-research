"""Hyperparameter search entry-point for the GoRight MCTS agent.

Each invocation samples a configuration from a fixed search space using the
provided seed, runs the experiment, and logs results via MLflow.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jrng

from goright.jax.env import GoRightJaxEnv, EnvParams

from rl_research.agents import RMaxMCTSAgent, RMaxMCTSAgentParams, goright_expectation_model
from rl_research.experiment import ExperimentParams, log_experiment, run_experiment


@dataclass(frozen=True)
class SearchConfig:
    discount: float
    num_simulations: int
    max_depth: int
    exploration_constant: float
    m: int
    initial_value: float


def sample_hyperparameters(seed: int) -> tuple[SearchConfig, jax.Array]:
    """Sample a reproducible hyperparameter configuration for the given seed."""
    key = jrng.PRNGKey(seed)

    discount = float(0.9)

    key, sim_key = jrng.split(key)
    simulation_choices = jnp.array(
        [0, 1, 2, 16, 32, 48, 64, 96, 128, 160, 192, 256, 320], dtype=jnp.int32
    )
    num_simulations = int(jrng.choice(sim_key, simulation_choices))

    key, depth_key = jrng.split(key)
    depth_choices = jnp.array(
        [1, 2, 6, 8, 10, 12, 14, 16, 18, 20, 24, 32], dtype=jnp.int32
    )
    max_depth = int(jrng.choice(depth_key, depth_choices))

    key, m_key = jrng.split(key)
    m_choices = jnp.array(
        [1, 5, 10, 15, 20, 25, 30, 35], dtype=jnp.int32
    )
    m = int(jrng.choice(m_key, m_choices))

    key, exploration_key = jrng.split(key)
    log_span = jnp.log(jnp.asarray(5.0)) - jnp.log(jnp.asarray(0.25))
    log_sample = jnp.log(jnp.asarray(0.25)) + jrng.uniform(
        exploration_key, dtype=jnp.float32
    ) * log_span
    exploration_constant = float(jnp.exp(log_sample))

    initial_value = 0.0

    key, experiment_key = jrng.split(key)
    config = SearchConfig(
        discount=discount,
        num_simulations=num_simulations,
        max_depth=max_depth,
        exploration_constant=exploration_constant,
        m=m,
        initial_value=initial_value,
    )
    return config, experiment_key


def format_float(value: float, precision: int = 3) -> str:
    """Format float values for run names without introducing dots."""
    return f"{value:.{precision}f}".replace(".", "p")


def main(seed: int) -> None:
    config, experiment_key = sample_hyperparameters(seed)
    print(f"Sampled configuration for seed {seed}: {config}")

    env_params = EnvParams(
        length=21,
        num_indicators=2,
        num_actions=2,
        first_checkpoint=10,
        first_reward=3.0,
        second_checkpoint=20,
        second_reward=6.0,
        is_partially_obs=True,
        mapping="default",
    )
    env = GoRightJaxEnv(env_params)

    dynamics_model = goright_expectation_model(
        length=env_params.length,
        first_checkpoint=env_params.first_checkpoint,
        first_reward=env_params.first_reward,
        second_checkpoint=env_params.second_checkpoint,
        second_reward=env_params.second_reward,
        num_indicators=env_params.num_indicators,
        is_partially_obs=env_params.is_partially_obs,
    )

    agent_params = RMaxMCTSAgentParams(
        num_states=env.env.observation_space.n,
        num_actions=env.env.action_space.n,
        discount=config.discount,
        dynamics_model=dynamics_model,
        num_simulations=config.num_simulations,
        max_depth=config.max_depth,
        exploration_constant=config.exploration_constant,
        r_max=6.0,
        m=config.m,
        initial_value=config.initial_value,
    )
    agent = RMaxMCTSAgent(params=agent_params, seed=seed)

    experiment_params = ExperimentParams(
        num_seeds=10,
        total_train_episodes=600,
        episode_length=500,
        eval_every=1,
        num_eval_episodes=1,
    )
    results = run_experiment(
        env=env,
        agent=agent,
        rng=experiment_key,
        params=experiment_params,
    )

    run_suffix = "_".join(
        [
            f"s{config.num_simulations}",
            f"d{config.max_depth}",
            f"c{format_float(config.exploration_constant, precision=2)}",
            f"v{format_float(config.initial_value, precision=2)}",
            f"g{format_float(config.discount, precision=3)}",
            f"seed{seed}",
        ]
    )
    agent_name = f"mcts_rmax_search_{run_suffix}"

    log_experiment(
        experiment_name="doublegoright_mcts_rmax_hparam_search",
        parent_run_name=agent_name,
        agent_name=agent_name,
        agent_params=agent_params,
        experiment_params=experiment_params,
        env_params=env_params,
        experiment_results=results,
    )


if __name__ == "__main__":
    import os

    in_job_array = "SLURM_ARRAY_TASK_ID" in os.environ
    if in_job_array:
        array_task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        array_task_count = int(os.environ["SLURM_ARRAY_TASK_COUNT"])

        print(f"This job is at index {array_task_id} in a job array of size {array_task_count}")
        seed = array_task_id
    else:
        seed = 0

    main(seed)
