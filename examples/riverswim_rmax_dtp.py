import jax
from classic_pacmdp_envs import RiverSwimJaxEnv

from rl_research.agents import DTRMaxNStepAgent, DTRMaxNStepParams, riverswim_expectation_model
from rl_research.examples import ExperimentConfig, TrackingConfig, run_tabular_mlflow_example


def main():
    rng = jax.random.PRNGKey(0)

    env = RiverSwimJaxEnv()
    agent_params = DTRMaxNStepParams(
        num_states=6,
        num_actions=2,
        initial_value=0.0,
        learning_rate=0.03,
        horizon=7,
        r_max=10_000,
        discount=0.95,
        m=30,
        dynamics_model=riverswim_expectation_model(),
    )
    agent = DTRMaxNStepAgent(params=agent_params)

    run_tabular_mlflow_example(
        env=env,
        agent=agent,
        agent_params=agent_params,
        rng=rng,
        run_config=ExperimentConfig(
            num_seeds=30,
            total_train_episodes=1,
            episode_length=5_000,
            eval_every=0,
            num_eval_episodes=0,
        ),
        tracking=TrackingConfig(
            experiment_name="riverswim",
            agent_name="rmax_dt_rollout",
            parent_run_name="rmax_dt_rollout",
            seed_run_name_template="rmax_dt_rollout_seed_{seed:03d}",
            parent_tags={"agent": "rmax_dt_rollout"},
            seed_tags={"agent": "rmax_dt_rollout"},
        ),
    )


if __name__ == "__main__":
    main()
