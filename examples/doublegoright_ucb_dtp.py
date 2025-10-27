import jax
from goright.jax.env import GoRightJaxEnv, EnvParams

from rl_research.agents import DTUCBPlanner, DTUCBParams, goright_expectation_model
from rl_research.examples import ExperimentConfig, TrackingConfig, run_tabular_mlflow_example


def main():
    agent_name = "ucb_dtp_setting4"
    experiment_name = "doublegoright"
    rng = jax.random.PRNGKey(0)

    env = GoRightJaxEnv(
        EnvParams(
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
    )

    agent_params = DTUCBParams(
        num_states=env.env.observation_space.n,
        num_actions=2,
        discount=0.9,
        learning_rate=0.05,
        initial_value=0,
        horizon=3,
        beta=20.0,
        dynamics_model=goright_expectation_model(is_partially_obs=True),
        use_time_bonus=True
    )
    agent = DTUCBPlanner(params=agent_params)

    run_tabular_mlflow_example(
        env=env,
        agent=agent,
        agent_params=agent_params,
        rng=rng,
        run_config=ExperimentConfig(
            num_seeds=30,
            total_train_episodes=600,
            episode_length=500,
            eval_every=1,
            num_eval_episodes=1,
        ),
        tracking=TrackingConfig(
            experiment_name=experiment_name,
            agent_name=agent_name,
        ),
    )


if __name__ == "__main__":
    main()
