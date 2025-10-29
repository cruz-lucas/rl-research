import jax
from goright.jax.env import GoRightJaxEnv, EnvParams

from rl_research.agents import DTUCBPlanner, DTUCBParams, goright_expectation_model
from rl_research.experiment import run_experiment, log_experiment, ExperimentParams


def main():
    agent_name = "ucb_dtp_setting4"
    experiment_name = "doublegoright"
    rng = jax.random.PRNGKey(0)

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
    env = GoRightJaxEnv(
        env_params
    )

    agent_params = DTUCBParams(
        num_states=env.env.observation_space.n,
        num_actions=env.env.action_space.n,
        discount=0.9,
        learning_rate=0.05,
        initial_value=0,
        horizon=3,
        beta=20.0,
        dynamics_model=goright_expectation_model(is_partially_obs=True),
        use_time_bonus=True
    )
    agent = DTUCBPlanner(params=agent_params)

    experiment_params = ExperimentParams(
        num_seeds=30,
        total_train_episodes=600,
        episode_length=500,
        eval_every=1,
        num_eval_episodes=1,
    )
    results = run_experiment(
        env=env,
        agent=agent,
        rng=rng,
        params=experiment_params
    )

    log_experiment(
        experiment_name=experiment_name,
        parent_run_name=agent_name,
        agent_name=agent_name,
        agent_params=agent_params,
        experiment_params=experiment_params,
        env_params=env_params,
        experiment_results=results,
    )


if __name__ == "__main__":
    main()
