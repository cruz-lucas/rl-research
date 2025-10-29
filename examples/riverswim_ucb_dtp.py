import jax
from classic_pacmdp_envs import RiverSwimJaxEnv
from classic_pacmdp_envs.riverswim import EnvParams

from rl_research.agents import DTUCBPlanner, DTUCBParams, riverswim_expectation_model
from rl_research.experiment import run_experiment, log_experiment, ExperimentParams


def main():
    agent_name = "mbie-eb"
    experiment_name = "riverswim"
    rng = jax.random.PRNGKey(0)

    env_params = EnvParams()
    env = RiverSwimJaxEnv()
    
    agent_params = DTUCBParams(
        num_states=env.env.observation_space.n,
        num_actions=env.env.action_space.n,
        initial_value=0.0,
        learning_rate=0.4,
        horizon=7,
        discount=0.95,
        beta=40.0,
        use_time_bonus=False,
        dynamics_model=riverswim_expectation_model(),
    )
    agent = DTUCBPlanner(params=agent_params)

    experiment_params = ExperimentParams(
        num_seeds=30,
        total_train_episodes=1,
        episode_length=5000,
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
