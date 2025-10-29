import jax
from classic_pacmdp_envs import RiverSwimJaxEnv
from classic_pacmdp_envs.riverswim import EnvParams

from rl_research.agents import DTRMaxNStepAgent, DTRMaxNStepParams, riverswim_expectation_model
from rl_research.experiment import run_experiment, log_experiment, ExperimentParams


def main():
    agent_name = "rmax_dt_rollout"
    experiment_name = "riverswim"
    rng = jax.random.PRNGKey(0)

    env_params = EnvParams()
    env = RiverSwimJaxEnv()
    agent_params = DTRMaxNStepParams(
        num_states=env.env.observation_space.n,
        num_actions=env.env.action_space.n,
        initial_value=0.0,
        learning_rate=0.03,
        horizon=7,
        r_max=10_000,
        discount=0.95,
        m=30,
        dynamics_model=riverswim_expectation_model(),
    )
    agent = DTRMaxNStepAgent(params=agent_params)

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
