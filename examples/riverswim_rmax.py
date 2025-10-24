import jax
from classic_pacmdp_envs import RiverSwimJaxEnv

from rl_research.agents import RMaxAgent, RMaxParams
from rl_research.examples import ExperimentConfig, TrackingConfig, run_tabular_mlflow_example


def main():
    rng = jax.random.PRNGKey(0)

    env = RiverSwimJaxEnv()
    agent_params = RMaxParams(
        num_states=6,
        num_actions=2,
        threshold=0.01,
        r_max=10_000,
        discount=0.95,
        m=16,
    )
    agent = RMaxAgent(params=agent_params)

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
            agent_name="rmax",
            parent_run_name="rmax",
            seed_run_name_template="rmax_seed_{seed:03d}",
            parent_tags={"agent": "Rmax"},
            seed_tags={"agent": "Rmax"},
        ),
    )


if __name__ == "__main__":
    main()
