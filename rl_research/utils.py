from flax import nnx
import numpy as np
import gin
import mlflow

def extract_agent_ckpt(agent_state) -> dict:
    graphdef, state = nnx.split(agent_state.network)
    return {
        "graphdef": graphdef,
        "state": state,
        "step": agent_state.step,
    }

def restore_agent_ckpt(ckpt):
    agent_state = nnx.merge(ckpt["graphdef"], ckpt["state"])
    return agent_state.replace(step=ckpt["step"])


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


class RecordWriter:
    prev_metrics = None
    train_disc_returns = []
    train_returns = []

    def __call__(self, cur_metrics: dict | None):
        self.prev_metrics, log_metrics = cur_metrics, self.prev_metrics
        if log_metrics is None:
            return
        
        self.train_disc_returns.append(log_metrics["metrics"].train_discounted_returns)
        self.train_returns.append(log_metrics["metrics"].train_returns)
    
        metrics = {
            "train/return": float(log_metrics["metrics"].train_returns),
            "train/discounted_return": float(log_metrics["metrics"].train_discounted_returns),
            "train/loss": float(log_metrics["metrics"].train_losses),
            "train/episode_length": float(log_metrics["metrics"].train_lengths),
        }
        
        mlflow.log_metrics(
            metrics,
            step=int(log_metrics["step"]),
        )

    def flush_summary(self):
        self.__call__(None)  # flush last metrics

        mlflow.log_metrics(
            {
                "summary/last100_train_disc_return_mean": float(np.mean(self.train_disc_returns[-100:])),
                "summary/last100_train_return_mean": float(np.mean(self.train_returns[-100:])),
                "summary/train_disc_return_mean": float(np.mean(self.train_disc_returns)),
                "summary/train_return_mean": float(np.mean(self.train_returns)),
            },
        )