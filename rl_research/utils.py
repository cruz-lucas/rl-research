import time
from flax import nnx
import numpy as np
import gin
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import Metric

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
    metrics = []
    train_disc_returns = []
    train_returns = []
    timestamp = int(time.time() * 1000)
    batch_size = 100_000

    def __call__(self, log_metrics: dict):
        self.metrics.extend([
            Metric(
                key="train/return",
                value=float(log_metrics["metrics"].train_returns),
                step=int(log_metrics["step"]),
                timestamp=self.timestamp,
            ),
            Metric(
                key="train/discounted_return",
                value=float(log_metrics["metrics"].train_discounted_returns),
                step=int(log_metrics["step"]),
                timestamp=self.timestamp,
            ),
            Metric(
                key="train/loss",
                value=float(log_metrics["metrics"].train_losses),
                step=int(log_metrics["step"]),
                timestamp=self.timestamp,
            ),
            Metric(
                key="train/episode_length",
                value=float(log_metrics["metrics"].train_lengths),
                step=int(log_metrics["step"]),
                timestamp=self.timestamp,
            ),
        ])

        self.train_disc_returns.append(log_metrics["metrics"].train_discounted_returns)
        self.train_returns.append(log_metrics["metrics"].train_returns)

    def flush_summary(self):
        client = MlflowClient()
        run_id = mlflow.active_run().info.run_id
        
        for i in range(0, len(self.metrics), self.batch_size):
            client.log_batch(
                run_id=run_id,
                metrics=self.metrics[i:i + self.batch_size],
            )

        if len(self.train_disc_returns) >= 100:
            mlflow.log_metrics(
                {
                    "summary/last100_train_disc_return_mean": float(np.mean(self.train_disc_returns[-100:])),
                    "summary/last100_train_return_mean": float(np.mean(self.train_returns[-100:])),
                    "summary/train_disc_return_mean": float(np.mean(self.train_disc_returns)),
                    "summary/train_return_mean": float(np.mean(self.train_returns)),
                },
            )