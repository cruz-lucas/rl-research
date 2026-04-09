import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


BATCH_SUMMARY_FIELDS = [
    "env_step",
    "update_index",
    "epsilon",
    "q_loss_full",
    "q_loss_no_intrinsic",
    "rnd_loss",
    "reward_normalizer_mean",
    "reward_normalizer_std",
    "reward_normalizer_count",
    "r_ext_mean",
    "r_ext_std",
    "r_int_current_mean",
    "r_int_current_std",
    "r_int_stored_mean",
    "r_int_stored_std",
    "beta_r_int_mean",
    "beta_r_int_std",
    "bootstrap_term_mean",
    "bootstrap_term_std",
    "target_mean",
    "target_std",
    "q_selected_mean",
    "q_selected_std",
    "td_error_mean",
    "td_error_std",
    "ratio_mean",
    "ratio_std",
    "age_mean",
    "age_std",
    "drift_mean",
    "drift_std",
]

GRADIENT_FIELDS = [
    "env_step",
    "update_index",
    "g_full_norm",
    "g_no_intrinsic_norm",
    "g_intrinsic_only_norm",
    "cosine_full_vs_no_intrinsic",
    "pre_clip_grad_norm",
    "post_clip_grad_norm",
    "was_clipped",
]

OPTIMIZER_FIELDS = [
    "env_step",
    "update_index",
    "optimizer",
    "grad_norm_for_step",
    "parameter_update_norm",
    "effective_step_size",
    "pre_clip_grad_norm",
    "post_clip_grad_norm",
    "was_clipped",
]

OPTIMIZER_LAYER_FIELDS = [
    "env_step",
    "update_index",
    "network",
    "layer",
    "pre_clip_grad_norm",
    "post_clip_grad_norm",
    "update_norm",
    "effective_step_size",
]

REPLAY_DIAGNOSTIC_FIELDS = [
    "env_step",
    "update_index",
    "sample_index",
    "buffer_index",
    "state_id",
    "next_state_id",
    "age",
    "stored_intrinsic_reward",
    "current_intrinsic_reward",
    "drift",
]

EPISODE_FIELDS = [
    "env_step",
    "episode_index",
    "episode_length",
    "episode_return_extrinsic",
    "episode_return_intrinsic",
    "episode_return_total",
    "done",
]

CORRELATION_FIELDS = [
    "env_step",
    "q_visit_corr_all_states",
    "q_intrinsic_corr_all_states",
    "q_visit_corr_visited_states",
    "q_intrinsic_corr_visited_states",
]


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_to_serializable(payload), handle, indent=2)


class JsonlWriter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_to_serializable(record)))
            handle.write("\n")


class CsvWriter:
    def __init__(self, path: Path, fieldnames: list[str]):
        self.path = path
        self.fieldnames = fieldnames
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writeheader()

    def write(self, record: dict[str, Any]) -> None:
        row = {field: _to_serializable(record.get(field)) for field in self.fieldnames}
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writerow(row)


class StructuredLogger:
    def __init__(
        self,
        output_dir: Path,
        state_metadata: dict[int, dict[str, Any]],
    ):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.batch_detail_writer = JsonlWriter(output_dir / "batch_details.jsonl")
        self.batch_summary_writer = CsvWriter(
            output_dir / "batch_summary.csv", BATCH_SUMMARY_FIELDS
        )
        self.gradient_writer = CsvWriter(
            output_dir / "gradient_stats.csv", GRADIENT_FIELDS
        )
        self.optimizer_writer = CsvWriter(
            output_dir / "optimizer_stats.csv", OPTIMIZER_FIELDS
        )
        self.optimizer_layer_writer = CsvWriter(
            output_dir / "optimizer_layer_stats.csv",
            OPTIMIZER_LAYER_FIELDS,
        )
        self.replay_writer = CsvWriter(
            output_dir / "replay_diagnostics.csv",
            REPLAY_DIAGNOSTIC_FIELDS,
        )
        self.episode_writer = CsvWriter(
            output_dir / "episode_metrics.csv",
            EPISODE_FIELDS,
        )
        self.correlation_writer = CsvWriter(
            output_dir / "correlation_stats.csv",
            CORRELATION_FIELDS,
        )

        self.state_stats: dict[int, dict[str, Any]] = {}
        for state_id, metadata in state_metadata.items():
            self.state_stats[int(state_id)] = {
                "state_id": int(state_id),
                "metadata": metadata,
                "count": 0,
                "r_int_mean": 0.0,
                "cumulative_intrinsic_reward": 0.0,
                "r_int_history": [],
                "q_history": [],
                "visit_count_history": [],
                "cumulative_intrinsic_reward_history": [],
            }

    def log_state_visit(
        self,
        state_id: int,
        step: int,
        intrinsic_reward: float,
    ) -> None:
        entry = self.state_stats[int(state_id)]
        entry["count"] += 1
        entry["cumulative_intrinsic_reward"] += float(intrinsic_reward)
        entry["r_int_mean"] = (
            entry["cumulative_intrinsic_reward"] / max(1, entry["count"])
        )
        entry["r_int_history"].append(
            {"step": int(step), "value": float(intrinsic_reward)}
        )

    def log_batch_detail(self, record: dict[str, Any]) -> None:
        self.batch_detail_writer.write(record)

    def log_batch_summary(self, record: dict[str, Any]) -> None:
        self.batch_summary_writer.write(record)

    def log_gradient_stats(self, record: dict[str, Any]) -> None:
        self.gradient_writer.write(record)

    def log_optimizer_stats(self, record: dict[str, Any]) -> None:
        self.optimizer_writer.write(record)

    def log_optimizer_layer_stats(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            self.optimizer_layer_writer.write(row)

    def log_replay_rows(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            self.replay_writer.write(row)

    def log_episode(self, record: dict[str, Any]) -> None:
        self.episode_writer.write(record)

    def log_correlation(self, record: dict[str, Any]) -> None:
        self.correlation_writer.write(record)

    def log_q_snapshot(
        self,
        step: int,
        q_values: np.ndarray,
        visit_counts: np.ndarray,
        cumulative_intrinsic: np.ndarray,
    ) -> None:
        q_values = np.asarray(q_values, dtype=np.float32)
        visit_counts = np.asarray(visit_counts, dtype=np.int64)
        cumulative_intrinsic = np.asarray(cumulative_intrinsic, dtype=np.float32)

        for state_id, entry in self.state_stats.items():
            state_q = q_values[state_id]
            entry["q_history"].append(
                {
                    "step": int(step),
                    "q_values": state_q.tolist(),
                    "q_mean": float(np.mean(state_q)),
                }
            )
            entry["visit_count_history"].append(
                {"step": int(step), "value": int(visit_counts[state_id])}
            )
            entry["cumulative_intrinsic_reward_history"].append(
                {
                    "step": int(step),
                    "value": float(cumulative_intrinsic[state_id]),
                }
            )

    def finalize(self, run_summary: dict[str, Any]) -> None:
        write_json(self.output_dir / "state_stats.json", self.state_stats)
        write_json(self.output_dir / "run_summary.json", run_summary)
