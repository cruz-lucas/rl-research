import base64
import hashlib
import json
import time
from pathlib import Path
from threading import Lock
from typing import Any

import mlflow
import numpy as np


def pack_array(array: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(array))
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "data_b64": base64.b64encode(array.tobytes()).decode("ascii"),
    }


def unpack_array(payload: dict[str, Any]) -> np.ndarray:
    array = np.frombuffer(
        base64.b64decode(payload["data_b64"]),
        dtype=np.dtype(payload["dtype"]),
    )
    return array.reshape(payload["shape"])


def observation_key(array: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(array).reshape(-1))
    return hashlib.blake2b(array.tobytes(), digest_size=16).hexdigest()


def observation_keys(array: np.ndarray) -> list[str]:
    array = np.asarray(array)
    if array.ndim == 1:
        return [observation_key(array)]

    flat = np.ascontiguousarray(array.reshape(array.shape[0], -1))
    return [
        hashlib.blake2b(row.tobytes(), digest_size=16).hexdigest() for row in flat
    ]


class DQNRNDDebugLogger:
    def __init__(
        self,
        *,
        log_dir: str | Path,
        agent_class: str = "DQNRNDAgent",
        num_states: int,
        num_actions: int,
        discount: float,
        rnd_action_conditioning: str,
        normalize_observations: bool,
        compact_observations: bool = True,
        log_to_mlflow: bool = True,
    ) -> None:
        active_run = mlflow.active_run()
        if active_run is None:
            run_id = f"no_mlflow_{int(time.time() * 1000)}"
        else:
            run_id = active_run.info.run_id

        self.run_id = run_id
        self.log_dir = Path(log_dir).resolve() / run_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.decision_log_path = self.log_dir / "decision_trace.jsonl"
        self.update_log_path = self.log_dir / "update_trace.jsonl"
        self.observation_table_path = self.log_dir / "observation_table.jsonl"
        self.metadata_path = self.log_dir / "metadata.json"
        self._decision_file = self.decision_log_path.open(
            "a", encoding="utf-8", buffering=1
        )
        self._update_file = self.update_log_path.open(
            "a", encoding="utf-8", buffering=1
        )
        self._compact_observations = bool(compact_observations)
        self._observation_table_file = (
            self.observation_table_path.open("a", encoding="utf-8", buffering=1)
            if self._compact_observations
            else None
        )
        self._observation_ids: dict[str, int] = {}
        self._lock = Lock()
        self._closed = False
        self._log_to_mlflow = bool(log_to_mlflow)

        action_conditioning_note = (
            "Intrinsic rewards in the update trace are logged for both observation and "
            "next_observation. When rnd_action_conditioning != 'none', both are "
            "evaluated with the sampled action from the same transition."
        )
        metadata = {
            "agent_class": agent_class,
            "run_id": self.run_id,
            "num_states": int(num_states),
            "num_actions": int(num_actions),
            "discount": float(discount),
            "normalize_observations": bool(normalize_observations),
            "rnd_action_conditioning": rnd_action_conditioning,
            "decision_log_path": str(self.decision_log_path),
            "update_log_path": str(self.update_log_path),
            "observation_storage": {
                "mode": "table_ids" if self._compact_observations else "inline",
                "observation_table_path": (
                    str(self.observation_table_path)
                    if self._compact_observations
                    else None
                ),
                "payload_format": "base64-encoded raw bytes",
                "observation_key": "blake2b-128 digest of flattened raw bytes",
            },
            "notes": {
                "intrinsic_reward": action_conditioning_note,
            },
        }
        self.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def _write_jsonl(self, file_obj, payload: dict[str, Any]) -> None:
        file_obj.write(json.dumps(payload, separators=(",", ":")) + "\n")

    def _register_observation_locked(self, observation: np.ndarray) -> tuple[int, str]:
        observation = np.ascontiguousarray(np.asarray(observation))
        key = observation_key(observation)
        observation_id = self._observation_ids.get(key)
        if observation_id is None:
            observation_id = len(self._observation_ids)
            self._observation_ids[key] = observation_id
            if self._observation_table_file is not None:
                self._write_jsonl(
                    self._observation_table_file,
                    {
                        "observation_id": observation_id,
                        "observation_key": key,
                        "observation": pack_array(observation),
                    },
                )
        return observation_id, key

    def _register_observation_batch_locked(
        self,
        observations: np.ndarray,
    ) -> tuple[list[int], list[str]]:
        observation_ids: list[int] = []
        observation_keys_list: list[str] = []
        for observation in np.asarray(observations):
            observation_id, key = self._register_observation_locked(observation)
            observation_ids.append(observation_id)
            observation_keys_list.append(key)
        return observation_ids, observation_keys_list

    def log_decision(
        self,
        global_step: np.ndarray,
        observation: np.ndarray,
        q_values: np.ndarray,
        epsilon: np.ndarray,
        action: np.ndarray,
        decision_bonus: np.ndarray | None = None,
        decision_values: np.ndarray | None = None,
    ) -> None:
        observation = np.asarray(observation)
        with self._lock:
            record = {
                "global_step": int(np.asarray(global_step)),
                "q_values": np.asarray(q_values).tolist(),
                "action": int(np.asarray(action)),
            }
            if self._compact_observations:
                observation_id, _ = self._register_observation_locked(observation)
                record["observation_id"] = observation_id
            else:
                record["observation"] = pack_array(observation)
                record["observation_key"] = observation_key(observation)

            epsilon_value = float(np.asarray(epsilon))
            if np.isfinite(epsilon_value):
                record["epsilon"] = epsilon_value
            if decision_bonus is not None:
                record["decision_bonus"] = np.asarray(decision_bonus).tolist()
            if decision_values is not None:
                record["decision_values"] = np.asarray(decision_values).tolist()
            self._write_jsonl(self._decision_file, record)

    def log_update(
        self,
        global_step: np.ndarray,
        gradient_step: np.ndarray,
        observation: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_observation: np.ndarray,
        terminal: np.ndarray,
        q_values_observation: np.ndarray,
        q_values_next_observation_online: np.ndarray,
        q_values_next_observation_target: np.ndarray,
        intrinsic_reward_observation: np.ndarray,
        intrinsic_reward_next_observation: np.ndarray,
        intrinsic_reward_used: np.ndarray,
        target_without_intrinsic: np.ndarray,
        target_with_intrinsic: np.ndarray,
        q_loss: np.ndarray,
        rnd_loss: np.ndarray,
        q_grad_norm_with_intrinsic: np.ndarray,
        q_grad_norm_without_intrinsic: np.ndarray,
        rnd_grad_norm: np.ndarray,
    ) -> None:
        observation = np.asarray(observation)
        next_observation = np.asarray(next_observation)
        with self._lock:
            batch_payload = {
                "action": np.asarray(action).tolist(),
                "reward": np.asarray(reward).tolist(),
                "terminal": np.asarray(terminal).astype(bool).tolist(),
                "q_values_observation": np.asarray(q_values_observation).tolist(),
                "q_values_next_observation_online": np.asarray(
                    q_values_next_observation_online
                ).tolist(),
                "q_values_next_observation_target": np.asarray(
                    q_values_next_observation_target
                ).tolist(),
                "intrinsic_reward_observation": np.asarray(
                    intrinsic_reward_observation
                ).tolist(),
                "intrinsic_reward_next_observation": np.asarray(
                    intrinsic_reward_next_observation
                ).tolist(),
                "intrinsic_reward_used": np.asarray(intrinsic_reward_used).tolist(),
                "target_without_intrinsic": np.asarray(
                    target_without_intrinsic
                ).tolist(),
                "target_with_intrinsic": np.asarray(target_with_intrinsic).tolist(),
            }
            if self._compact_observations:
                observation_ids, _ = self._register_observation_batch_locked(observation)
                next_observation_ids, _ = self._register_observation_batch_locked(
                    next_observation
                )
                batch_payload["observation_ids"] = observation_ids
                batch_payload["next_observation_ids"] = next_observation_ids
            else:
                batch_payload["observation"] = pack_array(observation)
                batch_payload["observation_keys"] = observation_keys(observation)
                batch_payload["next_observation"] = pack_array(next_observation)
                batch_payload["next_observation_keys"] = observation_keys(
                    next_observation
                )

            record = {
                "global_step": int(np.asarray(global_step)),
                "gradient_step": int(np.asarray(gradient_step)),
                "batch": batch_payload,
                "q_loss": float(np.asarray(q_loss)),
                "rnd_loss": float(np.asarray(rnd_loss)),
                "loss": float(np.asarray(q_loss) + np.asarray(rnd_loss)),
                "q_grad_norm_with_intrinsic": float(
                    np.asarray(q_grad_norm_with_intrinsic)
                ),
                "q_grad_norm_without_intrinsic": float(
                    np.asarray(q_grad_norm_without_intrinsic)
                ),
                "rnd_grad_norm": float(np.asarray(rnd_grad_norm)),
            }
            self._write_jsonl(self._update_file, record)

    def close(self) -> None:
        if self._closed:
            return

        with self._lock:
            self._decision_file.close()
            self._update_file.close()
            if self._observation_table_file is not None:
                self._observation_table_file.close()
            self._closed = True

        if self._log_to_mlflow and mlflow.active_run() is not None:
            mlflow.log_artifacts(str(self.log_dir), artifact_path="debug_logs")
