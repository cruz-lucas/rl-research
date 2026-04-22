from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gin
import jax
import jax.numpy as jnp
import navix as nx
import numpy as np
from flax import nnx
from navix import observations
from navix.actions import _can_walk_there
from navix.grid import translate
from navix.states import State

from rl_research.agents import BaseAgent, DQNRmaxRND
from rl_research.buffers import ReplayBuffer as _ReplayBuffer
from rl_research.environments import NavixWrapper as _NavixWrapper
from rl_research.experiment import restore_agent_checkpoint
from rl_research.utils import setup_mlflow as _setup_mlflow


_ = (_ReplayBuffer, _NavixWrapper, _setup_mlflow)

DEFAULT_ENV_ID = "Navix-Empty-16x16-v0"
ACTION_NAMES = ("up", "down", "left", "right")
ACTION_TO_NAVIX_DIRECTION = {
    "up": 3,
    "down": 1,
    "left": 2,
    "right": 0,
}
POLICY_NAMES = ("random_policy", "agent_policy")
METADATA_FILENAME = "metadata.json"


@dataclass(frozen=True)
class CollectionSettings:
    output_dir: Path
    episodes: int
    seed: int = 0
    max_steps: int | None = None
    bonus_threshold: float = 1.0
    visitation_threshold: int = 1
    config_path: Path | None = None
    checkpoint_path: Path | None = None
    gin_bindings: tuple[str, ...] = ()
    env_id: str = DEFAULT_ENV_ID
    train_rnd_after_each_episode: bool = False
    rnd_train_epochs_per_episode: int = 1


@dataclass
class PolicyRollout:
    policy_name: str
    visitation_counts: np.ndarray
    state_visit_counts: np.ndarray
    online_bonus_sum: np.ndarray
    online_bonus_mean: np.ndarray
    online_bonus_eval_counts: np.ndarray
    final_bonus_sum: np.ndarray
    final_bonus_mean: np.ndarray
    final_bonus_eval_counts: np.ndarray
    trajectory_episode: np.ndarray
    trajectory_step: np.ndarray
    trajectory_row: np.ndarray
    trajectory_col: np.ndarray
    trajectory_action: np.ndarray
    trajectory_direction: np.ndarray
    trajectory_observation: np.ndarray
    trajectory_reward: np.ndarray
    trajectory_terminated: np.ndarray
    trajectory_truncated: np.ndarray
    trajectory_bonus: np.ndarray
    trajectory_chosen_bonus: np.ndarray
    episode_returns: np.ndarray
    episode_lengths: np.ndarray

    def to_npz_dict(self) -> dict[str, np.ndarray]:
        return {
            "visitation_counts": self.visitation_counts,
            "state_visit_counts": self.state_visit_counts,
            "online_bonus_sum": self.online_bonus_sum,
            "online_bonus_mean": self.online_bonus_mean,
            "online_bonus_eval_counts": self.online_bonus_eval_counts,
            "final_bonus_sum": self.final_bonus_sum,
            "final_bonus_mean": self.final_bonus_mean,
            "final_bonus_eval_counts": self.final_bonus_eval_counts,
            "trajectory_episode": self.trajectory_episode,
            "trajectory_step": self.trajectory_step,
            "trajectory_row": self.trajectory_row,
            "trajectory_col": self.trajectory_col,
            "trajectory_action": self.trajectory_action,
            "trajectory_direction": self.trajectory_direction,
            "trajectory_observation": self.trajectory_observation,
            "trajectory_reward": self.trajectory_reward,
            "trajectory_terminated": self.trajectory_terminated,
            "trajectory_truncated": self.trajectory_truncated,
            "trajectory_bonus": self.trajectory_bonus,
            "trajectory_chosen_bonus": self.trajectory_chosen_bonus,
            "episode_returns": self.episode_returns,
            "episode_lengths": self.episode_lengths,
        }

    @classmethod
    def from_npz(cls, policy_name: str, arrays: dict[str, np.ndarray]) -> PolicyRollout:
        final_bonus_sum = np.asarray(
            arrays.get("final_bonus_sum", arrays.get("bonus_sum"))
        )
        final_bonus_mean = np.asarray(
            arrays.get("final_bonus_mean", arrays.get("bonus_mean"))
        )
        final_bonus_eval_counts = np.asarray(
            arrays.get("final_bonus_eval_counts", arrays.get("bonus_eval_counts"))
        )
        online_bonus_sum = np.asarray(
            arrays.get("online_bonus_sum", final_bonus_sum)
        )
        online_bonus_mean = np.asarray(
            arrays.get("online_bonus_mean", final_bonus_mean)
        )
        online_bonus_eval_counts = np.asarray(
            arrays.get("online_bonus_eval_counts", final_bonus_eval_counts)
        )
        return cls(
            policy_name=policy_name,
            visitation_counts=np.asarray(arrays["visitation_counts"]),
            state_visit_counts=np.asarray(arrays["state_visit_counts"]),
            online_bonus_sum=online_bonus_sum,
            online_bonus_mean=online_bonus_mean,
            online_bonus_eval_counts=online_bonus_eval_counts,
            final_bonus_sum=final_bonus_sum,
            final_bonus_mean=final_bonus_mean,
            final_bonus_eval_counts=final_bonus_eval_counts,
            trajectory_episode=np.asarray(arrays["trajectory_episode"]),
            trajectory_step=np.asarray(arrays["trajectory_step"]),
            trajectory_row=np.asarray(arrays["trajectory_row"]),
            trajectory_col=np.asarray(arrays["trajectory_col"]),
            trajectory_action=np.asarray(arrays["trajectory_action"]),
            trajectory_direction=np.asarray(arrays["trajectory_direction"]),
            trajectory_observation=np.asarray(
                arrays.get("trajectory_observation", np.zeros((0,), dtype=np.uint8))
            ),
            trajectory_reward=np.asarray(arrays["trajectory_reward"]),
            trajectory_terminated=np.asarray(arrays["trajectory_terminated"]),
            trajectory_truncated=np.asarray(arrays["trajectory_truncated"]),
            trajectory_bonus=np.asarray(arrays["trajectory_bonus"]),
            trajectory_chosen_bonus=np.asarray(arrays["trajectory_chosen_bonus"]),
            episode_returns=np.asarray(arrays["episode_returns"]),
            episode_lengths=np.asarray(arrays["episode_lengths"]),
        )


def _move_absolute(state: State, direction: int) -> State:
    player = state.get_player(idx=0)
    absolute_direction = jnp.asarray(direction, dtype=jnp.int32)
    target_position = translate(player.position, absolute_direction)

    # Reuse Navix's own walkability/event logic so this wrapper only changes the
    # action semantics, not collisions, rewards, or termination behaviour.
    can_move, events = _can_walk_there(state, target_position)
    next_position = jnp.where(can_move, target_position, player.position)
    player = player.replace(position=next_position, direction=absolute_direction)

    return state.set_player(player).replace(events=events)


def _move_up(state: State) -> State:
    return _move_absolute(state, ACTION_TO_NAVIX_DIRECTION["up"])


def _move_down(state: State) -> State:
    return _move_absolute(state, ACTION_TO_NAVIX_DIRECTION["down"])


def _move_left(state: State) -> State:
    return _move_absolute(state, ACTION_TO_NAVIX_DIRECTION["left"])


def _move_right(state: State) -> State:
    return _move_absolute(state, ACTION_TO_NAVIX_DIRECTION["right"])


CARDINAL_ACTION_SET = (
    _move_up,
    _move_down,
    _move_left,
    _move_right,
)


class CardinalNavixWrapper:
    def __init__(self, env_id: str = DEFAULT_ENV_ID, max_steps: int | None = None):
        kwargs: dict[str, Any] = {
            "observation_fn": observations.symbolic,
            "action_set": CARDINAL_ACTION_SET,
        }
        if max_steps is not None:
            kwargs["max_steps"] = max_steps
        self.env = nx.make(env_id, **kwargs)

    def reset(self, key: jax.Array):
        timestep = self.env.reset(key)
        return timestep, timestep.observation

    def step(self, timestep, action: jax.Array):
        timestep = self.env.step(timestep, action)
        return (
            timestep,
            timestep.observation,
            timestep.reward,
            timestep.is_termination(),
            timestep.is_truncation(),
            timestep.info,
        )


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, type):
        return value.__name__
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    return str(value)


def _parse_agent_from_config(
    settings: CollectionSettings,
    observation_shape: tuple[int, ...],
    num_actions: int,
) -> tuple[BaseAgent, Any, dict[str, Any]]:
    gin.clear_config()

    if settings.config_path is not None:
        gin.parse_config_files_and_bindings(
            [str(settings.config_path)],
            list(settings.gin_bindings),
        )
        run_bindings = gin.get_bindings("run_loop")
        agent_cls = run_bindings.get("agent_cls", DQNRmaxRND)
    else:
        run_bindings = {}
        agent_cls = DQNRmaxRND

    num_states = int(np.prod(np.asarray(observation_shape)))
    agent = agent_cls(
        num_states=num_states,
        num_actions=num_actions,
        seed=settings.seed,
    )
    agent_state = agent.initial_state()

    checkpoint_mode = "random_init"
    if settings.checkpoint_path is not None:
        try:
            agent_state, checkpoint_mode = restore_agent_checkpoint(
                agent_state,
                settings.checkpoint_path,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to restore the analysis checkpoint. The checkpoint must match "
                "the 4-action cardinal-move agent architecture used by this script."
            ) from exc

    if not hasattr(agent, "_compute_decision_bonus"):
        raise TypeError(
            f"{agent.__class__.__name__} does not expose RND decision bonuses. "
            "Use a DQN+RND-style agent for this analysis."
        )

    metadata = {
        "agent_class": agent_cls.__name__,
        "agent_bindings": _jsonable(gin.get_bindings(agent_cls.__name__)),
        "run_loop_bindings": _jsonable(run_bindings),
        "checkpoint_mode": checkpoint_mode,
    }
    return agent, agent_state, metadata


def _compute_decision_bonus(
    agent: BaseAgent,
    agent_state: Any,
    observation: np.ndarray | jax.Array,
    num_actions: int,
) -> np.ndarray:
    raw_observation = jnp.asarray(observation).reshape(-1)
    bonus = agent._compute_decision_bonus(agent_state, raw_observation)
    bonus_np = np.asarray(jax.device_get(bonus), dtype=np.float32).reshape(-1)
    if bonus_np.shape != (num_actions,):
        raise ValueError(
            "Expected one RND bonus per action for the 4-action wrapper, "
            f"received shape {bonus_np.shape!r}."
        )
    return bonus_np


def _validate_rnd_training_support(agent: BaseAgent, agent_state: Any) -> None:
    required_agent_attrs = (
        "_compute_prediction_error",
        "_select_rnd_features",
        "_update_intrinsic_stats",
    )
    missing_agent_attrs = [
        attr for attr in required_agent_attrs if not hasattr(agent, attr)
    ]
    if missing_agent_attrs:
        raise TypeError(
            f"{agent.__class__.__name__} does not expose the RND hooks required for "
            f"episode-level RND training: {', '.join(missing_agent_attrs)}."
        )

    required_state_attrs = ("rnd_predictor_network", "rnd_optimizer")
    missing_state_attrs = [
        attr for attr in required_state_attrs if not hasattr(agent_state, attr)
    ]
    if missing_state_attrs:
        raise TypeError(
            "The restored agent state does not include the RND predictor/optimizer "
            f"needed for episode-level RND training: {', '.join(missing_state_attrs)}."
        )


def _train_rnd_after_episode(
    agent: BaseAgent,
    agent_state: Any,
    *,
    observations_batch: np.ndarray,
    actions_batch: np.ndarray,
    num_epochs: int,
) -> Any:
    if observations_batch.size == 0 or num_epochs < 1:
        return agent_state

    _validate_rnd_training_support(agent, agent_state)

    observations_jax = jnp.asarray(observations_batch).reshape(
        observations_batch.shape[0],
        -1,
    )
    actions_jax = jnp.asarray(actions_batch, dtype=jnp.int32)
    rnd_action = (
        actions_jax
        if getattr(agent, "rnd_action_conditioning", "none") != "none"
        else None
    )

    for _ in range(num_epochs):
        (
            prediction_error,
            rnd_input,
            rnd_target_features,
            _,
        ) = agent._compute_prediction_error(
            agent_state,
            observations_jax,
            rnd_action,
        )

        def rnd_loss_fn(network):
            predictor_features = agent._select_rnd_features(
                network(rnd_input),
                rnd_action,
            )
            return jnp.mean(jnp.square(predictor_features - rnd_target_features))

        _, rnd_grads = nnx.value_and_grad(rnd_loss_fn)(
            agent_state.rnd_predictor_network
        )
        agent_state.rnd_optimizer.update(agent_state.rnd_predictor_network, rnd_grads)
        agent_state = agent._update_intrinsic_stats(
            state=agent_state,
            prediction_error=jax.lax.stop_gradient(prediction_error),
        )

    return agent_state


def _compute_final_bonus_statistics(
    *,
    agent: BaseAgent,
    agent_state: Any,
    trajectory_observation: np.ndarray,
    trajectory_row: np.ndarray,
    trajectory_col: np.ndarray,
    height: int,
    width: int,
    num_actions: int,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    final_bonus_sum = np.zeros((height, width, num_actions), dtype=np.float32)
    final_bonus_eval_counts = np.zeros((height, width), dtype=np.int32)

    num_steps = int(trajectory_row.shape[0])
    if num_steps == 0:
        return (
            final_bonus_sum,
            np.zeros_like(final_bonus_sum),
            final_bonus_eval_counts,
        )

    for start_idx in range(0, num_steps, batch_size):
        end_idx = min(start_idx + batch_size, num_steps)
        observations_batch = trajectory_observation[start_idx:end_idx]
        rows_batch = trajectory_row[start_idx:end_idx]
        cols_batch = trajectory_col[start_idx:end_idx]
        observations_flat = observations_batch.reshape(observations_batch.shape[0], -1)

        bonus_batch = agent._compute_decision_bonus(
            agent_state,
            jnp.asarray(observations_flat),
        )
        bonus_batch = np.asarray(jax.device_get(bonus_batch), dtype=np.float32).reshape(
            -1, num_actions
        )

        np.add.at(final_bonus_eval_counts, (rows_batch, cols_batch), 1)
        for action_idx in range(num_actions):
            np.add.at(
                final_bonus_sum[..., action_idx],
                (rows_batch, cols_batch),
                bonus_batch[:, action_idx],
            )

    final_bonus_mean = np.divide(
        final_bonus_sum,
        final_bonus_eval_counts[..., None],
        out=np.zeros_like(final_bonus_sum),
        where=final_bonus_eval_counts[..., None] > 0,
    )
    return final_bonus_sum, final_bonus_mean, final_bonus_eval_counts


def _extract_player_position(timestep) -> tuple[int, int]:
    row, col = jax.device_get(timestep.state.get_player(idx=0).position)
    return int(row), int(col)


def _extract_player_direction(timestep) -> int:
    return int(jax.device_get(timestep.state.get_player(idx=0).direction))


def _stack_or_empty(
    values: list[np.ndarray],
    *,
    dtype: np.dtype[Any],
    trailing_shape: tuple[int, ...] = (),
) -> np.ndarray:
    if values:
        return np.asarray(values, dtype=dtype)
    return np.zeros((0, *trailing_shape), dtype=dtype)


def _collect_single_policy(
    *,
    policy_name: str,
    environment: CardinalNavixWrapper,
    agent: BaseAgent,
    agent_state: Any,
    settings: CollectionSettings,
) -> PolicyRollout:
    height = int(environment.env.height)
    width = int(environment.env.width)
    num_actions = int(environment.env.action_space.n)

    visitation_counts = np.zeros((height, width, num_actions), dtype=np.int32)
    state_visit_counts = np.zeros((height, width), dtype=np.int32)
    online_bonus_sum = np.zeros((height, width, num_actions), dtype=np.float32)
    online_bonus_eval_counts = np.zeros((height, width), dtype=np.int32)

    trajectory_episode: list[int] = []
    trajectory_step: list[int] = []
    trajectory_row: list[int] = []
    trajectory_col: list[int] = []
    trajectory_action: list[int] = []
    trajectory_direction: list[int] = []
    trajectory_observation: list[np.ndarray] = []
    trajectory_reward: list[float] = []
    trajectory_terminated: list[bool] = []
    trajectory_truncated: list[bool] = []
    trajectory_bonus: list[np.ndarray] = []
    trajectory_chosen_bonus: list[float] = []
    episode_returns: list[float] = []
    episode_lengths: list[int] = []

    key = jax.random.PRNGKey(settings.seed)
    rollout_agent_state = agent_state

    for episode_idx in range(settings.episodes):
        key, reset_key = jax.random.split(key)
        timestep, observation = environment.reset(reset_key)

        done = False
        step_in_episode = 0
        episode_return = 0.0
        episode_observations: list[np.ndarray] = []
        episode_actions: list[int] = []

        while not done:
            row, col = _extract_player_position(timestep)
            direction = _extract_player_direction(timestep)
            observation_np = np.asarray(jax.device_get(observation)).copy()
            decision_bonus = _compute_decision_bonus(
                agent,
                rollout_agent_state,
                observation_np,
                num_actions=num_actions,
            )

            # Track the online bonus seen during the rollout before any later
            # episode-level RND training changes the predictor. The final heatmap is
            # recomputed separately from the saved trajectory using the final RND.
            online_bonus_sum[row, col] += decision_bonus
            online_bonus_eval_counts[row, col] += 1
            state_visit_counts[row, col] += 1

            key, action_key = jax.random.split(key)
            if policy_name == "random_policy":
                action = jax.random.randint(
                    action_key,
                    shape=(),
                    minval=0,
                    maxval=num_actions,
                )
            elif policy_name == "agent_policy":
                rollout_agent_state, action = agent.select_action(
                    rollout_agent_state,
                    observation,
                    action_key,
                    is_training=False,
                )
            else:
                raise ValueError(f"Unknown policy '{policy_name}'.")

            action_int = int(jax.device_get(action))
            visitation_counts[row, col, action_int] += 1
            episode_observations.append(observation_np)
            episode_actions.append(action_int)

            timestep, observation, reward, terminated, truncated, _ = environment.step(
                timestep,
                action,
            )

            reward_float = float(jax.device_get(reward))
            terminated_bool = bool(jax.device_get(terminated))
            truncated_bool = bool(jax.device_get(truncated))

            trajectory_episode.append(episode_idx)
            trajectory_step.append(step_in_episode)
            trajectory_row.append(row)
            trajectory_col.append(col)
            trajectory_action.append(action_int)
            trajectory_direction.append(direction)
            trajectory_observation.append(observation_np)
            trajectory_reward.append(reward_float)
            trajectory_terminated.append(terminated_bool)
            trajectory_truncated.append(truncated_bool)
            trajectory_bonus.append(decision_bonus.copy())
            trajectory_chosen_bonus.append(float(decision_bonus[action_int]))

            episode_return += reward_float
            step_in_episode += 1
            done = terminated_bool or truncated_bool

        episode_returns.append(episode_return)
        episode_lengths.append(step_in_episode)

        if settings.train_rnd_after_each_episode:
            rollout_agent_state = _train_rnd_after_episode(
                agent,
                rollout_agent_state,
                observations_batch=np.asarray(episode_observations),
                actions_batch=np.asarray(episode_actions, dtype=np.int32),
                num_epochs=settings.rnd_train_epochs_per_episode,
            )

    online_bonus_mean = np.divide(
        online_bonus_sum,
        online_bonus_eval_counts[..., None],
        out=np.zeros_like(online_bonus_sum),
        where=online_bonus_eval_counts[..., None] > 0,
    )
    trajectory_observation_np = _stack_or_empty(
        trajectory_observation,
        dtype=np.uint8,
        trailing_shape=tuple(environment.env.observation_space.shape),
    )
    final_bonus_sum, final_bonus_mean, final_bonus_eval_counts = (
        _compute_final_bonus_statistics(
            agent=agent,
            agent_state=rollout_agent_state,
            trajectory_observation=trajectory_observation_np,
            trajectory_row=np.asarray(trajectory_row, dtype=np.int32),
            trajectory_col=np.asarray(trajectory_col, dtype=np.int32),
            height=height,
            width=width,
            num_actions=num_actions,
        )
    )

    return PolicyRollout(
        policy_name=policy_name,
        visitation_counts=visitation_counts,
        state_visit_counts=state_visit_counts,
        online_bonus_sum=online_bonus_sum,
        online_bonus_mean=online_bonus_mean,
        online_bonus_eval_counts=online_bonus_eval_counts,
        final_bonus_sum=final_bonus_sum,
        final_bonus_mean=final_bonus_mean,
        final_bonus_eval_counts=final_bonus_eval_counts,
        trajectory_episode=np.asarray(trajectory_episode, dtype=np.int32),
        trajectory_step=np.asarray(trajectory_step, dtype=np.int32),
        trajectory_row=np.asarray(trajectory_row, dtype=np.int32),
        trajectory_col=np.asarray(trajectory_col, dtype=np.int32),
        trajectory_action=np.asarray(trajectory_action, dtype=np.int32),
        trajectory_direction=np.asarray(trajectory_direction, dtype=np.int32),
        trajectory_observation=trajectory_observation_np,
        trajectory_reward=np.asarray(trajectory_reward, dtype=np.float32),
        trajectory_terminated=np.asarray(trajectory_terminated, dtype=np.bool_),
        trajectory_truncated=np.asarray(trajectory_truncated, dtype=np.bool_),
        trajectory_bonus=_stack_or_empty(
            trajectory_bonus,
            dtype=np.float32,
            trailing_shape=(num_actions,),
        ),
        trajectory_chosen_bonus=np.asarray(trajectory_chosen_bonus, dtype=np.float32),
        episode_returns=np.asarray(episode_returns, dtype=np.float32),
        episode_lengths=np.asarray(episode_lengths, dtype=np.int32),
    )


def compute_policy_summary(
    rollout: PolicyRollout,
    *,
    visitation_threshold: int,
    bonus_threshold: float,
) -> dict[str, Any]:
    cell_visit_mask = rollout.state_visit_counts > 0
    visited_cell_count = int(np.count_nonzero(cell_visit_mask))
    total_cell_count = int(rollout.state_visit_counts.size)

    state_action_mask = np.repeat(
        cell_visit_mask[..., None],
        rollout.visitation_counts.shape[-1],
        axis=-1,
    )
    known_by_visitation = state_action_mask & (
        rollout.visitation_counts >= visitation_threshold
    )
    known_by_bonus = state_action_mask & (rollout.final_bonus_mean <= bonus_threshold)

    def _stat_block(values: np.ndarray) -> dict[str, float | None]:
        if values.size == 0:
            return {"min": None, "max": None, "mean": None}
        return {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
        }

    known_total = int(np.count_nonzero(state_action_mask))
    visitation_cell_values = rollout.state_visit_counts[cell_visit_mask]
    visitation_state_action_values = rollout.visitation_counts[state_action_mask]
    final_bonus_values = rollout.final_bonus_mean[state_action_mask]
    online_bonus_values = rollout.online_bonus_mean[state_action_mask]

    return {
        "policy_name": rollout.policy_name,
        "episodes": int(rollout.episode_returns.size),
        "trajectory_steps": int(rollout.trajectory_action.size),
        "thresholds": {
            "visitation_threshold": int(visitation_threshold),
            "bonus_threshold": float(bonus_threshold),
        },
        "total_visited_cells": visited_cell_count,
        "visited_cell_fraction": (
            float(visited_cell_count / total_cell_count) if total_cell_count else 0.0
        ),
        "executed_state_actions": int(np.count_nonzero(rollout.visitation_counts)),
        "evaluated_state_actions": known_total,
        "visitation_known_fraction": (
            float(np.count_nonzero(known_by_visitation) / known_total)
            if known_total
            else 0.0
        ),
        "visitation_unknown_fraction": (
            float(
                np.count_nonzero(state_action_mask & ~known_by_visitation)
                / known_total
            )
            if known_total
            else 0.0
        ),
        "bonus_known_fraction": (
            float(np.count_nonzero(known_by_bonus) / known_total)
            if known_total
            else 0.0
        ),
        "bonus_unknown_fraction": (
            float(np.count_nonzero(state_action_mask & ~known_by_bonus) / known_total)
            if known_total
            else 0.0
        ),
        "cell_visitation_count_stats": _stat_block(visitation_cell_values),
        "state_action_visitation_count_stats": _stat_block(
            visitation_state_action_values
        ),
        "rnd_bonus_stats": _stat_block(final_bonus_values),
        "online_rnd_bonus_stats": _stat_block(online_bonus_values),
        "episode_return_stats": _stat_block(rollout.episode_returns),
        "episode_length_stats": _stat_block(rollout.episode_lengths),
    }


def _collection_metadata(
    *,
    settings: CollectionSettings,
    environment: CardinalNavixWrapper,
    agent_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "env_id": settings.env_id,
        "grid_shape": [int(environment.env.height), int(environment.env.width)],
        "action_names": list(ACTION_NAMES),
        "action_to_navix_direction": ACTION_TO_NAVIX_DIRECTION,
        "episodes": int(settings.episodes),
        "seed": int(settings.seed),
        "max_steps": int(environment.env.max_steps),
        "bonus_threshold": float(settings.bonus_threshold),
        "visitation_threshold": int(settings.visitation_threshold),
        "config_path": str(settings.config_path) if settings.config_path else None,
        "checkpoint_path": (
            str(settings.checkpoint_path) if settings.checkpoint_path else None
        ),
        "gin_bindings": list(settings.gin_bindings),
        "train_rnd_after_each_episode": bool(settings.train_rnd_after_each_episode),
        "rnd_train_epochs_per_episode": int(settings.rnd_train_epochs_per_episode),
        "online_bonus_aggregation": (
            "Mean of the four per-action RND bonuses evaluated at each visited state "
            "during the rollout, before any later episode-level RND training."
        ),
        "final_bonus_aggregation": (
            "Mean of the four per-action RND bonuses queried after collection with "
            "the final RND predictor over every visited trajectory state."
        ),
        **agent_metadata,
    }


def collect_knownness_rollouts(
    settings: CollectionSettings,
) -> tuple[dict[str, PolicyRollout], dict[str, Any]]:
    environment = CardinalNavixWrapper(
        env_id=settings.env_id,
        max_steps=settings.max_steps,
    )
    agent, agent_state, agent_metadata = _parse_agent_from_config(
        settings,
        observation_shape=tuple(environment.env.observation_space.shape),
        num_actions=int(environment.env.action_space.n),
    )

    rollouts = {
        policy_name: _collect_single_policy(
            policy_name=policy_name,
            environment=environment,
            agent=agent,
            agent_state=agent_state,
            settings=settings,
        )
        for policy_name in POLICY_NAMES
    }
    metadata = _collection_metadata(
        settings=settings,
        environment=environment,
        agent_metadata=agent_metadata,
    )
    return rollouts, metadata


def _policy_npz_path(output_dir: Path, policy_name: str) -> Path:
    return output_dir / f"{policy_name}.npz"


def _policy_summary_path(output_dir: Path, policy_name: str) -> Path:
    return output_dir / f"{policy_name}_summary.json"


def save_collection_metadata(output_dir: Path, metadata: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / METADATA_FILENAME
    metadata_path.write_text(
        json.dumps(_jsonable(metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata_path


def save_policy_rollout(output_dir: Path, rollout: PolicyRollout) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = _policy_npz_path(output_dir, rollout.policy_name)
    np.savez_compressed(npz_path, **rollout.to_npz_dict())
    return npz_path


def save_policy_summary(
    output_dir: Path,
    summary: dict[str, Any],
    *,
    policy_name: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = _policy_summary_path(output_dir, policy_name)
    summary_path.write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary_path


def load_collection_metadata(output_dir: Path) -> dict[str, Any]:
    metadata_path = output_dir / METADATA_FILENAME
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_policy_rollout(output_dir: Path, policy_name: str) -> PolicyRollout:
    npz_path = _policy_npz_path(output_dir, policy_name)
    with np.load(npz_path) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    return PolicyRollout.from_npz(policy_name=policy_name, arrays=arrays)
