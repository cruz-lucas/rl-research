from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gin
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from rl_research.agents import BaseAgent, DQNRmaxRND
from rl_research.buffers import ReplayBuffer as _ReplayBuffer
from rl_research.buffers import Transition
from rl_research.environments import CardinalNavixWrapper
from rl_research.environments import NavixWrapper as _NavixWrapper
from rl_research.environments.navix import ACTION_NAMES, ACTION_TO_NAVIX_DIRECTION
from rl_research.experiment import restore_agent_checkpoint
from rl_research.utils import setup_mlflow as _setup_mlflow


_ = (_ReplayBuffer, _NavixWrapper, _setup_mlflow)

DEFAULT_ENV_ID = "Navix-Empty-16x16-v0"
OBSERVATION_MODES = ("symbolic", "position", "onehot_position", "tabular")
ANALYSIS_KINDS = ("auto", "rnd", "model_based")
POLICY_NAMES = ("random_policy", "agent_policy")
# POLICY_NAMES = ["random_policy"]
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
    observation_mode: str = "symbolic"
    onehot_obs_action_pair: bool = False
    linear_function_approximation: bool = False
    analysis_kind: str = "auto"


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
    final_agent_visit_counts: np.ndarray
    final_knownness: np.ndarray
    final_q_values: np.ndarray
    valid_state_mask: np.ndarray
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
            "final_agent_visit_counts": self.final_agent_visit_counts,
            "final_knownness": self.final_knownness,
            "final_q_values": self.final_q_values,
            "valid_state_mask": self.valid_state_mask,
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
        online_bonus_sum = np.asarray(arrays.get("online_bonus_sum", final_bonus_sum))
        online_bonus_mean = np.asarray(
            arrays.get("online_bonus_mean", final_bonus_mean)
        )
        online_bonus_eval_counts = np.asarray(
            arrays.get("online_bonus_eval_counts", final_bonus_eval_counts)
        )
        visitation_counts = np.asarray(arrays["visitation_counts"])
        state_visit_counts = np.asarray(arrays["state_visit_counts"])
        final_agent_visit_counts = np.asarray(
            arrays.get("final_agent_visit_counts", visitation_counts)
        )
        final_knownness = np.asarray(
            arrays.get("final_knownness", np.zeros_like(visitation_counts))
        )
        final_q_values = np.asarray(
            arrays.get(
                "final_q_values", np.zeros_like(visitation_counts, dtype=np.float32)
            )
        )
        valid_state_mask = np.asarray(
            arrays.get("valid_state_mask", state_visit_counts > 0)
        )
        return cls(
            policy_name=policy_name,
            visitation_counts=visitation_counts,
            state_visit_counts=state_visit_counts,
            online_bonus_sum=online_bonus_sum,
            online_bonus_mean=online_bonus_mean,
            online_bonus_eval_counts=online_bonus_eval_counts,
            final_bonus_sum=final_bonus_sum,
            final_bonus_mean=final_bonus_mean,
            final_bonus_eval_counts=final_bonus_eval_counts,
            final_agent_visit_counts=final_agent_visit_counts,
            final_knownness=final_knownness,
            final_q_values=final_q_values,
            valid_state_mask=valid_state_mask,
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


def _callable_accepts_kwarg(fn: Any, kwarg: str) -> bool:
    signature = inspect.signature(fn)
    return kwarg in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _parse_agent_from_config(
    settings: CollectionSettings,
    num_states: int,
    num_actions: int,
) -> tuple[BaseAgent, Any, dict[str, Any]]:
    if settings.analysis_kind not in ANALYSIS_KINDS:
        raise ValueError(
            f"Unsupported analysis_kind {settings.analysis_kind!r}. "
            f"Choose from {', '.join(ANALYSIS_KINDS)}."
        )
    if settings.observation_mode not in OBSERVATION_MODES:
        raise ValueError(
            f"Unsupported observation_mode {settings.observation_mode!r}. "
            f"Choose from {', '.join(OBSERVATION_MODES)}."
        )
    if settings.onehot_obs_action_pair and settings.observation_mode not in {
        "tabular",
        "onehot_position",
    }:
        raise ValueError(
            "--onehot-obs-action-pair requires --observation-mode tabular "
            "or onehot_position."
        )

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

    agent_init_kwargs: dict[str, Any] = {
        "num_states": int(num_states),
        "num_actions": num_actions,
    }
    if _callable_accepts_kwarg(agent_cls, "seed"):
        agent_init_kwargs["seed"] = settings.seed
    if settings.onehot_obs_action_pair:
        agent_init_kwargs["rnd_action_conditioning"] = "pair"
    if settings.linear_function_approximation:
        linear_overrides = {
            "hidden_dims": (),
            "rnd_hidden_dims": (),
            "normalization": "none",
            "rnd_normalization": "none",
        }
        agent_init_kwargs.update(
            {
                key: value
                for key, value in linear_overrides.items()
                if _callable_accepts_kwarg(agent_cls, key)
            }
        )

    agent = agent_cls(**agent_init_kwargs)
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

    has_rnd_bonus = hasattr(agent, "_compute_decision_bonus")
    has_model_diagnostics = hasattr(agent_state, "q_table") and hasattr(
        agent_state,
        "visit_counts",
    )
    if settings.analysis_kind == "rnd" and not has_rnd_bonus:
        raise TypeError(
            f"{agent.__class__.__name__} does not expose RND decision bonuses."
        )
    if settings.analysis_kind == "model_based" and not has_model_diagnostics:
        raise TypeError(
            f"{agent.__class__.__name__} does not expose q_table and visit_counts."
        )

    resolved_analysis_kind = settings.analysis_kind
    if resolved_analysis_kind == "auto":
        resolved_analysis_kind = "rnd" if has_rnd_bonus else "model_based"
    if resolved_analysis_kind == "model_based" and not has_model_diagnostics:
        raise TypeError(
            f"{agent.__class__.__name__} does not expose model-based diagnostics. "
            "Use an R-Max/MBIE-EB-style tabular agent or set --analysis-kind rnd."
        )

    metadata = {
        "agent_class": agent_cls.__name__,
        "agent_bindings": _jsonable(gin.get_bindings(agent_cls.__name__)),
        "run_loop_bindings": _jsonable(run_bindings),
        "checkpoint_mode": checkpoint_mode,
        "agent_init_overrides": _jsonable(agent_init_kwargs),
        "analysis_kind": resolved_analysis_kind,
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
        # agent_state = agent._update_intrinsic_stats(
        #     state=agent_state,
        #     prediction_error=jax.lax.stop_gradient(prediction_error),
        # )

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

        bonus_batch = 500 * agent._compute_decision_bonus(
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


def _valid_empty_room_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.bool_)
    if height > 2 and width > 2:
        mask[1 : height - 1, 1 : width - 1] = True
    return mask


def _position_state_ids(height: int, width: int) -> np.ndarray:
    return np.arange(height * width, dtype=np.int32).reshape(height, width)


def _knownness_threshold(agent: BaseAgent, fallback: int) -> int:
    threshold = getattr(agent, "known_threshold", fallback)
    try:
        threshold_float = float(jax.device_get(threshold))
    except (TypeError, ValueError, OverflowError):
        threshold_float = float(fallback)
    if not np.isfinite(threshold_float):
        return int(fallback)
    return max(int(threshold_float), 1)


def _extract_grid_q_values(
    agent_state: Any,
    *,
    height: int,
    width: int,
    num_actions: int,
) -> np.ndarray:
    q_values = np.zeros((height, width, num_actions), dtype=np.float32)
    if not hasattr(agent_state, "q_table"):
        return q_values

    q_table = np.asarray(jax.device_get(agent_state.q_table), dtype=np.float32)
    state_ids = _position_state_ids(height, width)
    valid_ids = state_ids < q_table.shape[0]
    q_values[valid_ids] = q_table[state_ids[valid_ids], :num_actions]
    return q_values


def _extract_grid_visit_counts(
    agent_state: Any,
    *,
    height: int,
    width: int,
    num_actions: int,
) -> np.ndarray:
    counts = np.zeros((height, width, num_actions), dtype=np.float32)
    if not hasattr(agent_state, "visit_counts"):
        return counts

    visit_counts = np.asarray(
        jax.device_get(agent_state.visit_counts), dtype=np.float32
    )
    state_ids = _position_state_ids(height, width)
    valid_ids = state_ids < visit_counts.shape[0]
    counts[valid_ids] = visit_counts[state_ids[valid_ids], :num_actions]
    return counts


def _single_transition(
    observation: np.ndarray | jax.Array,
    action: int,
    reward: float,
    next_observation: np.ndarray | jax.Array,
    terminal: bool,
    discount: float,
) -> Transition:
    return Transition(
        observation=jnp.asarray(observation).reshape(1, -1),
        action=jnp.asarray([action], dtype=jnp.int32),
        reward=jnp.asarray([reward], dtype=jnp.float32),
        discount=jnp.asarray([discount], dtype=jnp.float32),
        next_observation=jnp.asarray(next_observation).reshape(1, -1),
        terminal=jnp.asarray([terminal], dtype=jnp.bool_),
        mask=jnp.ones((1,), dtype=jnp.bool_),
    )


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
    collect_rnd_bonus = settings.analysis_kind in {"auto", "rnd"} and hasattr(
        agent, "_compute_decision_bonus"
    )
    train_model_during_rollout = ("agent_policy" in POLICY_NAMES) and (settings.analysis_kind == "model_based") or (
        settings.analysis_kind == "auto" and hasattr(rollout_agent_state, "q_table")
    )

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
            if collect_rnd_bonus:
                decision_bonus = _compute_decision_bonus(
                    agent,
                    rollout_agent_state,
                    observation_np,
                    num_actions=num_actions,
                )
            else:
                decision_bonus = np.zeros((num_actions,), dtype=np.float32)

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
                    is_training=train_model_during_rollout,
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

            if train_model_during_rollout:
                rollout_agent_state, _ = agent.update(
                    rollout_agent_state,
                    _single_transition(
                        observation_np,
                        action_int,
                        reward_float,
                        np.asarray(jax.device_get(observation)).copy(),
                        terminated_bool,
                        discount=float(getattr(agent, "discount", 1.0)),
                    ),
                )

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

        if settings.train_rnd_after_each_episode and collect_rnd_bonus:
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
        dtype=environment.observation_dtype,
        trailing_shape=tuple(environment.observation_shape),
    )
    if collect_rnd_bonus:
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
    else:
        final_bonus_sum = np.zeros_like(online_bonus_sum)
        final_bonus_mean = np.zeros_like(online_bonus_sum)
        final_bonus_eval_counts = np.zeros((height, width), dtype=np.int32)

    valid_state_mask = _valid_empty_room_mask(height, width)
    final_agent_visit_counts = _extract_grid_visit_counts(
        rollout_agent_state,
        height=height,
        width=width,
        num_actions=num_actions,
    )
    final_q_values = _extract_grid_q_values(
        rollout_agent_state,
        height=height,
        width=width,
        num_actions=num_actions,
    )
    knownness_threshold = _knownness_threshold(agent, settings.visitation_threshold)
    final_knownness = (final_agent_visit_counts >= float(knownness_threshold)).astype(
        np.float32
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
        final_agent_visit_counts=final_agent_visit_counts,
        final_knownness=final_knownness,
        final_q_values=final_q_values,
        valid_state_mask=valid_state_mask,
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
    valid_state_action_mask = np.repeat(
        rollout.valid_state_mask[..., None].astype(bool),
        rollout.visitation_counts.shape[-1],
        axis=-1,
    )
    known_by_model = valid_state_action_mask & rollout.final_knownness.astype(bool)
    final_q_values = rollout.final_q_values[valid_state_action_mask]
    final_agent_visit_values = rollout.final_agent_visit_counts[valid_state_action_mask]
    model_known_total = int(np.count_nonzero(valid_state_action_mask))

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
                np.count_nonzero(state_action_mask & ~known_by_visitation) / known_total
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
        "model_known_fraction": (
            float(np.count_nonzero(known_by_model) / model_known_total)
            if model_known_total
            else 0.0
        ),
        "model_unknown_fraction": (
            float(
                np.count_nonzero(valid_state_action_mask & ~known_by_model)
                / model_known_total
            )
            if model_known_total
            else 0.0
        ),
        "model_visit_count_stats": _stat_block(final_agent_visit_values),
        "q_value_stats": _stat_block(final_q_values),
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
        "observation_mode": settings.observation_mode,
        "observation_shape": list(environment.observation_shape),
        "observation_dtype": np.dtype(environment.observation_dtype).name,
        "onehot_obs_action_pair": bool(settings.onehot_obs_action_pair),
        "linear_function_approximation": bool(settings.linear_function_approximation),
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
        "resolved_analysis_kind": agent_metadata.get("analysis_kind"),
    }


def collect_knownness_rollouts(
    settings: CollectionSettings,
) -> tuple[dict[str, PolicyRollout], dict[str, Any]]:
    environment = CardinalNavixWrapper(
        env_id=settings.env_id,
        max_steps=settings.max_steps,
        observation_mode=settings.observation_mode,
    )
    agent, agent_state, agent_metadata = _parse_agent_from_config(
        settings,
        num_states=int(environment.num_observation_states),
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
