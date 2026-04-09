# ruff: noqa: I001

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax import traverse_util
from jax.flatten_util import ravel_pytree

from rl_research.environments import NavixWrapper


if __package__ in {None, ""}:
    from dqn_rnd_analysis.analysis import generate_all_plots
    from dqn_rnd_analysis.logger import StructuredLogger, write_json
    from dqn_rnd_analysis.networks import MLP
    from dqn_rnd_analysis.replay_buffer import ReplayBatch, ReplayBuffer
else:
    from .analysis import generate_all_plots
    from .logger import StructuredLogger, write_json
    from .networks import MLP
    from .replay_buffer import ReplayBatch, ReplayBuffer


PLAYER_TAG = 10
DOOR_TAG = 4
KEY_TAG = 5


@dataclass
class Args:
    output_root: Path = Path("dqn_rnd_analysis/outputs/dqn_rnd_analysis")
    run_name: str | None = None
    env_id: str = "GridDoorKey-5x5-layout1-v0"
    seeds: tuple[int, ...] = (0,)
    total_steps: int = 2048
    max_episode_steps: int = 100
    replay_capacity: int = 16384
    batch_size: int = 16
    warmup_steps: int = 0
    train_frequency: int = 1
    gradient_updates_per_train_step: int = 1
    target_update_frequency: int = 2048
    gamma: float = 0.99
    learning_rate: float = 4.8e-06
    rnd_learning_rate: float = 1.6e-3
    optimizer: str = "adam"
    hidden_dims: tuple[int, int] = (64, 64)
    rnd_hidden_dims: tuple[int, int] = (64, 64)
    rnd_output_dim: int = 1
    activation: str = "relu"
    epsilon_start: float = 0.6
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 120_000
    beta: float = 0.1
    use_intrinsic_reward: bool = True
    grad_clip_norm: float | None = 10.0
    intrinsic_reward_clip: float | None = 10.0
    intrinsic_reward_epsilon: float = 1e-8
    loss_type: str = "mse"
    huber_delta: float = 1.0
    state_eval_interval: int = 250
    plot_top_k_states: int = 6


@dataclass
class RewardNormalizer:
    epsilon: float = 1e-8
    clip: float | None = 10.0
    mean: float = 0.0
    var: float = 1.0
    count: float = 1e-4

    @property
    def std(self) -> float:
        return math.sqrt(max(self.var, self.epsilon))

    def normalize(self, values: np.ndarray) -> np.ndarray:
        normalized = np.asarray(values, dtype=np.float32) / self.std
        if self.clip is not None:
            normalized = np.clip(normalized, 0.0, float(self.clip))
        return normalized.astype(np.float32)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.size == 0:
            return

        batch_mean = float(np.mean(values))
        batch_var = float(np.var(values))
        batch_count = float(values.size)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / total_count)

        moment_a = self.var * self.count
        moment_b = batch_var * batch_count
        moment_2 = (
            moment_a
            + moment_b
            + (delta**2) * (self.count * batch_count / total_count)
        )
        new_var = max(moment_2 / total_count, self.epsilon)

        self.mean = new_mean
        self.var = new_var
        self.count = total_count


@dataclass
class AgentState:
    q_params: Any
    target_params: Any
    q_opt_state: optax.OptState
    rnd_predictor_params: Any
    rnd_target_params: Any
    rnd_opt_state: optax.OptState
    update_index: int = 0


def linear_schedule(step: int, start: float, end: float, decay_steps: int) -> float:
    fraction = min(max(step / max(1, decay_steps), 0.0), 1.0)
    return float(start + fraction * (end - start))


def build_optimizer(name: str, learning_rate: float) -> optax.GradientTransformation:
    normalized = name.strip().lower()
    if normalized == "adam":
        return optax.adam(learning_rate=learning_rate)
    if normalized == "sgd":
        return optax.sgd(learning_rate=learning_rate)
    raise ValueError(f"Unsupported optimizer {name!r}. Choose from adam or sgd.")


def td_loss(td_error: jax.Array, loss_type: str, huber_delta: float) -> jax.Array:
    normalized = loss_type.strip().lower()
    if normalized == "mse":
        return jnp.square(td_error)
    if normalized == "huber":
        return optax.huber_loss(td_error, delta=huber_delta)
    raise ValueError(f"Unsupported loss_type {loss_type!r}.")


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.allclose(np.std(x), 0.0) or np.allclose(np.std(y), 0.0):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def safe_cosine(x: jax.Array, y: jax.Array) -> float:
    x_norm = float(jnp.linalg.norm(x))
    y_norm = float(jnp.linalg.norm(y))
    if x_norm == 0.0 and y_norm == 0.0:
        return 1.0
    if x_norm == 0.0 or y_norm == 0.0:
        return 0.0
    return float(jnp.dot(x, y) / (x_norm * y_norm + 1e-8))


def clip_gradients(
    grads: Any,
    max_norm: float | None,
) -> tuple[Any, float, float, bool]:
    pre_norm = float(optax.global_norm(grads))
    if max_norm is None or max_norm <= 0.0:
        return grads, pre_norm, pre_norm, False

    scale = min(1.0, float(max_norm) / (pre_norm + 1e-8))
    clipped = jax.tree.map(lambda value: value * scale, grads)
    post_norm = float(optax.global_norm(clipped))
    return clipped, pre_norm, post_norm, scale < 1.0


def group_norms(tree: Any) -> dict[str, float]:
    params_tree = tree["params"] if hasattr(tree, "keys") and "params" in tree else tree
    flat_tree = traverse_util.flatten_dict(params_tree)
    grouped: dict[str, float] = {}
    for path, leaf in flat_tree.items():
        layer = path[0] if path else "root"
        grouped.setdefault(str(layer), 0.0)
        grouped[str(layer)] += float(np.sum(np.square(np.asarray(leaf))))
    return {layer: math.sqrt(value) for layer, value in grouped.items()}


def extract_state_components(
    obs: np.ndarray,
    grid_size: int,
) -> tuple[int, int, int, int]:
    obs = np.asarray(obs, dtype=np.uint8).reshape(grid_size, grid_size, 3)
    inner = grid_size - 2
    tags = obs[..., 0]

    player_locations = np.argwhere(tags == PLAYER_TAG)
    if player_locations.shape[0] != 1:
        raise ValueError("Expected exactly one player in the symbolic observation.")
    player_row, player_col = player_locations[0]
    player_pos = int((player_row - 1) * inner + (player_col - 1))

    door_locations = np.argwhere(tags == DOOR_TAG)
    if door_locations.shape[0] == 0:
        door_open = 1
    elif door_locations.shape[0] == 1:
        door_row, door_col = door_locations[0]
        door_open = int(obs[door_row, door_col, 2] == 0)
    else:
        raise ValueError("Expected at most one door in the symbolic observation.")

    key_locations = np.argwhere(tags == KEY_TAG)
    if key_locations.shape[0] == 0:
        key_pos = 0
    else:
        key_row, key_col = key_locations[0]
        key_pos = int(1 + (key_row - 1) * inner + (key_col - 1))

    direction = int(obs[player_row, player_col, 2])
    return player_pos, key_pos, door_open, direction


def components_to_state_id(
    player_pos: int,
    key_pos: int,
    door_open: int,
    direction: int,
    grid_size: int,
) -> int:
    inner = grid_size - 2
    return int(
        (((key_pos * inner * inner) + player_pos) * 2 + door_open) * 4 + direction
    )


def state_id_to_components(state_id: int, grid_size: int) -> tuple[int, int, int, int]:
    direction = int(state_id % 4)
    remainder = state_id // 4
    door_open = int(remainder % 2)
    remainder //= 2
    inner = grid_size - 2
    player_pos = int(remainder % (inner * inner))
    key_pos = int(remainder // (inner * inner))
    return player_pos, key_pos, door_open, direction


def components_to_features(
    player_pos: int,
    key_pos: int,
    door_open: int,
    direction: int,
    grid_size: int,
) -> np.ndarray:
    inner = grid_size - 2
    player_dim = inner * inner
    key_dim = player_dim + 1
    features = np.zeros((player_dim + key_dim + 2 + 4,), dtype=np.float32)

    features[player_pos] = 1.0
    features[player_dim + key_pos] = 1.0
    features[player_dim + key_dim + door_open] = 1.0
    features[player_dim + key_dim + 2 + direction] = 1.0
    return features


def observation_to_features_and_state_id(
    obs: np.ndarray,
    grid_size: int,
) -> tuple[np.ndarray, int]:
    player_pos, key_pos, door_open, direction = extract_state_components(obs, grid_size)
    state_id = components_to_state_id(
        player_pos=player_pos,
        key_pos=key_pos,
        door_open=door_open,
        direction=direction,
        grid_size=grid_size,
    )
    features = components_to_features(
        player_pos=player_pos,
        key_pos=key_pos,
        door_open=door_open,
        direction=direction,
        grid_size=grid_size,
    )
    return features, state_id


def build_state_feature_catalog(
    grid_size: int,
) -> tuple[np.ndarray, dict[int, dict[str, Any]]]:
    inner = grid_size - 2
    num_states = (inner * inner + 1) * inner * inner * 2 * 4
    features = []
    metadata: dict[int, dict[str, Any]] = {}
    for state_id in range(num_states):
        player_pos, key_pos, door_open, direction = state_id_to_components(
            state_id, grid_size
        )
        features.append(
            components_to_features(
                player_pos=player_pos,
                key_pos=key_pos,
                door_open=door_open,
                direction=direction,
                grid_size=grid_size,
            )
        )

        player_row = player_pos // inner + 1
        player_col = player_pos % inner + 1
        if key_pos == 0:
            key_label = "picked"
            key_row = None
            key_col = None
        else:
            key_row = (key_pos - 1) // inner + 1
            key_col = (key_pos - 1) % inner + 1
            key_label = f"({key_row}, {key_col})"

        metadata[state_id] = {
            "player_row": player_row,
            "player_col": player_col,
            "key_row": key_row,
            "key_col": key_col,
            "key_picked": key_pos == 0,
            "door_open": bool(door_open),
            "direction": direction,
            "label": (
                f"player=({player_row}, {player_col}) "
                f"key={key_label} "
                f"door={'open' if door_open else 'closed'} "
                f"dir={direction}"
            ),
        }
    return np.asarray(features, dtype=np.float32), metadata


def compute_intrinsic_rewards(
    predictor_network: MLP,
    predictor_params: Any,
    target_network: MLP,
    target_params: Any,
    observations: np.ndarray,
    reward_normalizer: RewardNormalizer,
) -> tuple[np.ndarray, np.ndarray]:
    inputs = jnp.asarray(observations, dtype=jnp.float32)
    target_features = target_network.apply({"params": target_params}, inputs)
    predictor_features = predictor_network.apply({"params": predictor_params}, inputs)
    raw_errors = jnp.mean(
        jnp.square(predictor_features - jax.lax.stop_gradient(target_features)),
        axis=-1,
    )
    raw_errors_np = np.asarray(raw_errors, dtype=np.float32)
    normalized = reward_normalizer.normalize(raw_errors_np)
    return raw_errors_np, normalized


def q_values_for_states(
    q_network: MLP,
    q_params: Any,
    features: np.ndarray,
) -> np.ndarray:
    q_values = q_network.apply({"params": q_params}, jnp.asarray(features, jnp.float32))
    return np.asarray(q_values, dtype=np.float32)


def summarize(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float32)
    return float(np.mean(values)), float(np.std(values))


def serialize_args(args: Args, seed: int, output_dir: Path) -> dict[str, Any]:
    payload = asdict(args)
    payload["output_root"] = str(args.output_root)
    payload["output_dir"] = str(output_dir)
    payload["seed"] = seed
    return payload


def make_snapshot_and_log(
    logger: StructuredLogger,
    step: int,
    q_network: MLP,
    q_params: Any,
    all_state_features: np.ndarray,
    state_visit_counts: np.ndarray,
    state_cumulative_intrinsic: np.ndarray,
) -> None:
    q_values = q_values_for_states(q_network, q_params, all_state_features)
    logger.log_q_snapshot(
        step=step,
        q_values=q_values,
        visit_counts=state_visit_counts,
        cumulative_intrinsic=state_cumulative_intrinsic,
    )

    q_mean = q_values.mean(axis=1)
    visited_mask = state_visit_counts > 0
    logger.log_correlation(
        {
            "env_step": int(step),
            "q_visit_corr_all_states": safe_pearson(q_mean, state_visit_counts),
            "q_intrinsic_corr_all_states": safe_pearson(
                q_mean, state_cumulative_intrinsic
            ),
            "q_visit_corr_visited_states": safe_pearson(
                q_mean[visited_mask], state_visit_counts[visited_mask]
            ),
            "q_intrinsic_corr_visited_states": safe_pearson(
                q_mean[visited_mask], state_cumulative_intrinsic[visited_mask]
            ),
        }
    )


def update_agent(
    agent_state: AgentState,
    batch: ReplayBatch,
    q_network: MLP,
    rnd_predictor_network: MLP,
    rnd_target_network: MLP,
    q_optimizer: optax.GradientTransformation,
    rnd_optimizer: optax.GradientTransformation,
    reward_normalizer: RewardNormalizer,
    args: Args,
    env_step: int,
    epsilon: float,
) -> tuple[AgentState, dict[str, Any]]:
    observations = jnp.asarray(batch.observations, dtype=jnp.float32)
    next_observations = jnp.asarray(batch.next_observations, dtype=jnp.float32)
    actions = jnp.asarray(batch.actions, dtype=jnp.int32)
    rewards_ext = jnp.asarray(batch.extrinsic_rewards, dtype=jnp.float32)
    dones = jnp.asarray(batch.dones.astype(np.float32), dtype=jnp.float32)
    beta_scale = float(args.beta if args.use_intrinsic_reward else 0.0)

    raw_intrinsic_np, intrinsic_current_np = compute_intrinsic_rewards(
        predictor_network=rnd_predictor_network,
        predictor_params=agent_state.rnd_predictor_params,
        target_network=rnd_target_network,
        target_params=agent_state.rnd_target_params,
        observations=batch.next_observations,
        reward_normalizer=reward_normalizer,
    )
    intrinsic_current = jnp.asarray(intrinsic_current_np, dtype=jnp.float32)

    reward_mean_before_update = reward_normalizer.mean
    reward_std_before_update = reward_normalizer.std
    reward_count_before_update = reward_normalizer.count

    def q_loss_fn(q_params: Any, intrinsic_rewards: jax.Array):
        q_values = q_network.apply({"params": q_params}, observations)
        q_selected = jnp.take_along_axis(q_values, actions[:, None], axis=1).squeeze(-1)

        next_q_target = q_network.apply(
            {"params": agent_state.target_params},
            next_observations,
        )
        max_next_q = jnp.max(next_q_target, axis=1)
        bootstrap_term = (
            jnp.asarray(args.gamma, dtype=jnp.float32)
            * max_next_q
            * (1.0 - dones)
        )
        beta_intrinsic = jnp.asarray(beta_scale, dtype=jnp.float32) * intrinsic_rewards
        target = rewards_ext + beta_intrinsic + bootstrap_term
        td_error = q_selected - jax.lax.stop_gradient(target)
        losses = td_loss(
            td_error,
            loss_type=args.loss_type,
            huber_delta=args.huber_delta,
        )
        ratio = beta_intrinsic / (jnp.abs(rewards_ext + bootstrap_term) + 1e-8)

        aux = {
            "r_ext": rewards_ext,
            "r_int": intrinsic_rewards,
            "beta_r_int": beta_intrinsic,
            "bootstrap_term": bootstrap_term,
            "target": target,
            "q_selected": q_selected,
            "td_error": td_error,
            "ratio": ratio,
        }
        return jnp.mean(losses), aux

    (q_loss_full, aux_full), grads_full = jax.value_and_grad(
        q_loss_fn, has_aux=True
    )(agent_state.q_params, intrinsic_current)
    (q_loss_no_intrinsic, _), grads_no_intrinsic = jax.value_and_grad(
        q_loss_fn, has_aux=True
    )(agent_state.q_params, jnp.zeros_like(intrinsic_current))

    grads_full_vec, _ = ravel_pytree(grads_full)
    grads_no_intrinsic_vec, _ = ravel_pytree(grads_no_intrinsic)
    grads_intrinsic_vec = grads_full_vec - grads_no_intrinsic_vec

    gradient_record = {
        "env_step": int(env_step),
        "update_index": int(agent_state.update_index + 1),
        "g_full_norm": float(jnp.linalg.norm(grads_full_vec)),
        "g_no_intrinsic_norm": float(jnp.linalg.norm(grads_no_intrinsic_vec)),
        "g_intrinsic_only_norm": float(jnp.linalg.norm(grads_intrinsic_vec)),
        "cosine_full_vs_no_intrinsic": safe_cosine(
            grads_full_vec, grads_no_intrinsic_vec
        ),
    }

    clipped_grads, pre_clip_norm, post_clip_norm, was_clipped = clip_gradients(
        grads_full, args.grad_clip_norm
    )
    q_updates, new_q_opt_state = q_optimizer.update(
        clipped_grads,
        agent_state.q_opt_state,
        agent_state.q_params,
    )
    new_q_params = optax.apply_updates(agent_state.q_params, q_updates)
    parameter_update_norm = float(optax.global_norm(q_updates))

    gradient_record["pre_clip_grad_norm"] = pre_clip_norm
    gradient_record["post_clip_grad_norm"] = post_clip_norm
    gradient_record["was_clipped"] = was_clipped

    optimizer_record = {
        "env_step": int(env_step),
        "update_index": int(agent_state.update_index + 1),
        "optimizer": args.optimizer,
        "grad_norm_for_step": post_clip_norm,
        "parameter_update_norm": parameter_update_norm,
        "effective_step_size": parameter_update_norm / (post_clip_norm + 1e-8),
        "pre_clip_grad_norm": pre_clip_norm,
        "post_clip_grad_norm": post_clip_norm,
        "was_clipped": was_clipped,
    }

    layer_pre_clip = group_norms(grads_full)
    layer_post_clip = group_norms(clipped_grads)
    layer_update = group_norms(q_updates)
    layer_names = sorted(
        set(layer_pre_clip) | set(layer_post_clip) | set(layer_update)
    )
    layer_records = []
    for layer_name in layer_names:
        post_norm = layer_post_clip.get(layer_name, 0.0)
        update_norm = layer_update.get(layer_name, 0.0)
        layer_records.append(
            {
                "env_step": int(env_step),
                "update_index": int(agent_state.update_index + 1),
                "network": "q",
                "layer": layer_name,
                "pre_clip_grad_norm": layer_pre_clip.get(layer_name, 0.0),
                "post_clip_grad_norm": post_norm,
                "update_norm": update_norm,
                "effective_step_size": update_norm / (post_norm + 1e-8),
            }
        )

    def rnd_loss_fn(predictor_params: Any):
        predictor_features = rnd_predictor_network.apply(
            {"params": predictor_params}, next_observations
        )
        target_features = rnd_target_network.apply(
            {"params": agent_state.rnd_target_params}, next_observations
        )
        raw_error = jnp.mean(
            jnp.square(predictor_features - jax.lax.stop_gradient(target_features)),
            axis=-1,
        )
        return jnp.mean(raw_error), raw_error

    (rnd_loss, _), rnd_grads = jax.value_and_grad(rnd_loss_fn, has_aux=True)(
        agent_state.rnd_predictor_params
    )
    clipped_rnd_grads, _, _, _ = clip_gradients(rnd_grads, args.grad_clip_norm)
    rnd_updates, new_rnd_opt_state = rnd_optimizer.update(
        clipped_rnd_grads,
        agent_state.rnd_opt_state,
        agent_state.rnd_predictor_params,
    )
    new_rnd_predictor_params = optax.apply_updates(
        agent_state.rnd_predictor_params, rnd_updates
    )

    reward_normalizer.update(raw_intrinsic_np)

    new_update_index = agent_state.update_index + 1
    if new_update_index % max(1, args.target_update_frequency) == 0:
        new_target_params = new_q_params
    else:
        new_target_params = agent_state.target_params

    updated_state = AgentState(
        q_params=new_q_params,
        target_params=new_target_params,
        q_opt_state=new_q_opt_state,
        rnd_predictor_params=new_rnd_predictor_params,
        rnd_target_params=agent_state.rnd_target_params,
        rnd_opt_state=new_rnd_opt_state,
        update_index=new_update_index,
    )

    ages = env_step - batch.insertion_steps
    replay_drift = np.abs(
        intrinsic_current_np.astype(np.float32) - batch.stored_intrinsic_rewards
    )
    replay_rows = []
    for sample_index in range(batch.indices.shape[0]):
        replay_rows.append(
            {
                "env_step": int(env_step),
                "update_index": int(new_update_index),
                "sample_index": int(sample_index),
                "buffer_index": int(batch.indices[sample_index]),
                "state_id": int(batch.state_ids[sample_index]),
                "next_state_id": int(batch.next_state_ids[sample_index]),
                "age": int(ages[sample_index]),
                "stored_intrinsic_reward": float(
                    batch.stored_intrinsic_rewards[sample_index]
                ),
                "current_intrinsic_reward": float(intrinsic_current_np[sample_index]),
                "drift": float(replay_drift[sample_index]),
            }
        )

    r_ext_mean, r_ext_std = summarize(np.asarray(aux_full["r_ext"]))
    r_int_mean, r_int_std = summarize(intrinsic_current_np)
    r_int_stored_mean, r_int_stored_std = summarize(batch.stored_intrinsic_rewards)
    beta_r_int_mean, beta_r_int_std = summarize(np.asarray(aux_full["beta_r_int"]))
    bootstrap_mean, bootstrap_std = summarize(np.asarray(aux_full["bootstrap_term"]))
    target_mean, target_std = summarize(np.asarray(aux_full["target"]))
    q_selected_mean, q_selected_std = summarize(np.asarray(aux_full["q_selected"]))
    td_error_mean, td_error_std = summarize(np.asarray(aux_full["td_error"]))
    ratio_mean, ratio_std = summarize(np.asarray(aux_full["ratio"]))
    age_mean, age_std = summarize(ages.astype(np.float32))
    drift_mean, drift_std = summarize(replay_drift.astype(np.float32))

    batch_summary = {
        "env_step": int(env_step),
        "update_index": int(new_update_index),
        "epsilon": float(epsilon),
        "q_loss_full": float(q_loss_full),
        "q_loss_no_intrinsic": float(q_loss_no_intrinsic),
        "rnd_loss": float(rnd_loss),
        "reward_normalizer_mean": float(reward_mean_before_update),
        "reward_normalizer_std": float(reward_std_before_update),
        "reward_normalizer_count": float(reward_count_before_update),
        "r_ext_mean": r_ext_mean,
        "r_ext_std": r_ext_std,
        "r_int_current_mean": r_int_mean,
        "r_int_current_std": r_int_std,
        "r_int_stored_mean": r_int_stored_mean,
        "r_int_stored_std": r_int_stored_std,
        "beta_r_int_mean": beta_r_int_mean,
        "beta_r_int_std": beta_r_int_std,
        "bootstrap_term_mean": bootstrap_mean,
        "bootstrap_term_std": bootstrap_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "q_selected_mean": q_selected_mean,
        "q_selected_std": q_selected_std,
        "td_error_mean": td_error_mean,
        "td_error_std": td_error_std,
        "ratio_mean": ratio_mean,
        "ratio_std": ratio_std,
        "age_mean": age_mean,
        "age_std": age_std,
        "drift_mean": drift_mean,
        "drift_std": drift_std,
    }

    batch_detail = {
        "env_step": int(env_step),
        "update_index": int(new_update_index),
        "epsilon": float(epsilon),
        "reward_normalizer_mean": float(reward_mean_before_update),
        "reward_normalizer_std": float(reward_std_before_update),
        "r_ext": np.asarray(aux_full["r_ext"], dtype=np.float32),
        "r_int_current": intrinsic_current_np.astype(np.float32),
        "r_int_stored": batch.stored_intrinsic_rewards.astype(np.float32),
        "beta_r_int": np.asarray(aux_full["beta_r_int"], dtype=np.float32),
        "bootstrap_term": np.asarray(aux_full["bootstrap_term"], dtype=np.float32),
        "target": np.asarray(aux_full["target"], dtype=np.float32),
        "q_selected": np.asarray(aux_full["q_selected"], dtype=np.float32),
        "td_error": np.asarray(aux_full["td_error"], dtype=np.float32),
        "ratio": np.asarray(aux_full["ratio"], dtype=np.float32),
        "state_ids": batch.state_ids.astype(np.int32),
        "next_state_ids": batch.next_state_ids.astype(np.int32),
        "ages": ages.astype(np.int64),
        "replay_drift": replay_drift.astype(np.float32),
        "batch_indices": batch.indices.astype(np.int64),
        "summary": batch_summary,
    }

    metrics = {
        "batch_summary": batch_summary,
        "batch_detail": batch_detail,
        "gradient_record": gradient_record,
        "optimizer_record": optimizer_record,
        "optimizer_layer_records": layer_records,
        "replay_rows": replay_rows,
    }
    return updated_state, metrics


def run_single_seed(args: Args, seed: int, seed_dir: Path) -> None:
    env = NavixWrapper(args.env_id, max_steps=args.max_episode_steps)
    master_key = jax.random.PRNGKey(seed)
    master_key, reset_key = jax.random.split(master_key)
    env_state, obs = env.reset(reset_key)
    obs_np = np.asarray(obs, dtype=np.uint8)
    grid_size = int(obs_np.shape[0])

    all_state_features, state_metadata = build_state_feature_catalog(grid_size)
    num_states = int(all_state_features.shape[0])
    observation_dim = int(all_state_features.shape[1])
    num_actions = int(env.env.action_space.n)

    logger = StructuredLogger(seed_dir, state_metadata=state_metadata)
    write_json(
        seed_dir / "config.json",
        serialize_args(args, seed=seed, output_dir=seed_dir),
    )

    q_network = MLP(
        hidden_dims=args.hidden_dims,
        output_dim=num_actions,
        activation=args.activation,
    )
    rnd_predictor_network = MLP(
        hidden_dims=args.rnd_hidden_dims,
        output_dim=args.rnd_output_dim,
        activation=args.activation,
    )
    rnd_target_network = MLP(
        hidden_dims=args.rnd_hidden_dims,
        output_dim=args.rnd_output_dim,
        activation=args.activation,
    )

    dummy_input = jnp.zeros((1, observation_dim), dtype=jnp.float32)
    master_key, q_key, rnd_target_key, rnd_predictor_key = jax.random.split(
        master_key, 4
    )
    q_params = q_network.init(q_key, dummy_input)["params"]
    target_params = q_params
    rnd_target_params = rnd_target_network.init(rnd_target_key, dummy_input)["params"]
    rnd_predictor_params = rnd_predictor_network.init(
        rnd_predictor_key, dummy_input
    )["params"]

    q_optimizer = build_optimizer(args.optimizer, args.learning_rate)
    rnd_optimizer = build_optimizer(args.optimizer, args.rnd_learning_rate)

    agent_state = AgentState(
        q_params=q_params,
        target_params=target_params,
        q_opt_state=q_optimizer.init(q_params),
        rnd_predictor_params=rnd_predictor_params,
        rnd_target_params=rnd_target_params,
        rnd_opt_state=rnd_optimizer.init(rnd_predictor_params),
    )

    replay_buffer = ReplayBuffer(
        capacity=args.replay_capacity,
        observation_dim=observation_dim,
        seed=seed,
    )
    reward_normalizer = RewardNormalizer(
        epsilon=args.intrinsic_reward_epsilon,
        clip=args.intrinsic_reward_clip,
    )

    state_visit_counts = np.zeros((num_states,), dtype=np.int64)
    state_cumulative_intrinsic = np.zeros((num_states,), dtype=np.float32)

    current_features, current_state_id = observation_to_features_and_state_id(
        obs_np, grid_size
    )
    _, current_intrinsic = compute_intrinsic_rewards(
        predictor_network=rnd_predictor_network,
        predictor_params=agent_state.rnd_predictor_params,
        target_network=rnd_target_network,
        target_params=agent_state.rnd_target_params,
        observations=current_features[None, :],
        reward_normalizer=reward_normalizer,
    )
    state_visit_counts[current_state_id] += 1
    state_cumulative_intrinsic[current_state_id] += float(current_intrinsic[0])
    logger.log_state_visit(
        state_id=current_state_id,
        step=0,
        intrinsic_reward=float(current_intrinsic[0]),
    )
    make_snapshot_and_log(
        logger=logger,
        step=0,
        q_network=q_network,
        q_params=agent_state.q_params,
        all_state_features=all_state_features,
        state_visit_counts=state_visit_counts,
        state_cumulative_intrinsic=state_cumulative_intrinsic,
    )

    np_rng = np.random.default_rng(seed)
    beta_scale = float(args.beta if args.use_intrinsic_reward else 0.0)

    episode_index = 0
    episode_length = 0
    episode_return_extrinsic = 0.0
    episode_return_intrinsic = 0.0

    for env_step in range(1, args.total_steps + 1):
        epsilon = linear_schedule(
            env_step - 1,
            start=args.epsilon_start,
            end=args.epsilon_end,
            decay_steps=args.epsilon_decay_steps,
        )
        q_values = q_values_for_states(
            q_network, agent_state.q_params, current_features[None, :]
        )[0]
        if np_rng.random() < epsilon:
            action = int(np_rng.integers(num_actions))
        else:
            action = int(np.argmax(q_values))

        next_env_state, next_obs, reward_ext, terminated, truncated, _ = env.step(
            env_state, action
        )
        done = bool(terminated or truncated)
        next_obs_np = np.asarray(next_obs, dtype=np.uint8)
        next_features, next_state_id = observation_to_features_and_state_id(
            next_obs_np, grid_size
        )

        _, stored_intrinsic = compute_intrinsic_rewards(
            predictor_network=rnd_predictor_network,
            predictor_params=agent_state.rnd_predictor_params,
            target_network=rnd_target_network,
            target_params=agent_state.rnd_target_params,
            observations=next_features[None, :],
            reward_normalizer=reward_normalizer,
        )
        stored_intrinsic_value = float(stored_intrinsic[0])

        replay_buffer.add(
            observation=current_features,
            action=action,
            extrinsic_reward=float(reward_ext),
            discount=args.gamma,
            next_observation=next_features,
            done=done,
            insertion_step=env_step,
            stored_intrinsic_reward=stored_intrinsic_value,
            state_id=current_state_id,
            next_state_id=next_state_id,
        )

        state_visit_counts[next_state_id] += 1
        state_cumulative_intrinsic[next_state_id] += stored_intrinsic_value
        logger.log_state_visit(
            state_id=next_state_id,
            step=env_step,
            intrinsic_reward=stored_intrinsic_value,
        )

        episode_length += 1
        episode_return_extrinsic += float(reward_ext)
        episode_return_intrinsic += stored_intrinsic_value

        if (
            replay_buffer.ready(args.batch_size)
            and env_step >= args.warmup_steps
            and env_step % args.train_frequency == 0
        ):
            for _ in range(args.gradient_updates_per_train_step):
                batch = replay_buffer.sample(args.batch_size)
                agent_state, metrics = update_agent(
                    agent_state=agent_state,
                    batch=batch,
                    q_network=q_network,
                    rnd_predictor_network=rnd_predictor_network,
                    rnd_target_network=rnd_target_network,
                    q_optimizer=q_optimizer,
                    rnd_optimizer=rnd_optimizer,
                    reward_normalizer=reward_normalizer,
                    args=args,
                    env_step=env_step,
                    epsilon=epsilon,
                )
                logger.log_batch_detail(metrics["batch_detail"])
                logger.log_batch_summary(metrics["batch_summary"])
                logger.log_gradient_stats(metrics["gradient_record"])
                logger.log_optimizer_stats(metrics["optimizer_record"])
                logger.log_optimizer_layer_stats(metrics["optimizer_layer_records"])
                logger.log_replay_rows(metrics["replay_rows"])

        env_state = next_env_state
        current_features = next_features
        current_state_id = next_state_id

        should_snapshot = (
            args.state_eval_interval > 0 and env_step % args.state_eval_interval == 0
        )
        if should_snapshot or env_step == args.total_steps:
            make_snapshot_and_log(
                logger=logger,
                step=env_step,
                q_network=q_network,
                q_params=agent_state.q_params,
                all_state_features=all_state_features,
                state_visit_counts=state_visit_counts,
                state_cumulative_intrinsic=state_cumulative_intrinsic,
            )

        if done:
            logger.log_episode(
                {
                    "env_step": int(env_step),
                    "episode_index": int(episode_index),
                    "episode_length": int(episode_length),
                    "episode_return_extrinsic": float(episode_return_extrinsic),
                    "episode_return_intrinsic": float(episode_return_intrinsic),
                    "episode_return_total": float(
                        episode_return_extrinsic
                        + beta_scale * episode_return_intrinsic
                    ),
                    "done": True,
                }
            )

            episode_index += 1
            episode_length = 0
            episode_return_extrinsic = 0.0
            episode_return_intrinsic = 0.0

            master_key, reset_key = jax.random.split(master_key)
            env_state, obs = env.reset(reset_key)
            obs_np = np.asarray(obs, dtype=np.uint8)
            current_features, current_state_id = observation_to_features_and_state_id(
                obs_np, grid_size
            )
            _, reset_intrinsic = compute_intrinsic_rewards(
                predictor_network=rnd_predictor_network,
                predictor_params=agent_state.rnd_predictor_params,
                target_network=rnd_target_network,
                target_params=agent_state.rnd_target_params,
                observations=current_features[None, :],
                reward_normalizer=reward_normalizer,
            )
            state_visit_counts[current_state_id] += 1
            state_cumulative_intrinsic[current_state_id] += float(reset_intrinsic[0])
            logger.log_state_visit(
                state_id=current_state_id,
                step=env_step,
                intrinsic_reward=float(reset_intrinsic[0]),
            )

    visited_states = int(np.sum(state_visit_counts > 0))
    run_summary = {
        "seed": seed,
        "num_states": num_states,
        "num_actions": num_actions,
        "total_steps": args.total_steps,
        "total_updates": int(agent_state.update_index),
        "visited_states": visited_states,
        "final_reward_normalizer_mean": float(reward_normalizer.mean),
        "final_reward_normalizer_std": float(reward_normalizer.std),
        "final_reward_normalizer_count": float(reward_normalizer.count),
    }
    logger.finalize(run_summary=run_summary)
    generate_all_plots(seed_dir, top_k_states=args.plot_top_k_states)


def main(args: Args) -> None:
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        run_dir / "run_config.json",
        {
            "run_name": run_name,
            "args": asdict(args),
        },
    )

    for seed in args.seeds:
        seed_dir = run_dir / f"seed_{int(seed):03d}"
        run_single_seed(args=args, seed=int(seed), seed_dir=seed_dir)


if __name__ == "__main__":
    main(tyro.cli(Args))
