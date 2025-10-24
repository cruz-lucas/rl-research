"""Bounding-Box Inference (BBI) agent implemented with pure JAX."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from rl_research.agents.base import AgentState, TabularAgent, UpdateResult
from rl_research.policies import ActionSelectionPolicy, EpsilonGreedyPolicy


@struct.dataclass
class BBIAgentState(AgentState):
    """Mutable state tracked by the BBI agent."""

    visit_counts: jax.Array


def _softmin(values: jax.Array, tau: float) -> jax.Array:
    """Compute a softmin distribution over `values` using temperature `tau`."""

    x = jnp.asarray(values, dtype=jnp.float32)
    tau_arr = jnp.asarray(tau, dtype=jnp.float32)
    scaled = jnp.exp(-x / jnp.maximum(tau_arr, 1e-6))
    total = jnp.sum(scaled)
    total = jnp.where(total > 0.0, total, 1.0)
    return scaled / total


def _load_bounding_box_model(model_filename: str, fully_observed: bool) -> np.ndarray:
    """Load the pre-computed bounding box model from disk."""

    resource = resources.files("rl_research.environment.data") / model_filename
    try:
        with resources.as_file(resource) as resolved:
            data = np.load(Path(resolved), allow_pickle=True)
    except FileNotFoundError as exc:  # pragma: no cover - defensive.
        msg = (
            f"Could not locate bounding box model '{model_filename}'. "
            f"Tried path '{resource}'."
        )
        raise FileNotFoundError(msg) from exc

    key = "fully_obs" if fully_observed else "partially_obs"
    if key not in data:
        raise KeyError(f"Model '{model_filename}' missing key '{key}'.")
    return np.asarray(data[key], dtype=np.float32)


@dataclass(slots=True)
class BBIAgentConfig:
    """Configuration for the BBI agent."""

    discount: float = 0.95
    step_size: float = 1.0
    epsilon: float = 0.1
    tau: float = 1.0
    horizon: int = 5
    fully_observed: bool = False
    model_filename: str = "boundingbox_model.npz"
    grid_length: int = 21
    num_indicators: int = 2
    num_status_levels: int = 3


class BBIAgent(TabularAgent[BBIAgentState]):
    """Bounding-Box Inference agent with JAX-friendly updates."""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        *,
        config: BBIAgentConfig | None = None,
        seed: int | None = None,
        policy: ActionSelectionPolicy | None = None,
    ) -> None:
        self.config = config or BBIAgentConfig()
        self.step_size = float(self.config.step_size)
        self.tau = float(self.config.tau)
        self.horizon = int(self.config.horizon)
        self.epsilon = float(self.config.epsilon)
        self.fully_observed = bool(self.config.fully_observed)

        self._feature_shape = self._derive_obs_shape(num_states)
        self._feature_dim = len(self._feature_shape)
        self._obs_features_np = self._compute_observation_features_np()
        self._obs_features = jnp.asarray(self._obs_features_np, dtype=jnp.int32)

        model = _load_bounding_box_model(
            self.config.model_filename, self.fully_observed
        )
        if model.shape[0] != num_states or model.shape[1] != num_actions:
            raise ValueError(
                "Bounding box model shape mismatch: "
                f"expected (num_states={num_states}, num_actions={num_actions}), "
                f"got {model.shape[:2]}"
            )

        (
            self._lower_next_obs,
            self._lower_reward,
            self._expected_next_obs,
            self._expected_reward,
            self._upper_next_obs,
            self._upper_reward,
        ) = (model[..., idx] for idx in range(6))

        lower_next_obs = self._lower_next_obs.astype(np.int32)
        upper_next_obs = self._upper_next_obs.astype(np.int32)

        self._lower_next_obs = jnp.asarray(lower_next_obs, dtype=jnp.int32)
        self._upper_next_obs = jnp.asarray(upper_next_obs, dtype=jnp.int32)
        self._expected_next_obs = jnp.asarray(
            self._expected_next_obs.astype(np.int32), dtype=jnp.int32
        )

        self._lower_reward = jnp.asarray(self._lower_reward, dtype=jnp.float32)
        self._upper_reward = jnp.asarray(self._upper_reward, dtype=jnp.float32)
        self._expected_reward = jnp.asarray(
            self._expected_reward, dtype=jnp.float32
        )

        self._lower_next_features = jnp.asarray(
            self._obs_features_np[lower_next_obs], dtype=jnp.int32
        )
        self._upper_next_features = jnp.asarray(
            self._obs_features_np[upper_next_obs], dtype=jnp.int32
        )
        self._lower_next_features_f = self._lower_next_features.astype(jnp.float32)
        self._upper_next_features_f = self._upper_next_features.astype(jnp.float32)

        super().__init__(
            num_states=num_states,
            num_actions=num_actions,
            discount=self.config.discount,
            seed=seed,
            policy=policy,
        )

    # ------------------------------------------------------------------ #
    def _derive_obs_shape(self, num_states: int) -> Tuple[int, ...]:
        """Infer the observation tensor shape for Go-Right."""

        cfg = self.config
        indicators = (2,) * int(cfg.num_indicators)
        if self.fully_observed:
            shape = (
                int(cfg.grid_length),
                int(cfg.num_status_levels),
                int(cfg.num_status_levels),
            ) + indicators
        else:
            shape = (
                int(cfg.grid_length),
                int(cfg.num_status_levels),
            ) + indicators

        prod = int(np.prod(shape))
        if prod != num_states:
            raise ValueError(
                "Derived observation shape does not match `num_states`. "
                f"Computed shape {shape} -> product {prod}, expected {num_states}."
            )
        return shape

    def _compute_observation_features_np(self) -> np.ndarray:
        """Map flattened observation indices to their feature tuples."""

        num_states = int(np.prod(self._feature_shape))
        indices = np.arange(num_states, dtype=np.int32)
        coords = np.stack(
            np.unravel_index(indices, self._feature_shape),
            axis=-1,
        )
        return coords.astype(np.int32)

    # ------------------------------------------------------------------ #
    def _default_policy(self) -> ActionSelectionPolicy:
        return EpsilonGreedyPolicy(self.epsilon)

    def _initial_state(self, key: jax.Array) -> BBIAgentState:
        q_values = jnp.zeros(
            (self.num_states, self.num_actions),
            dtype=jnp.float32,
        )
        visit_counts = jnp.zeros_like(q_values)
        return BBIAgentState(q_values=q_values, visit_counts=visit_counts, rng=key)

    def train(self) -> None:
        if isinstance(self._policy, EpsilonGreedyPolicy):
            self._policy.epsilon = self.epsilon

    def eval(self) -> None:
        if isinstance(self._policy, EpsilonGreedyPolicy):
            self._policy.epsilon = 0.0

    # ------------------------------------------------------------------ #
    def update(
        self,
        agent_state: BBIAgentState,
        obs: jax.Array,
        action: jax.Array,
        reward: jax.Array,
        next_obs: jax.Array,
        *,
        terminated: jax.Array | bool = False,
    ) -> Tuple[BBIAgentState, UpdateResult]:
        obs_idx = jnp.asarray(obs, dtype=jnp.int32)
        action_idx = jnp.asarray(action, dtype=jnp.int32)
        next_obs_idx = jnp.asarray(next_obs, dtype=jnp.int32)
        reward_val = jnp.asarray(reward, dtype=jnp.float32)
        terminated_mask = jnp.asarray(terminated, dtype=jnp.bool_)

        q_values = agent_state.q_values
        q_sa = q_values[obs_idx, action_idx]
        next_value = jnp.max(q_values[next_obs_idx])

        gamma = jnp.asarray(self.config.discount, dtype=jnp.float32)
        base_target = reward_val + gamma * jnp.where(
            terminated_mask, 0.0, next_value
        )

        rollout_targets, rollout_uncertainties = jax.lax.cond(
            terminated_mask,
            lambda _: (
                jnp.zeros((self.horizon,), dtype=jnp.float32),
                jnp.full((self.horizon,), jnp.inf, dtype=jnp.float32),
            ),
            lambda _: self._bbi_rollout_targets(
                q_values=q_values,
                initial_obs=next_obs_idx,
                initial_reward=reward_val,
                discount=gamma,
            ),
            operand=None,
        )

        all_targets = jnp.concatenate(
            [base_target[None], rollout_targets], axis=0
        )
        all_uncertainties = jnp.concatenate(
            [jnp.zeros((1,), dtype=jnp.float32), rollout_uncertainties],
            axis=0,
        )

        weights = _softmin(all_uncertainties, self.tau)
        mean_target = jnp.sum(weights * all_targets)
        td_error = mean_target - q_sa

        new_q_values = q_values.at[obs_idx, action_idx].add(
            self.step_size * td_error
        )
        new_visit_counts = agent_state.visit_counts.at[obs_idx, action_idx].add(1.0)

        new_state = agent_state.replace(
            q_values=new_q_values,
            visit_counts=new_visit_counts,
        )
        return new_state, UpdateResult()

    # ------------------------------------------------------------------ #
    def _bbi_rollout_targets(
        self,
        q_values: jax.Array,
        initial_obs: jax.Array,
        initial_reward: jax.Array,
        discount: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        box = self._obs_features[initial_obs]
        action_mask = self._greedy_actions_mask(q_values, initial_obs)
        fallback_action = self._greedy_action_index(q_values, initial_obs)
        fallback_mask = jax.nn.one_hot(
            fallback_action, self.num_actions, dtype=jnp.float32
        ).astype(bool)
        action_mask = jnp.where(jnp.any(action_mask), action_mask, fallback_mask)

        carry = (
            box,  # box lower
            box,  # box upper
            action_mask,
            initial_reward,  # lower_return
            initial_reward,  # upper_return
            initial_reward,  # expected_return
            discount,  # current_discount
            discount,  # expected_discount
            initial_obs,  # expected_obs
            jnp.asarray(True, dtype=jnp.bool_),  # active
        )

        def body_fn(carry, _):
            (
                box_lower,
                box_upper,
                action_mask,
                lower_return,
                upper_return,
                expected_return,
                current_discount,
                expected_discount,
                expected_obs,
                active,
            ) = carry

            def do_step(args):
                (
                    box_lower,
                    box_upper,
                    action_mask,
                    lower_return,
                    upper_return,
                    expected_return,
                    current_discount,
                    expected_discount,
                    expected_obs,
                    _,
                ) = args

                expected_action = self._greedy_action_index(q_values, expected_obs)
                prev_lower_return = lower_return
                prev_upper_return = upper_return
                prev_expected_return = expected_return
                prev_current_discount = current_discount
                prev_expected_discount = expected_discount
                prev_expected_obs = expected_obs
                (
                    next_box_lower,
                    next_box_upper,
                    lower_reward,
                    upper_reward,
                    expected_reward,
                    lower_value,
                    upper_value,
                    expected_value,
                    expected_next_obs,
                    next_action_mask,
                    valid,
                ) = self._rollout_step(
                    q_values=q_values,
                    box_lower=box_lower,
                    box_upper=box_upper,
                    action_mask=action_mask,
                    expected_obs=expected_obs,
                    expected_action=expected_action,
                )

                lower_return_new = lower_return + current_discount * lower_reward
                upper_return_new = upper_return + current_discount * upper_reward
                current_discount_new = current_discount * self.config.discount

                expected_return_new = (
                    expected_return + expected_discount * expected_reward
                )
                expected_discount_new = expected_discount * self.config.discount

                lower_target = lower_return_new + current_discount_new * lower_value
                upper_target = upper_return_new + current_discount_new * upper_value
                expected_target = expected_return_new + expected_discount_new * expected_value

                nominal_target = jnp.where(valid, expected_target, 0.0)
                uncertainty = jnp.where(
                    valid, jnp.maximum(0.0, upper_target - lower_target), jnp.inf
                )

                next_carry = (
                    jnp.where(valid, next_box_lower, box_lower),
                    jnp.where(valid, next_box_upper, box_upper),
                    jnp.where(valid, next_action_mask, action_mask),
                    jnp.where(valid, lower_return_new, prev_lower_return),
                    jnp.where(valid, upper_return_new, prev_upper_return),
                    jnp.where(valid, expected_return_new, prev_expected_return),
                    jnp.where(valid, current_discount_new, prev_current_discount),
                    jnp.where(valid, expected_discount_new, prev_expected_discount),
                    jnp.where(valid, expected_next_obs, prev_expected_obs),
                    jnp.logical_and(valid, jnp.any(next_action_mask)),
                )
                return next_carry, (nominal_target, uncertainty)

            def skip_step(args):
                return args, (jnp.asarray(0.0, dtype=jnp.float32), jnp.asarray(jnp.inf, dtype=jnp.float32))

            new_carry, outputs = jax.lax.cond(
                active, do_step, skip_step, operand=carry
            )
            return new_carry, outputs

        _, (targets, uncertainties) = jax.lax.scan(
            body_fn, carry, jnp.arange(self.horizon, dtype=jnp.int32)
        )
        return targets.astype(jnp.float32), uncertainties.astype(jnp.float32)

    # ------------------------------------------------------------------ #
    def _rollout_step(
        self,
        q_values: jax.Array,
        box_lower: jax.Array,
        box_upper: jax.Array,
        action_mask: jax.Array,
        expected_obs: jax.Array,
        expected_action: jax.Array,
    ) -> Tuple[
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
    ]:
        state_mask = self._states_in_box(box_lower, box_upper)
        expected_action_mask = jax.nn.one_hot(
            expected_action, self.num_actions, dtype=jnp.float32
        ).astype(bool)
        combined_action_mask = jnp.logical_or(action_mask, expected_action_mask)
        sa_mask = jnp.logical_and(state_mask[:, None], combined_action_mask[None, :])
        processed = jnp.any(sa_mask)

        lower_reward = jnp.min(
            jnp.where(sa_mask, self._lower_reward, jnp.asarray(jnp.inf, dtype=jnp.float32))
        )
        upper_reward = jnp.max(
            jnp.where(sa_mask, self._upper_reward, jnp.asarray(-jnp.inf, dtype=jnp.float32))
        )

        lower_reward = jnp.where(processed, lower_reward, 0.0)
        upper_reward = jnp.where(processed, upper_reward, 0.0)

        expected_next_obs = self._expected_next_obs[expected_obs, expected_action]
        expected_reward = self._expected_reward[expected_obs, expected_action]
        expected_reward = jnp.where(jnp.isfinite(expected_reward), expected_reward, 0.0)

        expected_value = jnp.max(q_values[expected_next_obs])
        expected_value = jnp.where(jnp.isfinite(expected_value), expected_value, 0.0)

        mask_expanded = sa_mask[..., None]
        lower_features = self._lower_next_features_f
        upper_features = self._upper_next_features_f

        min_lower = jnp.min(
            jnp.where(mask_expanded, lower_features, jnp.asarray(jnp.inf, dtype=jnp.float32)),
            axis=(0, 1),
        )
        max_upper = jnp.max(
            jnp.where(mask_expanded, upper_features, jnp.asarray(-jnp.inf, dtype=jnp.float32)),
            axis=(0, 1),
        )

        min_lower = jnp.where(processed, min_lower, jnp.zeros_like(min_lower))
        max_upper = jnp.where(processed, max_upper, jnp.zeros_like(max_upper))

        combined_lower = jnp.minimum(min_lower, max_upper).astype(jnp.int32)
        combined_upper = jnp.maximum(min_lower, max_upper).astype(jnp.int32)

        lower_value, upper_value, actions_found = self._value_bounds_and_actions(
            q_values, combined_lower, combined_upper
        )

        valid = jnp.logical_and(
            processed, jnp.logical_and(jnp.isfinite(lower_value), jnp.isfinite(upper_value))
        )

        next_action_mask = jnp.logical_or(
            actions_found, self._greedy_actions_mask(q_values, expected_next_obs)
        )

        return (
            combined_lower,
            combined_upper,
            lower_reward,
            upper_reward,
            expected_reward,
            lower_value,
            upper_value,
            expected_value,
            expected_next_obs,
            next_action_mask,
            valid,
        )

    def _states_in_box(self, lower: jax.Array, upper: jax.Array) -> jax.Array:
        features = self._obs_features
        lower = jnp.asarray(lower, dtype=jnp.int32)
        upper = jnp.asarray(upper, dtype=jnp.int32)
        ge_lower = features >= lower
        le_upper = features <= upper
        return jnp.all(jnp.logical_and(ge_lower, le_upper), axis=1)

    def _value_bounds_and_actions(
        self,
        q_values: jax.Array,
        lower: jax.Array,
        upper: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        mask = self._states_in_box(lower, upper)
        has_state = jnp.any(mask)

        max_q = jnp.max(q_values, axis=1)
        greedy_mask = jnp.isclose(q_values, max_q[:, None])

        min_vals = jnp.min(
            jnp.where(mask, max_q, jnp.asarray(jnp.inf, dtype=jnp.float32))
        )
        max_vals = jnp.max(
            jnp.where(mask, max_q, jnp.asarray(-jnp.inf, dtype=jnp.float32))
        )

        lower_value = jnp.where(has_state, min_vals, 0.0)
        upper_value = jnp.where(has_state, max_vals, 0.0)

        action_mask = jnp.any(
            jnp.where(mask[:, None], greedy_mask, False), axis=0
        )
        action_mask = jnp.where(
            has_state,
            action_mask,
            jnp.zeros((self.num_actions,), dtype=bool),
        )

        return lower_value, upper_value, action_mask

    @staticmethod
    def _greedy_action_index(q_values: jax.Array, obs: jax.Array) -> jax.Array:
        return jnp.argmax(q_values[obs])

    def _greedy_actions_mask(
        self, q_values: jax.Array, obs: jax.Array
    ) -> jax.Array:
        row = q_values[obs]
        max_q = jnp.max(row)
        return jnp.isclose(row, max_q)
