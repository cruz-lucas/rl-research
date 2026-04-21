from collections.abc import Sequence
from typing import Tuple

import gin
import jax
import jax.numpy as jnp

from rl_research.agents.dqn_rnd import DQNRNDAgent, DQNRNDState
from rl_research.policies import _select_greedy


@gin.configurable
class DQNRNDUCBAgent(DQNRNDAgent):
    """DQN + RND with action-output conditioning and bonus-greedy exploration."""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        hidden_units: int = 64,
        hidden_dims: Sequence[int] | None = None,
        activation: str = "relu",
        normalization: str = "last",
        learning_rate: float = 1e-3,
        discount: float = 0.99,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        eps_decay_steps: int = 100_000,
        target_update_freq: int = 1000,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        optimizer_beta1: float = 0.9,
        optimizer_beta2: float = 0.999,
        optimizer_epsilon: float = 1e-8,
        optimizer_weight_decay: float = 0.0,
        optimizer_momentum: float = 0.0,
        optimizer_decay: float = 0.95,
        optimizer_centered: bool = False,
        loss_type: str = "mse",
        huber_delta: float = 1.0,
        double_q: bool = False,
        intrinsic_reward_scale: float = 1.0,
        intrinsic_stats_decay: float = 0.99,
        intrinsic_reward_epsilon: float = 1e-4,
        intrinsic_reward_clip: float | None = 10.0,
        rnd_hidden_units: int | None = None,
        rnd_hidden_dims: Sequence[int] | None = None,
        rnd_activation: str | None = None,
        rnd_normalization: str | None = None,
        rnd_output_dim: int = 64,
        rnd_optimizer: str | None = None,
        rnd_learning_rate: float | None = None,
        rnd_include_action: bool | None = None,
        rnd_action_conditioning: str = "output",
        rnd_update_period: int = 1,
        visit_count_table_size: int = 65536,
        decision_bonus_scale: float | None = None,
        decision_bonus_threshold: float | None = None,
        decision_visit_count_threshold: int | None = None,
        decision_optimistic_value: float | None = None,
        bootstrap_with_rnd_bonus: bool = False,
        bootstrap_bonus_scale: float | None = None,
        bootstrap_bonus_threshold: float | None = None,
        bootstrap_visit_count_threshold: int | None = None,
        bootstrap_optimistic_value: float | None = None,
        use_decision_bonus_in_eval: bool = False,
        normalize_observations: bool = False,
        obs_normalization_epsilon: float = 1e-8,
        obs_normalization_clip: float | None = 5.0,
        debug: bool = False,
        debug_log_dir: str = "tmp/debug_logs",
        debug_log_to_mlflow: bool = True,
        debug_compact_observations: bool = True,
        seed: int = 0,
    ):
        super().__init__(
            num_states=num_states,
            num_actions=num_actions,
            hidden_units=hidden_units,
            hidden_dims=hidden_dims,
            activation=activation,
            normalization=normalization,
            learning_rate=learning_rate,
            discount=discount,
            eps_start=eps_start,
            eps_end=eps_end,
            eps_decay_steps=eps_decay_steps,
            target_update_freq=target_update_freq,
            max_grad_norm=max_grad_norm,
            optimizer=optimizer,
            optimizer_beta1=optimizer_beta1,
            optimizer_beta2=optimizer_beta2,
            optimizer_epsilon=optimizer_epsilon,
            optimizer_weight_decay=optimizer_weight_decay,
            optimizer_momentum=optimizer_momentum,
            optimizer_decay=optimizer_decay,
            optimizer_centered=optimizer_centered,
            loss_type=loss_type,
            huber_delta=huber_delta,
            double_q=double_q,
            intrinsic_reward_scale=intrinsic_reward_scale,
            intrinsic_stats_decay=intrinsic_stats_decay,
            intrinsic_reward_epsilon=intrinsic_reward_epsilon,
            intrinsic_reward_clip=intrinsic_reward_clip,
            rnd_hidden_units=rnd_hidden_units,
            rnd_hidden_dims=rnd_hidden_dims,
            rnd_activation=rnd_activation,
            rnd_normalization=rnd_normalization,
            rnd_output_dim=rnd_output_dim,
            rnd_optimizer=rnd_optimizer,
            rnd_learning_rate=rnd_learning_rate,
            rnd_include_action=rnd_include_action,
            rnd_action_conditioning=rnd_action_conditioning,
            rnd_update_period=rnd_update_period,
            visit_count_table_size=visit_count_table_size,
            normalize_observations=normalize_observations,
            obs_normalization_epsilon=obs_normalization_epsilon,
            obs_normalization_clip=obs_normalization_clip,
            debug=debug,
            debug_log_dir=debug_log_dir,
            debug_log_to_mlflow=debug_log_to_mlflow,
            debug_compact_observations=debug_compact_observations,
            seed=seed,
        )
        if self.rnd_action_conditioning == "none":
            raise ValueError(
                "DQNRNDUCBAgent requires rnd_action_conditioning='output' "
                "or 'input' so the "
                "RND network emits one feature vector per action."
            )
        self.decision_bonus_scale = (
            self.intrinsic_reward_scale
            if decision_bonus_scale is None
            else float(decision_bonus_scale)
        )
        self.decision_bonus_threshold = (
            None
            if decision_bonus_threshold is None
            else float(decision_bonus_threshold)
        )
        self.decision_visit_count_threshold = (
            None
            if decision_visit_count_threshold is None
            else int(decision_visit_count_threshold)
        )
        self.decision_optimistic_value = (
            None
            if decision_optimistic_value is None
            else float(decision_optimistic_value)
        )
        self.bootstrap_with_rnd_bonus = bool(bootstrap_with_rnd_bonus)
        self.bootstrap_bonus_scale = (
            self.decision_bonus_scale
            if bootstrap_bonus_scale is None
            else float(bootstrap_bonus_scale)
        )
        self.bootstrap_bonus_threshold = (
            None
            if bootstrap_bonus_threshold is None
            else float(bootstrap_bonus_threshold)
        )
        self.bootstrap_visit_count_threshold = (
            None
            if bootstrap_visit_count_threshold is None
            else int(bootstrap_visit_count_threshold)
        )
        self.bootstrap_optimistic_value = (
            None
            if bootstrap_optimistic_value is None
            else float(bootstrap_optimistic_value)
        )
        self.use_decision_bonus_in_eval = bool(use_decision_bonus_in_eval)

    def _compute_decision_bonus(
        self,
        state: DQNRNDState,
        observation: jnp.ndarray,
    ) -> jax.Array:
        decision_bonus = super()._compute_decision_bonus(state, observation)
        if decision_bonus.ndim == 0 or decision_bonus.shape[-1] != self.num_actions:
            raise ValueError(
                "DQNRNDUCBAgent expected one RND bonus per action, but got "
                f"shape {decision_bonus.shape!r}."
            )
        return decision_bonus

    def _resolve_optimistic_values(
        self,
        values: jax.Array,
        optimistic_value: float | None,
    ) -> jax.Array:
        if optimistic_value is None:
            return jnp.max(values, axis=-1, keepdims=True) + jnp.asarray(
                1.0, dtype=values.dtype
            )
        return jnp.asarray(optimistic_value, dtype=values.dtype)

    def _apply_optimistic_mask(
        self,
        values: jax.Array,
        optimistic_mask: jax.Array,
        optimistic_value: float | None,
    ) -> jax.Array:
        optimistic_values = self._resolve_optimistic_values(values, optimistic_value)
        return jnp.where(optimistic_mask, optimistic_values, values)

    def _apply_bonus_threshold(
        self,
        values: jax.Array,
        bonus: jax.Array,
        threshold: float | None,
        optimistic_value: float | None,
    ) -> jax.Array:
        if threshold is None:
            return values

        optimistic_mask = bonus >= jnp.asarray(threshold, dtype=bonus.dtype)
        return self._apply_optimistic_mask(
            values,
            optimistic_mask,
            optimistic_value,
        )

    def _apply_visit_count_threshold(
        self,
        values: jax.Array,
        visit_counts: jax.Array,
        threshold: int | None,
        optimistic_value: float | None,
    ) -> jax.Array:
        if threshold is None:
            return values

        threshold_array = jnp.asarray(threshold, dtype=visit_counts.dtype)
        optimistic_mask = visit_counts < threshold_array
        return self._apply_optimistic_mask(
            values,
            optimistic_mask,
            optimistic_value,
        )

    def _compute_next_bootstrap_values(
        self,
        state: DQNRNDState,
        next_observation: jax.Array,
        next_online_q: jax.Array,
        next_target_q: jax.Array,
    ) -> jax.Array:
        if (
            not self.bootstrap_with_rnd_bonus
            and self.bootstrap_bonus_threshold is None
            and self.bootstrap_visit_count_threshold is None
        ):
            return super()._compute_next_bootstrap_values(
                state=state,
                next_observation=next_observation,
                next_online_q=next_online_q,
                next_target_q=next_target_q,
            )

        next_online_values = next_online_q
        next_target_values = next_target_q
        if self.bootstrap_with_rnd_bonus or self.bootstrap_bonus_threshold is not None:
            next_bonus = jax.lax.stop_gradient(
                self._compute_decision_bonus(state, next_observation)
            )
            if self.bootstrap_with_rnd_bonus:
                next_online_values = (
                    next_online_values + self.bootstrap_bonus_scale * next_bonus
                )
                next_target_values = (
                    next_target_values + self.bootstrap_bonus_scale * next_bonus
                )
            next_online_values = self._apply_bonus_threshold(
                next_online_values,
                next_bonus,
                self.bootstrap_bonus_threshold,
                self.bootstrap_optimistic_value,
            )
            next_target_values = self._apply_bonus_threshold(
                next_target_values,
                next_bonus,
                self.bootstrap_bonus_threshold,
                self.bootstrap_optimistic_value,
            )
        next_visit_counts = jax.lax.stop_gradient(
            self._get_state_action_visit_counts(state, next_observation)
        )
        next_online_values = self._apply_visit_count_threshold(
            next_online_values,
            next_visit_counts,
            self.bootstrap_visit_count_threshold,
            self.bootstrap_optimistic_value,
        )
        next_target_values = self._apply_visit_count_threshold(
            next_target_values,
            next_visit_counts,
            self.bootstrap_visit_count_threshold,
            self.bootstrap_optimistic_value,
        )

        if self.double_q:
            next_action = jnp.argmax(
                jax.lax.stop_gradient(next_online_values), axis=-1, keepdims=True
            )
            return jnp.take_along_axis(
                next_target_values,
                next_action,
                axis=-1,
            ).squeeze(-1)

        return jnp.max(next_target_values, axis=-1)

    def select_action(
        self,
        state: DQNRNDState,
        obs: jnp.ndarray,
        key: jax.Array,
        is_training: bool,
    ) -> Tuple[DQNRNDState, jnp.ndarray]:
        raw_obs = jnp.asarray(obs).reshape(-1)
        if is_training:
            state = self._maybe_update_obs_normalizer(state, raw_obs)

        obs = self._normalize_observation(state, raw_obs)
        q_vals = state.online_network(obs.reshape(-1))

        decision_bonus = None
        decision_values = None

        use_decision_optimism = is_training or self.use_decision_bonus_in_eval
        if use_decision_optimism:
            # decision_bonus = self._compute_decision_bonus(state, raw_obs)
            decision_visit_counts = self._get_state_action_visit_counts(state, raw_obs)
            decision_values = q_vals# + self.decision_bonus_scale * decision_bonus
            # decision_values = self._apply_bonus_threshold(
            #     decision_values,
            #     decision_bonus,
            #     self.decision_bonus_threshold,
            #     self.decision_optimistic_value,
            # )
            decision_values = self._apply_visit_count_threshold(
                decision_values,
                decision_visit_counts,
                self.decision_visit_count_threshold,
                self.decision_optimistic_value,
            )
            action = _select_greedy(decision_values, key)
        else:
            action = _select_greedy(q_vals, key)

        if is_training:
            self._log_decision_debug(
                global_step=jnp.asarray(state.step, dtype=jnp.int32),
                observation=raw_obs,
                q_values=q_vals,
                action=action,
                epsilon=jnp.asarray(jnp.nan, dtype=jnp.float32),
                decision_bonus=decision_bonus,
                decision_values=decision_values,
            )
            state = self._increment_state_action_visit_count(state, raw_obs, action)
            state = state.replace(step=state.step + 1)

        return state, action
