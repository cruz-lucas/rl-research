"""Replay-based R-max + DQN agent with RND-defined unknownness."""

from collections.abc import Sequence
from typing import Tuple

import distrax
import gin
import jax
import jax.numpy as jnp
from flax import nnx

from rl_research.agents._dqn_common import hard_update_network, temporal_difference_loss
from rl_research.agents.dqn_rnd import DQNRNDAgent, DQNRNDState
from rl_research.buffers import Transition


@gin.configurable
class DQNRmaxRND(DQNRNDAgent):
    """R-max style DQN agent that uses RND intrinsic reward as the unknownness signal."""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        hidden_units: int = 64,
        hidden_dims: Sequence[int] | None = None,
        activation: str = "relu",
        normalization: str = "last",
        learning_rate: float = 1e-1,
        discount: float = 0.99,
        r_max: float = 1.0,
        v_max: float = 10.0,
        use_vmax: bool = True,
        intrinsic_reward_threshold: float = 0.5,
        target_update_freq: int = 1000,
        max_grad_norm: float = 10.0,
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
        rnd_action_conditioning: str = "input",
        rnd_update_period: int = 1,
        normalize_observations: bool = False,
        obs_normalization_epsilon: float = 1e-8,
        obs_normalization_clip: float | None = 5.0,
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
            eps_start=0.0,
            eps_end=0.0,
            eps_decay_steps=1,
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
            intrinsic_reward_scale=0.0,
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
            visit_count_table_size=1,
            normalize_observations=normalize_observations,
            obs_normalization_epsilon=obs_normalization_epsilon,
            obs_normalization_clip=obs_normalization_clip,
            debug=False,
            seed=seed,
        )
        if self.rnd_action_conditioning == "none":
            raise ValueError(
                "DQNRmaxRND requires rnd_action_conditioning='output' or 'input' "
                "so unknownness can be evaluated per action."
            )
        if intrinsic_reward_threshold < 0.0:
            raise ValueError("intrinsic_reward_threshold must be non-negative.")

        self.r_max = float(r_max)
        self.v_max = float(v_max)
        self.use_vmax = bool(use_vmax)
        self.intrinsic_reward_threshold = float(intrinsic_reward_threshold)
        self.optimistic_value = (
            self.v_max if self.use_vmax else (self.r_max / (1.0 - self.discount))
        )

    def _action_intrinsic_reward(
        self,
        state: DQNRNDState,
        observation: jnp.ndarray,
    ) -> jax.Array:
        intrinsic_reward = self._compute_decision_bonus(state, observation)
        if intrinsic_reward.ndim == 0 or intrinsic_reward.shape[-1] != self.num_actions:
            raise ValueError(
                "DQNRmaxRND expected one intrinsic reward per action, but got "
                f"shape {intrinsic_reward.shape!r}."
            )
        return intrinsic_reward

    def _is_known(self, intrinsic_reward: jax.Array) -> jax.Array:
        return intrinsic_reward < jnp.asarray(
            self.intrinsic_reward_threshold, dtype=intrinsic_reward.dtype
        )

    def select_action(
        self, state: DQNRNDState, obs: jnp.ndarray, key: jax.Array, is_training: bool
    ) -> Tuple[DQNRNDState, jnp.ndarray]:
        raw_obs = jnp.asarray(obs).reshape(-1)
        if is_training:
            state = self._maybe_update_obs_normalizer(state, raw_obs)

        obs = self._normalize_observation(state, raw_obs)
        q_vals = state.online_network(obs.reshape(-1))
        intrinsic_reward = self._action_intrinsic_reward(state, raw_obs)
        values = jnp.where(self._is_known(intrinsic_reward), q_vals, self.optimistic_value)
        action = distrax.Greedy(values).sample(seed=key)

        if is_training:
            state = state.replace(step=state.step + 1)

        return state, action

    def update(
        self, state: DQNRNDState, batch: Transition
    ) -> tuple[DQNRNDState, jax.Array]:
        action = batch.action.astype(jnp.int32)
        terminal = batch.terminal.astype(jnp.float32)
        observation = self._normalize_observation(state, batch.observation)
        next_observation = self._normalize_observation(state, batch.next_observation)

        (
            prediction_error,
            rnd_input,
            rnd_target_features,
            _,
        ) = self._compute_prediction_error(
            state,
            batch.observation,
            action,
        )
        intrinsic_reward = self._normalize_intrinsic_reward(
            prediction_error=prediction_error,
            reward_var=state.intrinsic_reward_var,
        )
        known_mask_f = self._is_known(intrinsic_reward).astype(jnp.float32)
        denom = jnp.maximum(jnp.sum(known_mask_f), 1.0)

        next_intrinsic_reward = self._action_intrinsic_reward(state, batch.next_observation)
        next_has_unknown_action = jnp.any(~self._is_known(next_intrinsic_reward), axis=-1)

        def q_loss_fn(network):
            q_values = network(observation)
            q_sel = jnp.take_along_axis(q_values, action[:, None], axis=1).squeeze(-1)
            next_online_q = network(next_observation)
            next_target_q = state.target_network(next_observation)
            next_bootstrap = self._compute_next_bootstrap_values(
                state=state,
                next_observation=batch.next_observation,
                next_online_q=next_online_q,
                next_target_q=next_target_q,
            )
            max_next_q = jnp.where(
                next_has_unknown_action,
                self.optimistic_value,
                next_bootstrap,
            )
            target = batch.reward + batch.discount * max_next_q * (1.0 - terminal)
            td_error = q_sel - jax.lax.stop_gradient(target)
            per_sample_loss = temporal_difference_loss(
                td_error,
                loss_type=self.loss_type,
                huber_delta=self.huber_delta,
            )
            return jnp.sum(per_sample_loss * known_mask_f) / denom

        q_loss, q_grads = nnx.value_and_grad(q_loss_fn)(state.online_network)

        def rnd_loss_fn(network):
            predictor_features = self._select_rnd_features(network(rnd_input), action)
            return jnp.mean(jnp.square(predictor_features - rnd_target_features))

        should_update_rnd = self._should_update_rnd(state)

        def do_rnd_update(agent_state: DQNRNDState):
            rnd_loss, rnd_grads = nnx.value_and_grad(rnd_loss_fn)(
                agent_state.rnd_predictor_network
            )
            agent_state.rnd_optimizer.update(agent_state.rnd_predictor_network, rnd_grads)
            return agent_state, rnd_loss

        def skip_rnd_update(agent_state: DQNRNDState):
            return agent_state, rnd_loss_fn(agent_state.rnd_predictor_network)

        state, rnd_loss = nnx.cond(
            should_update_rnd,
            do_rnd_update,
            skip_rnd_update,
            state,
        )

        state.optimizer.update(state.online_network, q_grads)
        state = state.replace(gradient_steps=state.gradient_steps + 1)
        hard_update_network(
            source=state.online_network,
            target=state.target_network,
            should_update=state.gradient_steps % self.target_update_freq == 0,
        )
        state = self._update_intrinsic_stats(
            state=state,
            prediction_error=jax.lax.stop_gradient(prediction_error),
        )

        return state, q_loss + rnd_loss

    def bootstrap_value(
        self,
        state: DQNRNDState,
        next_observation: jnp.ndarray,
    ) -> jax.Array:
        raw_next_observation = jnp.asarray(next_observation).reshape(1, -1)
        next_observation = self._normalize_observation(state, raw_next_observation)
        next_online_q = state.online_network(next_observation)
        next_target_q = state.target_network(next_observation)
        next_bootstrap = self._compute_next_bootstrap_values(
            state=state,
            next_observation=raw_next_observation,
            next_online_q=next_online_q,
            next_target_q=next_target_q,
        ).squeeze(0)
        next_intrinsic_reward = self._action_intrinsic_reward(state, raw_next_observation)
        has_unknown_action = jnp.any(~self._is_known(next_intrinsic_reward.squeeze(0)))
        return jnp.where(has_unknown_action, self.optimistic_value, next_bootstrap)
