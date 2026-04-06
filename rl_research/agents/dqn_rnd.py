from collections.abc import Sequence
from typing import Tuple

import distrax
import gin
import jax
import jax.numpy as jnp
from flax import nnx, struct

from rl_research.agents._dqn_common import (
    MLPNetwork,
    ObservationNormalizerState,
    build_optimizer_transform,
    clone_module,
    hard_update_network,
    init_observation_normalizer,
    linear_epsilon,
    normalize_observation,
    temporal_difference_loss,
    update_observation_normalizer,
)
from rl_research.buffers import Transition


class DQNRNDState(struct.PyTreeNode):
    online_network: MLPNetwork
    target_network: MLPNetwork
    optimizer: nnx.Optimizer
    rnd_target_network: MLPNetwork
    rnd_predictor_network: MLPNetwork
    rnd_optimizer: nnx.Optimizer
    obs_normalizer: ObservationNormalizerState
    intrinsic_reward_mean: jax.Array
    intrinsic_reward_var: jax.Array
    step: int
    gradient_steps: int


@gin.configurable
class DQNRNDAgent:
    _RND_ACTION_CONDITIONING_MODES = ("none", "input", "output")

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
        rnd_action_conditioning: str = "none",
        normalize_observations: bool = False,
        obs_normalization_epsilon: float = 1e-8,
        obs_normalization_clip: float | None = 5.0,
        seed: int = 0,
    ):
        self.num_states = int(num_states)
        self.num_actions = int(num_actions)
        self.hidden_units = int(hidden_units)
        self.hidden_dims = (
            None if hidden_dims is None else tuple(int(dim) for dim in hidden_dims)
        )
        self.activation = activation
        self.normalization = normalization
        self.learning_rate = float(learning_rate)
        self.discount = float(discount)
        self.eps_start = float(eps_start)
        self.eps_end = float(eps_end)
        self.eps_decay_steps = int(eps_decay_steps)
        self.target_update_freq = int(target_update_freq)
        self.max_grad_norm = float(max_grad_norm)
        self.optimizer = optimizer
        self.optimizer_beta1 = float(optimizer_beta1)
        self.optimizer_beta2 = float(optimizer_beta2)
        self.optimizer_epsilon = float(optimizer_epsilon)
        self.optimizer_weight_decay = float(optimizer_weight_decay)
        self.optimizer_momentum = float(optimizer_momentum)
        self.optimizer_decay = float(optimizer_decay)
        self.optimizer_centered = bool(optimizer_centered)
        self.loss_type = loss_type
        self.huber_delta = float(huber_delta)
        self.double_q = bool(double_q)
        self.intrinsic_reward_scale = float(intrinsic_reward_scale)
        self.intrinsic_stats_decay = float(intrinsic_stats_decay)
        self.intrinsic_reward_epsilon = float(intrinsic_reward_epsilon)
        self.intrinsic_reward_clip = intrinsic_reward_clip
        self.rnd_hidden_units = (
            self.hidden_units if rnd_hidden_units is None else int(rnd_hidden_units)
        )
        self.rnd_hidden_dims = (
            tuple(int(dim) for dim in rnd_hidden_dims)
            if rnd_hidden_dims is not None
            else self.hidden_dims
        )
        self.rnd_activation = (
            self.activation if rnd_activation is None else rnd_activation
        )
        self.rnd_normalization = (
            self.normalization if rnd_normalization is None else rnd_normalization
        )
        self.rnd_output_dim = int(rnd_output_dim)
        self.rnd_optimizer = self.optimizer if rnd_optimizer is None else rnd_optimizer
        self.rnd_learning_rate = (
            self.learning_rate
            if rnd_learning_rate is None
            else float(rnd_learning_rate)
        )
        self.rnd_action_conditioning = self._resolve_rnd_action_conditioning(
            rnd_include_action=rnd_include_action,
            rnd_action_conditioning=rnd_action_conditioning,
        )
        self.rnd_include_action = self.rnd_action_conditioning == "input"
        self.normalize_observations = bool(normalize_observations)
        self.obs_normalization_epsilon = float(obs_normalization_epsilon)
        self.obs_normalization_clip = obs_normalization_clip
        self.seed = int(seed)

    def _canonicalize_rnd_action_conditioning(self, mode: str | bool) -> str:
        if isinstance(mode, bool):
            return "input" if mode else "none"

        normalized_mode = str(mode).strip().lower()
        aliases = {
            "none": "none",
            "state": "none",
            "observation": "none",
            "input": "input",
            "action_input": "input",
            "include_action": "input",
            "output": "output",
            "action_output": "output",
            "per_action": "output",
        }
        canonical_mode = aliases.get(normalized_mode)
        if canonical_mode is None:
            raise ValueError(
                f"Unsupported rnd_action_conditioning {mode!r}. "
                f"Choose from {', '.join(self._RND_ACTION_CONDITIONING_MODES)}."
            )
        return canonical_mode

    def _resolve_rnd_action_conditioning(
        self,
        rnd_include_action: bool | None,
        rnd_action_conditioning: str | bool,
    ) -> str:
        canonical_mode = self._canonicalize_rnd_action_conditioning(
            rnd_action_conditioning
        )
        if rnd_include_action is None:
            return canonical_mode

        legacy_mode = self._canonicalize_rnd_action_conditioning(rnd_include_action)
        if canonical_mode != "none" and canonical_mode != legacy_mode:
            raise ValueError(
                "rnd_include_action and rnd_action_conditioning disagree. "
                f"Got {rnd_include_action!r} and {rnd_action_conditioning!r}."
            )
        return legacy_mode

    def _rnd_input_dim(self) -> int:
        return self.num_states + (
            self.num_actions if self.rnd_action_conditioning == "input" else 0
        )

    def _rnd_network_output_dim(self) -> int:
        if self.rnd_action_conditioning == "output":
            return self.num_actions * self.rnd_output_dim
        return self.rnd_output_dim

    def initial_state(self) -> DQNRNDState:
        rng = jax.random.PRNGKey(self.seed)
        q_rng, rnd_target_rng, rnd_predictor_rng = jax.random.split(rng, 3)
        rnd_input_dim = self._rnd_input_dim()
        rnd_network_output_dim = self._rnd_network_output_dim()

        online_network = MLPNetwork(
            in_features=self.num_states,
            out_features=self.num_actions,
            rngs=nnx.Rngs(q_rng),
            hidden_features=self.hidden_units,
            hidden_dims=self.hidden_dims,
            activation=self.activation,
            normalization=self.normalization,
        )
        target_network = clone_module(online_network)
        optimizer = nnx.Optimizer(
            online_network,
            build_optimizer_transform(
                learning_rate=self.learning_rate,
                max_grad_norm=self.max_grad_norm,
                optimizer=self.optimizer,
                optimizer_beta1=self.optimizer_beta1,
                optimizer_beta2=self.optimizer_beta2,
                optimizer_epsilon=self.optimizer_epsilon,
                optimizer_weight_decay=self.optimizer_weight_decay,
                optimizer_momentum=self.optimizer_momentum,
                optimizer_decay=self.optimizer_decay,
                optimizer_centered=self.optimizer_centered,
            ),
            wrt=nnx.Param,
        )

        rnd_target_network = MLPNetwork(
            in_features=rnd_input_dim,
            out_features=rnd_network_output_dim,
            rngs=nnx.Rngs(rnd_target_rng),
            hidden_features=self.rnd_hidden_units,
            hidden_dims=self.rnd_hidden_dims,
            activation=self.rnd_activation,
            normalization=self.rnd_normalization,
        )
        rnd_predictor_network = MLPNetwork(
            in_features=rnd_input_dim,
            out_features=rnd_network_output_dim,
            rngs=nnx.Rngs(rnd_predictor_rng),
            hidden_features=self.rnd_hidden_units,
            hidden_dims=self.rnd_hidden_dims,
            activation=self.rnd_activation,
            normalization=self.rnd_normalization,
        )
        rnd_optimizer = nnx.Optimizer(
            rnd_predictor_network,
            build_optimizer_transform(
                learning_rate=self.rnd_learning_rate,
                max_grad_norm=self.max_grad_norm,
                optimizer=self.rnd_optimizer,
                optimizer_beta1=self.optimizer_beta1,
                optimizer_beta2=self.optimizer_beta2,
                optimizer_epsilon=self.optimizer_epsilon,
                optimizer_weight_decay=self.optimizer_weight_decay,
                optimizer_momentum=self.optimizer_momentum,
                optimizer_decay=self.optimizer_decay,
                optimizer_centered=self.optimizer_centered,
            ),
            wrt=nnx.Param,
        )

        return DQNRNDState(
            online_network=online_network,
            target_network=target_network,
            optimizer=optimizer,
            rnd_target_network=rnd_target_network,
            rnd_predictor_network=rnd_predictor_network,
            rnd_optimizer=rnd_optimizer,
            obs_normalizer=init_observation_normalizer(self.num_states),
            intrinsic_reward_mean=jnp.asarray(0.0, dtype=jnp.float32),
            intrinsic_reward_var=jnp.asarray(1.0, dtype=jnp.float32),
            step=0,
            gradient_steps=0,
        )

    def _maybe_update_obs_normalizer(
        self,
        state: DQNRNDState,
        observation: jnp.ndarray,
    ) -> DQNRNDState:
        if not self.normalize_observations:
            return state
        return state.replace(
            obs_normalizer=update_observation_normalizer(
                state.obs_normalizer, observation
            )
        )

    def _normalize_observation(
        self,
        state: DQNRNDState,
        observation: jnp.ndarray,
    ) -> jax.Array:
        if not self.normalize_observations:
            return jnp.asarray(observation, dtype=jnp.float32)
        return normalize_observation(
            observation,
            state.obs_normalizer,
            epsilon=self.obs_normalization_epsilon,
            clip=self.obs_normalization_clip,
        )

    def _build_rnd_input(
        self,
        state: DQNRNDState,
        observation: jnp.ndarray,
        action: jnp.ndarray | None = None,
    ) -> jax.Array:
        observation = self._normalize_observation(state, observation)
        if self.rnd_action_conditioning != "input":
            return observation

        if action is None:
            raise ValueError("Action-conditioned RND requires an action input.")

        action_features = jax.nn.one_hot(
            action.astype(jnp.int32),
            self.num_actions,
            dtype=jnp.float32,
        )
        return jnp.concatenate((observation, action_features), axis=-1)

    def _select_rnd_features(
        self,
        features: jax.Array,
        action: jnp.ndarray | None = None,
    ) -> jax.Array:
        if self.rnd_action_conditioning != "output":
            return features

        reshaped_features = features.reshape(
            *features.shape[:-1], self.num_actions, self.rnd_output_dim
        )
        if action is None:
            return reshaped_features

        action = action.astype(jnp.int32)
        if action.ndim == 0:
            return reshaped_features[action]

        gather_indices = action[..., None, None]
        return jnp.take_along_axis(
            reshaped_features,
            gather_indices,
            axis=-2,
        ).squeeze(axis=-2)

    def select_action(
        self,
        state: DQNRNDState,
        obs: jnp.ndarray,
        key: jax.Array,
        is_training: bool,
    ) -> Tuple[DQNRNDState, jnp.ndarray]:
        if is_training:
            state = self._maybe_update_obs_normalizer(state, obs.reshape(-1))

        obs = self._normalize_observation(state, obs.reshape(-1))
        q_vals = state.online_network(obs.reshape(-1))

        if is_training:
            eps = linear_epsilon(
                state.step,
                eps_start=self.eps_start,
                eps_end=self.eps_end,
                eps_decay_steps=self.eps_decay_steps,
            )
            action = distrax.EpsilonGreedy(q_vals, epsilon=eps).sample(seed=key)
            state = state.replace(step=state.step + 1)
        else:
            action = distrax.Greedy(q_vals).sample(seed=key)

        return state, action

    def _normalize_intrinsic_reward(
        self,
        prediction_error: jax.Array,
        reward_var: jax.Array,
    ) -> jax.Array:
        reward_scale = jnp.sqrt(jnp.maximum(reward_var, self.intrinsic_reward_epsilon))
        normalized_reward = prediction_error / reward_scale
        if self.intrinsic_reward_clip is not None:
            normalized_reward = jnp.clip(
                normalized_reward, 0.0, self.intrinsic_reward_clip
            )
        return normalized_reward

    def _update_intrinsic_stats(
        self,
        state: DQNRNDState,
        prediction_error: jax.Array,
    ) -> DQNRNDState:
        batch_mean = jnp.mean(prediction_error)
        batch_second_moment = jnp.mean(jnp.square(prediction_error))

        decay = self.intrinsic_stats_decay
        old_second_moment = state.intrinsic_reward_var + jnp.square(
            state.intrinsic_reward_mean
        )

        new_mean = decay * state.intrinsic_reward_mean + (1.0 - decay) * batch_mean
        new_second_moment = (
            decay * old_second_moment + (1.0 - decay) * batch_second_moment
        )
        new_var = jnp.maximum(
            new_second_moment - jnp.square(new_mean),
            self.intrinsic_reward_epsilon,
        )

        return state.replace(
            intrinsic_reward_mean=new_mean,
            intrinsic_reward_var=new_var,
        )

    def update(
        self, state: DQNRNDState, batch: Transition
    ) -> tuple[DQNRNDState, jax.Array]:
        action = batch.action.astype(jnp.int32)
        terminal = batch.terminal.astype(jnp.float32)
        observation = self._normalize_observation(state, batch.observation)
        next_observation = self._normalize_observation(state, batch.next_observation)
        rnd_observation = (
            batch.next_observation
            if self.rnd_action_conditioning == "none"
            else batch.observation
        )
        rnd_input = self._build_rnd_input(
            state,
            rnd_observation,
            action if self.rnd_action_conditioning == "input" else None,
        )

        rnd_target_features = jax.lax.stop_gradient(state.rnd_target_network(rnd_input))
        rnd_predictor_features = state.rnd_predictor_network(rnd_input)
        rnd_target_features = self._select_rnd_features(rnd_target_features, action)
        rnd_predictor_features = self._select_rnd_features(
            rnd_predictor_features, action
        )
        prediction_error = jnp.mean(
            jnp.square(rnd_predictor_features - rnd_target_features),
            axis=-1,
        )
        intrinsic_reward = self._normalize_intrinsic_reward(
            prediction_error=prediction_error,
            reward_var=state.intrinsic_reward_var,
        )
        total_reward = (
            batch.reward
            + self.intrinsic_reward_scale * jax.lax.stop_gradient(intrinsic_reward)
        )

        def q_loss_fn(network: MLPNetwork):
            q_values = network(observation)
            q_sel = jnp.take_along_axis(q_values, action[:, None], axis=1).squeeze(-1)

            if self.double_q:
                next_online_q = jax.lax.stop_gradient(network(next_observation))
                next_action = jnp.argmax(next_online_q, axis=1, keepdims=True)
                next_target_q = state.target_network(next_observation)
                max_next_q = jnp.take_along_axis(
                    next_target_q, next_action, axis=1
                ).squeeze(-1)
            else:
                next_q = state.target_network(next_observation)
                max_next_q = jnp.max(next_q, axis=1)

            target = total_reward + batch.discount * max_next_q * (1.0 - terminal)
            td_error = q_sel - jax.lax.stop_gradient(target)
            return jnp.mean(
                temporal_difference_loss(
                    td_error,
                    loss_type=self.loss_type,
                    huber_delta=self.huber_delta,
                )
            )

        q_loss, q_grads = nnx.value_and_grad(q_loss_fn)(state.online_network)
        state.optimizer.update(state.online_network, q_grads)

        def rnd_loss_fn(network: MLPNetwork):
            predictor_features = self._select_rnd_features(network(rnd_input), action)
            return jnp.mean(jnp.square(predictor_features - rnd_target_features))

        rnd_loss, rnd_grads = nnx.value_and_grad(rnd_loss_fn)(
            state.rnd_predictor_network
        )
        state.rnd_optimizer.update(state.rnd_predictor_network, rnd_grads)

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
        next_observation = self._normalize_observation(
            state, next_observation.reshape(1, -1)
        )
        q_vals = state.target_network(next_observation.reshape(1, -1))
        return jnp.max(q_vals, axis=-1).squeeze()
