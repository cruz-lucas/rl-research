from collections.abc import Sequence
from typing import Tuple

import distrax
import gin
import jax
import jax.numpy as jnp
import optax
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
from rl_research.debug_logging import DQNRNDDebugLogger


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
    step: jax.Array
    gradient_steps: jax.Array


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
        rnd_update_period: int = 1,
        normalize_observations: bool = False,
        obs_normalization_epsilon: float = 1e-8,
        obs_normalization_clip: float | None = 5.0,
        debug: bool = False,
        debug_log_dir: str = "tmp/debug_logs",
        debug_log_to_mlflow: bool = True,
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
        self.rnd_update_period = int(rnd_update_period)
        if self.rnd_update_period < 1:
            raise ValueError("rnd_update_period must be at least 1.")
        self.normalize_observations = bool(normalize_observations)
        self.obs_normalization_epsilon = float(obs_normalization_epsilon)
        self.obs_normalization_clip = obs_normalization_clip
        self.debug = bool(debug)
        self.debug_log_dir = debug_log_dir
        self.debug_log_to_mlflow = bool(debug_log_to_mlflow)
        self.seed = int(seed)
        self._debug_logger = (
            DQNRNDDebugLogger(
                log_dir=self.debug_log_dir,
                agent_class=self.__class__.__name__,
                num_states=self.num_states,
                num_actions=self.num_actions,
                discount=self.discount,
                rnd_action_conditioning=self.rnd_action_conditioning,
                normalize_observations=self.normalize_observations,
                log_to_mlflow=self.debug_log_to_mlflow,
            )
            if self.debug
            else None
        )

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
            step=jnp.asarray(0, dtype=jnp.int32),
            gradient_steps=jnp.asarray(0, dtype=jnp.int32),
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

    def _compute_prediction_error(
        self,
        state: DQNRNDState,
        observation: jnp.ndarray,
        action: jnp.ndarray | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        rnd_input = self._build_rnd_input(state, observation, action)
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
        return (
            prediction_error,
            rnd_input,
            rnd_target_features,
            rnd_predictor_features,
        )

    def _compute_intrinsic_reward(
        self,
        state: DQNRNDState,
        observation: jnp.ndarray,
        action: jnp.ndarray | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        prediction_error, _, _, _ = self._compute_prediction_error(
            state,
            observation,
            action,
        )
        intrinsic_reward = self._normalize_intrinsic_reward(
            prediction_error=prediction_error,
            reward_var=state.intrinsic_reward_var,
        )
        return intrinsic_reward, prediction_error

    def _compute_decision_bonus(
        self,
        state: DQNRNDState,
        observation: jnp.ndarray,
    ) -> jax.Array:
        observation = jnp.asarray(observation, dtype=jnp.float32)
        single_observation = observation.ndim == 1
        if single_observation:
            observation = observation.reshape(1, -1)
        if self.rnd_action_conditioning == "input":
            batch_size = observation.shape[0]
            observation_batch = jnp.broadcast_to(
                observation[:, None, :],
                (batch_size, self.num_actions, observation.shape[-1]),
            )
            action_batch = jnp.broadcast_to(
                jnp.arange(self.num_actions, dtype=jnp.int32),
                (batch_size, self.num_actions),
            )
            decision_bonus, _ = self._compute_intrinsic_reward(
                state,
                observation_batch.reshape(-1, observation.shape[-1]),
                action_batch.reshape(-1),
            )
            decision_bonus = decision_bonus.reshape(batch_size, self.num_actions)
        else:
            decision_bonus, _ = self._compute_intrinsic_reward(state, observation)

        if single_observation:
            return decision_bonus.squeeze(axis=0)
        return decision_bonus

    def _log_decision_debug(
        self,
        *,
        global_step: jax.Array,
        observation: jax.Array,
        q_values: jax.Array,
        action: jax.Array,
        epsilon: jax.Array | None = None,
        decision_bonus: jax.Array | None = None,
        decision_values: jax.Array | None = None,
    ) -> None:
        if self._debug_logger is None:
            return
        callback_kwargs = {}
        if decision_bonus is not None:
            callback_kwargs["decision_bonus"] = decision_bonus
        if decision_values is not None:
            callback_kwargs["decision_values"] = decision_values
        jax.debug.callback(
            self._debug_logger.log_decision,
            global_step,
            observation,
            q_values,
            jnp.asarray(jnp.nan, dtype=jnp.float32) if epsilon is None else epsilon,
            action,
            ordered=True,
            **callback_kwargs,
        )

    def _log_update_debug(
        self,
        *,
        global_step: jax.Array,
        gradient_step: jax.Array,
        observation: jax.Array,
        action: jax.Array,
        reward: jax.Array,
        next_observation: jax.Array,
        terminal: jax.Array,
        q_values_observation: jax.Array,
        q_values_next_observation_online: jax.Array,
        q_values_next_observation_target: jax.Array,
        intrinsic_reward_observation: jax.Array,
        intrinsic_reward_next_observation: jax.Array,
        intrinsic_reward_used: jax.Array,
        target_without_intrinsic: jax.Array,
        target_with_intrinsic: jax.Array,
        q_loss: jax.Array,
        rnd_loss: jax.Array,
        q_grad_norm_with_intrinsic: jax.Array,
        q_grad_norm_without_intrinsic: jax.Array,
        rnd_grad_norm: jax.Array,
    ) -> None:
        if self._debug_logger is None:
            return
        jax.debug.callback(
            self._debug_logger.log_update,
            global_step,
            gradient_step,
            observation,
            action,
            reward,
            next_observation,
            terminal,
            q_values_observation,
            q_values_next_observation_online,
            q_values_next_observation_target,
            intrinsic_reward_observation,
            intrinsic_reward_next_observation,
            intrinsic_reward_used,
            target_without_intrinsic,
            target_with_intrinsic,
            q_loss,
            rnd_loss,
            q_grad_norm_with_intrinsic,
            q_grad_norm_without_intrinsic,
            rnd_grad_norm,
            ordered=True,
        )

    def close_debug_logger(self) -> None:
        if self._debug_logger is not None:
            self._debug_logger.close()

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

        if is_training:
            decision_bonus = (
                self._compute_decision_bonus(state, raw_obs) if self.debug else None
            )
            eps = linear_epsilon(
                state.step,
                eps_start=self.eps_start,
                eps_end=self.eps_end,
                eps_decay_steps=self.eps_decay_steps,
            )
            action = distrax.EpsilonGreedy(q_vals, epsilon=eps).sample(seed=key)
            self._log_decision_debug(
                global_step=jnp.asarray(state.step, dtype=jnp.int32),
                observation=raw_obs,
                q_values=q_vals,
                action=action,
                epsilon=eps,
                decision_bonus=decision_bonus,
            )
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

    def _compute_next_bootstrap_values(
        self,
        state: DQNRNDState,
        next_observation: jax.Array,
        next_online_q: jax.Array,
        next_target_q: jax.Array,
    ) -> jax.Array:
        del state, next_observation
        if self.double_q:
            next_action = jnp.argmax(
                jax.lax.stop_gradient(next_online_q), axis=-1, keepdims=True
            )
            return jnp.take_along_axis(next_target_q, next_action, axis=-1).squeeze(-1)
        return jnp.max(next_target_q, axis=-1)

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

    def _should_update_rnd(self, state: DQNRNDState) -> jax.Array:
        if self.rnd_update_period == 1:
            return jnp.asarray(True)
        return (state.gradient_steps + 1) % self.rnd_update_period == 0

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
        rnd_action = action if self.rnd_action_conditioning != "none" else None
        (
            prediction_error,
            rnd_input,
            rnd_target_features,
            _,
        ) = self._compute_prediction_error(
            state,
            rnd_observation,
            rnd_action,
        )
        intrinsic_reward = self._normalize_intrinsic_reward(
            prediction_error=prediction_error,
            reward_var=state.intrinsic_reward_var,
        )
        total_reward = (
            batch.reward
            + self.intrinsic_reward_scale * jax.lax.stop_gradient(intrinsic_reward)
        )

        def q_loss_fn(network: MLPNetwork, reward_vector: jax.Array):
            q_values = network(observation)
            q_sel = jnp.take_along_axis(q_values, action[:, None], axis=1).squeeze(-1)
            next_online_q = network(next_observation)
            next_target_q = state.target_network(next_observation)
            max_next_q = self._compute_next_bootstrap_values(
                state=state,
                next_observation=batch.next_observation,
                next_online_q=next_online_q,
                next_target_q=next_target_q,
            )

            target = reward_vector + batch.discount * max_next_q * (1.0 - terminal)
            td_error = q_sel - jax.lax.stop_gradient(target)
            loss = jnp.mean(
                temporal_difference_loss(
                    td_error,
                    loss_type=self.loss_type,
                    huber_delta=self.huber_delta,
                )
            )
            aux = {
                "q_values": q_values,
                "next_online_q_values": next_online_q,
                "next_target_q_values": next_target_q,
                "target": target,
            }
            return loss, aux

        if self.debug:
            (q_loss, q_aux), q_grads = nnx.value_and_grad(
                lambda network: q_loss_fn(network, total_reward),
                has_aux=True,
            )(state.online_network)
            (
                (_, q_aux_without_intrinsic),
                q_grads_without_intrinsic,
            ) = nnx.value_and_grad(
                lambda network: q_loss_fn(network, batch.reward),
                has_aux=True,
            )(state.online_network)
            q_grad_norm_with_intrinsic = optax.global_norm(q_grads)
            q_grad_norm_without_intrinsic = optax.global_norm(q_grads_without_intrinsic)

            intrinsic_reward_observation, _ = self._compute_intrinsic_reward(
                state,
                batch.observation,
                rnd_action,
            )
            intrinsic_reward_next_observation, _ = self._compute_intrinsic_reward(
                state,
                batch.next_observation,
                rnd_action,
            )
        else:
            (q_loss, _), q_grads = nnx.value_and_grad(
                lambda network: q_loss_fn(network, total_reward),
                has_aux=True,
            )(state.online_network)

        def rnd_loss_fn(network: MLPNetwork):
            predictor_features = self._select_rnd_features(network(rnd_input), action)
            return jnp.mean(jnp.square(predictor_features - rnd_target_features))

        should_update_rnd = self._should_update_rnd(state)

        if self.debug:
            def do_rnd_update(agent_state: DQNRNDState):
                rnd_loss, rnd_grads = nnx.value_and_grad(rnd_loss_fn)(
                    agent_state.rnd_predictor_network
                )
                agent_state.rnd_optimizer.update(
                    agent_state.rnd_predictor_network, rnd_grads
                )
                rnd_grad_norm = optax.global_norm(rnd_grads)
                return agent_state, rnd_loss, rnd_grad_norm

            def skip_rnd_update(agent_state: DQNRNDState):
                rnd_loss = rnd_loss_fn(agent_state.rnd_predictor_network)
                return agent_state, rnd_loss, jnp.asarray(0.0, dtype=jnp.float32)

            state, rnd_loss, rnd_grad_norm = nnx.cond(
                should_update_rnd,
                do_rnd_update,
                skip_rnd_update,
                state,
            )
            self._log_update_debug(
                global_step=jnp.asarray(state.step, dtype=jnp.int32),
                gradient_step=jnp.asarray(state.gradient_steps + 1, dtype=jnp.int32),
                observation=batch.observation,
                action=action,
                reward=batch.reward,
                next_observation=batch.next_observation,
                terminal=batch.terminal,
                q_values_observation=q_aux["q_values"],
                q_values_next_observation_online=q_aux["next_online_q_values"],
                q_values_next_observation_target=q_aux["next_target_q_values"],
                intrinsic_reward_observation=intrinsic_reward_observation,
                intrinsic_reward_next_observation=intrinsic_reward_next_observation,
                intrinsic_reward_used=intrinsic_reward,
                target_without_intrinsic=q_aux_without_intrinsic["target"],
                target_with_intrinsic=q_aux["target"],
                q_loss=q_loss,
                rnd_loss=rnd_loss,
                q_grad_norm_with_intrinsic=q_grad_norm_with_intrinsic,
                q_grad_norm_without_intrinsic=q_grad_norm_without_intrinsic,
                rnd_grad_norm=rnd_grad_norm,
            )
        else:
            def do_rnd_update(agent_state: DQNRNDState):
                rnd_loss, rnd_grads = nnx.value_and_grad(rnd_loss_fn)(
                    agent_state.rnd_predictor_network
                )
                agent_state.rnd_optimizer.update(
                    agent_state.rnd_predictor_network, rnd_grads
                )
                return agent_state, rnd_loss

            def skip_rnd_update(agent_state: DQNRNDState):
                rnd_loss = rnd_loss_fn(agent_state.rnd_predictor_network)
                return agent_state, rnd_loss

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
        next_online_q = state.online_network(next_observation.reshape(1, -1))
        next_target_q = state.target_network(next_observation.reshape(1, -1))
        bootstrap_value = self._compute_next_bootstrap_values(
            state=state,
            next_observation=raw_next_observation,
            next_online_q=next_online_q,
            next_target_q=next_target_q,
        )
        return bootstrap_value.squeeze()
