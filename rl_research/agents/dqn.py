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


class DQNState(struct.PyTreeNode):
    online_network: MLPNetwork
    target_network: MLPNetwork
    optimizer: nnx.Optimizer
    obs_normalizer: ObservationNormalizerState
    step: jax.Array
    gradient_steps: jax.Array


@gin.configurable
class DQNAgent:
    def __init__(
        self,
        num_states: int,  # this is the input size
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
        self.normalize_observations = bool(normalize_observations)
        self.obs_normalization_epsilon = float(obs_normalization_epsilon)
        self.obs_normalization_clip = obs_normalization_clip
        self.seed = int(seed)

    def initial_state(self) -> DQNState:
        rng = jax.random.PRNGKey(self.seed)
        online_network = MLPNetwork(
            in_features=self.num_states,
            out_features=self.num_actions,
            rngs=nnx.Rngs(rng),
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

        return DQNState(
            online_network=online_network,
            target_network=target_network,
            optimizer=optimizer,
            obs_normalizer=init_observation_normalizer(self.num_states),
            step=jnp.asarray(0, dtype=jnp.int32),
            gradient_steps=jnp.asarray(0, dtype=jnp.int32),
        )

    def _maybe_update_obs_normalizer(
        self,
        state: DQNState,
        observation: jnp.ndarray,
    ) -> DQNState:
        if not self.normalize_observations:
            return state
        return state.replace(
            obs_normalizer=update_observation_normalizer(
                state.obs_normalizer, observation
            )
        )

    def _normalize_observation(
        self,
        state: DQNState,
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

    def select_action(
        self, state: DQNState, obs: jnp.ndarray, key: jax.Array, is_training: bool
    ) -> Tuple[DQNState, jnp.ndarray]:
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

    def update(self, state: DQNState, batch: Transition) -> tuple[DQNState, jax.Array]:
        action = batch.action.astype(jnp.int32)
        terminal = batch.terminal.astype(jnp.float32)
        observation = self._normalize_observation(state, batch.observation)
        next_observation = self._normalize_observation(state, batch.next_observation)

        def loss_fn(network: MLPNetwork):
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

            target = batch.reward + batch.discount * max_next_q * (1.0 - terminal)
            td_error = q_sel - jax.lax.stop_gradient(target)
            loss = jnp.mean(
                temporal_difference_loss(
                    td_error,
                    loss_type=self.loss_type,
                    huber_delta=self.huber_delta,
                )
            )
            return loss

        loss, grads = nnx.value_and_grad(loss_fn)(state.online_network)
        state.optimizer.update(state.online_network, grads)

        state = state.replace(gradient_steps=state.gradient_steps + 1)

        should_update = state.gradient_steps % self.target_update_freq == 0
        hard_update_network(
            source=state.online_network,
            target=state.target_network,
            should_update=should_update,
        )

        return state, loss

    def bootstrap_value(
        self, state: DQNState, next_observation: jnp.ndarray
    ) -> jax.Array:
        next_observation = self._normalize_observation(
            state, next_observation.reshape(1, -1)
        )
        q_vals = state.target_network(next_observation.reshape(1, -1))
        return jnp.max(q_vals, axis=-1).squeeze()
