from typing import Tuple

import distrax
import gin
import jax
import jax.numpy as jnp
import optax
from flax import nnx, struct

from rl_research.agents._dqn_common import (
    MLPNetwork,
    clone_module,
    hard_update_network,
    linear_epsilon,
)
from rl_research.buffers import Transition


class DQNRNDState(struct.PyTreeNode):
    online_network: MLPNetwork
    target_network: MLPNetwork
    optimizer: nnx.Optimizer
    rnd_target_network: MLPNetwork
    rnd_predictor_network: MLPNetwork
    rnd_optimizer: nnx.Optimizer
    intrinsic_reward_mean: jax.Array
    intrinsic_reward_var: jax.Array
    step: int
    gradient_steps: int


@gin.configurable
class DQNRNDAgent:
    def __init__(
        self,
        num_states: int,
        num_actions: int,
        hidden_units: int = 64,
        learning_rate: float = 1e-3,
        discount: float = 0.99,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        eps_decay_steps: int = 100_000,
        target_update_freq: int = 1000,
        max_grad_norm: float = 1.0,
        intrinsic_reward_scale: float = 1.0,
        intrinsic_stats_decay: float = 0.99,
        intrinsic_reward_epsilon: float = 1e-4,
        intrinsic_reward_clip: float | None = 10.0,
        rnd_hidden_units: int | None = None,
        rnd_output_dim: int = 64,
        rnd_learning_rate: float | None = None,
        seed: int = 0,
    ):
        self.num_states = int(num_states)
        self.num_actions = int(num_actions)
        self.hidden_units = int(hidden_units)
        self.learning_rate = float(learning_rate)
        self.discount = float(discount)
        self.eps_start = float(eps_start)
        self.eps_end = float(eps_end)
        self.eps_decay_steps = int(eps_decay_steps)
        self.target_update_freq = int(target_update_freq)
        self.max_grad_norm = float(max_grad_norm)
        self.intrinsic_reward_scale = float(intrinsic_reward_scale)
        self.intrinsic_stats_decay = float(intrinsic_stats_decay)
        self.intrinsic_reward_epsilon = float(intrinsic_reward_epsilon)
        self.intrinsic_reward_clip = intrinsic_reward_clip
        self.rnd_hidden_units = (
            self.hidden_units if rnd_hidden_units is None else int(rnd_hidden_units)
        )
        self.rnd_output_dim = int(rnd_output_dim)
        self.rnd_learning_rate = (
            self.learning_rate
            if rnd_learning_rate is None
            else float(rnd_learning_rate)
        )
        self.seed = int(seed)

    def initial_state(self) -> DQNRNDState:
        rng = jax.random.PRNGKey(self.seed)
        q_rng, rnd_target_rng, rnd_predictor_rng = jax.random.split(rng, 3)

        online_network = MLPNetwork(
            in_features=self.num_states,
            out_features=self.num_actions,
            rngs=nnx.Rngs(q_rng),
            hidden_features=self.hidden_units,
        )
        target_network = clone_module(online_network)
        optimizer = nnx.Optimizer(
            online_network,
            optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.adam(self.learning_rate),
            ),
            wrt=nnx.Param,
        )

        rnd_target_network = MLPNetwork(
            in_features=self.num_states,
            out_features=self.rnd_output_dim,
            rngs=nnx.Rngs(rnd_target_rng),
            hidden_features=self.rnd_hidden_units,
        )
        rnd_predictor_network = MLPNetwork(
            in_features=self.num_states,
            out_features=self.rnd_output_dim,
            rngs=nnx.Rngs(rnd_predictor_rng),
            hidden_features=self.rnd_hidden_units,
        )
        rnd_optimizer = nnx.Optimizer(
            rnd_predictor_network,
            optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.adam(self.rnd_learning_rate),
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
            intrinsic_reward_mean=jnp.asarray(0.0, dtype=jnp.float32),
            intrinsic_reward_var=jnp.asarray(1.0, dtype=jnp.float32),
            step=0,
            gradient_steps=0,
        )

    def select_action(
        self,
        state: DQNRNDState,
        obs: jnp.ndarray,
        key: jax.Array,
        is_training: bool,
    ) -> Tuple[DQNRNDState, jnp.ndarray]:
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

        rnd_target_features = jax.lax.stop_gradient(
            state.rnd_target_network(batch.next_observation)
        )
        rnd_predictor_features = state.rnd_predictor_network(batch.next_observation)
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
            q_values = network(batch.observation)
            q_sel = jnp.take_along_axis(q_values, action[:, None], axis=1).squeeze(-1)

            next_q = state.target_network(batch.next_observation)
            max_next_q = jnp.max(next_q, axis=1)

            target = total_reward + batch.discount * max_next_q * (1.0 - terminal)
            return jnp.mean((q_sel - jax.lax.stop_gradient(target)) ** 2)

        q_loss, q_grads = nnx.value_and_grad(q_loss_fn)(state.online_network)
        state.optimizer.update(state.online_network, q_grads)

        def rnd_loss_fn(network: MLPNetwork):
            predictor_features = network(batch.next_observation)
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
        q_vals = state.target_network(next_observation.reshape(1, -1))
        return jnp.max(q_vals, axis=-1).squeeze()
