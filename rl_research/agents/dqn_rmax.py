"""Replay-based R-max + DQN agent."""

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
)
from rl_research.buffers import Transition
from rl_research.environments.navix import obs_to_index


class DQNRmaxState(struct.PyTreeNode):
    online_network: MLPNetwork
    target_network: MLPNetwork
    optimizer: nnx.Optimizer
    step: int
    gradient_steps: int
    visit_counts: jnp.ndarray


@gin.configurable
class DQNRmaxAgent:
    def __init__(
        self,
        num_states: int,
        num_actions: int,
        grid_size: int,
        hidden_units: int = 64,
        learning_rate: float = 1e-3,
        discount: float = 0.99,
        r_max: float = 1.0,
        v_max: float = 1.0,
        use_vmax: bool = True,
        known_threshold: int = 1,
        target_update_freq: int = 1000,
        max_grad_norm: float = 1.0,
        seed: int = 0,
    ):
        self.num_states = int(num_states)
        self.num_actions = int(num_actions)
        self.grid_size = int(grid_size)
        # Specific for Door Key environment, needs to change if obs_to_index is modified
        self.num_obs_ids = ((grid_size - 2) ** 2) * (((grid_size - 2) ** 2) + 1) * 2 * 4
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.discount = discount
        self.known_threshold = int(known_threshold)
        self.optimistic_value = v_max if use_vmax else (r_max / (1.0 - discount))
        self.target_update_freq = int(target_update_freq)
        self.max_grad_norm = max_grad_norm
        self.seed = int(seed)

    def initial_state(self) -> DQNRmaxState:
        rng = jax.random.PRNGKey(self.seed)
        online_network = MLPNetwork(
            in_features=self.num_states,
            out_features=self.num_actions,
            rngs=nnx.Rngs(rng),
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

        visit_counts = jnp.zeros((self.num_obs_ids, self.num_actions), dtype=jnp.int32)

        return DQNRmaxState(
            online_network=online_network,
            target_network=target_network,
            optimizer=optimizer,
            visit_counts=visit_counts,
            step=0,
            gradient_steps=0,
        )

    def select_action(
        self, state: DQNRmaxState, obs: jnp.ndarray, key: jax.Array, is_training: bool
    ) -> Tuple[DQNRmaxState, jnp.ndarray]:
        q_vals = state.online_network(obs.reshape(-1))

        obs_id = jnp.squeeze(
            obs_to_index(obs.reshape(-1), grid_size=self.grid_size), axis=0
        )
        is_known = state.visit_counts[obs_id] >= self.known_threshold

        values = jnp.where(is_known, q_vals, self.optimistic_value)

        action = distrax.Greedy(values).sample(seed=key)

        if is_training:
            state = state.replace(
                step=state.step + 1,
                visit_counts=state.visit_counts.at[
                    obs_id, action.astype(jnp.int32)
                ].add(1),
            )

        return state, action

    def update(
        self, state: DQNRmaxState, batch: Transition
    ) -> tuple[DQNRmaxState, jax.Array]:
        obs_ids = obs_to_index(batch.observation, grid_size=self.grid_size)
        next_obs_ids = obs_to_index(batch.next_observation, grid_size=self.grid_size)
        action = batch.action.astype(jnp.int32)
        terminal = batch.terminal.astype(jnp.float32)

        known_mask = state.visit_counts[obs_ids, action] >= self.known_threshold
        known_mask_f = known_mask.astype(jnp.float32)
        denom = jnp.maximum(jnp.sum(known_mask_f), 1.0)

        def loss_fn(network: MLPNetwork):
            q_values = network(batch.observation)
            q_sel = jnp.take_along_axis(q_values, action[:, None], axis=1).squeeze(-1)

            next_q = state.target_network(batch.next_observation)

            max_next_q = jnp.where(
                jnp.any(
                    state.visit_counts[next_obs_ids, :] < self.known_threshold, axis=-1
                ),
                self.optimistic_value,
                jnp.max(next_q, axis=1),
            )

            target = batch.reward + batch.discount * max_next_q * (1.0 - terminal)
            per_sample_loss = (q_sel - jax.lax.stop_gradient(target)) ** 2
            masked_loss = per_sample_loss * known_mask_f
            mean_loss = jnp.sum(masked_loss) / denom
            return mean_loss

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
        self, state: DQNRmaxState, next_observation: jnp.ndarray
    ) -> jax.Array:
        obs_id = jnp.squeeze(
            obs_to_index(next_observation.reshape(-1), grid_size=self.grid_size), axis=0
        )
        q_vals = state.target_network(next_observation.reshape(1, -1)).squeeze(0)
        has_unknown_action = jnp.any(state.visit_counts[obs_id] < self.known_threshold)
        return jnp.where(has_unknown_action, self.optimistic_value, jnp.max(q_vals))
