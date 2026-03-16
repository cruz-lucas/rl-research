import gin
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from flax import struct
import distrax
from typing import Tuple
from dataclasses import dataclass

from rl_research.buffers import Transition


class Network(nnx.Module):
    def __init__(self, in_features: int , out_features: int, rngs: nnx.Rngs, hidden_features: int = 64):
        # self.in_layer = nnx.Linear(in_features=in_features, out_features=out_features, rngs=rngs)
        
        self.in_layer = nnx.Linear(in_features=in_features, out_features=hidden_features, rngs=rngs)
        self.hidden_layer = nnx.Linear(in_features=hidden_features, out_features=hidden_features, rngs=rngs)
        self.layernorm = nnx.LayerNorm(num_features=hidden_features, rngs=rngs)
        self.out_layer = nnx.Linear(in_features=hidden_features, out_features=out_features, rngs=rngs)
    
    def __call__(self, x):
        x = self.in_layer(x)
        x = nnx.relu(x)
        x = self.hidden_layer(x)
        x = self.layernorm(x)
        x = nnx.relu(x)
        x = self.out_layer(x)
        return x

class DQNState(struct.PyTreeNode):
    online_network: Network
    target_network: Network
    optimizer: nnx.Optimizer
    step: int


@gin.configurable
class DQNAgent:
    def __init__(
        self,
        num_states: int, # this is the input size
        num_actions: int,
        hidden_units: int = 64,
        learning_rate: float = 1e-3,
        discount: float = 0.99,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        eps_decay_steps: int = 100_000,
        target_update_freq: int = 1000,
        max_grad_norm: float = 1.0,
        seed: int = 0,
    ):
        self.num_states = int(num_states)
        self.num_actions = int(num_actions)
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.discount = discount
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.target_update_freq = int(target_update_freq)
        self.max_grad_norm = max_grad_norm
        self.seed = int(seed)

    def initial_state(self) -> DQNState:
        rng = jax.random.PRNGKey(self.seed)
        online_network = Network(in_features=self.num_states, out_features=self.num_actions, rngs=nnx.Rngs(rng), hidden_features=self.hidden_units)
        graphdef, online_params = nnx.split(online_network)
        target_network = nnx.merge(graphdef, online_params)
        optimizer = nnx.Optimizer(
            online_network,
            optax.chain(optax.clip_by_global_norm(self.max_grad_norm), optax.adam(self.learning_rate)),
            wrt=nnx.Param
        )
        
        return DQNState(
            online_network=online_network,
            target_network=target_network,
            optimizer=optimizer,
            step=0,
        )

    def select_action(self, state: DQNState, obs: jnp.ndarray, key: jax.Array, is_training: bool) -> Tuple[DQNState, jnp.ndarray]:
        q_vals = state.online_network(obs.reshape(-1))

        if is_training:
            frac = jnp.clip(state.step / max(1, self.eps_decay_steps), 0.0, 1.0)
            eps = self.eps_start + frac * (self.eps_end - self.eps_start)

            action = distrax.EpsilonGreedy(q_vals, epsilon=eps).sample(seed=key)

            state = state.replace(
                step = state.step + 1,
         
            )

        else:
            action = distrax.Greedy(q_vals).sample(seed=key)

        return state, action

    def update(self, state: DQNState, batch: Transition) -> tuple[DQNState, jax.Array]:
        def loss_fn(network: Network):
            q_values = network(batch.observation)
            q_sel = jnp.take_along_axis(q_values, batch.action[:, None], axis=1).squeeze()

            next_q = state.target_network(batch.next_observation)
            max_next_q = jnp.max(next_q, axis=1)

            target = batch.reward + self.discount * max_next_q * (1.0 - batch.terminal)
            loss = jnp.mean((q_sel - jax.lax.stop_gradient(target)) ** 2)
            return loss

        loss, grads = nnx.value_and_grad(loss_fn)(state.online_network)
        state.optimizer.update(state.online_network, grads)

        _, online_params = nnx.split(state.online_network)
        _, target_params = nnx.split(state.target_network)

        should_update = state.step % self.target_update_freq == 0
        new_target_params = jax.tree.map(
            lambda o, t: jnp.where(should_update, o, t),
            online_params,
            target_params,
        )

        nnx.update(state.target_network, new_target_params)

        return state, loss

    def bootstrap_value(self, state: DQNState, next_observation: jnp.ndarray) -> jax.Array:
        q_vals = state.target_network(next_observation.reshape(1, -1))
        return jnp.max(q_vals, axis=-1).squeeze()
