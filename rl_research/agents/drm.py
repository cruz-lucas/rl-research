"""Replay-based R-max + DQN agent."""
import gin
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from flax import struct
import distrax

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

class DRMState(struct.PyTreeNode):
    online_network: Network
    target_network: Network
    optimizer: nnx.Optimizer
    visit_counts: jnp.ndarray
    step: int


@gin.configurable
class DRMAgent:
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
        self.num_obs_ids = (grid_size ** 2) * 2 * 2 * 4
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.discount = discount
        self.known_threshold = int(known_threshold)
        self.optimistic_value = v_max if use_vmax else (r_max / (1.0 - discount))
        self.target_update_freq = int(target_update_freq)
        self.max_grad_norm = max_grad_norm
        self.seed = int(seed)

    def initial_state(self) -> DRMState:
        rng = jax.random.PRNGKey(self.seed)
        rng_online, rng_target = jax.random.split(rng)
        online_network = Network(
            in_features=self.num_states,
            out_features=self.num_actions,
            rngs=nnx.Rngs(rng_online),
            hidden_features=self.hidden_units,
        )
        target_network = Network(
            in_features=self.num_states,
            out_features=self.num_actions,
            rngs=nnx.Rngs(rng_target),
            hidden_features=self.hidden_units,
        )

        optimizer = nnx.Optimizer(
            online_network,
            optax.chain(optax.clip_by_global_norm(self.max_grad_norm), optax.adam(self.learning_rate)),
            wrt=nnx.Param,
        )

        visit_counts = jnp.zeros((self.num_obs_ids, self.num_actions), dtype=jnp.int32)

        return DRMState(
            online_network=online_network,
            target_network=target_network,
            optimizer=optimizer,
            visit_counts=visit_counts,
            step=0,
        )
    
    # def obs_to_index(self, obs: jax.Array) -> jax.Array:
    #     obs = obs.reshape((-1, self.num_states))

    #     pos_ids = self.grid_size ** 2
    #     f1 = jnp.argmax(obs[:, :pos_ids], axis=1)
    #     f2 = jnp.argmax(obs[:, (pos_ids+2):(pos_ids+4)], axis=1)
    #     f3 = jnp.argmax(obs[:, (pos_ids+4):(pos_ids+6)], axis=1)
    #     f4 = jnp.argmax(obs[:, -4:], axis=1)

    #     return f1 + 25 * (f2 + 2 * (f3 + 2 * f4))
    
    def obs_to_index(self, obs: jax.Array) -> jax.Array:
        obs = obs.reshape((-1, self.grid_size, self.grid_size, 3))
        B, G, _, _ = obs.shape

        player_mask = obs[:, :, :, 0] == 10
        _, prow, pcol = jnp.where(player_mask, size=B, fill_value=0)

        player_pos = (prow - 1) * (G - 2) + (pcol - 1)

        door_mask = obs[:, :, :, 0] == 4
        dbatch, drow, dcol = jnp.where(door_mask, size=B, fill_value=0)

        door_open = obs[dbatch, drow, dcol, -1] == 2

        key_mask = obs[:, :, :, 0] == 5
        _, krow, _ = jnp.where(key_mask, size=B, fill_value=0)

        kpicked = krow == 0

        direction = obs[jnp.arange(B), prow, pcol, -1]

        return jnp.int16(((player_pos * 2 + door_open) * 2 + kpicked) * 4 + direction)

    def select_action(self, state: DRMState, obs: jnp.ndarray, key: jax.Array, is_training: bool) -> jnp.ndarray:
        q_vals = state.online_network(obs.reshape(-1)).squeeze()

        obs_ids = self.obs_to_index(obs.reshape(-1))
        is_known = state.visit_counts[obs_ids] >= self.known_threshold

        values = jnp.where(is_known, q_vals, self.optimistic_value)

        action_dist = distrax.Greedy(values)
        return action_dist.sample(seed=key)[0]

    def update(self, state: DRMState, batch: Transition) -> tuple[DRMState, jax.Array]:
        obs_ids = self.obs_to_index(batch.observation)
        next_obs_ids = self.obs_to_index(batch.next_observation)
        new_visit_counts = state.visit_counts.at[obs_ids, batch.action].add(1)

        known_mask = new_visit_counts[obs_ids, batch.action] >= self.known_threshold
        known_mask_f = known_mask.astype(jnp.float32)
        denom = jnp.maximum(jnp.sum(known_mask_f), 1.0)

        def loss_fn(network: Network):
            q_values = network(batch.observation)
            q_sel = jnp.take_along_axis(q_values, batch.action[:, None], axis=1).squeeze()

            next_q = state.target_network(batch.next_observation)
            
            max_next_q = jnp.where(
                jnp.any(new_visit_counts[next_obs_ids, :] < self.known_threshold, axis=-1),
                self.optimistic_value,
                jnp.max(next_q, axis=1),
            )

            target = batch.reward + self.discount * max_next_q * (1.0 - batch.terminal)
            
            per_sample_loss = (q_sel - jax.lax.stop_gradient(target)) ** 2
            masked_loss = per_sample_loss * known_mask_f
            mean_loss = jnp.sum(masked_loss) / denom
            return mean_loss

        loss, grads = nnx.value_and_grad(loss_fn)(state.online_network)
        state.optimizer.update(state.online_network, grads)

        _graphdef, _state = nnx.cond(
            state.step % self.target_update_freq == 0,
            lambda _: nnx.split(state.online_network),
            lambda _: nnx.split(state.target_network),
            None,
        )
        nnx.update(state.target_network, _state)

        new_state = state.replace(
            visit_counts=new_visit_counts,
        )

        return new_state, loss

    def bootstrap_value(self, state: DRMState, next_observation: jnp.ndarray) -> jax.Array:
        # TODO: implement this properly.
        return jnp.array(0.0)
