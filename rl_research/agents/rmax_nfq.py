import gin
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from flax import struct
from flax.training.train_state import TrainState
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

class NFQState(struct.PyTreeNode):
    online_network: Network
    optimizer: nnx.Optimizer
    visitation_counts: jnp.ndarray
    step: int


@gin.configurable
class RMaxNFQAgent:
    def __init__(
        self,
        num_states: int,
        num_actions: int,
        grid_size: int,
        num_iters: int = 10,
        hidden_units: int = 64,
        learning_rate: float = 1e-3,
        discount: float = 0.99,
        max_grad_norm: float = 1.0,
        seed: int = 0,
        vmax: float = 1.0,
        min_visits: int = 5,
    ):
        self.num_states = int(num_states)
        self.num_actions = int(num_actions)
        self.num_iters = int(num_iters)
        self.grid_size = int(grid_size)
        self.num_obs_ids = (grid_size ** 2) * 2 * 2 * 4
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.discount = discount
        self.max_grad_norm = max_grad_norm
        self.seed = int(seed)
        self.vmax = vmax
        self.min_visits = min_visits

    def initial_state(self) -> NFQState:
        rng = jax.random.PRNGKey(self.seed)
        rng_online, rng_target = jax.random.split(rng)
        online_network = Network(in_features=self.num_states, out_features=self.num_actions, rngs=nnx.Rngs(rng_online), hidden_features=self.hidden_units)
        optimizer = nnx.Optimizer(
            online_network,
            optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.adam(self.learning_rate),
            ),
            wrt=nnx.Param
        )
        visitation_counts = jnp.zeros((self.num_obs_ids, self.num_actions), dtype=jnp.int32)
        
        return NFQState(
            online_network=online_network,
            optimizer=optimizer,
            visitation_counts=visitation_counts,
            step=0,
        )


    # def obs_to_index(self, obs: jnp.ndarray) -> jnp.ndarray:
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


    def select_action(self, state: NFQState, obs: jnp.ndarray, key: jax.Array, is_training: bool) -> jnp.ndarray:
        obs_idx = self.obs_to_index(obs)
        q_vals = state.online_network(obs.reshape(-1))
        counts = state.visitation_counts[obs_idx]
        q_vals = jnp.where(counts < self.min_visits, self.vmax, q_vals)
        action_dist = distrax.Greedy(q_vals)
        return action_dist.sample(seed=key)[0]


    def update(self, state: NFQState, batch: Transition) -> tuple[NFQState, jax.Array]:
        next_q = state.online_network(batch.next_observation)
        max_next_q = jnp.max(next_q, axis=1)

        obs_idx = self.obs_to_index(batch.observation)
        next_obs_idx = self.obs_to_index(batch.next_observation)
        
        state = state.replace(
            visitation_counts=state.visitation_counts.at[obs_idx, batch.action].add(1)
        )

        batch_counts_minimum = jnp.array([state.visitation_counts[next_obs_idx[j], :].min() for j in range(batch.observation.shape[0])])
        max_next_q = jnp.where(batch_counts_minimum < self.min_visits, self.vmax, max_next_q)

        for i in range(self.num_iters):
            def loss_fn(network: Network):
                q_values = network(batch.observation)
                q_sel = jnp.take_along_axis(q_values, batch.action[:, None], axis=1).squeeze()
                target = batch.reward + self.discount * max_next_q * (1.0 - batch.terminal)

                # Note: maybe we do want to regress Q to Vmax if the current state-action pair is unknown
                known_mask = state.visitation_counts[obs_idx, batch.action] >= self.min_visits
                loss = jnp.sum(jnp.where(known_mask, (q_sel - jax.lax.stop_gradient(target)) ** 2, 0.0))/ jnp.sum(known_mask)

                # target = jnp.where(known_mask, target, self.vmax)
                # loss = jnp.mean((q_sel - jax.lax.stop_gradient(target)) ** 2)
                
                return loss
            loss, grads = nnx.value_and_grad(loss_fn)(state.online_network)
            state.optimizer.update(state.online_network, grads)

        return state, loss

    def bootstrap_value(self, state: NFQState, next_observation: jnp.ndarray) -> jax.Array:
        # TODO: implement this properly. Don't need to implement this yet, not using MC returns.
        return jnp.array(0.0)
