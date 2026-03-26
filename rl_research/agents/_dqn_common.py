import jax
import jax.numpy as jnp
from flax import nnx, struct


class MLPNetwork(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rngs: nnx.Rngs,
        hidden_features: int = 64,
    ):
        self.in_layer = nnx.Linear(
            in_features=in_features,
            out_features=hidden_features,
            rngs=rngs,
        )
        self.hidden_layer = nnx.Linear(
            in_features=hidden_features,
            out_features=hidden_features,
            rngs=rngs,
        )
        self.layernorm = nnx.LayerNorm(num_features=hidden_features, rngs=rngs)
        self.out_layer = nnx.Linear(
            in_features=hidden_features,
            out_features=out_features,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.asarray(x, dtype=jnp.float32)
        x = self.in_layer(x)
        x = nnx.relu(x)
        x = self.hidden_layer(x)
        x = self.layernorm(x)
        x = nnx.relu(x)
        return self.out_layer(x)


class ObservationNormalizerState(struct.PyTreeNode):
    mean: jax.Array
    var: jax.Array
    count: jax.Array


def init_observation_normalizer(
    num_features: int,
    initial_count: float = 1e-4,
) -> ObservationNormalizerState:
    return ObservationNormalizerState(
        mean=jnp.zeros((num_features,), dtype=jnp.float32),
        var=jnp.ones((num_features,), dtype=jnp.float32),
        count=jnp.asarray(initial_count, dtype=jnp.float32),
    )


def update_observation_normalizer(
    state: ObservationNormalizerState,
    observations: jax.Array,
) -> ObservationNormalizerState:
    observations = jnp.asarray(observations, dtype=jnp.float32).reshape(
        -1, state.mean.shape[0]
    )
    batch_mean = jnp.mean(observations, axis=0)
    batch_var = jnp.var(observations, axis=0)
    batch_count = jnp.asarray(observations.shape[0], dtype=jnp.float32)

    delta = batch_mean - state.mean
    total_count = state.count + batch_count

    new_mean = state.mean + delta * (batch_count / total_count)
    m_a = state.var * state.count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + jnp.square(delta) * (state.count * batch_count / total_count)
    new_var = m2 / total_count

    return state.replace(mean=new_mean, var=new_var, count=total_count)


def normalize_observation(
    observation: jax.Array,
    state: ObservationNormalizerState,
    epsilon: float = 1e-8,
    clip: float | None = 5.0,
) -> jax.Array:
    normalized = (
        jnp.asarray(observation, dtype=jnp.float32) - state.mean
    ) / jnp.sqrt(jnp.maximum(state.var, epsilon))
    if clip is not None:
        normalized = jnp.clip(normalized, -clip, clip)
    return normalized


def clone_module(module: nnx.Module) -> nnx.Module:
    graphdef, module_state = nnx.split(module)
    return nnx.merge(graphdef, module_state)


def linear_epsilon(
    step: int | jax.Array,
    eps_start: float,
    eps_end: float,
    eps_decay_steps: int,
) -> jax.Array:
    frac = jnp.clip(step / max(1, eps_decay_steps), 0.0, 1.0)
    return eps_start + frac * (eps_end - eps_start)


def hard_update_network(
    source: nnx.Module,
    target: nnx.Module,
    should_update: bool | jax.Array,
) -> None:
    _, source_state = nnx.split(source)
    _, target_state = nnx.split(target)
    new_target_state = jax.tree.map(
        lambda source_value, target_value: jnp.where(
            should_update, source_value, target_value
        ),
        source_state,
        target_state,
    )
    nnx.update(target, new_target_state)
