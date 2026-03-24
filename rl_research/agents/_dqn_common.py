import jax
import jax.numpy as jnp
from flax import nnx


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
