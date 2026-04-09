from collections.abc import Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn


def resolve_activation(name: str):
    normalized = name.strip().lower()
    if normalized == "relu":
        return jax.nn.relu
    if normalized == "gelu":
        return jax.nn.gelu
    if normalized == "silu":
        return jax.nn.silu
    if normalized == "tanh":
        return jnp.tanh
    raise ValueError(
        f"Unsupported activation {name!r}. Choose from relu, gelu, silu, tanh."
    )


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    output_dim: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        activation_fn = resolve_activation(self.activation)

        x = jnp.asarray(x, dtype=jnp.float32)
        squeeze_output = False
        if x.ndim == 1:
            x = x[None, :]
            squeeze_output = True
        x = x.reshape((x.shape[0], -1))

        for width in self.hidden_dims:
            x = nn.Dense(int(width))(x)
            x = activation_fn(x)

        x = nn.Dense(int(self.output_dim))(x)
        if squeeze_output:
            return x[0]
        return x
