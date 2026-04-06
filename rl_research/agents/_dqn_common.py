from collections.abc import Sequence

import jax
import jax.numpy as jnp
import optax
from flax import nnx, struct


class IdentityLayer(nnx.Module):
    def __call__(self, x: jax.Array) -> jax.Array:
        return x


def coerce_hidden_dims(
    hidden_features: int | Sequence[int],
    hidden_dims: Sequence[int] | None = None,
) -> tuple[int, ...]:
    dims_source = hidden_features if hidden_dims is None else hidden_dims
    if isinstance(dims_source, Sequence) and not isinstance(dims_source, (str, bytes)):
        dims = tuple(int(dim) for dim in dims_source)
    else:
        width = int(dims_source)
        dims = (width, width)

    if not dims:
        raise ValueError("hidden_dims must contain at least one hidden layer.")
    if any(dim <= 0 for dim in dims):
        raise ValueError(f"hidden_dims must be positive, got {dims}.")
    return dims


def apply_activation(x: jax.Array, activation: str) -> jax.Array:
    activation = activation.lower()
    if activation == "relu":
        return jax.nn.relu(x)
    if activation == "gelu":
        return jax.nn.gelu(x)
    if activation == "silu":
        return jax.nn.silu(x)
    if activation == "tanh":
        return jnp.tanh(x)
    raise ValueError(
        f"Unsupported activation {activation!r}. Choose from relu, gelu, silu, tanh."
    )


def build_optimizer_transform(
    learning_rate: float,
    max_grad_norm: float,
    optimizer: str = "adam",
    optimizer_beta1: float = 0.9,
    optimizer_beta2: float = 0.999,
    optimizer_epsilon: float = 1e-8,
    optimizer_weight_decay: float = 0.0001,
    optimizer_momentum: float = 0.0,
    optimizer_decay: float = 0.9,
    optimizer_centered: bool = False,
) -> optax.GradientTransformationExtraArgs:
    optimizer = optimizer.lower()
    momentum = optimizer_momentum if optimizer_momentum > 0.0 else None

    if optimizer == "adam":
        update_rule = optax.adam(
            learning_rate=learning_rate,
            b1=optimizer_beta1,
            b2=optimizer_beta2,
            eps=optimizer_epsilon,
        )
    elif optimizer == "adamw":
        update_rule = optax.adamw(
            learning_rate=learning_rate,
            b1=optimizer_beta1,
            b2=optimizer_beta2,
            eps=optimizer_epsilon,
            weight_decay=optimizer_weight_decay,
        )
    elif optimizer == "rmsprop":
        update_rule = optax.rmsprop(
            learning_rate=learning_rate,
            decay=optimizer_decay,
            eps=optimizer_epsilon,
            centered=optimizer_centered,
            momentum=momentum,
        )
    elif optimizer == "sgd":
        update_rule = optax.sgd(
            learning_rate=learning_rate,
            momentum=momentum,
        )
    else:
        raise ValueError(
            f"Unsupported optimizer {optimizer!r}. "
            "Choose from adam, adamw, rmsprop, sgd."
        )

    if max_grad_norm > 0.0:
        return optax.chain(optax.clip_by_global_norm(max_grad_norm), update_rule)
    return update_rule


def temporal_difference_loss(
    td_error: jax.Array,
    loss_type: str,
    huber_delta: float,
) -> jax.Array:
    loss_type = loss_type.lower()
    if loss_type == "mse":
        return jnp.square(td_error)
    if loss_type == "huber":
        return optax.huber_loss(td_error, delta=huber_delta)
    raise ValueError(f"Unsupported loss_type {loss_type!r}. Choose from mse or huber.")


class MLPNetwork(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rngs: nnx.Rngs,
        hidden_features: int | Sequence[int] = 64,
        hidden_dims: Sequence[int] | None = None,
        activation: str = "relu",
        normalization: str = "last",
    ):
        self.hidden_dims = coerce_hidden_dims(
            hidden_features=hidden_features,
            hidden_dims=hidden_dims,
        )
        self.activation = activation.lower()
        self.normalization = normalization.lower()
        if self.normalization not in {"none", "last", "all"}:
            raise ValueError(
                f"Unsupported normalization {normalization!r}. "
                "Choose from none, last, all."
            )

        self.hidden_layers = nnx.List()
        self.normalization_layers = nnx.List()
        prev_features = in_features
        num_hidden_layers = len(self.hidden_dims)
        for idx, width in enumerate(self.hidden_dims):
            self.hidden_layers.append(
                nnx.Linear(
                    in_features=prev_features,
                    out_features=width,
                    rngs=rngs,
                )
            )
            use_norm = self.normalization == "all" or (
                self.normalization == "last" and idx == num_hidden_layers - 1
            )
            self.normalization_layers.append(
                nnx.LayerNorm(num_features=width, rngs=rngs)
                if use_norm
                else IdentityLayer()
            )
            prev_features = width
        self.out_layer = nnx.Linear(
            in_features=prev_features,
            out_features=out_features,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.asarray(x, dtype=jnp.float32)
        for hidden_layer, normalization_layer in zip(
            self.hidden_layers, self.normalization_layers
        ):
            x = hidden_layer(x)
            x = normalization_layer(x)
            x = apply_activation(x, self.activation)
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
    normalized = (jnp.asarray(observation, dtype=jnp.float32) - state.mean) / jnp.sqrt(
        jnp.maximum(state.var, epsilon)
    )
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
