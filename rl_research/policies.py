import jax
import jax.numpy as jnp
import jax.random as jrandom


def _break_ties_randomly(values: jnp.ndarray, key: jax.Array) -> jax.Array:
    """Break ties randomly among maximum values."""
    max_val = jnp.max(values)
    mask = values == max_val
    probs = mask / jnp.sum(mask)
    return jrandom.choice(key, values.shape[0], p=probs)


def _select_greedy(q_values: jnp.ndarray, key: jax.Array) -> jax.Array:
    """Select greedy action with random tie-breaking."""
    return _break_ties_randomly(q_values, key)


def _select_random(q_values: jnp.ndarray, key: jax.Array) -> jax.Array:
    """Random action selection."""
    return jrandom.choice(key, q_values.size)


def _select_epsilon_greedy(
    q_values: jnp.ndarray,
    epsilon: float,
    key: jax.Array,
) -> jax.Array:
    """Epsilon-greedy action selection."""
    k1, k2, k3 = jrandom.split(key, 3)

    random_action = _select_random(q_values, k1)
    greedy_action = _select_greedy(q_values, k2)

    use_random = jrandom.uniform(k3) < epsilon
    return jnp.where(use_random, random_action, greedy_action)


def _select_ucb(
    q_values: jnp.ndarray,
    visit_counts: jnp.ndarray,
    total_visits: jax.Array,
    ucb_c: float,
    key: jax.Array,
) -> jax.Array:
    """UCB action selection."""
    safe_counts = jnp.maximum(visit_counts, 1)
    exploration_bonus = ucb_c * jnp.sqrt(jnp.log(total_visits + 1) / safe_counts)

    exploration_bonus = jnp.where(visit_counts == 0, 1e6, exploration_bonus)

    ucb_values = q_values + exploration_bonus
    return _select_greedy(ucb_values, key)


__all__ = [
    "_select_greedy",
    "_select_random",
    "_select_epsilon_greedy",
    "_select_ucb",
]
