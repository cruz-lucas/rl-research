import jax
import jax.numpy as jnp


def obs_to_index(obs: jax.Array, grid_size: int) -> jax.Array:
    obs = obs.reshape((-1, grid_size, grid_size, 3))
    B, H, W, C = obs.shape

    player_mask = obs[:, :, :, 0] == 10
    _, prow, pcol = jnp.where(player_mask, size=B, fill_value=0)

    player_pos = (prow - 1) * (W - 2) + (pcol - 1)

    door_mask = obs[:, :, :, 0] == 4
    dbatch, drow, dcol = jnp.where(door_mask, size=B, fill_value=0)

    door_open = obs[dbatch, drow, dcol, -1] == 2

    key_mask = obs[:, :, :, 0] == 5
    _, krow, kcol = jnp.where(key_mask, size=B, fill_value=0)

    key_picked = kcol == -1
    key_pos = 1 + (krow - 1) * (W - 2) + (kcol - 1) # (W-2) * (H-2) + 1 possible positions
    key_pos = jnp.where(key_picked, 0, key_pos)

    direction = obs[jnp.arange(B), prow, pcol, -1]

    return jnp.int32(
        ((key_pos * (W - 2) * (H - 2) + player_pos) * 2 + door_open) * 4 + direction
    )