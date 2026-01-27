import jax
import jax.numpy as jnp
import navix as nx
from navix.environments.door_key import *


class FixedGridDoorKey(nx.environments.DoorKey):
    door_row: int = struct.field(pytree_node=False, default=1)
    door_col: int = struct.field(pytree_node=False, default=2)
    key_row: int = struct.field(pytree_node=False, default=2)
    key_col: int = struct.field(pytree_node=False, default=1)
    
    def encode_state(self, timestep: nx.Timestep) -> int:
        state = timestep.state

        prow, pcol = state.get_player().position
        player_pos = (prow - 1) * (self.width - 2) + (pcol - 1)

        door_open = state.get_doors().open

        # check pocket != -1
        krow, _ = state.get_keys().position[0]
        kpicked = krow == 0

        direction = state.get_player().direction
        return jnp.int16(((player_pos * 2 + door_open) * 2 + kpicked) * 4 + direction)[0]

    def _reset(self, key: Array, cache: Union[RenderingCache, None] = None) -> Timestep:
        # check minimum height and width
        assert (
            self.height > 3
        ), f"Room height must be greater than 3, got {self.height} instead"
        assert (
            self.width > 4
        ), f"Room width must be greater than 5, got {self.width} instead"

        key, k1, k2, k3, k4 = jax.random.split(key, 5)

        grid = room(height=self.height, width=self.width)

        # door positions
        # col can be between 1 and height - 2
        door_col = self.door_col
        # row can be between 1 and height - 2
        door_row = self.door_row
        door_pos = jnp.asarray((door_row, door_col))
        doors = Door.create(
            position=door_pos,
            requires=jnp.asarray(3),
            open=jnp.asarray(False),
            colour=PALETTE.YELLOW,
        )

        # wall positions
        wall_rows = jnp.arange(1, self.height - 1)
        wall_cols = jnp.asarray([door_col] * (self.height - 2))
        wall_pos = jnp.stack((wall_rows, wall_cols), axis=1)
        # remove wall where the door is
        wall_pos = jnp.delete(
            wall_pos, door_row - 1, axis=0, assume_unique_indices=True
        )
        walls = Wall.create(position=wall_pos)

        # get rooms
        first_room_mask = mask_by_coordinates(
            grid, (jnp.asarray(self.height), door_col), jnp.less
        )
        first_room = jnp.where(first_room_mask, grid, -1)  # put walls where not mask
        second_room_mask = mask_by_coordinates(
            grid, (jnp.asarray(0), door_col), jnp.greater
        )
        second_room = jnp.where(second_room_mask, grid, -1)  # put walls where not mask

        # set player and goal pos
        if self.random_start:
            player_pos = random_positions(k1, first_room)
            player_dir = random_directions(k2)
            goal_pos = random_positions(k2, second_room)
        else:
            player_pos = jnp.asarray([1, 1])
            player_dir = jnp.asarray(0)
            goal_pos = jnp.asarray([self.height - 2, self.width - 2])

        # spawn goal and player
        player = Player.create(
            position=player_pos, direction=player_dir, pocket=EMPTY_POCKET_ID
        )
        goals = Goal.create(position=goal_pos, probability=jnp.asarray(1.0))

        # spawn key
        key_pos = jnp.array((self.key_row, self.key_col))#random_positions(k2, first_room, exclude=player_pos)
        keys = Key.create(position=key_pos, id=jnp.asarray(3), colour=PALETTE.YELLOW)

        # remove the wall beneath the door
        grid = grid.at[tuple(door_pos)].set(0)

        entities = {
            "player": player[None],
            "key": keys[None],
            "door": doors[None],
            "goal": goals[None],
            "wall": walls,
        }

        state = State(
            key=key,
            grid=grid,
            cache=cache or RenderingCache.init(grid),
            entities=entities,
        )
        return Timestep(
            t=jnp.asarray(0, dtype=jnp.int32),
            observation=self.observation_fn(state),
            action=jnp.asarray(-1, dtype=jnp.int32),
            reward=jnp.asarray(0.0, dtype=jnp.float32),
            step_type=jnp.asarray(0, dtype=jnp.int32),
            state=state,
        )
    

nx.register_env(
    "FixedGridDoorKey-5x5-layout1-v0",
    lambda *args, **kwargs: FixedGridDoorKey.create(
    observation_fn=nx.observations.categorical,
    reward_fn=nx.rewards.on_goal_reached,
    termination_fn=nx.terminations.on_goal_reached,
    height=5,
    width=5,
    door_row=1,
    random_start=False,
    **kwargs,
    )
)

nx.register_env(
    "FixedGridDoorKey-5x5-layout2-v0",
    lambda *args, **kwargs: FixedGridDoorKey.create(
    observation_fn=nx.observations.categorical,
    reward_fn=nx.rewards.on_goal_reached,
    termination_fn=nx.terminations.on_goal_reached,
    height=5,
    width=5,
    door_row=2,
    random_start=False,
    **kwargs,
    )
)

nx.register_env(
    "FixedGridDoorKey-5x5-layout3-v0",
    lambda *args, **kwargs: FixedGridDoorKey.create(
    observation_fn=nx.observations.categorical,
    reward_fn=nx.rewards.on_goal_reached,
    termination_fn=nx.terminations.on_goal_reached,
    height=5,
    width=5,
    door_row=3,
    random_start=False,
    **kwargs,
    )
)

nx.register_env(
    "FixedGridDoorKey-16x16-layout1-v0",
    lambda *args, **kwargs: FixedGridDoorKey.create(
    observation_fn=nx.observations.categorical,
    reward_fn=nx.rewards.on_goal_reached,
    termination_fn=nx.terminations.on_goal_reached,
    height=16,
    width=16,
    door_row=14,
    door_col=13,
    random_start=False,
    **kwargs,
    )
)