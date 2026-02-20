from typing import Union, Callable
import jax
import jax.numpy as jnp
from jax import Array
from flax import struct
import navix as nx
import gin

from navix.states import State
from navix.environments.environment import Timestep
from navix.entities import Player, Key, Door, Goal, Wall
from navix.rendering.registry import PALETTE
from navix.rendering.cache import RenderingCache, TILE_SIZE
from navix.components import EMPTY_POCKET_ID
from navix import rewards, observations, terminations
from navix.grid import mask_by_coordinates, room
from navix.spaces import Space, Discrete, Continuous

from rl_research.environments import BaseJaxEnv



def tabular_obs_fn(state: State) -> Array:
    H, W = state.grid.shape

    prow, pcol = state.get_player().position
    player_pos = (prow - 1) * (W - 2) + (pcol - 1)

    door_open = state.get_doors().open

    # check pocket != -1 might be a better solution
    krow, _ = state.get_keys().position[0]
    kpicked = krow == 0

    direction = state.get_player().direction
    return jnp.int16(((player_pos * 2 + door_open) * 2 + kpicked) * 4 + direction)[0]


def onehot_tabular_obs_fn(state: State) -> Array:
    H, W = state.grid.shape

    prow, pcol = state.get_player().position
    player_pos = (prow - 1) * (W - 2) + (pcol - 1)

    door_open = state.get_doors().open

    # check pocket != -1 might be a better solution
    krow, _ = state.get_keys().position[0]
    kpicked = krow == 0

    direction = state.get_player().direction
    idx = jnp.int16(((player_pos * 2 + door_open) * 2 + kpicked) * 4 + direction)[0]
    return jnp.eye(H * W * 2 * 2 * 4)[idx]


class FixedGridDoorKey(nx.environments.DoorKey):
    door_row: int = struct.field(pytree_node=False, default=1)
    door_col: int = struct.field(pytree_node=False, default=2)
    key_row: int = struct.field(pytree_node=False, default=2)
    key_col: int = struct.field(pytree_node=False, default=1)
    goal_row: int = struct.field(pytree_node=False, default=2)
    goal_col: int = struct.field(pytree_node=False, default=1)
    

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
        door_col = jnp.asarray(self.door_col)
        # row can be between 1 and height - 2
        door_row = jnp.asarray(self.door_row)
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
        player_pos = jnp.asarray([1, 1])
        player_dir = jnp.asarray(0)
        goal_pos = jnp.asarray([self.goal_row, self.goal_col])

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
    
    @staticmethod
    def _get_obs_space_from_fn(
        width: int, height: int, observation_fn: Callable[[State], Array]
    ) -> Space:
        if observation_fn == observations.none:
            return Continuous.create(
                shape=(), minimum=jnp.asarray(0.0), maximum=jnp.asarray(0.0)
            )
        elif observation_fn == observations.categorical:
            return Discrete.create(n_elements=9, shape=(height, width))
        elif observation_fn == observations.categorical_first_person:
            radius = observations.RADIUS
            return Discrete.create(n_elements=9, shape=(radius * 2 + 1, radius * 2 + 1))
        elif observation_fn == observations.rgb:
            return Discrete.create(
                256,
                shape=(height * TILE_SIZE, width * TILE_SIZE, 3),
                dtype=jnp.uint8,
            )
        elif observation_fn == observations.rgb_first_person:
            radius = observations.RADIUS
            return Discrete.create(
                n_elements=256,
                shape=((radius * 2 + 1) * TILE_SIZE, (radius * 2 + 1) * TILE_SIZE, 3),
                dtype=jnp.uint8,
            )
        elif observation_fn == observations.symbolic:
            return Discrete.create(
                n_elements=9,
                shape=(height, width, 3),
                dtype=jnp.uint8,
            )
        elif observation_fn == observations.symbolic_first_person:
            radius = observations.RADIUS
            return Discrete.create(
                n_elements=256,
                shape=(radius * 2 + 1, radius * 2 + 1, 3),
                dtype=jnp.uint8,
            )
        elif observation_fn == tabular_obs_fn:
            return Discrete.create(
                n_elements=height * width * 2 * 2 * 4,
                shape=(1,),
                dtype=jnp.uint8,
            )
        elif observation_fn == onehot_tabular_obs_fn:
            return Discrete.create(
                n_elements=2,
                shape=(height * width * 2 * 2 * 4,),
                dtype=jnp.uint8,
            )
        else:
            raise NotImplementedError(
                "Unknown observation space for observation function {}".format(
                    observation_fn
                )
            )


@gin.configurable
class NavixWrapper(BaseJaxEnv):
    def __init__(self, env_id: str):
        self.env = nx.make(
            env_id,
        )

    def reset(self, key: Array):
        timestep = self.env.reset(key)
        return timestep, timestep.observation

    def step(self, timestep: Timestep, action: Array):
        timestep = self.env.step(timestep, action)
        return timestep, timestep.observation, timestep.reward, timestep.is_termination(), timestep.is_truncation(), timestep.info  # observations.rgb(timestep.state) # set info to rgb for debugging purposes


# TODO: fix and submit PR to remove max_steps default value.
nx.register_env(
    "TabularGridDoorKey-5x5-layout1-v0",
    lambda *args, **kwargs: FixedGridDoorKey.create(
    observation_fn=tabular_obs_fn,
    reward_fn=nx.rewards.on_goal_reached,
    termination_fn=nx.terminations.on_goal_reached,
    height=5,
    width=5,
    door_row=1,
    goal_row=1,
    goal_col=3,
    max_steps=100,
    random_start=False,
    # **kwargs,
    )
)

nx.register_env(
    "TabularGridDoorKey-5x5-layout2-v0",
    lambda *args, **kwargs: FixedGridDoorKey.create(
    observation_fn=tabular_obs_fn,
    reward_fn=nx.rewards.on_goal_reached,
    termination_fn=nx.terminations.on_goal_reached,
    height=5,
    width=5,
    door_row=3,
    goal_row=3,
    goal_col=3,
    max_steps=100,
    random_start=False,
    # **kwargs,
    )
)

nx.register_env(
    "TabularGridDoorKey-5x5-layout3-v0",
    lambda *args, **kwargs: FixedGridDoorKey.create(
    observation_fn=tabular_obs_fn,
    reward_fn=nx.rewards.on_goal_reached,
    termination_fn=nx.terminations.on_goal_reached,
    height=5,
    width=5,
    door_row=3,
    goal_row=1,
    goal_col=3,
    max_steps=100,
    random_start=False,
    # **kwargs,
    )
)

nx.register_env(
    "TabularGridDoorKey-16x16-layout1-v0",
    lambda *args, **kwargs: FixedGridDoorKey.create(
    observation_fn=tabular_obs_fn,
    reward_fn=nx.rewards.on_goal_reached,
    termination_fn=nx.terminations.on_goal_reached,
    height=16,
    width=16,
    door_row=1,
    door_col=13,
    goal_row=1,
    goal_col=14,
    max_steps=2048,
    random_start=False,
    # **kwargs,
    )
)

nx.register_env(
    "TabularGridDoorKey-16x16-layout2-v0",
    lambda *args, **kwargs: FixedGridDoorKey.create(
    observation_fn=tabular_obs_fn,
    reward_fn=nx.rewards.on_goal_reached,
    termination_fn=nx.terminations.on_goal_reached,
    height=16,
    width=16,
    door_row=14,
    door_col=13,
    goal_row=14,
    goal_col=14,
    max_steps=2048,
    random_start=False,
    # **kwargs,
    )
)

nx.register_env(
    "TabularGridDoorKey-16x16-layout3-v0",
    lambda *args, **kwargs: FixedGridDoorKey.create(
    observation_fn=tabular_obs_fn,
    reward_fn=nx.rewards.on_goal_reached,
    termination_fn=nx.terminations.on_goal_reached,
    height=16,
    width=16,
    door_row=14,
    door_col=13,
    goal_row=1,
    goal_col=14,
    max_steps=2048,
    random_start=False,
    # **kwargs,
    )
)



nx.register_env(
    "GridDoorKey-5x5-layout1-v0",
    lambda *args, **kwargs: FixedGridDoorKey.create(
    observation_fn=nx.observations.symbolic,
    reward_fn=nx.rewards.on_goal_reached,
    termination_fn=nx.terminations.on_goal_reached,
    height=5,
    width=5,
    door_row=1,
    goal_row=1,
    goal_col=3,
    max_steps=100,
    random_start=False,
    # **kwargs,
    )
)


nx.register_env(
    "GridDoorKey-5x5-layout2-v0",
    lambda *args, **kwargs: FixedGridDoorKey.create(
    observation_fn=nx.observations.symbolic,
    reward_fn=nx.rewards.on_goal_reached,
    termination_fn=nx.terminations.on_goal_reached,
    height=5,
    width=5,
    door_row=3,
    goal_row=3,
    goal_col=3,
    max_steps=100,
    random_start=False,
    # **kwargs,
    )
)

nx.register_env(
    "GridDoorKey-5x5-layout3-v0",
    lambda *args, **kwargs: FixedGridDoorKey.create(
    observation_fn=nx.observations.symbolic,
    reward_fn=nx.rewards.on_goal_reached,
    termination_fn=nx.terminations.on_goal_reached,
    height=5,
    width=5,
    door_row=3,
    goal_row=1,
    goal_col=3,
    max_steps=100,
    random_start=False,
    # **kwargs,
    )
)

nx.register_env(
    "GridDoorKey-16x16-layout1-v0",
    lambda *args, **kwargs: FixedGridDoorKey.create(
    observation_fn=nx.observations.symbolic,
    reward_fn=nx.rewards.on_goal_reached,
    termination_fn=nx.terminations.on_goal_reached,
    height=16,
    width=16,
    door_row=1,
    door_col=13,
    goal_row=1,
    goal_col=14,
    max_steps=2048,
    random_start=False,
    # **kwargs,
    )
)

nx.register_env(
    "GridDoorKey-16x16-layout2-v0",
    lambda *args, **kwargs: FixedGridDoorKey.create(
    observation_fn=nx.observations.symbolic,
    reward_fn=nx.rewards.on_goal_reached,
    termination_fn=nx.terminations.on_goal_reached,
    height=16,
    width=16,
    door_row=14,
    door_col=13,
    goal_row=14,
    goal_col=14,
    max_steps=2048,
    random_start=False,
    # **kwargs,
    )
)

nx.register_env(
    "GridDoorKey-16x16-layout3-v0",
    lambda *args, **kwargs: FixedGridDoorKey.create(
    observation_fn=nx.observations.symbolic,
    reward_fn=nx.rewards.on_goal_reached,
    termination_fn=nx.terminations.on_goal_reached,
    height=16,
    width=16,
    door_row=14,
    door_col=13,
    goal_row=1,
    goal_col=14,
    max_steps=2048,
    random_start=False,
    # **kwargs,
    )
)

nx.register_env(
    "OneHotTabularGridDoorKey-5x5-layout1-v0",
    lambda *args, **kwargs: FixedGridDoorKey.create(
    observation_fn=onehot_tabular_obs_fn,
    reward_fn=nx.rewards.on_goal_reached,
    termination_fn=nx.terminations.on_goal_reached,
    height=5,
    width=5,
    door_row=1,
    goal_row=1,
    goal_col=3,
    max_steps=100,
    random_start=False,
    # **kwargs,
    )
)