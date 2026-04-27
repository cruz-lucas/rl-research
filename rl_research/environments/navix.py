import re
from typing import Callable, Union

import gin
import jax
import jax.numpy as jnp
import navix as nx
import numpy as np
from flax import struct
from jax import Array
from navix import observations, rewards, terminations
from navix.actions import _can_walk_there
from navix.components import EMPTY_POCKET_ID
from navix.entities import Door, Goal, Key, Player, Wall
from navix.environments.empty import Room as EmptyRoom
from navix.environments.environment import Timestep
from navix.grid import room, translate
from navix.rendering.cache import TILE_SIZE, RenderingCache
from navix.rendering.registry import PALETTE
from navix.spaces import Continuous, Discrete, Space
from navix.states import State

from rl_research.environments import BaseJaxEnv


ACTION_NAMES = ("up", "down", "left", "right")
ACTION_TO_NAVIX_DIRECTION = {
    "up": 3,
    "down": 1,
    "left": 2,
    "right": 0,
}


def _move_absolute(state: State, direction: int) -> State:
    player = state.get_player(idx=0)
    absolute_direction = jnp.asarray(direction, dtype=jnp.int32)
    target_position = translate(player.position, absolute_direction)

    can_move, events = _can_walk_there(state, target_position)
    next_position = jnp.where(can_move, target_position, player.position)
    player = player.replace(position=next_position, direction=absolute_direction)

    return state.set_player(player).replace(events=events)


def _move_up(state: State) -> State:
    return _move_absolute(state, ACTION_TO_NAVIX_DIRECTION["up"])


def _move_down(state: State) -> State:
    return _move_absolute(state, ACTION_TO_NAVIX_DIRECTION["down"])


def _move_left(state: State) -> State:
    return _move_absolute(state, ACTION_TO_NAVIX_DIRECTION["left"])


def _move_right(state: State) -> State:
    return _move_absolute(state, ACTION_TO_NAVIX_DIRECTION["right"])


CARDINAL_ACTION_SET = (
    _move_up,
    _move_down,
    _move_left,
    _move_right,
)

ACTION_SETS = {
    "default": None,
    "cardinal": CARDINAL_ACTION_SET,
}


def position_obs_fn(state: State) -> Array:
    _, width = state.grid.shape
    row, col = state.get_player(idx=0).position
    return (row.astype(jnp.int32) * width + col.astype(jnp.int32)).reshape(1)


def onehot_position_obs_fn(state: State) -> Array:
    height, width = state.grid.shape
    idx = position_obs_fn(state).reshape(())
    return jax.nn.one_hot(idx, height * width, dtype=jnp.float32)


def _position_obs_space(height: int, width: int) -> Space:
    return Discrete.create(
        n_elements=height * width,
        shape=(1,),
        dtype=jnp.int32,
    )


def _onehot_position_obs_space(height: int, width: int) -> Space:
    return Discrete.create(
        n_elements=2,
        shape=(height * width,),
        dtype=jnp.float32,
    )


def _grid_shape_from_env_id(env_id: str) -> tuple[int, int]:
    match = re.search(r"(\d+)x(\d+)", env_id)
    if match is None:
        raise ValueError(
            "Could not infer grid shape from env_id "
            f"{env_id!r}. Use an env id containing '<height>x<width>'."
        )
    return int(match.group(1)), int(match.group(2))


def _observation_kwargs(
    observation_mode: str | None,
    *,
    height: int,
    width: int,
) -> dict:
    if observation_mode is None or observation_mode == "default":
        return {}

    if observation_mode == "symbolic":
        return {"observation_fn": observations.symbolic}
    if observation_mode in {"position", "tabular_position"}:
        return {
            "observation_fn": position_obs_fn,
            "observation_space": _position_obs_space(height, width),
        }
    if observation_mode in {"onehot_position", "tabular"}:
        return {
            "observation_fn": onehot_position_obs_fn,
            "observation_space": _onehot_position_obs_space(height, width),
        }

    raise ValueError(
        f"Unsupported observation_mode {observation_mode!r}. "
        "Choose from default, symbolic, position, or onehot_position."
    )


def _action_set_kwargs(action_set: str | None) -> dict:
    if action_set is None or action_set == "default":
        return {}
    try:
        resolved_action_set = ACTION_SETS[action_set]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported action_set {action_set!r}. "
            f"Choose from {', '.join(ACTION_SETS)}."
        ) from exc
    return {"action_set": resolved_action_set}


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

    key_picked = kcol == grid_size - 1
    key_pos = (
        1 + (krow - 1) * (W - 2) + (kcol - 1)
    )  # (W-2) * (H-2) + 1 possible positions
    key_pos = jnp.where(key_picked, 0, key_pos)

    direction = obs[jnp.arange(B), prow, pcol, -1]

    return jnp.int32(
        ((key_pos * (W - 2) * (H - 2) + player_pos) * 2 + door_open) * 4 + direction
    )


def tabular_obs_fn(state: State) -> Array:
    H, W = state.grid.shape

    prow, pcol = state.get_player().position
    player_pos = (prow - 1) * (W - 2) + (pcol - 1)  # (W-2) * (H-2) possible positions

    door_open = state.get_doors().open

    krow, kcol = state.get_keys().position[0]
    key_picked = kcol == -1
    key_pos = (
        1 + (krow - 1) * (W - 2) + (kcol - 1)
    )  # (W-2) * (H-2) + 1 possible positions
    key_pos = jnp.where(key_picked, 0, key_pos)

    direction = state.get_player().direction
    return jnp.int16(
        ((key_pos * (W - 2) * (H - 2) + player_pos) * 2 + door_open) * 4 + direction
    )[0]


def onehot_tabular_obs_fn(state: State) -> Array:
    H, W = state.grid.shape
    idx = tabular_obs_fn(state)
    return jnp.eye(((W - 2) * (H - 2) + 1) * (W - 2) * (H - 2) * 2 * 4)[idx]


def onehot_obs_fn(state: State) -> Array:
    H, W = state.grid.shape

    prow, pcol = state.get_player().position
    player_pos = (prow - 1) * (W - 2) + (pcol - 1)

    onehot_player_pos = jnp.eye((H - 2) * (W - 2))[player_pos]

    door_open = (state.get_doors().open) * 1

    onehot_door_open = jnp.eye(2)[door_open].reshape(-1)

    krow, kcol = state.get_keys().position[0]
    key_picked = kcol == -1
    key_pos = 1 + (krow - 1) * (W - 2) + (kcol - 1)
    key_pos = jnp.where(key_picked, 0, key_pos)

    onehot_key = jnp.eye((H - 2) * (W - 2) + 1)[key_pos]

    direction = state.get_player().direction
    onehot_direction = jnp.eye(4)[direction]
    return jnp.concatenate(
        [onehot_player_pos, onehot_key, onehot_door_open, onehot_direction]
    )


class FixedGridDoorKey(nx.environments.DoorKey):
    door_row: int = struct.field(pytree_node=False, default=1)
    door_col: int = struct.field(pytree_node=False, default=2)
    key_row: int = struct.field(pytree_node=False, default=2)
    key_col: int = struct.field(pytree_node=False, default=1)
    goal_row: int = struct.field(pytree_node=False, default=2)
    goal_col: int = struct.field(pytree_node=False, default=1)

    def _reset(self, key: Array, cache: Union[RenderingCache, None] = None) -> Timestep:
        # check minimum height and width
        assert self.height > 3, (
            f"Room height must be greater than 3, got {self.height} instead"
        )
        assert self.width > 4, (
            f"Room width must be greater than 5, got {self.width} instead"
        )

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
        key_pos = jnp.array(
            (self.key_row, self.key_col)
        )  # random_positions(k2, first_room, exclude=player_pos)
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
        elif observation_fn == onehot_obs_fn:
            return Discrete.create(
                n_elements=2,
                shape=(height * width + 2 + 2 + 4,),
                dtype=jnp.uint8,
            )
        elif observation_fn == position_obs_fn:
            return _position_obs_space(height, width)
        elif observation_fn == onehot_position_obs_fn:
            return _onehot_position_obs_space(height, width)
        else:
            raise NotImplementedError(
                "Unknown observation space for observation function {}".format(
                    observation_fn
                )
            )


@gin.configurable
class NavixWrapper(BaseJaxEnv):
    def __init__(
        self,
        env_id: str,
        max_steps: int = 100,
        action_set: str = "default",
        observation_mode: str = "default",
    ):
        height, width = _grid_shape_from_env_id(env_id)
        kwargs = {
            **_action_set_kwargs(action_set),
            **_observation_kwargs(observation_mode, height=height, width=width),
        }
        self.env = nx.make(
            env_id,
            max_steps=max_steps,
            **kwargs,
        )
        self.observation_shape = tuple(self.env.observation_space.shape)
        self.observation_dtype = np.dtype(self.env.observation_space.dtype)
        self.num_observation_states = (
            int(self.env.observation_space.n)
            if self.observation_shape in [(), (1,)]
            else int(np.prod(np.asarray(self.observation_shape)))
        )

    def reset(self, key: Array):
        timestep = self.env.reset(key)
        return timestep, timestep.observation

    def step(self, timestep: Timestep, action: Array):
        action = jnp.asarray(action).reshape(())
        timestep = self.env.step(timestep, action)
        return (
            timestep,
            timestep.observation,
            timestep.reward,
            timestep.is_termination(),
            timestep.is_truncation(),
            timestep.info,
        )  # observations.rgb(timestep.state) # set info to rgb for debugging purposes


class CardinalNavixWrapper(NavixWrapper):
    def __init__(
        self,
        env_id: str = "Navix-Empty-16x16-v0",
        max_steps: int = 1024,
        observation_mode: str = "symbolic",
    ):
        super().__init__(
            env_id=env_id,
            max_steps=max_steps,
            action_set="cardinal",
            observation_mode=observation_mode,
        )


def _register_empty_cardinal_envs(size: int = 16) -> None:
    env_name = f"Navix-Empty-{size}x{size}-Cardinal-v0"
    tabular_env_name = f"TabularNavix-Empty-{size}x{size}-Cardinal-v0"
    onehot_env_name = f"OneHotNavix-Empty-{size}x{size}-Cardinal-v0"

    nx.register_env(
        env_name,
        lambda *args, **kwargs: EmptyRoom.create(
            height=size,
            width=size,
            random_start=False,
            action_set=kwargs.pop("action_set", CARDINAL_ACTION_SET),
            observation_fn=kwargs.pop("observation_fn", observations.symbolic),
            reward_fn=kwargs.pop("reward_fn", rewards.on_goal_reached),
            termination_fn=kwargs.pop(
                "termination_fn",
                terminations.on_goal_reached,
            ),
            *args,
            **kwargs,
        ),
    )
    nx.register_env(
        tabular_env_name,
        lambda *args, **kwargs: EmptyRoom.create(
            height=size,
            width=size,
            random_start=False,
            action_set=kwargs.pop("action_set", CARDINAL_ACTION_SET),
            observation_fn=kwargs.pop("observation_fn", position_obs_fn),
            observation_space=kwargs.pop(
                "observation_space",
                _position_obs_space(size, size),
            ),
            reward_fn=kwargs.pop("reward_fn", rewards.on_goal_reached),
            termination_fn=kwargs.pop(
                "termination_fn",
                terminations.on_goal_reached,
            ),
            *args,
            **kwargs,
        ),
    )
    nx.register_env(
        onehot_env_name,
        lambda *args, **kwargs: EmptyRoom.create(
            height=size,
            width=size,
            random_start=False,
            action_set=kwargs.pop("action_set", CARDINAL_ACTION_SET),
            observation_fn=kwargs.pop("observation_fn", onehot_position_obs_fn),
            observation_space=kwargs.pop(
                "observation_space",
                _onehot_position_obs_space(size, size),
            ),
            reward_fn=kwargs.pop("reward_fn", rewards.on_goal_reached),
            termination_fn=kwargs.pop(
                "termination_fn",
                terminations.on_goal_reached,
            ),
            *args,
            **kwargs,
        ),
    )


_register_empty_cardinal_envs()


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
    ),
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
    ),
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
    ),
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
        max_steps=1024,
        random_start=False,
        # **kwargs,
    ),
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
        max_steps=1024,
        random_start=False,
        # **kwargs,
    ),
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
        max_steps=1024,
        random_start=False,
        # **kwargs,
    ),
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
    ),
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
    ),
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
    ),
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
        max_steps=1024,
        random_start=False,
        # **kwargs,
    ),
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
        max_steps=1024,
        random_start=False,
        # **kwargs,
    ),
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
        max_steps=1024,
        random_start=False,
        # **kwargs,
    ),
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
    ),
)


nx.register_env(
    "OneHotGridDoorKey-5x5-layout1-v0",
    lambda *args, **kwargs: FixedGridDoorKey.create(
        observation_fn=onehot_obs_fn,
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
    ),
)


nx.register_env(
    "OneHotGridDoorKey-16x16-layout1-v0",
    lambda *args, **kwargs: FixedGridDoorKey.create(
        observation_fn=onehot_obs_fn,
        reward_fn=nx.rewards.on_goal_reached,
        termination_fn=nx.terminations.on_goal_reached,
        height=16,
        width=16,
        door_row=1,
        door_col=13,
        goal_row=1,
        goal_col=14,
        max_steps=1024,
        random_start=False,
        # **kwargs,
    ),
)
