"""Go-Right environment implemented with the Gymnasium functional JAX API."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import TypeAlias

import gin
import jax
import jax.numpy as jnp
import jax.random as jrng
import numpy as np
from flax import struct
from gymnasium import spaces
from gymnasium.experimental.functional import FuncEnv


try:
    import pygame
except ModuleNotFoundError:  # pragma: no cover - optional dependency.
    pygame = None


PRNGKey: TypeAlias = jax.Array

_BG_COLOR = (20, 20, 28)
_TRACK_COLOR = (64, 66, 82)
_TRACK_BORDER = (110, 110, 130)
_AGENT_COLOR = (236, 197, 82)
_CHECKPOINT_COLOR = (91, 135, 211)
_PRIZE_ACTIVE = (255, 215, 0)
_PRIZE_INACTIVE = (90, 90, 90)
_TEXT_COLOR = (228, 228, 228)
_STATUS_COLORS = {
    0: (216, 94, 94),
    1: (94, 200, 132),
    2: (97, 119, 216),
}
_BOUND_COLORS = {
    "lower": (216, 94, 94),
    "expected": (97, 119, 216),
    "upper": (94, 200, 132),
}


def _load_pygame():
    if pygame is None:
        raise ModuleNotFoundError(
            "pygame is required to render the GoRight environment. "
            "Install it with `pip install pygame`."
        )
    return pygame


@struct.dataclass
class EnvState:
    """Stateless Go-Right environment state."""

    position: jax.Array
    previous_status: jax.Array
    status: jax.Array
    prize_inds: jax.Array
    rng: PRNGKey


@gin.configurable
@struct.dataclass
class EnvParams:
    """Static parameters for the Go-Right environment."""

    length: int = 21
    num_indicators: int = 2
    num_actions: int = 2
    first_checkpoint: int = 10
    first_reward: float = 3.0
    second_checkpoint: int = 20
    second_reward: float = 6.0
    is_partially_obs: bool = False
    mapping: str = "default"


class GoRightFunctional(
    FuncEnv[
        EnvState,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        None,
        EnvParams,
    ]
):
    """Go-Right environment expressed through the functional Gymnasium API."""

    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        params: EnvParams | None = None,
        use_precomputed: bool = False,
        transition_cache_path: str | None = None,
        force_recompute_cache: bool = False,
    ):
        """Initialise the functional environment."""
        self.default_params = self.get_default_params()
        self.params = self.default_params if params is None else params

        self._status_mapping = jnp.asarray(
            [
                [1, 0, 1],
                [2, 2, 2],
                [0, 1, 0],
            ],
            dtype=jnp.int32,
        )

        self._fully_obs_shape = (
            self.params.length,
            3,
            3,
            2,
            2,
        )

        self._partially_obs_shape = (
            self.params.length,
            3,
            2,
            2,
        )

        self.obs_shape = (
            self._partially_obs_shape
            if self.params.is_partially_obs
            else self._fully_obs_shape
        )
        self.action_space = spaces.Discrete(self.params.num_actions)
        self.observation_space = spaces.Discrete(int(math.prod(self.obs_shape)))

        self._use_precomputed = use_precomputed
        if self._use_precomputed:
            self._cache_path = self._resolve_cache_path(transition_cache_path)
            next_states, rewards, components = self._load_or_compute_cache(
                force_recompute_cache
            )
            self._precomputed_next_states = jnp.asarray(next_states, dtype=jnp.int32)
            self._precomputed_rewards = jnp.asarray(rewards, dtype=jnp.float32)
            self._state_components = jnp.asarray(components, dtype=jnp.int32)
            self._num_states = self._precomputed_next_states.shape[0]
        else:
            self._cache_path = None
            self._precomputed_next_states = None
            self._precomputed_rewards = None
            self._state_components = None
            self._num_states = None

    def initial(self, rng: PRNGKey, params: EnvParams | None = None) -> EnvState:
        """Sample an initial state."""
        next_rng, key_status, key_prev_status = jrng.split(rng, 3)

        status_levels = jnp.arange(3, dtype=jnp.int32)
        status = jrng.choice(key_status, status_levels)
        previous_status = jrng.choice(key_prev_status, status_levels)
        prize_inds = jnp.zeros((self.params.num_indicators,), dtype=jnp.int32)

        return EnvState(
            position=jnp.asarray(0, dtype=jnp.int32),
            previous_status=previous_status,
            status=status,
            prize_inds=prize_inds,
            rng=next_rng,
        )

    def observation(
        self, state: EnvState, params: EnvParams | None = None
    ) -> jax.Array:
        """Return the discrete observation index."""
        fully_obs = jnp.ravel_multi_index(
            jnp.array(
                [
                    state.position,
                    state.previous_status,
                    state.status,
                    state.prize_inds[0],
                    state.prize_inds[1],
                ]
            ),
            dims=self._fully_obs_shape,
            mode="clip",
        )

        partially_obs = jnp.ravel_multi_index(
            jnp.array(
                [
                    state.position,
                    state.status,
                    state.prize_inds[0],
                    state.prize_inds[1],
                ]
            ),
            dims=self._partially_obs_shape,
            mode="clip",
        )

        return (
            1 - self.params.is_partially_obs
        ) * fully_obs + self.params.is_partially_obs * partially_obs

    def state_info(
        self, state: EnvState, params: EnvParams | None = None
    ) -> dict[str, jax.Array]:
        """Diagnostic information for the given state."""
        return {
            "position": jnp.asarray(state.position, dtype=jnp.int32),
            "previous_status": jnp.asarray(state.previous_status, dtype=jnp.int32),
            "status": jnp.asarray(state.status, dtype=jnp.int32),
            "prize_inds": jnp.asarray(state.prize_inds, dtype=jnp.int32),
        }

    def _next_status(self, state: EnvState) -> jax.Array:
        """Compute the next status deterministically."""
        prev_status = jnp.asarray(state.previous_status, dtype=jnp.int32)
        curr_status = jnp.asarray(state.status, dtype=jnp.int32)
        row = jnp.take(self._status_mapping, prev_status, axis=0, mode="clip")
        return jnp.take(row, curr_status, axis=0, mode="clip")

    @staticmethod
    def _shift_prizes(prize_inds: jax.Array, params: EnvParams) -> jax.Array:
        """Shift the prize indicator vector according to the environment rules."""
        pad = jnp.where(jnp.any(prize_inds != 0), 0, 1)
        padded = jnp.concatenate(
            [
                jnp.asarray(prize_inds, dtype=jnp.int32),
                jnp.asarray([pad], dtype=jnp.int32),
            ],
            axis=0,
        )
        rolled = jnp.roll(padded, 1, axis=0)
        return rolled[: params.num_indicators]

    def _next_prize_inds(
        self,
        state: EnvState,
        action: jax.Array,
        next_status: jax.Array,
        params: EnvParams,
    ) -> jax.Array:
        """Compute the next prize indicator values."""
        prize = jnp.asarray(state.prize_inds, dtype=jnp.int32)
        is_jackpot = jnp.equal(next_status, 2)
        is_right = jnp.equal(jnp.asarray(action, dtype=jnp.int32), 1)
        triggered_now = jnp.all(prize != 0)

        position = jnp.asarray(state.position, dtype=jnp.int32)
        first_cp = params.first_checkpoint
        second_cp = params.second_checkpoint
        last_pos = params.length - 1

        is_before_first = jnp.equal(position, first_cp - 1)
        is_before_second = jnp.equal(position, second_cp - 1)
        is_before_pos = jnp.logical_or(is_before_first, is_before_second)

        is_first = jnp.equal(position, first_cp)
        is_second = jnp.equal(position, second_cp)
        is_prize_pos = jnp.logical_or(is_first, is_second)

        is_entering = jnp.logical_and(
            is_right, jnp.logical_and(is_before_pos, is_jackpot)
        )
        is_continuing = jnp.logical_and(
            triggered_now,
            jnp.logical_and(is_right, is_prize_pos),
        )
        is_triggering = jnp.logical_or(is_entering, is_continuing)

        is_at_end = jnp.equal(position, last_pos)
        needs_shifting = jnp.logical_and(
            jnp.logical_and(is_at_end, is_right), jnp.logical_not(is_triggering)
        )

        shifted = self._shift_prizes(prize, params)
        ones_like = jnp.ones_like(prize, dtype=jnp.int32)
        zeros_like = jnp.zeros_like(prize, dtype=jnp.int32)

        return jnp.where(
            is_triggering,
            ones_like,
            jnp.where(needs_shifting, shifted, zeros_like),
        )

    def transition(
        self,
        state: EnvState,
        action: jax.Array,
        params: EnvParams | None = None,
    ) -> EnvState:
        """Deterministically transition to the next state."""
        del params
        if self._use_precomputed:
            return self._precomputed_transition(state, action)
        return self._transition_core(state, action)

    def reward(
        self,
        state: EnvState,
        action: jax.Array,
        next_state: EnvState,
        params: EnvParams | None = None,
    ) -> jax.Array:
        """Compute the reward for a transition."""
        del params
        if self._use_precomputed:
            return self._precomputed_reward(state, action, next_state)
        return self._reward_core(state, action, next_state)

    def terminal(self, state: EnvState, params: EnvParams | None = None) -> jax.Array:
        """GoRight has no terminal states."""
        return jnp.asarray(False, dtype=jnp.bool_)

    def _transition_core(
        self,
        state: EnvState,
        action: jax.Array,
    ) -> EnvState:
        """Reference transition used for both on-the-fly and precomputed modes."""
        next_rng, _ = jrng.split(state.rng, 2)
        action = jnp.asarray(action, dtype=jnp.int32)
        position = jnp.asarray(state.position, dtype=jnp.int32)
        step = jnp.where(jnp.equal(action, 1), 1, -1)
        next_position = jnp.clip(position + step, 0, self.params.length - 1)

        next_status = self._next_status(state)
        next_prize = self._next_prize_inds(state, action, next_status, self.params)

        continue_triggering = jnp.logical_and(
            jnp.all(next_prize != 0),
            jnp.all(jnp.asarray(state.prize_inds, dtype=jnp.int32) != 0),
        )
        freeze_pos = jnp.logical_and(
            jnp.equal(position, self.params.first_checkpoint),
            continue_triggering,
        )
        next_position = jnp.where(freeze_pos, position, next_position)

        return EnvState(
            position=jnp.asarray(next_position, dtype=jnp.int32),
            previous_status=jnp.asarray(state.status, dtype=jnp.int32),
            status=jnp.asarray(next_status, dtype=jnp.int32),
            prize_inds=next_prize,
            rng=next_rng,
        )

    def _reward_core(
        self,
        state: EnvState,
        action: jax.Array,
        next_state: EnvState,
    ) -> jax.Array:
        """Reference reward used for both on-the-fly and precomputed modes."""
        action = jnp.asarray(action, dtype=jnp.int32)
        position = jnp.asarray(state.position, dtype=jnp.int32)
        prize = jnp.asarray(state.prize_inds, dtype=jnp.int32)
        next_prize = jnp.asarray(next_state.prize_inds, dtype=jnp.int32)

        is_right = jnp.equal(action, 1)
        penalty = jnp.where(is_right, -1.0, 0.0)

        is_first_cp = jnp.equal(position, self.params.first_checkpoint)
        is_second_cp = jnp.equal(position, self.params.second_checkpoint)
        is_cp = jnp.logical_or(is_first_cp, is_second_cp)

        triggered = jnp.logical_and(
            is_cp,
            jnp.logical_and(jnp.all(prize != 0), jnp.all(next_prize != 0)),
        )

        first_reward = jnp.where(is_first_cp, self.params.first_reward, 0.0)
        second_reward = jnp.where(is_second_cp, self.params.second_reward, 0.0)

        reward = jnp.where(triggered, first_reward + second_reward, penalty)
        return jnp.asarray(reward, dtype=jnp.float32)

    def _state_index(self, state: EnvState) -> jax.Array:
        """Flatten full observation features into a unique state index."""
        components = jnp.array(
            [
                state.position,
                state.previous_status,
                state.status,
                state.prize_inds[0],
                state.prize_inds[1],
            ],
            dtype=jnp.int32,
        )
        return jnp.ravel_multi_index(
            components, dims=self._fully_obs_shape, mode="clip"
        ).astype(jnp.int32)

    def _state_from_index(self, index: jax.Array, rng: PRNGKey) -> EnvState:
        """Reconstruct an EnvState from its flattened index."""
        components = self._state_components[index]
        return EnvState(
            position=components[0],
            previous_status=components[1],
            status=components[2],
            prize_inds=components[3:],
            rng=rng,
        )

    def _precomputed_transition(self, state: EnvState, action: jax.Array) -> EnvState:
        """Transition lookup using the cached table."""
        state_idx = self._state_index(state)
        action_idx = jnp.asarray(action, dtype=jnp.int32)
        next_idx = self._precomputed_next_states[state_idx, action_idx]
        next_rng, _ = jrng.split(state.rng, 2)
        return self._state_from_index(next_idx, next_rng)

    def _precomputed_reward(
        self,
        state: EnvState,
        action: jax.Array,
        next_state: EnvState,
    ) -> jax.Array:
        """Reward lookup using the cached table."""
        del next_state  # Reward is deterministic given (state, action) in this env.
        state_idx = self._state_index(state)
        action_idx = jnp.asarray(action, dtype=jnp.int32)
        return self._precomputed_rewards[state_idx, action_idx]

    def _compute_transition_table(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Enumerate all transitions and rewards for caching."""
        num_states = int(math.prod(self._fully_obs_shape))
        next_states = np.zeros((num_states, self.params.num_actions), dtype=np.int32)
        rewards = np.zeros((num_states, self.params.num_actions), dtype=np.float32)
        components = np.zeros((num_states, len(self._fully_obs_shape)), dtype=np.int32)

        dummy_rng = jrng.PRNGKey(0)
        for idx, multi_idx in enumerate(np.ndindex(self._fully_obs_shape)):
            position, prev_status, status, prize0, prize1 = multi_idx
            components[idx] = np.asarray(multi_idx, dtype=np.int32)
            env_state = EnvState(
                position=jnp.asarray(position, dtype=jnp.int32),
                previous_status=jnp.asarray(prev_status, dtype=jnp.int32),
                status=jnp.asarray(status, dtype=jnp.int32),
                prize_inds=jnp.asarray([prize0, prize1], dtype=jnp.int32),
                rng=dummy_rng,
            )

            for action in range(self.params.num_actions):
                act_arr = jnp.asarray(action, dtype=jnp.int32)
                next_state = self._transition_core(env_state, act_arr)
                next_components = (
                    int(next_state.position),
                    int(next_state.previous_status),
                    int(next_state.status),
                    int(next_state.prize_inds[0]),
                    int(next_state.prize_inds[1]),
                )
                next_idx = np.ravel_multi_index(
                    next_components, self._fully_obs_shape, mode="clip"
                )
                next_states[idx, action] = int(next_idx)
                rewards[idx, action] = float(
                    self._reward_core(env_state, act_arr, next_state)
                )

        return next_states, rewards, components

    def _cache_key(self) -> str:
        """Hash environment parameters to derive a cache filename."""
        params_dict = {
            "length": self.params.length,
            "num_indicators": self.params.num_indicators,
            "num_actions": self.params.num_actions,
            "first_checkpoint": self.params.first_checkpoint,
            "first_reward": self.params.first_reward,
            "second_checkpoint": self.params.second_checkpoint,
            "second_reward": self.params.second_reward,
            "is_partially_obs": self.params.is_partially_obs,
            "mapping": self.params.mapping,
        }
        serialized = json.dumps(params_dict, sort_keys=True)
        return hashlib.md5(serialized.encode("utf-8")).hexdigest()

    def _resolve_cache_path(self, override: str | None) -> Path:
        """Resolve a path for storing/loading the transition cache."""
        if override is not None:
            return Path(override)
        default_dir = Path("tmp")
        default_dir.mkdir(parents=True, exist_ok=True)
        return default_dir / f"goright_transitions_{self._cache_key()}.npz"

    def _load_or_compute_cache(
        self,
        force_recompute: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load cached transitions from disk or materialize them."""
        if self._cache_path.exists() and not force_recompute:
            data = np.load(self._cache_path)
            return data["next_states"], data["rewards"], data["components"]

        next_states, rewards, components = self._compute_transition_table()
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            self._cache_path,
            next_states=next_states,
            rewards=rewards,
            components=components,
        )
        return next_states, rewards, components

    def get_default_params(self, **kwargs) -> EnvParams:
        """Get the default params."""
        return EnvParams(**kwargs)


@gin.configurable
class GoRight:
    """Jax-friendly API around the functional environment."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30, "jax": True}

    def __init__(
        self,
        params: EnvParams | None = None,
        render_mode: str | None = None,
        use_precomputed: bool = False,
        transition_cache_path: str | None = None,
        force_recompute_cache: bool = False,
        **kwargs,
    ):
        """Wraps functional environment."""
        env = GoRightFunctional(
            params=params,
            use_precomputed=use_precomputed,
            transition_cache_path=transition_cache_path,
            force_recompute_cache=force_recompute_cache,
        )
        env.transform(jax.jit)
        self.env = env
        self._partial = bool(self.env.params.is_partially_obs)
        self._render_window = None
        self._render_clock = None
        self._render_font = None
        self._pygame_initialized = False
        self._render_padding = 32
        self._render_cell_width = 52
        self._render_cell_height = 72
        self._render_cell_gap = 6
        self._render_fps = 30

    def reset(self, rng: PRNGKey):
        """Resets the environment using the seed."""
        initial_state = self.env.initial(rng=rng)
        return initial_state, self.env.observation(initial_state)

    def step(self, state: EnvState, action: jax.Array):
        """Steps through the environment using the action."""
        next_state = self.env.transition(state, action)
        observation = self.env.observation(next_state)
        reward = self.env.reward(state, action, next_state)
        terminated = self.env.terminal(next_state)
        info = self.env.transition_info(state, action, next_state)

        return (
            next_state,
            observation,
            jnp.array(reward, dtype=float),
            jnp.array(terminated, dtype=bool),
            False,
            info,
        )

    def set_render_fps(self, fps: int) -> None:
        """Adjust the target frames-per-second for rendering."""
        self._render_fps = max(1, int(fps))

    def render(
        self,
        state: EnvState,
        last_reward: float = 0.0,
        last_action: int | None = None,
        info: str | None = None,
        bounds: dict | None = None,
    ) -> None:
        """Render the environment using pygame (for manual visualization/play)."""
        pygame_module = self._ensure_renderer()
        assert self._render_window is not None
        assert self._render_clock is not None
        assert self._render_font is not None

        self._render_clock.tick(self._render_fps)
        pygame_module.event.pump()

        surface = self._render_window
        surface.fill(_BG_COLOR)

        padding = self._render_padding
        cell_w = self._render_cell_width
        cell_h = self._render_cell_height
        gap = self._render_cell_gap

        track_y = padding
        prize_value = np.asarray(state.prize_inds)
        position_val = int(np.asarray(state.position))
        prev_status_val = int(np.asarray(state.previous_status))
        status_val = int(np.asarray(state.status))

        render_track_rects = []

        for position in range(self.env.params.length):
            x = padding + position * (cell_w + gap)
            rect = pygame_module.Rect(x, track_y, cell_w, cell_h)
            color = _TRACK_COLOR
            if position in (
                self.env.params.first_checkpoint,
                self.env.params.second_checkpoint,
            ):
                color = _CHECKPOINT_COLOR

            pygame_module.draw.rect(surface, color, rect, border_radius=8)
            pygame_module.draw.rect(
                surface, _TRACK_BORDER, rect, width=2, border_radius=8
            )
            render_track_rects.append(rect)

            if position_val == position:
                agent_rect = rect.inflate(-int(cell_w * 0.45), -int(cell_h * 0.45))
                pygame_module.draw.rect(
                    surface, _AGENT_COLOR, agent_rect, border_radius=8
                )

        # Prize indicators.
        prize_label_y = track_y + cell_h + 12
        self._draw_text(surface, "Prizes:", (padding, prize_label_y))

        prize_size = 24
        prize_y = prize_label_y + 22
        for idx, value in enumerate(prize_value.astype(int)):
            prize_x = padding + idx * (prize_size + 10)
            prize_rect = pygame_module.Rect(prize_x, prize_y, prize_size, prize_size)
            color = _PRIZE_ACTIVE if value else _PRIZE_INACTIVE
            pygame_module.draw.rect(surface, color, prize_rect, border_radius=6)
            pygame_module.draw.rect(
                surface, _TRACK_BORDER, prize_rect, width=1, border_radius=6
            )
            label = self._render_font.render(str(idx + 1), True, _TEXT_COLOR)
            label_rect = label.get_rect(
                center=(prize_rect.centerx, prize_rect.bottom + 12)
            )
            surface.blit(label, label_rect)

        # Status indicators.
        status_y = prize_y + prize_size + 28
        prev_rect = pygame_module.Rect(padding, status_y, 40, 40)
        curr_rect = pygame_module.Rect(padding + 200, status_y, 40, 40)

        prev_color = _STATUS_COLORS.get(prev_status_val, _TRACK_BORDER)
        curr_color = _STATUS_COLORS.get(status_val, _TRACK_BORDER)

        pygame_module.draw.rect(surface, prev_color, prev_rect, border_radius=8)
        pygame_module.draw.rect(surface, curr_color, curr_rect, border_radius=8)
        pygame_module.draw.rect(
            surface, _TRACK_BORDER, prev_rect, width=2, border_radius=8
        )
        pygame_module.draw.rect(
            surface, _TRACK_BORDER, curr_rect, width=2, border_radius=8
        )

        prev_text = self._render_font.render(str(prev_status_val), True, _BG_COLOR)
        curr_text = self._render_font.render(str(status_val), True, _BG_COLOR)

        surface.blit(prev_text, prev_text.get_rect(center=prev_rect.center))
        surface.blit(curr_text, curr_text.get_rect(center=curr_rect.center))

        self._draw_text(
            surface, "prev status", (prev_rect.right + 12, prev_rect.y + 10)
        )
        self._draw_text(
            surface, "curr status", (curr_rect.right + 12, curr_rect.y + 10)
        )

        reward_width = 120
        reward_rect = pygame_module.Rect(padding + 420, status_y, reward_width, 40)
        reward_value = float(last_reward)
        if reward_value > 0:
            reward_color = _STATUS_COLORS[1]
            reward_text_color = _BG_COLOR
        elif reward_value < 0:
            reward_color = _STATUS_COLORS[0]
            reward_text_color = _BG_COLOR
        else:
            reward_color = _TRACK_BORDER
            reward_text_color = _TEXT_COLOR
        pygame_module.draw.rect(surface, reward_color, reward_rect, border_radius=8)
        pygame_module.draw.rect(
            surface, _TRACK_BORDER, reward_rect, width=2, border_radius=8
        )
        reward_text = self._render_font.render(
            f"{reward_value:.2f}", True, reward_text_color
        )
        surface.blit(reward_text, reward_text.get_rect(center=reward_rect.center))
        self._draw_text(surface, "reward", (reward_rect.right + 12, reward_rect.y + 10))

        info_start_y = status_y + 60
        if bounds:
            card_w = 240
            card_h = 160
            card_gap = 24
            card_y = info_start_y
            label_text = {
                "lower": "lower bound",
                "expected": "expected",
                "upper": "upper bound",
            }
            for idx, key in enumerate(("lower", "expected", "upper")):
                data = bounds.get(key)
                if not data:
                    continue
                card_x = padding + idx * (card_w + card_gap)
                card_rect = pygame_module.Rect(card_x, card_y, card_w, card_h)
                color = _BOUND_COLORS.get(key, _TRACK_COLOR)
                pygame_module.draw.rect(surface, color, card_rect, border_radius=10)
                pygame_module.draw.rect(
                    surface, _TRACK_BORDER, card_rect, width=2, border_radius=10
                )

                title = self._render_font.render(label_text[key], True, _BG_COLOR)
                surface.blit(title, (card_rect.x + 18, card_rect.y + 16))

                summary_lines = self._obs_summary_lines(int(data["obs"]))
                summary_start = card_rect.y + 54
                line_spacing = 22
                for line_idx, text in enumerate(summary_lines):
                    line_surface = self._render_font.render(text, True, _BG_COLOR)
                    line_y = summary_start + line_idx * line_spacing
                    surface.blit(line_surface, (card_rect.x + 18, line_y))

                reward_surface = self._render_font.render(
                    f"reward: {data['reward']:.2f}", True, _BG_COLOR
                )
                reward_y = card_rect.bottom - 36
                surface.blit(reward_surface, (card_rect.x + 18, reward_y))

            info_start_y = card_y + card_h + 28

        prize_str = " ".join(str(int(v)) for v in prize_value) or "–"
        lines = [
            f"position: {position_val} / {self.env.params.length - 1}",
            f"prizes: {prize_str}",
            f"last reward: {reward_value:.2f}",
        ]

        if last_action is not None:
            direction = "right" if last_action == 1 else "left"
            lines.append(f"last action: {direction} ({last_action})")

        obs_idx = int(np.asarray(self.env.observation(state)))
        lines.extend(self._obs_summary_lines(obs_idx))

        if info:
            lines.append(info)

        for idx, line in enumerate(lines):
            self._draw_text(surface, line, (padding, info_start_y + idx * 22))

        pygame_module.display.flip()

    def close(self) -> None:
        """Release pygame resources allocated by the renderer."""
        if pygame is None:
            return

        if self._render_window is not None and pygame.display.get_init():
            pygame.display.quit()
        self._render_window = None
        self._render_clock = None
        self._render_font = None

        if self._pygame_initialized:
            pygame.quit()
            self._pygame_initialized = False

    def _ensure_renderer(self):
        pygame_module = _load_pygame()

        if not self._pygame_initialized:
            pygame_module.init()
            pygame_module.display.set_caption("GoRight")
            self._pygame_initialized = True

        if self._render_clock is None:
            self._render_clock = pygame_module.time.Clock()

        if self._render_font is None:
            self._render_font = pygame_module.font.SysFont("Menlo", 18)

        if self._render_window is None:
            width = (
                self._render_padding * 2
                + self.env.params.length * self._render_cell_width
                + (self.env.params.length - 1) * self._render_cell_gap
            )
            info_block = 440
            height = self._render_padding * 2 + self._render_cell_height + info_block
            self._render_window = pygame_module.display.set_mode((width, height))

        return pygame_module

    def _draw_text(self, surface, text: str, pos: tuple[int, int]) -> None:
        assert self._render_font is not None
        surface.blit(self._render_font.render(text, True, _TEXT_COLOR), pos)

    def _obs_summary_lines(self, obs_idx: int) -> list[str]:
        indices = np.unravel_index(int(obs_idx), self.env.obs_shape)
        if self._partial:
            position, status, *prize_inds = indices
            prize_str = " ".join(str(int(v)) for v in prize_inds) or "–"
            return [
                f"obs pos: {int(position)}",
                f"obs status: {int(status)}",
                f"obs prizes: {prize_str}",
            ]

        position, previous_status, status, *prize_inds = indices
        prize_str = " ".join(str(int(v)) for v in prize_inds) or "–"
        return [
            f"obs pos: {int(position)}",
            f"obs prev: {int(previous_status)}",
            f"obs status: {int(status)}",
            f"obs prizes: {prize_str}",
        ]
