"""Hand-crafted expectation dynamics for supported tabular environments."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from rl_research.models import StaticTabularModel


EXPECTED_STATUS_MAPPING = np.asarray(
    [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ],
    dtype=np.int32,
)


def goright_expectation_model(
    *,
    length: int = 21,
    first_checkpoint: int = 10,
    first_reward: float = 3.0,
    second_checkpoint: int = 20,
    second_reward: float = 6.0,
    num_indicators: int = 2,
    is_partially_obs: bool = False,
) -> np.ndarray:
    """Return a tabular expectation model for the GoRight environment.

    Parameters mirror `goright.jax.env.EnvParams` but avoid any dependency on JAX.
    When `is_partially_obs` is True, the observation space excludes the
    `previous_status` feature and the returned array has shape (252, 2, 2);
    otherwise it matches the fully observed case with shape (756, 2, 2).
    """

    if length <= 0:
        raise ValueError("`length` must be positive.")
    if num_indicators <= 0:
        raise ValueError("`num_indicators` must be positive.")

    mapping = EXPECTED_STATUS_MAPPING
    if mapping.shape != (3, 3):
        raise ValueError("`status_mapping` must have shape (3, 3).")

    fully_shape: Tuple[int, ...] = (length, 3, 3, 2, 2)
    partial_shape: Tuple[int, ...] = (length, 3, 2, 2)
    obs_shape = partial_shape if is_partially_obs else fully_shape

    num_observations = int(np.prod(obs_shape))
    model = np.zeros((num_observations, 2, 2), dtype=np.float32)

    def _next_status(prev_status: int, status: int) -> int:
        return int(mapping[prev_status, status])

    def _shift_prizes(prize: np.ndarray) -> np.ndarray:
        pad = 0 if np.any(prize != 0) else 1
        padded = np.concatenate([prize, np.asarray([pad], dtype=np.int32)], axis=0)
        rolled = np.roll(padded, 1)
        return rolled[:num_indicators]

    def _next_prize_indices(
        prize: np.ndarray, action: int, next_status: int, position: int
    ) -> np.ndarray:
        is_jackpot = next_status == 2
        is_right = action == 1
        triggered_now = bool(np.all(prize != 0))

        is_before_first = position == first_checkpoint - 1
        is_before_second = position == second_checkpoint - 1
        is_before_cp = is_before_first or is_before_second

        is_first_cp = position == first_checkpoint
        is_second_cp = position == second_checkpoint
        is_prize_cp = is_first_cp or is_second_cp

        is_entering = is_right and is_before_cp and is_jackpot
        is_continuing = triggered_now and is_right and is_prize_cp
        is_triggering = is_entering or is_continuing

        is_at_end = position == length - 1
        needs_shift = is_at_end and is_right and (not is_triggering)

        if is_triggering:
            return np.ones_like(prize, dtype=np.int32)
        if needs_shift:
            return _shift_prizes(prize)
        return np.zeros_like(prize, dtype=np.int32)

    def _transition(
        state: Tuple[int, int, int, int, int], action: int
    ) -> Tuple[int, int, int, int, int]:
        position, prev_status, status, prize0, prize1 = state
        prize = np.asarray([prize0, prize1], dtype=np.int32)
        delta = 1 if action == 1 else -1
        next_position = int(np.clip(position + delta, 0, length - 1))
        next_status = _next_status(prev_status, status)
        next_prize = _next_prize_indices(prize, action, next_status, position)

        continue_triggering = bool(np.all(next_prize != 0) and np.all(prize != 0))
        freeze_position = (position == first_checkpoint) and continue_triggering
        if freeze_position:
            next_position = position

        return (
            next_position,
            status,
            next_status,
            int(next_prize[0]),
            int(next_prize[1]),
        )

    def _reward(
        state: Tuple[int, int, int, int, int],
        action: int,
        next_state: Tuple[int, int, int, int, int],
    ) -> float:
        position, _prev_status, _status, prize0, prize1 = state
        next_prize = np.asarray(next_state[3:], dtype=np.int32)
        prize = np.asarray([prize0, prize1], dtype=np.int32)

        is_right = action == 1
        penalty = -1.0 if is_right else 0.0

        is_first_cp = position == first_checkpoint
        is_second_cp = position == second_checkpoint
        is_triggered = (is_first_cp or is_second_cp) and bool(
            np.all(prize != 0) and np.all(next_prize != 0)
        )

        if not is_triggered:
            return penalty

        reward = 0.0
        if is_first_cp:
            reward += first_reward
        if is_second_cp:
            reward += second_reward
        return reward

    for multi_index in np.ndindex(obs_shape):
        obs_index = np.ravel_multi_index(multi_index, obs_shape)

        if is_partially_obs:
            position, status, prize0, prize1 = (int(x) for x in multi_index)
            prev_status = 0  # Doesn't matter bc of mapping
            state = (position, prev_status, status, prize0, prize1)
            project = lambda st: (st[0], st[2], st[3], st[4])
        else:
            state = tuple(int(x) for x in multi_index)
            project = lambda st: st

        for action in (0, 1):
            next_state = _transition(state, action)
            projected_next = project(next_state)
            next_obs_index = np.ravel_multi_index(projected_next, obs_shape)
            reward = _reward(state, action, next_state)
            model[obs_index, action] = (int(next_obs_index), float(reward))

    return StaticTabularModel(model)


__all__ = [
    "goright_expectation_model",
]
