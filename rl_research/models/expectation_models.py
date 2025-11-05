"""Hand-crafted expectation dynamics for supported tabular environments."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
from rl_research.models import StaticTabularModel


def riverswim_expectation_model(
    *,
    num_states: int = 6,
    p_left: float = 0.1,
    p_stay: float = 0.6,
    p_right: float = 0.3,
    easy_reward: float = 5.0,
    hard_reward: float = 10_000.0,
    common_reward: float = 0.0,
) -> np.ndarray:
    """Return a (num_states, 2, 2) expectation model for the RiverSwim environment.

    The first axis corresponds to the tabular observation index, the second axis
    to the action (`0`=left, `1`=right), and the final dimension stores
    `(expected_next_obs, expected_reward)`.
    """
    num_actions = 2
    if num_states <= 0:
        raise ValueError("`num_states` must be positive.")

    transition_sums = p_left + p_stay + p_right
    if not np.isclose(transition_sums, 1.0):
        raise ValueError("RiverSwim transition probabilities must sum to 1.")

    model = np.zeros((num_states, num_actions, 2), dtype=np.float32)

    for state in range(num_states):
        # Action 0: move left deterministically.
        left_next = 0 if state == 0 else state - 1
        left_reward = easy_reward if state == 0 else common_reward
        model[state, 0] = (left_next, left_reward)

        # Action 1: stochastic movement biased to the right.
        if state == num_states - 1:
            probs = np.array([1.0 - p_right, 0.0, p_right], dtype=np.float32)
            candidates = np.array([state - 1, state, state], dtype=np.float32)
            reward = hard_reward * p_right
        else:
            probs = np.array([p_left, p_stay, p_right], dtype=np.float32)
            candidates = np.clip(
                np.array([state - 1, state, state + 1], dtype=np.float32),
                0,
                num_states - 1,
            )
            reward = common_reward

        expected_next = int(np.round(np.dot(probs, candidates), 0))
        model[state, 1] = (expected_next, reward)

    return model


def sixarms_expectation_model(
    *,
    success_probabilities: Iterable[float] = (1.0, 0.15, 0.10, 0.05, 0.03, 0.01),
    reward_by_state: Iterable[float] = (0.0, 50.0, 133.0, 300.0, 800.0, 1660.0, 6000.0),
) -> np.ndarray:
    """Return a (7, 6, 2) expectation model for the SixArms environment.

    The defaults match `classic_pacmdp_envs.sixarms.SixArmsFunctional`.
    """
    success_probabilities = tuple(float(p) for p in success_probabilities)
    reward_by_state = tuple(float(r) for r in reward_by_state)

    num_states = len(reward_by_state)
    num_actions = len(success_probabilities)
    if num_states != 7 or num_actions != 6:
        raise ValueError(
            "SixArms currently expects 7 states and 6 actions. "
            f"Received {num_states=}, {num_actions=}."
        )

    branch_transitions = np.asarray(
        [
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 1],
            [0, 2, 0, 0, 0, 0],
            [0, 0, 3, 0, 0, 0],
            [0, 0, 0, 4, 0, 0],
            [0, 0, 0, 0, 5, 0],
            [0, 0, 0, 0, 0, 6],
        ],
        dtype=np.int32,
    )

    model = np.zeros((num_states, num_actions, 2), dtype=np.float32)
    for state in range(num_states):
        for action in range(num_actions):
            if state == 0:
                p_success = success_probabilities[action]
                expected_next = int(np.round(p_success * (action + 1), 0))
                reward = 0.0
            else:
                next_state = int(branch_transitions[state, action])
                expected_next = int(next_state)
                reward = reward_by_state[next_state] if next_state == state else 0.0

            model[state, action] = (expected_next, reward)

    return model


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

    def _transition(state: Tuple[int, int, int, int, int], action: int) -> Tuple[int, int, int, int, int]:
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

    def _reward(state: Tuple[int, int, int, int, int], action: int, next_state: Tuple[int, int, int, int, int]) -> float:
        position, _prev_status, _status, prize0, prize1 = state
        next_prize = np.asarray(next_state[3:], dtype=np.int32)
        prize = np.asarray([prize0, prize1], dtype=np.int32)

        is_right = action == 1
        penalty = -1.0 if is_right else 0.0

        is_first_cp = position == first_checkpoint
        is_second_cp = position == second_checkpoint
        is_triggered = (is_first_cp or is_second_cp) and bool(np.all(prize != 0) and np.all(next_prize != 0))

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
            prev_status = 0 # Doesn't matter bc of mapping
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
    "riverswim_expectation_model",
    "sixarms_expectation_model",
    "goright_expectation_model",
]
