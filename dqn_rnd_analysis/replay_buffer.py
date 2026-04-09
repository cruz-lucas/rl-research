from dataclasses import dataclass

import numpy as np


@dataclass
class ReplayBatch:
    observations: np.ndarray
    actions: np.ndarray
    extrinsic_rewards: np.ndarray
    discounts: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    insertion_steps: np.ndarray
    stored_intrinsic_rewards: np.ndarray
    state_ids: np.ndarray
    next_state_ids: np.ndarray
    indices: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int, observation_dim: int, seed: int = 0):
        self.capacity = int(capacity)
        self.observation_dim = int(observation_dim)
        self.rng = np.random.default_rng(seed)

        self.observations = np.zeros(
            (self.capacity, self.observation_dim), dtype=np.float32
        )
        self.actions = np.zeros((self.capacity,), dtype=np.int32)
        self.extrinsic_rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.discounts = np.zeros((self.capacity,), dtype=np.float32)
        self.next_observations = np.zeros(
            (self.capacity, self.observation_dim), dtype=np.float32
        )
        self.dones = np.zeros((self.capacity,), dtype=np.bool_)
        self.insertion_steps = np.zeros((self.capacity,), dtype=np.int64)
        self.stored_intrinsic_rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.state_ids = np.zeros((self.capacity,), dtype=np.int32)
        self.next_state_ids = np.zeros((self.capacity,), dtype=np.int32)

        self.position = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def ready(self, batch_size: int) -> bool:
        return self.size >= int(batch_size)

    def add(
        self,
        observation: np.ndarray,
        action: int,
        extrinsic_reward: float,
        discount: float,
        next_observation: np.ndarray,
        done: bool,
        insertion_step: int,
        stored_intrinsic_reward: float,
        state_id: int,
        next_state_id: int,
    ) -> None:
        index = self.position % self.capacity

        self.observations[index] = np.asarray(observation, dtype=np.float32).reshape(-1)
        self.actions[index] = int(action)
        self.extrinsic_rewards[index] = float(extrinsic_reward)
        self.discounts[index] = float(discount)
        self.next_observations[index] = np.asarray(
            next_observation, dtype=np.float32
        ).reshape(-1)
        self.dones[index] = bool(done)
        self.insertion_steps[index] = int(insertion_step)
        self.stored_intrinsic_rewards[index] = float(stored_intrinsic_reward)
        self.state_ids[index] = int(state_id)
        self.next_state_ids[index] = int(next_state_id)

        self.position += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> ReplayBatch:
        if self.size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")

        indices = self.rng.integers(0, self.size, size=int(batch_size))
        return ReplayBatch(
            observations=self.observations[indices].copy(),
            actions=self.actions[indices].copy(),
            extrinsic_rewards=self.extrinsic_rewards[indices].copy(),
            discounts=self.discounts[indices].copy(),
            next_observations=self.next_observations[indices].copy(),
            dones=self.dones[indices].copy(),
            insertion_steps=self.insertion_steps[indices].copy(),
            stored_intrinsic_rewards=self.stored_intrinsic_rewards[indices].copy(),
            state_ids=self.state_ids[indices].copy(),
            next_state_ids=self.next_state_ids[indices].copy(),
            indices=indices.astype(np.int64, copy=False),
        )
