"""Tabular wrapper for Minigrid environments.

This module provides a tabular state representation for Minigrid environments
by converting the fully observable grid state into a discrete state index.
"""

from typing import Tuple

import gymnasium as gym
import numpy as np
from minigrid.wrappers import FullyObsWrapper


class TabularMinigridWrapper:
    """Wraps a Minigrid environment to provide tabular state representation.
    
    This wrapper:
    1. Creates a Minigrid environment with FullyObsWrapper
    2. Maintains a state discretization that maps grid observations to integers
    3. Provides gym-like interface (reset, step) with discrete state indices
    """

    def __init__(
        self,
        env_id: str,
        seed: int = 0,
    ):
        """Initialize the tabular wrapper.
        
        Args:
            env_id: Minigrid environment ID
            seed: Random seed
        """
        self.env_id = env_id
        self.seed = seed
        
        self.base_env = gym.make(env_id)
        self.base_env = FullyObsWrapper(self.base_env)
        
        self.action_space = self.base_env.action_space
        self.num_actions = self.action_space.n

        self.num_states = 0
        self.observation_space = gym.spaces.Discrete(1)  # Will be updated dynamically
        
        self.state_to_idx = {}
        self.idx_to_state = {}
        self.next_state_idx = 0
        
        self.reset()
        
    def _grid_to_hashable(self, grid: np.ndarray, agent_direction: np.int8) -> tuple:
        """Convert grid to a hashable representation."""
        return (agent_direction,) + tuple(grid.flatten().astype(int).tolist())

    def _get_state_index(self, grid: np.ndarray, agent_direction: np.int8) -> int:
        """Get or create state index for a grid observation."""
        grid_hash = self._grid_to_hashable(grid, agent_direction)

        if grid_hash not in self.state_to_idx:
            self.num_states += 1
            self.observation_space = gym.spaces.Discrete(self.num_states)

            self.state_to_idx[grid_hash] = self.next_state_idx
            self.idx_to_state[self.next_state_idx] = grid_hash
            self.next_state_idx += 1

        return self.state_to_idx[grid_hash]
    
    def reset(self, seed: int | None = None) -> Tuple[int, dict]:
        """Reset the environment.
        
        Args:
            seed: Optional seed for reproducibility
            
        Returns:
            Tuple of (state_index, info_dict)
        """
        if seed is not None:
            self.base_env.reset(seed=seed)
        else:
            self.base_env.reset(seed=self.seed)
        
        obs, info = self.base_env.reset()
        grid = obs["image"]
        agent_direction = obs["direction"]
        state_idx = self._get_state_index(grid, agent_direction)
        
        return state_idx, info
    
    def step(self, action: int) -> Tuple[int, float, bool, bool, dict]:
        """Execute one step in the environment.
        
        Args:
            action: Action to execute
            
        Returns:
            Tuple of (state_index, reward, terminated, truncated, info)
        """
        obs, reward, terminated, truncated, info = self.base_env.step(action)
        
        grid = obs["image"]
        agent_direction = obs["direction"]
        state_idx = self._get_state_index(grid, agent_direction)
        
        return state_idx, float(reward), terminated, truncated, info


def create_tabular_minigrid_env(
    env_id: str = "MiniGrid-DoorKey-16x16-v0",
    seed: int = 0,
) -> TabularMinigridWrapper:
    """Create a tabular Minigrid environment.
    
    Args:
        env_id: Minigrid environment ID
        max_episode_steps: Maximum steps per episode
        seed: Random seed
        
    Returns:
        TabularMinigridWrapper instance
    """
    return TabularMinigridWrapper(env_id, seed)
