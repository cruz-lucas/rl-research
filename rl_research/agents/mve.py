"""Q-learning agent that uses a learned model for rollouts."""

from importlib import resources
from pathlib import Path
from typing import NamedTuple

import numpy as np

from planning4exploration.agents.q_learning_agent import QLearningAgent, Transition


class MVEParams(NamedTuple):
    discount_factor: float = 0.9
    step_size: float = 1.0
    initial_epsilon: float = 1.0
    horizon: int = 5
    model_filename: str = 'expectation_model.npz'
    fully_obs: bool = False


def _load_model(model_filename: str) -> np.ndarray:
    """Resolve and load the expectation model with sensible fallbacks."""

    resource = resources.files("planning4exploration.environment.data") / model_filename
    try:
        with resources.as_file(resource) as resolved_path:
            return np.load(Path(resolved_path), allow_pickle=True)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Could not locate an expectation model for '{model_filename}'. Tried: {resource}"
        )

class MVEAgent(QLearningAgent):
    """MVE agent class."""
    def __init__(self, params: MVEParams, num_actions: int, num_states: int, seed: int):
        super().__init__(params, num_actions, num_states, seed)
        model = _load_model(params.model_filename)
        self.model = model['fully_obs'] if params.fully_obs else model['partially_obs']

    def update(self, transition: Transition) -> float:
        obs = transition.obs
        next_obs = transition.next_obs
        done = transition.done
        action = transition.action
        reward = transition.reward

        alpha = self.params.step_size
        gamma = self.params.discount_factor
        not_done = 1.0 - int(done)

        sim_obs = int(next_obs)
        sim_gamma = gamma
        sim_return = reward

        q_sa = self.Q[obs, action]
        target = reward + gamma * not_done * np.max(self.Q[next_obs])

        targets = [target]
        for h in range(self.params.horizon):
            sim_a = self._greedy_action(sim_obs)
            next_sim_obs, sim_reward = self.model[sim_obs, sim_a]

            next_sim_obs = int(next_sim_obs)

            sim_return += sim_gamma * sim_reward
            sim_gamma *= gamma
            sim_target = sim_return + sim_gamma * np.max(self.Q[next_sim_obs])
            targets.append(sim_target)

            sim_obs = next_sim_obs

        mean_target = np.mean(targets)
        td = mean_target - q_sa
        self.Q[obs, action] += alpha * td
        self.N[obs, action] += 1

        return td