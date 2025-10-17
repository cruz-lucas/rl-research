"""Agent package exposing registration helpers and built-in implementations."""

from rl_research.agents.base import AGENTS, Agent
from minimal_agents import QLearningAgent, IntrinsicQLearningAgent, RMaxAgent, MBIEAgent
from minimal_agents.policies import UCBPolicy, RandomWalkPolicy, EpsilonGreedyPolicy

AGENTS.register("q_learning_epsgreedy", lambda cfg: QLearningAgent(**cfg, policy=EpsilonGreedyPolicy()))
AGENTS.register("q_learning_ucb", lambda cfg: QLearningAgent(**cfg, policy=UCBPolicy()))
AGENTS.register("q_learning_randomwalk", lambda cfg: QLearningAgent(**cfg, policy=RandomWalkPolicy()))
AGENTS.register("q_learning_intrinsic", lambda cfg: IntrinsicQLearningAgent(**cfg, policy=EpsilonGreedyPolicy()))
AGENTS.register("rmax", lambda cfg: RMaxAgent(**cfg))
AGENTS.register("mbie", lambda cfg: MBIEAgent(**cfg))

__all__ = [
    "Agent",
    "AGENTS",
]
