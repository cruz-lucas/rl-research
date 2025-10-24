"""User-facing entry-points for the agents package."""

from rl_research.agents.base import TabularAgent, UpdateResult
from rl_research.agents.mbie import MBIEAgent, MBIEParams
from rl_research.agents.rmax import RMaxAgent, RMaxParams
from rl_research.agents.q_learning import QLearningAgent, QlearningParams

__all__ = [
    "TabularAgent",
    "UpdateResult",
    "RMaxAgent",
    "RMaxParams",
    "MBIEAgent",
    "MBIEParams",
    "QLearningAgent",
    "QlearningParams",
]
