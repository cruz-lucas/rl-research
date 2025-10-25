"""User-facing entry-points for the agents package."""

from rl_research.agents.base import TabularAgent, UpdateResult
from rl_research.agents.dt_planner import DTPAgent, DTPParams
from rl_research.agents.dt_ucb_planner import DTUCBPlanner, DTUCBParams
from rl_research.agents.expectation_models import (
    goright_expectation_model,
    riverswim_expectation_model,
    sixarms_expectation_model,
)
from rl_research.agents.mbie import MBIEAgent, MBIEParams
from rl_research.agents.rmax import RMaxAgent, RMaxParams
from rl_research.agents.q_learning import QLearningAgent, QlearningParams

__all__ = [
    "TabularAgent",
    "UpdateResult",
    "DTPAgent",
    "DTPParams",
    "DTUCBPlanner",
    "DTUCBParams",
    "RMaxAgent",
    "RMaxParams",
    "MBIEAgent",
    "MBIEParams",
    "QLearningAgent",
    "QlearningParams",
    "riverswim_expectation_model",
    "sixarms_expectation_model",
    "goright_expectation_model",
]
