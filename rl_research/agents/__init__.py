"""User-facing entry-points for the agents package."""

from rl_research.agents.base import TabularAgent, UpdateResult
from rl_research.agents.dt_lambda_planner import DTLambdaPAgent, DTLambdaPParams
from rl_research.agents.dt_nstep_planner import DTNStepPAgent, DTNStepPParams
from rl_research.agents.dt_rmax_planner import DTRMaxNStepAgent, DTRMaxNStepParams
from rl_research.agents.dt_ucb_planner import DTUCBPlanner, DTUCBParams
from rl_research.agents.mcts import MCTSAgent, MCTSAgentParams
from rl_research.agents.expectation_models import (
    goright_expectation_model,
    riverswim_expectation_model,
    sixarms_expectation_model,
)
from rl_research.agents.mbie import MBIEAgent, MBIEParams
from rl_research.agents.rmax import RMaxAgent, RMaxParams
from rl_research.agents.q_learning import QLearningAgent, QlearningParams
from rl_research.agents.rmax_mcts import (
    RMaxMCTSAgent,
    RMaxMCTSAgentParams,
    RMaxMCTSAgentState,
)

__all__ = [
    "TabularAgent",
    "UpdateResult",
    "DTLambdaPAgent",
    "DTLambdaPParams",
    "DTNStepPAgent",
    "DTNStepPParams",
    "DTRMaxNStepAgent",
    "DTRMaxNStepParams",
    "DTUCBPlanner",
    "DTUCBParams",
    "MCTSAgent",
    "MCTSAgentParams",
    "RMaxMCTSAgent",
    "RMaxMCTSAgentParams",
    "RMaxMCTSAgentState",
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
