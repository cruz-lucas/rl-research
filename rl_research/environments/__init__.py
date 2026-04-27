import gin
from classic_pacmdp_envs import RiverSwimJaxEnv, SixArmsJaxEnv

from rl_research.environments.base import BaseJaxEnv
from rl_research.environments.goright import GoRight
from rl_research.environments.navix import (
    ACTION_NAMES,
    CARDINAL_ACTION_SET,
    CardinalNavixWrapper,
    NavixWrapper,
)


gin.register(RiverSwimJaxEnv)
gin.register(SixArmsJaxEnv)

__all__ = [
    "BaseJaxEnv",
    "ACTION_NAMES",
    "CARDINAL_ACTION_SET",
    "CardinalNavixWrapper",
    "GoRight",
    "NavixWrapper",
    "RiverSwimJaxEnv",
    "SixArmsJaxEnv",
]
