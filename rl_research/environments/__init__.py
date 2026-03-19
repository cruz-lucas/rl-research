from rl_research.environments.base import BaseJaxEnv
from rl_research.environments.goright import GoRight
from rl_research.environments.navix import NavixWrapper
from classic_pacmdp_envs import RiverSwimJaxEnv, SixArmsJaxEnv
import gin

gin.register(RiverSwimJaxEnv)
gin.register(SixArmsJaxEnv)

__all__ = [
    "BaseJaxEnv",
    "GoRight",
    "NavixWrapper",
    "RiverSwimJaxEnv",
    "SixArmsJaxEnv",
]
