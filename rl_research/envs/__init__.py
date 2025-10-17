"""Environment registry and built-in implementations."""

from rl_research.envs.base import ENVIRONMENTS, Environment, EnvFactory, StepResult
from classic_pacmdp_envs import RiverSwimJaxEnv, SixArmsJaxEnv

ENVIRONMENTS.register("RiverSwim", RiverSwimJaxEnv)
ENVIRONMENTS.register("SixArms", SixArmsJaxEnv)

__all__ = [
    "ENVIRONMENTS",
    # "Environment",
    # "EnvFactory",
    # "StepResult",
    "RiverSwimJaxEnv",
    "SixArmsJaxEnv"
]
