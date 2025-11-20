from typing import Protocol, TypeAlias, Tuple
from abc import abstractmethod
import jax
from flax import struct

PRNGKey: TypeAlias = jax.Array

@struct.dataclass
class EnvState(Protocol):
    ...

class EnvParams(Protocol):
    ...

class BaseJaxEnv:
    metadata = {}

    @abstractmethod
    def __init__(
        self, params: EnvParams | None = None, render_mode: str | None = None, **kwargs
    ):
        ...

    @property
    @abstractmethod
    def env(self):
        ...

    @abstractmethod
    def reset(self, rng: PRNGKey) -> Tuple[EnvState, jax.Array]:
        ...

    def step(self, state: EnvState, action: jax.Array) -> Tuple[EnvState, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        ...