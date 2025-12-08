"""Interactive GoRight visualizer using pygame."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import tyro

from rl_research.environments.goright import EnvParams, GoRight, _load_pygame


@dataclass
class Args:
    """CLI options for playing the GoRight environment."""

    seed: int = 0
    fps: int = 10
    partial: bool = False
    no_precomputed: bool = False


def main(args: Args) -> int:
    pygame = _load_pygame()

    params = EnvParams(is_partially_obs=args.partial)
    env = GoRight(params=params, use_precomputed=not args.no_precomputed)
    env.set_render_fps(args.fps)

    key = jax.random.PRNGKey(args.seed)
    state, _ = env.reset(key)
    last_reward = 0.0
    last_action = None
    info = "Press left/right arrows or A/D. Esc to quit."

    running = True
    while running:
        env.render(
            state=state,
            last_reward=last_reward,
            last_action=last_action,
            info=info,
        )

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                    break
                if event.key in (pygame.K_LEFT, pygame.K_a):
                    action = 0
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    action = 1
                else:
                    continue

                next_state, _, reward, terminated, _truncated, _info = env.step(
                    state, jnp.asarray(action, dtype=jnp.int32)
                )
                state = next_state
                last_action = action
                last_reward = float(np.asarray(reward))

                if bool(np.asarray(terminated)):
                    running = False
                    break

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(tyro.cli(Args)))
