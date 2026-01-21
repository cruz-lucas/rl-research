"""
Simple script to measure the time to run 10M steps of the goright environment.

Usage:
    python scripts/benchmark_goright.py
"""

import time
import jax
import jax.lax as lax
import jax.random as jrng
from rl_research.environments.goright import GoRight, EnvParams


def make_step_fn(env):
    """Factory function to create a step function with env captured in closure."""
    @jax.jit
    def step_fn(carry, _):
        """Single step function for scan."""
        state, rng = carry
        rng, action_rng = jrng.split(rng)
        action = jrng.randint(action_rng, shape=(), minval=0, maxval=1)
        state, obs, reward, terminated, truncated, _ = env.step(state, action)
        
        state = jax.lax.cond(
            terminated | truncated,
            lambda s: env.reset(rng)[0],
            lambda s: s,
            state
        )
        
        return (state, rng), None
    
    return step_fn


def main():
    
    env_params = EnvParams()
    env = GoRight(env_params, use_precomputed=False)
    rng = jrng.PRNGKey(0)
    
    rng, reset_rng = jrng.split(rng)
    state, obs = env.reset(reset_rng)

    step_fn = make_step_fn(env)
    
    num_steps = 10_000_000
    
    print("Warming up...")
    carry = (state, rng)
    (state, rng), _ = lax.scan(step_fn, carry, None, length=100)
    
    print(f"Running {num_steps:,} steps...")
    start_time = time.perf_counter()
    
    carry = (state, rng)
    (state, rng), _ = lax.scan(step_fn, carry, None, length=num_steps)
    
    elapsed = time.perf_counter() - start_time
    steps_per_sec = num_steps / elapsed
    
    print(f"Time elapsed: {elapsed:.2f} seconds")
    print(f"Steps per second: {steps_per_sec:,.0f}")
    print(f"Time per step: {elapsed / num_steps * 1e6:.2f} microseconds")


if __name__ == "__main__":
    main()
