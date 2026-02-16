"""
Simple script to measure the time to run 10M steps of the goright environment.

Usage:
    python scripts/benchmark_goright.py
"""

import time
import jax
import jax.lax as lax
import jax.random as jrng
import navix as nx


def make_step_fn(env):
    """Factory function to create a step function with env captured in closure."""
    @jax.jit
    def step_fn(carry, _):
        """Single step function for scan."""
        timestep, rng = carry
        rng, action_rng = jrng.split(rng)
        action = jrng.randint(action_rng, shape=(), minval=0, maxval=6)
        timestep = env.step(timestep, action)
        
        timestep = jax.lax.cond(
            jax.numpy.not_equal(timestep.step_type, nx.StepType.TRANSITION),
            lambda s: env.reset(rng),
            lambda s: s,
            timestep
        )
        
        return (timestep, rng), None
    
    return step_fn


def main():
    
    env = nx.make("Navix-DoorKey-Random-16x16-v0", observation_fn=nx.observations.symbolic)
    rng = jrng.PRNGKey(0)
    
    rng, reset_rng = jrng.split(rng)
    timestep = env.reset(reset_rng)

    step_fn = make_step_fn(env)
    
    num_steps = 10_000_000
    
    print("Warming up...")
    carry = (timestep, rng)
    (timestep, rng), _ = lax.scan(step_fn, carry, None, length=100)
    
    print(f"Running {num_steps:,} steps...")
    start_time = time.perf_counter()
    
    carry = (timestep, rng)
    (timestep, rng), _ = lax.scan(step_fn, carry, None, length=num_steps)
    jax.block_until_ready(timestep)
    
    elapsed = time.perf_counter() - start_time
    steps_per_sec = num_steps / elapsed
    
    print(f"Time elapsed: {elapsed:.2f} seconds")
    print(f"Steps per second: {steps_per_sec:,.0f}")
    print(f"Time per step: {elapsed / num_steps * 1e6:.2f} microseconds")


if __name__ == "__main__":
    main()