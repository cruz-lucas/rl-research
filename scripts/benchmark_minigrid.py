"""
Simple script to measure the time to run 10M steps of the goright environment.

Usage:
    python scripts/benchmark_goright.py
"""

import time

import numpy as np
from rl_research.environments.tabular_minigrid import create_tabular_minigrid_env


def main():
    
    env = create_tabular_minigrid_env(
        env_id="MiniGrid-DoorKey-5x5-v0",
        seed=0,
    )
    
    state, _ = env.reset()
    
    num_steps = 10_000_000
    
    for _ in range(100):
        action = np.random.randint(env.num_actions)
        state, reward, terminated, truncated, _ = env.step(action)
    
    print(f"Running {num_steps:,} steps...")
    start_time = time.perf_counter()
    
    for i in range(num_steps):
        action = np.random.randint(env.num_actions)
        state, reward, terminated, truncated, _ = env.step(action)
        
        if terminated or truncated:
            state, obs = env.reset()
    
    elapsed = time.perf_counter() - start_time
    steps_per_sec = num_steps / elapsed
    
    print(f"Time elapsed: {elapsed:.2f} seconds")
    print(f"Steps per second: {steps_per_sec:,.0f}")
    print(f"Time per step: {elapsed / num_steps * 1e6:.2f} microseconds")


if __name__ == "__main__":
    main()
