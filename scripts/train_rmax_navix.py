"""
Fully JITted R-Max on Navix DoorKey using lax.scan.

- Environment stepping is JIT-compatible (Navix JAX backend)
- Learning + planning fully inside JAX
- Fixed-size tabular R-Max via hashing
- Single compiled scan over all steps
"""

import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.random as jrng
import navix as nx
import numpy as np
import pickle
from pathlib import Path
from functools import partial
from navix.environments.door_key import *

from rl_research.agents.rmax import RMaxAgent, RMaxState
from rl_research.buffers import Transition
from rl_research.environments import FixedGridDoorKey


NUM_STEPS = 1_000_000
DISCOUNT = 0.99
R_MAX = 1.0
KNOWN_THRESHOLD = 1
OUTPUT_DIR = "./outputs/rmax_navix"

MAX_STATES = 9 * 2 * 2 * 4
NUM_ACTIONS = 5

SEED = 2



def make_step_fn(env: nx.Environment, agent: RMaxAgent):
    @jax.jit
    def step_fn(carry, _):
        timestep, rng, state, ep_return, discount_acc = carry

        is_start = timestep.is_start()
        discount_acc = lax.cond(
            is_start,
            lambda discount_acc: 1.0,
            lambda discount_acc: discount_acc * DISCOUNT,
            discount_acc,
        )

        ep_return = lax.cond(
            is_start,
            lambda ep_return: 0.0,
            lambda ep_return: ep_return,
            ep_return,
        )

        s = env.encode_state(timestep)

        rng, a_rng = jrng.split(rng)
        a = agent.select_action(state, s, a_rng, is_training=True)

        # remap to remove unused actions
        env_a = jnp.where(jnp.equal(a, 4), 5, a)

        timestep = env.step(timestep, env_a)

        sp = env.encode_state(timestep)

        r = timestep.reward

        batch = Transition(
            observation=jnp.array([s]),
            action=jnp.array([a]),
            reward=jnp.array([r]),
            next_observation=jnp.array([sp]),
            terminal=jnp.array([timestep.is_termination()]),
            discount=jnp.array([DISCOUNT]),
        )

        next_state, _ = agent.update(state, batch)

        done = timestep.is_done()
        rng, reset_rng = jrng.split(rng)

        timestep = lax.cond(
            done,
            lambda ts: env.reset(reset_rng),
            lambda ts: ts,
            timestep,
        )

        ep_return += discount_acc * r

        return (timestep, rng, next_state, ep_return, discount_acc), (ep_return, done)

    return step_fn


def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = nx.make(
        "FixedGridDoorKey-5x5-layout1-v0",
    )

    rng = jrng.PRNGKey(SEED)
    rng, reset_rng = jrng.split(rng)
    timestep = env.reset(reset_rng)

    agent = RMaxAgent(
        num_states=MAX_STATES,
        num_actions=NUM_ACTIONS,
        r_max=R_MAX * (1 - DISCOUNT),
        discount=DISCOUNT,
        known_threshold=KNOWN_THRESHOLD,
    )

    state = agent.initial_state()

    step_fn = make_step_fn(env, agent)

    carry = (timestep, rng, state, 0.0, 1.0)

    print(f"Running {NUM_STEPS:,} steps...")

    carry, (episodic_returns, dones) = lax.scan(step_fn, carry, None, length=NUM_STEPS)

    jax.block_until_ready(carry)

    final_timestep, final_rng, final_agent_state, _, _ = carry

    print("Done. Saving results...")

    episodic_returns = np.asarray(episodic_returns)
    dones = np.asarray(dones)

    # Save all results
    results = {
        'episodic_returns': episodic_returns,
        'dones': dones,
        'num_steps': NUM_STEPS,
        'discount': DISCOUNT,
        'r_max': R_MAX,
        'known_threshold': KNOWN_THRESHOLD,
    }
    
    np.save(output_dir / "episodic_returns.npy", episodic_returns)
    np.save(output_dir / "dones.npy", dones)
    
    # Save agent state
    q_table_np = np.asarray(final_agent_state.q_table)
    transition_counts_np = np.asarray(final_agent_state.transition_counts)
    reward_sums_np = np.asarray(final_agent_state.reward_sums)
    visit_counts_np = np.asarray(final_agent_state.visit_counts)
    
    np.save(output_dir / "q_table.npy", q_table_np)
    np.save(output_dir / "transition_counts.npy", transition_counts_np)
    np.save(output_dir / "reward_sums.npy", reward_sums_np)
    np.save(output_dir / "visit_counts.npy", visit_counts_np)
    
    # Save metadata
    with open(output_dir / "metadata.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nSaved all results to {output_dir}/")
    print(f"  - episodic_returns.npy")
    print(f"  - dones.npy")
    print(f"  - q_table.npy (shape: {q_table_np.shape})")
    print(f"  - transition_counts.npy (shape: {transition_counts_np.shape})")
    print(f"  - reward_sums.npy (shape: {reward_sums_np.shape})")
    print(f"  - visit_counts.npy (shape: {visit_counts_np.shape})")
    print(f"  - metadata.pkl")


if __name__ == "__main__":
    main()
