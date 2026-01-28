"""
Fully JITted Batch Model-Free R-Max on Navix DoorKey using lax.scan with replay buffer.

- Environment stepping is JIT-compatible (Navix JAX backend)
- Replay buffer for trajectory data
- Batch updates from replay buffer
- Fixed-size tabular BMFRMax via hashing
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

from rl_research.agents.batch_modelfree_rmax import BMFRmaxAgent, BMFRmaxState
from rl_research.buffers import Transition, BufferState
from rl_research.environments import FixedGridDoorKey


NUM_STEPS = 1_000_000
DISCOUNT = 0.99
R_MAX = 1.0
KNOWN_THRESHOLD = 1
STEP_SIZE = 0.1
ENV_ID = "FixedGridDoorKey-16x16-layout3-v0"

OUTPUT_DIR = f"./outputs/rb_rmax_navix/{ENV_ID}"

MAX_STATES = 14 ** 2 * 2 * 2 * 4
NUM_ACTIONS = 5

NUM_SEEDS = 1
BASE_SEED = 0

# Replay buffer config
BUFFER_SIZE = 3136
BATCH_SIZE = 256
WARMUP_STEPS = 0  # Number of steps before training begins
UPDATE_FREQUENCY = 1  # Update every N environment steps
NUM_MINIBATCHES = 16  # Number of minibatches to sample per training step


def make_step_fn(env: nx.Environment, agent: BMFRmaxAgent):
    @jax.jit
    def step_fn(carry, step_idx):
        timestep, rng, agent_state, buffer_state, ep_return, discount_acc = carry

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
        a = agent.select_action(agent_state, s, a_rng, is_training=True)

        env_a = jnp.where(jnp.equal(a, 4), 5, a)

        timestep = env.step(timestep, env_a)

        sp = env.encode_state(timestep)

        r = timestep.reward

        batch = Transition(
            observation=s,
            action=a,
            reward=r,
            next_observation=sp,
            terminal=timestep.is_termination().astype(jnp.bool),
            discount=jnp.array(DISCOUNT, dtype=jnp.float32),
        )

        # Add transition to replay buffer
        buffer_state = buffer_state.push(batch)

        # Update agent after warmup if buffer is ready
        def update_from_buffer():
            def sample_and_update(carry_inner, _):
                agent_state_inner, rng_inner = carry_inner
                rng_inner, sample_rng = jrng.split(rng_inner)
                sampled_batch = buffer_state.sample(sample_rng, BATCH_SIZE)
                new_agent_state_inner, _ = agent.update(agent_state_inner, sampled_batch)
                return (new_agent_state_inner, rng_inner), None

            (new_agent_state, rng_new), _ = lax.scan(
                sample_and_update,
                (agent_state, rng),
                None,
                length=NUM_MINIBATCHES,
            )
            return new_agent_state, rng_new

        def no_update():
            return agent_state, rng

        is_past_warmup = step_idx >= WARMUP_STEPS
        is_update_step = (step_idx - WARMUP_STEPS) % UPDATE_FREQUENCY == 0
        buffer_ready = buffer_state.is_ready(BATCH_SIZE)
        should_update = is_past_warmup & is_update_step & buffer_ready

        next_agent_state, rng = lax.cond(
            should_update,
            update_from_buffer,
            no_update,
        )

        done = timestep.is_done()
        rng, reset_rng = jrng.split(rng)

        timestep = lax.cond(
            done,
            lambda ts: env.reset(reset_rng),
            lambda ts: ts,
            timestep,
        )

        ep_return += discount_acc * r

        return (timestep, rng, next_agent_state, buffer_state, ep_return, discount_acc), (ep_return, done)

    return step_fn


def run_one_seed(seed: jax.Array):
    env = nx.make(
        ENV_ID,
    )

    rng = jrng.PRNGKey(seed)
    rng, reset_rng = jrng.split(rng)
    timestep = env.reset(reset_rng)

    agent = BMFRmaxAgent(
        num_states=MAX_STATES,
        num_actions=NUM_ACTIONS,
        r_max=R_MAX * (1 - DISCOUNT),
        discount=DISCOUNT,
        step_size=STEP_SIZE,
        known_threshold=KNOWN_THRESHOLD,
    )

    agent_state = agent.initial_state()

    # Initialize replay buffer
    buffer_state = BufferState(
        observations=jnp.zeros((BUFFER_SIZE, 1), dtype=jnp.int32),
        actions=jnp.zeros((BUFFER_SIZE, 1), dtype=jnp.int32),
        rewards=jnp.zeros((BUFFER_SIZE, 1), dtype=jnp.float32),
        discounts=jnp.zeros((BUFFER_SIZE, 1), dtype=jnp.float32),
        next_observations=jnp.zeros((BUFFER_SIZE, 1), dtype=jnp.int32),
        terminals=jnp.zeros((BUFFER_SIZE, 1), dtype=jnp.bool_),
        position=0,
        size=0,
    )

    step_fn = make_step_fn(env, agent)

    carry = (timestep, rng, agent_state, buffer_state, 0.0, 1.0)

    carry, (episodic_returns, dones) = lax.scan(
        step_fn, carry, jnp.arange(NUM_STEPS), length=NUM_STEPS
    )

    final_timestep, final_rng, final_agent_state, final_buffer_state, _, _ = carry

    return episodic_returns, dones, final_agent_state


def run_many_seeds(seeds):
    # return jax.vmap(run_one_seed)(seeds)
    return run_one_seed(seeds)


def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = jnp.arange(BASE_SEED, BASE_SEED + NUM_SEEDS)[0]

    print(f"Running {NUM_SEEDS} seeds × {NUM_STEPS:,} steps...")
    print(f"Replay buffer size: {BUFFER_SIZE}, batch size: {BATCH_SIZE}")
    print(f"Warmup steps: {WARMUP_STEPS}, minibatches per step: {NUM_MINIBATCHES}")

    episodic_returns, dones, final_states = run_many_seeds(seeds)

    jax.block_until_ready(episodic_returns)

    print("Done. Saving results...")

    episodic_returns = np.asarray(episodic_returns)
    dones = np.asarray(dones)

    # Save per-seed trajectories
    np.save(output_dir / "episodic_returns.npy", episodic_returns)
    np.save(output_dir / "dones.npy", dones)

    # Save per-seed agent state
    np.save(
        output_dir / "q_table.npy",
        np.asarray(final_states.q_table),
    )
    np.save(
        output_dir / "visit_counts.npy",
        np.asarray(final_states.visit_counts),
    )

    metadata = {
        "num_seeds": NUM_SEEDS,
        "num_steps": NUM_STEPS,
        "discount": DISCOUNT,
        "r_max": R_MAX,
        "known_threshold": KNOWN_THRESHOLD,
        "base_seed": BASE_SEED,
        "buffer_size": BUFFER_SIZE,
        "batch_size": BATCH_SIZE,
        "warmup_steps": WARMUP_STEPS,
        "update_frequency": UPDATE_FREQUENCY,
        "num_minibatches": NUM_MINIBATCHES,
    }

    with open(output_dir / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"\nSaved all results to {output_dir}/")
    print(f"  episodic_returns.npy: {episodic_returns.shape}")
    print(f"  dones.npy: {dones.shape}")
    print(f"  q_table.npy: {final_states.q_table.shape}")


if __name__ == "__main__":
    main()
