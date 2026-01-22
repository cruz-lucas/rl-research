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

import matplotlib.pyplot as plt

from rl_research.agents.rmax import RMaxAgent, RMaxState
from rl_research.buffers import Transition


NUM_STEPS = 1_000_000
DISCOUNT = 0.99
R_MAX = 1.0
KNOWN_THRESHOLD = 1

MAX_STATES = 550
NUM_ACTIONS = 7


def encode_state(obs: jnp.ndarray, direction: jnp.int32) -> jnp.int32:
    flat = obs.reshape(-1).astype(jnp.int32)
    h = jnp.sum(flat * jnp.arange(flat.size, dtype=jnp.int32))
    h = h + direction * 1315423911
    return jnp.mod(h, MAX_STATES)



def make_step_fn(env: nx.Environment, agent: RMaxAgent):
    @jax.jit
    def step_fn(carry, _):
        timestep, rng, state, ep_return, discount_acc = carry

        obs = timestep.observation
        direction = timestep.state.get_player().direction
        s = encode_state(obs, direction)

        rng, a_rng = jrng.split(rng)
        a = agent.select_action(state, s, a_rng, is_training=True)

        timestep = env.step(timestep, a)

        next_obs = timestep.observation
        next_dir = timestep.state.get_player().direction
        sp = encode_state(next_obs, next_dir)

        r = timestep.reward

        batch = Transition(
            observation=jnp.array([s]),
            action=jnp.array([a]),
            reward=jnp.array([r]),
            next_observation=jnp.array([sp]),
            terminal=jnp.array([timestep.is_termination()]),
            discount=jnp.array([DISCOUNT]),
        )

        state, _ = agent.update(state, batch)

        done = timestep.is_done()
        timestep = lax.cond(
            done,
            lambda ts: env.reset(rng),
            lambda ts: ts,
            timestep,
        )

        discount_acc = discount_acc * DISCOUNT
        ep_return = ep_return + discount_acc * r

        return (timestep, rng, state, ep_return, discount_acc), (ep_return, done)

    return step_fn


def main():
    env = nx.make(
        "Navix-DoorKey-Random-5x5-v0",
        observation_fn=nx.observations.categorical,
    )

    rng = jrng.PRNGKey(0)
    rng, reset_rng = jrng.split(rng)
    timestep = env.reset(reset_rng)

    agent = RMaxAgent(
        num_states=MAX_STATES,
        num_actions=NUM_ACTIONS,
        r_max=R_MAX,
        discount=DISCOUNT,
        known_threshold=KNOWN_THRESHOLD,
    )
    
    state = agent.initial_state()

    step_fn = make_step_fn(env, agent)

    carry = (timestep, rng, state, 0.0, 1.0)
    print(f"Running {NUM_STEPS:,} fully-jitted R-Max steps...")
    carry, (episodic_returns, dones) = lax.scan(step_fn, carry, None, length=NUM_STEPS)

    jax.block_until_ready(carry)

    print("Done.")

    episodic_returns = np.asarray(episodic_returns)
    dones = np.asarray(dones)

    steps = np.arange(NUM_STEPS)

    terminal_steps = steps[dones]
    terminal_returns = episodic_returns[dones]

    plt.plot(terminal_steps, terminal_returns)
    plt.xlabel("Steps")
    plt.ylabel("Episodic Return")
    plt.title("R-Max Episodic Return vs Environment Steps")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
