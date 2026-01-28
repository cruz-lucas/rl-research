"""
DQN training script using Dopamine on FixedGridDoorKey environment.

Uses Dopamine's JAX DQN implementation with:
- 2-layer MLP with 64 hidden units
- Replay buffer: 128k capacity, 8k fill, 512 batch size
- Learning rate: 0.0003, max grad norm: 1.0
- Gamma: 0.95, epsilon decay over 50% of training
- Target update every 2k steps
"""

import jax
import jax.numpy as jnp
import jax.random as jrng
import navix as nx
import numpy as np
import pickle
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from pathlib import Path

from rl_research.environments import FixedGridDoorKey
import optax


# Configuration
NUM_STEPS = 1_000_000
ENV_ID = "FixedGridDoorKey-5x5-layout1-v0"
OUTPUT_DIR = f"./outputs/dqn_dopamine/{ENV_ID}"

# Agent hyperparameters
BUFFER_SIZE = 128_000
FILL_BUFFER = 8_000  # min_replay_history
BATCH_SIZE = 512
LEARNING_RATE = 0.0003
MAX_GRAD_NORM = 1.0
GAMMA = 0.95
EPS_START = 1.0
EPS_END = 0.01
EXPLORATION_FRACTION = 0.5
TARGET_UPDATE_FREQ = 2_000
NUM_EPOCHS = 1
NUM_ENVS = 1  # Single environment for now
ACTIVATION = "relu"

# Network configuration
HIDDEN_UNITS = 64
NUM_HIDDEN_LAYERS = 2

# Training configuration
UPDATE_PERIOD = 4  # Update every 4 steps
SEED = 0


class MLPQNetwork(nn.Module):
    """Simple 2-layer MLP Q-network."""
    
    num_actions: int
    hidden_units: int = 64
    num_hidden_layers: int = 2
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        x = nn.relu(x)
        
        x = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        x = nn.relu(x)

        q_values = nn.Dense(7, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            x
        )
        
        return q_values


def create_network(num_actions: int, observation_shape: tuple):
    """Create and initialize the Q-network."""
    network_def = MLPQNetwork(num_actions=num_actions)
    
    # Create dummy input for initialization
    rng = jrng.PRNGKey(SEED)
    # Input shape depends on observation format
    if len(observation_shape) == 3:  # (H, W, C)
        dummy_input = jnp.ones((1, observation_shape[0], observation_shape[1], observation_shape[2]))
    else:
        dummy_input = jnp.ones((1,) + observation_shape)
    
    params = network_def.init(rng, dummy_input)
    return network_def, params


def preprocess_observation(obs):
    """Preprocess observation: normalize pixel values."""
    obs = jnp.asarray(obs, dtype=jnp.float32)
    # Don't normalize for now, besides, it's not rgb
    # if obs.max() > 1.0:
    #     obs = obs / 255.0
    return obs


def create_optimizer():
    """Create optimizer with gradient clipping."""
    optimizer = optax.chain(
        optax.clip_by_global_norm(MAX_GRAD_NORM),
        optax.adam(learning_rate=LEARNING_RATE),
    )
    return optimizer


def epsilon_schedule(step: int, total_steps: int) -> float:
    """Linear epsilon decay schedule."""
    decay_steps = int(EXPLORATION_FRACTION * total_steps)
    if step < decay_steps:
        progress = step / decay_steps
        epsilon = EPS_START - (EPS_START - EPS_END) * progress
        return float(epsilon)
    else:
        return EPS_END


def select_action(
    network_def,
    params,
    state,
    rng,
    num_actions: int,
    epsilon: float,
) -> tuple:
    """Select action using epsilon-greedy policy."""
    rng, key_explore = jrng.split(rng)
    
    if jrng.uniform(key_explore) < epsilon:
        rng, key_action = jrng.split(rng)
        action = jrng.randint(key_action, (), 0, num_actions)
    else:
        q_vals = network_def.apply(params, state[None])[0]
        action = jnp.argmax(q_vals)
    
    return rng, int(action)


def compute_td_target(
    network_def,
    target_params,
    next_states,
    rewards,
    terminals,
) -> jnp.ndarray:
    """Compute TD target using target network."""
    next_q_vals = jax.vmap(
        lambda s: network_def.apply(target_params, s[None]).q_values[0]
    )(next_states)
    
    max_next_q = jnp.max(next_q_vals, axis=1)
    
    target = rewards + GAMMA * max_next_q * (1.0 - terminals)
    return jax.lax.stop_gradient(target)


def update_step(
    network_def,
    online_params,
    target_params,
    optimizer,
    optimizer_state,
    states,
    actions,
    next_states,
    rewards,
    terminals,
):
    """Single training step."""
    
    def loss_fn(params):
        # Get Q-values for actions taken
        q_vals = jax.vmap(
            lambda s: network_def.apply(params, s[None]).q_values[0]
        )(states)
        q_selected = jnp.take_along_axis(q_vals, actions[:, None], axis=1).squeeze()
        
        # Compute targets
        target_q = compute_td_target(network_def, target_params, next_states, rewards, terminals)
        
        # MSE loss
        loss = jnp.mean((q_selected - target_q) ** 2)
        return loss
    
    loss, grads = jax.value_and_grad(loss_fn)(online_params)
    updates, optimizer_state = optimizer.update(grads, optimizer_state)
    online_params = optax.apply_updates(online_params, updates)
    
    return online_params, optimizer_state, loss


def run_training():
    """Main training loop."""
    
    # Setup
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize environment
    env = nx.make(ENV_ID)
    rng = jrng.PRNGKey(SEED)
    rng, reset_rng = jrng.split(rng)
    timestep = env.reset(reset_rng)
    
    # Get environment info
    obs = preprocess_observation(timestep.observation)
    num_actions = env.action_space.n
    print(f"Observation shape: {obs.shape}")
    print(f"Number of actions: {num_actions}")
    
    # Create network and optimizer
    network_def, online_params = create_network(num_actions, obs.shape)
    target_params = online_params
    optimizer = create_optimizer()
    optimizer_state = optimizer.init(online_params)
    
    # Training state
    step = 0
    episode = 0
    episode_return = 0.0
    episode_length = 0
    episode_returns = []
    episode_lengths = []
    losses = []
    
    print(f"\nStarting training for {NUM_STEPS:,} steps...")
    print(f"Buffer size: {BUFFER_SIZE}, Batch size: {BATCH_SIZE}")
    print(f"Fill buffer: {FILL_BUFFER} steps, Target update: every {TARGET_UPDATE_FREQ} steps")
    print(f"Learning rate: {LEARNING_RATE}, Gamma: {GAMMA}")
    print()
    
    # Training loop
    replay_buffer_transitions = []
    
    while step < NUM_STEPS:
        epsilon = epsilon_schedule(step, NUM_STEPS)
        
        # Select and execute action
        obs_processed = preprocess_observation(timestep.observation)
        rng, action = select_action(network_def, online_params, obs_processed, rng, num_actions, epsilon)
        
        # Step environment
        last_obs = obs_processed
        timestep = env.step(timestep, action)
        obs_new = preprocess_observation(timestep.observation)
        reward = float(timestep.reward)
        done = bool(timestep.is_termination() or timestep.is_truncation())
        
        # Store transition
        transition = {
            'state': last_obs,
            'action': action,
            'reward': reward,
            'next_state': obs_new,
            'terminal': timestep.is_termination(),
        }
        replay_buffer_transitions.append(transition)
        
        # Keep buffer at max size
        if len(replay_buffer_transitions) > BUFFER_SIZE:
            replay_buffer_transitions.pop(0)
        
        # Update statistics
        episode_return += reward
        episode_length += 1
        
        # Training update (after filling buffer)
        if step >= FILL_BUFFER and step % UPDATE_PERIOD == 0 and len(replay_buffer_transitions) >= BATCH_SIZE:
            # Sample batch
            batch_indices = np.random.choice(len(replay_buffer_transitions), BATCH_SIZE, replace=False)
            batch = [replay_buffer_transitions[i] for i in batch_indices]
            
            # Stack batch
            states = jnp.stack([jnp.asarray(t['state']) for t in batch])
            actions = jnp.array([t['action'] for t in batch])
            rewards = jnp.array([t['reward'] for t in batch], dtype=jnp.float32)
            next_states = jnp.stack([jnp.asarray(t['next_state']) for t in batch])
            terminals = jnp.array([t['terminal'] for t in batch], dtype=jnp.float32)
            
            # Update network
            online_params, optimizer_state, loss = update_step(
                network_def,
                online_params,
                target_params,
                optimizer,
                optimizer_state,
                states,
                actions,
                next_states,
                rewards,
                terminals,
            )
            losses.append(float(loss))
        
        # Sync target network
        if step > 0 and step % TARGET_UPDATE_FREQ == 0:
            target_params = online_params
        
        # Episode end
        if done:
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
            
            if (episode + 1) % 10 == 0:
                avg_return = np.mean(episode_returns[-10:])
                avg_length = np.mean(episode_lengths[-10:])
                avg_loss = np.mean(losses[-100:]) if losses else 0.0
                print(
                    f"Episode {episode + 1:5d} | "
                    f"Step {step:7d} | "
                    f"Return {avg_return:7.2f} | "
                    f"Length {avg_length:6.1f} | "
                    f"Loss {avg_loss:.4f} | "
                    f"Epsilon {epsilon:.4f}"
                )
            
            # Reset episode
            episode_return = 0.0
            episode_length = 0
            episode += 1
            rng, reset_rng = jrng.split(rng)
            timestep = env.reset(reset_rng)
        
        step += 1
    
    # Save results
    print("\nTraining complete. Saving results...")
    
    results = {
        'episode_returns': np.array(episode_returns),
        'episode_lengths': np.array(episode_lengths),
        'losses': np.array(losses),
        'online_params': online_params,
        'target_params': target_params,
    }
    
    np.save(output_dir / 'episode_returns.npy', results['episode_returns'])
    np.save(output_dir / 'episode_lengths.npy', results['episode_lengths'])
    np.save(output_dir / 'losses.npy', results['losses'])
    
    metadata = {
        'num_steps': NUM_STEPS,
        'env_id': ENV_ID,
        'buffer_size': BUFFER_SIZE,
        'fill_buffer': FILL_BUFFER,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'max_grad_norm': MAX_GRAD_NORM,
        'gamma': GAMMA,
        'eps_start': EPS_START,
        'eps_end': EPS_END,
        'exploration_fraction': EXPLORATION_FRACTION,
        'target_update_freq': TARGET_UPDATE_FREQ,
        'update_period': UPDATE_PERIOD,
        'hidden_units': HIDDEN_UNITS,
        'num_hidden_layers': NUM_HIDDEN_LAYERS,
        'activation': ACTIVATION,
        'seed': SEED,
        'num_episodes': len(episode_returns),
    }
    
    with open(output_dir / 'metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\nSaved results to {output_dir}/")
    print(f"  Total steps: {NUM_STEPS:,}")
    print(f"  Total episodes: {len(episode_returns)}")
    print(f"  Avg return (last 10): {np.mean(episode_returns[-10:]):.2f}")
    print(f"  Avg length (last 10): {np.mean(episode_lengths[-10:]):.1f}")


if __name__ == '__main__':
    run_training()
