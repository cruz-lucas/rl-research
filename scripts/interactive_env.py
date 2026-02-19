#!/usr/bin/env python3
"""
Simple Terminal Q-Value Viewer

A lightweight terminal-based viewer for inspecting Q-values from trained checkpoints.

Usage:
    python simple_qvalue_viewer.py
    
Then modify the configuration variables at the top of main() as needed.
"""

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import orbax.checkpoint as ocp
from pathlib import Path
import matplotlib.pyplot as plt

from rl_research.agents import NFQAgent, DQNAgent, DRMAgent
from rl_research.environments.tabular_navix import NavixWrapper
from navix.observations import rgb


def print_qvalues(q_values_np, action_names):
    """Pretty print Q-values with highlighting"""
    print("\n" + "─" * 70)
    print("Q-VALUES")
    print("─" * 70)
    
    best_action = np.argmax(q_values_np)
    
    for i, (action_name, q_val) in enumerate(zip(action_names, q_values_np)):
        is_best = (i == best_action)
        
        # Create a bar visualization
        bar_length = int(abs(q_val) * 10)
        bar = "█" * min(bar_length, 40)
        
        # Color coding using ANSI
        if is_best:
            color = "\033[92m"  # Green
            marker = " ← BEST"
        else:
            color = "\033[94m"  # Blue
            marker = ""
        reset = "\033[0m"
        
        print(f"{color}[{i}] {action_name:12s}: {q_val:8.4f} {bar}{marker}{reset}")
    
    print("─" * 70)
    print(f"Max Q: {np.max(q_values_np):.4f} | Min Q: {np.min(q_values_np):.4f} | Mean Q: {np.mean(q_values_np):.4f}")
    print("─" * 70)


def print_state_info(episode_length, episode_return):
    """Print current state information"""
    print("\n" + "═" * 70)
    print(f"EPISODE INFO: Step {episode_length} | Return {episode_return:.2f}")
    # print(f"Observation: {obs}")
    print("═" * 70)


def main():
    # ==================== CONFIGURATION ====================
    
    # NFQAgent, DQNAgent, DRMAgent

    CHECKPOINT_PATH = "./tmp/ckpt/NFQAgent/checkpoint_950000"
    ENV_ID = "GridDoorKey-5x5-layout1-v0"
    AGENT_CLASS = NFQAgent
    SEED = 0
    
    # =======================================================
    
    print("\n" + "=" * 70)
    print("SIMPLE Q-VALUE VIEWER")
    print("=" * 70)
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Environment: {ENV_ID}")
    print(f"Agent: {AGENT_CLASS.__name__}")
    print("=" * 70)
    
    # Configure environment
    environment = NavixWrapper(ENV_ID)
    
    # Get dimensions
    obs_shape = environment.env.observation_space.shape
    n_states = environment.env.observation_space.n if obs_shape in [(), (1,)] else int(np.prod(np.array(obs_shape)))
    n_actions = environment.env.action_space.n
    
    print(f"\nObservation space: {obs_shape}")
    print(f"Number of states: {n_states}")
    print(f"Number of actions: {n_actions}")
    
    # Initialize agent
    agent = AGENT_CLASS(
        num_states=n_states,
        num_actions=n_actions,
        # grid_size=5
    )
    agent_state = agent.initial_state()
    
    # Load checkpoint
    print(f"\nLoading checkpoint from: {CHECKPOINT_PATH}")
    checkpointer = ocp.StandardCheckpointer()
    _, network_state = nnx.split(agent_state.online_network)
    
    checkpoint_path = Path(CHECKPOINT_PATH).resolve()
    restored_state = checkpointer.restore(checkpoint_path, network_state)
    nnx.update(agent_state.online_network, restored_state)
    
    print("✓ Checkpoint loaded successfully!")
    
    # Reset environment
    key = jax.random.PRNGKey(SEED)
    key, reset_key = jax.random.split(key)
    env_state, obs = environment.reset(reset_key)
    
    episode_return = 0.0
    episode_length = 0
    
    # Action names
    action_names = ["Turn Left", "Turn Right", "Move Forward", "Pickup", "Drop", "Toggle", "Done"]
    
    print("\n" + "=" * 70)
    print("CONTROLS")
    print("=" * 70)
    print("  0-5: Take action")
    print("  r:   Reset environment")
    print("  q:   Quit")
    print("  v:   View current Q-values (without taking action)")
    print("  a:   Auto-play best action")
    print("=" * 70)
    
    # Main interaction loop
    auto_play = False
    plt.ion()
    
    while True:
        # Display current state
        print_state_info(episode_length, episode_return)

        # plot obs as image 
        img = rgb(env_state.state)
        plt.imshow(img)
        plt.axis('off')
        
        # Get Q-values
        q_values = agent_state.online_network(obs.reshape(-1))
        q_values_np = np.array(q_values)  # Remove batch dimension
        
        # Display Q-values
        print_qvalues(q_values_np, action_names)
        
        # Get user input (or auto-play)
        if auto_play:
            action = int(np.argmax(q_values_np))
            print(f"\n[AUTO] Taking best action: {action} ({action_names[action]})")
            user_input = str(action)
        else:
            user_input = input("\nEnter command: ").strip().lower()
        
        # Process command
        if user_input == 'q':
            print("\n👋 Exiting...")
            break
            
        elif user_input == 'r':
            print("\n🔄 Resetting environment...")
            key, reset_key = jax.random.split(key)
            env_state, obs = environment.reset(reset_key)
            episode_return = 0.0
            episode_length = 0
            auto_play = False
            continue
            
        elif user_input == 'v':
            print("\n[Viewing Q-values, no action taken]")
            continue
            
        elif user_input == 'a':
            auto_play = not auto_play
            status = "ENABLED" if auto_play else "DISABLED"
            print(f"\n🤖 Auto-play {status}")
            continue
            
        # Try to parse as action
        try:
            action = int(user_input)
            
            if action < 0 or action >= n_actions:
                print(f"\n❌ Invalid action! Must be 0-{n_actions-1}")
                continue
            
            # Take step
            env_state, next_obs, reward, terminal, truncation, info = environment.step(env_state, action)
            
            # Update tracking
            obs = next_obs
            episode_return += reward
            episode_length += 1
            done = terminal or truncation
            
            # Display result
            print(f"\n✓ Action taken: {action_names[action]}")
            print(f"  Reward: {reward:.2f}")
            
            if done:
                print("\n" + "!" * 70)
                print("EPISODE COMPLETE!")
                print(f"Total Return: {episode_return:.2f}")
                print(f"Episode Length: {episode_length}")
                print("!" * 70)
                print("\nResetting environment...")
                
                key, reset_key = jax.random.split(key)
                env_state, obs = environment.reset(reset_key)
                episode_return = 0.0
                episode_length = 0
                auto_play = False
                
        except ValueError:
            print(f"\n❌ Invalid input: '{user_input}'. Enter 0-{n_actions-1}, 'r', 'v', 'a', or 'q'")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Exiting...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()