"""
Training script for R-Max agent on tabular Minigrid Door Key 16x16.

This script trains the R-Max tabular RL algorithm on a discretized version
of the Minigrid Door Key environment with full observability.

Installation:
    pip install minigrid gymnasium

Usage:
    python scripts/train_rmax_minigrid.py
    python scripts/train_rmax_minigrid.py --output_dir ./rmax_results
"""

from pathlib import Path

from tqdm import tqdm
import jax.numpy as jnp
import jax.random as jrng
import numpy as np
from absl import app, flags

from rl_research.agents.rmax import RMaxAgent, RMaxState
from rl_research.buffers import Transition
from rl_research.environments.tabular_minigrid import create_tabular_minigrid_env


FLAGS = flags.FLAGS
flags.DEFINE_string(
    "output_dir",
    "./outputs/rmax_minigrid",
    "Directory to store experiment outputs.",
)
flags.DEFINE_integer("num_steps", 10_000_000, "Number of training steps.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string(
    "env_id", "MiniGrid-DoorKey-5x5-v0", "Minigrid environment ID."
)
flags.DEFINE_float("r_max", 1.0, "R-Max reward optimism parameter.")
flags.DEFINE_float("discount", 0.99, "Discount factor.")
flags.DEFINE_integer("known_threshold", 5, "Threshold for known state-action pairs.")


class RMaxMinigridTrainer:
    """Trainer for R-Max agent on tabular Minigrid."""
    
    def __init__(
        self,
        output_dir: str,
        env_id: str,
        seed: int,
    ):
        """Initialize the trainer.
        
        Args:
            output_dir: Directory to save outputs
            env_id: Minigrid environment ID
            seed: Random seed
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        
        np.random.seed(seed)
        self.rng = jrng.PRNGKey(seed)
        
        self.train_env = create_tabular_minigrid_env(
            env_id=env_id,
            seed=seed,
        )
        
        self.num_states = self.train_env.observation_space.n
        self.num_actions = self.train_env.num_actions
        
        print(f"Environment: {env_id}")
        print(f"Number of actions: {self.num_actions}")
        
        self.agent = RMaxAgent(
            num_states=540,
            num_actions=self.num_actions,
            r_max=FLAGS.r_max,
            discount=FLAGS.discount,
            known_threshold=FLAGS.known_threshold,
        )
        
        self.agent_state = self.agent.initial_state()
    
    def _extend_agent_state(self, new_num_states: int) -> None:
        """Extend agent state arrays to accommodate new states.
        
        Args:
            new_num_states: New total number of states.
        """
        old_num_states = self.agent.num_states
        self.agent.num_states = new_num_states
        
        # Pre-allocate with a larger buffer to reduce frequency of re-allocations
        # Allocate 20% extra to amortize allocation costs
        buffer_num_states = int(new_num_states * 1.2)
        self.agent.num_states = buffer_num_states
        
        old_q_table = self.agent_state.q_table
        new_q_table = np.full(
            (buffer_num_states, self.num_actions),
            self.agent.optimistic_value,
            dtype=np.float32
        )
        new_q_table[:old_num_states] = np.asarray(old_q_table)
        new_q_table = jnp.asarray(new_q_table)
        
        old_trans_counts = self.agent_state.transition_counts
        new_trans_counts = np.zeros(
            (buffer_num_states, self.num_actions, buffer_num_states),
            dtype=np.float32
        )
        new_trans_counts[:old_num_states, :, :old_num_states] = np.asarray(old_trans_counts)
        new_trans_counts = jnp.asarray(new_trans_counts)
        
        old_reward_sums = self.agent_state.reward_sums
        new_reward_sums = np.zeros(
            (buffer_num_states, self.num_actions),
            dtype=np.float32
        )
        new_reward_sums[:old_num_states] = np.asarray(old_reward_sums)
        new_reward_sums = jnp.asarray(new_reward_sums)
        
        old_visit_counts = self.agent_state.visit_counts
        new_visit_counts = np.zeros(
            (buffer_num_states, self.num_actions),
            dtype=np.float32
        )
        new_visit_counts[:old_num_states] = np.asarray(old_visit_counts)
        new_visit_counts = jnp.asarray(new_visit_counts)
        
        self.agent_state = RMaxState(
            q_table=new_q_table,
            transition_counts=new_trans_counts,
            reward_sums=new_reward_sums,
            visit_counts=new_visit_counts,
            step=self.agent_state.step
        )

    def train(self):
        """Run training loop."""
        print(f"\nStarting training for {FLAGS.num_steps} steps...")
        
        episode_returns = []
        episode_lengths = []
        episode_idx = 0
        step = 0
        
        pbar = tqdm(total = FLAGS.num_steps)
        while step < FLAGS.num_steps:
            state, _ = self.train_env.reset(episode_idx)
            
            episode_return = 0.0
            episode_length = 0
            done = False
            
            while not done and step < FLAGS.num_steps:
                # self.rng, key_action = jrng.split(self.rng)
                state_idx = int(state) if isinstance(state, (int, np.integer)) else state
                q_values = np.asarray(self.agent_state.q_table[state_idx])
                max_val = np.max(q_values)
                max_indices = np.where(q_values == max_val)[0]
                idx = np.random.choice(max_indices)
                action = int(idx)
                # action = np.random.choice(self.train_env.action_space.n)
                
                next_state, reward, terminated, truncated, _ = self.train_env.step(action)
                done = terminated or truncated
                
                transition = Transition(
                    observation=jnp.array([state], dtype=jnp.int32),
                    action=jnp.array([action], dtype=jnp.int32),
                    reward=jnp.array([reward], dtype=jnp.float32),
                    discount=jnp.array([FLAGS.discount], dtype=jnp.float32),
                    next_observation=jnp.array([next_state], dtype=jnp.int32),
                    terminal=jnp.array([terminated], dtype=jnp.bool_),
                )
                
                episode_return += reward
                episode_length += 1
                step += 1
                pbar.update(1)
                state = next_state

                current_max_state = self.train_env.observation_space.n
                if self.agent.num_states < current_max_state:
                    self._extend_agent_state(current_max_state)

                self.agent_state, _ = self.agent.update(self.agent_state, transition)

            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
            episode_idx += 1
            print(
                f"Episode {episode_idx}: Return = {episode_return}, Length = {episode_length}, Num States = {self.train_env.observation_space.n}"
            )
        
        print("\nTraining complete!")
        return episode_returns, episode_lengths
    

def main(argv):
    """Main entry point."""
    del argv  # Unused
    
    trainer = RMaxMinigridTrainer(
        output_dir=FLAGS.output_dir,
        env_id=FLAGS.env_id,
        seed=FLAGS.seed,
    )
    
    returns, lengths = trainer.train()
    
    returns_path = Path(FLAGS.output_dir) / "returns.npy"
    lengths_path = Path(FLAGS.output_dir) / "lengths.npy"
    np.save(returns_path, np.array(returns))
    np.save(lengths_path, np.array(lengths))
    
    print(f"\nSaved results to {FLAGS.output_dir}/")
    print(f"  - returns.npy: {len(returns)} episodes")
    print(f"  - lengths.npy: {len(lengths)} episodes")


if __name__ == "__main__":
    app.run(main)
