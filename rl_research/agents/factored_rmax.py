import gin
import jax
import jax.numpy as jnp
from flax import struct

from rl_research.buffers import Transition
from rl_research.policies import _select_greedy


@struct.dataclass
class FactoredRMaxState:
    """State for Factored R-Max agent."""

    q_table: jnp.ndarray
    marginal_transition_counts: list[jnp.ndarray]
    reward_sums: jnp.ndarray
    visit_counts: jnp.ndarray
    step: int


@gin.configurable
class FactoredRMaxAgent:
    """R-Max agent with factored transition model."""

    def __init__(
        self,
        observation_shape: tuple[int, ...],
        num_states: int,
        num_actions: int,
        r_max: float = 6.0,
        discount: float = 0.9,
        known_threshold: int = 1,
        convergence_threshold: float = 1e-2,
    ):
        self.observation_shape = observation_shape
        self.num_states = int(jnp.prod(jnp.array(observation_shape)))

        assert num_states == self.num_states

        self.num_actions = num_actions
        self.num_dims = len(observation_shape)
        self.r_max = r_max
        self.discount = discount
        self.known_threshold = known_threshold
        self.convergence_threshold = convergence_threshold
        self.vi_iterations = int(
            jnp.ceil(
                jnp.log(1 / (convergence_threshold * (1 - discount))) / (1 - discount)
            )
        )
        self.optimistic_value = r_max / (1 - discount)

    def _flat_to_factored(self, flat_obs: jnp.ndarray) -> jnp.ndarray:
        """Convert flat observation index to factored representation."""
        indices = jnp.unravel_index(flat_obs, self.observation_shape)
        return jnp.array(indices)

    def _factored_to_flat(self, factored_obs: jnp.ndarray) -> jnp.ndarray:
        """Convert factored observation to flat index."""
        return jnp.ravel_multi_index(tuple(factored_obs), self.observation_shape, mode="clip")

    def initial_state(self) -> FactoredRMaxState:
        """Initialize with optimistic Q-values and empty marginal counts."""
        marginal_transition_counts = [
            jnp.zeros((dim_size, self.num_actions, dim_size))
            for dim_size in self.observation_shape
        ]
        
        return FactoredRMaxState(
            q_table=jnp.full(
                (self.num_states, self.num_actions), self.optimistic_value
            ),
            marginal_transition_counts=marginal_transition_counts,
            reward_sums=jnp.zeros((self.num_states, self.num_actions)),
            visit_counts=jnp.zeros((self.num_states, self.num_actions)),
            step=0,
        )

    def select_action(
        self, state: FactoredRMaxState, obs: jnp.ndarray, key: jax.Array, is_training: bool
    ) -> jnp.ndarray:
        """Select greedy action with random tie-breaking."""
        q_values = state.q_table[obs]
        return _select_greedy(q_values, key)

    def _compute_joint_transitions(self, marginal_counts, visit_counts):
        """Compute joint transition probabilities from marginal counts."""
        # Compute marginal probabilities for each dimension
        marginal_probs = []
        for dim_idx, counts in enumerate(marginal_counts):
            dim_size = self.observation_shape[dim_idx]
            # counts shape: (dim_size, num_actions, dim_size)
            safe_counts = jnp.maximum(
                jnp.sum(counts, axis=2, keepdims=True), 1
            )
            probs = counts / safe_counts
            marginal_probs.append(probs)
        
        # Compute joint probabilities by multiplying marginals
        # Result shape: (num_states, num_actions, num_states)
        joint_transitions = jnp.zeros((self.num_states, self.num_actions, self.num_states))
        
        def compute_joint_prob(s, a, s_next):
            """Compute P(s_next | s, a) as product of marginals."""
            s_factored = self._flat_to_factored(s)
            s_next_factored = self._flat_to_factored(s_next)
            
            prob = 1.0
            for dim_idx in range(self.num_dims):
                prob *= marginal_probs[dim_idx][s_factored[dim_idx], a, s_next_factored[dim_idx]]
            return prob
        
        # Vectorize over all state-action-nextstate triples
        for s in range(self.num_states):
            for a in range(self.num_actions):
                for s_next in range(self.num_states):
                    joint_transitions = joint_transitions.at[s, a, s_next].set(
                        compute_joint_prob(s, a, s_next)
                    )
        
        return joint_transitions

    def _value_iteration_step(self, q_table, is_known, rewards, transitions):
        """Single step of value iteration."""
        v_table = jnp.max(q_table, axis=1)

        def compute_q(s, a):
            expected_next_value = jnp.sum(transitions[s, a] * v_table)
            q_val = rewards[s, a] + self.discount * expected_next_value
            q_val = jnp.where(is_known[s, a], q_val, self.optimistic_value)
            return q_val

        new_q = jax.vmap(
            lambda s: jax.vmap(lambda a: compute_q(s, a))(jnp.arange(self.num_actions))
        )(jnp.arange(self.num_states))

        return new_q

    def update(
        self, state: FactoredRMaxState, batch: Transition, batch_mask: jnp.ndarray | None = None
    ) -> tuple[FactoredRMaxState, jax.Array]:
        """Update marginal models and recompute Q-values."""
        batch_size = batch.observation.shape[0]
        if batch_mask is None:
            batch_mask = jnp.ones((batch_size,), dtype=bool)
        batch_mask = batch_mask.astype(jnp.bool_)

        new_marginal_counts = state.marginal_transition_counts
        new_reward_sums = state.reward_sums
        new_visit_counts = state.visit_counts
        prev_is_known = state.visit_counts >= self.known_threshold

        def update_body(i, carry):
            marg_counts, rew_sums, vis_counts = carry
            s = batch.observation[i].astype(jnp.int32).squeeze()
            a = batch.action[i].astype(jnp.int32).squeeze()
            r = batch.reward[i]
            s_next = batch.next_observation[i].astype(jnp.int32).squeeze()
            terminal = batch.terminal[i]

            # Convert to factored representation
            s_factored = self._flat_to_factored(s)
            s_next_factored = self._flat_to_factored(s_next)

            valid_unknown = jnp.logical_and(
                batch_mask[i], vis_counts[s, a] < self.known_threshold
            )
            inc = jnp.where(
                jnp.logical_and(valid_unknown, jnp.logical_not(terminal)), 1, 0
            )

            # Update marginal counts for each dimension
            new_marg_counts = []
            for dim_idx in range(self.num_dims):
                dim_counts = marg_counts[dim_idx]
                dim_counts = dim_counts.at[
                    s_factored[dim_idx], a, s_next_factored[dim_idx]
                ].add(inc)
                new_marg_counts.append(dim_counts)
            
            rew_sums = rew_sums.at[s, a].add(r * inc)
            vis_counts = vis_counts.at[s, a].add(inc)
            return new_marg_counts, rew_sums, vis_counts

        new_marginal_counts, new_reward_sums, new_visit_counts = jax.lax.fori_loop(
            0,
            batch_size,
            update_body,
            (new_marginal_counts, new_reward_sums, new_visit_counts),
        )

        q_table = state.q_table
        new_is_known = new_visit_counts >= self.known_threshold
        newly_known = jnp.logical_and(jnp.logical_not(prev_is_known), new_is_known)
        newly_known_any = jnp.any(newly_known)

        def run_value_iteration(q_init):
            safe_counts = jnp.maximum(new_visit_counts, 1)
            avg_rewards = new_reward_sums / safe_counts
            
            # Compute joint transitions from marginals
            transition_probs = self._compute_joint_transitions(
                new_marginal_counts, new_visit_counts
            )

            def body(_, carry):
                q_prev, converged = carry
                new_q = self._value_iteration_step(
                    q_prev, new_is_known, avg_rewards, transition_probs
                )
                max_change = jnp.max(jnp.abs(new_q - q_prev))
                converged_now = max_change < self.convergence_threshold
                converged = jnp.logical_or(converged, converged_now)
                q_next = jnp.where(converged, q_prev, new_q)
                return q_next, converged

            q_final, _ = jax.lax.fori_loop(0, self.vi_iterations, body, (q_init, False))
            return q_final

        q_table = jax.lax.cond(
            newly_known_any, run_value_iteration, lambda q: q, q_table
        )

        new_state = state.replace(
            q_table=q_table,
            marginal_transition_counts=new_marginal_counts,
            reward_sums=new_reward_sums,
            visit_counts=new_visit_counts,
            step=state.step + jnp.sum(batch_mask),
        )

        loss = jnp.mean(jnp.abs(q_table - state.q_table))
        return new_state, loss

    def bootstrap_value(
        self, state: FactoredRMaxState, next_observation: jnp.ndarray
    ) -> jax.Array:
        s_next = next_observation.astype(jnp.int32).squeeze()
        return jnp.max(state.q_table[s_next])