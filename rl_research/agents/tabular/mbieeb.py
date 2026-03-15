import gin
import jax
import jax.numpy as jnp
from flax import struct
from typing import Tuple
import distrax

from rl_research.buffers import Transition
from rl_research.agents.tabular.rmax import RMaxAgent, RMaxState


@gin.configurable
class MBIEEBAgent(RMaxAgent):
    """MBIE-EB agent."""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        exploration_bonus: float,
        r_max: float = 1.0,
        v_max: float = 1.0,
        use_vmax: bool = False,
        discount: float = 0.9,
        known_threshold: int | None = None,
        convergence_threshold: float = 1e-6,
    ):
        self.terminal_state = num_states
        self.num_states = num_states + 1
        self.num_actions = num_actions
        self.discount = discount
        self.exploration_bonus = exploration_bonus * r_max
        self.known_threshold = jnp.inf if known_threshold is None else known_threshold
        self.convergence_threshold = convergence_threshold
        self.vi_iterations = (
            jnp.ceil(
                jnp.log(1 / (convergence_threshold * (1 - discount))) / (1 - discount)
            )).astype(int)
        self.optimistic_value = v_max if use_vmax else (r_max / (1.0 - discount))


    def update(
        self, state: RMaxState, batch: Transition
    ) -> tuple[RMaxState, jax.Array]:
        """Update model and recompute Q-values."""
        s = batch.observation.astype(jnp.int32)
        a = batch.action.astype(jnp.int32)
        r = batch.reward
        s_next = batch.next_observation.astype(jnp.int32)
        terminal = batch.terminal.astype(jnp.bool)

        s_next = jnp.where(terminal, self.terminal_state, s_next)
        
        def update_model(state: RMaxState) -> RMaxState:
            return state.replace(
                transition_counts=state.transition_counts.at[s, a, s_next].add(1.0),
                reward_sums=state.reward_sums.at[s, a].add(r),
                visit_counts=state.visit_counts.at[s, a].add(1.0)
            )

        is_unknown = (state.visit_counts[s, a] < self.known_threshold).squeeze()
        state = jax.lax.cond(
            is_unknown,
            lambda st: update_model(st),
            lambda st: st,
            state
        )

        def run_value_iteration(state: RMaxState):
            safe_counts = jnp.maximum(state.visit_counts, 1)
            avg_rewards = state.reward_sums / safe_counts
            transition_probs = state.transition_counts / safe_counts[:, :, None]

            def not_converged(carry):
                _, delta, it_step = carry
                return jnp.logical_and(delta > self.convergence_threshold, it_step < self.vi_iterations)

            def _vi_step(carry):
                q_table, delta, it_step = carry
                
                v_table = jnp.max(q_table, axis=1)
                v_table = v_table.at[self.terminal_state].set(0.0)

                expected_next_v = jnp.einsum(
                    "sat,t->sa",
                    transition_probs,
                    v_table
                )

                bonus = jnp.where(
                    state.visit_counts > 0,
                    self.exploration_bonus / jnp.sqrt(state.visit_counts),
                    0.0
                )

                q_known = avg_rewards + self.discount * expected_next_v + bonus
                new_q = jnp.where(
                    state.visit_counts > 0,
                    q_known,
                    self.optimistic_value
                )
                new_q = new_q.at[self.terminal_state].set(0.0)
  
                return new_q, jnp.max(jnp.abs(new_q - q_table)), it_step + 1

            q_final, final_delta, _ = jax.lax.while_loop(not_converged, _vi_step, (state.q_table, jnp.inf, 0))
            return state.replace(
                q_table=q_final
            ), final_delta

        return run_value_iteration(state)
