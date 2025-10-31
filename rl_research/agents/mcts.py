"""Monte Carlo Tree Search agent for tabular environments."""

from __future__ import annotations

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import jax.random as jrandom
from flax import struct

from rl_research.agents.base import AgentParams, AgentState, TabularAgent, UpdateResult
from rl_research.policies import ActionSelectionPolicy, GreedyPolicy


@struct.dataclass
class MCTSAgentState(AgentState):
    """Tracks Q-estimates, visitation statistics, and mode flags."""

    visit_counts: jax.Array
    state_counts: jax.Array
    eval: bool = False


@struct.dataclass
class MCTSAgentParams(AgentParams):
    """Static configuration for the Monte Carlo Tree Search agent."""

    dynamics_model: jax.Array
    num_simulations: int
    max_depth: int
    exploration_constant: float
    initial_value: float = 0.0


class MCTSAgent(TabularAgent[MCTSAgentState, MCTSAgentParams]):
    """Monte Carlo Tree Search planner operating on tabular dynamics."""

    def __init__(
        self,
        params: MCTSAgentParams,
        seed: int | None = None,
        policy: ActionSelectionPolicy | None = None,
    ) -> None:
        if params.dynamics_model is None:
            raise ValueError("`dynamics_model` must be provided for MCTSAgent.")

        simulations = int(params.num_simulations)
        depth = int(params.max_depth)
        if simulations <= 0:
            raise ValueError("`num_simulations` must be a positive integer.")
        if depth <= 0:
            raise ValueError("`max_depth` must be a positive integer.")

        dynamics = jnp.asarray(params.dynamics_model)
        if (
            dynamics.ndim != 3
            or dynamics.shape[0] != params.num_states
            or dynamics.shape[1] != params.num_actions
            or dynamics.shape[2] < 2
        ):
            raise ValueError(
                "Expected `dynamics_model` with shape "
                f"(num_states={params.num_states}, num_actions={params.num_actions}, >=2). "
                f"Received shape {dynamics.shape}."
            )

        next_obs = dynamics[..., 0]
        rewards = dynamics[..., 1]

        max_state = jnp.asarray(params.num_states - 1, dtype=jnp.int32)
        min_state = jnp.asarray(0, dtype=jnp.int32)
        rounded_next_obs = jnp.rint(next_obs).astype(jnp.int32)
        self._model_next_obs = jnp.clip(rounded_next_obs, min_state, max_state)
        self._model_rewards = jnp.asarray(rewards, dtype=jnp.float32)

        self._num_simulations = simulations
        self._max_depth = depth
        self._discount = jnp.asarray(params.discount, dtype=jnp.float32)
        self._exploration_train = jnp.asarray(
            float(params.exploration_constant), dtype=jnp.float32
        )
        self._initial_value = float(params.initial_value)
        self._num_actions = int(params.num_actions)
        self._action_indices = jnp.arange(self._num_actions, dtype=jnp.int32)

        super().__init__(params=params, seed=seed, policy=policy)

    def _default_policy(self) -> ActionSelectionPolicy:
        return GreedyPolicy()

    def _policy_extras(
        self, state: MCTSAgentState, obs: int
    ) -> Dict[str, jnp.ndarray]:
        counts_row = state.visit_counts[obs]
        total = jnp.sum(counts_row) + 1.0
        return {"counts": counts_row.astype(jnp.float32), "total": total}

    def _initial_state(self, key: jax.Array) -> MCTSAgentState:
        q_values = jnp.full(
            (self.num_states, self.num_actions),
            self._initial_value,
            dtype=jnp.float32,
        )
        visit_counts = jnp.zeros_like(q_values, dtype=jnp.float32)
        state_counts = jnp.zeros((self.num_states,), dtype=jnp.float32)
        return MCTSAgentState(
            q_values=q_values,
            rng=key,
            visit_counts=visit_counts,
            state_counts=state_counts,
            eval=False,
        )

    def select_action(
        self, state: MCTSAgentState, obs: jax.Array
    ) -> Tuple[jax.Array, MCTSAgentState, Dict[str, float]]:
        obs_idx = jnp.asarray(obs, dtype=jnp.int32)

        if state.eval:
            q_row = state.q_values[obs_idx]
            extras = self._policy_extras(state, obs_idx)
            action, new_rng, policy_info = self._policy.select(state.rng, q_row, extras)
            info = {
                **policy_info,
                "q_row": q_row,
                "visit_counts": state.visit_counts[obs_idx],
            }
            return action, state.replace(rng=new_rng), info

        exploration = self._exploration_train
        (
            q_values,
            visit_counts,
            state_counts,
            rng,
            root_values,
            root_counts,
        ) = self._run_mcts(
            state.q_values,
            state.visit_counts,
            state.state_counts,
            obs_idx,
            state.rng,
            exploration,
        )

        extras = {
            "counts": visit_counts[obs_idx].astype(jnp.float32),
            "total": jnp.sum(visit_counts[obs_idx]) + 1.0,
        }
        action, new_rng, policy_info = self._policy.select(rng, root_values, extras)

        info = {
            **policy_info,
            "plan_values": root_values,
            "visit_counts": root_counts,
            "q_row": state.q_values[obs_idx],
        }

        new_state = state.replace(
            q_values=q_values,
            visit_counts=visit_counts,
            state_counts=state_counts,
            rng=new_rng,
        )
        return action, new_state, info

    def update(
        self,
        agent_state: MCTSAgentState,
        obs: jax.Array,
        action: jax.Array,
        reward: jax.Array,
        next_obs: jax.Array,
        terminated: jax.Array | bool = False,
    ) -> Tuple[MCTSAgentState, UpdateResult]:
        del obs, action, reward, next_obs, terminated
        return agent_state, UpdateResult()

    def train(self, state: MCTSAgentState) -> MCTSAgentState:
        return state.replace(eval=False)

    def eval(self, state: MCTSAgentState) -> MCTSAgentState:
        return state.replace(eval=True)

    def _run_mcts(
        self,
        q_values: jax.Array,
        sa_counts: jax.Array,
        state_counts: jax.Array,
        root_state: jax.Array,
        rng: jax.Array,
        exploration: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """Executes `num_simulations` tree-search rollouts."""

        def simulate_body(_, carry):
            q_vals, counts, state_visits, key = carry
            key, sim_key = jrandom.split(key)
            q_vals, counts, state_visits = self._simulate_once(
                q_vals, counts, state_visits, root_state, sim_key, exploration
            )
            return q_vals, counts, state_visits, key

        q_values, sa_counts, state_counts, rng = jax.lax.fori_loop(
            0,
            self._num_simulations,
            simulate_body,
            (q_values, sa_counts, state_counts, rng),
        )

        root_values = q_values[root_state]
        root_counts = sa_counts[root_state]
        return q_values, sa_counts, state_counts, rng, root_values, root_counts

    def _simulate_once(
        self,
        q_values: jax.Array,
        sa_counts: jax.Array,
        state_counts: jax.Array,
        root_state: jax.Array,
        rng: jax.Array,
        exploration: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Runs a single selection-expansion-backup pass."""

        path_states = jnp.zeros((self._max_depth,), dtype=jnp.int32)
        path_actions = jnp.zeros((self._max_depth,), dtype=jnp.int32)
        path_rewards = jnp.zeros((self._max_depth,), dtype=jnp.float32)

        def selection_body(step, carry):
            state_idx, depth, done, states, actions, rewards, key = carry
            operand = (state_idx, depth, states, actions, rewards, key)

            def do_step(args):
                state_idx, depth, states, actions, rewards, key = args
                key, select_key = jrandom.split(key)
                random_key, untried_key, tie_key = jrandom.split(select_key, 3)

                counts_row = sa_counts[state_idx]
                q_row = q_values[state_idx]
                state_visits = state_counts[state_idx]

                def sample_random() -> jax.Array:
                    return jrandom.randint(
                        random_key,
                        shape=(),
                        minval=0,
                        maxval=self._num_actions,
                        dtype=jnp.int32,
                    )

                def sample_untried(mask: jax.Array) -> jax.Array:
                    probs = mask.astype(jnp.float32)
                    probs = probs / jnp.sum(probs)
                    return jrandom.choice(untried_key, self._action_indices, p=probs)

                def sample_ucb() -> jax.Array:
                    log_term = jnp.log(state_visits + 1.0)
                    bonuses = jnp.sqrt(log_term / (counts_row + 1.0))
                    scores = q_row + exploration * bonuses
                    best = jnp.max(scores)
                    mask = jnp.where(scores == best, 1.0, 0.0)
                    probs = mask / jnp.sum(mask)
                    return jrandom.choice(tie_key, self._action_indices, p=probs)

                zero_mask = counts_row == 0.0

                action = jax.lax.cond(
                    state_visits < 1.0,
                    lambda _: sample_random(),
                    lambda _: jax.lax.cond(
                        jnp.any(zero_mask),
                        lambda _: sample_untried(zero_mask),
                        lambda _: sample_ucb(),
                        operand=None,
                    ),
                    operand=None,
                )

                reward = self._model_rewards[state_idx, action]
                next_state = self._model_next_obs[state_idx, action]

                states = states.at[depth].set(state_idx)
                actions = actions.at[depth].set(action)
                rewards = rewards.at[depth].set(reward)

                counts_action = sa_counts[state_idx, action]
                expanded = jnp.logical_or(state_visits < 1.0, counts_action == 0.0)
                new_depth = depth + 1
                depth_cap = new_depth >= self._max_depth
                new_done = jnp.logical_or(expanded, depth_cap)

                return (
                    next_state,
                    new_depth,
                    new_done,
                    states,
                    actions,
                    rewards,
                    key,
                )

            state_idx, depth, done, states, actions, rewards, key = jax.lax.cond(
                done,
                lambda _: (state_idx, depth, done, states, actions, rewards, key),
                do_step,
                operand,
            )
            return state_idx, depth, done, states, actions, rewards, key

        init_carry = (
            root_state,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(False),
            path_states,
            path_actions,
            path_rewards,
            rng,
        )
        state_idx, depth, _, path_states, path_actions, path_rewards, rng = jax.lax.fori_loop(
            0, self._max_depth, selection_body, init_carry
        )

        state_counts = state_counts.at[state_idx].add(1.0)
        value = jnp.asarray(0.0, dtype=jnp.float32)

        def backup_body(step, carry):
            q_vals, counts, state_visits, val = carry

            def update(args):
                q_vals, counts, state_visits, val = args
                idx = depth - 1 - step
                st = path_states[idx]
                act = path_actions[idx]
                rew = path_rewards[idx]
                val = rew + self._discount * val
                count = counts[st, act] + 1.0
                q_old = q_vals[st, act]
                q_new = q_old + (val - q_old) / count
                q_vals = q_vals.at[st, act].set(q_new)
                counts = counts.at[st, act].set(count)
                state_visits = state_visits.at[st].add(1.0)
                return q_vals, counts, state_visits, val

            return jax.lax.cond(
                step < depth,
                update,
                lambda args: args,
                (q_vals, counts, state_visits, val),
            )

        q_values, sa_counts, state_counts, value = jax.lax.fori_loop(
            0,
            self._max_depth,
            backup_body,
            (q_values, sa_counts, state_counts, value),
        )
        del value
        return q_values, sa_counts, state_counts


__all__ = ["MCTSAgent", "MCTSAgentParams", "MCTSAgentState"]
