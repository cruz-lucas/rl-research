"""Monte Carlo Tree Search agent for tabular environments."""

from __future__ import annotations

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import jax.random as jrandom
from flax import struct

from rl_research.agents.base import AgentParams, TabularAgent, UpdateResult
from rl_research.models.tabular import TabularDynamicsModel
from rl_research.policies import ActionSelectionPolicy, UCBPolicy


@struct.dataclass
class MCTSAgentState:
    """Tracks Q-estimates, visitation statistics, and mode flags."""

    sa_counts: jax.Array
    rng: jax.Array
    eval: bool = False


@struct.dataclass
class MCTSAgentParams(AgentParams):
    """Static configuration for the Monte Carlo Tree Search agent."""

    dynamics_model: TabularDynamicsModel
    num_simulations: int
    max_depth: int
    rollout_length: int
    ucb_bonus: int
    tree_policy_bonus: float


class MCTSAgent(TabularAgent[MCTSAgentState, MCTSAgentParams]):
    """Monte Carlo Tree Search planner operating on tabular dynamics."""

    def __init__(
        self,
        params: MCTSAgentParams,
        seed: int | None = None,
        policy: ActionSelectionPolicy | None = None,
    ) -> None:
        model = params.dynamics_model

        if (
            model.num_states != params.num_states
            or model.num_actions != params.num_actions
        ):
            raise ValueError(
                "Dynamics model dimensions do not match agent configuration: "
                f"expected ({params.num_states}, {params.num_actions}), "
                f"received ({model.num_states}, {model.num_actions})."
            )

        self._model = model

        self._num_simulations = params.num_simulations
        self._max_depth = params.max_depth
        self._rollout_length = params.rollout_length
        self._discount = params.discount
        self.tree_policy_bonus = params.tree_policy_bonus
        self._num_actions = int(params.num_actions)
        self._action_indices = jnp.arange(self._num_actions, dtype=jnp.int32)
        self._params = params

        policy = UCBPolicy(confidence=params.ucb_bonus)
        super().__init__(params=params, seed=seed, policy=policy)

    def _default_policy(self) -> ActionSelectionPolicy:
        return UCBPolicy(confidence=self._params.ucb_bonus)

    def _policy_extras(
        self, state: MCTSAgentState, obs: int
    ) -> Dict[str, jnp.ndarray]:
        counts_row = state.sa_counts[obs]
        total = jnp.sum(state.sa_counts)
        return {"counts": counts_row.astype(jnp.float32), "total": total}


    def _initial_state(self, key: jax.Array) -> MCTSAgentState:
        sa_counts = jnp.zeros((self.num_states, self.num_actions), dtype=jnp.float32)
        return MCTSAgentState(
            sa_counts=sa_counts,
            rng=key,
            eval=False,
        )

    def select_action(
        self, state: MCTSAgentState, obs: jax.Array
    ) -> Tuple[jax.Array, MCTSAgentState, Dict[str, float]]:
        obs_idx = jnp.asarray(obs, dtype=jnp.int32)
        rng, mcts_key, action_key = jrandom.split(state.rng, 3)
        exploration = jnp.where(state.eval, 0.0, self.tree_policy_bonus)
        root_values = self._run_mcts(obs_idx, mcts_key, exploration)

        extras = self._policy_extras(state, obs_idx)
        action, _, policy_info = self._policy.select(action_key, root_values, extras)

        info = {
            **policy_info,
        }

        new_state = state.replace(rng=rng)
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
        obs_idx = jnp.asarray(obs, dtype=jnp.int32)
        action_idx = jnp.asarray(action, dtype=jnp.int32)
        reward_val = jnp.asarray(reward, dtype=jnp.float32)
        next_obs_idx = jnp.asarray(next_obs, dtype=jnp.int32)
        terminated_flag = jnp.asarray(terminated, dtype=jnp.bool_)

        self._model.update(
            obs_idx,
            action_idx,
            reward_val,
            next_obs_idx,
            terminated_flag,
        )

        sa_counts = agent_state.sa_counts.at[obs_idx, action_idx].add(1.0)
        agent_state = agent_state.replace(sa_counts=sa_counts)

        return agent_state, UpdateResult()

    def train(self, state: MCTSAgentState) -> MCTSAgentState:
        return state.replace(eval=False)

    def eval(self, state: MCTSAgentState) -> MCTSAgentState:
        return state.replace(eval=True)

    def _run_mcts(
        self,
        root_state: jax.Array,
        rng: jax.Array,
        exploration_bonus: jax.Array,
    ) -> jax.Array:
        """Executes `num_simulations` tree-search rollouts."""

        q_values = jnp.zeros((self.num_states, self.num_actions), dtype=jnp.float32)
        sa_counts = jnp.zeros_like(q_values, dtype=jnp.float32)
        state_counts = jnp.zeros((self.num_states,), dtype=jnp.float32)

        def simulate_body(_, carry):
            q_vals, counts, state_visits, key = carry
            key, sim_key = jrandom.split(key)
            q_vals, counts, state_visits = self._simulate_once(
                q_vals, counts, state_visits, root_state, sim_key, exploration_bonus
            )
            return q_vals, counts, state_visits, key

        q_values, _, _, _ = jax.lax.fori_loop(
            0,
            self._num_simulations,
            simulate_body,
            (q_values, sa_counts, state_counts, rng),
        )

        return q_values[root_state]

    def _simulate_once(
        self,
        q_values: jax.Array,
        sa_counts: jax.Array,
        state_counts: jax.Array,
        root_state: jax.Array,
        rng: jax.Array,
        exploration_bonus: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Performs one selection-expansion-rollout-backup pass."""

        path_states = jnp.zeros((self._max_depth,), dtype=jnp.int32)
        path_actions = jnp.zeros((self._max_depth,), dtype=jnp.int32)
        path_rewards = jnp.zeros((self._max_depth,), dtype=jnp.float32)

        def selection_body(step, carry):
            (
                state_idx,
                depth,
                done,
                states,
                actions,
                rewards,
                key,
            ) = carry
            operand = (state_idx, depth, states, actions, rewards, key)

            def select_step(args):
                state_idx, depth, states, actions, rewards, key = args

                counts_row = sa_counts[state_idx]
                q_row = q_values[state_idx]
                state_visits = state_counts[state_idx]

                key, select_key = jrandom.split(key)
                random_key, untried_key, tie_key = jrandom.split(select_key, 3)

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
                    scores = q_row + exploration_bonus * bonuses
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

                next_state, reward = self._model.query(state_idx, action)

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

            return jax.lax.cond(
                done,
                lambda _: (
                    state_idx,
                    depth,
                    done,
                    states,
                    actions,
                    rewards,
                    key,
                ),
                select_step,
                operand,
            )

        init_carry = (
            root_state,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(False),
            path_states,
            path_actions,
            path_rewards,
            rng,
        )
        (
            leaf_state,
            depth,
            _,
            path_states,
            path_actions,
            path_rewards,
            rng,
        ) = jax.lax.fori_loop(0, self._max_depth, selection_body, init_carry)

        def rollout_step(carry, _):
            state, key, discount = carry
            key, act_key = jrandom.split(key)
            action = jrandom.choice(act_key, self._action_indices)
            next_state, reward = self._model.query(state, action)
            value = discount * reward
            discount = discount * self._discount
            return (next_state, key, discount), value

        rollout_discount = jnp.asarray(1.0, dtype=jnp.float32)
        (_, rng, _), rollout_returns = jax.lax.scan(
            rollout_step,
            (leaf_state, rng, rollout_discount),
            xs=None,
            length=self._rollout_length,
        )
        leaf_value = jnp.sum(rollout_returns)

        state_counts = state_counts.at[leaf_state].add(1.0)

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

        q_values, sa_counts, state_counts, _ = jax.lax.fori_loop(
            0,
            self._max_depth,
            backup_body,
            (q_values, sa_counts, state_counts, leaf_value),
        )

        return q_values, sa_counts, state_counts
        

__all__ = ["MCTSAgent", "MCTSAgentParams", "MCTSAgentState"]
