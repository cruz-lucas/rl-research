#!/usr/bin/env python3
"""Interactive Navix viewer with live Q-values, RND, and visitation counts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import gin
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tyro
from matplotlib.backend_bases import KeyEvent
from navix.observations import rgb

from rl_research.agents import BaseAgent, DQNRNDAgent
from rl_research.environments import BaseJaxEnv, NavixWrapper
from rl_research.environments.navix import tabular_obs_fn
from rl_research.experiment import restore_agent_checkpoint


ACTION_NAMES = (
    "turn_left",
    "turn_right",
    "forward",
    "pickup",
    "drop",
    "toggle",
    "done",
)

ACTION_KEYS = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "left": 0,
    "right": 1,
    "up": 2,
    "p": 3,
    "d": 4,
    "t": 5,
    "enter": 6,
    "return": 6,
    "space": 6,
    " ": 6,
}

CONTROL_HELP = (
    "Controls",
    "  left/right/up: turn, turn, forward",
    "  p / d / t    : pickup, drop, toggle",
    "  enter / space: done",
    "  0-6          : direct action keys",
    "  r            : reset episode",
    "  q / escape   : quit",
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = {
    "doorkey_5x5_layout1": REPO_ROOT
    / "rl_research/configs/navix/doorkey/5x5_layout1/dqn_rnd.gin",
    "doorkey_16x16_layout1": REPO_ROOT
    / "rl_research/configs/navix/doorkey/16x16_layout1/dqn_rnd.gin",
}


class Preset(str, Enum):
    DOORKEY_5X5_LAYOUT1 = "doorkey_5x5_layout1"
    DOORKEY_16X16_LAYOUT1 = "doorkey_16x16_layout1"


@dataclass
class Args:
    preset: Preset = Preset.DOORKEY_5X5_LAYOUT1
    config: Path | None = None
    checkpoint: Path | None = None
    seed: int = 0
    binding: list[str] = field(default_factory=list)


@dataclass
class Runtime:
    config_path: Path
    environment: BaseJaxEnv
    agent: BaseAgent
    agent_state: Any
    env_state: Any
    obs: jax.Array
    key: jax.Array
    env_id: str
    checkpoint_mode: str
    n_actions: int


@dataclass
class Diagnostics:
    q_values: np.ndarray
    intrinsic_rewards: np.ndarray | None
    total_value: np.ndarray | None
    state_index: int
    state_visits: int
    state_action_visits: list[int]
    best_q_action: int
    best_intrinsic_action: int | None


@dataclass
class EpisodeStats:
    step: int = 0
    episode_return: float = 0.0
    last_action: int | None = None
    last_reward: float = 0.0
    last_terminal: bool = False
    status: str = "Ready"


def _default_config_for_preset(preset: Preset) -> Path:
    return DEFAULT_CONFIGS[preset.value]


def _config_path(args: Args) -> Path:
    if args.config is not None:
        return args.config
    return _default_config_for_preset(args.preset)


def _normalize_observation(
    agent: BaseAgent, agent_state: Any, obs: jax.Array
) -> jax.Array:
    raw_obs = jnp.asarray(obs).reshape(-1)
    normalize = getattr(agent, "_normalize_observation", None)
    if callable(normalize):
        return normalize(agent_state, raw_obs)
    return raw_obs.astype(jnp.float32)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def build_runtime(args: Args) -> Runtime:
    config_path = _config_path(args).resolve()
    gin.clear_config()
    gin.parse_config_files_and_bindings([str(config_path)], args.binding)

    run_bindings = gin.get_bindings("run_loop")
    env_cls = run_bindings.get("env_cls", NavixWrapper)
    agent_cls = run_bindings.get("agent_cls", BaseAgent)

    environment = env_cls()
    obs_shape = environment.env.observation_space.shape
    n_states = (
        environment.env.observation_space.n
        if obs_shape in [(), (1,)]
        else int(np.prod(np.asarray(obs_shape)))
    )
    n_actions = int(environment.env.action_space.n)

    agent = agent_cls(num_states=n_states, num_actions=n_actions, seed=args.seed)
    agent_state = agent.initial_state()

    checkpoint_mode = "random_init"
    if args.checkpoint is not None:
        agent_state, checkpoint_mode = restore_agent_checkpoint(
            agent_state,
            args.checkpoint,
        )

    key = jax.random.PRNGKey(args.seed)
    key, reset_key = jax.random.split(key)
    env_state, obs = environment.reset(reset_key)

    env_id = gin.get_bindings("NavixWrapper").get("env_id", "<unknown>")

    return Runtime(
        config_path=config_path,
        environment=environment,
        agent=agent,
        agent_state=agent_state,
        env_state=env_state,
        obs=obs,
        key=key,
        env_id=env_id,
        checkpoint_mode=checkpoint_mode,
        n_actions=n_actions,
    )


class InteractiveEnvViewer:
    def __init__(self, runtime: Runtime):
        self.runtime = runtime
        self.stats = EpisodeStats(status="Episode reset")
        self.state_visits: dict[int, int] = {}
        self.state_action_visits: dict[tuple[int, int], int] = {}
        self.done = False

        self.fig = plt.figure(figsize=(15, 8), constrained_layout=True)
        grid = self.fig.add_gridspec(
            2, 2, width_ratios=[1.15, 1.0], height_ratios=[1, 1]
        )
        self.render_ax = self.fig.add_subplot(grid[:, 0])
        self.table_ax = self.fig.add_subplot(grid[0, 1])
        self.info_ax = self.fig.add_subplot(grid[1, 1])

        self.render_ax.set_xticks([])
        self.render_ax.set_yticks([])
        self.table_ax.axis("off")
        self.info_ax.axis("off")

        self.image_artist = self.render_ax.imshow(self._render_image())
        self.table_text = self.table_ax.text(
            0.01,
            0.99,
            "",
            va="top",
            ha="left",
            family="monospace",
            fontsize=10,
        )
        self.info_text = self.info_ax.text(
            0.01,
            0.99,
            "",
            va="top",
            ha="left",
            family="monospace",
            fontsize=10,
        )

        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        manager = getattr(self.fig.canvas, "manager", None)
        if manager is not None:
            manager.set_window_title("Navix Interactive Viewer")

        self._register_state_visit()
        self._refresh()

    def _current_state_index(self) -> int:
        return int(jax.device_get(tabular_obs_fn(self.runtime.env_state.state)))

    def _render_image(self) -> np.ndarray:
        return np.asarray(jax.device_get(rgb(self.runtime.env_state.state)))

    def _register_state_visit(self) -> None:
        state_index = self._current_state_index()
        self.state_visits[state_index] = self.state_visits.get(state_index, 0) + 1

    def _compute_q_values(self) -> np.ndarray:
        if not hasattr(self.runtime.agent_state, "online_network"):
            raise NotImplementedError(
                "Interactive viewer requires an agent state with an online_network."
            )

        obs = _normalize_observation(
            self.runtime.agent,
            self.runtime.agent_state,
            self.runtime.obs,
        )
        q_values = self.runtime.agent_state.online_network(obs.reshape(-1))
        return np.asarray(jax.device_get(q_values), dtype=np.float32)

    def _compute_intrinsic_rewards(self) -> np.ndarray | None:
        if not isinstance(self.runtime.agent, DQNRNDAgent):
            return None

        raw_obs = jnp.asarray(self.runtime.obs).reshape(-1)
        rewards: list[float] = []

        if self.runtime.agent.rnd_action_conditioning == "none":
            intrinsic_reward, _ = self.runtime.agent._compute_intrinsic_reward(
                self.runtime.agent_state,
                raw_obs,
                None,
            )
            value = float(jax.device_get(intrinsic_reward))
            rewards = [value for _ in range(self.runtime.n_actions)]
        else:
            for action in range(self.runtime.n_actions):
                intrinsic_reward, _ = self.runtime.agent._compute_intrinsic_reward(
                    self.runtime.agent_state,
                    raw_obs,
                    jnp.asarray(action, dtype=jnp.int32),
                )
                rewards.append(float(jax.device_get(intrinsic_reward)))

        return np.asarray(rewards, dtype=np.float32)

    def _diagnostics(self) -> Diagnostics:
        q_values = self._compute_q_values()
        intrinsic_rewards = self._compute_intrinsic_rewards()
        state_index = self._current_state_index()
        state_visits = self.state_visits.get(state_index, 0)
        state_action_visits = [
            self.state_action_visits.get((state_index, action), 0)
            for action in range(self.runtime.n_actions)
        ]
        best_q_action = int(np.argmax(q_values))
        best_intrinsic_action = None
        total_value = np.zeros_like(q_values)
        if intrinsic_rewards is not None:
            beta = getattr(self.runtime.agent, "intrinsic_reward_scale", 0)
            total_value = q_values + beta * intrinsic_rewards
        #     and getattr(self.runtime.agent, "rnd_action_conditioning", "none") != "none"
        # ):
            best_intrinsic_action = int(np.argmax(total_value))

        return Diagnostics(
            q_values=q_values,
            intrinsic_rewards=intrinsic_rewards,
            total_value=total_value,
            state_index=state_index,
            state_visits=state_visits,
            state_action_visits=state_action_visits,
            best_q_action=best_q_action,
            best_intrinsic_action=best_intrinsic_action,
        )

    def _format_action_table(self, diagnostics: Diagnostics) -> str:
        lines = [
            "Action diagnostics",
            "idx  key     action        q_value    intrinsic    total_value   N(s,a)  tag",
            "---  ------  ------------  ---------  ----------   -----------   ------  --------",
        ]

        key_labels = ("left", "right", "up", "p", "d", "t", "enter")
        for action, action_name in enumerate(ACTION_NAMES):
            tags: list[str] = []
            if action == diagnostics.best_q_action:
                tags.append("best_q")
            if (
                diagnostics.best_intrinsic_action is not None
                and action == diagnostics.best_intrinsic_action
            ):
                tags.append("best_rnd")
            if self.stats.last_action == action:
                tags.append("last")

            intrinsic = (
                f"{diagnostics.intrinsic_rewards[action]:10.4f}"
                if diagnostics.intrinsic_rewards is not None
                else f"{'n/a':>10}"
            )
            lines.append(
                f"{action:>3}  {key_labels[action]:<6}  {action_name:<12}  "
                f"{diagnostics.q_values[action]:>9.4f}  {intrinsic}  {diagnostics.total_value[action]:>9.4f}"
                f"{diagnostics.state_action_visits[action]:>6}  {','.join(tags) or '-'}"
            )

        return "\n".join(lines)

    def _format_info_panel(self, diagnostics: Diagnostics) -> str:
        last_action_name = (
            ACTION_NAMES[self.stats.last_action]
            if self.stats.last_action is not None
            else "-"
        )
        lines = [
            "Environment",
            f"  env_id           : {self.runtime.env_id}",
            f"  config           : {_display_path(self.runtime.config_path)}",
            f"  checkpoint       : {self.runtime.checkpoint_mode}",
            "  rnd_mode         : "
            f"{getattr(self.runtime.agent, 'rnd_action_conditioning', 'n/a')}",
            "",
            "Episode",
            f"  step             : {self.stats.step}",
            f"  return           : {self.stats.episode_return:.4f}",
            f"  done             : {self.done}",
            f"  last_action      : {last_action_name}",
            f"  last_reward      : {self.stats.last_reward:.4f}",
            f"  state_index      : {diagnostics.state_index}",
            f"  N(s)             : {diagnostics.state_visits}",
            f"  unique_states    : {len(self.state_visits)}",
            f"  total_actions    : {sum(self.state_action_visits.values())}",
            "",
            "Status",
            f"  {self.stats.status}",
            "",
            *CONTROL_HELP,
        ]
        return "\n".join(lines)

    def _refresh(self) -> None:
        diagnostics = self._diagnostics()
        self.image_artist.set_data(self._render_image())
        self.render_ax.set_title(self.runtime.env_id)
        self.table_text.set_text(self._format_action_table(diagnostics))
        self.info_text.set_text(self._format_info_panel(diagnostics))
        self.fig.canvas.draw_idle()

    def _reset_episode(self, *, status: str) -> None:
        self.runtime.key, reset_key = jax.random.split(self.runtime.key)
        self.runtime.env_state, self.runtime.obs = self.runtime.environment.reset(
            reset_key
        )
        self.done = False
        self.stats = EpisodeStats(status=status)
        self._register_state_visit()
        self._refresh()

    def _step(self, action: int) -> None:
        if self.done:
            self.stats.status = "Episode is done. Press r to reset."
            self._refresh()
            return

        state_index = self._current_state_index()
        self.state_action_visits[(state_index, action)] = (
            self.state_action_visits.get((state_index, action), 0) + 1
        )

        (
            self.runtime.env_state,
            self.runtime.obs,
            reward,
            terminal,
            truncation,
            _,
        ) = self.runtime.environment.step(self.runtime.env_state, jnp.asarray(action))

        reward_value = float(jax.device_get(reward))
        terminal_value = bool(jax.device_get(jnp.logical_or(terminal, truncation)))

        self.stats.step += 1
        self.stats.episode_return += reward_value
        self.stats.last_action = action
        self.stats.last_reward = reward_value
        self.stats.last_terminal = terminal_value
        self.stats.status = (
            f"Action {action} ({ACTION_NAMES[action]}) -> reward {reward_value:.4f}"
        )
        self.done = terminal_value

        self._register_state_visit()
        if self.done:
            self.stats.status += " | episode done, press r to reset"
        self._refresh()

    def _on_key_press(self, event: KeyEvent) -> None:
        key = (event.key or "").lower()
        if key in ACTION_KEYS:
            self._step(ACTION_KEYS[key])
            return
        if key == "r":
            self._reset_episode(status="Episode reset")
            return
        if key in {"q", "escape"}:
            plt.close(self.fig)

    def run(self) -> None:
        plt.show()


def main(args: Args) -> None:
    runtime = build_runtime(args)
    viewer = InteractiveEnvViewer(runtime)
    viewer.run()


if __name__ == "__main__":
    main(tyro.cli(Args))
