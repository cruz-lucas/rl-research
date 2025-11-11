"""Hydra-powered command line interface for composing experiments."""

from __future__ import annotations

import logging
import os
import random
import re
from copy import deepcopy
from itertools import product
from typing import Any, Dict, Iterator, Mapping, MutableMapping, Sequence, Tuple

import jax
from hydra import main as hydra_main
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from rl_research.experiment import ExperimentParams, log_experiment, run_experiment

LOGGER = logging.getLogger(__name__)

def _resolve_space_cardinality(env: Any, attr: str) -> int:
    """Traverses nested gym wrappers to recover the tabular cardinality."""
    current = env
    while current is not None:
        space = getattr(current, attr, None)
        if space is not None and hasattr(space, "n"):
            return int(space.n)
        current = getattr(current, "env", None)
    raise ValueError(f"Unable to infer `{attr}.n` from environment {type(env)!r}.")


def _prepare_environment(env_cfg: DictConfig) -> Tuple[Any, Any]:
    """Instantiates the environment, its params, and optional expectation model."""
    if env_cfg is None:
        raise ValueError("Environment config must be provided.")

    builder_cfg = env_cfg.builder if "builder" in env_cfg else env_cfg
    params_cfg = env_cfg.params if "params" in env_cfg else None

    env_params = instantiate(params_cfg) if params_cfg is not None else None
    if env_params is None:
        env = instantiate(builder_cfg)
    else:
        env = instantiate(builder_cfg, env_params)

    return env, env_params


def _prepare_agent(
    agent_cfg: DictConfig,
    env: Any,
    seed: int | None,
) -> Tuple[Any, Any]:
    """Instantiates an agent and its parameter struct from config."""
    if agent_cfg is None:
        raise ValueError("Agent config must be provided.")

    agent_dict = OmegaConf.to_container(agent_cfg, resolve=False)
    if agent_dict is None:
        raise ValueError("Failed to materialise agent config.")

    agent_dict = deepcopy(agent_dict)
    builder_dict = agent_dict.pop("builder", None)
    if builder_dict is None:
        raise ValueError("Agent config must define a `builder` section with `_target_`.")

    params_dict = agent_dict.pop("params", None)
    if params_dict is None:
        raise ValueError("Agent config must define a `params` section.")

    autofill = agent_dict.pop("autofill", {}) or {}

    def set_if_missing(params: Dict[str, Any], key: str, value: Any) -> Dict[str, Any]:
        if key not in params or params[key] is None:
            params[key] = value
        return params

    dyn_cfg = params_dict.get("dynamics_model")

    if autofill.get("num_states") == "tabular_num_states":
        n_states = _resolve_space_cardinality(env, "observation_space")
        params_dict = set_if_missing(params_dict, "num_states", n_states)
        if dyn_cfg is not None:
            if "num_states" in dyn_cfg:
                dyn_cfg = set_if_missing(dyn_cfg, "num_states", n_states)

    if autofill.get("num_actions") == "tabular_num_actions":
        n_actions = _resolve_space_cardinality(env, "action_space")
        params_dict = set_if_missing(params_dict, "num_actions", n_actions)
        if dyn_cfg is not None:
            if "num_actions" in dyn_cfg:
                dyn_cfg = set_if_missing(dyn_cfg, "num_actions", n_actions)

    if isinstance(dyn_cfg, MutableMapping) and "_target_" in dyn_cfg:
        params_dict["dynamics_model"] = instantiate(dyn_cfg)

    agent_params = instantiate(params_dict)

    builder_kwargs: Dict[str, Any] = {"params": agent_params}
    if seed is not None:
        builder_kwargs["seed"] = int(seed)

    policy_cfg = agent_dict.pop("policy", None)
    if isinstance(policy_cfg, MutableMapping):
        builder_kwargs["policy"] = instantiate(policy_cfg)

    agent = instantiate(builder_dict, **builder_kwargs)
    return agent, agent_params


def _prepare_experiment_params(experiment_cfg: DictConfig) -> ExperimentParams:
    params_dict = OmegaConf.to_container(experiment_cfg, resolve=True)
    if params_dict is None:
        raise ValueError("Experiment config must define the training parameters.")
    return ExperimentParams(**params_dict)  # type: ignore[arg-type]


def _make_sweep_suffix(
    sweep_cfg: DictConfig,
    overrides: Mapping[str, Any],
    trial_index: int | None = None,
) -> str | None:
    template = sweep_cfg.get("name_template")
    context: Dict[str, Any] = dict(overrides)
    if trial_index is not None:
        context.setdefault("trial_index", trial_index)

    if template is None:
        parts: list[str] = []
        if trial_index is not None:
            parts.append(f"trial_{trial_index}")
        parts.extend(f"{key.replace('.', '_')}={value}" for key, value in overrides.items())
        return "__".join(parts) if parts else None

    pattern = re.compile(r"\{([^}]+)\}")

    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        return str(context.get(key, match.group(0)))

    return pattern.sub(repl, template)


def _iter_grid_overrides(
    sweep_cfg: DictConfig | None,
) -> Iterator[Tuple[Dict[str, Any], str | None]]:
    if sweep_cfg is None or sweep_cfg.mode in (None, "none"):
        return

    if sweep_cfg.mode != "grid":
        raise ValueError(f"Unsupported sweep mode `{sweep_cfg.mode}`. Expected `grid`.")

    parameters = OmegaConf.to_container(sweep_cfg.parameters, resolve=True) or {}
    if not parameters:
        return

    keys = list(parameters.keys())
    value_lists = [parameters[key] for key in keys]

    for values in product(*value_lists):
        overrides = dict(zip(keys, values))
        yield overrides, _make_sweep_suffix(sweep_cfg, overrides)


def _sample_random_overrides(
    parameters: Mapping[str, Sequence[Any]],
    base_seed: int,
    trial_index: int,
) -> Dict[str, Any]:
    rng = random.Random(base_seed + trial_index)
    overrides: Dict[str, Any] = {}
    for key, values in parameters.items():
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
            raise ValueError(
                f"Random sweep parameter `{key}` must be a sequence of choices, got {type(values)!r}."
            )

        options = list(values)
        if not options:
            raise ValueError(f"Random sweep parameter `{key}` cannot be empty.")

        overrides[key] = rng.choice(options)
    return overrides


def _apply_overrides(base: DictConfig, overrides: Mapping[str, Any]) -> DictConfig:
    if not overrides:
        return base

    container = OmegaConf.to_container(base, resolve=False)
    container = deepcopy(container)

    for path, value in overrides.items():
        segments = path.split(".")
        node = container
        for segment in segments[:-1]:
            if segment not in node or not isinstance(node[segment], MutableMapping):
                node[segment] = {}
            node = node[segment]
        node[segments[-1]] = value

    return OmegaConf.create(container)


def _select(cfg: DictConfig, path: str, default: Any = None) -> Any:
    value = OmegaConf.select(cfg, path, default=None)
    return default if value is None else value


def _run_single(
    cfg: DictConfig,
    sweep_suffix: str | None = None,
    sweep_overrides: Mapping[str, Any] | None = None,
) -> None:
    env, env_params = _prepare_environment(cfg.env)
    agent_seed = _select(cfg, "agent_seed")
    agent, agent_params = _prepare_agent(cfg.agent, env, agent_seed)

    experiment_params = _prepare_experiment_params(cfg.experiment)
    rng_seed = int(_select(cfg, "rng_seed", 0))
    rng = jax.random.PRNGKey(rng_seed)

    results = run_experiment(env=env, agent=agent, rng=rng, params=experiment_params)

    experiment_name = _select(cfg, "experiment_name") or _select(cfg, "env.name") or "experiment"
    mlflow_enabled = bool(_select(cfg, "mlflow.enabled", True))
    if mlflow_enabled:        
        base_agent_name = _select(cfg, "agent_name") or _select(cfg, "agent.name") or agent.__class__.__name__
        agent_name = base_agent_name if sweep_suffix is None else f"{base_agent_name}_{sweep_suffix}"

        extra_params = dict(sweep_overrides or {})
        if sweep_suffix is not None:
            extra_params["sweep_suffix"] = sweep_suffix

        log_experiment(
            experiment_name=experiment_name,
            parent_run_name=agent_name,
            agent_name=agent_name,
            agent_params=agent_params,
            experiment_params=experiment_params,
            env_params=env_params,
            experiment_results=results,
            extra_params=extra_params or None,
            log_artifacts=bool(sweep_suffix is None)
        )

    LOGGER.info(
        "Completed run agent=%s experiment=%s",
        agent.__class__.__name__,
        experiment_name,
    )


def _run_grid_sweep(cfg: DictConfig, sweep_cfg: DictConfig) -> None:
    sweep_iter = list(_iter_grid_overrides(sweep_cfg))
    if not sweep_iter:
        _run_single(cfg)
        return

    in_job_array = "SLURM_ARRAY_TASK_ID" in os.environ
    if in_job_array:
        array_task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        array_task_count = int(os.environ["SLURM_ARRAY_TASK_COUNT"])

        LOGGER.info(
            "Job array index %s/%s over %s configurations.",
            array_task_id,
            array_task_count,
            len(sweep_iter),
        )
        overrides, suffix = sweep_iter[array_task_id]
        run_cfg = _apply_overrides(cfg, overrides)
        _run_single(run_cfg, sweep_suffix=suffix, sweep_overrides=overrides)
        return

    LOGGER.info("Running all %s hyperparameter configurations.", len(sweep_iter))
    for overrides, suffix in sweep_iter:
        run_cfg = _apply_overrides(cfg, overrides)
        _run_single(run_cfg, sweep_suffix=suffix, sweep_overrides=overrides)


def _run_random_sweep(cfg: DictConfig, sweep_cfg: DictConfig) -> None:
    parameters = OmegaConf.to_container(sweep_cfg.parameters, resolve=True) or {}
    if not parameters:
        raise ValueError("Random sweep requires at least one parameter to sample.")

    choice_map: Dict[str, Sequence[Any]] = {}
    for key, values in parameters.items():
        if not isinstance(values, Sequence):
            raise ValueError(
                f"Random sweep parameter `{key}` must be a sequence of choices, got {type(values)!r}."
            )
        sequence_values = list(values)
        if not sequence_values:
            raise ValueError(f"Random sweep parameter `{key}` cannot be empty.")
        choice_map[key] = sequence_values

    base_seed = int(sweep_cfg.get("seed", 0))
    configured_samples = sweep_cfg.get("num_samples")
    total_configured = int(configured_samples) if configured_samples is not None else None

    def run_trial(trial_index: int) -> None:
        overrides = _sample_random_overrides(choice_map, base_seed, trial_index)
        suffix = _make_sweep_suffix(sweep_cfg, overrides, trial_index=trial_index)
        run_cfg = _apply_overrides(cfg, overrides)
        _run_single(run_cfg, sweep_suffix=suffix, sweep_overrides=overrides)

    in_job_array = "SLURM_ARRAY_TASK_ID" in os.environ
    if in_job_array:
        array_task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        array_task_count = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
        if total_configured is not None and array_task_count != total_configured:
            LOGGER.warning(
                "Job array size %s overrides sweep.num_samples=%s.",
                array_task_count,
                total_configured,
            )
        if total_configured is not None and array_task_id >= total_configured:
            raise ValueError(
                f"Array index {array_task_id} exceeds configured number of random samples {total_configured}."
            )

        LOGGER.info(
            "Job array index %s/%s – sampling a random configuration.",
            array_task_id,
            array_task_count,
        )
        run_trial(array_task_id)
        return

    if total_configured is None:
        raise ValueError(
            "Random sweep must set `sweep.num_samples` when not running inside a SLURM array job."
        )

    LOGGER.info("Sampling %s random hyperparameter configurations.", total_configured)
    for idx in range(total_configured):
        run_trial(idx)


def _run(cfg: DictConfig) -> None:
    sweep_cfg = cfg.sweep if "sweep" in cfg else None
    if sweep_cfg is None or sweep_cfg.mode in (None, "none"):
        _run_single(cfg)
        return

    mode = sweep_cfg.mode
    if mode == "grid":
        _run_grid_sweep(cfg, sweep_cfg)
    elif mode == "random":
        _run_random_sweep(cfg, sweep_cfg)
    else:
        raise ValueError(f"Unsupported sweep mode `{mode}`. Expected `grid`, `random`, or `none`.")


@hydra_main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Entry-point for the `rl-run` console script."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    _run(cfg)


if __name__ == "__main__":
    main()
