"""Hydra-powered command line interface for composing experiments."""

from __future__ import annotations

from copy import deepcopy
from itertools import product
import re
from typing import Any, Dict, Iterable, Iterator, Mapping, MutableMapping, Tuple

import jax
from hydra import main as hydra_main
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from rl_research.experiment import ExperimentParams, log_experiment, run_experiment


def _resolve_space_cardinality(env: Any, attr: str) -> int:
    """Traverses nested gym wrappers to recover the tabular cardinality."""
    current = env
    while current is not None:
        space = getattr(current, attr, None)
        if space is not None and hasattr(space, "n"):
            return int(space.n)
        current = getattr(current, "env", None)
    raise ValueError(f"Unable to infer `{attr}.n` from environment {type(env)!r}.")


def _prepare_environment(env_cfg: DictConfig) -> Tuple[Any, Any, Any]:
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

    expectation_model = None
    if "expectation_model" in env_cfg and env_cfg.expectation_model is not None:
        expectation_model = instantiate(env_cfg.expectation_model)

    return env, env_params, expectation_model


def _prepare_agent(
    agent_cfg: DictConfig,
    env: Any,
    expectation_model: Any,
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

    def set_if_missing(key: str, value: Any) -> None:
        if key not in params_dict or params_dict[key] is None:
            params_dict[key] = value

    if autofill.get("num_states") == "tabular_num_states":
        set_if_missing("num_states", _resolve_space_cardinality(env, "observation_space"))

    if autofill.get("num_actions") == "tabular_num_actions":
        set_if_missing("num_actions", _resolve_space_cardinality(env, "action_space"))

    if autofill.get("dynamics_model") == "env_expectation":
        if expectation_model is None:
            raise ValueError(
                "Agent requested `env_expectation` dynamics_model but the environment "
                "config did not define `expectation_model`."
            )
        params_dict["dynamics_model"] = expectation_model
    else:
        dyn_cfg = params_dict.get("dynamics_model")
        if isinstance(dyn_cfg, MutableMapping) and "_target_" in dyn_cfg:
            params_dict["dynamics_model"] = instantiate(OmegaConf.create(dyn_cfg))

    params_cfg = OmegaConf.create(params_dict)
    agent_params = instantiate(params_cfg)

    builder_cfg = OmegaConf.create(builder_dict)
    builder_kwargs: Dict[str, Any] = {"params": agent_params}
    if seed is not None:
        builder_kwargs["seed"] = int(seed)

    policy_cfg = agent_dict.pop("policy", None)
    if isinstance(policy_cfg, MutableMapping):
        builder_kwargs["policy"] = instantiate(OmegaConf.create(policy_cfg))

    agent = instantiate(builder_cfg, **builder_kwargs)
    return agent, agent_params


def _prepare_experiment_params(experiment_cfg: DictConfig) -> ExperimentParams:
    params_dict = OmegaConf.to_container(experiment_cfg, resolve=True)
    if params_dict is None:
        raise ValueError("Experiment config must define the training parameters.")
    return ExperimentParams(**params_dict)  # type: ignore[arg-type]


def _iter_sweep_overrides(
    sweep_cfg: DictConfig | None,
) -> Iterator[Tuple[Dict[str, Any], str | None]]:
    if sweep_cfg is None or sweep_cfg.mode in (None, "none"):
        return

    mode = sweep_cfg.mode
    if mode != "grid":
        raise ValueError(f"Unsupported sweep mode `{mode}`. Only `grid` is implemented.")

    parameters = OmegaConf.to_container(sweep_cfg.parameters, resolve=True) or {}
    if not parameters:
        return

    keys = list(parameters.keys())
    value_lists = [parameters[key] for key in keys]

    def make_suffix(overrides: Dict[str, Any]) -> str | None:
        template = sweep_cfg.get("name_template")
        if template is None:
            parts = [f"{key.replace('.', '_')}={value}" for key, value in overrides.items()]
            return "__".join(parts)

        pattern = re.compile(r"\{([^}]+)\}")

        def repl(match: re.Match[str]) -> str:
            key = match.group(1)
            return str(overrides.get(key, match.group(0)))

        return pattern.sub(repl, template)

    for values in product(*value_lists):
        overrides = dict(zip(keys, values))
        yield overrides, make_suffix(overrides)


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
    env, env_params, expectation_model = _prepare_environment(cfg.env)
    agent_seed = _select(cfg, "agent_seed")
    agent, agent_params = _prepare_agent(cfg.agent, env, expectation_model, agent_seed)

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
        )

    # TODO: replace print with propper logging
    print(
        f"[rl-research] Completed run agent={agent.__class__.__name__} experiment={experiment_name}"
    )


def _run(cfg: DictConfig) -> None:
    sweep_cfg = cfg.sweep if "sweep" in cfg else None
    sweep_iter = list(_iter_sweep_overrides(sweep_cfg))

    if not sweep_iter:
        _run_single(cfg)
        return

    for overrides, suffix in sweep_iter:
        run_cfg = _apply_overrides(cfg, overrides)
        _run_single(run_cfg, sweep_suffix=suffix, sweep_overrides=overrides)


@hydra_main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Entry-point for the `rl-run` console script."""
    _run(cfg)


if __name__ == "__main__":
    main()
