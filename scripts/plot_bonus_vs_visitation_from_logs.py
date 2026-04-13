from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import matplotlib
import numpy as np
import tyro


matplotlib.use("Agg")
import matplotlib.pyplot as plt


SourceMode = Literal["decision", "update"]
EntityMode = Literal["state", "state_action"]


@dataclass
class Args:
    input_paths: tuple[Path, ...]
    output_dir: Path | None = None
    output_name: str = "rnd_bonus_vs_visitation"
    source: SourceMode = "decision"
    entity_mode: EntityMode = "state"
    max_visit: int | None = 100
    top_k_entities: int | None = 20
    entity_alpha: float = 0.15
    log_x: bool = False
    reference_scale: float = 1.0
    title: str | None = None


def resolve_run_dirs(input_paths: tuple[Path, ...]) -> list[Path]:
    resolved: set[Path] = set()

    for raw_path in input_paths:
        path = raw_path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")

        if path.is_file():
            if path.name not in {"metadata.json", "decision_trace.jsonl", "update_trace.jsonl"}:
                raise ValueError(
                    "Expected a debug log directory or one of metadata.json, "
                    f"decision_trace.jsonl, update_trace.jsonl; got {path}"
                )
            if not (path.parent / "metadata.json").exists():
                raise ValueError(f"Missing metadata.json next to {path}")
            resolved.add(path.parent)
            continue

        if (path / "metadata.json").exists():
            resolved.add(path)
            continue

        direct_run_dirs = sorted(
            child for child in path.iterdir() if child.is_dir() and (child / "metadata.json").exists()
        )
        if direct_run_dirs:
            resolved.update(direct_run_dirs)
            continue

        recursive_run_dirs = sorted(candidate.parent for candidate in path.glob("**/metadata.json"))
        if recursive_run_dirs:
            resolved.update(recursive_run_dirs)
            continue

        raise ValueError(f"No debug run directories with metadata.json found under {path}")

    return sorted(resolved)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path}:{line_number}") from exc


def load_metadata(run_dir: Path) -> dict:
    return json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))


def append_value(histories: dict[str, list[float]], entity_key: str, value: float) -> None:
    histories.setdefault(entity_key, []).append(float(value))


def add_entity(
    *,
    histories: dict[str, list[float]],
    entity_state_keys: dict[str, str],
    run_prefix: str,
    state_key: str,
    action: int,
    entity_mode: EntityMode,
    value: float,
) -> None:
    scoped_state_key = f"{run_prefix}:{state_key}"
    if entity_mode == "state":
        entity_key = scoped_state_key
    else:
        entity_key = f"{scoped_state_key}:a={action}"
    entity_state_keys[entity_key] = scoped_state_key
    append_value(histories, entity_key, value)


def _selected_decision_bonus(record: dict) -> float:
    if "decision_bonus" not in record:
        raise ValueError(
            "Decision trace is missing decision_bonus. Re-run with the updated "
            "DQNRNDAgent debug logging."
        )

    bonus = np.asarray(record["decision_bonus"], dtype=np.float32)
    if bonus.ndim == 0:
        return float(bonus)

    action = int(record["action"])
    if action < 0 or action >= bonus.shape[-1]:
        raise ValueError(
            f"Action {action} is out of bounds for decision_bonus shape {bonus.shape!r}."
        )
    return float(bonus[action])


def load_histories_from_decision_trace(
    run_dir: Path,
    entity_mode: EntityMode,
) -> tuple[dict[str, list[float]], dict[str, str]]:
    trace_path = run_dir / "decision_trace.jsonl"
    if not trace_path.exists():
        raise FileNotFoundError(f"Missing decision trace: {trace_path}")

    histories: dict[str, list[float]] = {}
    entity_state_keys: dict[str, str] = {}
    run_prefix = run_dir.name

    for record in iter_jsonl(trace_path):
        state_key = str(record["observation_key"])
        action = int(record["action"])
        bonus = _selected_decision_bonus(record)
        add_entity(
            histories=histories,
            entity_state_keys=entity_state_keys,
            run_prefix=run_prefix,
            state_key=state_key,
            action=action,
            entity_mode=entity_mode,
            value=bonus,
        )

    return histories, entity_state_keys


def load_histories_from_update_trace(
    run_dir: Path,
    entity_mode: EntityMode,
) -> tuple[dict[str, list[float]], dict[str, str]]:
    trace_path = run_dir / "update_trace.jsonl"
    if not trace_path.exists():
        raise FileNotFoundError(f"Missing update trace: {trace_path}")

    histories: dict[str, list[float]] = {}
    entity_state_keys: dict[str, str] = {}
    run_prefix = run_dir.name

    for record in iter_jsonl(trace_path):
        batch = record["batch"]
        state_keys = list(batch["observation_keys"])
        actions = np.asarray(batch["action"], dtype=np.int32)
        bonuses = np.asarray(batch["intrinsic_reward_observation"], dtype=np.float32)
        if bonuses.ndim != 1:
            raise ValueError(
                "Expected intrinsic_reward_observation to be a 1D array of per-sample "
                f"bonuses, got shape {bonuses.shape!r}."
            )
        if not (len(state_keys) == len(actions) == len(bonuses)):
            raise ValueError(
                "Mismatched batch lengths in update trace: "
                f"{len(state_keys)=}, {len(actions)=}, {len(bonuses)=}."
            )

        for state_key, action, bonus in zip(state_keys, actions.tolist(), bonuses.tolist()):
            add_entity(
                histories=histories,
                entity_state_keys=entity_state_keys,
                run_prefix=run_prefix,
                state_key=str(state_key),
                action=int(action),
                entity_mode=entity_mode,
                value=float(bonus),
            )

    return histories, entity_state_keys


def load_histories(
    run_dirs: list[Path],
    source: SourceMode,
    entity_mode: EntityMode,
) -> tuple[dict[str, list[float]], dict[str, str], list[dict]]:
    histories: dict[str, list[float]] = {}
    entity_state_keys: dict[str, str] = {}
    metadata_by_run: list[dict] = []

    for run_dir in run_dirs:
        metadata = load_metadata(run_dir)
        metadata_by_run.append(metadata)
        if source == "decision":
            run_histories, run_state_keys = load_histories_from_decision_trace(
                run_dir, entity_mode
            )
        else:
            run_histories, run_state_keys = load_histories_from_update_trace(
                run_dir, entity_mode
            )

        histories.update(run_histories)
        entity_state_keys.update(run_state_keys)

    return histories, entity_state_keys, metadata_by_run


def select_top_histories(
    histories: dict[str, list[float]],
    top_k_entities: int | None,
) -> list[tuple[str, list[float]]]:
    ranked = sorted(histories.items(), key=lambda item: (-len(item[1]), item[0]))
    if top_k_entities is None:
        return ranked
    return ranked[: max(0, int(top_k_entities))]


def build_curve_matrix(
    selected_histories: list[tuple[str, list[float]]],
    max_visit: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if not selected_histories:
        return np.zeros((0,), dtype=np.int32), np.zeros((0, 0), dtype=np.float32)

    effective_max_visit = max(len(history) for _, history in selected_histories)
    if max_visit is not None:
        effective_max_visit = min(effective_max_visit, int(max_visit))

    visit_counts = np.arange(1, effective_max_visit + 1, dtype=np.int32)
    matrix = np.full(
        (len(selected_histories), effective_max_visit),
        np.nan,
        dtype=np.float32,
    )
    for row_index, (_, history) in enumerate(selected_histories):
        limit = min(len(history), effective_max_visit)
        matrix[row_index, :limit] = np.asarray(history[:limit], dtype=np.float32)
    return visit_counts, matrix


def aggregate_curves(
    visit_counts: np.ndarray,
    curve_matrix: np.ndarray,
) -> dict[str, np.ndarray]:
    if curve_matrix.size == 0:
        empty = np.zeros((0,), dtype=np.float32)
        return {
            "visit_count": visit_counts,
            "mean_bonus": empty,
            "std_bonus": empty,
            "sem_bonus": empty,
            "num_entities": np.zeros((0,), dtype=np.int32),
        }

    num_entities = np.sum(~np.isnan(curve_matrix), axis=0).astype(np.int32)
    mean_bonus = np.nanmean(curve_matrix, axis=0).astype(np.float32)
    std_bonus = np.nanstd(curve_matrix, axis=0).astype(np.float32)
    sem_bonus = (std_bonus / np.sqrt(np.maximum(num_entities, 1))).astype(np.float32)

    return {
        "visit_count": visit_counts,
        "mean_bonus": mean_bonus,
        "std_bonus": std_bonus,
        "sem_bonus": sem_bonus,
        "num_entities": num_entities,
    }


def write_summary_csv(
    output_path: Path,
    aggregate: dict[str, np.ndarray],
    reference: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = "visit_count,mean_bonus,std_bonus,sem_bonus,num_entities,reference\n"
    rows = [header]
    for idx in range(len(aggregate["visit_count"])):
        rows.append(
            (
                f"{int(aggregate['visit_count'][idx])},"
                f"{float(aggregate['mean_bonus'][idx])},"
                f"{float(aggregate['std_bonus'][idx])},"
                f"{float(aggregate['sem_bonus'][idx])},"
                f"{int(aggregate['num_entities'][idx])},"
                f"{float(reference[idx])}\n"
            )
        )
    output_path.write_text("".join(rows), encoding="utf-8")


def default_output_dir(run_dirs: list[Path]) -> Path:
    if len(run_dirs) == 1:
        return run_dirs[0]
    return run_dirs[0].parent


def make_default_title(source: SourceMode, entity_mode: EntityMode) -> str:
    entity_label = "State" if entity_mode == "state" else "State-Action"
    source_label = "Decision-Time" if source == "decision" else "Update-Time"
    return f"{source_label} RND Bonus vs Visitation ({entity_label})"


def plot_curves(
    *,
    output_path: Path,
    title: str,
    aggregate: dict[str, np.ndarray],
    curve_matrix: np.ndarray,
    entity_alpha: float,
    log_x: bool,
    reference_scale: float,
    source: SourceMode,
) -> np.ndarray:
    visits = aggregate["visit_count"]
    reference = reference_scale / (np.sqrt(visits.astype(np.float32)))

    fig, ax = plt.subplots(figsize=(10, 6))
    for row in curve_matrix:
        mask = ~np.isnan(row)
        if not np.any(mask):
            continue
        ax.plot(
            visits[mask],
            row[mask],
            color="tab:blue",
            linewidth=1.0,
            alpha=entity_alpha,
        )

    ax.plot(
        visits,
        aggregate["mean_bonus"],
        color="tab:blue",
        linewidth=2.5,
        label="Mean RND bonus",
    )
    ax.plot(
        visits,
        reference,
        color="tab:orange",
        linewidth=2.0,
        linestyle="--",
        label=r"$1 / (\sqrt{N} + 1)$",
    )

    if log_x:
        ax.set_xscale("log")

    x_label = "Visit count N" if source == "decision" else "Update count N"
    y_label = "Decision-time RND bonus" if source == "decision" else "Update-time RND bonus"
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return reference


def main(args: Args) -> None:
    run_dirs = resolve_run_dirs(args.input_paths)
    histories, entity_state_keys, metadata_by_run = load_histories(
        run_dirs=run_dirs,
        source=args.source,
        entity_mode=args.entity_mode,
    )
    if not histories:
        raise ValueError("No non-empty histories were found in the selected debug logs.")

    selected_histories = select_top_histories(histories, args.top_k_entities)
    if not selected_histories:
        raise ValueError("No entities were selected for plotting.")

    visit_counts, curve_matrix = build_curve_matrix(selected_histories, args.max_visit)
    aggregate = aggregate_curves(visit_counts, curve_matrix)

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else default_output_dir(run_dirs)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    title = args.title or make_default_title(args.source, args.entity_mode)
    output_plot = output_dir / f"{args.output_name}.pdf"
    output_csv = output_dir / f"{args.output_name}.csv"

    reference = plot_curves(
        output_path=output_plot,
        title=title,
        aggregate=aggregate,
        curve_matrix=curve_matrix,
        entity_alpha=args.entity_alpha,
        log_x=args.log_x,
        reference_scale=args.reference_scale,
        source=args.source,
    )
    write_summary_csv(output_csv, aggregate, reference)

    selected_entity_keys = [entity_key for entity_key, _ in selected_histories]
    unique_states_total = len(set(entity_state_keys.values()))
    unique_states_selected = len({entity_state_keys[key] for key in selected_entity_keys})
    unique_entities_total = len(histories)
    unique_entities_selected = len(selected_entity_keys)
    entity_label = "states" if args.entity_mode == "state" else "state-action pairs"

    print(f"Runs: {len(run_dirs)}")
    print(f"Source: {args.source}")
    print(f"Granularity: {args.entity_mode}")
    print(f"Unique states total: {unique_states_total}")
    if args.entity_mode == "state_action":
        print(f"Unique state-action pairs total: {unique_entities_total}")
    print(f"Selected {unique_entities_selected} {entity_label} for plotting")
    print(f"Unique states selected: {unique_states_selected}")
    print(f"Max plotted visit/update count: {len(visit_counts)}")
    print(f"Plot: {output_plot}")
    print(f"CSV: {output_csv}")

    metadata = {
        "args": {
            **asdict(args),
            "input_paths": [str(path) for path in args.input_paths],
            "output_dir": str(output_dir),
        },
        "run_dirs": [str(path) for path in run_dirs],
        "agent_classes": sorted(
            {
                str(metadata.get("agent_class", "unknown"))
                for metadata in metadata_by_run
            }
        ),
        "num_runs": len(run_dirs),
        "source": args.source,
        "entity_mode": args.entity_mode,
        "num_unique_states_total": unique_states_total,
        "num_unique_entities_total": unique_entities_total,
        "num_selected_entities": unique_entities_selected,
        "num_unique_states_selected": unique_states_selected,
        "selected_entity_visit_counts": {
            entity_key: len(history) for entity_key, history in selected_histories
        },
        "output_plot": str(output_plot),
        "output_csv": str(output_csv),
    }
    (output_dir / f"{args.output_name}_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
