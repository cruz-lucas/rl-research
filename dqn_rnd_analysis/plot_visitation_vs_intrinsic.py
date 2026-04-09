from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import tyro


matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class Args:
    input_paths: tuple[Path, ...]
    output_dir: Path | None = None
    output_name: str = "visitation_vs_intrinsic"
    max_visit: int | None = None
    show_seed_curves: bool = True
    log_x: bool = False
    reference_scale: float = 1.0
    title: str = "Intrinsic Reward vs Visit Count"


def resolve_seed_dirs(input_paths: tuple[Path, ...]) -> list[Path]:
    resolved: set[Path] = set()

    for raw_path in input_paths:
        path = raw_path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")

        if path.is_file():
            if path.name != "state_stats.json":
                raise ValueError(
                    f"Expected a seed folder or state_stats.json file, got {path}"
                )
            resolved.add(path.parent)
            continue

        direct_state_stats = path / "state_stats.json"
        if direct_state_stats.exists():
            resolved.add(path)
            continue

        direct_seed_dirs = sorted(
            child
            for child in path.glob("seed_*")
            if (child / "state_stats.json").exists()
        )
        if direct_seed_dirs:
            resolved.update(direct_seed_dirs)
            continue

        recursive_seed_dirs = sorted(
            candidate.parent for candidate in path.glob("**/seed_*/state_stats.json")
        )
        if recursive_seed_dirs:
            resolved.update(recursive_seed_dirs)
            continue

        raise ValueError(f"No seed folders with state_stats.json found under {path}")

    return sorted(resolved)


def load_state_histories(seed_dir: Path) -> list[np.ndarray]:
    path = seed_dir / "state_stats.json"
    payload = json.loads(path.read_text(encoding="utf-8"))

    histories: list[np.ndarray] = []
    for state_payload in payload.values():
        history = state_payload.get("r_int_history", [])
        if not history:
            continue
        values = np.asarray(
            [float(item["value"]) for item in history],
            dtype=np.float32,
        )
        if values.size > 0:
            histories.append(values)
    return histories


def compute_seed_curve(
    histories: list[np.ndarray],
    max_visit: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not histories:
        return (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )

    effective_max_visit = max(len(history) for history in histories)
    if max_visit is not None:
        effective_max_visit = min(effective_max_visit, int(max_visit))

    per_visit_values: list[list[float]] = [[] for _ in range(effective_max_visit)]
    for history in histories:
        limit = min(len(history), effective_max_visit)
        for visit_index in range(limit):
            per_visit_values[visit_index].append(float(history[visit_index]))

    visit_counts = np.arange(1, effective_max_visit + 1, dtype=np.int32)
    mean_intrinsic = np.asarray(
        [
            float(np.mean(values)) if values else np.nan
            for values in per_visit_values
        ],
        dtype=np.float32,
    )
    num_states = np.asarray(
        [len(values) for values in per_visit_values],
        dtype=np.int32,
    )
    return visit_counts, mean_intrinsic, num_states


def aggregate_across_seeds(
    seed_curves: list[dict[str, np.ndarray | Path]],
) -> dict[str, np.ndarray]:
    if not seed_curves:
        return {
            "visit_count": np.zeros((0,), dtype=np.int32),
            "mean_intrinsic": np.zeros((0,), dtype=np.float32),
            "std_intrinsic": np.zeros((0,), dtype=np.float32),
            "sem_intrinsic": np.zeros((0,), dtype=np.float32),
            "num_seeds": np.zeros((0,), dtype=np.int32),
            "mean_num_states": np.zeros((0,), dtype=np.float32),
        }

    max_visit = max(int(curve["visit_count"][-1]) for curve in seed_curves)
    visits = np.arange(1, max_visit + 1, dtype=np.int32)

    mean_intrinsic = []
    std_intrinsic = []
    sem_intrinsic = []
    num_seeds = []
    mean_num_states = []

    for visit in visits:
        per_seed_values = []
        per_seed_state_counts = []
        for curve in seed_curves:
            visit_counts = np.asarray(curve["visit_count"])
            if visit > len(visit_counts):
                continue
            value = float(np.asarray(curve["mean_intrinsic"])[visit - 1])
            if np.isnan(value):
                continue
            per_seed_values.append(value)
            per_seed_state_counts.append(
                float(np.asarray(curve["num_states"])[visit - 1])
            )

        if per_seed_values:
            values = np.asarray(per_seed_values, dtype=np.float32)
            mean_intrinsic.append(float(np.mean(values)))
            std_intrinsic.append(float(np.std(values)))
            sem_intrinsic.append(float(np.std(values) / np.sqrt(len(values))))
            num_seeds.append(len(values))
            mean_num_states.append(float(np.mean(per_seed_state_counts)))
        else:
            mean_intrinsic.append(np.nan)
            std_intrinsic.append(np.nan)
            sem_intrinsic.append(np.nan)
            num_seeds.append(0)
            mean_num_states.append(np.nan)

    return {
        "visit_count": visits,
        "mean_intrinsic": np.asarray(mean_intrinsic, dtype=np.float32),
        "std_intrinsic": np.asarray(std_intrinsic, dtype=np.float32),
        "sem_intrinsic": np.asarray(sem_intrinsic, dtype=np.float32),
        "num_seeds": np.asarray(num_seeds, dtype=np.int32),
        "mean_num_states": np.asarray(mean_num_states, dtype=np.float32),
    }


def write_summary_csv(
    path: Path,
    aggregate: dict[str, np.ndarray],
    reference_values: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "visit_count,mean_intrinsic,std_intrinsic,sem_intrinsic,"
        "num_seeds,mean_num_states,reference\n"
    )
    rows = [header]
    for idx in range(len(aggregate["visit_count"])):
        rows.append(
            (
                f"{int(aggregate['visit_count'][idx])},"
                f"{float(aggregate['mean_intrinsic'][idx])},"
                f"{float(aggregate['std_intrinsic'][idx])},"
                f"{float(aggregate['sem_intrinsic'][idx])},"
                f"{int(aggregate['num_seeds'][idx])},"
                f"{float(aggregate['mean_num_states'][idx])},"
                f"{float(reference_values[idx])}\n"
            )
        )
    path.write_text("".join(rows), encoding="utf-8")


def plot_curves(
    output_path: Path,
    title: str,
    aggregate: dict[str, np.ndarray],
    seed_curves: list[dict[str, np.ndarray | Path]],
    show_seed_curves: bool,
    log_x: bool,
    reference_scale: float,
) -> None:
    visits = aggregate["visit_count"]
    mean_intrinsic = aggregate["mean_intrinsic"]
    sem_intrinsic = aggregate["sem_intrinsic"]
    reference = reference_scale / (np.sqrt(visits.astype(np.float32)) + 1.0)

    fig, ax = plt.subplots(figsize=(10, 5))

    if show_seed_curves:
        for curve in seed_curves:
            ax.plot(
                np.asarray(curve["visit_count"]),
                np.asarray(curve["mean_intrinsic"]),
                alpha=0.2,
                linewidth=1.0,
                color="tab:blue",
            )

    ax.plot(
        visits,
        mean_intrinsic,
        color="tab:blue",
        linewidth=2.5,
        label="RND intrinsic reward",
    )
    ax.fill_between(
        visits,
        mean_intrinsic - sem_intrinsic,
        mean_intrinsic + sem_intrinsic,
        color="tab:blue",
        alpha=0.18,
        label="seed mean ± SEM",
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

    ax.set_title(title)
    ax.set_xlabel("Visit count N")
    ax.set_ylabel("Mean intrinsic reward")
    ax.grid(alpha=0.3)
    ax.legend()

    summary = (
        f"seeds={int(np.nanmax(aggregate['num_seeds'])) if len(visits) else 0}, "
        f"max_visit={int(visits[-1]) if len(visits) else 0}"
    )
    ax.text(
        0.99,
        0.02,
        summary,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        alpha=0.75,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def default_output_dir(seed_dirs: list[Path]) -> Path:
    if len(seed_dirs) == 1:
        return seed_dirs[0]
    return seed_dirs[0].parent


def main(args: Args) -> None:
    seed_dirs = resolve_seed_dirs(args.input_paths)
    curves = []
    for seed_dir in seed_dirs:
        histories = load_state_histories(seed_dir)
        visit_count, mean_intrinsic, num_states = compute_seed_curve(
            histories=histories,
            max_visit=args.max_visit,
        )
        if len(visit_count) == 0:
            continue
        curves.append(
            {
                "seed_dir": seed_dir,
                "visit_count": visit_count,
                "mean_intrinsic": mean_intrinsic,
                "num_states": num_states,
            }
        )

    if not curves:
        raise ValueError("No non-empty per-state intrinsic histories were found.")

    aggregate = aggregate_across_seeds(curves)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else default_output_dir(seed_dirs)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    reference = args.reference_scale / (
        np.sqrt(aggregate["visit_count"].astype(np.float32)) + 1.0
    )

    plot_curves(
        output_path=output_dir / f"{args.output_name}.png",
        title=args.title,
        aggregate=aggregate,
        seed_curves=curves,
        show_seed_curves=args.show_seed_curves,
        log_x=args.log_x,
        reference_scale=args.reference_scale,
    )
    write_summary_csv(
        path=output_dir / f"{args.output_name}.csv",
        aggregate=aggregate,
        reference_values=reference,
    )

    metadata = {
        "seed_dirs": [str(path) for path in seed_dirs],
        "num_seeds": len(curves),
        "max_visit": int(aggregate["visit_count"][-1]),
        "output_plot": str(output_dir / f"{args.output_name}.png"),
        "output_csv": str(output_dir / f"{args.output_name}.csv"),
    }
    (output_dir / f"{args.output_name}_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
