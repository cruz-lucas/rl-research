import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="darkgrid", context="talk")


ALGORITHMS = {
    "rb_rmax_navix": {
        "label": "RB-R-Max (Ours)",
        "color": "C0",
        "linestyle": "-",
    },
    "rmax_navix": {
        "label": "R-Max",
        "color": "C1",
        "linestyle": "-",
    },
}


PLOT_STAGES = [
    ("optimal", []),
    ("rmax", ["rmax_navix"]),
    ("rb_rmax", ["rmax_navix", "rb_rmax_navix"]),
]


def load_experiment(output_dir):
    episodic_returns = np.load(output_dir / "episodic_returns.npy")
    dones = np.load(output_dir / "dones.npy")

    steps = np.arange(len(episodic_returns))
    terminal_steps = steps[dones.astype(bool)]
    terminal_returns = episodic_returns[dones.astype(bool)]

    return steps, terminal_steps, terminal_returns


def plot_env_stage(
    env_id,
    title,
    best_score,
    stage_name,
    experiments,
    reference_experiment="rmax_navix",
):
    fig, ax = plt.subplots(figsize=(12, 6))

    # --- Always load steps from a reference experiment ---
    ref_dir = Path(f"./outputs/{reference_experiment}/{env_id}")
    if not ref_dir.exists():
        raise FileNotFoundError(
            f"Reference experiment not found: {ref_dir}"
        )

    steps, _, _ = load_experiment(ref_dir)


    # --- Algorithms ---
    for exp_name in experiments:
        meta = ALGORITHMS[exp_name]
        output_dir = Path(f"./outputs/{exp_name}/{env_id}")

        if not output_dir.exists():
            print(f"[WARN] Missing results: {output_dir}")
            continue

        _, terminal_steps, terminal_returns = load_experiment(output_dir)

        ax.plot(
            terminal_steps,
            terminal_returns,
            label=meta["label"],
            color=meta["color"],
            linestyle=meta["linestyle"],
            linewidth=2.8,
        )


    # --- Optimal return ---
    ax.plot(
        steps,
        np.ones_like(steps) * best_score,
        label="Optimal Return",
        color="black",
        linestyle="--",
        linewidth=2.5,
        zorder=10,
    )

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Discounted Return")
    ax.set_title(title)

    plt.ylim(-0.05, 1)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(6, 6))
    ax.legend(frameon=True)

    plt.tight_layout()

    save_dir = Path(f"./outputs/plots/{env_id}/{stage_name}")
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        save_dir / f"{env_id}_{stage_name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    titles = {
        "FixedGridDoorKey-5x5-layout1-v0": "Navix DoorKey (5×5), Layout 1",
        "FixedGridDoorKey-5x5-layout2-v0": "Navix DoorKey (5×5), Layout 2",
        "FixedGridDoorKey-5x5-layout3-v0": "Navix DoorKey (5×5), Layout 3",
        "FixedGridDoorKey-16x16-layout1-v0": "Navix DoorKey (16×16), Layout 1",
        "FixedGridDoorKey-16x16-layout2-v0": "Navix DoorKey (16×16), Layout 2",
        "FixedGridDoorKey-16x16-layout3-v0": "Navix DoorKey (16×16), Layout 3",
    }

    best_returns = {
        "FixedGridDoorKey-5x5-layout1-v0": 0.9509900498999999,
        "FixedGridDoorKey-5x5-layout2-v0": 0.9320653479069899,
        "FixedGridDoorKey-5x5-layout3-v0": 0.9043822,
        "FixedGridDoorKey-16x16-layout1-v0": 0.851458,
        "FixedGridDoorKey-16x16-layout2-v0": 0.7471725,
        "FixedGridDoorKey-16x16-layout3-v0": 0.64910316,
    }

    for env_id, title in titles.items():
        for stage_name, experiments in PLOT_STAGES:
            plot_env_stage(
                env_id=env_id,
                title=title,
                best_score=best_returns[env_id],
                stage_name=stage_name,
                experiments=experiments,
            )
