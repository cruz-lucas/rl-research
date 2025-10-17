"""Utilities for generating SLURM scripts targeting Compute Canada clusters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from rl_research.dispatch.base import Dispatcher, ExperimentJob


@dataclass(slots=True)
class SlurmResources:
    time: str = "02:00:00"
    account: str = "def-somepi"
    cpus_per_task: int = 4
    gpus_per_node: int = 0
    mem: str = "8G"
    partition: str | None = None
    nodes: int = 1
    constraint: str | None = None
    qos: str | None = None


@dataclass(slots=True)
class ComputeCanadaDispatcher(Dispatcher):
    """Dispatcher that emits SLURM submission scripts for Compute Canada."""

    output_dir: Path
    resources: SlurmResources = field(default_factory=SlurmResources)
    modules: Sequence[str] = field(default_factory=tuple)
    setup_commands: Sequence[str] = field(default_factory=tuple)
    conda_env: str | None = None
    submit: bool = False

    def dispatch(self, job: ExperimentJob) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        script_path = self.output_dir / f"{job.name}.sbatch"
        script_content = self._build_script(job)
        script_path.write_text(script_content)
        if self.submit:
            import subprocess

            subprocess.run(["sbatch", str(script_path)], check=True)

    def _build_script(self, job: ExperimentJob) -> str:
        header = [
            "#!/bin/bash",
            f"#SBATCH --account={self.resources.account}",
            f"#SBATCH --time={self.resources.time}",
            f"#SBATCH --cpus-per-task={self.resources.cpus_per_task}",
            f"#SBATCH --mem={self.resources.mem}",
            f"#SBATCH --nodes={self.resources.nodes}",
        ]
        if self.resources.partition:
            header.append(f"#SBATCH --partition={self.resources.partition}")
        if self.resources.gpus_per_node:
            header.append(f"#SBATCH --gres=gpu:{self.resources.gpus_per_node}")
        if self.resources.constraint:
            header.append(f"#SBATCH --constraint={self.resources.constraint}")
        if self.resources.qos:
            header.append(f"#SBATCH --qos={self.resources.qos}")
        header.append(f"#SBATCH --job-name={job.name}")
        header.append(f"#SBATCH --output={job.name}-%j.out")
        header.append("")

        body: list[str] = []
        for module in self.modules:
            body.append(f"module load {module}")

        if self.conda_env:
            body.append(f"source activate {self.conda_env}")

        body.extend(self.setup_commands)

        import shlex

        hydra_cmd = " ".join(shlex.quote(part) for part in job.hydra_command())
        body.append(f"cd {job.working_dir}")
        env_exports = [f"export {k}={v}" for k, v in job.env.items()]
        body.extend(env_exports)
        body.append(hydra_cmd)

        return "\n".join(header + body) + "\n"
