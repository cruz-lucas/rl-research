"""Local dispatcher that executes jobs on the current machine."""

from __future__ import annotations

import os
import subprocess
from typing import Sequence

from rl_research.dispatch.base import Dispatcher, ExperimentJob


class LocalDispatcher(Dispatcher):
    def __init__(self, *, dry_run: bool = False, capture_output: bool = False) -> None:
        self.dry_run = dry_run
        self.capture_output = capture_output

    def dispatch(self, job: ExperimentJob) -> None:
        command = job.hydra_command()
        if self.dry_run:
            print(" ".join(command))
            return

        env = os.environ.copy()
        env.update(job.env)

        subprocess.run(
            command,
            cwd=job.working_dir,
            env=env,
            check=True,
            capture_output=self.capture_output,
        )

