#!/usr/bin/env python3
"""
Execute a packed-job manifest sequentially and append JSONL progress records.

The progress file is append-only. On reruns, any task with a recorded `success`
event is skipped, which makes interrupted jobs easy to resubmit.
"""

from __future__ import annotations

import json
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

import tyro


CURRENT_RUN_ID: str | None = None
CURRENT_PROGRESS_FILE: Path | None = None


@dataclass
class Args:
    manifest: Annotated[
        tyro.conf.Positional[Path],
        tyro.conf.arg(help="Packed manifest JSON file to execute."),
    ]
    progress_file: Annotated[
        Path | None,
        tyro.conf.arg(
            help=(
                "Optional progress JSONL file. Defaults to "
                "<manifest parent>/../progress/<manifest stem>.progress.jsonl."
            )
        ),
    ] = None


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def default_progress_file(manifest: Path) -> Path:
    return manifest.parent.parent / "progress" / f"{manifest.stem}.progress.jsonl"


def append_event(progress_file: Path, event: str, **payload: Any) -> None:
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": now_utc(),
        "event": event,
        **payload,
    }
    with progress_file.open("a") as f:
        f.write(json.dumps(record) + "\n")


def load_successful_run_ids(progress_file: Path) -> set[str]:
    successful: set[str] = set()
    if not progress_file.exists():
        return successful

    with progress_file.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") == "success" and "run_id" in record:
                successful.add(record["run_id"])
    return successful


def build_run_command(task: dict[str, Any]) -> list[str]:
    cmd = [
        "uv",
        "run",
        "--active",
        "--offline",
        "python",
        "-m",
        "rl_research.main",
        "--config",
        str(task["config_path"]),
        "--seed",
        str(task["seed"]),
    ]

    bindings = task.get("bindings", None)
    if bindings is not None:
        cmd.append("--binding")
        cmd.extend(str(b) for b in bindings)
    return cmd


def handle_signal(signum: int, _frame: object) -> None:
    if CURRENT_PROGRESS_FILE is not None:
        append_event(
            CURRENT_PROGRESS_FILE,
            "signal",
            signal=signal.Signals(signum).name,
            run_id=CURRENT_RUN_ID,
        )
    raise SystemExit(128 + signum)


def main(args: Args) -> None:
    manifest_path = args.manifest.resolve()
    manifest = load_json(manifest_path)
    progress_file = (
        args.progress_file.resolve()
        if args.progress_file is not None
        else default_progress_file(manifest_path).resolve()
    )

    global CURRENT_PROGRESS_FILE
    CURRENT_PROGRESS_FILE = progress_file

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    tasks = manifest.get("tasks", [])
    successful = load_successful_run_ids(progress_file)
    append_event(
        progress_file,
        "job_start",
        batch_name=manifest.get("batch_name"),
        job_index=manifest.get("job_index"),
        manifest=str(manifest_path),
        total_tasks=len(tasks),
        already_successful=len(successful),
    )

    skipped = 0
    executed = 0
    succeeded = 0
    failed = 0

    for task in tasks:
        run_id = task["run_id"]
        if run_id in successful:
            skipped += 1
            print(f"[skip] {run_id} already marked successful", flush=True)
            append_event(progress_file, "skip_completed", run_id=run_id)
            continue

        global CURRENT_RUN_ID
        CURRENT_RUN_ID = run_id
        cmd = build_run_command(task)
        start_time = time.time()
        append_event(
            progress_file,
            "start",
            run_id=run_id,
            seed=task["seed"],
            combo_idx=task.get("combo_idx"),
            group_name=task.get("group_name"),
            config_path=task["config_path"],
            command=cmd,
        )
        print(f"[start] {run_id}", flush=True)
        print(f"  {shlex.join(cmd)}", flush=True)

        result = subprocess.run(cmd, check=False)
        duration_seconds = round(time.time() - start_time, 3)
        executed += 1

        if result.returncode == 0:
            succeeded += 1
            successful.add(run_id)
            append_event(
                progress_file,
                "success",
                run_id=run_id,
                returncode=result.returncode,
                duration_seconds=duration_seconds,
            )
            print(
                f"[success] {run_id} in {duration_seconds:.1f}s",
                flush=True,
            )
        else:
            failed += 1
            append_event(
                progress_file,
                "failure",
                run_id=run_id,
                returncode=result.returncode,
                duration_seconds=duration_seconds,
            )
            print(
                f"[failure] {run_id} exited with {result.returncode} "
                f"after {duration_seconds:.1f}s",
                flush=True,
            )

        CURRENT_RUN_ID = None

    append_event(
        progress_file,
        "job_end",
        batch_name=manifest.get("batch_name"),
        job_index=manifest.get("job_index"),
        total_tasks=len(tasks),
        skipped=skipped,
        executed=executed,
        succeeded=succeeded,
        failed=failed,
        successful_runs=len(successful),
    )

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main(tyro.cli(Args))
