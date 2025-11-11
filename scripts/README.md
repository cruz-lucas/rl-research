# Scripts

Utilities that support day-to-day experimentation. They live outside of the Python package so they
can be reused from SLURM clusters or local shells without installing `rl_research` as a module.

## `job.sh`

SLURM launcher used for large sweeps. Highlights:

- Requests 10 CPUs, 16 GB of RAM, and a 12 hour window by default. Tweak the headers to match the
  cluster you are targeting.
- Emits a short diagnostic preamble (job id, node, start time, resources) so logs are easier to
  parse.
- Loads the `python/3.11`, `gcc`, and `arrow` modules because those are required by `jax` + MLflow
  on Compute Canada. Change the module stanza if your scheduler uses different names.
- Executes `uv run --offline --active rl-run run=...` so the CLI entrypoint is used consistently in
  both local and remote contexts. Replace the `run=` override or append `sweep=` as needed.

Before submitting a job, make sure the `logs/` directory exists (`mkdir -p logs`) so that `sbatch`
can write stdout/stderr.
