# Run Catalog

Hydra makes it trivial to combine any environment/agent/experiment triple, but well-curated presets
are what reviewers and collaborators will actually run. This catalog lists every `run=<name>`
composition under `rl_research/conf/run` so you can quickly reference the exact setting that backs a
figure or MLflow dashboard. The `agent` column points to the relative YAML path inside
`conf/agent/`, and `experiment` references `conf/experiment/<name>.yaml`.

## Double GoRight (partially observable)

| run key | agent | experiment | sweep |
| --- | --- | --- | --- |
| `doublegoright_mcts` | `mcts/doublegoright` | `default` | none |
| `doublegoright_mcts_rmax` | `rmax_mcts/doublegoright` | `default` | none |
| `doublegoright_rmax` | `rmax/doublegoright` | `default` | none |
| `doublegoright_rmax_dtp` | `dt_rmax_nstep/doublegoright` | `default` | none |
| `doublegoright_rmax_ucb` | `dt_rmax_ucb/doublegoright` | `default` | none |
| `doublegoright_ucb_dtp` | `dt_ucb/doublegoright` | `default` | none |
| `doublegoright_qlearning` | `q_learning/doublegoright_randomwalk` | `goright` | none |

## Double GoRight (fully observed)

| run key | agent | experiment | sweep |
| --- | --- | --- | --- |
| `doublegoright_fullyobs_rmax` | `rmax/doublegoright_fullyobs` | `goright` | none |
| `doublegoright_fullyobs_qlearning` | `q_learning/doublegoright_fullyobs_randomwalk` | `goright` | none |
| `doublegoright_fullyobs_mcts_rmax_handcoded` | `rmax_mcts/handcoded_doublegoright` | `goright` | none |
| `doublegoright_fullyobs_mcts_rmax_empirical` | `rmax_mcts/empirical_doublegoright` | `goright` | none |
| `doublegoright_fullyobs_mcts_rmax_handcoded_sweep` | `rmax_mcts/base` | `goright` | `rmax_mcts` |
| `doublegoright_fullyobs_mcts_rmax_empirical_sweep` | `rmax_mcts/empirical` | `goright` | `rmax_mcts` |

## RiverSwim

| run key | agent | experiment | sweep |
| --- | --- | --- | --- |
| `riverswim_rmax` | `rmax/riverswim` | `riverswim` | none |
| `riverswim_rmax_dtp` | `dt_rmax_nstep/riverswim` | `riverswim` | none |
| `riverswim_rmax_mcts` | `rmax_mcts/empirical_riverswim` | `riverswim` | none |
| `riverswim_mbie` | `mbie/riverswim` | `riverswim` | none |
| `riverswim_mbieeb` | `mbie/riverswim_eb` | `riverswim` | none |
| `riverswim_mcts_rmax_empirical_sweep` | `rmax_mcts/empirical` | `riverswim` | `rmax_mcts` |

## SixArms

| run key | agent | experiment | sweep |
| --- | --- | --- | --- |
| `sixarms_rmax` | `rmax/sixarms` | `sixarms` | none |
| `sixarms_mbie` | `mbie/sixarms` | `sixarms` | none |
| `sixarms_mbieeb` | `mbie/sixarms_eb` | `sixarms` | none |

## Utilities

- `conf/run/base.yaml` is intentionally empty; it exists so that Hydra allows `run=base` which simply
  falls back to whatever defaults are listed in `conf/config.yaml`.
- All sweeps referenced above (`none`, `rmax_mcts`) are defined in `conf/sweep/`. Use
  `rl-run run=<preset> sweep=<sweep_name>` to combine a preset with a different sweep at runtime.

Use this catalog to cross-reference MLflow experiment names or to cite the exact config you used
when describing results in your PhD materials.
