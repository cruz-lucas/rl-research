# Best hyperparameters

Ranking by mean online return averaged across 10 seeds.

## Double GoRight

### Rmax
- converge_threshold: 4.2e-5
- known threshold: 14

### Model-free Batch Rmax
- step_size: 0.00106934
- known_threshold: 46
- batch_size: 326
- update_frequency: 21
- replay_ratio: 5
- warmup_steps: 2150
- buffer_size: 3571

### Model-free Batch MBIE
- step_size: 0.00231278
- beta: 69.1775
- batch_size: 357
- update_frequency: 12
- replay_ratio: 2
- warmup_steps: 944
- buffer_size: 1926

## Double GoRight (fully observable variant)

### Rmax
- converge_threshold: 9.4e-5
- known threshold: 12

### Model-free Batch Rmax
- step_size: 0.358055
- known_threshold: 2
- batch_size: 1
- update_frequency: 48
- replay_ratio: 24
- warmup_steps: 1238
- buffer_size: 3203

### Model-free Batch MBIE
- step_size: 0.00302162
- beta: 22.3098
- batch_size: 388
- update_frequency: 46
- replay_ratio: 87
- warmup_steps: 2327
- buffer_size: 2347