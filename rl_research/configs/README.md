# Best hyperparameters

Ranking by mean online return averaged across 10 seeds.

## Double GoRight

### Rmax
- converge_threshold: 4.2e-5
- known threshold: 14

### Optimistic Q-learning (or model-free batch Rmax)
- step_size:
- known_threshold:
- convergence_threshold:

## Double GoRight (fully observable variant)

### Rmax
- converge_threshold: 9.4e-5
- known threshold: 12

### Optimistic Q-learning (or model-free batch Rmax)
Trained on the entire buffer deduplicated.
- step_size: 0.126
- known_threshold: 23
- convergence_threshold: 2.1e-4