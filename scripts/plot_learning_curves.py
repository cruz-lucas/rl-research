#!/usr/bin/env python3
"""
Plot learning curves with bootstrap confidence intervals.
Handles unaligned episode data and produces publication-ready figures.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from scipy import interpolate

# Publication-ready plot settings
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'lines.linewidth': 2.0,
    'patch.linewidth': 0.5,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.minor.width': 0.8,
    'ytick.minor.width': 0.8,
    'mathtext.default': 'regular',
})

# Default configurations (can be overridden)
DEFAULT_GROUP_COLORS = {
    "dqn": "#1f77b4",
    "drm": "#9467bd",
    "qlearning_epsgreedy": "#ff7f0e",
    "rmax": "#2ca02c",
    "replaybased_rmax": "#d62728",
    "batch_modelfree_rmax": "#d62728",
}

DEFAULT_GROUP_LABELS = {
    "dqn": "DQN",
    "drm": "DR-max",
    "qlearning_epsgreedy": "Q-learning (Epsilon-Greedy)",
    "rmax": "R-Max",
    "replaybased_rmax": "Replay-based R-Max",
    "batch_modelfree_rmax": "Replay-based R-Max",
}

DEFAULT_GROUP_ORDER = [
    "dqn",
    "drm",
    "qlearning_epsgreedy",
    "rmax",
    "replaybased_rmax",
    "batch_modelfree_rmax"
]

def parse_mlflow_run(run_dir: Path):
    """
    Parse a single MLflow run directory and extract all information.
    
    Args:
        run_dir: Path to the run directory (e.g., mlruns/0/abc123def456/)
    
    Returns:
        Dictionary containing run info, params, tags, and summary metrics
    """
    run_data = {}
    
    # Parse meta.yaml for run info
    meta_file = run_dir / "meta.yaml"
    if meta_file.exists():
        import yaml
        with open(meta_file, 'r') as f:
            meta = yaml.safe_load(f)
            run_data['run_id'] = meta.get('run_id', '')
            run_data['run_name'] = meta.get('run_name', '')
            run_data['experiment_id'] = meta.get('experiment_id', '')
            run_data['status'] = meta.get('status', '')
            run_data['start_time'] = meta.get('start_time', '')
            run_data['end_time'] = meta.get('end_time', '')
            run_data['artifact_uri'] = meta.get('artifact_uri', '')
            run_data['lifecycle_stage'] = meta.get('lifecycle_stage', '')

    if run_data.get('lifecycle_stage') == 'deleted':
        # print(f"Skipping deleted run: {run_data.get('run_id', '')}")
        return None
    
    # Parse params
    params_dir = run_dir / "params"
    if params_dir.exists():
        for param_file in params_dir.iterdir():
            if param_file.is_file():
                with open(param_file, 'r') as f:
                    run_data[f'param_{param_file.name}'] = f.read().strip()
    
    # Parse tags
    tags_dir = run_dir / "tags"
    if tags_dir.exists():
        for tag_file in tags_dir.iterdir():
            if tag_file.is_file():
                with open(tag_file, 'r') as f:
                    run_data[f'tag_{tag_file.name}'] = f.read().strip()
    
    # Parse metrics (summary values only)
    metrics_dir = run_dir / "metrics" / "summary"
    if metrics_dir.exists():
        for metric_file in metrics_dir.iterdir():
            if metric_file.is_file():
                # Read the last line of the metric file (summary value)
                with open(metric_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        # Metric files are in format: timestamp value step
                        last_line = lines[-1].strip().split()
                        if len(last_line) >= 2:
                            run_data[f'metric_{metric_file.name}'] = float(last_line[1])
    
    return run_data

def parse_mlflow_experiment(mlruns_path: str = "mlruns") -> pd.DataFrame:
    """
    Parse all MLflow experiments and runs from the mlruns directory.
    
    Args:
        mlruns_path: Path to the mlruns directory (default: "mlruns")
    
    Returns:
        pandas DataFrame with one row per run
    """
    mlruns_dir = Path(mlruns_path)
    
    if not mlruns_dir.exists():
        raise FileNotFoundError(f"MLflow directory not found: {mlruns_path}")
    
    all_runs = []
    
    # Iterate through experiment directories
    for exp_dir in mlruns_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        # Skip .trash directory
        if exp_dir.name == '.trash':
            continue
        
        # Read experiment metadata if available
        exp_meta_file = exp_dir / "meta.yaml"
        experiment_name = exp_dir.name
        if exp_meta_file.exists():
            import yaml
            with open(exp_meta_file, 'r') as f:
                exp_meta = yaml.safe_load(f)
                experiment_name = exp_meta.get('name', exp_dir.name)
        
        # Iterate through run directories
        for run_dir in exp_dir.iterdir():
            if not run_dir.is_dir():
                continue
            
            try:
                run_data = parse_mlflow_run(run_dir)
                if run_data is not None:
                    run_data['experiment_name'] = experiment_name
                    all_runs.append(run_data)
            except Exception as e:
                print(f"Error parsing run {run_dir}: {e}")
                continue
    
    if not all_runs:
        print("No runs found in the MLflow directory.")
        return pd.DataFrame()
    
    # Create DataFrame
    df = pd.DataFrame(all_runs)
    
    # Reorder columns to put basic info first
    basic_cols = ['experiment_id', 'experiment_name', 'run_id', 'run_name', 
                  'status', 'start_time', 'end_time', 'lifecycle_stage']
    basic_cols = [col for col in basic_cols if col in df.columns]
    
    # Separate params, tags, and metrics
    param_cols = sorted([col for col in df.columns if col.startswith('param_')])
    tag_cols = sorted([col for col in df.columns if col.startswith('tag_')])
    metric_cols = sorted([col for col in df.columns if col.startswith('metric_')])
    
    # Other columns
    other_cols = [col for col in df.columns if col not in basic_cols + param_cols + tag_cols + metric_cols]
    
    # Reorder
    column_order = basic_cols + param_cols + tag_cols + metric_cols + other_cols
    df = df[column_order]
    
    return df

class LearningCurvePlotter:
    """Plot learning curves with bootstrap confidence intervals."""
    
    def __init__(self,
                 mlruns_path: str = "mlruns",
                 group_colors: Optional[Dict] = None,
                 group_labels: Optional[Dict] = None,
                 group_order: Optional[List] = None):
        """
        Initialize plotter.
        
        Args:
            mlflow_data_path: Path to CSV with MLflow data
            mlruns_path: Path to mlruns directory
            group_colors: Dict mapping group names to colors
            group_labels: Dict mapping group names to display labels
            group_order: List of group names in desired order
        """
        self.df = parse_mlflow_experiment(mlruns_path)
        self.mlruns_path = Path(mlruns_path)
        
        self.group_colors = group_colors or DEFAULT_GROUP_COLORS
        self.group_labels = group_labels or DEFAULT_GROUP_LABELS
        self.group_order = group_order or DEFAULT_GROUP_ORDER
        
    def load_returns_from_run(self, run_id: str, experiment_id: str, 
                             metric_name: str = "episode_return") -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Load episode returns and steps from a single run.
        
        Args:
            run_id: MLflow run ID
            experiment_id: Experiment ID
            metric_name: Name of the metric file
            
        Returns:
            Tuple of (steps, values) or None if not found
        """
        metric_file = self.mlruns_path / experiment_id / run_id / "metrics" / metric_name
        
        if not metric_file.exists():
            return None
        
        # Read metric file: timestamp value step
        steps = []
        values = []
        with open(metric_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    # parts[0] = timestamp, parts[1] = value, parts[2] = step
                    values.append(float(parts[1]))
                    steps.append(float(parts[2]))
                elif len(parts) >= 2:
                    # Fallback if step is missing - use index
                    values.append(float(parts[1]))
                    steps.append(len(steps))
                    print(f"Warning: Step information missing in {metric_file}, using index as steps.")
        
        if not values:
            return None
            
        return np.array(steps), np.array(values)
    
    def load_group_data(self,
                       experiment_name: str,
                       group_name: str,
                       metric_name: str = "episode_return") -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Load returns and steps for all seeds in a group.
        
        Args:
            experiment_name: Experiment name
            group_name: Group/tag name
            metric_name: Metric file name
            
        Returns:
            List of (steps, values) tuples (one per seed)
        """
        # Filter to experiment and group
        group_df = self.df[
            (self.df['experiment_name'] == experiment_name) &
            (self.df['tag_group'] == group_name)
        ]
        
        if len(group_df) == 0:
            print(f"Warning: No runs found for group '{group_name}'")
            return []
        
        all_data = []
        
        for _, row in group_df.iterrows():
            run_id = row['run_id']
            exp_id = str(row['experiment_id'])
            
            data = self.load_returns_from_run(run_id, exp_id, metric_name)
            
            if data is not None:
                steps, values = data
                if len(steps) > 0 and len(values) > 0:
                    all_data.append((steps, values))
        
        return all_data
    
    def align_curves(self,
                    curves_data: List[Tuple[np.ndarray, np.ndarray]],
                    num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align curves of different lengths using interpolation based on steps.
        
        Args:
            curves_data: List of (steps, values) tuples
            num_points: Number of interpolation points
            
        Returns:
            step_points, aligned_curves (num_points x num_seeds)
        """
        if not curves_data:
            return np.array([]), np.array([])
        
        # Determine step range across all curves
        min_step = min(steps[0] for steps, _ in curves_data)
        max_step = max(steps[-1] for steps, _ in curves_data)
        
        # Create common step axis
        step_points = np.linspace(min_step, max_step, num_points)
        
        # Interpolate each curve
        aligned = []
        for steps, values in curves_data:
            if len(steps) == 0 or len(values) == 0:
                continue
            
            # Interpolate to common step axis
            f = interpolate.interp1d(steps, values, 
                                    kind='linear',
                                    bounds_error=False,
                                    fill_value=(values[0], values[-1]))
            
            aligned_curve = f(step_points)
            aligned.append(aligned_curve)
        
        aligned_array = np.array(aligned).T  # num_points x num_seeds
        
        return step_points, aligned_array
    
    def bootstrap_ci(self,
                    data: np.ndarray,
                    n_bootstrap: int = 10000,
                    ci: float = 0.95,
                    random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate bootstrap confidence intervals.
        
        Args:
            data: Array of shape (num_points, num_seeds)
            n_bootstrap: Number of bootstrap samples
            ci: Confidence interval (0.95 = 95%)
            random_seed: Random seed for reproducibility
            
        Returns:
            mean, lower_bound, upper_bound
        """
        np.random.seed(random_seed)
        
        num_points, num_seeds = data.shape
        
        # Calculate mean
        mean = np.mean(data, axis=1)
        
        # Bootstrap
        alpha = 1 - ci
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        bootstrap_means = np.zeros((n_bootstrap, num_points))
        
        for i in range(n_bootstrap):
            # Resample seeds with replacement
            idx = np.random.randint(0, num_seeds, size=num_seeds)
            bootstrap_sample = data[:, idx]
            bootstrap_means[i] = np.mean(bootstrap_sample, axis=1)
        
        # Calculate percentiles
        lower = np.percentile(bootstrap_means, lower_percentile, axis=0)
        upper = np.percentile(bootstrap_means, upper_percentile, axis=0)
        
        return mean, lower, upper
    
    def plot_learning_curves(self,
                           experiment_name: str,
                           groups: Optional[List[str]] = None,
                           metric_name: str = "episode_return",
                           num_points: int = 1000,
                           n_bootstrap: int = 10000,
                           ci: float = 0.95,
                           xlabel: str = "Steps",
                           ylabel: str = "Discounted Return",
                           title: Optional[str] = None,
                           figsize: Tuple[int, int] = (10, 6),
                           grid: bool = True,
                           legend_loc: str = "best",
                           save_path: Optional[str] = None,
                           dpi: int = 300,
                           show_seeds: bool = False,
                           seed_alpha: float = 0.1) -> plt.Figure:
        """
        Plot learning curves with bootstrap confidence intervals.
        
        Args:
            experiment_name: Experiment name to filter
            groups: List of groups to plot (None = all in group_order)
            metric_name: Metric file name
            num_points: Number of interpolation points
            n_bootstrap: Number of bootstrap samples
            ci: Confidence interval
            xlabel: X-axis label
            ylabel: Y-axis label
            title: Plot title
            figsize: Figure size
            grid: Show grid
            legend_loc: Legend location
            save_path: Path to save figure
            dpi: DPI for saved figure
            show_seeds: Show individual seed trajectories
            seed_alpha: Alpha for seed trajectories
            
        Returns:
            Figure object
        """
        if groups is None:
            groups = self.group_order
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for group in groups:
            # Load data for this group
            curves_data = self.load_group_data(experiment_name, group, metric_name)
            
            if not curves_data:
                print(f"Skipping group '{group}' - no data")
                continue
            
            print(f"Group '{group}': {len(curves_data)} seeds")
            
            # Align curves
            steps, aligned_data = self.align_curves(curves_data, num_points)
            
            if aligned_data.size == 0:
                continue
            
            # Calculate bootstrap CI
            mean, lower, upper = self.bootstrap_ci(aligned_data, n_bootstrap, ci)
            
            # Get color and label
            color = self.group_colors.get(group, None)
            label = self.group_labels.get(group, group)
            
            # Plot individual seeds if requested
            if show_seeds:
                for i in range(aligned_data.shape[1]):
                    ax.plot(steps, aligned_data[:, i], 
                           color=color, alpha=seed_alpha, linewidth=0.8, zorder=1)
            
            # Plot mean
            ax.plot(steps, mean, color=color, label=label, linewidth=2.5, zorder=3)
            
            # Plot CI
            ax.fill_between(steps, lower, upper, 
                           color=color, alpha=0.2, linewidth=0, zorder=2)
        
        # Formatting
        ax.set_xlabel(xlabel, fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        
        if title:
            ax.set_title(title, fontweight='bold')
        
        if grid:
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        
        ax.legend(loc=legend_loc, framealpha=0.95, edgecolor='black', fancybox=True)
        
        # Tight layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"Saved figure to: {save_path}")
        
        return fig
    
    def plot_multiple_metrics(self,
                            experiment_name: str,
                            metrics: List[Tuple[str, str]],  # [(metric_name, ylabel), ...]
                            groups: Optional[List[str]] = None,
                            **plot_kwargs) -> List[plt.Figure]:
        """
        Plot multiple metrics in separate figures.
        
        Args:
            experiment_name: Experiment name
            metrics: List of (metric_name, ylabel) tuples
            groups: Groups to plot
            **plot_kwargs: Additional arguments for plot_learning_curves
            
        Returns:
            List of figure objects
        """
        figures = []
        
        for metric_name, ylabel in metrics:
            print(f"\nPlotting {metric_name}...")
            
            fig = self.plot_learning_curves(
                experiment_name=experiment_name,
                groups=groups,
                metric_name=metric_name,
                ylabel=ylabel,
                **plot_kwargs
            )
            
            figures.append(fig)
        
        return figures


def main():
    parser = argparse.ArgumentParser(
        description='Plot learning curves with bootstrap confidence intervals',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic plot
  python plot_learning_curves.py "my_experiment" --mlruns-path mlruns
  
  # Customize appearance
  python plot_learning_curves.py "my_experiment" \\
      --metric train/discounted_return \\
      --groups dqn drm rmax \\
      --n-bootstrap 5000 \\
      --output learning_curves.png
  
  # Show individual seeds
  python plot_learning_curves.py "my_experiment" \\
      --show-seeds --seed-alpha 0.15
        """
    )
    
    parser.add_argument('experiment_name', help='Experiment name to plot')
    parser.add_argument('--mlruns-path', default='./mlruns',
                       help='Path to mlruns directory (default: ./mlruns)')
    parser.add_argument('--metric', default='train/discounted_return',
                       help='Metric name to plot (default: train/discounted_return)')
    parser.add_argument('--groups', nargs='+',
                       help='Groups to plot (default: all in GROUP_ORDER)')
    parser.add_argument('--num-points', type=int, default=1000,
                       help='Number of interpolation points (default: 1000)')
    parser.add_argument('--n-bootstrap', type=int, default=10000,
                       help='Number of bootstrap samples (default: 10000)')
    parser.add_argument('--ci', type=float, default=0.95,
                       help='Confidence interval (default: 0.95)')
    parser.add_argument('--xlabel', default='Steps',
                       help='X-axis label')
    parser.add_argument('--ylabel', default='Discounted Return',
                       help='Y-axis label')
    parser.add_argument('--title', help='Plot title')
    parser.add_argument('--figsize', nargs=2, type=int, default=[10, 6],
                       help='Figure size (width height)')
    parser.add_argument('--no-grid', action='store_true',
                       help='Disable grid')
    parser.add_argument('--legend-loc', default='best',
                       help='Legend location (default: best)')
    parser.add_argument('--output', '-o', default='plot.pdf',
                       help='Output file path (e.g., plot.png, plot.pdf)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for saved figure (default: 300)')
    parser.add_argument('--show-seeds', action='store_true',
                       help='Show individual seed trajectories')
    parser.add_argument('--seed-alpha', type=float, default=0.1,
                       help='Alpha for seed trajectories (default: 0.1)')
    parser.add_argument('--show', action='store_true',
                       help='Display plot interactively')
    
    args = parser.parse_args()
    
    # Create plotter
    plotter = LearningCurvePlotter(
        mlruns_path=args.mlruns_path
    )
    
    # Plot
    fig = plotter.plot_learning_curves(
        experiment_name=args.experiment_name,
        groups=args.groups,
        metric_name=args.metric,
        num_points=args.num_points,
        n_bootstrap=args.n_bootstrap,
        ci=args.ci,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        title=args.title,
        figsize=tuple(args.figsize),
        grid=not args.no_grid,
        legend_loc=args.legend_loc,
        save_path=args.output,
        dpi=args.dpi,
        show_seeds=args.show_seeds,
        seed_alpha=args.seed_alpha
    )
    
    if args.show:
        plt.show()
    elif not args.output:
        print("\nUse --output to save or --show to display the plot")


if __name__ == "__main__":
    main()