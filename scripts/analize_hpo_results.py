import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class HyperparameterAnalyzer:
    """Analyze hyperparameter search results from MLflow."""
    
    def __init__(self, csv_path: str):
        """Load MLflow data from CSV."""
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} runs")
        
    def filter_experiment(self, experiment_name: str) -> 'HyperparameterAnalyzer':
        """Filter to specific experiment."""
        self.exp_df = self.df[self.df['experiment_name'] == experiment_name].copy()
        print(f"Filtered to {len(self.exp_df)} runs for experiment: {experiment_name}")
        return self
    
    def analyze(self, 
                metric: str,
                higher_is_better: bool = True,
                min_seeds: int = 1) -> pd.DataFrame:
        """
        Group by agent_class and tag_group, calculate statistics.
        
        Args:
            metric: Metric column to analyze (e.g., 'metric_final_reward')
            higher_is_better: Whether higher is better
            min_seeds: Minimum number of seeds required
            
        Returns:
            DataFrame with statistics per group
        """
        self.metric = metric
        self.higher_is_better = higher_is_better
        
        # Convert metric to numeric
        self.exp_df[metric] = pd.to_numeric(self.exp_df[metric], errors='coerce')
        
        # Group by agent_class and hyperparameter group
        grouped = self.exp_df.groupby(['param_agent_class', 'tag_group'])
        
        # Calculate statistics
        stats_list = []
        for (agent_class, group), group_df in grouped:
            n_seeds = len(group_df)
            
            if n_seeds < min_seeds:
                continue
            
            metric_values = group_df[metric].dropna()
            
            if len(metric_values) == 0:
                continue
            
            stats = {
                'agent_class': agent_class,
                'group': group,
                'n_seeds': n_seeds,
                'mean': metric_values.mean(),
                'std': metric_values.std(),
                'sem': metric_values.sem(),
                'min': metric_values.min(),
                'max': metric_values.max(),
                'median': metric_values.median(),
                'q25': metric_values.quantile(0.25),
                'q75': metric_values.quantile(0.75),
            }
            
            # Extract all param values
            param_cols = [col for col in group_df.columns if col.startswith('param_')]
            for param_col in param_cols:
                val = group_df[param_col].dropna().iloc[0] if len(group_df[param_col].dropna()) > 0 else None
                stats[param_col] = val
            
            stats_list.append(stats)
        
        self.stats_df = pd.DataFrame(stats_list)
        self.stats_df = self.stats_df.sort_values('mean', ascending=not higher_is_better)
        
        return self.stats_df
    
    def get_best_per_agent(self, top_n: int = 5) -> pd.DataFrame:
        """Get top N configurations for each agent class."""
        best_configs = []
        
        for agent_class in self.stats_df['agent_class'].unique():
            agent_df = self.stats_df[self.stats_df['agent_class'] == agent_class]
            best_configs.append(agent_df.head(top_n))
        
        return pd.concat(best_configs, ignore_index=True)
    
    def check_boundaries(self, 
                        params: Optional[List[str]] = None,
                        top_n: int = 10,
                        threshold: float = 0.3) -> Dict:
        """
        Check if best hyperparameters are hitting search boundaries.
        
        Args:
            params: List of parameter names to check (auto-detect if None)
            top_n: Number of top configs to analyze per agent
            threshold: Fraction of top configs near boundary to trigger warning
            
        Returns:
            Dictionary with boundary analysis per agent
        """
        boundary_results = {}
        
        for agent_class in self.stats_df['agent_class'].unique():
            agent_df = self.stats_df[self.stats_df['agent_class'] == agent_class]
            top_configs = agent_df.head(top_n)
            
            # Auto-detect numeric parameters
            if params is None:
                numeric_params = []
                param_cols = [col for col in agent_df.columns if col.startswith('param_')]
                for col in param_cols:
                    try:
                        pd.to_numeric(agent_df[col], errors='raise')
                        numeric_params.append(col)
                    except:
                        pass
            else:
                numeric_params = [f'param_{p}' if not p.startswith('param_') else p for p in params]
            
            agent_boundaries = {}
            
            for param_col in numeric_params:
                if param_col not in agent_df.columns:
                    continue
                
                values = pd.to_numeric(agent_df[param_col], errors='coerce').dropna()
                top_values = pd.to_numeric(top_configs[param_col], errors='coerce').dropna()
                
                if len(values) == 0 or len(top_values) == 0:
                    continue
                
                min_val = values.min()
                max_val = values.max()
                if type(min_val) == np.bool:
                    min_val = int(min_val)
                if type(max_val) == np.bool:
                    max_val = int(max_val)

                range_val = max_val - min_val
                
                # Check clustering near boundaries (within 10% of range)
                near_min = sum(top_values <= min_val + 0.1 * range_val)
                near_max = sum(top_values >= max_val - 0.1 * range_val)
                
                warning = None
                if near_min >= len(top_values) * threshold:
                    warning = f"⚠️ EXPAND LOWER ({near_min}/{len(top_values)} configs)"
                elif near_max >= len(top_values) * threshold:
                    warning = f"⚠️ EXPAND UPPER ({near_max}/{len(top_values)} configs)"
                
                agent_boundaries[param_col.replace('param_', '')] = {
                    'min': min_val,
                    'max': max_val,
                    'top_mean': top_values.mean(),
                    'top_std': top_values.std(),
                    'near_min': near_min,
                    'near_max': near_max,
                    'warning': warning
                }
            
            boundary_results[agent_class] = agent_boundaries
        
        return boundary_results
    
    def plot_performance_distribution(self, agent_class: Optional[str] = None):
        """Plot performance distribution across hyperparameter groups."""
        plot_df = self.stats_df.copy()
        
        if agent_class:
            plot_df = plot_df[plot_df['agent_class'] == agent_class]
            title = f'Performance Distribution - {agent_class}'
        else:
            title = 'Performance Distribution - All Agents'
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Box plot
        if agent_class:
            plot_df_sorted = plot_df.sort_values('mean', ascending=False)
            axes[0].errorbar(range(len(plot_df_sorted)), 
                           plot_df_sorted['mean'],
                           yerr=plot_df_sorted['std'],
                           fmt='o', capsize=5, alpha=0.7)
            axes[0].set_xlabel('Group (sorted by performance)')
            axes[0].set_ylabel(self.metric)
            axes[0].set_title('Mean ± Std by Group')
            axes[0].grid(True, alpha=0.3)
        else:
            for agent in plot_df['agent_class'].unique():
                agent_data = plot_df[plot_df['agent_class'] == agent]
                axes[0].scatter(agent_data['group'], agent_data['mean'], 
                              label=agent, alpha=0.6, s=100)
            axes[0].set_xlabel('Group')
            axes[0].set_ylabel(f'{self.metric} (mean)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Distribution of means
        for agent in plot_df['agent_class'].unique():
            agent_data = plot_df[plot_df['agent_class'] == agent]['mean']
            axes[1].hist(agent_data, alpha=0.5, label=agent, bins=20)
        
        axes[1].set_xlabel(f'{self.metric} (mean)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Group Means')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
    
    def plot_param_importance(self, param_name: str, agent_class: Optional[str] = None):
        """Plot how a parameter affects performance."""
        plot_df = self.stats_df.copy()
        
        if agent_class:
            plot_df = plot_df[plot_df['agent_class'] == agent_class]
        
        param_col = f'param_{param_name}' if not param_name.startswith('param_') else param_name
        
        if param_col not in plot_df.columns:
            print(f"Parameter {param_name} not found")
            return
        
        # Try to convert to numeric
        plot_df[param_col] = pd.to_numeric(plot_df[param_col], errors='ignore')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if pd.api.types.is_numeric_dtype(plot_df[param_col]):
            # Scatter plot for numeric params
            for agent in plot_df['agent_class'].unique():
                agent_data = plot_df[plot_df['agent_class'] == agent]
                ax.scatter(agent_data[param_col], agent_data['mean'],
                          label=agent, alpha=0.6, s=100)
            ax.set_xlabel(param_name)
            ax.set_ylabel(f'{self.metric} (mean)')
        else:
            # Box plot for categorical params
            plot_df.boxplot(column='mean', by=param_col, ax=ax)
            ax.set_xlabel(param_name)
            ax.set_ylabel(f'{self.metric} (mean)')
            plt.suptitle('')  # Remove default title
        
        ax.set_title(f'Effect of {param_name} on Performance')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def summary_report(self, top_n: int = 5):
        """Print a comprehensive summary report."""
        print("="*80)
        print(f"HYPERPARAMETER SEARCH SUMMARY")
        print(f"Metric: {self.metric} ({'↑ higher is better' if self.higher_is_better else '↓ lower is better'})")
        print("="*80)
        
        for agent_class in self.stats_df['agent_class'].unique():
            agent_df = self.stats_df[self.stats_df['agent_class'] == agent_class]
            
            print(f"\n{'─'*80}")
            print(f"Agent: {agent_class}")
            print(f"Total configurations tested: {len(agent_df)}")
            print(f"{'─'*80}")
            
            print(f"\nTop {top_n} Configurations:\n")
            
            for idx, (_, row) in enumerate(agent_df.head(top_n).iterrows(), 1):
                print(f"#{idx} Group {row['group']}:")
                print(f"   Performance: {row['mean']:.4f} ± {row['std']:.4f}")
                print(f"   SEM: {row['sem']:.4f}, Seeds: {int(row['n_seeds'])}")
                print(f"   Range: [{row['min']:.4f}, {row['max']:.4f}]")
                
                # Show params (excluding agent_class)
                params = {k.replace('param_', ''): v 
                         for k, v in row.items() 
                         if k.startswith('param_') and k != 'param_agent_class'}
                
                if params:
                    print(f"   Params: {params}")
                print()
        
        # Boundary check
        print(f"\n{'='*80}")
        print("BOUNDARY ANALYSIS")
        print(f"{'='*80}")
        
        boundaries = self.check_boundaries(top_n=top_n)
        
        for agent_class, params in boundaries.items():
            print(f"\n{agent_class}:")
            for param_name, analysis in params.items():
                warning = analysis['warning'] if analysis['warning'] else '✓ OK'
                print(f"  {param_name}: {warning}")
                print(f"    Range: [{analysis['min']:.4f}, {analysis['max']:.4f}]")
                print(f"    Top {top_n} mean: {analysis['top_mean']:.4f} ± {analysis['top_std']:.4f}")


if __name__ == "__main__":
    results_name = "hpo_navix2"
    experiment = "navix_doorkey_16x16_layout3"
    analyzer = HyperparameterAnalyzer(f"./{results_name}.csv")
    analyzer.filter_experiment(experiment)
    stats = analyzer.analyze('metric_summary_train_disc_return_mean', higher_is_better=True)

    analyzer.summary_report(top_n=10)

    best = analyzer.get_best_per_agent(top_n=10)
    print(best)

    boundaries = analyzer.check_boundaries(top_n=10)

    # analyzer.plot_performance_distribution('NFQAgent')
    # analyzer.plot_param_importance('buffer_buffer_size', 'NFQAgent')
    # analyzer.plot_param_importance('minibatch_size', 'NFQAgent')
    # analyzer.plot_param_importance('update_frequency', 'NFQAgent')
    # analyzer.plot_param_importance('warmup_steps', 'NFQAgent')
    # analyzer.plot_param_importance('agent_num_iters', 'NFQAgent')
    # analyzer.plot_param_importance('agent_max_grad_norm', 'NFQAgent')
    # analyzer.plot_param_importance('agent_eps_start', 'NFQAgent')
    # analyzer.plot_param_importance('agent_eps_end', 'NFQAgent')
    # analyzer.plot_param_importance('agent_eps_decay_steps', 'NFQAgent')
    # analyzer.plot_param_importance('agent_learning_rate', 'NFQAgent')

    # stats.to_csv(f'{experiment}__{results_name}__hyperparameter_stats.csv', index=False)
    best.to_csv(f'{experiment}__{results_name}__best_configs.csv', index=False)