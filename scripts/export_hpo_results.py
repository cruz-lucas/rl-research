#!/usr/bin/env python3
"""
Parse MLflow experiment files directly without using the MLflow API.
Creates a pandas DataFrame with one row per run, including all tags, params, and summary metrics.
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any


def parse_mlflow_run(run_dir: Path) -> Dict[str, Any]:
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


def main():
    """Main function to demonstrate usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Parse MLflow files into a pandas DataFrame')
    parser.add_argument('--mlruns-path', default='/home/lcruz1/mlruns', 
                        help='Path to the mlruns directory (default: mlruns)')
    parser.add_argument('--output', '-o', help='Output CSV file path (optional)')
    parser.add_argument('--show', action='store_true', 
                        help='Print the DataFrame to console')
    
    args = parser.parse_args()
    
    # Parse MLflow data
    print(f"Parsing MLflow data from: {args.mlruns_path}")
    df = parse_mlflow_experiment(args.mlruns_path)
    
    print(f"\nFound {len(df)} runs across {df['experiment_name'].nunique()} experiments")
    print(f"Columns: {len(df.columns)}")
    
    # Show DataFrame info
    if args.show:
        print("\nDataFrame preview:")
        print(df.head())
        print("\nDataFrame info:")
        print(df.info())
    
    # Save to CSV if requested
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nSaved to: {args.output}")
    
    return df


if __name__ == "__main__":
    df = main()