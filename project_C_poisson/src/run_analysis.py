
import pandas as pd
import numpy as np
import os
import argparse
from scipy.stats import wilcoxon, ttest_rel
from sklearn.utils import resample

def load_results(results_dir):
    # Find latest parquet or merge all?
    # For now, simplistic load
    files = [f for f in os.listdir(results_dir) if f.endswith(".parquet")]
    if not files:
        raise FileNotFoundError(f"No parquet files in {results_dir}")
    
    # Sort by time
    files.sort(reverse=True)
    latest = files[0]
    print(f"Loading {latest}...")
    return pd.read_parquet(os.path.join(results_dir, latest))

def compute_stats(df):
    """
    Aggregates results: Mean +/- Std, CI, Tests.
    """
    # Group by config (rho, bias, task, metric)
    grouped = df.groupby(['rho', 'bias', 'task', 'metric'])
    
    agg_df = grouped['value'].agg(['mean', 'std', 'count']).reset_index()
    
    # Calculate CI (Bootstrap) - Placeholder logic
    # Real impl would iterate groups and bootstrap 'value'
    
    return agg_df

def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

def run_analysis(results_dir, output_dir):
    df = load_results(results_dir)
    
    # Filter for valid data
    if 'baseline' not in df.columns:
        print("CRITICAL: 'baseline' column missing.")
        return

    # Pivot for paired tests?
    # We aggregate by (rho, bias, task, metric).
    # We want to compare 'value' vs 'baseline' distribution across seeds.
    
    tasks = df['task'].unique()
    metrics = df['metric'].unique()
    
    results = []
    
    # Identify best config per Task (based on mean metric)
    # This logic requires knowing which metric is 'good' (higher or lower).
    # NRMSE -> Lower is better. Accuracy/Capacity -> Higher is better.
    
    for task in tasks:
        task_df = df[df['task'] == task]
        for metric in task_df['metric'].unique():
            # Get metric direction
            higher_better = True
            if 'nrmse' in metric.lower() or 'mse' in metric.lower() or 'error' in metric.lower():
                higher_better = False
                
            # Iterate (rho, bias) groups
            for (rho, bias), group in task_df[task_df['metric']==metric].groupby(['rho', 'bias']):
                 vals = group['value'].values
                 bases = group['baseline'].values
                 
                 # Stats
                 mean_val = np.mean(vals)
                 std_val = np.std(vals)
                 mean_base = np.mean(bases)
                 
                 # Wilcoxon Signed-Rank Test (Paired)
                 if len(vals) > 1 and not np.array_equal(vals, bases):
                     try:
                        stat, p_val = wilcoxon(vals, bases)
                     except:
                        p_val = 1.0 # Identical or error
                 else:
                     p_val = 1.0
                     
                 # Cohen's d
                 d_val = cohen_d(vals, bases)
                 
                 results.append({
                     'task': task, 'metric': metric, 'rho': rho, 'bias': bias,
                     'mean': mean_val, 'std': std_val, 'mean_base': mean_base,
                     'p_wilcoxon': p_val, 'cohens_d': d_val,
                     'n_samples': len(vals)
                 })
                 
    stats_df = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    stats_df.to_csv(os.path.join(output_dir, "rigorous_stats.csv"), index=False)
    print(f"SAVED rigorous stats to {output_dir}/rigorous_stats.csv ({len(stats_df)} rows)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="report/tables")
    args = parser.parse_args()
    
    run_analysis(args.results_dir, args.output_dir)
