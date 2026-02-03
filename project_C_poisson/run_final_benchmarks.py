#!/usr/bin/env python3
import sys, os
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, accuracy_score

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import ExperimentConfig, HHConfig, TaskConfig
from rng_manager import RNGManager
from hh_model import HHModel
from utils import filter_and_downsample

def narma10_gen(n_samples, seed=123):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 0.5, n_samples)
    y = np.zeros(n_samples)
    for t in range(10, n_samples):
        y[t] = 0.3 * y[t-1] + 0.05 * y[t-1] * np.sum(y[t-10:t]) + 1.5 * u[t-10] * u[t-1] + 0.1
    return u, y

def xor_gen(n_samples, delay=1, seed=124):
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, n_samples)
    targets = np.zeros(n_samples)
    for t in range(delay, n_samples):
        targets[t] = bits[t] ^ bits[t-delay]
    return bits, targets

def mc_gen(n_samples, seed=125):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 1, n_samples)
    return u

def run_task(task_name, model_params, task_cfg, common_cfg, seed_net=42):
    # Setup
    rng_mgr = RNGManager(2025)
    tg = rng_mgr.get_trial_generators(seed_net)
    st = rng_mgr.get_trial_seeds_tuple(seed_net)
    
    n_total = common_cfg['n_train'] + common_cfg['n_test'] + common_cfg['warmup_symbols']
    
    # Generate Data
    if task_name == 'narma10':
        u, y = narma10_gen(n_total)
        rates = task_cfg['rate_min'] + u * (task_cfg['rate_max'] - task_cfg['rate_min'])
        
    elif task_name == 'xor':
        delay = task_cfg['delays'][0] # Use the configured delay (e.g. 2)
        u, y = xor_gen(n_total, delay=delay) 
        rates = np.where(u == 0, task_cfg['rate_0'], task_cfg['rate_1'])
        
    elif task_name == 'mc':
        u = mc_gen(n_total)
        y = u # Target is same as input for MC
        rates = task_cfg['rate_min'] + u * (task_cfg['rate_max'] - task_cfg['rate_min'])
    
    symbol_ms = task_cfg['symbol_ms']

    # Simulate
    cfg = ExperimentConfig()
    cfg.hh = HHConfig(
        N=100, 
        gL=0.3, 
        gA=model_params['gA'],
        in_gain=model_params.get('in_gain', 10.0),      # Default to old if missing
        in_density=model_params.get('in_density', 1.0)  # Default to 1.0 if missing
    )
    cfg.task = TaskConfig(dt=common_cfg['dt'], symbol_ms=symbol_ms)
    sps = int(cfg.task.symbol_ms / cfg.task.dt)
    
    rates_up = np.repeat(rates, sps)
    # 1D Input Spike Train Generation
    # We generate a single input stream, which HHModel will distribute sparsely via W_in
    spikes_in = (tg['in'].random(len(rates_up)) < (rates_up * cfg.task.dt * 1e-3)).astype(float)
    
    # Ensure spikes_in is 2D (time, 1 channel) to match HHModel expectation for W_in broadcast
    spikes_in = spikes_in[:, np.newaxis]
    
    hh = HHModel(cfg, tg, st)
    res = hh.simulate(model_params['rho'], model_params['bias'], spikes_in, task_name, trim_steps=0)
    
    # Extract States (phi) - mean trace per symbol
    # We can try the "2 bins per symbol" trick from validation if needed, but let's stick to standard mean for benchmarks first
    # Or maybe strictly follow the validation script that showed promise (2 bins)?
    # Let's use standard mean first for consistency with literature, or maybe 1 bin is safer to avoid OOM for long runs.
    # Validation used 2 bins. Benchmarks use mean (1 bin). Let's stick to mean first.
    phi = filter_and_downsample(res['spikes'], sps, cfg.task.dt, cfg.task.tau_trace)
    
    # Check for saturation
    if np.mean(res['mean_rate']) < 1.0 or np.mean(res['mean_rate']) > 150.0:
        print(f" Warning: Abnormal FR {np.mean(res['mean_rate']):.1f} Hz!", end="")
    
    # Split
    warm = common_cfg['warmup_symbols']
    mid = warm + common_cfg['n_train']
    
    X_train = phi[warm:mid]
    y_train = y[warm:mid]
    X_test = phi[mid:]
    y_test = y[mid:]
    
    # Train Readout
    if task_name == 'xor':
        clf = Ridge(alpha=1.0)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds > 0.5)
        return acc
    elif task_name == 'mc':
        mc_sum = 0
        for k in range(1, task_cfg['max_lag']):
            if mid-k > warm and n_total-k > mid:
                y_train_k = u[warm-k:mid-k]
                y_test_k = u[mid-k:n_total-k]
                reg = Ridge(alpha=1.0)
                reg.fit(X_train, y_train_k)
                r2 = reg.score(X_test, y_test_k)
                mc_sum += max(0, r2)
        return mc_sum
    else: # NARMA
        reg = Ridge(alpha=1.0)
        reg.fit(X_train, y_train)
        preds = reg.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        nrmse = np.sqrt(mse) / np.std(y_test)
        return nrmse

def main():
    print("STAGE C: FINAL BENCHMARKS (Sparse Encoding)")
    with open('task_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    rows = []
    # Order: Stable, Edge, Chaotic
    state_order = ['stable', 'edge', 'chaotic']
    
    for state_name in state_order:
        if state_name not in config['triplets']['seed_42']: continue
        params = config['triplets']['seed_42'][state_name]
        
        print(f"\nProcessing state: {state_name.upper()} (rho={params['rho']})")
        
        # MC
        print(" -> Running MC...", end=" ", flush=True)
        mc = run_task('mc', params, config['tasks']['mc'], config['tasks']['common'])
        print(f"Result: {mc:.2f}")
        
        # NARMA
        print(" -> Running NARMA-10...", end=" ", flush=True)
        narma = run_task('narma10', params, config['tasks']['narma10'], config['tasks']['common'])
        print(f"Result (NRMSE): {narma:.4f}")
        
        # XOR
        print(f" -> Running XOR (Delay={config['tasks']['xor']['delays'][0]})...", end=" ", flush=True)
        xor = run_task('xor', params, config['tasks']['xor'], config['tasks']['common'])
        print(f"Result (Acc): {xor:.2%}")
        
        rows.append({'state': state_name, 'rho': params['rho'], 'MC': mc, 'NARMA_NRMSE': narma, 'XOR_Acc': xor})

    df = pd.DataFrame(rows)
    print("\n" + "="*50)
    print("FINAL STAGE C RESULTS")
    print("="*50)
    print(df)
    df.to_csv("results_stage_C.csv", index=False)

if __name__ == "__main__":
    main()
