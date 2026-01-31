
import numpy as np
import pandas as pd
import os
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List

from config import load_config, ExperimentConfig
from rng_manager import RNGManager
from baselines_rc.esn import EchoStateNetwork
from readout import ReadoutModule
from utils import get_git_hash

# Tasks
from tasks.narma import NARMA
from tasks.xor import DelayedXOR
from tasks.mc import MemoryCapacity

def run_esn_trial(rho_esn: float, trial_idx: int, cfg: ExperimentConfig, base_seed: int = 2025) -> List[Dict[str, Any]]:
    results = []
    
    # 1. Setup RNG
    rng_mgr = RNGManager(base_seed)
    # Use SAME trial_idx to get SAME input streams as HH experiment
    trial_generators = rng_mgr.get_trial_generators(trial_idx)
    seeds_tuple = rng_mgr.get_trial_seeds_tuple(trial_idx)
    
    # 2. Setup ESN
    # ESN runs at symbol rate
    esn = EchoStateNetwork(
        N=cfg.esn.N,
        spectral_radius=rho_esn,
        input_scale=cfg.esn.input_scale,
        density=cfg.esn.density,
        leaking_rate=cfg.esn.leaking_rate,
        seed=seeds_tuple[0] # Use rec seed for ESN weights
    )
    
    readout = ReadoutModule(trial_generators['readout'], cv_folds=cfg.cv_folds, cv_gap=cfg.cv_gap)
    
    try:
        # ==========================
        # TASK A: NARMA
        # ==========================
        narma = NARMA(trial_generators['in'])    
        u_nm, y_nm = narma.generate_data(2000, order=cfg.task.narma_order)
        
        # Run ESN
        # ESN takes analog u directly.
        X_esn = esn.simulate(u_nm, washout=cfg.task.narma_order) # Washout? Or just run.
        # HH uses trim_steps=500 -> 500*dt / 20ms = 25ms? No.
        # dt=0.05, symbol=20. steps=400.
        # trim=500 steps is 25ms. One symbol is 20ms.
        # So trim is ~1.2 symbols? Wait.
        # config: dt=0.05. symbol_ms=20. steps_per_symbol = 400.
        # trim_steps=500 means 500 * 0.05 = 25 ms.
        # So we trim ~1 symbol.
        # ESN washout should be comparable. Let's use 100 symbols (standard).
        washout = 100
        
        # Re-generate data longer? No, generate_data(2000).
        # HH trims 500 *high_freq_steps*. That is 25ms.
        # If HH trims 1 symbol, ESN should trim 1 symbol.
        # But HH has transient.
        # ESN has transient too.
        # safely skip 200 symbols.
        
        # Actually, let's stick to what ReadoutModule expects.
        # ReadoutModule gets (X, y).
        # We need to ensure X and y are aligned.
        
        # Code above (NARMA) generates 2000.
        X_nm = esn.simulate(u_nm, washout=0) # Get all
        
        # discard washout
        X_nm = X_nm[washout:]
        y_nm = y_nm[washout:]
        
        # Standardize?
        # HH does StandardScaler. ESN usually does too or Ridge handles it.
        # Config has zscore_features.
        if cfg.task.zscore_features:
            from sklearn.preprocessing import StandardScaler
            X_nm = StandardScaler().fit_transform(X_nm)
            
        met_nm = readout.train_ridge_cv(X_nm, y_nm, task_type='regression', alphas=cfg.ridge_alphas)
        
        # Baseline (AR)
        # We need independent baseline calc or comparable.
        # narma.compute_baseline takes y_tgt.
        # Matches y_nm aligned.
        order = cfg.task.narma_order
        # Need to ensure lengths valid
        if len(y_nm) > order:
            X_ar = np.zeros((len(y_nm)-order, order))
            y_ar = y_nm[order:]
            # Construct AR
            for i in range(order):
                # predictors from y_nm full (before y_ar)
                X_ar[:, i] = y_nm[order-(i+1) : len(y_nm)-(i+1)]
            
            base_res = readout.train_ridge_cv(X_ar, y_ar, task_type='regression', alphas=cfg.ridge_alphas)
            base_nm = base_res['nrmse']
            
            # Align metric to same target y
            # met_nm was trained on y_nm (full).
            # AR models on y_nm[order:].
            # Strictly, we should eval metric on y_nm[order:] too for apples-to-apples.
            # But if T is large (2000), difference between T and T-10 is negligible.
            # However, Research Engineer demands rigor.
            # Let's truncate X_nm as well.
            met_nm = readout.train_ridge_cv(X_nm[order:], y_nm[order:], task_type='regression', alphas=cfg.ridge_alphas)
        else:
            base_nm = 1.0; met_nm = {'nrmse': 1.0}
            
        imp_nm = (base_nm - met_nm['nrmse']) / (base_nm + 1e-12)
        
        res_base = dict(rho_esn=rho_esn, seed=trial_idx)
        results.append({**res_base, 'task': 'NARMA', 'metric': 'nrmse', 'value': met_nm['nrmse'], 'baseline': base_nm, 'improvement': imp_nm})


        # ==========================
        # TASK B: XOR
        # ==========================
        xor = DelayedXOR(trial_generators['in'])
        u_xor, y_xor = xor.generate_data(2000, delay=cfg.task.xor_delay)
        
        X_xor = esn.simulate(u_xor, washout=0)
        X_xor = X_xor[washout:]
        y_xor = y_xor[washout:]
        
        if cfg.task.zscore_features:
             from sklearn.preprocessing import StandardScaler
             X_xor = StandardScaler().fit_transform(X_xor)
             
        met_xor = readout.train_ridge_cv(X_xor, y_xor, task_type='classification', alphas=cfg.ridge_alphas)
        base_xor = xor.compute_baseline(y_xor, readout)
        
        results.append({**res_base, 'task': 'XOR', 'metric': 'accuracy', 'value': met_xor.get('acc', 0.5), 'baseline': base_xor, 'improvement': met_xor.get('acc', 0.5) - base_xor})
        results.append({**res_base, 'task': 'XOR', 'metric': 'auc', 'value': met_xor.get('auc', 0.5), 'baseline': 0.5, 'improvement': met_xor.get('auc', 0.5) - 0.5})

        # ==========================
        # TASK C: MC
        # ==========================
        mc = MemoryCapacity(trial_generators['in'])
        u_mc = mc.generate_data(2000)
        
        X_mc = esn.simulate(u_mc, washout=0)
        X_mc = X_mc[washout:]
        u_mc = u_mc[washout:]
        
        if cfg.task.zscore_features:
             from sklearn.preprocessing import StandardScaler
             X_mc = StandardScaler().fit_transform(X_mc)
        
        res_mc = mc.run_mc_analysis(X_mc, u_mc, readout, max_lag=cfg.task.mc_max_lag)
        base_mc = mc.compute_baseline(u_mc, X_mc, readout, rng_shuffle=trial_generators['readout'], max_lag=cfg.task.mc_max_lag)
        
        results.append({**res_base, 'task': 'MC', 'metric': 'capacity', 'value': res_mc['mc'], 'baseline': base_mc, 'improvement': res_mc['mc'] - base_mc})
        
    except Exception as e:
        print(f"ERROR ESN Trial {trial_idx} rho={rho_esn}: {e}")
        pass
        
    return results

def run_esn_sweep(cfg_path: str):
    cfg = load_config(cfg_path)
    os.makedirs(cfg.results_dir, exist_ok=True)
    
    # ESN Sweep Parameters
    # As per plan: rho in [0.1, 1.2]
    # We define a grid here
    esn_rhos = np.round(np.arange(0.1, 1.3, 0.1), 2).tolist()
    seeds = range(cfg.seeds_fine) # Use Fine seeds for robust baseline
    
    print(f"Starting ESN Sweep: Rho={esn_rhos}, Seeds={len(seeds)}")
    
    tasks = []
    for r in esn_rhos:
        for s in seeds:
            tasks.append((r, s))
            
    all_results = []
    with ProcessPoolExecutor(max_workers=os.cpu_count() - 2) as executor:
        futures = {executor.submit(run_esn_trial, r, s, cfg): (r, s) for r, s in tasks}
        
        for i, future in enumerate(as_completed(futures)):
            res = future.result()
            all_results.extend(res)
            if i % 10 == 0:
                print(f"ESN Progress {i}/{len(tasks)}")
                
    df = pd.DataFrame(all_results)
    df['timestamp'] = time.strftime("%Y%m%d-%H%M%S")
    df['model_type'] = 'ESN'
    
    out_path = os.path.join(cfg.results_dir, f"results_esn_{int(time.time())}.parquet")
    df.to_parquet(out_path)
    print(f"Saved ESN results to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    run_esn_sweep(args.config)
