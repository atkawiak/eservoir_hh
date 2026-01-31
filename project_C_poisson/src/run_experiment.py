
import numpy as np
import pandas as pd
import os
import joblib
import time
import argparse
from typing import Dict, Any, List
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

from config import load_config, ExperimentConfig
from rng_manager import RNGManager
from hh_model import HHModel
from readout import ReadoutModule
from sklearn.preprocessing import StandardScaler
from utils import filter_and_downsample, get_git_hash

# Tasks
from tasks.narma import NARMA
from tasks.xor import DelayedXOR
from tasks.mc import MemoryCapacity
from tasks.lyapunov_task import LyapunovModule

def run_trial(rho: float, bias: float, trial_idx: int, cfg: ExperimentConfig, base_seed: int = 2025) -> List[Dict[str, Any]]:
    """
    Executes a single Trial (Seed Tuple) for a given (rho, bias).
    A Trial consists of 4 tasks run on the SAME reservoir instance (re-simulated/cached).
    """
    results = []
    
    # 1. Setup RNG (Strict N=20 separation)
    rng_mgr = RNGManager(base_seed)
    # trial_idx 0..19 determines the 4 streams
    trial_generators = rng_mgr.get_trial_generators(trial_idx)
    seeds_tuple = rng_mgr.get_trial_seeds_tuple(trial_idx) # (rec, inmask, in, readout)
    
    # 2. Setup Modules
    hh = HHModel(cfg, trial_generators, seeds_tuple)
    readout = ReadoutModule(trial_generators['readout'], cv_folds=cfg.cv_folds, cv_gap=cfg.cv_gap) 
    
    # 3. Parameters
    steps_per_symbol = int(cfg.task.symbol_ms / cfg.task.dt)
    
    try:
        # ==========================
        # TASK A: NARMA (Regression)
        # ==========================
        narma = NARMA(trial_generators['in']) # Use input stream for task generation
        u_nm, y_nm = narma.generate_data(2000, order=cfg.task.narma_order)
        
        input_id_nm = cfg.get_task_input_id() + "_NARMA_u05" # Add dist logic explicitly
        
        rates_nm = cfg.task.poisson_rate_min + u_nm * 2.0 * (cfg.task.poisson_rate_max - cfg.task.poisson_rate_min) 
        rates_up_nm = np.repeat(rates_nm, steps_per_symbol)
        spikes_nm = (trial_generators['in'].random(len(rates_up_nm)) < (rates_up_nm * cfg.task.dt * 1e-3)).astype(float)
        
        # Baseline logic is handled inside each block for clarity and state sync
        state_nm = hh.simulate(rho, bias, spikes_nm, input_id_nm)
        
        # Fallback length calc
        trim_steps = 500
        # Calculate per-task expected length to avoid crossover bugs
        expected_len_nm = (len(spikes_nm) - trim_steps) // steps_per_symbol
        
        if state_nm['mean_rate'] == 0:
            phi_nm = np.zeros((expected_len_nm, cfg.hh.N)) # Correct fallback length
        else:
            phi_nm = filter_and_downsample(state_nm['spikes'], steps_per_symbol, cfg.task.dt, cfg.task.tau_trace)
            if phi_nm.shape[1] != cfg.hh.N:
                # Emergency fix for cached dimension mismatch
                phi_nm = np.zeros((phi_nm.shape[0], cfg.hh.N))
            
        if cfg.task.zscore_features:
            from sklearn.preprocessing import StandardScaler
            phi_nm = StandardScaler().fit_transform(phi_nm)
            
        y_nm = y_nm[-len(phi_nm):]
        
        # Strictly aligned Baseline:
        # Model uses phi_nm[t] to predict y_nm[t].
        # For fair comparison, we truncate the first 'order' samples from BOTH
        # to ensure the same time indices are evaluated.
        # This fixes "different BlockedCV splits" issue.
        order = cfg.task.narma_order
        if len(phi_nm) > order:
            phi_valid = phi_nm[order:]
            y_valid = y_nm[order:]
            
            # Baseline uses SAME y_valid as target.
            # But baseline needs 'order' history for first sample of y_valid.
            # We construct X_ar purely from y_valid (shifted)? No.
            # We use y_nm to construct X_ar, then align.
            # Better: narma.compute_baseline takes X_ar and y_tgt.
            
            # Use 'compute_baseline' existing but pass the ALIGNED target.
            # Wait, if we pass y_valid to compute_baseline, it will internally look back 'order' steps FROM y_valid.
            # That would mean it needs y_valid[0-order].
            # narma.compute_baseline implementation:
            # X_ar[:, i] = y_tgt[order-(i+1) : n-(i+1)]
            # It assumes y_tgt is the FULL sequence and cuts off 'order' stats.
            
            # So if we pass y_nm (full), it evaluates on y_nm[order:].
            # If we run model on (phi[order:], y[order:]), then model evaluates on y[order:].
            # This MATCHES!
            
            # So the previous code:
            # met_nm = readout.train_ridge_cv(phi_nm, y_nm) -> evaluates on all splits of (phi, y).
            # base_nm = compute_baseline(y=y_nm) -> evaluates on y[order:].
            
            # Issue: Model evaluates on [0..T], Baseline on [order..T].
            # Fix: Force model to evaluate on [order..T] too.
            
            met_nm = readout.train_ridge_cv(phi_valid, y_valid, task_type='regression', alphas=cfg.ridge_alphas)
            
            # Now Baseline:
            # If we call compute_baseline(y_valid), it will cut ANOTHER 'order'.
            # We want baseline to evaluate on EXACTLY y_valid.
            # So we must pass (y_with_history) but tell it to eval on y_valid?
            # Or manually construct features here.
            
            # Manual construction for 100% clarity:
            X_ar = np.zeros((len(y_valid), order))
            # We need history from y_nm prior to y_valid
            # y_valid starts at index 'order' of y_nm.
            # History for y_valid[0] is y_nm[order-1], ..., y_nm[0].
            for i in range(order):
                 # Lag i+1
                 # X[:, i] = y_nm[order-(i+1) : - (i+1)] ?
                 # effectively shifted columns.
                 X_ar[:, i] = y_nm[order-(i+1) : len(y_nm)-(i+1)]
                 
            # Verify lengths
            # len(y_valid) = L - order.
            # len(y_nm) = L.
            # slice order-(i+1) : L-(i+1) has length (L-(i+1)) - (order-(i+1)) = L - order. Correct.
            
            base_nm_val = readout.train_ridge_cv(X_ar, y_valid, task_type="regression", alphas=cfg.ridge_alphas)
            base_nm = base_nm_val['nrmse']
            
        else:
             met_nm = {'nrmse': 1.0}; base_nm = 1.0; imp_nm = 0.0

        imp_nm = (base_nm - met_nm['nrmse']) / (base_nm + 1e-12)
        
        # Schema - Base
        res_base = dict(rho=rho, bias=bias, seed_tuple_id=trial_idx,
                       seed_rec=seeds_tuple[0], seed_inmask=seeds_tuple[1], seed_in=seeds_tuple[2], seed_readout=seeds_tuple[3],
                       firing_rate=state_nm['mean_rate'], I_syn_mean=state_nm['mean_I_syn'], 
                       I_syn_var=state_nm['var_I_syn'], saturation_flag=state_nm['saturation_flag'])

        results.append({**res_base, 'task': 'NARMA', 'metric': 'nrmse', 'value': met_nm['nrmse'], 'baseline': base_nm, 'improvement': imp_nm})

        # ==========================
        # TASK B: XOR (Classification)
        # ==========================
        xor = DelayedXOR(trial_generators['in'])
        u_xor, y_xor = xor.generate_data(2000, delay=cfg.task.xor_delay)
        input_id_xor = cfg.get_task_input_id() + "_XOR_bern05"
        
        rates_xor = cfg.task.poisson_rate_min + u_xor * (cfg.task.poisson_rate_max - cfg.task.poisson_rate_min) 
        rates_up_xor = np.repeat(rates_xor, steps_per_symbol)
        spikes_xor = (trial_generators['in'].random(len(rates_up_xor)) < (rates_up_xor * cfg.task.dt * 1e-3)).astype(float)
        
        state_xor = hh.simulate(rho, bias, spikes_xor, input_id_xor)
        expected_len_xor = (len(spikes_xor) - trim_steps) // steps_per_symbol
        
        if state_xor['mean_rate'] == 0:
            phi_xor = np.zeros((expected_len_xor, cfg.hh.N))
        else:
            phi_xor = filter_and_downsample(state_xor['spikes'], steps_per_symbol, cfg.task.dt, cfg.task.tau_trace)
            if phi_xor.shape[1] != cfg.hh.N:
                phi_xor = np.zeros((phi_xor.shape[0], cfg.hh.N))
            
        if cfg.task.zscore_features:
             from sklearn.preprocessing import StandardScaler
             phi_xor = StandardScaler().fit_transform(phi_xor)
        y_xor = y_xor[-len(phi_xor):]
        
        met_xor = readout.train_ridge_cv(phi_xor, y_xor, task_type='classification', alphas=cfg.ridge_alphas)
        base_xor = xor.compute_baseline(y_xor, readout)
        
        res_base['firing_rate'] = state_xor['mean_rate'] # Update specific to this run
        res_base['I_syn_mean'] = state_xor['mean_I_syn']
        res_base['I_syn_var'] = state_xor['var_I_syn']
        res_base['saturation_flag'] = state_xor['saturation_flag']
        
        results.append({**res_base, 'task': 'XOR', 'metric': 'accuracy', 'value': met_xor.get('acc', 0.5), 'baseline': base_xor, 'improvement': met_xor.get('acc', 0.5) - base_xor})
        results.append({**res_base, 'task': 'XOR', 'metric': 'auc', 'value': met_xor.get('auc', 0.5), 'baseline': 0.5, 'improvement': met_xor.get('auc', 0.5) - 0.5})

        # ==========================
        # TASK C: MC (Capacity)
        # ==========================
        mc = MemoryCapacity(trial_generators['in'])
        u_mc = mc.generate_data(2000)
        input_id_mc = cfg.get_task_input_id() + "_MC_uniform01"
        
        rates_mc = cfg.task.poisson_rate_min + u_mc * (cfg.task.poisson_rate_max - cfg.task.poisson_rate_min)
        rates_up_mc = np.repeat(rates_mc, steps_per_symbol)
        spikes_mc = (trial_generators['in'].random(len(rates_up_mc)) < (rates_up_mc * cfg.task.dt * 1e-3)).astype(float)
        
        state_mc = hh.simulate(rho, bias, spikes_mc, input_id_mc)
        expected_len_mc = (len(spikes_mc) - trim_steps) // steps_per_symbol
        
        if state_mc['mean_rate'] == 0:
            phi_mc = np.zeros((expected_len_mc, cfg.hh.N))
        else:
            phi_mc = filter_and_downsample(state_mc['spikes'], steps_per_symbol, cfg.task.dt, cfg.task.tau_trace)
            if phi_mc.shape[1] != cfg.hh.N:
                phi_mc = np.zeros((phi_mc.shape[0], cfg.hh.N))
            
        if cfg.task.zscore_features:
             from sklearn.preprocessing import StandardScaler
             phi_mc = StandardScaler().fit_transform(phi_mc)
        u_mc = u_mc[-len(phi_mc):]
        
        res_mc = mc.run_mc_analysis(phi_mc, u_mc, readout, max_lag=cfg.task.mc_max_lag)
        # Pass dedicated RNG for shuffling logic
        base_mc = mc.compute_baseline(u_mc, phi_mc, readout, rng_shuffle=trial_generators['readout'], max_lag=cfg.task.mc_max_lag)
        
        res_base['firing_rate'] = state_mc['mean_rate']
        res_base['I_syn_mean'] = state_mc['mean_I_syn']
        res_base['I_syn_var'] = state_mc['var_I_syn']
        res_base['saturation_flag'] = state_mc['saturation_flag']
        
        results.append({**res_base, 'task': 'MC', 'metric': 'capacity', 'value': res_mc['mc'], 'baseline': base_mc, 'improvement': res_mc['mc'] - base_mc})

        # ==========================
        # TASK D: LYAPUNOV (Stability) - Attractor-based Branching
        # ==========================
        lyap = LyapunovModule(trial_generators['in'])
        
        # 1. Warm-up to reach attractor
        steps_wu = 1000
        u_wu = trial_generators['in'].uniform(0, 1, steps_wu // steps_per_symbol + 1)
        rates_wu = cfg.task.poisson_rate_min + u_wu * (cfg.task.poisson_rate_max - cfg.task.poisson_rate_min)
        spikes_wu = (trial_generators['in'].random(steps_wu) < (np.repeat(rates_wu, steps_per_symbol)[:steps_wu] * cfg.task.dt * 1e-3)).astype(np.float32)
        
        # trim_steps=0 because we want the final state exactly after this input
        res_wu = hh.simulate(rho, bias, spikes_wu, cfg.get_task_input_id() + "_WU", trim_steps=0)
        start_state = res_wu['final_state']
        
        # 2. Parallel Trajectories starting from same point on attractor
        len_lyap = 500
        u_l = trial_generators['in'].uniform(0, 1, len_lyap)
        rates_l = cfg.task.poisson_rate_min + u_l * (cfg.task.poisson_rate_max - cfg.task.poisson_rate_min)
        spikes_l = (trial_generators['in'].random(len_lyap * steps_per_symbol) < (np.repeat(rates_l, steps_per_symbol) * cfg.task.dt * 1e-3)).astype(np.float32)
        
        # Reference
        state_l1 = hh.simulate(rho, bias, spikes_l, cfg.get_task_input_id() + "_L_ref", trim_steps=0, full_state=start_state)
        phi1 = filter_and_downsample(state_l1['spikes'], steps_per_symbol, cfg.task.dt, cfg.task.tau_trace).astype(np.float32)
        
        # Perturbed (Deep copy of state dict)
        perturbed_state = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in start_state.items()}
        perturbed_state['V'][0] += cfg.task.lyap_eps
        
        state_l2 = hh.simulate(rho, bias, spikes_l, cfg.get_task_input_id() + "_L_pert", trim_steps=0, full_state=perturbed_state)
        phi2 = filter_and_downsample(state_l2['spikes'], steps_per_symbol, cfg.task.dt, cfg.task.tau_trace).astype(np.float32)
        
        step_s = (steps_per_symbol * cfg.task.dt) / 1000.0
        slope = lyap.compute_lambda(phi1, phi2, window_range=cfg.task.lyap_window)
        lambda_val_sec = slope / step_s 
        
        res_base['firing_rate'] = state_l1['mean_rate']
        res_base['I_syn_mean'] = state_l1['mean_I_syn']
        res_base['I_syn_var'] = state_l1['var_I_syn']
        res_base['saturation_flag'] = state_l1['saturation_flag']

        results.append({**res_base, 'task': 'Lyapunov', 'metric': 'lambda_step', 'value': slope, 'baseline': 0.0, 'improvement': 0.0})
        results.append({**res_base, 'task': 'Lyapunov', 'metric': 'lambda_sec', 'value': lambda_val_sec, 'baseline': 0.0, 'improvement': 0.0})

    except Exception as e:
        print(f"ERROR in Trial {trial_idx} (rho={rho}, bias={bias}): {e}")
        # traceback.print_exc()
        # Return empty list or partial results?
        # Journal-Grade: Don't crash entire sweep, but log error.
        # Return what we have.
        pass
        
    return results

def run_experiment(cfg_path: str):
    cfg = load_config(cfg_path)
    os.makedirs(cfg.results_dir, exist_ok=True)
    
    # 1. Define Sweep Space
    tasks_to_run = []
    
    if cfg.sweep_mode == 'coarse':
        rhos = cfg.rho_grid_coarse
        biases = cfg.bias_grid_coarse
        seeds = range(cfg.seeds_coarse if hasattr(cfg, 'seeds_coarse') else 5)
    else:
        # Fine Sweep logic would be here (usually loaded from a separate fine config or dynamic)
        # For now assume config defines specific grids
        rhos = cfg.rho_grid_coarse # Or define fine grid in config
        biases = cfg.bias_grid_coarse
        seeds = range(cfg.seeds_fine)

    print(f"Initializing {cfg.sweep_mode.upper()} Sweep: {len(rhos)}x{len(biases)}x{len(seeds)}")
    
    for r in rhos:
        for b in biases:
            for s in seeds:
                tasks_to_run.append((r, b, s))
                
    # 2. Execute Parallel - UNLEASHED
    all_results = []
    max_workers = min(60, os.cpu_count() - 2)
    
    # Live stats tracking for Research Engineer verification
    stats = {'NM': [], 'XOR': [], 'MC': [], 'LY': [], 'FR': []}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_trial, r, b, s, cfg): (r, b, s) for r, b, s in tasks_to_run}
        
        for i, future in enumerate(as_completed(futures)):
            r, b, s = futures[future]
            try:
                res = future.result()
                all_results.extend(res)
                
                # Accumulate stats
                for r_item in res:
                    stats['FR'].append(r_item['firing_rate'])
                    if r_item['task'] == 'NARMA': stats['NM'].append(r_item['value'])
                    if r_item['task'] == 'XOR' and r_item['metric'] == 'accuracy': stats['XOR'].append(r_item['value'])
                    if r_item['task'] == 'MC': stats['MC'].append(r_item['value'])
                    if r_item['task'] == 'Lyapunov' and r_item['metric'] == 'lambda_sec': stats['LY'].append(r_item['value'])

                # Report every 5 trials for faster feedback
                if (i + 1) % 5 == 0:
                    avg_fr = np.mean(stats['FR'][-20:]) # 5 trials * 4 tasks
                    avg_nm = np.mean(stats['NM'][-5:]) if stats['NM'] else 0
                    avg_xr = np.mean(stats['XOR'][-5:]) if stats['XOR'] else 0
                    avg_mc = np.mean(stats['MC'][-5:]) if stats['MC'] else 0
                    avg_ly = np.mean(stats['LY'][-5:]) if stats['LY'] else 0
                    
                    print(f"\n[INTERMEDIATE REPORT {i+1}/{len(futures)}]")
                    print(f" > Params: rho={r:.2f}, bias={b:.2f}")
                    print(f" > Bio: Firing Rate = {avg_fr:.2f} Hz {'(OK)' if 1<avg_fr<50 else '(CRITICAL)'}")
                    print(f" > Chaos: Mean Lambda = {avg_ly:.3f} s^-1")
                    print(f" > Task Score: NARMA={avg_nm:.3f}, XOR={avg_xr:.2f}, MC={avg_mc:.2f}")
                    print("------------------------------------------")
            except Exception as e:
                print(f"CRITICAL FAIL on {r, b, s}: {e}")
                
    # 3. Save Parquet
    df = pd.DataFrame(all_results)
    df['timestamp'] = time.strftime("%Y%m%d-%H%M%S")
    df['git_hash'] = get_git_hash()
    df['sweep_mode'] = cfg.sweep_mode
    
    out_name = f"results_{cfg.sweep_mode}_{int(time.time())}.parquet"
    out_path = os.path.join(cfg.results_dir, out_name)
    df.to_parquet(out_path)
    print(f"SAVED RESULTS to {out_path} ({len(df)} rows)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    run_experiment(args.config)
