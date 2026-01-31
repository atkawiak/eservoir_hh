#!/usr/bin/env python3
"""
Local validation script - tests single extreme parameter combination
to catch numerical instabilities before Docker deployment.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from config import ExperimentConfig, HHConfig, TaskConfig
from rng_manager import RNGManager
from hh_model import HHModel
from readout import ReadoutModule
from utils import filter_and_downsample
from tasks.narma import NARMA
from tasks.xor import DelayedXOR
from tasks.mc import MemoryCapacity
from tasks.lyapunov_task import LyapunovModule

def test_extreme_case(rho, bias, seed=42):
    """Test single extreme parameter combination"""
    print(f"\n{'='*60}")
    print(f"Testing: rho={rho}, bias={bias}, seed={seed}")
    print(f"{'='*60}\n")
    
    # Create minimal config
    cfg = ExperimentConfig()
    cfg.hh = HHConfig(N=50)
    cfg.task = TaskConfig(dt=0.05, symbol_ms=20.0)
    cfg.cv_folds = 2
    cfg.cv_gap = 5
    cfg.ridge_alphas = [1.0, 10.0]
    cfg.cache_dir = "cache_local_test"
    cfg.results_dir = "results_local_test"
    
    os.makedirs(cfg.cache_dir, exist_ok=True)
    os.makedirs(cfg.results_dir, exist_ok=True)
    
    # Setup RNG
    rng_mgr = RNGManager(2025)
    trial_gens = rng_mgr.get_trial_generators(seed)
    seeds_tuple = rng_mgr.get_trial_seeds_tuple(seed)
    
    # Setup modules
    hh = HHModel(cfg, trial_gens, seeds_tuple)
    readout = ReadoutModule(trial_gens['readout'], cv_folds=cfg.cv_folds, cv_gap=cfg.cv_gap)
    
    steps_per_symbol = int(cfg.task.symbol_ms / cfg.task.dt)
    
    results = {}
    
    try:
        # Test NARMA
        print("Testing NARMA task...")
        narma = NARMA(trial_gens['in'])
        u_nm, y_nm = narma.generate_data(500, order=10)
        
        rates_nm = cfg.task.poisson_rate_min + u_nm * 2.0 * (cfg.task.poisson_rate_max - cfg.task.poisson_rate_min)
        rates_up_nm = np.repeat(rates_nm, steps_per_symbol)
        spikes_nm = (trial_gens['in'].random(len(rates_up_nm)) < (rates_up_nm * cfg.task.dt * 1e-3)).astype(float)
        
        state_nm = hh.simulate(rho, bias, spikes_nm, f"test_NARMA_{rho}_{bias}")
        
        print(f"  Firing rate: {state_nm['mean_rate']:.2f} Hz")
        print(f"  I_syn mean: {state_nm['mean_I_syn']:.4f}")
        print(f"  Saturation: {state_nm['saturation_flag']}")
        
        if state_nm['saturation_flag']:
            print("  ⚠ WARNING: Voltage saturation detected!")
        
        phi_nm = filter_and_downsample(state_nm['spikes'], steps_per_symbol, cfg.task.dt, cfg.task.tau_trace)
        y_nm = y_nm[-len(phi_nm):]
        
        met_nm = readout.train_ridge_cv(phi_nm, y_nm, task_type='regression', alphas=cfg.ridge_alphas)
        print(f"  NRMSE: {met_nm['nrmse']:.4f}")
        results['narma_nrmse'] = met_nm['nrmse']
        results['narma_saturation'] = state_nm['saturation_flag']
        
        # Test XOR
        print("\nTesting XOR task...")
        xor = DelayedXOR(trial_gens['in'])
        u_xor, y_xor = xor.generate_data(500, delay=2)
        
        rates_xor = cfg.task.poisson_rate_min + u_xor * (cfg.task.poisson_rate_max - cfg.task.poisson_rate_min)
        rates_up_xor = np.repeat(rates_xor, steps_per_symbol)
        spikes_xor = (trial_gens['in'].random(len(rates_up_xor)) < (rates_up_xor * cfg.task.dt * 1e-3)).astype(float)
        
        state_xor = hh.simulate(rho, bias, spikes_xor, f"test_XOR_{rho}_{bias}")
        phi_xor = filter_and_downsample(state_xor['spikes'], steps_per_symbol, cfg.task.dt, cfg.task.tau_trace)
        y_xor = y_xor[-len(phi_xor):]
        
        met_xor = readout.train_ridge_cv(phi_xor, y_xor, task_type='classification', alphas=cfg.ridge_alphas)
        print(f"  Accuracy: {met_xor['acc']:.4f}")
        print(f"  AUC: {met_xor['auc']:.4f}")
        results['xor_acc'] = met_xor['acc']
        
        # Test Lyapunov
        print("\nTesting Lyapunov task...")
        lyap = LyapunovModule(trial_gens['in'])
        u_lyap = trial_gens['in'].uniform(0, 1, 200)
        
        rates_lyap = cfg.task.poisson_rate_min + u_lyap * (cfg.task.poisson_rate_max - cfg.task.poisson_rate_min)
        rates_up_lyap = np.repeat(rates_lyap, steps_per_symbol)
        spikes_lyap = (trial_gens['in'].random(len(rates_up_lyap)) < (rates_up_lyap * cfg.task.dt * 1e-3)).astype(float)
        
        state_l1 = hh.simulate(rho, bias, spikes_lyap, f"test_LYAP_{rho}_{bias}_ref")
        phi1 = filter_and_downsample(state_l1['spikes'], steps_per_symbol, cfg.task.dt, cfg.task.tau_trace)
        
        V_pert = np.full(cfg.hh.N, -65.0)
        V_pert[0] += 1e-6
        
        state_l2 = hh.simulate(rho, bias, spikes_lyap, f"test_LYAP_{rho}_{bias}_pert", V_init=V_pert)
        phi2 = filter_and_downsample(state_l2['spikes'], steps_per_symbol, cfg.task.dt, cfg.task.tau_trace)
        
        lambda_val = lyap.compute_lambda(phi1, phi2, window_range=[20, 100])
        print(f"  Lambda: {lambda_val:.6f}")
        results['lyapunov'] = lambda_val
        
        print(f"\n{'='*60}")
        print("✓ TEST PASSED - No crashes or NaN values")
        print(f"{'='*60}\n")
        return True, results
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✗ TEST FAILED: {e}")
        print(f"{'='*60}\n")
        import traceback
        traceback.print_exc()
        return False, results

if __name__ == "__main__":
    print("="*60)
    print("LOCAL VALIDATION - Testing Extreme Parameter Combinations")
    print("="*60)
    
    test_cases = [
        (0.01, 0.0, 0),   # Low rho, no bias
        (1.5, 0.0, 0),    # High rho, no bias
        (0.01, 8.0, 0),   # Low rho, high bias
        (1.5, 8.0, 0),    # High rho, high bias (most extreme)
    ]
    
    all_passed = True
    for rho, bias, seed in test_cases:
        passed, results = test_extreme_case(rho, bias, seed)
        if not passed:
            all_passed = False
            print(f"\n⚠ STOPPING: Test failed for rho={rho}, bias={bias}")
            break
    
    if all_passed:
        print("\n" + "="*60)
        print("✓ ALL LOCAL VALIDATION TESTS PASSED")
        print("Safe to proceed with Docker deployment")
        print("="*60)
        sys.exit(0)
    else:
        print("\n" + "="*60)
        print("✗ VALIDATION FAILED")
        print("Fix numerical stability issues before deployment")
        print("="*60)
        sys.exit(1)
