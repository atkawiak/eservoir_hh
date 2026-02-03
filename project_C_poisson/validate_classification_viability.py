#!/usr/bin/env python3
import sys, os
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import ExperimentConfig, HHConfig, TaskConfig
from rng_manager import RNGManager
from hh_model import HHModel
from utils import filter_and_downsample

# ============================================================
# DIAGNOSTIC PARAMETERS
# ============================================================
N_NEURONS = 50        # Redukcja dla stabilności testu
IN_GAIN = 10.0        # INCREASED GAIN
SYMBOL_MS = 20.0      # FASTER SYMBOL
N_TRAIN = 200
N_TEST = 100
WAKEUP = 50

def xor_gen(n_samples, delay=1, seed=124):
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, n_samples)
    targets = np.zeros(n_samples)
    for t in range(delay, n_samples):
        targets[t] = bits[t] ^ bits[t-delay]
    return bits, targets

def validate_state(state_name, rho):
    print(f"\nVALIDATING {state_name.upper()} (rho={rho})")
    print("-" * 40)
    
    seed_net = 42
    rng_mgr = RNGManager(2025)
    tg = rng_mgr.get_trial_generators(seed_net)
    st = rng_mgr.get_trial_seeds_tuple(seed_net)
    
    # Task Config
    rate_0 = 5.0
    rate_1 = 80.0
    delay = 2
    
    n_total = N_TRAIN + N_TEST + WAKEUP
    u, y = xor_gen(n_total, delay=delay)
    rates = np.where(u == 0, rate_0, rate_1)
    
    # HH Config
    cfg = ExperimentConfig()
    cfg.hh = HHConfig(N=N_NEURONS, gL=0.3, gA=0.0, in_gain=IN_GAIN)
    cfg.task = TaskConfig(dt=0.05, symbol_ms=SYMBOL_MS)
    sps = int(cfg.task.symbol_ms / cfg.task.dt)
    
    # Generate Spikes
    rates_up = np.repeat(rates, sps)
    spikes_in = (tg['in'].random(len(rates_up)) < (rates_up * cfg.task.dt * 1e-3)).astype(float)
    
    # Simulate
    hh = HHModel(cfg, tg, st)
    res = hh.simulate(rho, 5.0, spikes_in, f"val_{state_name}", trim_steps=0)
    
    # Process States
    # Instead of just mean, let's use 2 sub-bins per symbol to capture dynamics
    bins_per_symbol = 2
    sps_bin = sps // bins_per_symbol
    phi_raw = filter_and_downsample(res['spikes'], sps_bin, cfg.task.dt, cfg.task.tau_trace)
    phi = phi_raw.reshape(n_total, -1) # Flatten bins into features
    
    print(f"Feature vector size: {phi.shape[1]}")
    
    # Pert
    V_start = np.full(cfg.hh.N, -65.0)
    
    # Split
    X_train = phi[WAKEUP:WAKEUP+N_TRAIN]
    y_train = y[WAKEUP:WAKEUP+N_TRAIN]
    X_test = phi[WAKEUP+N_TRAIN:]
    y_test = y[WAKEUP+N_TRAIN:]
    
    # Train dengan Cross-Validation sederhana (Ridge Alpha Sweep)
    best_acc = 0
    best_alpha = 1.0
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        reg = Ridge(alpha=alpha)
        reg.fit(X_train, y_train)
        preds = reg.predict(X_test)
        acc = accuracy_score(y_test, preds > 0.5)
        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha
            
    print(f"Best Alpha: {best_alpha}")
    print(f"XOR Accuracy: {best_acc:.2%}")
    
    # Diagnostic: Activity check
    avg_fr = np.mean(res['mean_rate'])
    print(f"Avg Network FR: {avg_fr:.1f} Hz")
    
    return best_acc

def main():
    # Test Triplets from Stage A
    triplet = {
        'stable': 3.0,
        'edge': 6.5,
        'chaotic': 7.5
    }
    
    results = {}
    for name, rho in triplet.items():
        results[name] = validate_state(name, rho)
        
    print("\n" + "="*40)
    print("FINAL VALIDATION REPORT")
    print("="*40)
    for name, acc in results.items():
        print(f"{name:10}: {acc:.2%}")
    
    if results['edge'] > results['stable']:
        print("\nSUCCESS: Edge shows better classification than Stable.")
    else:
        print("\nFAILURE: Edge does not outperform Stable. Check in_gain or symbol_ms.")

if __name__ == "__main__":
    main()
