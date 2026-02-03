#!/usr/bin/env python3
"""
Find Triplet WITHOUT A-current (gA=0) for easier chaos detection.
"""
import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from config import ExperimentConfig, HHConfig, TaskConfig
from rng_manager import RNGManager
from hh_model import HHModel
from utils import filter_and_downsample

SEED = 42

def measure(hh, cfg, rho, bias, spikes_in):
    LYAP_EPS = 0.1
    s1 = hh.simulate(rho, bias, spikes_in, "ref", trim_steps=500)
    
    V_start = np.full(cfg.hh.N, -65.0)
    V_start[0] += LYAP_EPS
    m0, h0, n0 = hh.get_steady_state(V_start)
    pert = {'V': V_start, 'm': m0, 'h': h0, 'n': n0, 
            'b_gate': np.full(cfg.hh.N, 1.0), 's_trace': np.zeros(cfg.hh.N), 's_in_trace': 0.0}
    
    s2 = hh.simulate(rho, bias, spikes_in, "pert", trim_steps=500, full_state=pert)
    diff = np.sum(s1['spikes'] != s2['spikes'])
    return {'rho': rho, 'diff': diff, 'fr': s1['mean_rate'], 'sat': s1['saturation_flag']}

def main():
    print("=" * 60)
    print("TRIPLET SEARCH WITHOUT A-CURRENT (gA=0)")
    print("=" * 60)
    
    rng_mgr = RNGManager(2025)
    trial_gens = rng_mgr.get_trial_generators(SEED)
    seeds_tuple = rng_mgr.get_trial_seeds_tuple(SEED)
    
    cfg = ExperimentConfig()
    cfg.hh = HHConfig(N=100, gL=0.3, gA=0.0)  # NO A-CURRENT
    cfg.task = TaskConfig(dt=0.05, symbol_ms=20.0)
    
    steps_per_symbol = int(cfg.task.symbol_ms / cfg.task.dt)
    u = trial_gens['in'].uniform(0, 1, 150)
    rates = cfg.task.poisson_rate_min + u * (cfg.task.poisson_rate_max - cfg.task.poisson_rate_min)
    rates_up = np.repeat(rates, steps_per_symbol)
    spikes_in = (trial_gens['in'].random(len(rates_up)) < (rates_up * cfg.task.dt * 1e-3)).astype(float)

    hh = HHModel(cfg, trial_gens, seeds_tuple)
    
    # Sweep rho
    results = []
    for rho in [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        m = measure(hh, cfg, rho, 5.0, spikes_in)
        sat_str = "SAT!" if m['sat'] else "OK"
        print(f"rho={rho:4.1f} | FR={m['fr']:6.1f}Hz | Diff={m['diff']:6d} | {sat_str}")
        results.append(m)
    
    # Classify
    stable = [r for r in results if r['diff'] < 100 and not r['sat']]
    edge = [r for r in results if 100 <= r['diff'] < 5000 and not r['sat'] and r['fr'] < 100]
    chaotic = [r for r in results if r['diff'] >= 5000 and not r['sat'] and r['fr'] < 500]
    
    print("\n--- TRIPLET CANDIDATES ---")
    if stable: print(f"STABLE:  rho={stable[0]['rho']}")
    if edge: print(f"EDGE:    rho={edge[0]['rho']}")
    if chaotic: print(f"CHAOTIC: rho={chaotic[0]['rho']}")

if __name__ == "__main__":
    main()
