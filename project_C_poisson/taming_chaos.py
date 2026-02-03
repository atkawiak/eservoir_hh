#!/usr/bin/env python3
import sys, os
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
from config import ExperimentConfig, HHConfig, TaskConfig
from rng_manager import RNGManager
from hh_model import HHModel

SEED = 42

def measure(hh, cfg, rho, bias, spikes_in, gA):
    # We update gA in the config and recreate/reset HH if needed
    # But for a sweep, we can just pass it directly if the model allows
    s1 = hh.simulate(rho, bias, spikes_in, f"ref_gA_{gA}", trim_steps=500, gL=None)
    
    LYAP_EPS = 0.1
    V_start = np.full(cfg.hh.N, -65.0); V_start[0] += LYAP_EPS
    m0, h0, n0 = hh.get_steady_state(V_start)
    pert = {'V': V_start, 'm': m0, 'h': h0, 'n': n0, 'b_gate': np.full(cfg.hh.N, 1.0), 's_trace': np.zeros(cfg.hh.N), 's_in_trace': 0.0}
    
    s2 = hh.simulate(rho, bias, spikes_in, f"pert_gA_{gA}", trim_steps=500, full_state=pert, gL=None)
    diff = np.sum(s1['spikes'] != s2['spikes'])
    return diff, s1['mean_rate'], s1['saturation_flag']

def main():
    print("=" * 60)
    print("TAMING CHAOS: Sweep gA for fixed RHO=7.5")
    print("=" * 60)
    
    rng_mgr = RNGManager(2025)
    tg = rng_mgr.get_trial_generators(SEED)
    st = rng_mgr.get_trial_seeds_tuple(SEED)
    
    # We want a high RHO that was chaotic before
    FIXED_RHO = 7.5
    BIAS = 5.0
    
    gA_values = [0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0]
    results = []

    for gA in gA_values:
        print(f"gA = {gA:4.1f} |", end=" ", flush=True)
        
        cfg = ExperimentConfig()
        cfg.hh = HHConfig(N=100, gL=0.3, gA=gA)
        cfg.task = TaskConfig(dt=0.05, symbol_ms=20.0)
        sps = int(cfg.task.symbol_ms / cfg.task.dt)
        
        # Consistent input
        tg_in = RNGManager(2025).get_trial_generators(SEED)['in']
        u = tg_in.uniform(0, 1, 150)
        rates = cfg.task.poisson_rate_min + u * (cfg.task.poisson_rate_max - cfg.task.poisson_rate_min)
        rates_up = np.repeat(rates, sps)
        spikes_in = (tg_in.random(len(rates_up)) < (rates_up * cfg.task.dt * 1e-3)).astype(float)
        
        # Fresh model with current gA
        hh = HHModel(cfg, tg, st)
        
        diff, fr, sat = measure(hh, cfg, FIXED_RHO, BIAS, spikes_in, gA)
        
        print(f" FR={fr:5.1f}Hz | Diff={diff:6d} | SAT={sat}", end=" ")
        if diff == 0:
            print(" << STABLE >>")
        elif diff < 500:
            print(" << EDGE CANDIDATE >>")
        else:
            print(" << CHAOTIC >>")
            
        results.append({'gA': gA, 'fr': fr, 'diff': diff, 'sat': sat})

    df = pd.DataFrame(results)
    os.makedirs("results_bifurcation", exist_ok=True)
    df.to_csv("results_bifurcation/gA_taming_sweep.csv", index=False)

if __name__ == "__main__":
    main()
