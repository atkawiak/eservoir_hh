#!/usr/bin/env python3
import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
from config import ExperimentConfig, HHConfig, TaskConfig
from rng_manager import RNGManager
from hh_model import HHModel
from tasks.lyapunov_task import LyapunovModule
from utils import filter_and_downsample

# ============================================================
# SEARCH FOR BIOLOGICAL EDGE
# ============================================================
SEED = 42
N_NEURONS = 100
GA = 20.0          # Standard
GL = 0.3          # Standard
BIAS = 5.0        # Standard
IN_GAIN = 5.0     # Standard Poission Input

# We sweep RHO and BIAS to find a region where it diverges but stays in 10-80Hz
RHO_VALUES = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0]
BIAS_VALUES = [0.0, 2.0, 5.0, 10.0]

T_SYMBOLS = 150
LYAP_EPS = 0.1     # Moderate perturbation

def main():
    print("=" * 70)
    print("SEARCHING FOR BIOLOGICAL EDGE OF CHAOS")
    print(f"Seed={SEED}, gA={GA}, gL={GL}")
    print("=" * 70)
    
    rng_mgr = RNGManager(2025)
    trial_gens = rng_mgr.get_trial_generators(SEED)
    seeds_tuple = rng_mgr.get_trial_seeds_tuple(SEED)
    
    cfg = ExperimentConfig()
    cfg.hh = HHConfig(N=N_NEURONS, gL=GL, gA=GA, in_gain=IN_GAIN)
    cfg.task = TaskConfig(dt=0.05, symbol_ms=20.0)
    
    steps_per_symbol = int(cfg.task.symbol_ms / cfg.task.dt)
    
    # Generate common input
    u_input = trial_gens['in'].uniform(0, 1, T_SYMBOLS)
    rates = cfg.task.poisson_rate_min + u_input * (cfg.task.poisson_rate_max - cfg.task.poisson_rate_min)
    rates_up = np.repeat(rates, steps_per_symbol)
    spikes_in = (trial_gens['in'].random(len(rates_up)) < (rates_up * cfg.task.dt * 1e-3)).astype(float)

    results = []
    
    for rho in RHO_VALUES:
        for bias in BIAS_VALUES:
            print(f"rho={rho:4.1f} bias={bias:4.1f} |", end=" ", flush=True)
            
            trial_gens_fresh = rng_mgr.get_trial_generators(SEED)
            hh = HHModel(cfg, trial_gens_fresh, seeds_tuple)
            
            # Ref
            s1 = hh.simulate(rho, bias, spikes_in, "ref", trim_steps=500)
            
            # Pert
            V_start = np.full(N_NEURONS, -65.0)
            V_start[0] += LYAP_EPS
            m0, h0, n0 = hh.get_steady_state(V_start)
            pert_state = {'V': V_start, 'm': m0, 'h': h0, 'n': n0, 'b_gate': np.full(N_NEURONS, 1.0), 's_trace': np.zeros(N_NEURONS), 's_in_trace': 0.0}
            s2 = hh.simulate(rho, bias, spikes_in, "pert", trim_steps=500, full_state=pert_state)
            
            diff_spks = np.sum(s1['spikes'] != s2['spikes'])
            fr = s1['mean_rate']
            sat = s1['saturation_flag']
            
            print(f" FR={fr:5.1f}Hz | DiffSpks={diff_spks:5d} | SAT={sat}", end=" ")
            
            if diff_spks > 0 and 1.0 < fr < 100.0 and not sat:
                print(" << POTENTIAL EDGE >>")
            else:
                print("")
                
            results.append({'rho': rho, 'bias': bias, 'fr': fr, 'diff': diff_spks, 'sat': sat})

    df = pd.DataFrame(results)
    df.to_csv("results_bifurcation/edge_search.csv", index=False)

if __name__ == "__main__":
    main()
