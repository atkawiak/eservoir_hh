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
# SUPER-EXCITABLE CONFIG (Autonomous Chaos Search)
# ============================================================
SEED = 42
N_NEURONS = 100
GA = 0.0                     # NO A-CURRENT (easier chaos)
BIAS = 15.0                  # HIGH BIAS
GL = 0.1                     # LOW LEAK
IN_GAIN = 0.0                # NO INPUT

RHO_MIN = 1.0
RHO_MAX = 20.0
RHO_STEPS = 10

T_SYMBOLS = 100
LYAP_EPS = 1.0

def main():
    print("=" * 60)
    print("AUTONOMOUS CHAOS SEARCH (No Input)")
    print(f"Seed={SEED}, bias={BIAS}, gL={GL}, gA={GA}")
    print("=" * 60)
    
    rng_mgr = RNGManager(2025)
    trial_gens = rng_mgr.get_trial_generators(SEED)
    seeds_tuple = rng_mgr.get_trial_seeds_tuple(SEED)
    
    cfg = ExperimentConfig()
    cfg.hh = HHConfig(N=N_NEURONS, gL=GL, gA=GA, in_gain=IN_GAIN)
    cfg.task = TaskConfig(dt=0.05, symbol_ms=20.0, lyap_eps=LYAP_EPS, lyap_window=(30, 80))
    
    steps_per_symbol = int(cfg.task.symbol_ms / cfg.task.dt)
    spikes_in = np.zeros(T_SYMBOLS * steps_per_symbol) # Empty input
    
    rho_values = np.linspace(RHO_MIN, RHO_MAX, RHO_STEPS)
    results = []
    
    for rho in rho_values:
        print(f"rho = {rho:5.2f}", end=" ", flush=True)
        trial_gens_fresh = rng_mgr.get_trial_generators(SEED)
        hh = HHModel(cfg, trial_gens_fresh, seeds_tuple)
        
        # Sim 1
        s1 = hh.simulate(rho, BIAS, spikes_in, "ref", trim_steps=0)
        phi1 = filter_and_downsample(s1['spikes'], steps_per_symbol, cfg.task.dt, cfg.task.tau_trace)
        
        # Sim 2 (Perturbed)
        V_start = np.full(N_NEURONS, -65.0)
        V_start[0] += LYAP_EPS
        m0, h0, n0 = hh.get_steady_state(V_start)
        pert_state = {'V': V_start, 'm': m0, 'h': h0, 'n': n0, 
                      'b_gate': np.full(N_NEURONS, 1.0), 
                      's_trace': np.zeros(N_NEURONS), 's_in_trace': 0.0}
        
        s2 = hh.simulate(rho, BIAS, spikes_in, "pert", trim_steps=0, full_state=pert_state)
        phi2 = filter_and_downsample(s2['spikes'], steps_per_symbol, cfg.task.dt, cfg.task.tau_trace)
        
        # DEBUG
        V1_final = s1['final_state']['V']
        V2_final = s2['final_state']['V']
        v_diff = np.linalg.norm(V1_final - V2_final)
        print(f"| VDiff={v_diff:.4e}", end=" ")
        
        diff_spks = np.sum(s1['spikes'] != s2['spikes'])
        lyap = LyapunovModule(np.random.default_rng(0))
        slope = lyap.compute_lambda(phi1, phi2, window_range=cfg.task.lyap_window)
        
        lam = slope / ((steps_per_symbol * cfg.task.dt) / 1000.0)
        regime = "STABLE" if lam < -0.1 else ("EDGE" if abs(lam) < 0.1 else "CHAOTIC")
        print(f"| λ={lam:+.4f} | DiffSpks={diff_spks} | FR={s1['mean_rate']:.1f}Hz | {regime}")
        results.append({'rho': rho, 'lambda': lam, 'fr': s1['mean_rate']})

if __name__ == "__main__":
    main()
