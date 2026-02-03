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
# TARGET REGIONS
# ============================================================
# Stable:   DiffSpks < 10,   λ < 0
# Edge:     DiffSpks > 500,  FR ~ 30Hz, Not Saturated
# Chaotic:  DiffSpks > 5000, FR < 100Hz, Not Saturated

def measure(hh, cfg, rho, bias, spikes_in, label):
    s1 = hh.simulate(rho, bias, spikes_in, f"{label}_ref", trim_steps=500)
    
    LYAP_EPS = 0.1
    V_start = np.full(cfg.hh.N, -65.0)
    V_start[0] += LYAP_EPS
    m0, h0, n0 = hh.get_steady_state(V_start)
    pert_state = {'V': V_start, 'm': m0, 'h': h0, 'n': n0, 'b_gate': np.full(cfg.hh.N, 1.0), 's_trace': np.zeros(cfg.hh.N), 's_in_trace': 0.0}
    
    s2 = hh.simulate(rho, bias, spikes_in, f"{label}_pert", trim_steps=500, full_state=pert_state)
    diff = np.sum(s1['spikes'] != s2['spikes'])
    
    return {'rho': rho, 'bias': bias, 'diff': diff, 'fr': s1['mean_rate'], 'sat': s1['saturation_flag']}

def main():
    SEED = 42
    print(f"GENERATING PERFECT TRIPLET FOR SEED {SEED}")
    
    rng_mgr = RNGManager(2025)
    trial_gens = rng_mgr.get_trial_generators(SEED)
    seeds_tuple = rng_mgr.get_trial_seeds_tuple(SEED)
    
    cfg = ExperimentConfig()
    cfg.hh = HHConfig(N=100, gL=0.3, gA=20.0)
    cfg.task = TaskConfig(dt=0.05, symbol_ms=20.0)
    
    steps_per_symbol = int(cfg.task.symbol_ms / cfg.task.dt)
    u_input = trial_gens['in'].uniform(0, 1, 150)
    rates = cfg.task.poisson_rate_min + u_input * (cfg.task.poisson_rate_max - cfg.task.poisson_rate_min)
    rates_up = np.repeat(rates, steps_per_symbol)
    spikes_in = (trial_gens['in'].random(len(rates_up)) < (rates_up * cfg.task.dt * 1e-3)).astype(float)

    hh = HHModel(cfg, trial_gens, seeds_tuple)

    # 1. THE EDGE (Confirmed)
    edge = measure(hh, cfg, 5.0, 5.0, spikes_in, "edge")
    print(f"EDGE:    rho={edge['rho']} bias={edge['bias']} | FR={edge['fr']:.1f}Hz | Diff={edge['diff']}")

    # 2. STABLE (Increase gA)
    # Actually, let's just use lower rho
    stable = measure(hh, cfg, 0.1, 5.0, spikes_in, "stable")
    print(f"STABLE:  rho={stable['rho']} bias={stable['bias']} | FR={stable['fr']:.1f}Hz | Diff={stable['diff']}")

    # 3. CHAOTIC (Find point between 5 and 10)
    for test_rho in [5.2, 5.5, 5.7, 6.0, 6.5]:
        c = measure(hh, cfg, test_rho, 5.0, spikes_in, "chaotic_test")
        print(f"TEST {test_rho}: FR={c['fr']:.1f}Hz | Diff={c['diff']} | SAT={c['sat']}")
        if not c['sat'] and c['diff'] > 2000:
            chaotic = c
            break

    print("\n--- FINAL TRIPLET ---")
    print(f"STABLE:  rho={stable['rho']}, bias={stable['bias']}, FR={stable['fr']:.1f}")
    print(f"EDGE:    rho={edge['rho']}, bias={edge['bias']}, FR={edge['fr']:.1f}")
    print(f"CHAOTIC: rho={chaotic['rho']}, bias={chaotic['bias']}, FR={chaotic['fr']:.1f}")

if __name__ == "__main__":
    main()
