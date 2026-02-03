#!/usr/bin/env python3
import sys, os
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from config import ExperimentConfig, HHConfig, TaskConfig
from rng_manager import RNGManager
from hh_model import HHModel

SEED = 42

def measure(hh, cfg, rho, bias, spikes_in):
    s1 = hh.simulate(rho, bias, spikes_in, "ref", trim_steps=500)
    V_start = np.full(cfg.hh.N, -65.0); V_start[0] += 0.1
    m0, h0, n0 = hh.get_steady_state(V_start)
    pert = {'V': V_start, 'm': m0, 'h': h0, 'n': n0, 'b_gate': np.full(cfg.hh.N, 1.0), 's_trace': np.zeros(cfg.hh.N), 's_in_trace': 0.0}
    s2 = hh.simulate(rho, bias, spikes_in, "pert", trim_steps=500, full_state=pert)
    return np.sum(s1['spikes'] != s2['spikes']), s1['mean_rate'], s1['saturation_flag']

def main():
    print("FINE-GRAINED TRIPLET SEARCH (gA=0)")
    rng_mgr = RNGManager(2025)
    tg = rng_mgr.get_trial_generators(SEED)
    st = rng_mgr.get_trial_seeds_tuple(SEED)
    
    cfg = ExperimentConfig()
    cfg.hh = HHConfig(N=100, gL=0.3, gA=0.0)
    cfg.task = TaskConfig(dt=0.05, symbol_ms=20.0)
    sps = int(cfg.task.symbol_ms / cfg.task.dt)
    u = tg['in'].uniform(0, 1, 150)
    rates = cfg.task.poisson_rate_min + u * (cfg.task.poisson_rate_max - cfg.task.poisson_rate_min)
    rates_up = np.repeat(rates, sps)
    spikes_in = (tg['in'].random(len(rates_up)) < (rates_up * cfg.task.dt * 1e-3)).astype(float)
    
    hh = HHModel(cfg, tg, st)
    
    for rho in np.arange(3.0, 8.0, 0.5):
        diff, fr, sat = measure(hh, cfg, rho, 5.0, spikes_in)
        print(f"rho={rho:4.1f} | FR={fr:6.1f}Hz | Diff={diff:6d} | SAT={sat}")

if __name__ == "__main__":
    main()
