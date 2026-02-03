#!/usr/bin/env python3
import sys, os
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, accuracy_score

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import ExperimentConfig, HHConfig, TaskConfig
from rng_manager import RNGManager
from hh_model import HHModel
from utils import filter_and_downsample

def xor_gen(n_samples, delay=1, seed=124):
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, n_samples)
    targets = np.zeros(n_samples)
    for t in range(delay, n_samples):
        targets[t] = bits[t] ^ bits[t-delay]
    return bits, targets

def mc_gen(n_samples, seed=125):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, n_samples)

def run_simulation(hh, rho, bias, rates, task_cfg, common_cfg):
    dt = common_cfg['dt']
    sps = int(task_cfg['symbol_ms'] / dt)
    rates_up = np.repeat(rates, sps)
    
    # Poisson Spike Generation (Spatial)
    # Each symbol is encoded as Poisson trains for n_in channels. 
    # Here n_in = 1 for simple tasks.
    rng_in = np.random.default_rng(42)
    spikes_in = (rng_in.random(len(rates_up)) < (rates_up * dt * 1e-3)).astype(float)
    spikes_in = spikes_in[:, np.newaxis] # (T, 1)
    
    res = hh.simulate(rho, bias, spikes_in, "benchmark", trim_steps=0)
    # Features: Mean membrane trace per symbol (standard LSM extraction)
    phi = filter_and_downsample(res['spikes'], sps, dt, 10.0) # tau_trace=10ms
    return phi, res['mean_rate']

def main():
    print("--- STAGE D: RIGOROUS BENCHMARKING ---")
    with open('task_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup Trial
    seed_net = 42
    rng_mgr = RNGManager(seed_net)
    tg = rng_mgr.get_trial_generators(seed_net)
    st = rng_mgr.get_trial_seeds_tuple(seed_net)
    
    common = config['tasks']['common']
    n_total = common['n_train'] + common['n_test'] + common['warmup_symbols']
    
    results = []
    
    # Triplets Sweep
    for state_name, params in config['triplets'].items():
        print(f"\nEvaluating State: {state_name.upper()} (rho={float(params['rho'])})")
        
        # Initialize Model
        cfg = ExperimentConfig()
        cfg.hh = HHConfig(
            N=config['network']['N'],
            density=config['network']['sparsity'],
            in_density=config['network']['in_density'],
            in_gain=config['network']['in_gain'],
            gA=0.0 # Focusing on gA=0 first
        )
        hh = HHModel(cfg, tg, st)

        # Zadanie XOR Delay=1
        print(" -> XOR (d=1)...", end=" ", flush=True)
        xor1_cfg = config['tasks']['xor']
        u, y = xor_gen(n_total, delay=1)
        rates = np.where(u == 0, xor1_cfg['rate_0'], xor1_cfg['rate_1'])
        phi, fr = run_simulation(hh, params['rho'], params['bias'], rates, xor1_cfg, common)
        
        X_train = phi[common['warmup_symbols']:common['warmup_symbols']+common['n_train']]
        y_train = y[common['warmup_symbols']:common['warmup_symbols']+common['n_train']]
        X_test = phi[common['warmup_symbols']+common['n_train']:]
        y_test = y[common['warmup_symbols']+common['n_train']:]
        
        clf = Ridge(alpha=1.0).fit(X_train, y_train)
        acc1 = accuracy_score(y_test, clf.predict(X_test) > 0.5)
        print(f"FR={fr:.1f}Hz | Acc={acc1:.2%}")

        # Zadanie XOR Delay=2
        print(" -> XOR (d=2)...", end=" ", flush=True)
        u, y = xor_gen(n_total, delay=2)
        rates = np.where(u == 0, xor1_cfg['rate_0'], xor1_cfg['rate_1'])
        phi, _ = run_simulation(hh, params['rho'], params['bias'], rates, xor1_cfg, common)
        
        X_train = phi[common['warmup_symbols']:common['warmup_symbols']+common['n_train']]
        y_train = y[common['warmup_symbols']:common['warmup_symbols']+common['n_train']]
        X_test = phi[common['warmup_symbols']+common['n_train']:]
        y_test = y[common['warmup_symbols']+common['n_train']:]
        
        clf = Ridge(alpha=1.0).fit(X_train, y_train)
        acc2 = accuracy_score(y_test, clf.predict(X_test) > 0.5)
        print(f"Acc={acc2:.2%}")

        # Zadanie Memory Capacity
        print(" -> MC...", end=" ", flush=True)
        mc_cfg = config['tasks']['mc']
        u = mc_gen(n_total)
        rates = mc_cfg['rate_min'] + u * (mc_cfg['rate_max'] - mc_cfg['rate_min'])
        phi, _ = run_simulation(hh, params['rho'], params['bias'], rates, mc_cfg, common)
        
        X_train = phi[common['warmup_symbols']:common['warmup_symbols']+common['n_train']]
        X_test = phi[common['warmup_symbols']+common['n_train']:]
        
        mc_sum = 0
        for k in range(1, mc_cfg['max_lag']):
            y_train_k = u[common['warmup_symbols']-k : common['warmup_symbols']+common['n_train']-k]
            y_test_k = u[common['warmup_symbols']+common['n_train']-k : n_total-k]
            reg = Ridge(alpha=1.0).fit(X_train, y_train_k)
            mc_sum += max(0, reg.score(X_test, y_test_k))
        print(f"Result={mc_sum:.2f}")

        results.append({
            'state': state_name,
            'rho': params['rho'],
            'FR': fr,
            'XOR_d1': acc1,
            'XOR_d2': acc2,
            'MC': mc_sum
        })

    df = pd.DataFrame(results)
    print("\n--- FINAL REPORT ---")
    print(df)
    df.to_csv("summary_stage_D.csv", index=False)

if __name__ == "__main__":
    main()
