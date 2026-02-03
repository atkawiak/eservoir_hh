#!/usr/bin/env python3
import sys, os
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import ExperimentConfig, HHConfig, TaskConfig
from rng_manager import RNGManager
from hh_model import HHModel
from utils import filter_and_downsample

# ============================================================
# RESEARCH ENGINEER PROTOCOL: SPARSE INPUT ENCODING
# ============================================================
N_NEURONS = 100
INPUT_SPARSITY = 0.2  # Tylko 20% neuronów dostaje wejście
IN_GAIN = 10.0
SYMBOL_MS = 50.0      # Zgodnie z Maass (2002)
N_TRAIN = 500
WAKEUP = 50

def run_sparse_xor(rho):
    seed = 42
    rng_mgr = RNGManager(2025)
    tg = rng_mgr.get_trial_generators(seed)
    st = rng_mgr.get_trial_seeds_tuple(seed)
    
    # Task data
    n_total = N_TRAIN + WAKEUP + 50
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, n_total)
    targets = np.zeros(n_total)
    for t in range(1, n_total): targets[t] = bits[t] ^ bits[t-1]
    
    # Input Encoding
    rates = np.where(bits == 0, 5.0, 80.0)
    cfg = ExperimentConfig()
    cfg.hh = HHConfig(N=N_NEURONS, gL=0.3, gA=0.0, in_gain=IN_GAIN)
    cfg.task = TaskConfig(dt=0.05, symbol_ms=SYMBOL_MS)
    sps = int(cfg.task.symbol_ms / cfg.task.dt)
    
    rates_up = np.repeat(rates, sps)
    
    # --- SPARSE INPUT MASKING ---
    mask = tg['in'].random(N_NEURONS) < INPUT_SPARSITY
    # Tylko wybrane neurony dostają Poisson spike train
    spikes_in_all = (tg['in'].random((len(rates_up), N_NEURONS)) < (rates_up[:, None] * cfg.task.dt * 1e-3)).astype(float)
    spikes_in_masked = spikes_in_all * mask[None, :]
    # Sumujemy wejście do jednego wektora, ale HHModel oczekuje 1D inputu? 
    # Muszę sprawdzić hh_model.py, jak obsługuje wejście. 
    # Jeśli obsługuje tylko 1D (wstrzykiwany prąd do wszystkich), muszę to poprawić w modelu.
    
    return spikes_in_masked, targets, sps, cfg, tg, st

def main():
    # Sprawdzam hh_model.py
    pass

if __name__ == "__main__":
    main()
