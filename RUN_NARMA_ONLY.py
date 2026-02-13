
import yaml
import logging
from src.reservoir.build_reservoir import build_reservoir
from src.benchmarks.narma10 import run_narma10_benchmark

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def run():
    print("--- RUNNING NARMA-10 (LEGENSTEIN SPEC) ---")
    with open("configs/scientific/FULL_narma_legenstein.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    res = build_reservoir(cfg, seed=1000)
    results = run_narma10_benchmark(res, cfg, inh_scaling=1.0, seed_input=1000)
    print(f"\nFINAL NARMA-10 NRMSE: {results['nrmse']:.4f}")

if __name__ == "__main__":
    run()
