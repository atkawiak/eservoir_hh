
import yaml
import logging
from src.reservoir.build_reservoir import build_reservoir
from src.benchmarks.mc import run_mc_benchmark

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def run():
    print("--- RUNNING MC (MAASS SPEC) ---")
    with open("configs/scientific/FULL_mc_maass.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    res = build_reservoir(cfg, seed=1000)
    results = run_mc_benchmark(res, cfg, inh_scaling=1.0, seed_input=1000)
    print(f"\nFINAL MC CAPACITY: {results['mc_total']:.4f}")

if __name__ == "__main__":
    run()
