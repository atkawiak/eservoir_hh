
import yaml
import logging
from src.reservoir.build_reservoir import build_reservoir
from src.benchmarks.delayed_xor import run_delayed_xor_benchmark

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def run():
    print("--- RUNNING DELAYED XOR (STANDARD SPEC) ---")
    with open("configs/scientific/FULL_xor_standard.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    res = build_reservoir(cfg, seed=1000)
    # Using specific delay 1 for test
    results = run_delayed_xor_benchmark(res, cfg, inh_scaling=1.0, seed_input=1000)
    print(f"\nFINAL XOR ACCURACY: {results['accuracy']:.4f}")

if __name__ == "__main__":
    run()
