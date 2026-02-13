"""
Main entry point for HH Reservoir Computing Framework.
"""

import argparse
import sys
import os
import yaml
import logging

def main():
    parser = argparse.ArgumentParser(description="HH Reservoir Computing Framework CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Command: generate
    parser_gen = subparsers.add_parser("generate", help="Generate and freeze reservoir topologies")
    parser_gen.add_argument("--config", type=str, default="configs/default.yaml")
    parser_gen.add_argument("--seeds", type=int, default=30)
    parser_gen.add_argument("--sizes", type=int, nargs="+", default=[100, 150, 200])

    # Command: run-one
    parser_one = subparsers.add_parser("run-one", help="Run simulation for a single seed")
    parser_one.add_argument("--path", type=str, required=True, help="Path to frozen reservoir")
    parser_one.add_argument("--config", type=str, default="configs/default.yaml")

    # Command: sweep
    parser_sweep = subparsers.add_parser("sweep", help="Run full sweep across sizes and seeds")
    parser_sweep.add_argument("--config", type=str, default="configs/default.yaml")
    parser_sweep.add_argument("--sizes", type=int, nargs="+", default=[100, 150, 200])

    # Command: stats
    parser_stats = subparsers.add_parser("stats", help="Aggregate results and compute statistics")
    parser_stats.add_argument("--results_dir", type=str, default="results")

    args = parser.parse_args()

    if args.command == "generate":
        from src.experiments.generate_reservoirs import main as gen_main
        sys.argv = ["generate_reservoirs.py", "--config", args.config, "--seeds_count", str(args.seeds), "--sizes"] + [str(s) for s in args.sizes]
        gen_main()

    elif args.command == "run-one":
        from src.experiments.run_one_seed import main as run_main
        sys.argv = ["run_one_seed.py", "--reservoir_path", args.path, "--config", args.config]
        run_main()

    elif args.command == "sweep":
        from src.experiments.sweep_sizes import main as sweep_main
        sys.argv = ["sweep_sizes.py", "--config", args.config, "--sizes"] + [str(s) for s in args.sizes]
        sweep_main()

    elif args.command == "stats":
        from src.experiments.aggregate_stats import main as stats_main
        sys.argv = ["aggregate_stats.py", "--results_dir", args.results_dir]
        stats_main()

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
