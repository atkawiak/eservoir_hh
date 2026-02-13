# Hodgkin-Huxley Reservoir Computing Framework

Computational model for proving the **Edge of Chaos** hypothesis in biologically realistic neural networks.

## Hypothesis

Spiking reservoirs based on Hodgkin-Huxley (HH) neurons achieve maximal computational performance (Memory Capacity, NARMA-10, Delayed XOR) at the edge of chaos ($\lambda \approx 0$).

## Prerequisites

- Python 3.8+
- Requirements: `numpy`, `scipy`, `pandas`, `pyyaml`

## Structure

- `src/reservoir/`: Core neuron models (HH), synapses, encoding, and Lyapunov estimation.
- `src/benchmarks/`: RC benchmark protocols (MC, NARMA-10, Delayed XOR).
- `src/experiments/`: Pipeline for frozen topology experiments and statistical verification.
- `docs/`: Detailed research pack (D1) and design decisions (D2).

## Quick Start (Full Experiment)

To run the complete pipeline (N=100, 150, 200 across 30 seeds each):

```bash
python3 cli.py sweep --sizes 100 150 200
```

## Individual Steps

1. **Generate topologies:**

   ```bash
   python3 cli.py generate --seeds 30 --sizes 100
   ```

2. **Analysis for one seed:**

   ```bash
   python3 cli.py run-one --path frozen/N_100/seed_1000
   ```

3. **Aggregate statistics:**

   ```bash
   python3 cli.py stats --results_dir results
   ```

## Key Implementation Details

- **Neuron:** Hodgkin-Huxley + A-current option.
- **Rules:** Dale's law (80/20 E/I), Poisson encoding.
- **Stability:** RK4 integration ($dt=0.01$ ms) with NaN/blow-up health checks.
- **Edge Search:** Hybrid Dense Scan + Brentq Refinement.
- **Readout:** Ridge regression with filtered firing rate features and anti-leakage scaling.
- **Plausibility:** Biological boundary conditions for firing rates and synchrony.

## Deliverables

- `docs/D1_research_pack.md`: Literature review and benchmarks definitions.
- `docs/D2_design_decisions.md`: Multi-agent review and experiment plan.
- `final_stats.csv`: Statistical significance tests and performance aggregations.
