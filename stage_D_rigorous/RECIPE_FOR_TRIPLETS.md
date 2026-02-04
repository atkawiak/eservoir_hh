# Recipe for Obtaining the Edge of Chaos in HH Reservoirs

This document describes the rigorous procedure for identifying the Stable, Edge, and Chaotic states in a Hodgkin-Huxley Spiking Neural Network.

## 1. Network Configuration
| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| N | 100 | Sufficient dimensionality for standard benchmarks. |
| Recurrent Density | 0.2 | Maintains sparsity required for complex dynamics. |
| E/I Ratio | 80/20 | Biologically plausible (Dale's Law). |
| Inhibition Scale | 4.0 | Critical for E/I balance; prevents runaway excitation. |

## 2. Input Encoding (The "LSM Protocol")
| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| Symbol Duration | 50.0 ms | Matches HH integration constants (~10-20ms). |
| Active Rate | 45.0 Hz | High enough for signal, low enough to avoid saturation. |
| Background Rate| 2.0 Hz | Keeps ion channels active ("warm-up"). |
| Input Density | 0.3 | Spatial diversity; allows 70% of network to process freely. |
| Input Gain | 3.0 | Ensures input acts as a perturbation, not a driver. |

## 3. Identification Protocol
1. **Initialize** the network with the spatial $W_{in}$ mask.
2. **Sweep** the Spectral Radius ($\rho$) from 0.5 to 10.0.
3. **Measure Spike Divergence (Diff):** Simulate two identical networks, perturbing one by 0.1mV in the first step.
4. **Classify:**
   - **Stable:** $\rho$ where `Diff` returns to 0 within 1 symbol.
   - **Edge:** $\rho$ where `Diff` is maximized but firing rate remains balanced (20-40 Hz).
   - **Chaotic:** $\rho$ where `Diff` is large (>1000) and firing rates become erratic.

## 4. Benchmark Validation
Run the `run_rigorous_benchmarks.py` script. The Edge of Chaos is confirmed if:
- **Memory Capacity (MC)** is maximal at the Edge state.
- **XOR (Delay 2)** shows higher accuracy at the Edge vs Stable state.
