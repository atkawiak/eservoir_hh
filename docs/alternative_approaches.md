# Alternative Approaches to Prove Edge of Chaos Hypothesis

## Why current approach struggles:
1. XOR with D=5 may be too hard for network size (200-500)
2. Random weights are not normalized (no spectral radius control)
3. Lyapunov exponent (FTLE) may not capture the right dynamics for HH model

## Recommended alternatives:

### Option 1: Memory Capacity Test ⭐ (Most Standard)
**What:** Measure how many steps back the network can remember.

**Method:**
- Input: random signal u(t)
- Tasks: Predict u(t-1), u(t-2), ..., u(t-k)
- Metric: R² (coefficient of determination) for each delay k
- Memory Capacity (MC) = sum of all R² values above threshold (e.g., 0.5)

**Expected result:**
- Stable regime (w small): Low MC (forgets quickly)
- Edge (w ≈ 0.2): **Maximum MC** ← PROOF
- Chaos (w large): Low MC (noise destroys memory)

**References:** Jaeger (2001), Legenstein & Maass (2007)

---

### Option 2: Spectral Radius Normalization ⭐⭐
**What:** Properly control the "strength" of recurrent dynamics using eigenvalue scaling.

**Problem with current approach:**
We use raw random weights (w_scale * randn()). The actual "effective strength" depends on:
- Network size N
- Connectivity p
- Weight distribution

**Solution:**
1. Generate random weight matrix W
2. Compute spectral radius: ρ = max|eigenvalue(W)|
3. Normalize: W_target = (target_ρ / ρ) * W
4. Sweep target_ρ from 0.5 to 1.5

**Expected result:**
- ρ < 1: Stable (Echo State Property satisfied)
- ρ ≈ 1: **Edge of Chaos** ← Standard ESN definition
- ρ > 1: Chaos (unstable)

**This is THE standard method in Echo State Networks literature.**

---

### Option 3: Simpler Task
- Use XOR with D=1 or D=2 (much easier)
- Or even simpler: Echo task y(t) = u(t-1)

Edge effect should be more visible on simpler tasks.

---

## Recommendation:
**Implement Memory Capacity test with Spectral Radius control.**

This combination:
1. Is standard in RC literature (easy to compare)
2. Directly measures memory (the property we care about)
3. Uses proper weight normalization (fixes our current issue)
