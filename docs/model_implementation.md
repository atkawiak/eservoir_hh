# Documentation: Implementing Shriki A-Current & Inhibition

## 1. Shriki-Huxley Neuron Model
We integrated the $I_A$ current based on Shriki et al. (2003) "Rate distributions and balance...".

### Equations
Added to standard Hodgkin-Huxley:
$$ C \frac{dV}{dt} = -I_{Na} - I_K - I_L - I_A + I_{inj} + I_{syn} $$

where:
$$ I_A = g_A a_\infty^3 b (V - E_A) $$
$$ \tau_A \frac{db}{dt} = b_\infty - b $$

Implementational details:
- `gA`: 20.0 mS/cm2 (default in paper)
- `tauA`: 20.0 ms
- `EA`: -80.0 mV

### Impact
- Linearizes the frequency-current (f-I) curve.
- Enables low-frequency firing (Type I dynamics approximation).
- Acts as a strong stabilizer for network activity.

## 2. Refactoring Inhibition (Stability Fix)
**Problem:** Initial simulations crashed at high weights ($w > 0.05$).
**Diagnosis:** The original implementation used "negative conductance" to represent inhibition ($I = -g \cdot (V - E_{exc})$). This is physically invalid and numerically unstable.
**Solution:**
Split synapses into Excitatory and Inhibitory pathways with distinct Reversal Potentials:
1. **Excitatory:** $E_{rev} = 0$ mV.
2. **Inhibitory:** $E_{rev} = -80$ mV.

$$ I_{syn} = g_{exc}(0 - V) + g_{inh}(-80 - V) $$

This guarantees that inhibition saturates at -80mV, preventing voltage divergences ("explosions").

**Result:**
The network became stable up to extremely high weights ($w=5.0$), identifying the "Ultra-Stable" regime caused by the combination of $I_A$ and strong physiological inhibition.
