# D1 — Research Pack: HH-Rezerwuar i Edge of Chaos

## 1. Hodgkin-Huxley Reservoir Computing — Fundamenty

### 1.1 Model Hodgkina-Huxleya (HH)

Model HH opisuje generowanie potencjałów czynnościowych w neuronach za pomocą nieliniowych
równań różniczkowych zwyczajnych (ODE). Oryginalny model dotyczy aksonu olbrzymiego kałamarnicy.

**Równanie membranowe:**
```
C_m * dV/dt = -g_Na * m^3 * h * (V - E_Na) - g_K * n^4 * (V - E_K) - g_L * (V - E_L) + I_ext
```

**Zmienne bramkowe (gating variables):**
```
dm/dt = α_m(V) * (1-m) - β_m(V) * m
dh/dt = α_h(V) * (1-h) - β_h(V) * h
dn/dt = α_n(V) * (1-n) - β_n(V) * n
```

**Standardowe parametry (squid giant axon):**

| Parametr | Wartość | Jednostka | Źródło |
|----------|---------|-----------|--------|
| C_m | 1.0 | µF/cm² | Hodgkin & Huxley, 1952 [1] |
| g_Na | 120.0 | mS/cm² | [1] |
| g_K | 36.0 | mS/cm² | [1] |
| g_L | 0.3 | mS/cm² | [1] |
| E_Na | 50.0 | mV | [1] |
| E_K | -77.0 | mV | [1] |
| E_L | -54.4 | mV | [1] |
| V_rest | ≈ -65.0 | mV | [1] |

**Cytowanie:**
[1] Hodgkin, A.L. & Huxley, A.F. (1952). "A quantitative description of membrane current and
    its application to conduction and excitation in nerve." J. Physiol., 117(4), 500-544.

### 1.2 A-current (IA) — rozszerzenie Connor-Stevens

A-current to przejściowy prąd potasowy, który wpływa na adaptację częstotliwości wypalania
i umożliwia ekscytabilność Typu I (ciągły zakres firing rates od zera).

**Dodatkowy prąd:**
```
I_A = g_A * a^3 * b * (V - E_A)
da/dt = (a_inf(V) - a) / tau_a(V)
db/dt = (b_inf(V) - b) / tau_b(V)
```

**Parametry Connor-Stevens A-current:**

| Parametr | Wartość | Jednostka | Źródło |
|----------|---------|-----------|--------|
| g_A | 47.7 | mS/cm² | Connor & Stevens, 1971 [2] |
| E_A | -75.0 | mV | [2] |

**Cytowanie:**
[2] Connor, J.A. & Stevens, C.F. (1971). "Prediction of repetitive firing behaviour from
    voltage clamp data on an isolated neurone soma." J. Physiol., 213(1), 31-53.

### 1.3 Sieci HH jako rezerwuary

Sieci neuronów HH zostały użyte jako biologicznie wiarygodne rezerwuary w kontekście
Liquid State Machine (LSM) i Reservoir Computing (RC).

**Zasady architektoniczne:**
- Losowo połączona sieć rekurencyjna neuronów HH
- Wagi stałe (nie uczą się); uczony jest tylko readout
- Dynamika sieci projektuje sygnał wejściowy w wysokowymiarową przestrzeń stanów
- Readout (liniowy regressor) ekstrahuje informację ze stanów rezerwuaru

**Cytowania:**
[3] Maass, W., Natschläger, T. & Markram, H. (2002). "Real-time computing without stable
    states: A new framework for neural computation based on perturbations."
    Neural Computation, 14(11), 2531-2560.
[4] Maass, W., Natschläger, T. & Markram, H. (2004). "Computational Models for Generic
    Cortical Microcircuits." In Computational Neuroscience: A Comprehensive Approach. Chapman & Hall.

---

## 2. Benchmarki RC — Protokoły i Definicje

### 2.1 Memory Capacity (MC)

**Definicja:**
MC mierzy zdolność rezerwuaru do rekonstrukcji przeszłych wartości wejścia.

```
MC_k = [cov(u(t-k), y_k(t))]^2 / [var(u(t-k)) * var(y_k(t))]
MC_total = Σ_{k=1}^{K_max} MC_k
```

Gdzie u(t) to wejście, y_k(t) to wytrenowany output dla opóźnienia k.

**Protokół:**
- Wejście: i.i.d. uniform [-1, 1] lub [-0.5, 0.5]
- K_max: typowo N (rozmiar rezerwuaru) lub do MC_k < próg (np. 0.01)
- Readout: osobny regressor liniowy (ridge) dla każdego k
- Washout: 100-500 kroków (odrzucone z początku)
- Metryka: MC_total = Σ MC_k
- Podział: train/test z proporcją ok. 80/20
- Granica teoretyczna: MC ≤ N (dla sieci liniowych; Jaeger 2002)

**Cytowania:**
[5] Jaeger, H. (2001). "The 'echo state' approach to analysing and training recurrent neural
    networks." GMD Report 148, German National Research Center for Information Technology.
[6] Jaeger, H. (2002). "Short term memory in echo state networks." GMD Report 152.

### 2.2 NARMA-10

**Definicja:**
NARMA-10 to 10-rzędowy nieliniowy autoregresyjny model z ruchomą średnią.

```
y(t+1) = 0.3*y(t) + 0.05*y(t)*[Σ_{i=0}^{9} y(t-i)] + 1.5*u(t-9)*u(t) + 0.1
```

Gdzie u(t) ~ Uniform(0, 0.5).

**Protokół:**
- Wejście: u(t) ~ U(0, 0.5)
- Inicjalizacja y(0)...y(9) = 0
- Metryka: NRMSE = sqrt(Σ(y_pred - y_true)^2 / Σ(y_true - mean(y_true))^2)
  lub NMSE = MSE / var(y_true)
- Washout: 100-200 kroków
- Długość sekwencji: 3000-10000 kroków
- Podział: typowo 50%/50% lub 80%/20%
- Readout: Ridge regression

**UWAGA:** NARMA-10 zawiera region niestabilny; zakres wejścia musi być ograniczony.

**Cytowania:**
[7] Atiya, A.F. & Parlos, A.G. (2000). "New results on recurrent network training:
    Unifying the algorithms and accelerating convergence." IEEE Trans. Neural Networks, 11(3), 697-709.
[8] Rodan, A. & Tino, P. (2011). "Minimum complexity echo state network." IEEE Trans. Neural
    Networks, 22(1), 131-144.

### 2.3 Delayed XOR

**Definicja:**
Delayed XOR to zadanie klasyfikacyjne testujące nieliniowość i pamięć rezerwuaru.

```
target(t) = u(t) XOR u(t - τ)
```

Gdzie u(t) ∈ {0, 1} to losowe bity, τ to opóźnienie (typowo 3-10).

**Protokół:**
- Wejście: losowe bity 0/1 z równym prawdopodobieństwem
- Czas trwania symbolu: 10-50 ms (dla spiking RC)
- Metryka: dokładność klasyfikacji (accuracy) lub error rate
- Readout: logistyczna regresja lub liniowy SVM
- Washout: dostosowany do τ (min. 2*τ symboli)
- Podział: 70/30 lub 80/20

**Cytowania:**
[9] Bertschinger, N. & Natschläger, T. (2004). "Real-time computation at the edge of chaos
    in recurrent neural networks." Neural Computation, 16(7), 1413-1436.

---

## 3. Edge of Chaos w RC / Spiking RC

### 3.1 Definicja i teoria

**Edge of chaos** to krytyczny reżim dynamiczny na granicy między zachowaniem uporządkowanym
a chaotycznym. Systemy operujące w tym reżimie wykazują maksymalną zdolność obliczeniową.

- λ < 0: reżim stabilny — perturbacje zanikają, sieć "zapomina" szybko
- λ ≈ 0: edge of chaos — optymalna równowaga pamięci i nieliniowości
- λ > 0: reżim chaotyczny — perturbacje rosną eksponencjalnie, degradacja sygnału

**Kluczowe wyniki:**
- Bertschinger & Natschläger (2004): Sieci rekurencyjne osiągają maksymalną wydajność
  obliczeniową na granicy chaosu. Testy na sieciach threshold gates (model spiking) [9].
- Langton (1990): Oryginalna hipoteza "computation at the edge of chaos" [10].
- Legenstein & Maass (2007): "Edge of chaos and prediction of computational performance
  for neural circuit models" — formalna analiza dla sieci spiking [11].

**Cytowania:**
[9] Bertschinger & Natschläger (2004) — j.w.
[10] Langton, C.G. (1990). "Computation at the edge of chaos: Phase transitions and
     emergent computation." Physica D, 42(1-3), 12-37.
[11] Legenstein, R. & Maass, W. (2007). "Edge of chaos and prediction of computational
     performance for neural circuit models." Neural Networks, 20(3), 323-334.

### 3.2 Wykładnik Lyapunova — pomiar

**Maksymalny Wykładnik Lyapunova (MLE / λ_max):**
Mierzy szybkość eksponencjalnej dywergencji/konwergencji bliskich trajektorii.

**Metoda perturbacyjna (finite-difference) dla sieci spiking:**

1. Uruchom referencyjną symulację sieci
2. Wprowadź małą perturbację δ₀ do stanu (np. V jednego neuronu)
3. Symuluj obie sieci równolegle
4. Mierz odległość δ(t) = ||x_ref(t) - x_pert(t)||
5. Periodycznie renormalizuj: λ_i = ln(δ_i / δ₀), potem δ → δ₀ * δ̂
6. Uśredniaj: λ = (1/T) * Σ λ_i

**Parametry:**
- δ₀: 1e-7 do 1e-5 (mały, ale powyżej szumu numerycznego)
- Okres renormalizacji: 1-10 ms
- Okno pomiarowe: po washout 200-500 ms
- Czas pomiaru: 500-2000 ms
- Washout Lyapunova: oddzielny od washout benchmarku

**Pułapki:**
- Zbyt duże δ₀: nieliniowe efekty, przeszacowanie λ
- Zbyt małe δ₀: szum numeryczny dominuje
- Zbyt krótkie okno: niestabilne estymaty
- Brak washout: transjenty zanieczyszczają pomiar
- Zależność od dt: artefakt numeryczny, nie chaos biologiczny

**Cytowania:**
[12] Sprott, J.C. (2003). "Chaos and time-series analysis." Oxford University Press.
[13] Benettin, G., et al. (1980). "Lyapunov characteristic exponents for smooth dynamical
     systems and for Hamiltonian systems." Meccanica, 15(1), 9-20.
[14] Wolf, A., et al. (1985). "Determining Lyapunov exponents from a time series."
     Physica D, 16(3), 285-317.

### 3.3 Surrogate Metrics

Alternatywne metryki korelujące z edge of chaos (gdy λ jest kosztowne):
- **Separation Ratio / Fading Memory Ratio**: stosunek separacji par trajektorii
- **Firing Rate Variance**: nadmierna (chaos) vs zbyt niska (stabilność)
- **Entropy produkcji spike'ów**: maksymalna przy edge

**Cytowania:**
[15] Boedecker, J., et al. (2012). "Information processing in echo state networks at the
     edge of chaos." Theory in Biosciences, 131(3), 205-213.

---

## 4. Numeryka HH: RK4 i Stabilność

### 4.1 Runge-Kutta 4 (RK4)

RK4 to metoda jawna 4. rzędu do rozwiązywania ODE. Dla układu stiff (jakim jest HH),
krok czasowy musi być dostatecznie mały.

**Algorytm:**
```
k1 = f(t, y)
k2 = f(t + dt/2, y + dt/2 * k1)
k3 = f(t + dt/2, y + dt/2 * k2)
k4 = f(t + dt, y + dt * k3)
y(t+dt) = y(t) + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
```

### 4.2 Dobór dt

| dt (ms) | Stabilność | Dokładność | Wydajność | Źródło |
|---------|-----------|------------|-----------|--------|
| 0.001 | Doskonała | Doskonała | Niska | [16] |
| 0.005 | Bardzo dobra | Bardzo dobra | Średnia | Praktyka |
| 0.01 | Dobra | Dobra | Dobra | [16], [17] |
| 0.025 | Graniczna | Akceptowalna | Wysoka | [16] |
| 0.05 | Ryzykowna | Niska | Bardzo wysoka | — |
| > 0.05 | Niestabilna | Niedopuszczalna | — | [16] |

**Zalecenie:** dt = 0.01 ms jako domyślne; dt = 0.005 ms dla weryfikacji.
dt > 0.025 ms: ryzyko katastroficznej niestabilności podczas fazy narastania AP.

### 4.3 Sanity Checks

- **NaN/Inf check**: po każdym kroku RK4
- **V range check**: V ∈ [-100, +100] mV fizjologicznie; V > 200 mV = blow-up
- **Gating variables**: m, h, n ∈ [0, 1]; clipping z logowaniem
- **dt sensitivity test**: porównaj wyniki dla dt i dt/2; jeśli wyniki się znacząco różnią,
  chaos może być artefaktem numerycznym

**Cytowania:**
[16] Börgers, C. (2017). "An Introduction to Modeling Neuronal Dynamics." Springer.
[17] Dayan, P. & Abbott, L.F. (2001). "Theoretical Neuroscience." MIT Press.

---

## 5. Biologiczna Wierność

### 5.1 Dale's Law

**Zasada:** Każdy neuron uwalnia ten sam typ neurotransmiterów ze wszystkich swoich terminali
synaptycznych — jest albo excytatoryczny (E), albo inhibicyjny (I), nigdy oba.

**Implementacja:**
- 80% neuronów E (wagi wyjściowe > 0)
- 20% neuronów I (wagi wyjściowe < 0)
- Znaki wag stałe — nie zmieniają się w trakcie symulacji

**Cytowania:**
[18] Dale, H.H. (1935). "Pharmacology and nerve-endings." Proc. Royal Soc. Med., 28, 319-332.
[19] Eccles, J.C. (1976). "From electrical to chemical transmission in the central nervous
     system." Notes and Records of the Royal Society, 30(2), 219-230.

### 5.2 E/I Ratio

- Biologiczny stosunek w korze: ~80% E / 20% I
- Excytatoryczne: glutamate → AMPA/NMDA → E_syn ≈ 0 mV
- Inhibicyjne: GABA_A → E_syn ≈ -80 mV
- Balans E/I jest krytyczny dla dynamiki sieci
- Inhibicja powinna być silniejsza (typowo 4-5x wagi E) dla balansu

**Cytowania:**
[20] Brunel, N. (2000). "Dynamics of sparsely connected networks of excitatory and
     inhibitory spiking neurons." J. Comput. Neurosci., 8(3), 183-208.
[21] Vogels, T.P. & Abbott, L.F. (2005). "Signal propagation and logic gating in networks
     of integrate-and-fire neurons." J. Neurosci., 25(46), 10786-10795.

### 5.3 Synapsy

**Model synaptyczny (conductance-based):**
```
I_syn_i = Σ_j w_ij * g_syn * s_j(t) * (V_i - E_syn)
ds_j/dt = -s_j / tau_syn + Σ_spike δ(t - t_spike)
```

- E_syn_exc ≈ 0 mV (AMPA)
- E_syn_inh ≈ -80 mV (GABA_A)
- tau_syn_exc ≈ 5 ms (AMPA)
- tau_syn_inh ≈ 10 ms (GABA_A)

Alternatywa (prostsza): current-based synapses
```
I_syn_i = Σ_j w_ij * s_j(t)
```

**Cytowania:**
[22] Destexhe, A., Mainen, Z.F. & Sejnowski, T.J. (1994). "An efficient method for computing
     synaptic conductances based on a kinetic model of receptor binding."
     Neural Computation, 6(1), 14-18.

### 5.4 Poisson Input Encoding

Kodowanie sygnału ciągłego na spike train Poissonowski:
```
rate(t) = f_base + gain * signal(t)
P(spike in [t, t+dt]) = rate(t) * dt
```

- f_base: tło (np. 2-5 Hz)
- gain: wzmocnienie (np. 40-100 Hz na jednostkę)
- Niezależne generatory Poissona per neuron wejściowy
- Deterministyczny RNG z jawnym seed

**Cytowania:**
[23] Dayan, P. & Abbott, L.F. (2001). "Theoretical Neuroscience." — Rozdział o kodowaniu.

---

## 6. Readout / Klasyfikatory

### 6.1 Feature Extraction ze stanów HH

**Metody ekstrakcji cech z rezerwuaru spiking:**

| Metoda | Opis | Zastosowanie |
|--------|------|-------------|
| Filtered Firing Rate | Spike train → filtr wykładniczy z tau_readout | Standardowe |
| Mean Window | Średni firing rate w oknie czasowym | Proste |
| Last State | Ostatni stan (V lub s) | Kontrola |
| Concat-K | Konkatenacja K próbek z okna | Więcej informacji |
| Downsample | Próbkowanie co K kroków | Wydajność |

**Preferowana metoda: Filtered Firing Rate**
```
r_i(t) = r_i(t-1) * exp(-dt/tau_readout) + spike_i(t) / tau_readout
```

tau_readout: 10-50 ms (typowo 20 ms)

**Cytowania:**
[24] Lukoševičius, M. (2012). "A practical guide to applying echo state networks."
     In Neural Networks: Tricks of the Trade, Springer. pp. 659-686.
[25] Lukoševičius, M. & Jaeger, H. (2009). "Reservoir computing approaches to recurrent
     neural network training." Computer Science Review, 3(3), 127-149.

### 6.2 Modele Readout

| Model | Zastosowanie | Hiperparametry | Protokół |
|-------|-------------|----------------|----------|
| Ridge Regression | Regresja (MC, NARMA) | α ∈ logspace(-6, 6) | CV/walidacja |
| Logistic Regression | Klasyfikacja (XOR) | C ∈ logspace(-4, 4) | CV/walidacja |
| Linear SVM | Klasyfikacja (XOR) | C ∈ logspace(-4, 4) | CV/walidacja |

### 6.3 Protokół Anti-Leakage

1. Podział: train / val / test (np. 60/20/20 lub 70/15/15)
2. Scaler (StandardScaler): fit TYLKO na train, transform na val i test
3. Hyper-sweep: wybór najlepszego α/C po walidacji
4. Końcowy wynik: TYLKO na test
5. Raportuj: train score vs test score (gap = overfitting indicator)

**Cytowania:**
[24] Lukoševičius (2012) — j.w.
[26] Verstraeten, D., Schrauwen, B. & Stroobandt, D. (2006). "Reservoir-based techniques for
     speech recognition." Proc. Int. Joint Conf. Neural Networks (IJCNN), 1050-1053.

---

## 7. Chaos Drivers w sieciach HH i Plausibility

### 7.1 Źródła chaosu w sieciach HH

| Czynnik | Mechanizm | Źródło |
|---------|----------|--------|
| Silne sprzężenie | Nadmierna synchronizacja → destabilizacja | [20], [27] |
| Zbyt słaba inhibicja | Runaway excitation | [20] |
| Zbyt silna inhibicja | Tłumienie → rebound spiking → desynchronizacja | [21] |
| Heterogeniczność parametrów | Różne częstotliwości rezonansowe | [28] |
| Opóźnienia synaptyczne | Phase-dependent destabilizacja | [29] |
| Szum (stochastyczność) | Noise-induced transitions | [30] |

**Chaos biologiczny vs artefakt numeryczny:**

| Kryterium | Chaos biologiczny | Artefakt numeryczny |
|-----------|------------------|-------------------|
| Zależność od dt | Niezależny | Znika przy mniejszym dt |
| Zależność od integratora | Niezależny | Zmienia się (RK4 vs RK2) |
| Firing rates | Biologicznie plausible (1-100 Hz) | Niefizjologiczne (>500 Hz) |
| V range | -80 do +40 mV | V → ±∞ |
| Reprodukowalność | Deterministyczny z tym samym seed | Zależy od precyzji numerycznej |

**Cytowania:**
[27] Hansel, D. & Sompolinsky, H. (1992). "Synchronization and computation in a chaotic
     neural network." Phys. Rev. Lett., 68(5), 718-721.
[28] White, J.A., et al. (1998). "Networks of interneurons with fast and slow γ-aminobutyric
     acid type A (GABAA) kinetics provide substrate for mixed gamma-theta rhythm."
     PNAS, 97(14), 8128-8133.
[29] Ernst, U., Pawelzik, K. & Geisel, T. (1995). "Synchronization induced by temporal delays
     in pulse-coupled oscillators." Phys. Rev. Lett., 74(9), 1570-1573.
[30] Gammaitoni, L., et al. (1998). "Stochastic resonance." Rev. Mod. Phys., 70(1), 223-287.

### 7.2 Biologiczne Warunki Brzegowe (Plausibility Checks)

| Wskaźnik | Zakres biologiczny | Patologia |
|----------|-------------------|-----------|
| Mean Firing Rate | 1-50 Hz | < 0.1 Hz (dead) lub > 200 Hz (epileptiform) |
| CV of ISI | 0.5-1.5 (irregular) | < 0.1 (clock-like) lub > 3.0 (extreme burst) |
| % Active Neurons | > 20% | < 5% (dead network) |
| Synchrony Index | 0.05 - 0.5 | > 0.8 (hypersynchrony / epileptiform) |
| V range | [-80, +40] mV | V > 100 mV (blow-up) |
| Sustained Depolarization | < 10 ms | > 50 ms (patologiczne) |

**Cytowania:**
[20] Brunel (2000) — j.w.
[31] Softky, W.R. & Koch, C. (1993). "The highly irregular firing of cortical cells is
     inconsistent with temporal integration of random EPSPs." J. Neurosci., 13(1), 334-350.
[32] Kumar, A., Schrader, S., Aertsen, A. & Rotter, S. (2008). "The high-conductance state
     of cortical networks." Neural Computation, 20(1), 1-43.

---

## 8. Tabela Zbiorcza Parametrów

| Parametr | Zalecany zakres | Uzasadnienie | Źródło |
|----------|----------------|-------------|--------|
| N (rozmiar sieci) | 100-500 | Trade-off: wyrażalność vs koszt | [3], [5] |
| E/I ratio | 80/20 | Biologiczna proporcja korowa | [20], [21] |
| g_syn_exc | 0.01-0.5 nS | Zakres fizjologiczny AMPA | [22] |
| g_syn_inh | 0.04-2.0 nS | 4-5x silniejsza niż E | [20] |
| tau_syn_exc | 2-5 ms | Kinetyka AMPA | [22] |
| tau_syn_inh | 5-10 ms | Kinetyka GABA_A | [22] |
| Connection prob. | 0.1-0.3 | Sparse connectivity | [3], [20] |
| dt (RK4) | 0.005-0.01 ms | Stabilność HH stiff | [16], [17] |
| Washout (benchmark) | 200-500 ms | Eliminacja transientów | [5], [9] |
| Washout (Lyapunov) | 200-500 ms | j.w. | [12] |
| δ₀ (perturbacja λ) | 1e-7 - 1e-5 | Powyżej szumu, poniżej nieliniowości | [12], [14] |
| Renorm. period (λ) | 1-5 ms | Balance: resolution vs noise | [12] |
| tau_readout | 10-50 ms | Filtracja spike → rate | [24] |
| α (Ridge) | 1e-6 - 1e6 | Logspace sweep | [24] |
| Poisson base rate | 2-5 Hz | Tło aktywności | [23] |
| Poisson input gain | 20-100 Hz | Zakres kodowania | [23] |
| Symbol duration | 10-50 ms | Czas na symbol wejścia | [9] |
| inh_scaling range | 0.5-12.0 | Skanowanie reżimów | Projektowa decyzja |
| Seeds count | ≥ 30 | Statystyczna istotność | Praktyka statystyczna |

---

## Bibliografia kompletna

[1] Hodgkin, A.L. & Huxley, A.F. (1952). J. Physiol., 117(4), 500-544.
[2] Connor, J.A. & Stevens, C.F. (1971). J. Physiol., 213(1), 31-53.
[3] Maass, W., Natschläger, T. & Markram, H. (2002). Neural Computation, 14(11), 2531-2560.
[4] Maass, W., Natschläger, T. & Markram, H. (2004). Computational Neuroscience. Chapman & Hall.
[5] Jaeger, H. (2001). GMD Report 148.
[6] Jaeger, H. (2002). GMD Report 152.
[7] Atiya, A.F. & Parlos, A.G. (2000). IEEE Trans. Neural Networks, 11(3), 697-709.
[8] Rodan, A. & Tino, P. (2011). IEEE Trans. Neural Networks, 22(1), 131-144.
[9] Bertschinger, N. & Natschläger, T. (2004). Neural Computation, 16(7), 1413-1436.
[10] Langton, C.G. (1990). Physica D, 42(1-3), 12-37.
[11] Legenstein, R. & Maass, W. (2007). Neural Networks, 20(3), 323-334.
[12] Sprott, J.C. (2003). Chaos and time-series analysis. Oxford University Press.
[13] Benettin, G., et al. (1980). Meccanica, 15(1), 9-20.
[14] Wolf, A., et al. (1985). Physica D, 16(3), 285-317.
[15] Boedecker, J., et al. (2012). Theory in Biosciences, 131(3), 205-213.
[16] Börgers, C. (2017). An Introduction to Modeling Neuronal Dynamics. Springer.
[17] Dayan, P. & Abbott, L.F. (2001). Theoretical Neuroscience. MIT Press.
[18] Dale, H.H. (1935). Proc. Royal Soc. Med., 28, 319-332.
[19] Eccles, J.C. (1976). Notes and Records of the Royal Society, 30(2), 219-230.
[20] Brunel, N. (2000). J. Comput. Neurosci., 8(3), 183-208.
[21] Vogels, T.P. & Abbott, L.F. (2005). J. Neurosci., 25(46), 10786-10795.
[22] Destexhe, A., Mainen, Z.F. & Sejnowski, T.J. (1994). Neural Computation, 6(1), 14-18.
[23] Dayan, P. & Abbott, L.F. (2001). Theoretical Neuroscience. MIT Press.
[24] Lukoševičius, M. (2012). Neural Networks: Tricks of the Trade. Springer, 659-686.
[25] Lukoševičius, M. & Jaeger, H. (2009). Computer Science Review, 3(3), 127-149.
[26] Verstraeten, D., Schrauwen, B. & Stroobandt, D. (2006). Proc. IJCNN, 1050-1053.
[27] Hansel, D. & Sompolinsky, H. (1992). Phys. Rev. Lett., 68(5), 718-721.
[28] White, J.A., et al. (1998). PNAS, 97(14), 8128-8133.
[29] Ernst, U., Pawelzik, K. & Geisel, T. (1995). Phys. Rev. Lett., 74(9), 1570-1573.
[30] Gammaitoni, L., et al. (1998). Rev. Mod. Phys., 70(1), 223-287.
[31] Softky, W.R. & Koch, C. (1993). J. Neurosci., 13(1), 334-350.
[32] Kumar, A., Schrader, S., Aertsen, A. & Rotter, S. (2008). Neural Computation, 20(1), 1-43.
