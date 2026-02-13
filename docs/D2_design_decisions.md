# D2 — Design Decisions (Brainstorm Output)

## Multi-Agent Design Review — Final Decisions

---

## Phase 1 — Primary Designer: Specyfikacja Projektu

### 1. Operacyjna Definicja Reżimów

| Reżim | Definicja operacyjna | Kryterium selekcji |
|-------|---------------------|-------------------|
| **Stable** | λ < -0.5 (wyraźnie ujemny) | Punkt z dense scan: λ najbliższy -1.0, plausibility OK |
| **Edge** | \|λ\| < 0.1 (blisko zera) | Wynik hybrydy Dense+Brentq, plausibility OK |
| **Chaotic** | λ > 0.5 (wyraźnie dodatni) | Punkt z dense scan: λ najbliższy +1.0, plausibility OK |

**Uzasadnienie:** Progi ±0.5 zapewniają wyraźną separację reżimów. λ jest mierzone
w jednostkach 1/ms po washout w oknie ≥500 ms.

### 2. Estimator λ — strategia stabilizacji

**Metoda:**

1. Perturbacja δ₀ = 1e-6 (all V_i of all neurons shifted by δ₀)  
2. Renormalizacja co T_renorm = 2 ms
3. Okno pomiarowe: 1000 ms po 500 ms washout
4. **Powtórzenia:** 3 niezależne perturbacje (różne seed_lyapunov), uśrednianie median
5. **Test stabilności:** odchylenie < 0.3 między powtórzeniami; jeśli > 0.3, oznacz jako
   "unstable λ estimate"

**Uzasadnienie:**

- Wielokrotne powtórzenia redukują szum
- Median jest odporny na outliers
- Próg 0.3 to kompromis: zbyt niski = false rejects, zbyt wysoki = przepuszcza szum

### 3. Progi Plausibility

| Wskaźnik | Min | Max | Akcja na fail |
|----------|-----|-----|--------------|
| Mean Firing Rate (Hz) | 0.5 | 200 | REJECT |
| % Active Neurons | 10% | — | REJECT |
| Synchrony Index | — | 0.85 | WARN (nie reject) |
| Max V (mV) | — | 100 | REJECT (blow-up) |
| CV of ISI | 0.1 | 5.0 | WARN |
| NaN/Inf count | — | 0 | REJECT |

**Kalibracja progów:** Na podstawie rozkładów z pilotażowych 10 seedów dla N=100.
Jeśli > 50% seedów failuje, progi należy złagodzić.

### 4. Frozen Topology — szczegóły

- W_base generowane raz per seed_topology
- Struktura (maska połączeń), znaki (E/I assignment), wartości bazowe wag — zamrożone
- Skalowanie: W_inh = W_base_inh * inh_scaling
- W_exc = W_base_exc (nie skalowane)
- Zapisane do pliku: W_base (dense matrix), E/I masks, config

### 5. Kontrola wejść

Dla porównań stable/edge/chaos na tej samej topologii:

- Ten sam seed_input → identyczny spike train Poissona
- Ten sam seed_train_split → identyczny podział danych
- Różni się TYLKO inh_scaling

---

## Phase 2 — Skeptic Review

### S1. "Czy λ jest wiarygodne?"

**Objection:** λ mierzone perturbacyjnie w sieciach spiking może być niestabilne
ze względu na dyskretne zdarzenia spike'owe (spike/no-spike flip).

**Resolution:**

- 3 powtórzenia z medianą
- Test dt-sensitivity (dt vs dt/2)
- Odrzucenie punktów z niestabilnym λ
- **Decyzja:** ACCEPTED — dodano test dt-sensitivity jako obowiązkową kontrolę

### S2. "Czy 30 seedów wystarczy?"

**Objection:** Dla dużej wariancji między seedami, 30 może nie wystarczyć.

**Resolution:**

- 30 seedów to minimum (odpowiada CLT)
- Bootstrap CI (95%) do oceny precyzji
- Jeśli CI jest zbyt szerokie, zwiększ do 50 seedów
- **Decyzja:** ACCEPTED — 30 jako minimum, raportuj CI, opcja zwiększenia

### S3. "Czy V-readout nie jest artefaktem?"

**Objection:** Użycie V jako cechy readoutu daje artefaktyczne wyniki (V ≠ biologiczny output).

**Resolution:**

- V jest TYLKO kontrolą, nie głównym readoutem
- Główny readout: filtered firing rate
- Raportuj oba, ale wnioski wyciągaj z firing rate
- **Decyzja:** ACCEPTED — V jako kontrola, nie jako główny wynik

### S4. "Czy current-based vs conductance-based synapsy wpływa na wynik?"

**Objection:** Current-based synapsy są prostsze ale mniej biologiczne.

**Resolution:**

- Domyślnie: current-based (prostsza implementacja, szybsza)
- Ablacja: conductance-based jako wariant (opcjonalnie)
- **Decyzja:** ACCEPTED — current-based jako domyślne, conductance-based jako flaga

---

## Phase 2 — Constraint Guardian Review

### C1. Wydajność obliczeniowa

- N=100, dt=0.01 ms, 2000 ms symulacji = 200,000 kroków × 100 neuronów
- 30 seedów × 3 reżimy × 3 benchmarki + kalibracja (100 dense + brentq) =
  ~300 pełnych symulacji per N
- Szacowany czas: ~2-4h per N na jednym CPU
- **Decyzja:** Akceptowalne. Możliwość paralelizacji per seed.

### C2. Pamięć

- Macierz W: N×N float64 = 80 KB dla N=100
- Stany: N×4 float64 per krok (V, m, h, n) + synapsy
- Nie przechowujemy pełnej historii — tylko features po ekstrakcji
- **Decyzja:** Brak problemu pamięciowego

### C3. Determinizm

- Oddzielne RNG dla: topology, input, lyapunov, split
- np.random.default_rng(seed) — izolowane generatory
- **Decyzja:** ACCEPTED

---

## Phase 2 — User Advocate Review

### U1. "Jak łatwo jest dodać nowe benchmarki?"

**Resolution:** Abstrakcyjna klasa Benchmark z interfejsem generate/evaluate.
Łatwe rozszerzenie. **ACCEPTED.**

### U2. "Czy wyniki są łatwo interpretowalne?"

**Resolution:** CSV z per-run danymi + agregacje z p-values.
Opcjonalne wykresy. **ACCEPTED.**

---

## Phase 3 — Integrator/Arbiter Decision

**Disposition: APPROVED**

Wszystkie objections rozwiązane. Kluczowe decyzje:

1. λ stabilizowany przez 3 powtórzenia + median + dt-test
2. 30 seedów minimum z bootstrap CI
3. Filtered firing rate jako główny readout; V jako kontrola
4. Current-based synapsy domyślnie; conductance-based jako opcja
5. Progi plausibility z kalibracji pilotażowej

---

## Plan Statystyczny

### Testy porównawcze (stable vs edge vs chaos)

1. **Dla każdego N:** 30+ seedów, 3 reżimy per seed (paired design)
2. **Test główny:** Friedman test (non-parametric, paired, 3 grupy)
3. **Post-hoc:** Wilcoxon signed-rank test z korekcją Bonferroni (3 porównania)
4. **Efekt:** Cliff's delta (non-parametric effect size)
5. **CI:** Bootstrap 95% CI (10,000 prób)
6. **Raportowanie:**
   - Tabela: N, regime, mean±std, median, CI_low, CI_high
   - p-values Friedman i post-hoc
   - Cliff's delta per porównanie
   - % odrzuconych seedów i powody

### Hipoteza

- H0: Brak różnic w benchmarkach między reżimami
- H1: Edge > Stable AND Edge > Chaos (dla MC, NARMA, XOR)
- α = 0.05 po korekcji Bonferroni

---

## Audit Checklist

| # | Potencjalny artefakt | Kontrola | Status |
|---|---------------------|---------|--------|
| A1 | dt artefakt | Test dt vs dt/2 na 5 seedach | MUST |
| A2 | Brak washout | Washout 500 ms obowiązkowy | DONE by design |
| A3 | Readout leakage | Scaler fit only on train | DONE by design |
| A4 | V jako feature | V only as control, fire rate primary | DONE by design |
| A5 | Zbyt krótki Lyapunov | 1000 ms pomiar + 3 powtórzenia | DONE by design |
| A6 | Overfitting readoutu | Logspace hyper-sweep + val selection | DONE by design |
| A7 | Różne wejścia per reżim | Frozen input seed | DONE by design |
| A8 | Różna topologia per reżim | Frozen topology (W_base) | DONE by design |
| A9 | Hipersynchronia | Synchrony Index check | DONE by design |
| A10 | Numeryczny blow-up | V > 100 mV check + NaN check | DONE by design |
| A11 | Dead network | FR < 0.5 Hz check | DONE by design |

---

## Ablations Plan (WYŁĄCZNIE w HH)

| Ablacja | Co zmieniamy | Cecha |
|---------|-------------|-------|
| HH vs HH+A-current | g_A = 0 vs g_A = 47.7 | Wpływ A-current na edge |
| Current vs Conductance syn | Typ synapsy | Wpływ modelu synaptycznego |
| tau_readout | 10, 20, 50 ms | Wpływ filtracji |
| Feature type | fire_rate vs V vs concat | Wpływ ekstrakcji |
| Readout type | ridge vs logreg vs SVM | Wrażliwość na model |
| dt test | 0.01 vs 0.005 ms | Artefakt numeryczny? |
| E/I ratio | 80/20 vs 70/30 | Wrażliwość na balans |

---

## Experiment Matrix

```
for N in [100, 150, 200]:
    for seed_topo in range(30):
        generate_reservoir(N, seed_topo) → W_base
        calibrate_edge(W_base) → inh_scaling_edge, inh_scaling_stable, inh_scaling_chaos
        for regime in [stable, edge, chaos]:
            for benchmark in [MC, NARMA10, XOR]:
                run_benchmark(W_base, inh_scaling[regime], benchmark, seed_input=42)
                → results[N][seed_topo][regime][benchmark]
    aggregate_stats(results[N]) → means, stds, CIs, p_values
```
