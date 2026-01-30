# Shriki Neuron & Edge of Chaos Exploration

Ten podprojekt bada hipotezę "Edge of Chaos" w sieciach neuronów Hodgkin-Huxley z prądem A (mechanizm Shriki).

## Podział Projektów

### Project A: Analog Current Injection
Klasyczne podejście, gdzie informacja wejściowa jest wtryskiwana jako ciągły analogowy prąd.
- **Lokalizacja:** `project_A_analog/`
- **Główne skrypty:** `run_henon_sweep.py`, `run_mc_sweep.py`.

### Project B: Poisson Spike Encoding
Podejście bardziej biologiczne, gdzie wejście jest kodowane jako ciągi impulsów (rate coding), a odczyt polega na filtrowaniu spike'ów rezerwuaru.
- **Lokalizacja:** `project_B_poisson/`
- **Główne skrypty:** `run_poisson_henon_sweep.py`, `run_poisson_mc_sweep.py`.
- **Weryfikacja:** `calc_lyapunov.py` (liczenie stabilności).

## Kluczowe Cechy (Wspólne)
- **Prawo Dale'a**: Podział na populacje ekscytacyjną (80%) i inhibicyjną (20%).
- **Detekcja zbocza**: Precyzyjna detekcja spike'ów (rising-edge).
- **Prądy konduktancyjne**: Realistyczna fizyka synaps.

## Jak uruchomić
Przejdź do folderu wybranego projektu i uruchom skrypty sweep z folderu `scripts/`.
Wyniki (wykresy i CSV) zostaną zapisane w folderze `results/`.
