# HH Reservoir Computing - Docker Deployment Guide

## Przegląd

Ten folder zawiera kompletne środowisko Docker do uruchamiania eksperymentów Journal-Grade z sieciami Hodgkin-Huxley.

## Struktura Plików

```
project_C_poisson/
├── Dockerfile              # Definicja obrazu Docker
├── docker-compose.yml      # Konfiguracja z limitami zasobów (80% CPU/RAM)
├── requirements.txt        # Zależności Python
├── deploy.sh              # Automatyczny skrypt wdrożenia
├── validate_local.py      # Lokalna walidacja przed Dockerem
├── configs/
│   ├── validation_config.yaml    # 8 trials (ekstremalne przypadki)
│   └── production_config.yaml    # 1000 trials (pełny sweep)
├── src/                   # Kod źródłowy
├── results/               # Wyniki (montowane jako volume)
└── cache/                 # Cache symulacji (montowane jako volume)
```

## Workflow Wdrożenia

### 1. Lokalna Walidacja (OBOWIĄZKOWA)

Przed budowaniem Dockera, uruchom lokalną walidację:

```bash
python3 validate_local.py
```

To testuje 4 ekstremalne kombinacje parametrów:
- rho=0.01, bias=0.0 (minimalna aktywność)
- rho=1.5, bias=0.0 (maksymalna rekurencja)
- rho=0.01, bias=8.0 (maksymalny bias)
- rho=1.5, bias=8.0 (najbardziej ekstremalny)

**Jeśli którykolwiek test się nie powiedzie, NIE uruchamiaj Dockera!**

### 2. Automatyczne Wdrożenie

Po pomyślnej walidacji lokalnej:

```bash
./deploy.sh
```

Skrypt automatycznie:
1. Oblicza limity zasobów (80% CPU i RAM)
2. Buduje obraz Docker
3. Uruchamia walidację w Dockerze (8 trials)
4. Jeśli walidacja przejdzie, pyta o uruchomienie produkcji (1000 trials)

### 3. Ręczne Wdrożenie (Opcjonalne)

Jeśli wolisz kontrolować każdy krok:

```bash
# Budowanie obrazu
docker-compose build

# Walidacja (8 trials)
docker-compose run --rm hh_experiment python3 src/run_experiment.py --config configs/validation_config.yaml

# Produkcja (1000 trials) - tylko jeśli walidacja przeszła
docker-compose run --rm hh_experiment python3 src/run_experiment.py --config configs/production_config.yaml
```

## Konfiguracja Zasobów

Docker jest automatycznie ograniczony do:
- **80% dostępnych rdzeni CPU**
- **80% dostępnej pamięci RAM**

Limity są obliczane dynamicznie przez `deploy.sh` i zapisywane do `docker-compose.yml`.

## Monitorowanie

### Sprawdzanie Postępu

```bash
# Logi w czasie rzeczywistym
docker logs -f hh_reservoir_experiment

# Liczba wygenerowanych wyników
ls -lh results/*.parquet | wc -l

# Rozmiar cache
du -sh cache/
```

### Analiza Wyników

```bash
# Szybki podgląd
docker run --rm -v $(pwd)/results:/app/results hh_reservoir_experiment:latest \
    python3 -c "import pandas as pd; import glob; \
    df=pd.concat([pd.read_parquet(f) for f in glob.glob('/app/results/*.parquet')]); \
    print(df.groupby(['task', 'metric'])['value'].describe())"
```

## Parametry Eksperymentu

### Validation Config (validation_config.yaml)
- **Sieć:** N=50 neuronów
- **Sweep:** 2 rho × 2 bias × 2 seeds = **8 trials**
- **Czas:** ~5-10 minut
- **Cel:** Wykrycie błędów numerycznych

### Production Config (production_config.yaml)
- **Sieć:** N=100 neuronów
- **Sweep:** 10 rho × 5 bias × 20 seeds = **1000 trials**
- **Czas:** ~3-6 godzin (zależy od CPU)
- **Cel:** Pełne dane do publikacji

## Rozwiązywanie Problemów

### "Saturation Flag" w wynikach
- Normalne dla wysokich rho/bias
- Jeśli >50% trials ma saturację, rozważ zmniejszenie `dt` w config

### Brak plików w `results/`
- Sprawdź logi: `docker logs hh_reservoir_experiment`
- Upewnij się, że volume jest poprawnie zamontowany
- Sprawdź uprawnienia do zapisu

### Out of Memory
- Zmniejsz `seeds_coarse` w config
- Zwiększ `cv_gap` (mniej foldów CV)
- Uruchom w mniejszych partiach

## Wdrożenie na Serwer

### Rsync na retrai

```bash
# Z lokalnej maszyny
rsync -avz --exclude 'cache*' --exclude 'results*' --exclude '__pycache__' \
    project_C_poisson/ retrai:~/eservoir_hh/project_C_poisson/

# Na serwerze
ssh retrai
cd ~/eservoir_hh/project_C_poisson
./deploy.sh
```

### Uruchomienie w tle (nohup)

```bash
nohup ./deploy.sh > deployment.log 2>&1 &
tail -f deployment.log
```

## Wyniki

Pliki `.parquet` w `results/` zawierają:
- `rho`, `bias`, `seed_tuple_id` - parametry
- `task` - NARMA, XOR, MC, Lyapunov
- `metric` - nrmse, accuracy, auc, capacity, lambda_cond
- `value` - wartość metryki
- `baseline` - baseline dla porównania
- `improvement` - value - baseline
- `firing_rate`, `I_syn_mean`, `saturation_flag` - diagnostyka
- `timestamp`, `git_hash`, `sweep_mode` - metadane

## Bezpieczeństwo

✓ **Lokalna walidacja** przed pełnym sweepem
✓ **Limity zasobów** (80% CPU/RAM)
✓ **Izolacja Docker** - nie wpływa na system
✓ **Cache** - unika powtarzania symulacji
✓ **Deterministyczne RNG** - pełna reprodukowalność
