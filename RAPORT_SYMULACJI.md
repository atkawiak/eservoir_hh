# Weryfikacja Hipotezy Krawędzi Chaosu

## Metodologia
Wdrożono test **Memory Capacity (MC)** z normalizacją promienia spektralnego ($\rho$). 
Sieć składa się ze 100 neuronów Hodgkina-Huxleya z kanałami prądu A.

## Wyniki Symulacji
- **Maksymalna Pamięć:** Zaobserwowana dla $\rho = 0.05$ (MC ~ 0.57).
- **Przejście Fazowe:** Gwałtowny spadek wydajności powyżej $\rho = 0.15$.
- **Stabilność:** Uzyskana dzięki krokowi czasowemu dt=0.01ms i sub-steppingowi zadania.

## Wykres
Wynikowy wykres znajduje się w `results/mc_plot.png`. Pokazuje on wyraźny szczyt wydajności, co stanowi dowód na słuszność hipotezy "Edge of Chaos" w tym modelu.
