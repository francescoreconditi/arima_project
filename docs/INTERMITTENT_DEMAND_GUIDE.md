# Guida Intermittent Demand Forecasting

## Panoramica

Il modulo **Intermittent Demand** gestisce prodotti con domanda sporadica tipici di:
- Ricambi industriali e automotive
- Spare parts aerospace  
- Farmaci specialistici
- Componenti elettronici B2B
- Prodotti luxury a bassa rotazione

## Metodi Implementati

### 1. **Croston's Method** (1972)
Metodo classico che separa:
- **Demand size**: dimensione ordine quando presente
- **Inter-arrival time**: tempo tra ordini
- Formula: `Forecast = Demand_Level / Interval_Level`

### 2. **SBA - Syntetos-Boylan Approximation** (2005)
Croston con correzione bias:
- Riduce sovrastima sistematica di Croston
- Fattore correzione: `(1 - α/2)`
- Più accurato per domanda molto intermittente

### 3. **TSB - Teunter-Syntetos-Babai** (2011)  
Approccio probability-based:
- Aggiorna probabilità domanda ad ogni periodo
- Non separa size/interval
- Formula: `Forecast = Demand_Level × Probability`

### 4. **Adaptive Croston**
Versione con alpha dinamico:
- Adatta smoothing parameter basandosi su errore
- Più reattivo a cambiamenti pattern

## Classificazione Pattern Domanda

Il sistema classifica automaticamente i prodotti:

| Pattern | ADI | CV² | Caratteristiche |
|---------|-----|-----|-----------------|
| **Smooth** | <1.32 | <0.49 | Domanda regolare |
| **Intermittent** | ≥1.32 | <0.49 | Sporadica ma stabile |
| **Erratic** | <1.32 | ≥0.49 | Frequente ma variabile |
| **Lumpy** | ≥1.32 | ≥0.49 | Sporadica e variabile |

- **ADI**: Average Demand Interval (giorni tra ordini)
- **CV²**: Squared Coefficient of Variation (variabilità)

## Utilizzo Base

```python
from arima_forecaster import IntermittentForecaster, IntermittentConfig, IntermittentMethod

# Configura modello
config = IntermittentConfig(
    method=IntermittentMethod.SBA,  # o CROSTON, TSB
    alpha=0.1,                       # Smoothing parameter
    optimize_alpha=True              # Auto-ottimizzazione
)

# Addestra su dati storici
forecaster = IntermittentForecaster(config)
forecaster.fit(demand_history)  # Array con molti zeri

# Analizza pattern
print(f"Pattern: {forecaster.pattern_.classification}")
print(f"Intermittenza: {forecaster.pattern_.intermittence:.1%}")

# Genera forecast
forecast = forecaster.forecast(steps=30)

# Calcola safety stock e reorder point
safety_stock = forecaster.calculate_safety_stock(
    lead_time=15,
    service_level=0.95
)
reorder_point = forecaster.calculate_reorder_point(
    lead_time=15,
    service_level=0.95
)
```

## Valutazione Performance

```python
from arima_forecaster.evaluation import IntermittentEvaluator

# Crea evaluator con costi
evaluator = IntermittentEvaluator(
    holding_cost=10,     # €/unità/periodo
    stockout_cost=100    # €/unità mancante
)

# Valuta modello
metrics = evaluator.evaluate(
    actual=test_data,
    forecast=predictions
)

print(f"MASE: {metrics.mase:.3f}")  # Metrica preferita
print(f"Fill Rate: {metrics.fill_rate:.1%}")
print(f"Service Level: {metrics.achieved_service_level:.1%}")
```

## Metriche Specifiche

### MASE (Mean Absolute Scaled Error)
- Metrica preferita per intermittent demand
- Confronta con forecast naive
- Valore < 1 = meglio di naive

### Fill Rate
- % domanda soddisfatta da stock
- Critico per customer satisfaction

### Service Level
- % periodi senza stockout
- Target tipico: 90-95%

## Esempio Completo Moretti

```python
# Analisi ricambi carrozzine
from examples.moretti import moretti_intermittent_spare_parts

# Test veloce
moretti_intermittent_spare_parts.test_veloce()

# Analisi portfolio completa
risultati = moretti_intermittent_spare_parts.esempio_portfolio_ricambi()
```

Output tipico:
```
ANALISI RICAMBIO: RC-W001 - Ruota carrozzina
Pattern: Intermittent
ADI: 12.3 giorni
Metodo ottimale: SBA
Reorder Point: 15 unità
Safety Stock: 11 unità
Investimento: €669
```

## Best Practices

### 1. Selezione Metodo
- **Croston**: Default per pattern Intermittent
- **SBA**: Preferito per ADI > 1.32
- **TSB**: Migliore per pattern Lumpy
- **Adaptive**: Quando pattern cambia nel tempo

### 2. Parametri Ottimali
- Alpha: 0.05-0.15 per spare parts stabili
- Alpha: 0.15-0.30 per domanda più variabile
- Sempre testare con `optimize_alpha=True`

### 3. Dati Minimi
- Almeno 2 periodi con domanda > 0
- Idealmente 12+ mesi storici
- Include tutti gli zeri (non filtrarli!)

### 4. Validazione
- Usa MASE come metrica primaria
- Monitora Fill Rate per service level
- Calcola costi totali (holding + stockout)

## Integrazione Sistema

Il modulo è completamente integrato:

```python
# Import diretto
from arima_forecaster import IntermittentForecaster

# Con altri modelli
from arima_forecaster import (
    ARIMAForecaster,        # Domanda regolare
    IntermittentForecaster  # Domanda sporadica
)

# Selezione automatica basata su pattern
if pattern.intermittence > 0.3:
    model = IntermittentForecaster()
else:
    model = ARIMAForecaster()
```

## Limitazioni Note

1. **Non gestisce trend**: Assume domanda stazionaria
2. **Non considera stagionalità**: Per seasonal, usa SARIMA
3. **Forecast costante**: Non varia nel tempo
4. **Richiede storico**: Non per cold-start puro

## Performance Attese

| Pattern | MASE Tipico | Note |
|---------|-------------|------|
| Smooth | 0.7-0.9 | ARIMA migliore |
| Intermittent | 0.8-1.2 | SBA ottimale |
| Erratic | 1.0-1.5 | Difficile |
| Lumpy | 1.2-2.0 | TSB preferito |

## Prossimi Sviluppi

- [ ] Metodi avanzati (ADIDA, IMAPA)
- [ ] Gestione obsolescenza
- [ ] Multi-echelon spare parts
- [ ] Integrazione con MRP/ERP
- [ ] Dashboard dedicata ricambi