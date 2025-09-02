# API Reference - Intermittent Demand Module

## Panoramica

Il modulo `intermittent_model` fornisce implementazioni specializzate per forecasting di domanda intermittente, tipica di spare parts, ricambi e prodotti a bassa rotazione.

## Classi Principali

### `IntermittentForecaster`

Classe principale per forecasting domanda intermittente.

```python
from arima_forecaster import IntermittentForecaster, IntermittentConfig, IntermittentMethod
```

#### Costruttore

```python
IntermittentForecaster(config: Optional[IntermittentConfig] = None)
```

**Parametri:**
- `config` (IntermittentConfig, optional): Configurazione del modello. Se None, usa configurazione default.

#### Metodi Principali

##### `fit(data: Union[pd.Series, np.ndarray]) -> IntermittentForecaster`

Addestra il modello sui dati storici.

**Parametri:**
- `data`: Serie storica con domanda (può contenere molti zeri)

**Returns:**
- Self per method chaining

**Esempio:**
```python
forecaster = IntermittentForecaster()
forecaster.fit(demand_history)
```

##### `forecast(steps: int = 1) -> np.ndarray`

Genera previsioni per periodi futuri.

**Parametri:**
- `steps`: Numero di periodi da prevedere

**Returns:**
- Array numpy con previsioni

**Esempio:**
```python
predictions = forecaster.forecast(steps=30)
```

##### `analyze_pattern(data: Union[pd.Series, np.ndarray]) -> IntermittentPattern`

Analizza il pattern di domanda per classificazione.

**Parametri:**
- `data`: Serie storica da analizzare

**Returns:**
- IntermittentPattern con metriche dettagliate

**Esempio:**
```python
pattern = forecaster.analyze_pattern(demand_data)
print(f"Classificazione: {pattern.classification}")
print(f"ADI: {pattern.adi:.2f}")
print(f"CV²: {pattern.cv2:.2f}")
```

##### `calculate_safety_stock(lead_time: int, service_level: float = 0.95) -> float`

Calcola il safety stock per il livello di servizio target.

**Parametri:**
- `lead_time`: Lead time del fornitore in periodi
- `service_level`: Livello di servizio target (0-1)

**Returns:**
- Quantità di safety stock suggerita

**Esempio:**
```python
safety_stock = forecaster.calculate_safety_stock(
    lead_time=15,
    service_level=0.95
)
```

##### `calculate_reorder_point(lead_time: int, service_level: float = 0.95) -> float`

Calcola il punto di riordino ottimale.

**Parametri:**
- `lead_time`: Lead time del fornitore
- `service_level`: Livello di servizio target

**Returns:**
- Quantità reorder point

**Esempio:**
```python
rop = forecaster.calculate_reorder_point(lead_time=15)
print(f"Riordina quando stock < {rop:.0f} unità")
```

##### `predict_intervals(steps: int = 1, confidence: float = 0.95) -> Dict[str, np.ndarray]`

Calcola intervalli di confidenza per le previsioni.

**Parametri:**
- `steps`: Numero di periodi
- `confidence`: Livello di confidenza (es. 0.95 per 95%)

**Returns:**
- Dizionario con 'forecast', 'lower', 'upper'

**Esempio:**
```python
intervals = forecaster.predict_intervals(steps=30, confidence=0.95)
print(f"Forecast: {intervals['forecast']}")
print(f"Lower bound: {intervals['lower']}")
print(f"Upper bound: {intervals['upper']}")
```

##### `get_metrics() -> Dict[str, Any]`

Ritorna metriche di performance e diagnostica.

**Returns:**
- Dizionario con metriche dettagliate

**Esempio:**
```python
metrics = forecaster.get_metrics()
print(f"Metodo: {metrics['method']}")
print(f"Alpha: {metrics['alpha']}")
print(f"Pattern: {metrics['pattern']['classification']}")
```

### `IntermittentConfig`

Configurazione per modelli Intermittent Demand.

```python
from arima_forecaster import IntermittentConfig, IntermittentMethod
```

#### Attributi

- `method` (IntermittentMethod): Metodo di forecasting
  - `CROSTON`: Croston's Method originale
  - `SBA`: Syntetos-Boylan Approximation
  - `TSB`: Teunter-Syntetos-Babai
  - `ADAPTIVE_CROSTON`: Croston con alpha adattivo
- `alpha` (float): Smoothing parameter (0-1), default 0.1
- `initial_level` (float, optional): Livello iniziale domanda
- `initial_interval` (float, optional): Intervallo iniziale
- `bias_correction` (bool): Applica correzione bias per SBA, default True
- `optimize_alpha` (bool): Ottimizza alpha automaticamente, default False
- `min_demand_periods` (int): Minimo periodi con domanda > 0, default 2

**Esempio:**
```python
config = IntermittentConfig(
    method=IntermittentMethod.SBA,
    alpha=0.15,
    optimize_alpha=True,
    bias_correction=True
)
```

### `IntermittentPattern`

Dataclass per analisi pattern domanda.

#### Attributi

- `adi` (float): Average Demand Interval
- `cv2` (float): Squared Coefficient of Variation
- `demand_periods` (int): Numero periodi con domanda > 0
- `zero_periods` (int): Numero periodi con domanda = 0
- `intermittence` (float): Percentuale periodi zero (0-1)
- `lumpiness` (float): Variabilità dimensione ordini
- `classification` (str): Classificazione pattern
  - "Smooth": Domanda regolare
  - "Intermittent": Sporadica ma stabile
  - "Erratic": Frequente ma variabile
  - "Lumpy": Sporadica e variabile

## Evaluator Specializzato

### `IntermittentEvaluator`

Valutatore specializzato per modelli Intermittent Demand.

```python
from arima_forecaster.evaluation import IntermittentEvaluator
```

#### Costruttore

```python
IntermittentEvaluator(holding_cost: float = 1.0, stockout_cost: float = 10.0)
```

**Parametri:**
- `holding_cost`: Costo unitario di giacenza per periodo
- `stockout_cost`: Costo unitario stockout per periodo

#### Metodi

##### `evaluate(actual, forecast, initial_stock=0) -> IntermittentMetrics`

Valuta performance del forecast.

**Parametri:**
- `actual`: Valori reali
- `forecast`: Valori previsti
- `initial_stock`: Stock iniziale per simulazione

**Returns:**
- IntermittentMetrics con tutte le metriche

**Esempio:**
```python
evaluator = IntermittentEvaluator(holding_cost=10, stockout_cost=100)
metrics = evaluator.evaluate(test_data, predictions)
print(f"MASE: {metrics.mase:.3f}")
print(f"Fill Rate: {metrics.fill_rate:.1f}%")
print(f"Service Level: {metrics.achieved_service_level:.1f}%")
```

##### `compare_methods(actual, forecasts) -> pd.DataFrame`

Confronta performance di metodi diversi.

**Parametri:**
- `actual`: Serie reale
- `forecasts`: Dict con nome_metodo -> forecast

**Returns:**
- DataFrame con confronto metriche

**Esempio:**
```python
forecasts = {
    'Croston': croston_forecast,
    'SBA': sba_forecast,
    'TSB': tsb_forecast
}
comparison = evaluator.compare_methods(actual_data, forecasts)
print(comparison)
```

### `IntermittentMetrics`

Dataclass contenente metriche di valutazione.

#### Attributi

- `mse`: Mean Squared Error
- `rmse`: Root Mean Squared Error
- `mae`: Mean Absolute Error
- `mase`: Mean Absolute Scaled Error (metrica preferita)
- `periods_in_stock`: % periodi con stock disponibile
- `fill_rate`: % domanda soddisfatta
- `bias`: Mean Error (positivo = overforecast)
- `pbias`: Percentage Bias
- `mae_demand`: MAE solo su periodi con domanda > 0
- `mape_demand`: MAPE solo su periodi con domanda > 0
- `achieved_service_level`: Livello servizio raggiunto
- `stockout_periods`: Numero periodi in stockout
- `overstock_periods`: Numero periodi con eccesso stock
- `total_holding_cost`: Costo totale giacenza (optional)
- `total_stockout_cost`: Costo totale stockout (optional)
- `total_cost`: Costo totale (optional)

## Esempi Completi

### Esempio 1: Analisi Base Ricambio

```python
from arima_forecaster import IntermittentForecaster, IntermittentConfig, IntermittentMethod
import numpy as np

# Genera dati esempio (domanda sporadica)
np.random.seed(42)
demand = np.random.choice([0, 0, 0, 1, 2, 3, 5], size=365, p=[0.7, 0.1, 0.05, 0.05, 0.05, 0.03, 0.02])

# Configura e addestra
config = IntermittentConfig(
    method=IntermittentMethod.SBA,
    optimize_alpha=True
)

forecaster = IntermittentForecaster(config)
forecaster.fit(demand[:300])

# Analizza pattern
pattern = forecaster.pattern_
print(f"Pattern: {pattern.classification}")
print(f"Intermittenza: {pattern.intermittence:.1%}")

# Forecast e inventory
forecast = forecaster.forecast(steps=30)
rop = forecaster.calculate_reorder_point(lead_time=15, service_level=0.95)

print(f"Forecast medio: {forecast.mean():.2f} unità/giorno")
print(f"Reorder Point: {rop:.0f} unità")
```

### Esempio 2: Confronto Metodi

```python
from arima_forecaster import IntermittentForecaster, IntermittentConfig, IntermittentMethod
from arima_forecaster.evaluation import IntermittentEvaluator

# Dati training e test
train_data = demand[:250]
test_data = demand[250:300]

# Test diversi metodi
methods = [IntermittentMethod.CROSTON, IntermittentMethod.SBA, IntermittentMethod.TSB]
results = {}

for method in methods:
    config = IntermittentConfig(method=method, optimize_alpha=True)
    model = IntermittentForecaster(config)
    model.fit(train_data)
    forecast = model.forecast(len(test_data))
    results[method.value] = forecast

# Valuta e confronta
evaluator = IntermittentEvaluator(holding_cost=5, stockout_cost=50)
comparison = evaluator.compare_methods(test_data, results)
print(comparison)

# Migliore metodo
best_method = comparison.iloc[0]['Method']
print(f"\nMigliore metodo: {best_method}")
```

### Esempio 3: Ottimizzazione Portfolio Ricambi

```python
import pandas as pd
from arima_forecaster import IntermittentForecaster, IntermittentConfig, IntermittentMethod

# Portfolio ricambi
spare_parts = {
    'SP001': {'history': demand1, 'cost': 45, 'lead_time': 15},
    'SP002': {'history': demand2, 'cost': 120, 'lead_time': 30},
    'SP003': {'history': demand3, 'cost': 85, 'lead_time': 20}
}

results = []

for code, data in spare_parts.items():
    # Analizza ogni ricambio
    config = IntermittentConfig(method=IntermittentMethod.SBA, optimize_alpha=True)
    model = IntermittentForecaster(config)
    model.fit(data['history'])
    
    # Calcola parametri inventory
    rop = model.calculate_reorder_point(
        lead_time=data['lead_time'],
        service_level=0.95
    )
    
    results.append({
        'Code': code,
        'Pattern': model.pattern_.classification,
        'ROP': rop,
        'Investment': rop * data['cost']
    })

# Summary
df_results = pd.DataFrame(results)
print(df_results)
print(f"\nInvestimento totale: €{df_results['Investment'].sum():,.2f}")
```

## Best Practices

### 1. Selezione del Metodo

```python
# Logica di selezione basata su pattern
pattern = forecaster.analyze_pattern(data)

if pattern.classification == 'Intermittent':
    method = IntermittentMethod.SBA  # Migliore per intermittent puro
elif pattern.classification == 'Lumpy':
    method = IntermittentMethod.TSB  # Migliore per lumpy
elif pattern.classification == 'Smooth':
    # Usa ARIMA tradizionale invece
    from arima_forecaster import ARIMAForecaster
    model = ARIMAForecaster()
else:
    method = IntermittentMethod.CROSTON  # Default
```

### 2. Ottimizzazione Alpha

```python
# Sempre consigliato per spare parts
config = IntermittentConfig(
    method=IntermittentMethod.SBA,
    optimize_alpha=True  # Trova alpha ottimale automaticamente
)
```

### 3. Validazione con Walk-Forward

```python
# Walk-forward validation per spare parts
window_size = 200
step_size = 20
errors = []

for i in range(0, len(data) - window_size - step_size, step_size):
    train = data[i:i+window_size]
    test = data[i+window_size:i+window_size+step_size]
    
    model = IntermittentForecaster()
    model.fit(train)
    forecast = model.forecast(step_size)
    
    error = np.mean(np.abs(test - forecast))
    errors.append(error)

print(f"MAE medio: {np.mean(errors):.3f}")
```

## Note Tecniche

### Complessità Computazionale

- **Croston/SBA/TSB**: O(n) dove n = lunghezza serie
- **Optimize Alpha**: O(n × m) dove m = numero valori alpha testati
- **Memory footprint**: Minimo, solo stati finali salvati

### Limitazioni

1. **No Trend**: I metodi assumono domanda stazionaria
2. **No Stagionalità**: Per pattern stagionali usa SARIMA
3. **Forecast Costante**: Le previsioni non variano nel tempo
4. **Dati Minimi**: Richiede almeno 2 periodi con domanda > 0

### Performance Attese

| Pattern | MASE Range | Metodo Consigliato |
|---------|------------|-------------------|
| Smooth | 0.6-0.9 | ARIMA/SARIMA |
| Intermittent | 0.8-1.2 | SBA |
| Erratic | 1.0-1.5 | Croston Adaptive |
| Lumpy | 1.2-2.0 | TSB |

## Integrazione con Pipeline

```python
# Pipeline completa spare parts
from arima_forecaster import IntermittentForecaster, TimeSeriesPreprocessor
from arima_forecaster.evaluation import IntermittentEvaluator
from arima_forecaster.visualization import ForecastPlotter

# 1. Preprocessing
preprocessor = TimeSeriesPreprocessor()
clean_data, metadata = preprocessor.preprocess_pipeline(
    raw_data,
    remove_outliers=False,  # Mantieni spike domanda
    handle_missing='forward_fill'
)

# 2. Forecasting
forecaster = IntermittentForecaster()
forecaster.fit(clean_data)

# 3. Evaluation
evaluator = IntermittentEvaluator()
metrics = evaluator.evaluate(test_data, forecast)

# 4. Visualization
plotter = ForecastPlotter()
plotter.plot_forecast(
    historical=clean_data,
    forecast=forecast,
    title=f"Spare Part Forecast - {metrics.mase:.2f} MASE"
)
```