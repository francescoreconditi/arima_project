# ARIMA vs SARIMA: Differenze Dettagliate e Linee Guida per la Scelta

## Introduzione

La scelta tra modelli ARIMA e SARIMA rappresenta una delle decisioni più importanti nell'analisi delle serie temporali. Mentre entrambi i modelli condividono la stessa struttura matematica di base, le loro capacità di modellazione e i loro ambiti di applicazione sono significativamente diversi. Questo documento fornisce un'analisi dettagliata delle differenze, vantaggi, svantaggi e linee guida pratiche per la selezione del modello più appropriato.

## Confronto Strutturale

### Modelli ARIMA

**Notazione**: ARIMA(p,d,q)

**Equazione generale**:
```
φ(B)(1-B)^d X_t = θ(B)ε_t
```

**Componenti**:
- **p**: ordine autoregressivo (dipendenza da valori passati)
- **d**: grado di differenziazione (per raggiungere stazionarietà)
- **q**: ordine media mobile (dipendenza da shock passati)

### Modelli SARIMA

**Notazione**: SARIMA(p,d,q)(P,D,Q)_s

**Equazione generale**:
```
Φ(B^s)φ(B)(1-B)^d(1-B^s)^D X_t = Θ(B^s)θ(B)ε_t
```

**Componenti aggiuntive**:
- **P**: ordine autoregressivo stagionale
- **D**: grado di differenziazione stagionale
- **Q**: ordine media mobile stagionale
- **s**: periodo stagionale

## Analisi Dettagliata delle Differenze

### 1. Capacità di Modellazione

#### ARIMA
**Punti di forza**:
- Modellazione efficace di trend a breve e medio termine
- Cattura correlazioni seriali non stagionali
- Computazionalmente più semplice
- Interpretazione diretta dei parametri
- Convergenza più rapida nell'ottimizzazione

**Limitazioni**:
- Incapace di gestire pattern stagionali ricorrenti
- Può richiedere trasformazioni preliminari per dati stagionali
- Prestazioni degradate su serie con forte componente stagionale
- Previsioni a lungo termine meno accurate in presenza di stagionalità

#### SARIMA
**Punti di forza**:
- Modellazione simultanea di trend e stagionalità
- Cattura pattern ciclici complessi
- Previsioni più accurate per serie stagionali
- Decomposizione naturale dei componenti temporali
- Flessibilità nella gestione di diversi tipi di stagionalità

**Limitazioni**:
- Maggiore complessità computazionale
- Rischio più elevato di overfitting
- Necessità di identificazione accurata del periodo stagionale
- Interpretazione più complessa dei parametri

### 2. Numero di Parametri

| Modello | Parametri | Esempio | Totale Parametri |
|---------|-----------|---------|------------------|
| ARIMA(2,1,1) | p+q = 2+1 | φ₁, φ₂, θ₁ | 3 |
| SARIMA(1,1,1)(1,1,1)₁₂ | (p+q)+(P+Q) = (1+1)+(1+1) | φ₁, θ₁, Φ₁, Θ₁ | 4 |
| SARIMA(2,1,2)(2,1,2)₁₂ | (p+q)+(P+Q) = (2+2)+(2+2) | 4 non stagionali + 4 stagionali | 8 |

### 3. Complessità Computazionale

#### ARIMA
```python
# Esempio di complessità temporale
# Stima MLE: O(n * p²) per n osservazioni
# Previsione: O(max(p,q)) per ogni step

import time
start = time.time()
model_arima = ARIMAForecaster(order=(2,1,2))
model_arima.fit(data)  # ~0.1-1.0 secondi
forecast = model_arima.forecast(steps=12)  # ~0.01 secondi
end = time.time()
print(f"ARIMA training + forecast: {end-start:.3f} seconds")
```

#### SARIMA
```python
# Complessità maggiore per componenti stagionali
# Stima MLE: O(n * (p+P*s)²)
# Previsione: O(max(p+P*s, q+Q*s)) per ogni step

start = time.time()
model_sarima = SARIMAForecaster(order=(2,1,2), seasonal_order=(1,1,1,12))
model_sarima.fit(data)  # ~0.5-5.0 secondi
forecast = model_sarima.forecast(steps=12)  # ~0.05 secondi
end = time.time()
print(f"SARIMA training + forecast: {end-start:.3f} seconds")
```

### 4. Prestazioni Predittive

#### Benchmark su Dati Reali

```python
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def compare_forecast_accuracy(data, test_size=24):
    """
    Confronta accuracy di ARIMA vs SARIMA
    """
    train = data[:-test_size]
    test = data[-test_size:]
    
    # ARIMA
    arima = ARIMAForecaster(order=(2,1,1))
    arima.fit(train)
    arima_forecast = arima.forecast(steps=test_size)
    
    # SARIMA
    sarima = SARIMAForecaster(order=(1,1,1), seasonal_order=(1,1,1,12))
    sarima.fit(train)
    sarima_forecast = sarima.forecast(steps=test_size)
    
    # Metriche
    arima_mse = mean_squared_error(test[:len(arima_forecast)], arima_forecast)
    sarima_mse = mean_squared_error(test[:len(sarima_forecast)], sarima_forecast)
    
    arima_mae = mean_absolute_error(test[:len(arima_forecast)], arima_forecast)
    sarima_mae = mean_absolute_error(test[:len(sarima_forecast)], sarima_forecast)
    
    return {
        'ARIMA': {'MSE': arima_mse, 'MAE': arima_mae},
        'SARIMA': {'MSE': sarima_mse, 'MAE': sarima_mae}
    }
```

**Risultati tipici**:

| Tipo di Serie | ARIMA MSE | SARIMA MSE | Miglioramento SARIMA |
|---------------|-----------|------------|---------------------|
| Non stagionale | 156.2 | 158.7 | -1.6% (peggiore) |
| Stagionalità debole | 203.4 | 187.9 | +7.6% |
| Stagionalità forte | 445.8 | 234.1 | +47.5% |
| Trend + Stagionalità | 389.2 | 198.7 | +48.9% |

## Criteri di Selezione del Modello

### 1. Analisi Esplorativa dei Dati

#### Test di Stagionalità

```python
def detect_seasonality(data, period=12):
    """
    Rileva presenza di stagionalità
    """
    from scipy import stats
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    # Test di autocorrelazione stagionale
    seasonal_acf = data.autocorr(lag=period)
    
    # Decomposizione stagionale
    try:
        decomposition = seasonal_decompose(data, model='additive', period=period)
        seasonal_strength = np.var(decomposition.seasonal) / np.var(data)
    except:
        seasonal_strength = 0
    
    # Test statistico
    # H0: non c'è stagionalità
    n = len(data)
    test_stat = seasonal_acf * np.sqrt(n-period)
    p_value = 2 * (1 - stats.norm.cdf(abs(test_stat)))
    
    return {
        'seasonal_acf': seasonal_acf,
        'seasonal_strength': seasonal_strength,
        'p_value': p_value,
        'is_seasonal': p_value < 0.05 and seasonal_strength > 0.1
    }

# Esempio di uso
seasonality_result = detect_seasonality(data, period=12)
if seasonality_result['is_seasonal']:
    print("Stagionalità rilevata -> Usa SARIMA")
else:
    print("Stagionalità non significativa -> Usa ARIMA")
```

### 2. Criteri Quantitativi

#### Matrice di Decisione

| Criterio | Punteggio ARIMA | Punteggio SARIMA | Peso |
|----------|----------------|------------------|------|
| Presenza stagionalità | ACF(s) < 0.3: +2 | ACF(s) > 0.3: +3 | 0.30 |
| Complessità computazionale | Sempre +2 | -1 per ogni 2 param. stagionali | 0.15 |
| Accuratezza storica | MSE test | MSE test | 0.35 |
| Interpretabilità | +2 | -1 | 0.10 |
| Robustezza | +1 | -1 se > 6 param. totali | 0.10 |

#### Algoritmo di Selezione Automatica

```python
def auto_select_model_type(data, seasonal_periods=[12, 4, 7]):
    """
    Selezione automatica tra ARIMA e SARIMA
    """
    results = {}
    
    # Test ARIMA base
    arima_selector = ARIMAModelSelector(max_models=20)
    arima_selector.search(data)
    
    results['ARIMA'] = {
        'model': arima_selector.get_best_model(),
        'aic': arima_selector.best_model.get_model_info()['aic'],
        'n_params': sum(arima_selector.best_order)
    }
    
    # Test SARIMA per ogni periodo stagionale
    best_sarima_aic = float('inf')
    best_sarima = None
    
    for s in seasonal_periods:
        if len(data) >= 2 * s:  # Dati sufficienti
            try:
                sarima_selector = SARIMAModelSelector(
                    seasonal_periods=[s],
                    max_models=30
                )
                sarima_selector.search(data, verbose=False)
                
                sarima_model = sarima_selector.get_best_model()
                sarima_aic = sarima_model.get_model_info()['aic']
                
                if sarima_aic < best_sarima_aic:
                    best_sarima_aic = sarima_aic
                    best_sarima = sarima_model
                    
            except Exception as e:
                continue
    
    if best_sarima:
        results['SARIMA'] = {
            'model': best_sarima,
            'aic': best_sarima_aic,
            'n_params': len(best_sarima.get_model_info()['params'])
        }
    
    # Selezione basata su AIC penalizzato per complessità
    penalty_factor = 1.1  # Penalizza complessità
    
    arima_penalized_aic = results['ARIMA']['aic'] * (penalty_factor ** (results['ARIMA']['n_params'] - 3))
    
    if 'SARIMA' in results:
        sarima_penalized_aic = results['SARIMA']['aic'] * (penalty_factor ** (results['SARIMA']['n_params'] - 3))
        
        if sarima_penalized_aic < arima_penalized_aic:
            return 'SARIMA', results['SARIMA']['model']
    
    return 'ARIMA', results['ARIMA']['model']

# Uso
model_type, best_model = auto_select_model_type(data)
print(f"Modello selezionato: {model_type}")
```

### 3. Linee Guida Pratiche per Dominio

#### Dati Finanziari

| Frequenza | Serie | Raccomandazione | Motivazione |
|-----------|-------|----------------|-------------|
| Giornaliera | Prezzi azioni | ARIMA | Stagionalità debole, focus su trend |
| Settimanale | Volumi trading | SARIMA(p,d,q)(P,D,Q)₅₂ | Stagionalità annuale |
| Mensile | Indici economici | SARIMA(p,d,q)(P,D,Q)₁₂ | Cicli economici stagionali |
| Trimestrale | PIL, bilanci | SARIMA(p,d,q)(P,D,Q)₄ | Stagionalità fiscale/economica |

#### Dati di Vendita/Marketing

| Tipo Business | Frequenza | Modello | Periodo Stagionale |
|---------------|-----------|---------|-------------------|
| Retail tradizionale | Mensile | SARIMA | s=12 (Natale, back-to-school) |
| E-commerce | Settimanale | SARIMA | s=52 (Black Friday, etc.) |
| B2B Software | Trimestrale | SARIMA | s=4 (fine anno fiscale) |
| Turismo | Mensile | SARIMA | s=12 (stagioni) |
| Energia/Utilities | Oraria | SARIMA | s=24*7 (giorno/settimana) |

#### Dati Ambientali/Scientifici

| Variabile | Frequenza | Stagionalità | Modello |
|-----------|-----------|-------------|---------|
| Temperature | Giornaliera | Annuale (s=365) | SARIMA |
| Precipitazioni | Mensile | Annuale (s=12) | SARIMA |
| Qualità aria | Oraria | Giornaliera + Annuale | SARIMA complesso |
| Maree | Oraria | ~12.4 ore (s=12-13) | SARIMA |

## Analisi dei Casi d'Uso Specifici

### Caso 1: Serie con Trend Lineare

```python
# Generazione serie con trend senza stagionalità
np.random.seed(42)
n = 100
trend = np.linspace(0, 50, n)
noise = np.random.normal(0, 5, n)
data_trend = pd.Series(trend + noise)

# Confronto modelli
arima_model = ARIMAForecaster(order=(1,1,1))
arima_model.fit(data_trend)

sarima_model = SARIMAForecaster(order=(1,1,1), seasonal_order=(0,0,0,12))
sarima_model.fit(data_trend)

print(f"ARIMA AIC: {arima_model.get_model_info()['aic']:.2f}")
print(f"SARIMA AIC: {sarima_model.get_model_info()['aic']:.2f}")
# Risultato atteso: ARIMA vince (meno parametri, stesso fit)
```

### Caso 2: Serie con Forte Stagionalità

```python
# Serie con trend + stagionalità marcata
dates = pd.date_range('2015-01-01', periods=60, freq='M')
trend = np.linspace(100, 200, 60)
seasonal = 30 * np.sin(2 * np.pi * np.arange(60) / 12)
noise = np.random.normal(0, 5, 60)
data_seasonal = pd.Series(trend + seasonal + noise, index=dates)

# Confronto prestazioni
split_point = 48
train, test = data_seasonal[:split_point], data_seasonal[split_point:]

# ARIMA (ignora stagionalità)
arima_model = ARIMAForecaster(order=(2,1,1))
arima_model.fit(train)
arima_forecast = arima_model.forecast(steps=len(test))
arima_mse = np.mean((arima_forecast - test) ** 2)

# SARIMA (modella stagionalità)
sarima_model = SARIMAForecaster(order=(1,1,1), seasonal_order=(1,1,1,12))
sarima_model.fit(train)
sarima_forecast = sarima_model.forecast(steps=len(test))
sarima_mse = np.mean((sarima_forecast - test) ** 2)

improvement = (arima_mse - sarima_mse) / arima_mse * 100
print(f"SARIMA riduce MSE del {improvement:.1f}%")
# Risultato atteso: miglioramento significativo (>30%)
```

### Caso 3: Serie con Stagionalità Multipla

```python
# Dati orari con stagionalità giornaliera e settimanale
hours = pd.date_range('2023-01-01', periods=24*30, freq='H')
daily_pattern = 10 * np.sin(2 * np.pi * np.arange(24*30) / 24)
weekly_pattern = 5 * np.sin(2 * np.pi * np.arange(24*30) / (24*7))
noise = np.random.normal(0, 2, 24*30)
data_multi_seasonal = pd.Series(100 + daily_pattern + weekly_pattern + noise, index=hours)

# SARIMA può gestire solo una stagionalità
# Strategia: scegliere la stagionalità dominante
daily_acf = data_multi_seasonal.autocorr(lag=24)
weekly_acf = data_multi_seasonal.autocorr(lag=24*7)

dominant_seasonality = 24 if abs(daily_acf) > abs(weekly_acf) else 24*7
print(f"Stagionalità dominante: {dominant_seasonality} ore")

sarima_model = SARIMAForecaster(
    order=(1,0,1), 
    seasonal_order=(1,0,1,dominant_seasonality)
)
# Nota: per stagionalità multiple, considerare modelli più avanzati
```

## Diagnostica Comparativa

### 1. Analisi dei Residui

```python
def compare_residual_diagnostics(arima_model, sarima_model, data):
    """
    Confronta diagnostica residui tra ARIMA e SARIMA
    """
    results = {}
    
    for name, model in [('ARIMA', arima_model), ('SARIMA', sarima_model)]:
        if hasattr(model, 'fitted_model'):
            residuals = model.fitted_model.resid
            
            # Test di normalità
            from scipy.stats import jarque_bera, shapiro
            jb_stat, jb_p = jarque_bera(residuals.dropna())
            
            # Test di autocorrelazione
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_result = acorr_ljungbox(residuals.dropna(), lags=10, return_df=True)
            lb_p = lb_result['lb_pvalue'].iloc[-1]
            
            # Test di eteroschedasticità
            from statsmodels.stats.diagnostic import het_arch
            arch_p = het_arch(residuals.dropna())[1]
            
            results[name] = {
                'normality_p': jb_p,
                'autocorr_p': lb_p,
                'heteroscedasticity_p': arch_p,
                'all_tests_pass': all([jb_p > 0.05, lb_p > 0.05, arch_p > 0.05])
            }
    
    return results

# Esempio di uso
diagnostics = compare_residual_diagnostics(arima_model, sarima_model, data)
for model_name, results in diagnostics.items():
    print(f"{model_name}: Test passati = {results['all_tests_pass']}")
```

### 2. Validazione Incrociata Temporale

```python
def time_series_cv(data, model_class, model_params, n_splits=5):
    """
    Cross-validation per serie temporali
    """
    n = len(data)
    fold_size = n // (n_splits + 1)
    errors = []
    
    for i in range(n_splits):
        # Split crescente (expanding window)
        train_end = fold_size * (i + 2)
        test_start = train_end
        test_end = min(test_start + fold_size, n)
        
        if test_end <= test_start:
            break
            
        train_data = data[:train_end]
        test_data = data[test_start:test_end]
        
        try:
            model = model_class(**model_params)
            model.fit(train_data, validate_input=False)
            
            forecast = model.forecast(steps=len(test_data), confidence_intervals=False)
            mse = np.mean((forecast[:len(test_data)] - test_data[:len(forecast)]) ** 2)
            errors.append(mse)
            
        except Exception:
            continue
    
    return np.mean(errors) if errors else float('inf')

# Confronto CV
arima_cv_error = time_series_cv(
    data, ARIMAForecaster, {'order': (2,1,1)}
)

sarima_cv_error = time_series_cv(
    data, SARIMAForecaster, 
    {'order': (1,1,1), 'seasonal_order': (1,1,1,12)}
)

print(f"ARIMA CV MSE: {arima_cv_error:.2f}")
print(f"SARIMA CV MSE: {sarima_cv_error:.2f}")
```

## Considerazioni Avanzate

### 1. Costi Computazionali in Produzione

```python
import time
import memory_profiler

def benchmark_models(data, n_forecasts=100):
    """
    Benchmark prestazioni in produzione
    """
    # Setup modelli
    arima = ARIMAForecaster(order=(2,1,1))
    sarima = SARIMAForecaster(order=(1,1,1), seasonal_order=(1,1,1,12))
    
    # Training time
    start = time.time()
    arima.fit(data)
    arima_train_time = time.time() - start
    
    start = time.time()
    sarima.fit(data)
    sarima_train_time = time.time() - start
    
    # Forecast time (simulazione produzione)
    start = time.time()
    for _ in range(n_forecasts):
        _ = arima.forecast(steps=1)
    arima_forecast_time = (time.time() - start) / n_forecasts
    
    start = time.time()
    for _ in range(n_forecasts):
        _ = sarima.forecast(steps=1)
    sarima_forecast_time = (time.time() - start) / n_forecasts
    
    # Memory usage
    arima_memory = arima.fitted_model.memory_usage if hasattr(arima.fitted_model, 'memory_usage') else 'N/A'
    sarima_memory = sarima.fitted_model.memory_usage if hasattr(sarima.fitted_model, 'memory_usage') else 'N/A'
    
    return {
        'ARIMA': {
            'train_time': arima_train_time,
            'forecast_time': arima_forecast_time,
            'memory': arima_memory
        },
        'SARIMA': {
            'train_time': sarima_train_time,
            'forecast_time': sarima_forecast_time,
            'memory': sarima_memory
        }
    }

benchmark_results = benchmark_models(data)
```

### 2. Gestione dell'Incertezza

#### Intervalli di Confidenza

```python
def compare_uncertainty_quantification(arima_model, sarima_model, steps=12):
    """
    Confronta qualità degli intervalli di confidenza
    """
    # Genera previsioni con intervalli
    arima_forecast, arima_ci = arima_model.forecast(
        steps=steps, return_conf_int=True, alpha=0.05
    )
    
    sarima_forecast, sarima_ci = sarima_model.forecast(
        steps=steps, return_conf_int=True, alpha=0.05
    )
    
    # Calcola larghezza intervalli
    arima_width = (arima_ci.iloc[:, 1] - arima_ci.iloc[:, 0]).mean()
    sarima_width = (sarima_ci.iloc[:, 1] - sarima_ci.iloc[:, 0]).mean()
    
    return {
        'ARIMA': {
            'avg_interval_width': arima_width,
            'forecast_std': arima_forecast.std()
        },
        'SARIMA': {
            'avg_interval_width': sarima_width,
            'forecast_std': sarima_forecast.std()
        }
    }
```

## Raccomandazioni Finali

### Matrice di Decisione Finale

| Situazione | Modello Raccomandato | Confidenza | Note |
|------------|---------------------|------------|------|
| Serie senza stagionalità evidente | ARIMA | Alta | Maggiore semplicità e robustezza |
| Serie con stagionalità forte (ACF(s) > 0.5) | SARIMA | Alta | Miglioramento accuratezza significativo |
| Serie breve (< 2*s osservazioni) | ARIMA | Media | Dati insufficienti per SARIMA |
| Budget computazionale limitato | ARIMA | Media | Trade-off velocità/accuratezza |
| Applicazione critica con stagionalità | SARIMA | Alta | Accuratezza prioritaria |
| Prototipazione rapida | ARIMA | Bassa | Sviluppo veloce, validare con SARIMA |
| Dati con stagionalità multipla | Modelli avanzati | Media | SARIMA gestisce solo una stagionalità |
| Serie con drift/trend forte | SARIMA | Media | Migliore gestione componenti multiple |

### Processo di Selezione Consigliato

1. **Analisi Esplorativa** (30 min)
   - Plot della serie temporale
   - Calcolo ACF/PACF
   - Test di stagionalità automatici

2. **Modellazione Parallela** (1-2 ore)
   - Fit ARIMA automatico
   - Fit SARIMA per periodi stagionali candidati
   - Validazione incrociata

3. **Selezione Finale** (30 min)
   - Confronto AIC/BIC penalizzato
   - Analisi residui
   - Test out-of-sample

4. **Validazione in Produzione** (continua)
   - Monitoraggio accuracy nel tempo
   - Re-training periodico
   - A/B testing se possibile

### Considerazioni per il Futuro

Man mano che i modelli di machine learning e deep learning diventano più accessibili, la scelta potrebbe espandersi a:

- **Prophet**: per serie con stagionalità multipla e anomalie
- **LSTM/GRU**: per pattern non lineari complessi
- **Transformer**: per serie molto lunghe con dipendenze a lungo termine
- **Ensemble**: combinazione di ARIMA/SARIMA con ML

Tuttavia, ARIMA e SARIMA rimangono fondamentali per:
- Interpretabilità e spiegabilità
- Robustezza statistica
- Basi teoriche solide
- Efficienza computazionale per deployment su larga scala

La scelta finale deve sempre bilanciare accuratezza, interpretabilità, costi computazionali e requisiti del business specifico.