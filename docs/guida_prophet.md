# Guida Pratica Facebook Prophet

## 🚀 Quick Start

### Installazione e Setup

```bash
# Installa Prophet
uv add prophet

# Verifica installazione
python -c "import prophet; print('Prophet OK!')"
```

### Primo Modello Prophet

```python
from arima_forecaster.core import ProphetForecaster
import pandas as pd

# Carica dati
data = pd.read_csv('vendite.csv', index_col=0, parse_dates=True)
series = data['vendite']  # pd.Series con DatetimeIndex

# Crea modello
model = ProphetForecaster(
    growth='linear',
    yearly_seasonality=True,
    weekly_seasonality=True,
    country_holidays='IT'
)

# Addestra
model.fit(series)

# Previsioni
forecast = model.forecast(steps=30, confidence_level=0.95)
print(f"Previsioni: {forecast.mean():.2f}")
```

## 🎛️ Configurazione Parametri

### Parametri Base

```python
model = ProphetForecaster(
    # Tipo di crescita
    growth='linear',              # 'linear', 'logistic', 'flat'
    
    # Stagionalità
    yearly_seasonality='auto',     # True, False, 'auto', o int (fourier terms)
    weekly_seasonality='auto',     # True, False, 'auto', o int  
    daily_seasonality=False,       # True, False, 'auto', o int
    
    # Modalità stagionalità
    seasonality_mode='additive',   # 'additive' o 'multiplicative'
    
    # Festività
    country_holidays='IT',         # 'IT', 'US', 'UK', 'DE', 'FR', 'ES'
    
    # Parametri di regolarizzazione
    changepoint_prior_scale=0.05,     # Flessibilità trend (0.001-0.5)
    seasonality_prior_scale=10.0,     # Flessibilità stagionalità (0.01-50)
    holidays_prior_scale=10.0         # Flessibilità festività (0.01-50)
)
```

### Trend Logistico con Capacità

```python
# Per crescita con saturazione
model = ProphetForecaster(
    growth='logistic'
)

# Aggiungi capacità ai dati
data_with_cap = series.to_frame('y')
data_with_cap['cap'] = 1000000  # Limite superiore
model.fit(data_with_cap)

# Previsioni con capacità futura
future_cap = pd.DataFrame({
    'cap': [1000000] * 30  # Capacità per 30 giorni futuri
})
forecast = model.forecast(steps=30, exog_future=future_cap)
```

## 🔍 Selezione Automatica Parametri

### Auto-Selection Semplice

```python
from arima_forecaster.core import ProphetModelSelector

# Configurazione ricerca
selector = ProphetModelSelector(
    growth_types=['linear', 'logistic'],
    seasonality_modes=['additive', 'multiplicative'], 
    country_holidays=['IT', 'US', None],
    max_models=20
)

# Esecuzione ricerca
selector.search(series, verbose=True)

# Miglior modello
best_model = selector.get_best_model()
best_params = selector.get_best_params()

print("Migliori parametri:")
for param, value in best_params.items():
    print(f"  {param}: {value}")
```

### Ricerca Avanzata con Cross-Validation

```python
# Ricerca più approfondita
selector = ProphetModelSelector(
    growth_types=['linear', 'logistic', 'flat'],
    seasonality_modes=['additive', 'multiplicative'],
    country_holidays=['IT', 'US', 'UK', 'DE', None],
    max_models=50,
    cv_horizon='60 days',        # Orizzonte cross-validation
    cv_period='30 days',         # Frequenza evaluation
    cv_initial='365 days'        # Training iniziale minimo
)

selector.search(series)

# Risultati dettagliati
results_df = selector.get_results_summary()
print(results_df.head())

# Metriche del miglior modello
print(f"MAPE: {selector.best_mape:.2f}%")
print(f"MAE: {selector.best_mae:.2f}")
```

## 📊 Gestione Stagionalità

### Stagionalità Multiple

```python
# Modello con multiple stagionalità
model = ProphetForecaster(
    yearly_seasonality=10,    # 10 termini di Fourier per anno
    weekly_seasonality=3,     # 3 termini di Fourier per settimana
    daily_seasonality=False   # Disabilita stagionalità giornaliera
)

# Stagionalità custom
model.fit(series)
model.add_seasonality(
    name='monthly',
    period=30.5,      # Periodo in giorni
    fourier_order=5   # Numero termini di Fourier
)
```

### Stagionalità Condizionali

```python
# Stagionalità diverse per condizioni
model.add_seasonality(
    name='weekly_business',
    period=7,
    fourier_order=3,
    condition_name='is_business_day'  
)

# Aggiungi condizione ai dati
data_extended = series.to_frame('y')
data_extended['is_business_day'] = data_extended.index.weekday < 5
model.fit(data_extended)
```

## 🎉 Gestione Festività

### Festività Automatiche per Paese

```python
# Festività italiane incluse automaticamente
model = ProphetForecaster(country_holidays='IT')
```

### Festività Custom

```python
import pandas as pd

# Definisci festività personalizzate
custom_holidays = pd.DataFrame({
    'holiday': ['black_friday', 'black_friday', 'black_friday'],
    'ds': pd.to_datetime(['2022-11-25', '2023-11-24', '2024-11-29']),
    'lower_window': -1,    # Effetto inizia 1 giorno prima
    'upper_window': 0      # Effetto termina il giorno stesso
})

model = ProphetForecaster()
model.add_holidays(custom_holidays)
```

### Festività con Finestre Estese

```python
# Natale con effetti prolungati
christmas = pd.DataFrame({
    'holiday': 'christmas',
    'ds': pd.to_datetime(['2022-12-25', '2023-12-25', '2024-12-25']),
    'lower_window': -7,    # Settimana prima
    'upper_window': 2      # Due giorni dopo
})

# Capodanno
new_year = pd.DataFrame({
    'holiday': 'new_year', 
    'ds': pd.to_datetime(['2023-01-01', '2024-01-01', '2025-01-01']),
    'lower_window': -1,
    'upper_window': 1
})

holidays = pd.concat([christmas, new_year])
model.add_holidays(holidays)
```

## 🔧 Tuning dei Prior

### Flessibilità del Trend

```python
# Trend molto rigido (per serie stabili)
model_rigid = ProphetForecaster(changepoint_prior_scale=0.001)

# Trend molto flessibile (per serie con cambiamenti frequenti)
model_flexible = ProphetForecaster(changepoint_prior_scale=0.5)

# Auto-tuning del trend
import numpy as np
scales = np.logspace(-3, 0, 10)  # Da 0.001 a 1.0
best_scale = None
best_mape = float('inf')

for scale in scales:
    model = ProphetForecaster(changepoint_prior_scale=scale)
    model.fit(series[:-30])  # Train su tutto meno ultimi 30
    forecast = model.forecast(steps=30)
    mape = np.mean(np.abs((series[-30:] - forecast.values) / series[-30:].values)) * 100
    
    if mape < best_mape:
        best_mape = mape
        best_scale = scale

print(f"Miglior changepoint_prior_scale: {best_scale:.4f}")
```

### Flessibilità Stagionalità

```python
# Stagionalità regolare
model_regular = ProphetForecaster(seasonality_prior_scale=1.0)

# Stagionalità molto variabile  
model_variable = ProphetForecaster(seasonality_prior_scale=50.0)
```

## 📈 Previsioni e Intervalli

### Previsioni Base

```python
# Previsioni puntuali
forecast = model.forecast(steps=30)

# Con intervalli di confidenza
forecast_with_ci = model.forecast(
    steps=30, 
    confidence_level=0.95,
    return_confidence_intervals=True
)

print(f"Previsione: {forecast_with_ci['forecast'].mean():.2f}")
print(f"Intervallo 95%: [{forecast_with_ci['lower'].mean():.2f}, {forecast_with_ci['upper'].mean():.2f}]")
```

### Previsioni con Regressori

```python
# Aggiungi regressori al modello
model.add_regressor('temperature')
model.add_regressor('promotion', standardize=False)

# Dati con regressori
data_with_regressors = series.to_frame('y')
data_with_regressors['temperature'] = temperature_data
data_with_regressors['promotion'] = promotion_data

model.fit(data_with_regressors)

# Previsioni con regressori futuri
future_regressors = pd.DataFrame({
    'temperature': [20.5] * 30,   # Temperatura costante
    'promotion': [0, 1, 1, 0] + [0] * 26  # Promozione per 2 giorni
})

forecast = model.forecast(steps=30, exog_future=future_regressors)
```

## 📊 Visualizzazioni

### Componenti del Modello

```python
from arima_forecaster.visualization import ForecastPlotter

plotter = ForecastPlotter()

# Plot principale con forecast
fig = plotter.plot_forecast(
    data=series,
    forecast=forecast,
    confidence_intervals=True,
    title="Previsioni Prophet - Vendite"
)

# Decomposizione componenti
components_fig = plotter.plot_prophet_components(model, series)
```

### Analisi Trend e Stagionalità

```python
# Estrai componenti
components = model.predict_components(series)

# Plot trend
plt.figure(figsize=(12, 4))
plt.plot(series.index, components['trend'])
plt.title('Componente di Trend')
plt.xlabel('Data')
plt.ylabel('Valore')

# Plot stagionalità annuale
plt.figure(figsize=(12, 4)) 
plt.plot(range(365), components.groupby(components.index.dayofyear)['yearly'].mean())
plt.title('Stagionalità Annuale Media')
plt.xlabel('Giorno dell\'anno')
plt.ylabel('Effetto')
```

## 🏥 Diagnostica e Validazione

### Cross-Validation

```python
from arima_forecaster.evaluation import ModelEvaluator

evaluator = ModelEvaluator()

# Cross-validation time series
cv_results = evaluator.cross_validate_prophet(
    model=model,
    data=series,
    initial='365 days',    # Training iniziale
    period='90 days',      # Frequenza evaluation
    horizon='30 days'      # Orizzonte previsione
)

print("Risultati Cross-Validation:")
print(f"MAPE medio: {cv_results['mape'].mean():.2f}%")
print(f"MAE medio: {cv_results['mae'].mean():.2f}")
print(f"Coverage: {cv_results['coverage'].mean():.2f}")
```

### Analisi Residui

```python
# Calcola residui
fitted_values = model.predict(steps=len(series))
residuals = series - fitted_values

# Test normalità
from scipy.stats import jarque_bera
stat, p_value = jarque_bera(residuals.dropna())
print(f"Test Jarque-Bera: statistic={stat:.4f}, p-value={p_value:.4f}")

# Test autocorrelazione
from statsmodels.stats.diagnostic import acorr_ljungbox
ljung_box = acorr_ljungbox(residuals.dropna(), lags=10)
print(f"Test Ljung-Box: p-value={ljung_box['lb_pvalue'].iloc[-1]:.4f}")

# Plot residui
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Residui nel tempo
axes[0,0].plot(series.index, residuals)
axes[0,0].set_title('Residui nel Tempo')

# Q-Q plot
from scipy.stats import probplot
probplot(residuals.dropna(), dist="norm", plot=axes[0,1])
axes[0,1].set_title('Q-Q Plot')

# Istogramma
axes[1,0].hist(residuals.dropna(), bins=30, alpha=0.7)
axes[1,0].set_title('Distribuzione Residui')

# ACF residui
from statsmodels.tsa.stattools import acf
lags = range(1, min(20, len(residuals)//4))
autocorr = acf(residuals.dropna(), nlags=max(lags))
axes[1,1].bar(lags, autocorr[1:len(lags)+1])
axes[1,1].set_title('Autocorrelazione Residui')

plt.tight_layout()
plt.show()
```

## ⚙️ Ottimizzazione Performance

### Parallelizzazione

```python
# Prophet supporta automaticamente il parallelismo
import os
os.environ['PROPHET_NUM_THREADS'] = '4'  # Usa 4 core

# Per auto-selection su dataset grandi
selector = ProphetModelSelector(
    max_models=100,
    parallel=True,        # Abilita parallelizzazione
    n_jobs=4             # Numero processi paralleli
)
```

### Gestione Memoria

```python
# Per serie molto lunghe (>10k osservazioni)
model = ProphetForecaster(
    # Riduce termini di Fourier per efficienza
    yearly_seasonality=5,    # Invece di 10 default
    weekly_seasonality=2,    # Invece di 3 default
    
    # Meno changepoint per velocità
    n_changepoints=15        # Invece di 25 default
)

# Preprocessing per velocità
series_resampled = series.resample('D').mean()  # Giornaliero invece di orario
model.fit(series_resampled)
```

## 🔗 Integrazione Avanzata

### Con Dashboard Streamlit

```python
# Il modello si integra automaticamente nella dashboard
# Vai su: http://localhost:8501 dopo aver avviato
# python scripts/run_dashboard.py

# Seleziona "Prophet" nel menu modelli
# Configura parametri tramite UI
# Visualizza risultati interattivi
```

### Con API REST

```bash
# Addestra modello Prophet via API
curl -X POST "http://localhost:8000/models/train/prophet" \
     -H "Content-Type: application/json" \
     -d '{
       "data": {
         "timestamps": ["2023-01-01", "2023-01-02", "2023-01-03"],
         "values": [100, 105, 98]
       },
       "growth": "linear",
       "yearly_seasonality": "auto",
       "country_holidays": "IT"
     }'

# Selezione automatica
curl -X POST "http://localhost:8000/models/train/prophet/auto-select" \
     -H "Content-Type: application/json" \
     -d '{
       "data": {...},
       "growth_types": ["linear", "logistic"],
       "max_models": 20
     }'
```

### Con Report Quarto

```python
from arima_forecaster.reporting import QuartoReportGenerator

# Genera report completo Prophet
reporter = QuartoReportGenerator()
report = reporter.generate_prophet_report(
    model=model,
    data=series,
    forecast=forecast,
    title="Analisi Prophet - Vendite Q4",
    language='it'
)

# Output: report HTML, PDF, DOCX
report.save('report_prophet_vendite.html')
```

## 🎯 Best Practices

### 1. **Preparazione Dati**
```python
# ✅ Buone pratiche
- Frequenza regolare (giornaliera, settimanale)
- Almeno 2 cicli stagionali completi
- Gestisci outliers estremi
- Index datetime corretto

# ❌ Evita
- Frequenze irregolari
- Gap troppo lunghi nei dati
- Serie troppo corte (<100 osservazioni)
```

### 2. **Selezione Parametri**
```python
# ✅ Strategie vincenti
- Usa auto-selection per primo approccio
- Testa più paesi per festività se applicabile  
- Valida con cross-validation
- Monitora coverage intervalli confidenza

# ❌ Errori comuni
- changepoint_prior_scale troppo alto (overfitting)
- Ignorare validazione out-of-sample
- Non testare stagionalità moltiplicativa
```

### 3. **Interpretazione Business**
```python
# ✅ Sfrutta interpretabilità
- Analizza componenti separatamente
- Identifica pattern nel trend
- Quantifica impatto festività
- Comunica risultati con grafici

# ❌ Trattamento black-box
- Non ignorare componenti
- Non limitarti solo alle previsioni
- Spiega risultati al business
```

---

## 🚀 Prossimi Passi

Dopo aver padroneggiato le basi:

1. **[Teoria Prophet](teoria_prophet.md)** - Approfondimenti matematici
2. **[Prophet vs ARIMA](prophet_vs_arima.md)** - Confronti dettagliati  
3. **[Casi Studio](esempi_prophet.md)** - Applicazioni reali
4. **[Troubleshooting](prophet_faq.md)** - Risoluzione problemi comuni

**Happy Forecasting with Prophet! 📈**