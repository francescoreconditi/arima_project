# Prophet vs ARIMA/SARIMA: Guida Comparativa Completa

## 📚 Panoramica

Questo documento fornisce una comparazione dettagliata tra **Facebook Prophet** e i modelli **ARIMA/SARIMA**, due delle metodologie più utilizzate per il forecasting di serie temporali. La scelta tra questi approcci dipende dalle caratteristiche specifiche dei dati, dal contesto applicativo e dai requisiti di business.

---

## 🔬 Fondamenti Teorici

### ARIMA/SARIMA: Approccio Statistico Classico

**ARIMA** (AutoRegressive Integrated Moving Average) è una famiglia di modelli statistici basati su:

#### Componenti ARIMA(p,d,q):
- **AR(p)** - AutoRegressivo: `Y_t = φ₁Y_{t-1} + φ₂Y_{t-2} + ... + φₚY_{t-p} + ε_t`
- **I(d)** - Integrato: Differenziazione per rendere la serie stazionaria
- **MA(q)** - Media Mobile: Dipendenza dagli errori passati

#### SARIMA(p,d,q)(P,D,Q,s):
Estende ARIMA con componenti stagionali:
- **(P,D,Q,s)**: Parametri stagionali con periodo s
- **Esempio**: SARIMA(1,1,1)(1,1,1,12) per dati mensili con stagionalità annuale

```python
from arima_forecaster.core import SARIMAForecaster

# Modello SARIMA manuale
model = SARIMAForecaster(
    order=(1, 1, 1),           # ARIMA componenti
    seasonal_order=(1, 1, 1, 12),  # Stagionalità annuale
    trend='c'                  # Con costante
)
model.fit(series)
forecast = model.forecast(steps=12)
```

### Prophet: Approccio Decomposizionale Moderno

**Prophet** decompone la serie temporale in componenti interpretabili:

#### Formula Prophet:
```
y(t) = g(t) + s(t) + h(t) + ε_t
```

Dove:
- **g(t)**: Trend (lineare o logistico con saturazione)
- **s(t)**: Stagionalità (Fourier series per weekly/yearly)
- **h(t)**: Effetti holiday e eventi speciali
- **ε_t**: Rumore idiosincratico

#### Caratteristiche Uniche:
1. **Changepoint Detection**: Rileva automaticamente cambi di trend
2. **Stagionalità Multipla**: Gestisce simultaneamente daily/weekly/yearly
3. **Holiday Effects**: Incorpora festività e eventi speciali
4. **Uncertainty Quantification**: Intervalli di confidenza realistici

```python
from arima_forecaster.core import ProphetForecaster

# Modello Prophet con features avanzate
model = ProphetForecaster(
    growth='linear',              # o 'logistic' con cap
    yearly_seasonality=True,      # Stagionalità annuale
    weekly_seasonality=True,      # Stagionalità settimanale
    daily_seasonality=False,      # Per dati orari
    country_holidays='IT',        # Festività italiane
    seasonality_mode='additive'   # o 'multiplicative'
)
model.fit(series)
forecast = model.forecast(steps=30, confidence_level=0.95)
```

---

## 📊 Confronto Dettagliato

### 🎯 Accuratezza e Performance

| Aspetto | ARIMA/SARIMA | Prophet |
|---------|--------------|---------|
| **Accuratezza su serie regolari** | ⭐⭐⭐⭐⭐ Eccellente | ⭐⭐⭐⭐ Molto buona |
| **Robustezza a outlier** | ⭐⭐ Sensibile | ⭐⭐⭐⭐ Robusta |
| **Gestione dati mancanti** | ⭐⭐ Richiede preprocessing | ⭐⭐⭐⭐⭐ Gestisce automaticamente |
| **Serie con changepoint** | ⭐⭐ Difficile | ⭐⭐⭐⭐⭐ Eccellente |
| **Stagionalità complessa** | ⭐⭐⭐ Buona (con SARIMA) | ⭐⭐⭐⭐⭐ Eccellente |
| **Holiday effects** | ⭐ Manuale | ⭐⭐⭐⭐⭐ Automatica |

### ⚡ Facilità d'Uso e Interpretabilità

| Aspetto | ARIMA/SARIMA | Prophet |
|---------|--------------|---------|
| **Semplicità implementazione** | ⭐⭐ Richiede expertise | ⭐⭐⭐⭐⭐ User-friendly |
| **Selezione parametri** | ⭐⭐ Complessa (p,d,q,P,D,Q,s) | ⭐⭐⭐⭐ Intuitiva |
| **Interpretabilità componenti** | ⭐⭐ Solo residui/ACF | ⭐⭐⭐⭐⭐ Trend/stagionalità/holiday |
| **Diagnostica modello** | ⭐⭐⭐⭐⭐ Ricca (Ljung-Box, etc.) | ⭐⭐⭐ Limitata |
| **Controllo fine** | ⭐⭐⭐⭐⭐ Massimo | ⭐⭐⭐ Buono |

### 🚀 Performance Computazionale

| Aspetto | ARIMA/SARIMA | Prophet |
|---------|--------------|---------|
| **Velocità training** | ⭐⭐⭐⭐ Veloce | ⭐⭐⭐ Moderata |
| **Velocità forecast** | ⭐⭐⭐⭐⭐ Istantanea | ⭐⭐⭐⭐ Veloce |
| **Scalabilità** | ⭐⭐⭐ Buona | ⭐⭐⭐⭐ Molto buona |
| **Requisiti memoria** | ⭐⭐⭐⭐ Bassi | ⭐⭐⭐ Moderati |

---

## 🎨 Quando Usare Ciascun Modello

### 🏆 Usa ARIMA/SARIMA quando:

#### ✅ **Ideale per:**
- **Serie temporali regolari e stazionarie**
- **Dati di alta qualità senza outlier significativi**
- **Controllo statistico rigoroso necessario**
- **Orizzonte forecast breve-medio (< 6 mesi)**
- **Contesti accademici/scientifici dove servono test statistici**
- **Budget computazionale limitato**

#### 📈 **Esempi Applicativi:**
```python
# Caso: Previsioni finanziarie giornaliere
from arima_forecaster.core import ARIMAModelSelector

# Auto-selezione ARIMA per dati finanziari
selector = ARIMAModelSelector(max_p=3, max_d=2, max_q=3)
selector.search(stock_prices)
best_model = selector.get_best_model()

# Diagnostica rigorosa per validazione
diagnostics = best_model.get_diagnostics()
print(f"Ljung-Box p-value: {diagnostics['ljung_box_pvalue']}")
print(f"ADF stationarity: {diagnostics['adf_statistic']}")
```

#### 🏭 **Settori Tipici:**
- **Finanza**: Prezzi azioni, tassi di cambio, bond yields
- **Economia**: PIL, inflazione, indici economici
- **Produzione**: Output industriale regolare
- **Energia**: Consumi elettrici stabili

### 🚀 Usa Prophet quando:

#### ✅ **Ideale per:**
- **Serie con trend non lineari e changepoint**
- **Stagionalità multipla complessa (daily + weekly + yearly)**
- **Presenza di holiday ed eventi speciali**
- **Dati con outlier e valori mancanti**
- **Business analytics e KPI tracking**
- **Orizzonte forecast lungo (> 6 mesi)**
- **Necessità di interpretabilità business**

#### 📊 **Esempi Applicativi:**
```python
# Caso: E-commerce con stagionalità complessa
from arima_forecaster.core import ProphetForecaster

# Prophet per vendite e-commerce con festività
model = ProphetForecaster(
    growth='logistic',           # Crescita con saturazione
    yearly_seasonality=True,     # Stagionalità annuale (Natale, estate)
    weekly_seasonality=True,     # Weekend vs weekdays
    country_holidays='IT',       # Black Friday, Natale, etc.
    seasonality_mode='multiplicative'  # Effetti proporzionali
)

# Aggiunge regressori personalizzati
model.add_regressor('marketing_spend')  # Budget marketing
model.add_regressor('competitor_price')  # Prezzi concorrenti

model.fit(sales_data, exog_data)
forecast = model.forecast(steps=365)  # 1 anno

# Analisi componenti per insights business
components = model.predict_components(sales_data)
print("Trend annuale:", components['trend'].iloc[-1] - components['trend'].iloc[0])
print("Impatto weekend:", components['weekly'].max() - components['weekly'].min())
```

#### 🏪 **Settori Tipici:**
- **Retail**: Vendite, traffico negozi, e-commerce
- **Marketing**: Website traffic, conversioni, engagement
- **Hospitality**: Prenotazioni hotel, ristoranti
- **Healthcare**: Pazienti, emergenze ospedaliere
- **Utilities**: Consumi energetici con pattern complessi

---

## 🔍 Esempi Pratici Comparativi

### Scenario 1: Vendite Retail con Stagionalità

```python
import pandas as pd
import numpy as np
from arima_forecaster.core import SARIMAForecaster, ProphetForecaster
from arima_forecaster.evaluation import ModelEvaluator

# Dati di vendita retail (2 anni)
dates = pd.date_range('2022-01-01', '2024-01-01', freq='D')
trend = np.linspace(100, 200, len(dates))
yearly_season = 30 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
weekly_season = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
holidays_boost = np.where((dates.month == 12) & (dates.day >= 20), 50, 0)
noise = np.random.normal(0, 10, len(dates))

sales = trend + yearly_season + weekly_season + holidays_boost + noise
sales_series = pd.Series(sales, index=dates)

# Split train/test
train_data = sales_series[:'2023-06-30']
test_data = sales_series['2023-07-01':]

# SARIMA Approach
sarima_model = SARIMAForecaster(
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),  # Stagionalità settimanale
    trend='c'
)
sarima_model.fit(train_data)
sarima_forecast = sarima_model.forecast(steps=len(test_data))

# Prophet Approach
prophet_model = ProphetForecaster(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    country_holidays='IT'
)
prophet_model.fit(train_data)
prophet_forecast = prophet_model.forecast(steps=len(test_data))

# Valutazione
evaluator = ModelEvaluator()
sarima_metrics = evaluator.calculate_forecast_metrics(test_data, sarima_forecast)
prophet_metrics = evaluator.calculate_forecast_metrics(test_data, prophet_forecast)

print("SARIMA Performance:")
print(f"  MAPE: {sarima_metrics['mape']:.2f}%")
print(f"  RMSE: {sarima_metrics['rmse']:.2f}")

print("Prophet Performance:")
print(f"  MAPE: {prophet_metrics['mape']:.2f}%")
print(f"  RMSE: {prophet_metrics['rmse']:.2f}")
```

### Scenario 2: Dati Finanziari con High Frequency

```python
# Caso: Trading intraday con dati alta frequenza
from arima_forecaster.core import ARIMAModelSelector

# Dati finanziari (ogni 5 minuti)
dates = pd.date_range('2024-01-01', '2024-02-01', freq='5min')
prices = pd.Series(100 + np.cumsum(np.random.normal(0, 0.1, len(dates))), index=dates)

# ARIMA è preferibile per dati finanziari
selector = ARIMAModelSelector(
    max_p=5, max_d=1, max_q=5,
    information_criterion='aic',
    seasonal=False  # Dati intraday raramente stagionali
)

selector.search(prices)
best_arima = selector.get_best_model()

# Prophet meno adatto per alta frequenza finanziaria
# (ma si può provare)
prophet_financial = ProphetForecaster(
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=True,  # Pattern intraday
    seasonality_prior_scale=0.1  # Bassa per dati finanziari
)

print(f"ARIMA order trovato: {best_arima.order}")
print("Prophet configurato per alta frequenza")
```

---

## 🛠️ Linee Guida per la Scelta

### 📋 Decision Framework

#### 1. **Analisi Preliminare dei Dati**
```python
def analyze_time_series(series):
    """Analizza serie temporale per suggerire modello ottimale."""
    
    analysis = {}
    
    # Lunghezza serie
    analysis['length'] = len(series)
    analysis['frequency'] = pd.infer_freq(series.index)
    
    # Stazionarietà
    from statsmodels.tsa.stattools import adfuller
    adf_stat, adf_pvalue = adfuller(series.dropna())[:2]
    analysis['stationary'] = adf_pvalue < 0.05
    
    # Outlier
    q75, q25 = np.percentile(series, [75, 25])
    iqr = q75 - q25
    outlier_count = len(series[(series < q25 - 1.5*iqr) | (series > q75 + 1.5*iqr)])
    analysis['outlier_ratio'] = outlier_count / len(series)
    
    # Valori mancanti
    analysis['missing_ratio'] = series.isnull().sum() / len(series)
    
    # Stagionalità (test Friedman)
    if len(series) >= 24:
        from scipy import stats
        if analysis['frequency'] and 'D' in analysis['frequency']:
            weekly_groups = [series[series.index.dayofweek == i].dropna() for i in range(7)]
            if all(len(g) > 0 for g in weekly_groups):
                friedman_stat, friedman_p = stats.friedmanchisquare(*weekly_groups)
                analysis['weekly_seasonality'] = friedman_p < 0.05
    
    return analysis

def recommend_model(analysis):
    """Raccomanda modello basato su analisi."""
    
    score_arima = 0
    score_prophet = 0
    
    # Lunghezza serie
    if analysis['length'] < 100:
        score_arima += 2  # ARIMA meglio su serie corte
    else:
        score_prophet += 1
    
    # Stazionarietà
    if analysis['stationary']:
        score_arima += 2
    else:
        score_prophet += 1  # Prophet gestisce meglio non-stazionarietà
    
    # Outlier
    if analysis['outlier_ratio'] > 0.05:
        score_prophet += 3  # Prophet molto più robusto
    else:
        score_arima += 1
    
    # Valori mancanti
    if analysis['missing_ratio'] > 0.02:
        score_prophet += 3  # Prophet gestisce meglio missing values
    
    # Stagionalità
    if analysis.get('weekly_seasonality', False):
        score_prophet += 2  # Prophet migliore per stagionalità complessa
    
    if score_prophet > score_arima:
        return "Prophet", score_prophet, score_arima
    else:
        return "ARIMA/SARIMA", score_arima, score_prophet

# Esempio d'uso
analysis = analyze_time_series(your_series)
recommendation, score_winner, score_loser = recommend_model(analysis)
print(f"Modello raccomandato: {recommendation}")
print(f"Score: {score_winner} vs {score_loser}")
```

#### 2. **Matrice Decisionale Rapida**

| Criterio | ARIMA/SARIMA | Prophet | Peso |
|----------|--------------|---------|------|
| Serie regolare, pochi outlier | ✅ | ❌ | 🔥🔥🔥 |
| Changepoint frequenti | ❌ | ✅ | 🔥🔥🔥 |
| Holiday effects | ❌ | ✅ | 🔥🔥 |
| Controllo statistico rigoroso | ✅ | ❌ | 🔥🔥 |
| Interpretabilità business | ❌ | ✅ | 🔥🔥 |
| Velocità training | ✅ | ❌ | 🔥 |
| Robustezza a missing data | ❌ | ✅ | 🔥 |

---

## 🔗 Approccio Ibrido

### 🌟 Combine Both: Ensemble Methods

```python
from arima_forecaster.ensemble import EnsembleForecaster
from arima_forecaster.core import SARIMAForecaster, ProphetForecaster

# Ensemble SARIMA + Prophet
sarima_model = SARIMAForecaster(order=(1,1,1), seasonal_order=(1,1,1,12))
prophet_model = ProphetForecaster(yearly_seasonality=True, weekly_seasonality=True)

# Ensemble con pesi ottimizzati
ensemble = EnsembleForecaster([
    ('sarima', sarima_model, 0.6),    # 60% peso SARIMA
    ('prophet', prophet_model, 0.4)   # 40% peso Prophet
])

ensemble.fit(train_data)
ensemble_forecast = ensemble.forecast(steps=30)

print("Ensemble combina i punti di forza di entrambi i modelli!")
```

### 🔄 Sequential Approach

```python
# Approccio sequenziale: Prophet per trend, ARIMA per residui
def hybrid_forecast(series, steps):
    """Forecasting ibrido: Prophet + ARIMA sui residui."""
    
    # Fase 1: Prophet per trend e stagionalità
    prophet = ProphetForecaster(yearly_seasonality=True, weekly_seasonality=True)
    prophet.fit(series)
    prophet_fitted = prophet.predict()
    
    # Fase 2: ARIMA sui residui
    residuals = series - prophet_fitted
    residuals = residuals.dropna()
    
    arima = ARIMAForecaster(order=(1,1,1))
    arima.fit(residuals)
    
    # Forecast combinato
    prophet_forecast = prophet.forecast(steps=steps)
    arima_residual_forecast = arima.forecast(steps=steps)
    
    hybrid_forecast = prophet_forecast + arima_residual_forecast
    
    return hybrid_forecast

# Uso
hybrid_result = hybrid_forecast(your_series, steps=30)
print("Hybrid forecast: trend (Prophet) + noise (ARIMA)")
```

---

## 📚 Best Practices e Consigli Avanzati

### 🎯 Per ARIMA/SARIMA

#### ✅ **Do's:**
1. **Pre-processing rigoroso**: Stazionarietà, outlier removal, missing value imputation
2. **Diagnostica completa**: Ljung-Box, Jarque-Bera, ACF/PACF analysis
3. **Cross-validation**: Time series split, rolling window validation
4. **Model selection**: AIC/BIC optimization, auto-selection con `ARIMAModelSelector`

```python
# Esempio best practices ARIMA
from arima_forecaster.core import ARIMAModelSelector
from arima_forecaster.data import TimeSeriesPreprocessor

# 1. Pre-processing completo
preprocessor = TimeSeriesPreprocessor(
    missing_strategy='interpolate',
    outlier_method='iqr',
    stationarity_method='difference'
)
clean_series = preprocessor.preprocess_pipeline(raw_series)

# 2. Model selection automatica
selector = ARIMAModelSelector(
    p_range=(0, 3), d_range=(0, 2), q_range=(0, 3),
    scoring='aic', max_models=50, verbose=True
)
best_model, results = selector.search(clean_series)

# 3. Diagnostica post-fitting
diagnostics = best_model.diagnostic_plots()
print(f"Ljung-Box p-value: {diagnostics['ljung_box_pvalue']}")
```

#### ❌ **Don'ts:**
1. **Non** ignorare i test di stazionarietà
2. **Non** usare modelli troppo complessi senza giustificazione (overfitting)
3. **Non** applicare ARIMA direttamente a serie con trend forte senza differenziazione
4. **Non** dimenticare la validazione out-of-sample

### 🎯 Per Prophet

#### ✅ **Do's:**
1. **Eventi e holiday**: Sfruttare la gestione nativa di festività e eventi speciali
2. **Saturazione**: Usare logistic growth per serie con limiti fisici
3. **Regolarization**: Tuning di `changepoint_prior_scale` per trend flexibility
4. **Cross-validation**: Built-in time series CV con `cross_validation()`

```python
# Esempio best practices Prophet  
from arima_forecaster.core import ProphetForecaster
import pandas as pd

# 1. Preparazione eventi custom
custom_events = pd.DataFrame({
    'holiday': ['black_friday', 'cyber_monday'],
    'ds': pd.to_datetime(['2023-11-24', '2023-11-27']),
    'lower_window': [0, 0],
    'upper_window': [1, 1],
})

# 2. Configurazione ottimale
model = ProphetForecaster(
    growth='logistic',  # Per serie con saturazione
    yearly_seasonality=True,
    weekly_seasonality='auto',  # Automatico
    daily_seasonality=False,
    holidays=custom_events,
    seasonality_mode='multiplicative',  # Se effetto stagionale proporzionale
    changepoint_prior_scale=0.05,  # Controllo flessibilità trend
    seasonality_prior_scale=10.0   # Forza stagionalità
)

# 3. Fitting con capacity (per logistic growth)
df = pd.DataFrame({'ds': series.index, 'y': series.values})
df['cap'] = 1000  # Capacità massima
model.fit(df)

# 4. Cross-validation built-in
from prophet.diagnostics import cross_validation, performance_metrics
cv_results = cross_validation(model.model, initial='730 days', period='180 days', horizon='365 days')
metrics = performance_metrics(cv_results)
print(f"Average MAPE: {metrics['mape'].mean():.2%}")
```

#### ❌ **Don'ts:**
1. **Non** ignorare outlier estremi (Prophet è robusto ma non immune)
2. **Non** over-parameterizzare con troppi regressori custom
3. **Non** aspettarsi buone performance su serie molto corte (<2 anni)
4. **Non** dimenticare di settare `cap` per logistic growth

---

## 📊 Tabelle Comparative Dettagliate

### 🆚 Confronto Performance per Tipologia Dati

| Tipologia Serie | ARIMA/SARIMA | Prophet | Vincitore | Note |
|-----------------|--------------|---------|-----------|------|
| **Vendite Retail** (stagionalità forte) | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **Prophet** | Holiday effects, multiple seasonalities |
| **Finanza/Trading** (alta frequenza) | ⭐⭐⭐⭐⭐ | ⭐⭐ | **ARIMA** | Volatility clustering, no clear seasonality |
| **IoT Sensors** (continuous data) | ⭐⭐⭐⭐ | ⭐⭐⭐ | **ARIMA** | Short-term correlations, noise |
| **Energy Consumption** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **Prophet** | Multiple seasonalities, weather effects |
| **Manufacturing KPIs** | ⭐⭐⭐⭐ | ⭐⭐⭐ | **ARIMA** | Process control, stationary |
| **Website Traffic** | ⭐⭐ | ⭐⭐⭐⭐⭐ | **Prophet** | Holiday effects, growth trends |
| **Medical Diagnostics** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | **ARIMA** | Precision critical, explainable |
| **Economic Indicators** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | **ARIMA** | Economic theory grounding |

### 🔧 Confronto Implementativo

| Aspetto | ARIMA/SARIMA | Prophet | Note |
|---------|--------------|---------|------|
| **Setup Time** | 🔴 Alto (parameter tuning) | 🟢 Basso (auto-config) | Prophet più user-friendly |
| **Compute Cost** | 🟢 Basso | 🟡 Medio | ARIMA più efficiente |
| **Memory Usage** | 🟢 Basso | 🔴 Alto | Prophet usa più RAM |
| **Scalabilità** | 🟢 Ottima | 🟡 Media | ARIMA scala meglio |
| **Maintenance** | 🔴 Alto | 🟢 Basso | Prophet più "fire-and-forget" |
| **Debug Difficulty** | 🔴 Alto | 🟢 Basso | Prophet più interpretabile |

### 📈 Performance Metrics Tipiche

| Metrica | Serie Finanziarie | Serie Retail | Serie IoT | 
|---------|-------------------|--------------|-----------|
| **ARIMA MAPE** | 8-15% | 15-25% | 5-12% |
| **Prophet MAPE** | 12-25% | 8-18% | 10-20% |
| **ARIMA Training Time** | 2-30 sec | 5-60 sec | 1-10 sec |
| **Prophet Training Time** | 10-120 sec | 30-300 sec | 15-180 sec |

---

## 🎯 Decision Framework: Quale Modello Scegliere?

### 🚀 Quick Decision Tree

```
📊 HAI DATI SERIE TEMPORALI?
├── ✅ SI
│   ├── 🎯 OBIETTIVO PRIMARIO?
│   │   ├── 💰 **Business Forecast** (vendite, marketing)
│   │   │   ├── 📅 Effetti festività/eventi importanti?
│   │   │   │   ├── ✅ SI → **PROPHET** 🏆
│   │   │   │   └── ❌ NO → ARIMA vs Prophet (A/B test)
│   │   │   └── 📈 Trend growth con saturazione?
│   │   │       ├── ✅ SI → **PROPHET** 🏆
│   │   │       └── ❌ NO → **ARIMA/SARIMA** 🏆
│   │   ├── 🔬 **Statistical Analysis** (ricerca, medicina)
│   │   │   └── **ARIMA/SARIMA** 🏆 (interpretabilità)
│   │   ├── ⚡ **Real-time/High-frequency** (trading, IoT)
│   │   │   └── **ARIMA** 🏆 (velocità)
│   │   └── 🤖 **Automated Pipeline** (ML production)
│   │       └── **PROPHET** 🏆 (minimal tuning)
│   └── 📊 CARATTERISTICHE DATI?
│       ├── ⏰ Frequenza giornaliera/settimanale + eventi → **PROPHET**
│       ├── 📈 Serie stazionaria + correlazioni lag → **ARIMA**
│       ├── 🔄 Multiple seasonalities → **PROPHET**
│       └── 📉 Volatility clustering → **ARIMA**
└── ❌ NO → Considera modelli ML classici
```

### 📋 Checklist Pre-Implementation

#### ✅ **Scegli ARIMA/SARIMA se:**
- [ ] Hai esperienza in econometria/statistica
- [ ] Serving real-time con latency <100ms
- [ ] Serie principalmente stazionaria
- [ ] Precisione matematica critica
- [ ] Budget computazionale limitato
- [ ] Spiegabilità del modello richiesta dal business
- [ ] Dati finanziari o economici

#### ✅ **Scegli Prophet se:**
- [ ] Team business-oriented senza background statistico
- [ ] Serie con trend growth e stagionalità multiple
- [ ] Eventi/festività influenzano significativamente i dati
- [ ] Hai >2 anni di dati storici
- [ ] Tolleranza per maggior compute cost
- [ ] Quick time-to-market prioritario
- [ ] Serie retail, marketing, web analytics

#### ✅ **Considera Ensemble se:**
- [ ] High-stakes forecasting (revenue critical)
- [ ] Hai risorse computazionali abbondanti
- [ ] Accuracy > interpretability
- [ ] Vuoi robustezza contro diversi pattern
- [ ] Team skills per modelli complessi

---

## 🏆 Raccomandazioni Finali

### 🎯 **Per Beginners**
1. **Inizia con Prophet** per learning curve più dolce
2. **Usa Auto-Selection** per ARIMA se hai background statistico
3. **Studia i residui** in entrambi i casi per validation
4. **A/B test** sempre i modelli sui tuoi dati specifici

### 🚀 **Per Advanced Users**
1. **Ensemble approach** per production critical systems
2. **Custom seasonalities** in Prophet per pattern business-specific
3. **Exogenous variables** con SARIMAX per fattori esterni
4. **Hierarchical forecasting** per serie correlate multiple

### 🛠️ **Setup Production-Ready**

```python
# Pipeline completo enterprise-grade
from arima_forecaster.core import ProphetForecaster, SARIMAForecaster
from arima_forecaster.evaluation import ModelEvaluator
from arima_forecaster.visualization import ForecastPlotter

class ProductionForecaster:
    """Forecasting engine production-ready."""
    
    def __init__(self):
        self.models = {
            'prophet': ProphetForecaster(yearly_seasonality='auto'),
            'sarima': SARIMAForecaster()  # Auto-fit with best params
        }
        self.evaluator = ModelEvaluator()
        self.plotter = ForecastPlotter()
        
    def forecast_with_validation(self, series, steps=30, validation_split=0.8):
        """Forecast con validation e model selection automatica."""
        
        # Split data
        split_idx = int(len(series) * validation_split)
        train, validation = series[:split_idx], series[split_idx:]
        
        results = {}
        
        # Test multiple models
        for name, model in self.models.items():
            try:
                # Training
                model.fit(train)
                
                # Validation forecast
                val_forecast = model.forecast(steps=len(validation))
                
                # Metrics
                metrics = self.evaluator.calculate_metrics(validation, val_forecast)
                
                # Production forecast
                model.fit(series)  # Refit on full data
                production_forecast = model.forecast(steps=steps)
                
                results[name] = {
                    'model': model,
                    'validation_mape': metrics['mape'],
                    'forecast': production_forecast,
                    'metrics': metrics
                }
                
            except Exception as e:
                print(f"Model {name} failed: {e}")
                continue
        
        # Select best model
        best_model_name = min(results.keys(), 
                             key=lambda x: results[x]['validation_mape'])
        
        best_result = results[best_model_name]
        
        print(f"🏆 Best model: {best_model_name} (MAPE: {best_result['validation_mape']:.2%})")
        
        return best_result['forecast'], results

# Uso in production
forecaster = ProductionForecaster()
forecast, all_results = forecaster.forecast_with_validation(your_series)
print(f"Production forecast ready: {len(forecast)} periods")
```

### 📚 **Learning Path Consigliato**

1. **Settimana 1-2**: Familiarizzare con Prophet su dati business
2. **Settimana 3-4**: Imparare ARIMA theory e diagnostics  
3. **Settimana 5-6**: Implementare auto-selection per entrambi
4. **Settimana 7-8**: Sperimentare ensemble e hybrid approaches
5. **Settimana 9+**: Specialized techniques (VAR, LSTM, etc.)

### 🎉 **Conclusioni**

**Non esiste un modello universalmente superiore.** La scelta tra Prophet e ARIMA/SARIMA dipende dal:

1. **Context**: Business vs. Statistical analysis
2. **Data**: Seasonality, trend, frequency, holidays
3. **Resources**: Compute, team skills, maintenance
4. **Requirements**: Accuracy vs. interpretability vs. speed

**Best practice**: Implementa entrambi, testa sui tuoi dati, usa ensemble per production critical applications.

La libreria `arima-forecaster` ti offre implementazioni ottimizzate di entrambi gli approcci con Auto-Selection per facilificare la scelta. **Start simple, iterate fast, scale smart!** 🚀

---

*📝 Documento aggiornato: Agosto 2024*  
*🔄 Prossimi aggiornamenti: Seasonal-Trend decomposition, Advanced ensemble methods*

```python
# Best practice SARIMA
from arima_forecaster.core import SARIMAModelSelector
from arima_forecaster.data import TimeSeriesPreprocessor

# Preprocessing robusto
preprocessor = TimeSeriesPreprocessor()
clean_series = preprocessor.preprocess_pipeline(
    series,
    handle_missing='interpolate',
    remove_outliers=True,
    outlier_method='iqr',
    make_stationary=True
)

# Auto-selection con validazione
selector = SARIMAModelSelector(
    max_p=3, max_d=2, max_q=3,
    max_P=2, max_D=1, max_Q=2,
    seasonal_periods=[7, 12],
    information_criterion='bic',  # Penalizza complessità
    validate_on_test=True
)

selector.search(clean_series)
best_sarima = selector.get_best_model()

# Diagnostica finale
diagnostics = best_sarima.get_diagnostics()
assert diagnostics['ljung_box_pvalue'] > 0.05, "Residui non indipendenti"
```

#### ❌ **Don'ts:**
- Non ignorare stazionarietà e diagnostica residui
- Non over-parametrizzare (evita order alti senza giustificazione)
- Non usare per serie con changepoint frequenti
- Non applicare su serie con missing data senza preprocessing

### 🚀 Per Prophet

#### ✅ **Do's:**
1. **Tuning stagionalità**: Adatta `yearly/weekly/daily_seasonality` ai dati
2. **Holiday customization**: Usa holiday nazionali e eventi custom
3. **Regressori esterni**: Aggiungi variabili esplicative rilevanti
4. **Uncertainty tuning**: Ajusta `interval_width` per business needs

```python
# Best practice Prophet
from arima_forecaster.core import ProphetModelSelector

# Custom holidays per business
custom_holidays = pd.DataFrame({
    'holiday': 'Black Friday',
    'ds': pd.to_datetime(['2023-11-24', '2024-11-29']),
    'lower_window': -1,  # Inizia 1 giorno prima
    'upper_window': 2    # Finisce 2 giorni dopo
})

# Auto-tuning con Bayesian optimization
selector = ProphetModelSelector(
    changepoint_prior_scales=[0.001, 0.01, 0.05, 0.1, 0.5],
    seasonality_modes=['additive', 'multiplicative'],
    scoring='mape',
    max_models=50
)

best_model, results = selector.search(
    series,
    method='bayesian',  # Ottimizzazione intelligente
    custom_holidays=custom_holidays
)

print(f"Migliori parametri: {selector.get_best_params()}")
print(f"Cross-validation MAPE: {selector.get_best_score():.3f}%")
```

#### ❌ **Don'ts:**
- Non ignorare la scelta tra crescita lineare/logistica
- Non usare default seasonality su dati high-frequency
- Non trascurare country_holidays per serie business
- Non over-tuning su dataset piccoli

---

## 🎓 Conclusioni e Raccomandazioni

### 🏆 **Quando NON c'è una scelta chiara**

Se l'analisi preliminare non dà un vincitore chiaro, la strategia migliore è:

1. **📊 A/B Testing**: Implementa entrambi e confronta su validation set
2. **🔄 Ensemble**: Combina i due approcci per robustezza
3. **📈 Business Context**: Considera requisiti non-tecnici (interpretabilità, velocità, risorse)

### 🎯 **Recap Decisionale Finale**

```
┌─ Serie < 100 osservazioni ──► ARIMA
├─ Dati finanziari/economici ──► ARIMA/SARIMA  
├─ Alta frequenza (min/sec) ────► ARIMA
├─ Controllo statistico rigoroso ► ARIMA/SARIMA
├─ Business analytics ─────────► Prophet
├─ E-commerce/retail ──────────► Prophet
├─ Marketing/web analytics ────► Prophet
├─ Serie con changepoint ──────► Prophet
├─ Holiday effects ────────────► Prophet
└─ Robustezza richiesta ───────► Prophet
```

### 🚀 **Next Steps**

Ora che hai una comprensione completa delle differenze, puoi:

1. **🧪 Sperimentare**: Usa gli esempi di codice sui tuoi dati
2. **📈 Auto-Selection**: Sfrutta `ARIMAModelSelector` e `ProphetModelSelector` 
3. **🔍 Ensemble**: Combina approcci per risultati ottimali
4. **📊 Monitoraggio**: Implementa pipeline di valutazione continua

La scelta ottimale dipende sempre dal **contesto specifico** - usa questa guida come framework, ma **valida sempre empiricamente** sui tuoi dati reali!

---

*Per ulteriori dettagli tecnici, consulta la documentazione specifica di ciascun modello nella cartella `docs/`.*