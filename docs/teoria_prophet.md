# Teoria Facebook Prophet - Guida Completa

## 📊 Introduzione a Prophet

Facebook Prophet è un modello di forecasting sviluppato da Meta (ex-Facebook) specificamente progettato per previsioni di serie temporali con forte stagionalità e trend che cambiano nel tempo. Rappresenta un approccio innovativo che combina semplicità d'uso con potenza predittiva avanzata.

### 🎯 Filosofia di Prophet

Prophet adotta un approccio **decomposizione + regressione**:
- **Decomposizione**: Separa la serie in componenti interpretabili (trend, stagionalità, festività)
- **Regressione**: Usa modelli robusti per ogni componente
- **Interpretabilità**: Ogni componente ha significato business chiaro

## 🧮 Formulazione Matematica

### Modello Base

La formulazione fondamentale di Prophet è:

```
y(t) = g(t) + s(t) + h(t) + εₜ
```

Dove:
- **g(t)**: Componente di trend (crescita)
- **s(t)**: Componente stagionale periodica
- **h(t)**: Effetti delle festività irregolari
- **εₜ**: Termine di errore (rumore)

### 1. Componente di Trend - g(t)

Prophet supporta due tipi di trend:

#### A) Trend Lineare
```
g(t) = (k + a(t)ᵀ δ) t + (m + a(t)ᵀ γ)
```

- **k**: Tasso di crescita base
- **δ**: Vettore dei cambiamenti nel tasso di crescita
- **m**: Offset iniziale
- **γ**: Vettore degli aggiustamenti offset
- **a(t)**: Vettore di indicatori per punti di cambiamento

#### B) Trend Logistico (Saturazione)
```
g(t) = C(t) / (1 + exp(-(k + a(t)ᵀ δ)(t - (m + a(t)ᵀ γ))))
```

- **C(t)**: Capacità portante (limite superiore)
- Altri parametri come nel trend lineare

### 2. Componente Stagionale - s(t)

Prophet usa **serie di Fourier** per modellare la stagionalità:

```
s(t) = Σₙ₌₁ᴺ [aₙ cos(2πnt/P) + bₙ sin(2πnt/P)]
```

- **P**: Periodo della stagionalità (365.25 per annuale, 7 per settimanale)
- **N**: Numero di termini di Fourier
- **aₙ, bₙ**: Coefficienti stimati dai dati

### 3. Effetti Festività - h(t)

Per ogni festività i:
```
h(t) = Σᵢ κᵢ · 1[t ∈ Dᵢ]
```

- **κᵢ**: Effetto della festività i
- **Dᵢ**: Insieme di date per la festività i
- **1[·]**: Funzione indicatrice

## 🔄 Algoritmo di Stima

### Approccio Bayesiano

Prophet utilizza inferenza bayesiana con prior informativi:

1. **Prior sui Parametri**:
   - Trend: `k ~ Normal(0, 5)`
   - Stagionalità: `aₙ, bₙ ~ Normal(0, σ²)`
   - Festività: `κᵢ ~ Normal(0, ν²)`

2. **Iperparametri di Regolarizzazione**:
   - `σ`: Controlla flessibilità stagionalità
   - `ν`: Controlla flessibilità festività  
   - `τ`: Controlla flessibilità trend

### Ottimizzazione MAP

Prophet trova la stima **Maximum A Posteriori**:

```
θ* = argmax_θ [log p(y|θ) + log p(θ)]
```

Implementata tramite L-BFGS per efficienza computazionale.

## 📈 Tipi di Stagionalità

### 1. Stagionalità Additiva
```
y(t) = trend + stagionalità + rumore
```

### 2. Stagionalità Moltiplicativa  
```
y(t) = trend × (1 + stagionalità) + rumore
```

### Selezione Automatica

Prophet sceglie automaticamente il tipo basandosi su:
- **Crescita della varianza** nel tempo → Moltiplicativa
- **Varianza costante** → Additiva

## 🎛️ Iperparametri Chiave

### Flessibilità del Trend

**changepoint_prior_scale** (default: 0.05)
- **Valori bassi** (0.001-0.01): Trend più rigido
- **Valori alti** (0.1-0.5): Trend più flessibile
- **Rischio overfitting**: Valori troppo alti

### Flessibilità Stagionalità

**seasonality_prior_scale** (default: 10.0)
- **Valori bassi** (0.01-1.0): Stagionalità regolare
- **Valori alti** (10-50): Stagionalità più variabile

### Flessibilità Festività

**holidays_prior_scale** (default: 10.0)
- Simile logica alla stagionalità
- Controlla quanto le festività possono deviare dalla norma

## 🌍 Gestione Festività

### Categorie di Festività

1. **Fisse**: Stessa data ogni anno (Natale: 25 dicembre)
2. **Mobili**: Data variabile (Pasqua, Ramadan)  
3. **Nazionali**: Specifiche per paese
4. **Custom**: Definite dall'utente

### Finestre Temporali

Prophet permette di definire **finestre di influenza**:
```python
christmas = pd.DataFrame({
    'holiday': 'christmas',
    'ds': pd.to_datetime(['2020-12-25', '2021-12-25']),
    'lower_window': -2,  # 2 giorni prima
    'upper_window': 1    # 1 giorno dopo
})
```

## ⚡ Vantaggi di Prophet

### 1. **Robustezza**
- Gestisce automaticamente valori mancanti
- Robusto agli outlier
- Non richiede preprocessing elaborato

### 2. **Interpretabilità**
- Componenti separabili e visualizzabili
- Trend chiaro e comprensibile
- Effetti festività quantificabili

### 3. **Scalabilità**
- Veloce anche su serie lunghe
- Parallelizzazione automatica
- API semplice e intuitiva

### 4. **Flessibilità**
- Gestisce trend complessi (non lineari)
- Multiple stagionalità simultanee
- Calendari festività personalizzabili

## 🎯 Quando Usare Prophet

### ✅ Casi Ideali

- **Serie con forte stagionalità**: Vendite retail, traffico web
- **Trend che cambiano**: Crescita aziendale, adoption nuovi prodotti
- **Effetti festività importanti**: E-commerce, settore hospitality
- **Dati business**: Serie con interpretazione economica chiara

### ❌ Limitazioni

- **Serie molto corte**: < 2 cicli stagionali completi
- **Stagionalità debole**: ARIMA potrebbe essere migliore
- **Relazioni multivariate**: VAR più appropriato
- **Alta frequenza**: Dati intra-giornalieri complessi

## 🔍 Diagnostica e Validazione

### Cross-Validation

Prophet implementa **time series cross-validation**:

```python
df_cv = cross_validation(model, 
                        initial='730 days',    # Training iniziale
                        period='180 days',     # Frequenza evaluation  
                        horizon='365 days')    # Orizzonte previsione
```

### Metriche di Performance

1. **MAPE**: Mean Absolute Percentage Error
2. **MAE**: Mean Absolute Error  
3. **RMSE**: Root Mean Square Error
4. **Coverage**: % di valori entro intervalli confidenza

### Analisi Residui

- **Normalità**: Q-Q plot dei residui
- **Autocorrelazione**: ACF dei residui
- **Eteroschedasticità**: Varianza costante nel tempo

## 🛠️ Implementazione Pratica

### Esempio Base

```python
from arima_forecaster.core import ProphetForecaster

# Crea e addestra
model = ProphetForecaster(
    growth='linear',
    yearly_seasonality=True,
    weekly_seasonality=True,
    country_holidays='IT'
)

model.fit(data)

# Previsioni
forecast = model.forecast(steps=30, confidence_level=0.95)
```

### Auto-Selection

```python
from arima_forecaster.core import ProphetModelSelector

selector = ProphetModelSelector(
    growth_types=['linear', 'logistic'],
    seasonality_modes=['additive', 'multiplicative'],
    country_holidays=['IT', 'US', None]
)

selector.search(data)
best_model = selector.get_best_model()
```

## 📚 Confronto con Altri Modelli

| Caratteristica | Prophet | ARIMA | SARIMA |
|---|---|---|---|
| **Stagionalità** | ✅ Automatica | ❌ Manuale | ✅ Manuale |
| **Trend non-lineare** | ✅ | ❌ | ❌ |
| **Festività** | ✅ Built-in | ❌ | ⚠️ Manuale |
| **Interpretabilità** | ✅ Alta | ⚠️ Media | ⚠️ Media |
| **Preprocessing** | ✅ Minimo | ❌ Richiesto | ❌ Richiesto |
| **Velocità** | ✅ Veloce | ⚠️ Media | ❌ Lenta |
| **Valori mancanti** | ✅ Automatico | ❌ Preprocessing | ❌ Preprocessing |

## 🎓 Approfondimenti Teorici

### Letteratura di Riferimento

1. **Taylor, S.J., Letham, B. (2018)**: "Forecasting at Scale" - Paper originale
2. **Hyndman, R.J. (2018)**: "Forecasting: Principles and Practice" - Cap. 12
3. **Prophet Documentation**: Guida ufficiale Meta

### Estensioni Avanzate

1. **Prophet+**: Integrazione con variabili esogene
2. **NeuralProphet**: Versione neural network
3. **Orbit**: Implementazione Bayesiana alternativa

---

## 🔗 Integrazione nel Framework

Prophet è completamente integrato in **ARIMA Forecaster** con:

- ✅ API unificata con ARIMA/SARIMA
- ✅ Selezione automatica parametri
- ✅ Visualizzazioni avanzate
- ✅ Report multi-formato  
- ✅ Dashboard interattiva
- ✅ Endpoint REST API

**→ Prossimo**: [Confronto Prophet vs ARIMA](prophet_vs_arima.md)