# Confronto Completo: ARIMA vs SARIMA vs SARIMAX

## Guida alla Scelta del Modello Ottimale per Serie Temporali

---

## 🎯 **Panoramica Generale**

Questa guida fornisce un confronto dettagliato tra i tre principali modelli della famiglia ARIMA per aiutarti a scegliere l'approccio più appropriato per il tuo caso specifico.

### 📊 **Quadro di Confronto Rapido**

| Caratteristica | ARIMA | SARIMA | SARIMAX |
|----------------|-------|--------|---------|
| **Stagionalità** | ❌ No | ✅ Sì | ✅ Sì |
| **Variabili Esogene** | ❌ No | ❌ No | ✅ Sì |
| **Complessità** | 🟢 Bassa | 🟡 Media | 🔴 Alta |
| **Parametri** | 3 (p,d,q) | 7 (p,d,q)(P,D,Q,s) | 7+ (p,d,q)(P,D,Q,s) + variabili |
| **Performance** | 🟢 Veloce | 🟡 Media | 🔴 Lenta |
| **Interpretabilità** | 🟢 Alta | 🟡 Media | 🔴 Bassa |
| **Accuratezza** | 🟡 Base | 🟢 Buona | 🟢 Eccellente |

---

## 📈 **ARIMA - AutoRegressive Integrated Moving Average**

### 🔍 **Quando Usare ARIMA**

**✅ Ideale per:**
- Serie temporali **non stagionali** 
- Dati con trend lineare o stazionari
- Forecasting a breve-medio termine (< 12 periodi)
- Prototipazione rapida e analisi esplorativa
- Casi con risorse computazionali limitate

**❌ Evitare quando:**
- Sono presenti pattern stagionali chiari
- Servono previsioni a lungo termine
- I dati mostrano forte influenza di fattori esterni

### ⚙️ **Struttura del Modello**

**Formula**: ARIMA(p,d,q)
- **p**: Ordine autoregressivo (AR) - dipendenza dai valori passati
- **d**: Grado di differenziazione (I) - per rendere stazionaria la serie
- **q**: Ordine media mobile (MA) - dipendenza dagli errori passati

### 📝 **Esempio Pratico: Vendite Giornaliere**

```python
from arima_forecaster import ARIMAForecaster, ARIMAModelSelector
import pandas as pd

# Dati senza stagionalità (es. vendite online daily)
data = pd.read_csv('vendite_daily.csv', index_col='data', parse_dates=True)
serie = data['vendite']

# Selezione automatica parametri
selector = ARIMAModelSelector()
selector.search(serie)
print(f"Modello ottimale: ARIMA{selector.best_order}")

# Addestramento e forecast
arima_model = ARIMAForecaster(order=selector.best_order)
arima_model.fit(serie)
forecast = arima_model.forecast(steps=7)  # 7 giorni
print(f"Previsioni settimanali: {forecast}")
```

### 📊 **Vantaggi e Svantaggi**

**➕ Vantaggi:**
- **Semplicità**: Solo 3 parametri da ottimizzare
- **Velocità**: Training e inference molto rapidi
- **Interpretabilità**: Coefficienti facilmente interpretabili
- **Stabilità**: Meno soggetto a overfitting
- **Memoria**: Richiede poca RAM

**➖ Svantaggi:**
- **Limitazioni stagionali**: Non cattura pattern ricorrenti
- **Accuratezza limitata**: Su serie complesse
- **Portata temporale**: Efficace solo nel breve termine

---

## 🌊 **SARIMA - Seasonal ARIMA**

### 🔍 **Quando Usare SARIMA**

**✅ Ideale per:**
- Serie con **stagionalità chiara** (mensile, trimestrale, annuale)
- Dati business con pattern ricorrenti (vendite retail, turismo)
- Forecasting medio-lungo termine (12-24+ periodi)
- Analisi di decomposizione stagionale
- Settori con ciclicità naturale

**❌ Evitare quando:**
- La serie non mostra stagionalità
- Fattori esterni influenzano fortemente i risultati
- Servono previsioni in tempo reale

### ⚙️ **Struttura del Modello**

**Formula**: SARIMA(p,d,q)(P,D,Q,s)
- **Componenti non stagionali**: (p,d,q) come in ARIMA
- **Componenti stagionali**: 
  - **P**: AR stagionale
  - **D**: Differenziazione stagionale
  - **Q**: MA stagionale
  - **s**: Periodo stagionale (12 per dati mensili, 4 per trimestrali)

### 📝 **Esempio Pratico: Vendite Mensili Retail**

```python
from arima_forecaster import SARIMAForecaster, SARIMAModelSelector

# Dati con forte stagionalità mensile
data_monthly = pd.read_csv('vendite_retail_monthly.csv', 
                          index_col='mese', parse_dates=True)
serie_mensile = data_monthly['vendite']

# Selezione automatica con focus sulla stagionalità
selector = SARIMAModelSelector(
    seasonal_periods=[12],  # Stagionalità annuale
    max_models=50
)
selector.search(serie_mensile)

# Risultati selezione
print(f"Modello ottimale: SARIMA{selector.best_order}x{selector.best_seasonal_order}")
print(f"AIC: {selector.best_score}")

# Addestramento modello
sarima_model = SARIMAForecaster(
    order=selector.best_order,
    seasonal_order=selector.best_seasonal_order
)
sarima_model.fit(serie_mensile)

# Decomposizione stagionale
decomposizione = sarima_model.get_seasonal_decomposition()
print("Componenti:", list(decomposizione.keys()))

# Forecast con stagionalità
forecast_12_mesi = sarima_model.forecast(
    steps=12, 
    confidence_intervals=True
)
print("Forecast annuale:", forecast_12_mesi['forecast'])
```

### 📊 **Vantaggi e Svantaggi**

**➕ Vantaggi:**
- **Cattura stagionalità**: Modella pattern ricorrenti automaticamente
- **Accuratezza superiore**: Su dati stagionali rispetto ad ARIMA
- **Decomposizione**: Separa trend, stagionalità e rumore
- **Flessibilità**: Supporta multiple stagionalità
- **Long-term forecasting**: Eccellente per previsioni a lungo termine

**➖ Svantaggi:**
- **Complessità**: 7 parametri da ottimizzare
- **Tempo training**: Più lento di ARIMA
- **Overfitting**: Rischio maggiore con serie corte
- **Interpretabilità**: Più difficile interpretare coefficienti stagionali

---

## 🌐 **SARIMAX - SARIMA with eXogenous Variables**

### 🔍 **Quando Usare SARIMAX**

**✅ Ideale per:**
- Serie influenzate da **fattori esterni** (meteo, economia, marketing)
- Business con variabili controllabili (prezzo, promozioni, advertising)
- Scenari "what-if" e simulazioni
- Forecasting con informazioni future disponibili
- Analisi causale e attribution

**❌ Evitare quando:**
- Non hai variabili esogene affidabili
- Le variabili esterne non sono predittive
- Mancano dati futuri per le variabili esogene
- Serve interpretabilità semplice

### ⚙️ **Struttura del Modello**

**Formula**: SARIMAX(p,d,q)(P,D,Q,s) + X(k)
- **Componenti SARIMA**: Come SARIMA standard
- **Variabili esogene**: k variabili esterne che influenzano la serie target
- **Coefficienti esogeni**: Peso e significatività di ogni variabile

### 📝 **Esempio Pratico: Vendite E-commerce con Marketing**

```python
from arima_forecaster import SARIMAXForecaster, SARIMAXModelSelector
import pandas as pd

# Dati con serie target e variabili esogene
vendite_data = pd.read_csv('vendite_ecommerce.csv', index_col='data', parse_dates=True)
serie_vendite = vendite_data['vendite']

# Variabili esogene che influenzano le vendite
variabili_esogene = pd.DataFrame({
    'spesa_marketing': vendite_data['marketing_budget'],
    'temperatura_media': vendite_data['temperature'],
    'indice_economico': vendite_data['economic_index'],
    'promozioni_attive': vendite_data['active_promotions']
})

print("Variabili esogene:", list(variabili_esogene.columns))
print("Correlazioni con vendite:")
for col in variabili_esogene.columns:
    corr = serie_vendite.corr(variabili_esogene[col])
    print(f"  {col}: {corr:.3f}")

# Selezione automatica SARIMAX
selector = SARIMAXModelSelector(
    seasonal_periods=[12],
    exog_names=list(variabili_esogene.columns)
)
selector.search(serie_vendite, exog=variabili_esogene)

# Miglior modello
sarimax_model = SARIMAXForecaster(
    order=selector.best_order,
    seasonal_order=selector.best_seasonal_order,
    exog_names=list(variabili_esogene.columns)
)
sarimax_model.fit(serie_vendite, exog=variabili_esogene)

# Analisi importanza variabili esogene
importance = sarimax_model.get_exog_importance()
print("\nImportanza variabili esogene:")
for _, row in importance.iterrows():
    significance = "✅ Significativa" if row['significant'] else "❌ Non significativa"
    print(f"  {row['variable']}: coeff={row['coefficient']:.4f}, p-value={row['pvalue']:.4f} {significance}")

# Forecast con scenari futuri
# Scenario 1: Incremento marketing del 20%
scenario_marketing_up = pd.DataFrame({
    'spesa_marketing': [1200, 1300, 1400, 1500] * 1.2,  # +20%
    'temperatura_media': [22, 25, 28, 30],  # Dati previsti
    'indice_economico': [105, 106, 107, 108],  # Trend crescente
    'promozioni_attive': [1, 1, 0, 1]  # Piano promozionale
})

forecast_scenario1 = sarimax_model.forecast(
    steps=4,
    exog_future=scenario_marketing_up,
    confidence_intervals=True
)

# Scenario 2: Marketing invariato
scenario_marketing_baseline = pd.DataFrame({
    'spesa_marketing': [1200, 1300, 1400, 1500],  # Baseline
    'temperatura_media': [22, 25, 28, 30],
    'indice_economico': [105, 106, 107, 108],
    'promozioni_attive': [1, 1, 0, 1]
})

forecast_scenario2 = sarimax_model.forecast(
    steps=4,
    exog_future=scenario_marketing_baseline
)

# Confronto scenari
print("\nConfronto scenari:")
print(f"Scenario Marketing +20%: {forecast_scenario1['forecast'].mean():.0f}")
print(f"Scenario Baseline:      {forecast_scenario2['forecast'].mean():.0f}")
incremento = forecast_scenario1['forecast'].mean() - forecast_scenario2['forecast'].mean()
print(f"Incremento assoluto:    {incremento:.0f} (+{incremento/forecast_scenario2['forecast'].mean()*100:.1f}%)")
```

### 📊 **Vantaggi e Svantaggi**

**➕ Vantaggi:**
- **Accuratezza massima**: Incorpora informazioni esterne rilevanti
- **Scenario analysis**: Simulazioni what-if con variabili controllabili
- **Business insight**: Quantifica impatto di fattori esterni
- **Causalità**: Identifica drivers significativi
- **Flessibilità**: Adattabile a contesti complessi

**➖ Svantaggi:**
- **Complessità elevata**: Molti parametri e validazioni necessarie
- **Dipendenza dati**: Richiede variabili esogene future affidabili
- **Overfitting**: Alto rischio con troppe variabili
- **Interpretabilità**: Difficile spiegare interazioni complesse
- **Performance**: Lento su dataset grandi

---

## 🎯 **Guida alla Scelta del Modello**

### 🤔 **Flowchart Decisionale**

```
Hai dati con stagionalità evidente?
├── No → La serie ha trend o è stazionaria?
│   ├── Sì → ARIMA (semplice e veloce)
│   └── No → Considera preprocessing o altri modelli
│
└── Sì → Hai variabili esogene significative?
    ├── No → SARIMA (cattura stagionalità)
    └── Sì → Hai previsioni future per le variabili esogene?
        ├── No → SARIMA (più sicuro)
        └── Sì → SARIMAX (massima accuratezza)
```

### 📋 **Matrice di Decisione per Settore**

| Settore/Caso d'Uso | Modello Raccomandato | Motivazione |
|-------------------|---------------------|-------------|
| **E-commerce Daily Sales** | SARIMA | Stagionalità settimanale/mensile forte |
| **Retail con Marketing** | SARIMAX | Promozioni e spesa pubblicitaria misurabili |
| **Servizi Finanziari** | ARIMA/SARIMA | Focus su trend, stagionalità limitata |
| **Energia/Utilities** | SARIMAX | Meteo e fattori esterni critici |
| **Manufacturing** | SARIMAX | Materie prime, domanda, capacità |
| **Travel/Tourism** | SARIMA | Stagionalità molto forte |
| **Food & Beverage** | SARIMAX | Meteo, eventi, marketing |
| **Healthcare** | SARIMA | Pattern stagionali (influenza, allergie) |
| **Real Estate** | SARIMAX | Tassi interesse, economia, demografia |
| **Logistica** | ARIMA | Trend lineari, meno stagionalità |

### ⚖️ **Criteri di Valutazione Comparativa**

#### 🎯 **Accuratezza**
```python
# Confronto sistematico delle performance
from arima_forecaster.evaluation import ModelEvaluator
evaluator = ModelEvaluator()

# Test su stesso dataset
results = {}
for model_name, model in [('ARIMA', arima_model), 
                         ('SARIMA', sarima_model), 
                         ('SARIMAX', sarimax_model)]:
    predictions = model.predict(steps=len(test_data))
    metrics = evaluator.calculate_forecast_metrics(test_data, predictions)
    results[model_name] = metrics

# Classifica modelli
print("Classifica per MAPE (Mean Absolute Percentage Error):")
sorted_models = sorted(results.items(), key=lambda x: x[1]['mape'])
for i, (name, metrics) in enumerate(sorted_models, 1):
    print(f"{i}. {name}: MAPE={metrics['mape']:.2f}%, RMSE={metrics['rmse']:.2f}")
```

#### ⏱️ **Performance Computazionale**
```python
import time

# Benchmark training time
models_benchmark = {
    'ARIMA': lambda: ARIMAForecaster((2,1,2)).fit(serie),
    'SARIMA': lambda: SARIMAForecaster((1,1,1), (1,1,1,12)).fit(serie),
    'SARIMAX': lambda: SARIMAXForecaster((1,1,1), (1,1,1,12), ['temp']).fit(serie, exog=temp_data)
}

print("Benchmark Training Time:")
for name, train_func in models_benchmark.items():
    start_time = time.time()
    train_func()
    training_time = time.time() - start_time
    print(f"{name}: {training_time:.2f} seconds")
```

### 💡 **Best Practices per Modello**

#### 🎯 **ARIMA Best Practices**
- **Stazionarietà**: Sempre verificare con test ADF/KPSS prima del training
- **Differenziazione**: Limitare d≤2 per evitare over-differencing
- **Parsimonia**: Preferire modelli semplici (p+q≤5)
- **Validazione**: Cross-validation temporale essenziale

#### 🌊 **SARIMA Best Practices**
- **Periodo stagionale**: Verificare con analisi spettrale o ACF
- **Stagionalità multipla**: Considerare modelli nested per stagionalità complesse
- **Selezione parametri**: Usare grid search limitato per evitare overfitting
- **Diagnostica**: Controllare residui sia per componenti regolari che stagionali

#### 🌐 **SARIMAX Best Practices**
- **Selezione variabili**: Analisi di correlazione e significatività statistica
- **Multicollinearità**: Evitare variabili esogene correlate tra loro
- **Validazione out-of-sample**: Essenziale data la complessità
- **Scenario planning**: Preparare multipli scenari per le variabili future
- **Preprocessing esogene**: Standardizzazione/normalizzazione spesso necessaria

---

## 📊 **Caso Studio Comparativo Completo**

### 🏪 **Dataset: Vendite Retail Chain**

Confrontiamo i tre modelli su un dataset realistico di vendite mensili di una catena retail, con:
- **Serie target**: Vendite mensili (36 mesi)
- **Stagionalità**: Pattern natalizi e estivi
- **Variabili esogene**: Spesa marketing, temperatura, indice economico

```python
import pandas as pd
import numpy as np
from arima_forecaster import ARIMAForecaster, SARIMAForecaster, SARIMAXForecaster
from arima_forecaster.evaluation import ModelEvaluator
from arima_forecaster.visualization import ForecastPlotter

# Carica dataset
data = pd.read_csv('retail_sales_case_study.csv', index_col='month', parse_dates=True)
vendite = data['sales']
exog_vars = data[['marketing_spend', 'temperature', 'economic_index']]

# Split temporale: 30 mesi training, 6 mesi test
split_date = vendite.index[-6]
train_sales = vendite[:split_date]
test_sales = vendite[split_date:]
train_exog = exog_vars[:split_date]
test_exog = exog_vars[split_date:]

print(f"Training set: {len(train_sales)} mesi")
print(f"Test set: {len(test_sales)} mesi")

# 1. ARIMA Model
arima = ARIMAForecaster(order=(2,1,2))
arima.fit(train_sales)
arima_forecast = arima.forecast(steps=6)

# 2. SARIMA Model  
sarima = SARIMAForecaster(order=(1,1,1), seasonal_order=(1,1,1,12))
sarima.fit(train_sales) 
sarima_forecast = sarima.forecast(steps=6)

# 3. SARIMAX Model
sarimax = SARIMAXForecaster(
    order=(1,1,1), 
    seasonal_order=(1,1,1,12),
    exog_names=['marketing_spend', 'temperature', 'economic_index']
)
sarimax.fit(train_sales, exog=train_exog)
sarimax_forecast = sarimax.forecast(steps=6, exog_future=test_exog)

# Valutazione comparativa
evaluator = ModelEvaluator()
models_results = {}

for name, forecast in [('ARIMA', arima_forecast), 
                      ('SARIMA', sarima_forecast), 
                      ('SARIMAX', sarimax_forecast)]:
    metrics = evaluator.calculate_forecast_metrics(
        actual=test_sales, 
        predicted=forecast if isinstance(forecast, pd.Series) else forecast['forecast']
    )
    models_results[name] = metrics

# Tabella comparativa risultati
print("\n📊 RISULTATI CONFRONTO MODELLI")
print("="*50)
print(f"{'Modello':<10} {'MAPE':<8} {'RMSE':<8} {'MAE':<8} {'R²':<8}")
print("-"*50)

for model, metrics in models_results.items():
    print(f"{model:<10} {metrics['mape']:<8.2f} {metrics['rmse']:<8.2f} "
          f"{metrics['mae']:<8.2f} {metrics['r_squared']:<8.3f}")

# Identifica il migliore
best_model = min(models_results.items(), key=lambda x: x[1]['mape'])
print(f"\n🏆 Modello vincente: {best_model[0]} (MAPE: {best_model[1]['mape']:.2f}%)")

# Visualizzazione comparativa
plotter = ForecastPlotter()
plt.figure(figsize=(15, 10))

# Plot 1: Serie originale e forecasts
plt.subplot(2, 2, 1)
plt.plot(train_sales.index, train_sales.values, label='Training', color='blue')
plt.plot(test_sales.index, test_sales.values, label='Actual Test', color='black', linewidth=2)
plt.plot(test_sales.index, arima_forecast, label='ARIMA', linestyle='--', alpha=0.8)
plt.plot(test_sales.index, sarima_forecast, label='SARIMA', linestyle='--', alpha=0.8)
plt.plot(test_sales.index, sarimax_forecast['forecast'] if isinstance(sarimax_forecast, dict) else sarimax_forecast, 
         label='SARIMAX', linestyle='--', alpha=0.8)
plt.title('Confronto Forecasts')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Errori percentuali
plt.subplot(2, 2, 2)
for name, forecast in [('ARIMA', arima_forecast), 
                      ('SARIMA', sarima_forecast), 
                      ('SARIMAX', sarimax_forecast)]:
    forecast_values = forecast if isinstance(forecast, pd.Series) else forecast['forecast']
    errors = ((forecast_values - test_sales) / test_sales) * 100
    plt.plot(test_sales.index, errors, marker='o', label=f'{name}')
plt.title('Errori Percentuali per Periodo')
plt.ylabel('Errore %')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)

# Plot 3: Distribuzione errori
plt.subplot(2, 2, 3)
errors_data = []
model_names = []
for name, forecast in [('ARIMA', arima_forecast), 
                      ('SARIMA', sarima_forecast), 
                      ('SARIMAX', sarimax_forecast)]:
    forecast_values = forecast if isinstance(forecast, pd.Series) else forecast['forecast']
    errors = np.abs(((forecast_values - test_sales) / test_sales) * 100)
    errors_data.extend(errors)
    model_names.extend([name] * len(errors))

plt.boxplot([models_results['ARIMA']['absolute_errors'],
            models_results['SARIMA']['absolute_errors'], 
            models_results['SARIMAX']['absolute_errors']], 
           labels=['ARIMA', 'SARIMA', 'SARIMAX'])
plt.title('Distribuzione Errori Assoluti')
plt.ylabel('Errore Assoluto')

# Plot 4: Metriche radar chart (se disponibili)
plt.subplot(2, 2, 4)
metrics_names = ['MAPE', 'RMSE', 'MAE']
arima_scores = [models_results['ARIMA'][m.lower()] for m in ['mape', 'rmse', 'mae']]
sarima_scores = [models_results['SARIMA'][m.lower()] for m in ['mape', 'rmse', 'mae']]  
sarimax_scores = [models_results['SARIMAX'][m.lower()] for m in ['mape', 'rmse', 'mae']]

x = np.arange(len(metrics_names))
width = 0.25

plt.bar(x - width, arima_scores, width, label='ARIMA', alpha=0.8)
plt.bar(x, sarima_scores, width, label='SARIMA', alpha=0.8)
plt.bar(x + width, sarimax_scores, width, label='SARIMAX', alpha=0.8)

plt.xlabel('Metriche')
plt.ylabel('Valore')
plt.title('Confronto Metriche Performance')
plt.xticks(x, metrics_names)
plt.legend()

plt.tight_layout()
plt.savefig('confronto_modelli_arima_completo.png', dpi=300, bbox_inches='tight')
print(f"\n📈 Grafici salvati: confronto_modelli_arima_completo.png")

# Conclusioni e raccomandazioni
print(f"\n📋 CONCLUSIONI E RACCOMANDAZIONI")
print("="*50)
print(f"Dataset: {len(vendite)} osservazioni mensili con stagionalità")
print(f"Best performer: {best_model[0]} con MAPE {best_model[1]['mape']:.2f}%")

if best_model[0] == 'ARIMA':
    print("💡 ARIMA ha vinto: serie probabilmente con trend semplice, stagionalità debole")
elif best_model[0] == 'SARIMA':
    print("💡 SARIMA ha vinto: stagionalità significativa, variabili esogene non decisive")
else:
    print("💡 SARIMAX ha vinto: variabili esogene forniscono valore predittivo significativo")

print(f"\nVariabili esogene più importanti:")
if 'sarimax' in locals():
    importance = sarimax.get_exog_importance()
    for _, row in importance.head(3).iterrows():
        impact = "Alto" if abs(row['coefficient']) > 0.5 else "Medio" if abs(row['coefficient']) > 0.1 else "Basso"
        print(f"  • {row['variable']}: {row['coefficient']:.3f} (impatto {impact})")
```

### 📈 **Interpretazione Risultati**

I risultati del caso studio mostrano tipicamente:

1. **Se ARIMA vince**: La serie ha trend semplici senza componenti stagionali forti
2. **Se SARIMA vince**: La stagionalità è il driver principale, fattori esterni meno importanti  
3. **Se SARIMAX vince**: Variabili esogene aggiungono valore predittivo significativo

---

## 🎓 **Conclusioni e Raccomandazioni Finali**

### 🎯 **Principi Guida per la Scelta**

1. **Inizia Semplice**: Prova sempre ARIMA prima come baseline
2. **Analizza Stagionalità**: Se presente, SARIMA è spesso la scelta giusta
3. **Valuta Variabili Esterne**: SARIMAX solo se hai variabili predittive affidabili
4. **Testa Empiricamente**: Non affidarti solo alla teoria, valida con dati reali
5. **Considera il Contesto**: Complessità vs accuratezza vs interpretabilità

### ⚡ **Quick Decision Framework**

```python
def suggerisci_modello(serie, ha_stagionalita=None, variabili_esogene=None):
    """
    Framework decisionale automatico per scelta modello ARIMA
    """
    # Test stagionalità automatico se non specificato
    if ha_stagionalita is None:
        from scipy import stats
        # Test semplificato: varianza per stagione
        if len(serie) >= 24:  # Almeno 2 anni di dati
            stagioni = [serie[i::12] for i in range(12)]
            _, p_value = stats.kruskal(*[s.dropna() for s in stagioni if len(s.dropna()) > 2])
            ha_stagionalita = p_value < 0.05
        else:
            ha_stagionalita = False
    
    # Valuta correlazione variabili esogene
    exog_significative = False
    if variabili_esogene is not None:
        correlazioni = [abs(serie.corr(var)) for var in variabili_esogene.T]
        exog_significative = any(corr > 0.3 for corr in correlazioni)
    
    # Decisione
    if not ha_stagionalita:
        return "ARIMA", "Serie senza stagionalità evidente"
    elif not exog_significative:
        return "SARIMA", "Serie stagionale, variabili esogene poco correlate"  
    else:
        return "SARIMAX", "Serie stagionale con variabili esogene significative"

# Esempio utilizzo
raccomandazione, motivazione = suggerisci_modello(
    serie=vendite, 
    variabili_esogene=exog_vars
)
print(f"💡 Raccomandazione: {raccomandazione}")
print(f"📋 Motivazione: {motivazione}")
```

### 🚀 **Prossimi Passi**

Dopo aver letto questa guida:

1. **Sperimenta**: Testa tutti e tre i modelli sui tuoi dati
2. **Documenta**: Mantieni record delle performance per domini diversi
3. **Automatizza**: Implementa pipeline che testino multiple opzioni
4. **Approfondisci**: Studia modelli ibridi e ensemble methods
5. **Contribuisci**: Condividi i tuoi risultati con la community

---

### 📚 **Risorse Aggiuntive**

- **[Teoria ARIMA](teoria_arima.md)** - Fondamenti matematici 
- **[Teoria SARIMA](teoria_sarima.md)** - Stagionalità e implementazione
- **[Teoria SARIMAX](teoria_sarimax.md)** - Variabili esogene e validazione
- **[Guida Utente](guida_utente.md)** - Esempi pratici completi
- **[API Reference](../README.md)** - Documentazione tecnica completa

---

*📝 Documento aggiornato per ARIMA Forecaster v0.4.0 - Gennaio 2025*