# ARIMA Forecaster 🚀

## Libreria Avanzata per Forecasting Serie Temporali con Modelli ARIMA, SARIMA, SARIMAX, VAR e Prophet

Una libreria Python professionale e completa per l'analisi, modellazione e previsione di serie temporali utilizzando modelli ARIMA, SARIMA (Seasonal ARIMA), SARIMAX (con variabili esogene), VAR (Vector Autoregression) e **Facebook Prophet**. Include funzionalità avanzate di Auto-ML, API REST, dashboard interattiva multilingue (5 lingue), sistema traduzioni centralizzato e ottimizzazione automatica dei parametri per applicazioni enterprise-grade.

---

### 🌟 **Nuove Funzionalità Avanzate**

- **🚀 GPU/CUDA Acceleration**: Supporto GPU nativo per training parallelo ad alta velocità (5-15x speedup)
- **⚙️ Configuration Management**: Sistema configurazione avanzato con .env file e auto-detection hardware
- **📈 Facebook Prophet**: Modelli avanzati per serie con stagionalità complessa e festività
- **🔥 Cold Start Problem**: Transfer Learning per forecasting di nuovi prodotti senza dati storici
- **🌊 Modelli SARIMA**: Gestione completa della stagionalità con parametri (P,D,Q,s)
- **🌐 Modelli SARIMAX**: Modelli con variabili esogene per incorporare fattori esterni
- **⭐ Advanced Exog Handling**: Selezione automatica feature, preprocessing intelligente, diagnostica
- **📊 Modelli VAR**: Forecasting multivariato con analisi di causalità e impulse response
- **🤖 Auto-ML**: Ottimizzazione automatica con Optuna, Hyperopt e Scikit-Optimize  
- **🌐 API REST**: Servizi di forecasting production-ready con FastAPI multilingue
- **💻 Dashboard Streamlit**: Interfaccia web interattiva multilingue (IT, EN, ES, FR, ZH)
- **🌍 Sistema Traduzioni**: Gestione centralizzata traduzioni per 5 lingue
- **📄 Report Quarto**: Generazione report dinamici multilingue con analisi automatiche
- **🎯 Ensemble Methods**: Combinazione intelligente di modelli diversi
- **⚡ GPU Parallel Processing**: Training fino a 500+ modelli in parallelo su GPU NVIDIA
- **🏭 Inventory Optimization**: Sistema avanzato ottimizzazione magazzino enterprise-grade
- **⏰ Slow/Fast Moving**: Gestione differenziata prodotti bassa/alta rotazione
- **📦 Perishable/FEFO**: Gestione prodotti deperibili con markdown automatico
- **🏢 Multi-Echelon**: Ottimizzazione inventory reti distribuite multi-livello
- **⚖️ Capacity Constraints**: Gestione vincoli capacità (volume, budget, pallet)
- **🔧 Kitting/Bundle**: Ottimizzazione kit vs componenti separati

### ✨ **Caratteristiche Core**

- **🎯 Selezione Automatica Modello**: Grid search intelligente per trovare parametri ottimali
- **🔧 Preprocessing Avanzato**: Gestione valori mancanti, rimozione outlier, test stazionarietà
- **📊 Valutazione Completa**: 15+ metriche accuratezza, diagnostica residui, test statistici
- **📈 Visualizzazioni Professionali**: Dashboard interattivi, grafici con intervalli confidenza
- **⚡ Gestione Errori Robusta**: Eccezioni personalizzate e logging configurabile
- **🧪 Testing Estensivo**: Suite test completa con alta coverage
- **📚 Documentazione Completa**: Guide teoriche e pratiche in italiano

---

### 🏗️ **Architettura Modulare Avanzata**

```
├── src/arima_forecaster/           # Package principale
│   ├── config/                     # 🚀 Sistema configurazione avanzato
│   │   ├── settings.py            # Configuration management con .env support
│   │   ├── gpu_config.py          # GPU detection e ottimizzazione automatica
│   │   └── __init__.py            # Configurazioni globali
│   ├── core/                       # Modelli ARIMA, SARIMA, SARIMAX, VAR, Prophet e GPU-accelerated
│   │   ├── arima_model.py         # Implementazione ARIMA base
│   │   ├── sarima_model.py        # Modelli SARIMA con stagionalità
│   │   ├── sarimax_model.py       # Modelli SARIMAX con variabili esogene
│   │   ├── sarimax_auto_selector.py  # ⭐ Advanced Exog Handling con auto feature selection
│   │   ├── gpu_model_selector.py  # 🚀 GPU-accelerated ARIMA/SARIMA selectors
│   │   ├── var_model.py           # Vector Autoregression multivariato
│   │   ├── prophet_model.py       # 📈 Facebook Prophet per serie con trend complessi
│   │   ├── prophet_selection.py   # 📈 Selezione automatica Prophet
│   │   ├── cold_start.py          # 🔥 Cold Start Problem - Transfer Learning per nuovi prodotti
│   │   ├── model_selection.py     # Selezione automatica ARIMA
│   │   ├── sarima_selection.py    # Selezione automatica SARIMA
│   │   └── sarimax_selection.py   # Selezione automatica SARIMAX
│   ├── data/                       # Caricamento dati e preprocessing
│   ├── evaluation/                 # Metriche valutazione e diagnostica
│   ├── visualization/              # Grafici e dashboard avanzati
│   ├── reporting/                  # Sistema reporting Quarto dinamico
│   │   ├── generator.py           # Generatore report con template automatici
│   │   └── __init__.py            # Import opzionali per reporting
│   ├── api/                        # REST API con FastAPI
│   │   ├── main.py                # Applicazione API principale
│   │   ├── models.py              # Modelli Pydantic per validazione
│   │   └── services.py            # Servizi di business logic
│   ├── dashboard/                  # Dashboard interattiva Streamlit
│   ├── automl/                     # Auto-ML e ottimizzazione avanzata
│   │   ├── optimizer.py           # Ottimizzatori con Optuna/Hyperopt
│   │   └── tuner.py               # Hyperparameter tuning avanzato
│   ├── inventory/                   # 🏭 Sistema Ottimizzazione Magazzino Enterprise
│   │   └── balance_optimizer.py    # Bilanciamento scorte: Slow/Fast, Perishable, Multi-Echelon, Capacity, Kitting
│   ├── utils/                       # Logging, eccezioni, traduzioni, GPU e Advanced Exog Utils
│   │   ├── gpu_utils.py            # 🚀 GPU/CUDA utilities e array management
│   │   ├── translations.py         # Sistema traduzioni centralizzato multilingue
│   │   ├── preprocessing.py        # ⭐ Preprocessing avanzato variabili esogene
│   │   ├── exog_diagnostics.py     # ⭐ Diagnostica completa variabili esogene
│   │   ├── logger.py               # Logging configurabile
│   │   └── exceptions.py           # Eccezioni personalizzate
│   └── assets/                      # Risorse static del progetto
│       └── locales/                 # File traduzioni JSON (5 lingue)
├── docs/                           # Documentazione completa
│   ├── GPU_SETUP.md               # 🚀 Setup completo GPU/CUDA acceleration
│   ├── teoria_arima.md            # Teoria matematica ARIMA
│   ├── teoria_sarima.md           # Teoria matematica SARIMA
│   ├── teoria_sarimax.md          # Teoria matematica SARIMAX
│   ├── teoria_prophet.md          # 📈 Teoria matematica Facebook Prophet
│   ├── guida_prophet.md           # 📈 Guida pratica uso Prophet
│   ├── arima_vs_sarima.md         # Confronto dettagliato modelli
│   ├── sarima_vs_sarimax.md       # Confronto SARIMA vs SARIMAX
│   ├── prophet_vs_arima.md        # 📈 Confronto Prophet vs ARIMA
│   ├── slow_fast_moving_theory.md # 🏭 Teoria e pratica gestione Slow/Fast Moving
│   └── inventory_optimization_guide.md # 🏭 Guida completa ottimizzazione magazzino
├── .env.example                   # 🚀 Template configurazione completo per GPU/CPU
├── examples/                       # Script esempio pratici
│   ├── advanced_forecasting_showcase.py  # Demo funzionalità avanzate
│   ├── sarimax_example.py         # Esempio completo modelli SARIMAX
│   ├── moretti/                   # ⭐ Caso pratico completo Moretti S.p.A.
│   │   ├── test_advanced_exog_handling.py  # ⭐ Demo Advanced Exog Handling completo
│   │   └── moretti_dashboard.py   # Dashboard multilingue sistema medicale
│   ├── slow_fast_moving_demo.py   # 🏭 Demo classificazione e ottimizzazione Slow/Fast Moving
│   ├── advanced_features_demo.py  # 🏭 Demo completa 4 casistiche avanzate inventory
│   └── forecasting_base.py        # Esempi base ARIMA/SARIMA
├── notebooks/                      # Jupyter notebooks per ricerca e sviluppo
│   └── research_and_development.ipynb # Ambiente R&D per sperimentazione algoritmi
├── scripts/                        # Script di utilità
│   ├── run_api.py                 # Lancia API server
│   └── run_dashboard.py           # Lancia dashboard Streamlit
├── test_sarimax_api.py             # Test script per SARIMAX API
├── tests/                          # Suite test completa
│   ├── test_arima_model.py        # Test modelli ARIMA
│   ├── test_sarima_model.py       # Test modelli SARIMA  
│   ├── test_sarimax_model.py      # Test modelli SARIMAX
│   ├── test_var_model.py          # Test modelli VAR
│   └── test_api.py                # Test API REST
└── outputs/                        # Output generati
    ├── models/                    # Modelli salvati e metadata
    ├── plots/                     # Visualizzazioni generate
    └── reports/                   # Report Quarto in HTML/PDF/DOCX
```

---

### 🚀 **Installazione e Setup**

#### Opzione 1: Con UV (Raccomandato - 10x più veloce) ⚡

```bash
# Installa uv se non ce l'hai già
curl -LsSf https://astral.sh/uv/install.sh | sh
# oppure per Windows: winget install --id=astral-sh.uv

# Clona il repository
git clone https://github.com/tuonome/arima-forecaster.git
cd arima-forecaster

# Installa con tutte le funzionalità avanzate
uv sync --all-extras

# Attiva ambiente virtuale
source .venv/bin/activate  # Linux/macOS
# oppure: .venv\Scripts\activate  # Windows

# Verifica installazione completa
uv run pytest tests/ -v --cov=src/arima_forecaster
```

#### Opzione 2: Installazione Selettiva

```bash
# Solo funzionalità base
uv sync

# Con accelerazione GPU/CUDA 🚀
uv sync --extra gpu

# Con processing ad alte performance
uv sync --extra high-performance

# Con modelli neurali (LSTM, GRU)
uv sync --extra neural

# Con API REST
uv sync --extra api

# Con dashboard interattiva  
uv sync --extra dashboard

# Con Auto-ML
uv sync --extra automl

# Con funzionalità di sviluppo
uv sync --extra dev

# Con reporting Quarto
uv sync --extra reports

# Tutte le funzionalità
uv sync --all-extras
```

#### Opzione 3: Con pip (Alternativa tradizionale)

```bash
git clone https://github.com/tuonome/arima-forecaster.git
cd arima-forecaster

# Installazione completa
pip install -e ".[all]"

# Con GPU acceleration
pip install -e ".[gpu]"

# Con tutte le funzionalità
pip install -e ".[all]"

# Verifica installazione
python -m pytest tests/ -v
```

#### 🚀 Setup GPU/CUDA (Opzionale - Per Performance Estreme)

```bash
# 1. Verifica GPU compatibile
nvidia-smi

# 2. Installa dipendenze GPU
uv sync --extra gpu

# 3. Installa CuPy per la tua versione CUDA
pip install cupy-cuda12x  # Per CUDA 12.x
# oppure: pip install cupy-cuda11x  # Per CUDA 11.x

# 4. Configura accelerazione GPU
cp .env.example .env
# Modifica .env: ARIMA_BACKEND=auto

# 5. Test GPU acceleration
python -c "
from arima_forecaster.config import detect_gpu_capability
cap = detect_gpu_capability()
print(f'GPU Available: {cap.has_cuda}')
print(f'GPU Name: {cap.gpu_name}')
print(f'GPU Memory: {cap.gpu_memory:.1f}GB')
"
```

> **📖 Guida Completa GPU**: Vedi [docs/GPU_SETUP.md](docs/GPU_SETUP.md) per setup dettagliato GPU/CUDA

#### Verifica Installazione Rapida

```bash
# Test delle funzionalità principali
uv run python examples/advanced_forecasting_showcase.py

# Test SARIMAX con variabili esogene
uv run python examples/sarimax_example.py

# Test API REST completo
uv run python test_sarimax_api.py

# Lancia API (in background)  
uv run python scripts/run_api.py &

# Verifica API endpoints
curl http://localhost:8000/            # Info API
curl http://localhost:8000/docs        # Swagger UI
curl http://localhost:8000/scalar      # Scalar UI (moderna)
curl http://localhost:8000/redoc       # ReDoc

# Lancia dashboard (nuovo terminale)
uv run python scripts/run_dashboard.py
```

#### 📈 Installazione Facebook Prophet

Per utilizzare i modelli Prophet, installa la dipendenza aggiuntiva:

```bash
# Installazione Prophet con UV (raccomandato)
uv add prophet

# Oppure con pip
pip install prophet

# Verifica installazione Prophet
python -c "from arima_forecaster.core import ProphetForecaster; print('✅ Prophet OK!')"

# Test Prophet con esempio rapido
uv run python -c "
from arima_forecaster.core import ProphetForecaster
import pandas as pd
import numpy as np

# Crea dati di test
dates = pd.date_range('2023-01-01', periods=100, freq='D')
values = 100 + np.cumsum(np.random.randn(100) * 0.1) + 10 * np.sin(2 * np.pi * np.arange(100) / 7)
series = pd.Series(values, index=dates)

# Test Prophet
model = ProphetForecaster(country_holidays='IT')
model.fit(series)
forecast = model.forecast(steps=7)
print(f'📈 Prophet Forecast OK: {forecast.mean():.2f}')
"
```

**Note:**
- Prophet è una dipendenza opzionale per mantenere il package leggero
- Se Prophet non è installato, i modelli sono disabilitati automaticamente con graceful fallback
- Supporto per festività integrate: IT, US, UK, DE, FR, ES
- Performance ottimali su Python 3.8+ con numpy/pandas recenti

---

### 💡 **Esempi di Utilizzo**

#### 1. Forecasting Base con ARIMA

```python
from arima_forecaster import ARIMAForecaster, TimeSeriesPreprocessor, ForecastPlotter
from arima_forecaster.core import ARIMAModelSelector
import pandas as pd

# Carica e preprocessa dati
dati = pd.read_csv('vendite.csv', index_col='data', parse_dates=True)
preprocessore = TimeSeriesPreprocessor()
serie_pulita, metadata = preprocessore.preprocess_pipeline(dati['vendite'])

# Selezione automatica modello ottimale
selettore = ARIMAModelSelector(p_range=(0,3), d_range=(0,2), q_range=(0,3))
selettore.search(serie_pulita)
modello_migliore = selettore.get_best_model()

# Previsioni con intervalli di confidenza
previsioni, intervalli = modello_migliore.forecast(steps=12, return_conf_int=True)
print(f"Previsioni 12 mesi: {previsioni.describe()}")
```

#### 2. Forecasting Stagionale con SARIMA

```python
from arima_forecaster import SARIMAForecaster, SARIMAModelSelector

# Selezione automatica SARIMA per dati mensili
selettore_sarima = SARIMAModelSelector(
    seasonal_periods=[12],  # Stagionalità annuale
    max_models=50
)
selettore_sarima.search(serie_pulita)

# Migliore modello SARIMA
sarima_model = selettore_sarima.get_best_model()
print(f"Ordine SARIMA: {selettore_sarima.best_order}x{selettore_sarima.best_seasonal_order}")

# Decomposizione stagionale
decomposizione = sarima_model.get_seasonal_decomposition()
print("Componenti:", decomposizione.keys())

# Forecast stagionale
forecast_sarima = sarima_model.forecast(steps=24)  # 2 anni
```

#### 3. 📈 Forecasting Avanzato con Facebook Prophet (NUOVO)

```python
from arima_forecaster.core import ProphetForecaster, ProphetModelSelector

# Modello Prophet con festività italiane
prophet_model = ProphetForecaster(
    growth='linear',              # Crescita lineare
    yearly_seasonality=True,      # Stagionalità annuale automatica
    weekly_seasonality=True,      # Stagionalità settimanale automatica
    country_holidays='IT',        # Festività italiane
    seasonality_mode='additive'   # Stagionalità additiva
)

prophet_model.fit(serie_pulita)

# Previsioni Prophet con intervalli confidenza
forecast_prophet = prophet_model.forecast(steps=30, confidence_level=0.95)
print(f"Forecast Prophet: {forecast_prophet.mean():.2f}")

# Selezione automatica parametri Prophet con 3 metodi
prophet_selector = ProphetModelSelector(
    changepoint_prior_scales=[0.001, 0.01, 0.05, 0.1, 0.5],
    seasonality_modes=['additive', 'multiplicative'],
    scoring='mape',
    max_models=50,
    verbose=True
)

# Metodo 1: Grid Search (esaustivo)
best_model, results = prophet_selector.search(serie_pulita, method='grid_search')

# Metodo 2: Bayesian Optimization (intelligente, richiede Optuna)
# best_model, results = prophet_selector.search(serie_pulita, method='bayesian')

# Metodo 3: Random Search (veloce)
# best_model, results = prophet_selector.search(serie_pulita, method='random_search')

print(f"Migliori parametri: {prophet_selector.get_best_params()}")
print(f"Miglior score MAPE: {prophet_selector.get_best_score():.3f}%")
print(f"Modelli testati: {len(results)}")

# Analisi componenti Prophet
componenti = best_model.predict_components(serie_pulita)
print("Componenti Prophet:", componenti.columns.tolist())

# Summary completo della ricerca
print(prophet_selector.summary())
```

#### 4. ⭐ Advanced Exogenous Handling (NUOVO)

```python
from arima_forecaster.core import SARIMAXAutoSelector
from arima_forecaster.utils import ExogenousPreprocessor, ExogDiagnostics, analyze_feature_relationships
import pandas as pd

# Dati con molte variabili esogene (es. caso Moretti medicale)
serie_vendite = pd.read_csv('vendite_carrozzine.csv', parse_dates=True, index_col=0)
variabili_esogene = pd.DataFrame({
    'popolazione_over65': demographics_data,
    'temperatura_media': weather_data,
    'spesa_sanitaria': healthcare_budget_data,
    'covid_impact': covid_data,
    'competitor_price': competitor_data,
    'marketing_budget': marketing_data,
    'pil_regionale': economic_data,
    'search_trends': google_trends_data,
    'supply_disruption': supply_chain_data,
    # ... fino a 20+ variabili esogene
})

# 1. Analisi preliminare relazioni
relationships = analyze_feature_relationships(variabili_esogene, serie_vendite)
print("High correlation features:", [f for f, c in relationships['correlations'].items() if abs(c) > 0.5])

# 2. Preprocessing avanzato automatico
preprocessor = ExogenousPreprocessor(
    method='auto',  # Sceglie metodo ottimale automaticamente
    handle_outliers=True,
    outlier_method='modified_zscore',
    missing_strategy='interpolate',
    detect_multicollinearity=True,
    stationarity_test=True
)

exog_processed = preprocessor.fit_transform(variabili_esogene)
print(f"Features dopo preprocessing: {exog_processed.shape[1]} (da {variabili_esogene.shape[1]})")

# 3. SARIMAX con selezione automatica feature
selector = SARIMAXAutoSelector(
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),  # Stagionalità settimanale
    max_features=8,  # Seleziona top 8 feature automaticamente
    selection_method='stepwise',  # 'lasso', 'elastic_net', 'f_test'
    feature_engineering=['lags', 'differences'],  # Feature engineering auto
    preprocessing_method='robust',
    validation_split=0.2
)

# 4. Training con auto-selezione
selector.fit_with_exog(serie_vendite, variabili_esogene)

# 5. Analisi feature selezionate
feature_analysis = selector.get_feature_analysis()
print(f"Features selezionate: {feature_analysis['feature_details']['selected_features']}")
print(f"Selection ratio: {feature_analysis['selection_summary']['selection_ratio']:.1%}")

# 6. Diagnostica completa
diagnostics = ExogDiagnostics(max_lag=7)
diagnostic_results = diagnostics.full_diagnostic_suite(
    target_series=serie_vendite,
    exog_data=variabili_esogene,
    fitted_model=selector.fitted_model
)

print(f"Overall assessment: {diagnostic_results['summary']['overall_assessment']}")
for rec in diagnostic_results['recommendations'][:3]:
    print(f"Recommendation: {rec}")

# 7. Forecasting con variabili esogene future
forecast_exog = variabili_esogene.iloc[-30:].copy()  # Ultimi 30 giorni per pattern
forecast_result = selector.forecast_with_exog(
    steps=30,
    exog=forecast_exog,
    confidence_intervals=True
)

print(f"Forecast SARIMAX con {len(selector.selected_features)} features selezionate automaticamente")
print(f"Accuracy (AIC): {selector.fitted_model.aic:.1f}")
```

#### 4. Forecasting con Variabili Esogene (SARIMAX)

```python
from arima_forecaster import SARIMAXForecaster, SARIMAXModelSelector, TimeSeriesPreprocessor
import pandas as pd

# Dati con variabili esogene
serie_vendite = pd.read_csv('vendite.csv', index_col='data', parse_dates=True)['vendite']
variabili_esogene = pd.DataFrame({
    'temperature': temperature_data,
    'marketing_spend': marketing_data,
    'economic_indicator': economic_data
})

# Preprocessing integrato per serie target e variabili esogene
preprocessor = TimeSeriesPreprocessor()
serie_processed, exog_processed, metadata = preprocessor.preprocess_pipeline_with_exog(
    series=serie_vendite,
    exog=variabili_esogene,
    scale_exog=True,  # Standardizza variabili esogene
    validate_exog=True
)

# Selezione automatica SARIMAX
selector = SARIMAXModelSelector(
    seasonal_periods=[12],
    exog_names=list(variabili_esogene.columns),
    max_models=30
)
selector.search(serie_processed, exog=exog_processed)

# Miglior modello con variabili esogene
sarimax_model = selector.get_best_model()
print(f"Modello: SARIMAX{sarimax_model.order}x{sarimax_model.seasonal_order}")

# Analisi importanza variabili esogene
importance = sarimax_model.get_exog_importance()
print("\nImportanza variabili esogene:")
for _, row in importance.iterrows():
    sig = "✅" if row['significant'] else "❌"
    print(f"  {row['variable']}: coeff={row['coefficient']:.4f}, p-value={row['pvalue']:.4f} {sig}")

# Forecast con variabili esogene future (richieste!)
exog_future = pd.DataFrame({
    'temperature': [22.5, 23.1, 21.8, 20.5],  # Dati futuri
    'marketing_spend': [1200, 1300, 1100, 1400],
    'economic_indicator': [105.2, 106.1, 105.8, 107.0]
})

forecast_sarimax = sarimax_model.forecast(
    steps=4, 
    exog_future=exog_future,
    confidence_intervals=True
)
print(f"\nPrevisioni SARIMAX: {forecast_sarimax}")
```

#### 4. Forecasting Multivariato con VAR

```python
from arima_forecaster import VARForecaster
import pandas as pd

# Dati multivariati (es. vendite, marketing, competitori)
dati_mv = pd.DataFrame({
    'vendite': vendite_series,
    'marketing_spend': marketing_series,
    'competitor_index': competitor_series
})

# Modello VAR con selezione automatica lag
var_model = VARForecaster()
var_model.fit(dati_mv)

# Forecast multivariato
var_forecast = var_model.forecast(steps=6)
print("Forecast multivariato:", var_forecast['forecast'])

# Analisi causalità di Granger
causalita = var_model.granger_causality('vendite', ['marketing_spend'])
print("Test causalità:", causalita)

# Impulse Response Functions
irf = var_model.impulse_response(periods=10)
```

#### 5. 🚀 GPU-Accelerated Parallel Training (NUOVO)

```python
from arima_forecaster.core import GPUARIMAModelSelector, GPUSARIMAModelSelector
from arima_forecaster.config import get_config
import pandas as pd

# Configura GPU acceleration
config = get_config()
print(f"Backend: {config.backend}")
print(f"GPU Available: {config.backend == 'cuda'}")

# Dataset di molte serie temporali (es. 100+ prodotti)
series_list = []
for i in range(100):
    # Genera serie temporali realistiche per test
    series_data = generate_time_series_data(product_id=i)
    series = pd.Series(series_data, name=f"product_{i}")
    series_list.append(series)

print(f"Training {len(series_list)} serie temporali in parallelo...")

# GPU-accelerated ARIMA grid search per 100 serie
gpu_selector = GPUARIMAModelSelector(
    use_gpu=True,  # Auto-fallback su CPU se GPU non disponibile
    max_parallel_models=200  # Fino a 200 modelli in parallelo su GPU
)

# Training parallelo - da 45 minuti (CPU) a 8 minuti (GPU)
import time
start_time = time.time()
results = gpu_selector.search_multiple_series(series_list)
training_time = time.time() - start_time

print(f"Training completato in {training_time:.2f}s")
print(f"Speedup: ~{45*60/training_time:.1f}x vs CPU sequenziale")

# Risultati per ogni serie
for result in results[:5]:  # Mostra primi 5
    print(f"{result['series_name']}: {result['best_order']} "
          f"(AIC: {result['best_score']:.2f}, Status: {result['status']})")

# GPU SARIMA per serie con stagionalità
gpu_sarima = GPUSARIMAModelSelector(
    use_gpu=True,
    max_parallel_models=50  # SARIMA più pesante
)

seasonal_results = gpu_sarima.search_multiple_series(series_list[:20])
print(f"GPU SARIMA completato per {len(seasonal_results)} serie")

# Configurazione personalizzata GPU
from arima_forecaster.utils.gpu_utils import get_gpu_manager

gpu_manager = get_gpu_manager()
memory_info = gpu_manager.get_memory_info()
if memory_info:
    free_gb = memory_info[0] / (1024**3)
    total_gb = memory_info[1] / (1024**3) 
    print(f"GPU Memory: {free_gb:.1f}GB free / {total_gb:.1f}GB total")

# Context manager per operazioni GPU specifiche
with gpu_manager.device_context(device_id=0):
    # Operazioni su GPU 0
    custom_gpu_operations()
```

#### 6. Auto-ML e Ottimizzazione Avanzata

```python
from arima_forecaster.automl import ARIMAOptimizer, HyperparameterTuner

# Ottimizzazione singola con Optuna
optimizer = ARIMAOptimizer(objective_metric='aic')
risultato = optimizer.optimize_optuna(serie_pulita, n_trials=100)

print(f"Parametri ottimali: {risultato['best_params']}")
print(f"Score migliore: {risultato['best_score']}")

# Multi-objective optimization
tuner = HyperparameterTuner(
    objective_metrics=['aic', 'bic', 'mse'],
    ensemble_method='weighted_average'
)

# Ottimizzazione multi-obiettivo
multi_result = tuner.multi_objective_optimization('arima', serie_pulita)
print(f"Soluzioni Pareto: {multi_result['n_pareto_solutions']}")

# Ensemble di modelli
ensemble_result = tuner.ensemble_optimization('arima', serie_pulita, n_models=5)
ensemble_forecast = tuner.forecast_ensemble(steps=12, method='weighted')
```

#### 7. API REST - Client Python

```python
import requests
import json

# Training di un modello SARIMAX via API
api_url = "http://localhost:8000"

# Prepara dati con variabili esogene per API
data_payload = {
    "data": {
        "timestamps": serie_pulita.index.strftime('%Y-%m-%d').tolist(),
        "values": serie_pulita.values.tolist()
    },
    "model_type": "sarimax",
    "exogenous_data": {
        "variables": {
            "temperature": temperature_data.tolist(),
            "marketing_spend": marketing_data.tolist(),
            "economic_indicator": economic_data.tolist()
        }
    },
    "auto_select": True
}

# Invia richiesta training SARIMAX
response = requests.post(f"{api_url}/models/train", json=data_payload)
model_info = response.json()
model_id = model_info['model_id']

# Genera forecast SARIMAX via API (richiede variabili esogene future)
forecast_payload = {
    "steps": 12,
    "confidence_level": 0.95,
    "return_intervals": True,
    "exogenous_future": {
        "variables": {
            "temperature": [22.0] * 12,
            "marketing_spend": [1500.0] * 12,
            "economic_indicator": [105.0] * 12
        }
    }
}

forecast_response = requests.post(
    f"{api_url}/models/{model_id}/forecast", 
    json=forecast_payload
)
forecast_result = forecast_response.json()
print("Forecast SARIMAX API:", forecast_result['forecast_values'][:3])

# Esempio anche con SARIMA tradizionale
sarima_payload = {
    "data": {
        "timestamps": serie_pulita.index.strftime('%Y-%m-%d').tolist(),
        "values": serie_pulita.values.tolist()
    },
    "model_type": "sarima",
    "auto_select": True
}
sarima_response = requests.post(f"{api_url}/models/train", json=sarima_payload)
print("SARIMA Model ID:", sarima_response.json()['model_id'])
```

#### 7. Dashboard Interattiva (Script di Lancio)

```python
# Lancia dashboard Streamlit
import subprocess
import sys

# Via script
subprocess.run([sys.executable, "scripts/run_dashboard.py"])

# Oppure direttamente
import streamlit as st
from arima_forecaster.dashboard.main import ARIMADashboard

dashboard = ARIMADashboard()
dashboard.run()
```

#### 8. 🏭 Inventory Optimization - Slow/Fast Moving (NUOVO)

```python
from arima_forecaster.inventory import (
    MovementClassifier, SlowFastOptimizer, PerishableManager,
    MultiEchelonOptimizer, CapacityConstrainedOptimizer, KittingOptimizer
)
import pandas as pd

# 1. CLASSIFICAZIONE SLOW/FAST MOVING
classifier = MovementClassifier()

# Classifica prodotto per velocità movimento
turnover_annuo = 8.5  # rotazioni/anno
categoria = classifier.classify_by_movement(turnover_annuo)
print(f"Categoria: {categoria.value[1]}")  # "Media rotazione"

# Analisi portfolio completo
products_data = pd.DataFrame({
    'product_id': ['A001', 'B002', 'C003'],
    'name': ['Mascherine FFP2', 'Carrozzina Standard', 'Kit Chirurgico'],
    'fatturato': [500000, 250000, 75000]
})

sales_history = pd.DataFrame({
    'product_id': ['A001'] * 365 + ['B002'] * 365,
    'quantity': np.random.poisson(50, 365).tolist() + np.random.poisson(5, 365).tolist()
})

# Analisi completa con classificazione ABC-XYZ
analysis = classifier.analyze_product_portfolio(products_data, sales_history)
print(analysis[['product_id', 'movimento', 'turnover', 'cv']].head())

# 2. OTTIMIZZAZIONE DIFFERENZIATA
optimizer = SlowFastOptimizer(costi_giacenza)

# Per prodotti Fast Moving
fast_optimization = optimizer.optimize_fast_moving(
    demand_history=daily_sales_fast,
    unit_cost=25.0,
    lead_time=7,
    supplier_constraints={'min_order_qty': 100, 'truck_capacity': 1000}
)

print(f"Fast Moving Strategy:")
print(f"  Safety Stock: {fast_optimization['safety_stock']} unità")
print(f"  EOQ: {fast_optimization['eoq']} unità") 
print(f"  Service Level: {fast_optimization['service_level']:.0%}")

# Per prodotti Slow Moving
slow_optimization = optimizer.optimize_slow_moving(
    demand_history=daily_sales_slow,
    unit_cost=450.0,
    lead_time=21,
    shelf_life_days=730  # 2 anni shelf life
)

print(f"Slow Moving Strategy:")
print(f"  Safety Stock: {slow_optimization['safety_stock']} unità")
print(f"  Rischio Obsolescenza: {slow_optimization['obsolescence_risk']:.0%}")
print(f"  Make-to-Order Threshold: {slow_optimization['make_to_order_threshold']}")

# 3. GESTIONE PRODOTTI DEPERIBILI (FEFO)
from datetime import datetime, timedelta

perishable_mgr = PerishableManager()

# Lotti con scadenze diverse
lotti_farmaci = [
    {
        'lotto_id': 'LOT001',
        'quantita': 500,
        'data_produzione': datetime.now() - timedelta(days=200),
        'data_scadenza': datetime.now() + timedelta(days=165),
        'valore_unitario': 12.50
    }
]

# Analisi FEFO (First Expired, First Out)
lotti_analizzati = perishable_mgr.analizza_lotti(lotti_farmaci)
lotto_critico = lotti_analizzati[0]

# Calcola markdown automatico per smaltimento
markdown = perishable_mgr.calcola_markdown_ottimale(
    lotto_critico,
    domanda_giornaliera=15,
    elasticita_prezzo=2.0
)

print(f"Lotto {lotto_critico.lotto_id}:")
print(f"  Giorni rimanenti: {lotto_critico.giorni_residui}")
print(f"  Markdown suggerito: {markdown['markdown_suggerito']:.0%}")
print(f"  Azione: {markdown['azione']}")
```

#### 9. 🏢 Multi-Echelon & Capacity Optimization (NUOVO)

```python
# MULTI-ECHELON: Ottimizzazione rete distributiva
from arima_forecaster.inventory import MultiEchelonOptimizer, NodoInventory, LivelloEchelon

# Setup rete: Centrale → 3 Regionali → 6 Locali
rete_nodi = {
    'CENTRAL': NodoInventory(
        nodo_id='CENTRAL',
        nome='Deposito Centrale Milano',
        livello=LivelloEchelon.CENTRALE,
        capacita_max=50000,
        stock_attuale=15000,
        demand_rate=0,
        lead_time_fornitori={'SUPPLIER': 14},
        costi_trasporto={'REG_NORD': 2.5, 'REG_CENTRO': 3.0},
        nodi_figli=['REG_NORD', 'REG_CENTRO'],
        nodi_genitori=['SUPPLIER']
    )
}

optimizer_me = MultiEchelonOptimizer(rete_nodi)

# Safety stock con risk pooling
ss_info = optimizer_me.calcola_safety_stock_echelon(
    'CENTRAL',
    service_level_target=0.95,
    variabilita_domanda=25
)

print(f"Safety Stock Centrale: {ss_info['safety_stock']} unità")
print(f"Beneficio Risk Pooling: {ss_info['beneficio_pooling_pct']:.1f}%")

# Allocation ottimale tra nodi
allocation = optimizer_me.ottimizza_allocation(
    stock_disponibile_centrale=8000,
    richieste_nodi={'REG_NORD': 3500, 'REG_CENTRO': 2800},
    priorita_nodi={'REG_NORD': 1.5, 'REG_CENTRO': 1.0}
)

print(f"Fill Rate: {allocation['fill_rate_medio']:.0%}")
for nodo, dettaglio in allocation['dettaglio_nodi'].items():
    print(f"  {nodo}: {dettaglio['allocato']}/{dettaglio['richiesta']}")

# CAPACITY CONSTRAINTS: Gestione vincoli fisici/budget
from arima_forecaster.inventory import CapacityConstrainedOptimizer, VincoloCapacita, TipoCapacita

vincoli = {
    'volume': VincoloCapacita(
        tipo=TipoCapacita.VOLUME,
        capacita_massima=1200.0,  # m³
        utilizzo_corrente=850.0,
        unita_misura="m³",
        costo_per_unita=150.0  # €/m³ espansione
    ),
    'budget': VincoloCapacita(
        tipo=TipoCapacita.BUDGET,
        capacita_massima=500000.0,  # €500k
        utilizzo_corrente=320000.0,
        unita_misura="€"
    )
}

capacity_optimizer = CapacityConstrainedOptimizer(vincoli)

# Richieste che superano capacity
richieste_riordino = {
    'PROD_A': 1500,
    'PROD_B': 2000,  # Alto impatto volume/budget
    'PROD_C': 8000
}

# Ottimizzazione con vincoli
result = capacity_optimizer.ottimizza_con_vincoli(
    richieste_riordino,
    priorita_prodotti={'PROD_A': 1.0, 'PROD_B': 1.8, 'PROD_C': 0.6}
)

print(f"Fill Rate con vincoli: {result['fill_rate']:.0%}")
print("Vincoli saturati:", result['vincoli_saturati'])

# Suggerimenti investimenti
if result['fill_rate'] < 0.9:
    richieste_mancanti = {
        pid: richieste_riordino[pid] - result['quantita_approvate'][pid]
        for pid in richieste_riordino
        if result['quantita_approvate'][pid] < richieste_riordino[pid]
    }
    
    espansioni = capacity_optimizer.suggerisci_espansione_capacita(richieste_mancanti)
    print(f"Investimento suggerito: €{espansioni['investimento_totale']:,.0f}")
    print(f"ROI medio: {espansioni['roi_medio_ponderato']:.1f}%/anno")
```

#### 10. 🔧 Kit/Bundle Optimization (NUOVO)

```python
# KITTING: Ottimizzazione kit vs componenti separati
from arima_forecaster.inventory import KittingOptimizer, ComponenteKit, DefinzioneKit, TipoComponente

# Definisci componenti kit medico
componenti = [
    ComponenteKit(
        componente_id='STETOSCOPIO',
        nome='Stetoscopio Standard',
        tipo=TipoComponente.MASTER,
        quantita_per_kit=1,
        costo_unitario=45.0,
        lead_time=14,
        criticalita=0.9,
        sostituibili=['STETOSCOPIO_PRO']
    ),
    ComponenteKit(
        componente_id='TERMOMETRO',
        nome='Termometro Digitale',
        tipo=TipoComponente.STANDARD,
        quantita_per_kit=1,
        costo_unitario=15.0,
        lead_time=7,
        sostituibili=['TERMOMETRO_IR']
    )
]

# Definisci kit completo
kit_medico = DefinzioneKit(
    kit_id='KIT_MEDICO_BASE',
    nome='Kit Medico Base',
    componenti=componenti,
    prezzo_vendita_kit=150.0,
    margine_target=0.35,
    domanda_storica_kit=[25, 30, 28, 35],
    can_sell_components_separately=True
)

kitting_optimizer = KittingOptimizer({'KIT_MEDICO_BASE': kit_medico})

# Setup inventory componenti
kitting_optimizer.aggiorna_inventory_componente('STETOSCOPIO', 150)
kitting_optimizer.aggiorna_inventory_componente('TERMOMETRO', 200)

# Analisi assemblabilità
kit_info = kitting_optimizer.calcola_kit_assemblabili('KIT_MEDICO_BASE')
print(f"Kit assemblabili: {kit_info['kit_assemblabili']}")
print(f"Componente limitante: {kit_info['componente_limitante']}")

# Decisione: Kit vs Componenti Separati
forecast_kit = np.array([28, 32, 30, 25])
forecast_componenti = {
    'STETOSCOPIO': np.array([5, 7, 4, 6]),
    'TERMOMETRO': np.array([12, 15, 10, 8])
}

strategy = kitting_optimizer.ottimizza_kit_vs_componenti(
    'KIT_MEDICO_BASE',
    forecast_kit,
    forecast_componenti
)

print(f"Strategia: {strategy['strategia_consigliata']}")
print(f"Focus: {strategy['focus_principale']}")
print(f"ROI kit vs componenti: {strategy['analisi_finanziaria']['roi_kit_vs_componenti']:.2f}x")

# Disassembly planning
disassembly = kitting_optimizer.pianifica_disassembly(
    'KIT_MEDICO_BASE',
    20,  # Disfa 20 kit
    {'TERMOMETRO': 25, 'STETOSCOPIO': 10}  # Domanda urgente componenti
)

print(f"Disassembly convenienza: €{disassembly['convenienza_netta']:,.0f}")
print(f"Raccomandazione: {disassembly['raccomandazione']}")
```

#### 11. Report Dinamici con Quarto

```python
from arima_forecaster import ARIMAForecaster, SARIMAForecaster, SARIMAXForecaster
from arima_forecaster.reporting import QuartoReportGenerator
from arima_forecaster.visualization import ForecastPlotter

# Addestra modelli
arima_model = ARIMAForecaster(order=(2,1,2))
sarima_model = SARIMAForecaster(order=(1,1,1), seasonal_order=(1,1,1,12))
sarimax_model = SARIMAXForecaster(order=(1,1,1), seasonal_order=(1,1,1,12), exog_names=['temperature', 'marketing'])

arima_model.fit(serie_pulita)
sarima_model.fit(serie_pulita)
sarimax_model.fit(serie_pulita, exog=variabili_esogene)

# Crea visualizzazioni
plotter = ForecastPlotter()
forecast_arima = arima_model.forecast(steps=12, confidence_intervals=True)
forecast_sarima = sarima_model.forecast(steps=12, confidence_intervals=True)

# Salva grafici
plots_data = {}
plots_data['arima_forecast'] = plotter.plot_forecast(
    actual=serie_pulita, 
    forecast=forecast_arima['forecast'],
    confidence_intervals=forecast_arima['confidence_intervals'],
    save_path="outputs/plots/arima_forecast.png"
)

# Genera report individuale ARIMA
arima_report = arima_model.generate_report(
    plots_data=plots_data,
    report_title="Analisi Completa Vendite ARIMA",
    output_filename="vendite_arima_analysis",
    format_type="html",
    include_diagnostics=True,
    include_forecast=True,
    forecast_steps=24
)
print(f"Report ARIMA generato: {arima_report}")

# Genera report comparativo
generator = QuartoReportGenerator()
comparison_report = generator.create_comparison_report(
    models_results={
        'ARIMA(2,1,2)': {
            'model_type': 'ARIMA',
            'order': arima_model.order,
            'model_info': arima_model.get_model_info(),
            'metrics': evaluator.calculate_forecast_metrics(
                serie_pulita, arima_model.predict()
            )
        },
        'SARIMA(1,1,1)x(1,1,1,12)': {
            'model_type': 'SARIMA', 
            'order': sarima_model.order,
            'seasonal_order': sarima_model.seasonal_order,
            'model_info': sarima_model.get_model_info(),
            'metrics': evaluator.calculate_forecast_metrics(
                serie_pulita, sarima_model.predict()
            )
        }
    },
    report_title="Confronto Modelli ARIMA vs SARIMA - Vendite",
    output_filename="vendite_models_comparison",
    format_type="html"
)
print(f"Report comparativo generato: {comparison_report}")

# Export in formati multipli
pdf_report = arima_model.generate_report(
    format_type="pdf",  # Richiede LaTeX
    output_filename="vendite_executive_summary"
)

docx_report = arima_model.generate_report(
    format_type="docx",  # Richiede pandoc
    output_filename="vendite_technical_doc"
)
```

---

### 📊 **Capacità Avanzate per Tipo di Modello**

#### 🌊 SARIMA - Modelli Stagionali
- **Stagionalità Multipla**: Supporto per diversi periodi stagionali (12, 4, 7, 24)
- **Decomposizione Automatica**: Separazione trend, stagionalità, residui
- **Selezione Parametri**: Grid search per (p,d,q)(P,D,Q)_s ottimali
- **Validazione Stagionale**: Test specifici per pattern stagionali

#### 🌐 SARIMAX - Con Variabili Esogene
- **Variabili Esterne**: Integrazione fattori economici, meteorologici, marketing
- **Preprocessing Automatico**: Gestione e validazione variabili esogene
- **Analisi Importanza**: Coefficienti e significatività statistica variabili
- **Visualizzazioni Dedicate**: Dashboard specializzate per analisi esogene

#### 📈 VAR - Multivariato
- **Analisi Causalità**: Test di Granger per relazioni causa-effetto
- **Impulse Response**: Analisi propagazione shock tra variabili
- **Cointegrazione**: Test relazioni di equilibrio a lungo termine
- **FEVD**: Decomposizione varianza errore di previsione

#### 🤖 Auto-ML e Ottimizzazione
- **Algoritmi Avanzati**: Optuna (TPE), Hyperopt, Scikit-Optimize
- **Multi-Obiettivo**: Ottimizzazione simultanea di AIC, BIC, MSE
- **Ensemble Methods**: Combinazione intelligente modelli diversi
- **Early Stopping**: Prevenzione overfitting automatica

#### 🌐 API REST Production-Ready
- **Async Processing**: Training modelli in background
- **Model Registry**: Gestione persistente modelli trained
- **Batch Forecasting**: Previsioni per dataset multipli
- **Auto-Scaling**: Supporto deployment con load balancing
- **📚 Scalar UI**: Documentazione API moderna e interattiva
- **Multiple Doc Formats**: Swagger UI, ReDoc, Scalar per ogni esigenza

#### 📊 Dashboard Interattiva Multilingue
- **5 Lingue Supportate**: Italiano, English, Español, Français, 中文 (Cinese)
- **Data Exploration**: Upload CSV, statistiche, visualizzazioni localizzate
- **Model Comparison**: Confronto performance modelli con interfaccia tradotta
- **Interactive Plotting**: Grafici Plotly con zoom, filtering, labels multilingue
- **Export Results**: Download forecast e report multilingue in CSV/PDF
- **Smart Filtering**: Filtri "Tutti" per visualizzazioni aggregate

#### 📄 Reporting Dinamico Multilingue con Quarto
- **5 Lingue Supportate**: Report automatici in IT, EN, ES, FR, ZH
- **Report Automatici**: Template professionali con analisi integrate e tradotte
- **Multi-Formato**: Export HTML, PDF, DOCX con caratteri Unicode corretti
- **Analisi Intelligenti**: Interpretazione automatica metriche localizzate
- **Visualizzazioni Embed**: Grafici integrati con titoli e legende tradotti
- **Report Comparativi**: Confronto automatico tra modelli multipli
- **Personalizzazione**: Template Quarto modificabili e estendibili

#### Preprocessing Intelligente (Esteso)
- **Valori Mancanti**: 5 strategie (interpolazione, drop, forward/backward fill, seasonally-adjusted)
- **Outlier**: 4 metodi rilevamento (IQR, Z-score, Z-score modificato, Isolation Forest)
- **Stazionarietà**: Test ADF, KPSS, PP con differenziazione automatica
- **Frequency Detection**: Rilevamento automatico frequenza dati

#### Selezione Modelli Automatica (Avanzata)
- **Smart Grid Search**: Pruning intelligente spazio parametri
- **Cross-Validation**: Time series split per validazione robusta
- **Information Criteria**: AIC, BIC, HQIC con penalizzazione complessità
- **Parallel Processing**: Selezione veloce su hardware multi-core

#### Valutazione Completa (20+ Metriche)
- **Accuracy**: MAE, RMSE, MAPE, SMAPE, R², MASE, Theil's U, MSIS
- **Residual Tests**: Jarque-Bera, Ljung-Box, Durbin-Watson, Breusch-Pagan, ARCH
- **Interval Quality**: PICP, MPIW, ACE per intervalli confidenza
- **Business Metrics**: Directional accuracy, hit rate, profit-based scoring

---

### 📚 **Documentazione Completa**

| Documento | Descrizione | Livello |
|-----------|-------------|---------|
| **[Teoria ARIMA](docs/teoria_arima.md)** | Fondamenti matematici, componenti AR/I/MA, diagnostica | Teorico |
| **[Teoria SARIMA](docs/teoria_sarima.md)** | Matematica SARIMA, stagionalità, implementazione | Teorico |
| **[Teoria SARIMAX](docs/teoria_sarimax.md)** | Matematica SARIMAX, variabili esogene, validazione | Teorico |
| **[Confronto ARIMA/SARIMA/SARIMAX](docs/confronto_modelli_arima.md)** | Guida completa alla scelta del modello ottimale | Pratico |
| **[ARIMA vs SARIMA](docs/arima_vs_sarima.md)** | Confronto dettagliato, scelta del modello, casi d'uso | Pratico |
| **[SARIMA vs SARIMAX](docs/sarima_vs_sarimax.md)** | Quando usare variabili esogene, esempi pratici | Pratico |
| **[Guida Utente](docs/guida_utente.md)** | Esempi pratici, API, workflow completo | Pratico |
| **[Funzionalità Avanzate](ADVANCED_FEATURES.md)** | VAR, Auto-ML, API, Dashboard - guida completa | Avanzato |
| **[CLAUDE.md](CLAUDE.md)** | Architettura, sviluppo, comandi dettagliati | Sviluppatori |

### 🎯 **Guide Rapide per Caso d'Uso**

#### 📈 Business/Finance
```bash
# Vendite mensili con stagionalità
uv run python examples/retail_sales_forecasting.py

# Serie finanziarie daily
uv run python examples/financial_time_series.py
```

#### 🏭 Industria/IoT
```bash  
# Dati sensori multivariati
uv run python examples/iot_sensor_forecasting.py

# Produzione con downtime
uv run python examples/manufacturing_forecasting.py
```

#### 🌐 Web/Digital
```bash
# Traffico web con stagionalità multipla
uv run python examples/web_traffic_forecasting.py

# Metriche utente
uv run python examples/user_engagement_forecasting.py
```

---

### 📊 **Benchmark Performance Dettagliati**

#### Velocità Training Modelli
| Modello | Tempo Medio | Memoria | Note |
|---------|-------------|---------|------|
| **ARIMA** | ~0.1-1.0s | 1-10MB | Performance baseline |
| **SARIMA** | ~0.5-5.0s | 5-20MB | Con componenti stagionali |
| **SARIMAX** | ~1.0-8.0s | 10-50MB | Con variabili esogene |
| **VAR** | ~0.1-2.0s | 5-25MB | Multivariato |
| **Prophet** | ~3.0-8.0s | 20-100MB | Con MCMC sampling |
| **Auto-ML** | ~10-300s | 50-500MB | Ottimizzazione completa |
| **Forecast** | < 100ms | < 1MB | Post-training |
| **Report** | 10-20s | 10-50MB | HTML/PDF con Quarto |

#### Efficienza Ottimizzazione
| Algoritmo | Convergenza | Parallelizzazione | Casi d'Uso |
|-----------|-------------|-------------------|-------------|
| **Grid Search** | Baseline | ✅ Multi-core | Spazi piccoli |
| **Optuna TPE** | 2-5x più veloce | ✅ Distributed | Spazi complessi |
| **Bayesian** | 3-7x più veloce | ⚠️ Limitata | Funzioni costose |
| **Multi-obiettivo** | 10-50 soluzioni Pareto | ✅ Multi-core | Trade-off multipli |
| **Ensemble** | 3-7 modelli diversi | ✅ GPU-ready | Massima accuratezza |

#### Utilizzo Memoria
| Componente | Memoria Base | Memoria Massima | Scaling |
|------------|--------------|-----------------|----------|
| **Singolo Modello** | 1-10MB | 50MB | Lineare con dati |
| **Ensemble (5 modelli)** | 5-50MB | 250MB | Lineare con N modelli |
| **Server API** | 50-200MB | 1GB | Con cache modelli |
| **Dashboard** | 100-300MB | 800MB | Con grafici interattivi |
| **GPU Acceleration** | +500MB VRAM | +4GB VRAM | Con batch processing |

### 🔧 **Configurazione Production-Ready**

#### Impostazioni Ottimizzazione Auto-ML
```python
from arima_forecaster.automl import ARIMAOptimizer, HyperparameterTuner

# Configurazione ottimizzazione personalizzata
optimizer = ARIMAOptimizer(
    objective_metric='aic',
    cv_folds=3,
    test_size=0.2,
    n_jobs=4,  # Processamento parallelo
    random_state=42,
    timeout=3600,  # Timeout 1 ora
    memory_limit='4GB'
)

# Impostazioni tuner avanzate
tuner = HyperparameterTuner(
    objective_metrics=['aic', 'bic', 'mse'],
    ensemble_method='pareto',
    meta_learning=True,
    early_stopping_patience=10,
    pruning_enabled=True,  # Pruning trials non promettenti
    storage='sqlite:///optuna_studies.db'  # Persistenza studi
)
```

#### Configurazione API Enterprise
```python
from arima_forecaster.api import create_app

# Configurazione API production-ready
app = create_app(
    model_storage_path="/data/models",
    enable_scalar=True,
    production_mode=True,
    max_concurrent_models=10,
    model_cache_size=100,
    request_timeout=300,
    cors_origins=["https://dashboard.company.com"],
    rate_limit="100/minute",
    auth_enabled=True
)

# Esegui con impostazioni enterprise
import uvicorn
uvicorn.run(
    app, 
    host="0.0.0.0", 
    port=8080, 
    workers=4,
    worker_class="uvicorn.workers.UvicornWorker",
    access_log=True,
    log_config="logging.conf"
)
```

#### Variabili d'Ambiente Complete
```bash
# .env file production
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
PRODUCTION_MODE=true
ENABLE_SCALAR=true
MAX_WORKERS=4
REQUEST_TIMEOUT=300

# Model Storage
MODEL_STORAGE_PATH=/data/models
MODEL_CACHE_SIZE=100
MAX_CONCURRENT_MODELS=10

# Security
CORS_ORIGINS=["https://dashboard.company.com", "https://api.company.com"]
RATE_LIMIT=100/minute
AUTH_ENABLED=true
SECRET_KEY=your-secret-key-here

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/arima-forecaster.log
LOG_ROTATION=daily

# Database (for model registry)
DATABASE_URL=postgresql://user:pass@localhost/arima_models
REDIS_URL=redis://localhost:6379/0

# Monitoring
METRICS_ENABLED=true
HEALTH_CHECK_ENDPOINT=/health
PROMETHEUS_METRICS=/metrics

# GPU Configuration
ARIMA_BACKEND=auto
CUDA_DEVICE=0
MAX_GPU_MODELS_PARALLEL=100
GPU_MEMORY_LIMIT=0.8

# Auto-ML Settings
OPTUNA_STORAGE=postgresql://user:pass@localhost/optuna
MAX_TRIALS=1000
OPTIMIZATION_TIMEOUT=3600
```

### 🚀 **Estendere la Libreria**

#### Aggiungere Nuovi Ottimizzatori
```python
from arima_forecaster.automl.base import BaseOptimizer

class CustomOptimizer(BaseOptimizer):
    """Ottimizzatore personalizzato con algoritmo proprietario."""
    
    def optimize_custom(self, series, **kwargs):
        """Implementa logica di ottimizzazione personalizzata."""
        # Implementazione algoritmo custom
        best_params = self._custom_search_algorithm(series)
        return {
            'best_params': best_params,
            'best_score': self._evaluate_params(series, best_params),
            'search_history': self.history,
            'convergence_info': self.convergence_metrics
        }
    
    def _custom_search_algorithm(self, series):
        """Algoritmo di ricerca proprietario."""
        # Logica custom per ottimizzazione
        pass
```

#### Estendere API con Endpoint Personalizzati
```python
from fastapi import APIRouter
from arima_forecaster.api.main import app

# Router personalizzato
custom_router = APIRouter(prefix="/custom", tags=["Custom"])

@custom_router.post("/proprietary-forecast")
async def custom_forecasting_endpoint(request: CustomRequest):
    """Endpoint per forecasting con algoritmi proprietari."""
    # Implementa logica business specifica
    return {"forecast": custom_results, "metadata": custom_info}

# Registra router personalizzato
app.include_router(custom_router)
```

#### Template Report Personalizzati
```python
from arima_forecaster.reporting import QuartoReportGenerator

# Estendi generatore con template custom
class CompanyReportGenerator(QuartoReportGenerator):
    def create_branded_report(self, model_results, **kwargs):
        """Genera report con template aziendale."""
        template_config = {
            'template': 'templates/company_template.qmd',
            'logo': 'assets/company_logo.png',
            'colors': ['#0066CC', '#FF6600', '#339933'],
            'font_family': 'Arial',
            'header_footer': True
        }
        
        return self.create_comparison_report(
            models_results=model_results,
            custom_template=template_config,
            **kwargs
        )
```

### 🧪 **Testing e Qualità**

#### 🧪 **Ambiente R&D per Ricerca Algoritmi**

```bash
# Lancia Jupyter Lab per ambiente ricerca
jupyter lab notebooks/

# Oppure esegui il notebook R&D direttamente
jupyter nbconvert --execute --to notebook --inplace notebooks/research_and_development.ipynb

# Notebook R&D include:
# - Benchmarking sistematico di algoritmi ARIMA, SARIMA, ML
# - Generazione dataset sintetici con diverse caratteristiche
# - Analisi performance e visualizzazioni avanzate
# - Metodi ensemble e confronti statistici
# - Export automatico risultati e report di ricerca
```

#### Test Completi delle Funzionalità

```bash
# Test base (ARIMA core)
uv run pytest tests/test_arima_model.py -v

# Test funzionalità avanzate
uv run pytest tests/test_sarima_model.py tests/test_sarimax_model.py tests/test_var_model.py -v

# Test Auto-ML e ottimizzazione
uv run pytest tests/test_automl.py -v --timeout=300

# Test API REST
uv run pytest tests/test_api.py -v

# Test Dashboard (richiede browser headless)
uv run pytest tests/test_dashboard.py -v --browser=chrome

# Test Reporting Quarto (richiede dipendenze reports)
uv run pytest tests/test_reporting.py -v

# Tutti i test con coverage dettagliata
uv run pytest tests/ -v --cov=src/arima_forecaster --cov-report=html --cov-report=term-missing
```

#### Quality Assurance Pipeline

```bash
# 1. Formatting automatico
uv run black src/ tests/ examples/ scripts/

# 2. Linting e security check  
uv run ruff check src/ tests/ examples/ scripts/
uv run ruff format src/ tests/ examples/ scripts/

# 3. Type checking statico
uv run mypy src/arima_forecaster/

# 4. Security scanning
uv run bandit -r src/

# 5. Dependency scanning  
uv run safety check

# 6. Documentation check
uv run pydocstyle src/arima_forecaster/

# Tutto insieme con pre-commit
uv run pre-commit run --all-files
```

#### Performance Benchmarking

```bash
# Benchmark funzionalità base
uv run python tests/performance/benchmark_basic.py

# Benchmark Auto-ML (più lento)
uv run python tests/performance/benchmark_automl.py

# Benchmark API load test
uv run python tests/performance/benchmark_api.py --requests=1000

# Memory profiling
uv run python -m memory_profiler examples/advanced_forecasting_showcase.py

# 🚀 GPU Performance Benchmarks
uv run python -c "
from arima_forecaster.utils.gpu_utils import benchmark_gpu_vs_cpu
import numpy as np

# Test GPU vs CPU performance
def matrix_ops(gpu_mgr, data):
    a = gpu_mgr.array(data)
    return gpu_mgr.xp.dot(a, a.T)

test_data = np.random.randn(1000, 1000)
results = benchmark_gpu_vs_cpu(matrix_ops, test_data)
print(f'CPU: {results[\"cpu_time\"]:.3f}s')
print(f'GPU: {results[\"gpu_time\"]:.3f}s') 
print(f'Speedup: {results[\"speedup\"]:.1f}x')
"
```

#### Test Integration in CI/CD

```bash
# Test matrix per diverse versioni Python
uv run --python 3.9 pytest tests/ -x
uv run --python 3.10 pytest tests/ -x  
uv run --python 3.11 pytest tests/ -x
uv run --python 3.12 pytest tests/ -x

# Test con diverse configurazioni hardware
uv run pytest tests/ -v --benchmark-only --benchmark-autosave
```

---

### 🎨 **Esempi Pratici e Workflows**

#### 🚀 Demo Funzionalità Complete

```bash
# Showcase completo (tutte le funzionalità avanzate)
uv run python examples/advanced_forecasting_showcase.py

# Workshop interattivo step-by-step  
uv run python examples/forecasting_workshop.py

# Confronto modelli su dataset reali
uv run python examples/model_comparison_study.py
```

#### 📊 Esempi per Dominio Specifico

```bash
# Forecasting base (series singole)
uv run python examples/forecasting_base.py
uv run python examples/sarimax_example.py
uv run python examples/selezione_automatica.py

# Reporting dinamico con Quarto
uv run python examples/quarto_reporting.py

# Business Intelligence
uv run python examples/business_metrics_forecasting.py
uv run python examples/seasonal_sales_analysis.py

# Finanza e Investimenti
uv run python examples/financial_returns_forecasting.py
uv run python examples/portfolio_risk_modeling.py

# Industria 4.0 e IoT
uv run python examples/sensor_data_forecasting.py
uv run python examples/predictive_maintenance.py

# Marketing e Web Analytics
uv run python examples/web_traffic_analysis.py
uv run python examples/customer_behavior_forecasting.py
```

#### 🔧 Utility e Tools

```bash
# Data generator per testing
uv run python scripts/generate_test_data.py --type=seasonal --length=365

# Model comparison tool
uv run python scripts/model_comparator.py --data=data/sample.csv --models=arima,sarima,var

# Batch forecasting per multiple serie
uv run python scripts/batch_forecaster.py --input-dir=data/series/ --output-dir=results/

# Performance profiler
uv run python scripts/performance_analyzer.py --model-type=all --data-size=large
```

#### 🌐 Deployment Examples

```bash
# Docker deployment
docker build -t arima-forecaster .
docker run -p 8000:8000 arima-forecaster

# Kubernetes deployment  
kubectl apply -f k8s/deployment.yaml

# Cloud deployment (AWS/GCP/Azure)
uv run python scripts/deploy_cloud.py --platform=aws --region=us-east-1
```

---

### 🛠️ **Stack Tecnologico e Dipendenze**

#### 📚 Core Dependencies
| Libreria | Scopo | Versione | Funzionalità |
|----------|-------|----------|--------------|
| **statsmodels** | Modelli statistici | >=0.14.0 | ARIMA, SARIMA, VAR, test statistici |
| **pandas** | Data manipulation | >=2.0.0 | Serie temporali, preprocessing |
| **numpy** | Computing numerico | >=1.24.0 | Array operations, linear algebra |
| **scipy** | Algoritmi scientifici | >=1.10.0 | Ottimizzazione, test statistici |
| **scikit-learn** | ML utilities | >=1.3.0 | Preprocessing, metriche, validation |

#### 🧪 **R&D Stack (Notebook Ricerca)**
| Libreria | Scopo | Funzionalità |
|----------|-------|--------------|
| **jupyter** | Notebook environment | >=1.0.0 | Ambiente interattivo ricerca |
| **psutil** | System monitoring | >=5.9.0 | Monitoraggio performance algoritmi |
| **xgboost** | Gradient boosting | >=1.7.0 | Algoritmi ML avanzati per confronti |
| **memory-profiler** | Memory profiling | >=0.60.0 | Analisi utilizzo memoria |

#### 📄 Reporting Stack (Opzionale)
| Libreria | Scopo | Funzionalità |
|----------|-------|--------------|
| **quarto** | Document generation | >=1.3.0 | Report dinamici HTML/PDF/DOCX |
| **jupyter** | Notebook support | >=1.0.0 | Esecuzione codice nei report |
| **nbformat** | Notebook format | >=5.8.0 | Supporto formato notebook |

#### 📊 Visualization Stack
| Libreria | Scopo | Funzionalità |
|----------|-------|--------------|
| **matplotlib** | Plotting base | >=3.6.0 | Grafici statici, pubblicazione |
| **seaborn** | Statistical plotting | >=0.12.0 | Grafici statistici avanzati |
| **plotly** | Interactive plots | >=5.15.0 | Dashboard interattivi, web |

#### 🤖 Auto-ML Stack  
| Libreria | Scopo | Algoritmo |
|----------|-------|-----------|
| **optuna** | Hyperparameter optimization | Tree-structured Parzen Estimator |
| **hyperopt** | Bayesian optimization | Tree Parzen Estimator, Random |
| **scikit-optimize** | Sequential optimization | Gaussian Processes, Random Forest |

#### 🌐 Web & API Stack
| Libreria | Scopo | Funzionalità |
|----------|-------|--------------|
| **fastapi** | REST API framework | Async API, auto docs, validation |
| **scalar-fastapi** | API documentation | Modern interactive API docs |
| **uvicorn** | ASGI server | High-performance async server |
| **pydantic** | Data validation | Type checking, serialization |
| **streamlit** | Web dashboards | Interactive web apps |

#### 🧪 Development Stack
| Libreria | Scopo | Funzionalità |
|----------|-------|--------------|
| **pytest** | Testing framework | Unit tests, fixtures, coverage |
| **black** | Code formatting | PEP8 compliant formatting |
| **ruff** | Linting | Fast Python linter & formatter |
| **mypy** | Type checking | Static type analysis |
| **pre-commit** | Git hooks | Code quality automation |

---

### ✅ **Roadmap e Funzionalità Implementate**

#### 🎉 Completate (v0.4.0)
- [x] **Modelli SARIMA**: Stagionalità completa con selezione automatica
- [x] **Modelli SARIMAX**: Supporto completo per variabili esogene con preprocessing automatico
- [x] **VAR Multivariato**: Forecasting serie multiple con causalità
- [x] **API REST**: Servizi production-ready con FastAPI (incluso SARIMAX)
- [x] **Dashboard Streamlit**: Interfaccia web completa con supporto SARIMAX
- [x] **Auto-ML**: Ottimizzazione con Optuna, Hyperopt, Scikit-Optimize
- [x] **Ensemble Methods**: Combinazione intelligente modelli
- [x] **Report Quarto**: Generazione automatica report dinamici HTML/PDF/DOCX
- [x] **Documentazione**: Teoria completa ARIMA, SARIMA, SARIMAX e confronti

#### ✅ Implementato (v0.4.1)
- [x] **Advanced Exog Handling**: Selezione automatica feature, preprocessing avanzato, diagnostica per SARIMAX

#### ✅ Implementato (v0.5.0 - Agosto 2024)
- [x] **Prophet Integration**: Facebook Prophet models con supporto completo stagionalità e holidays ✅
- [x] **Prophet Auto-Selection**: Ottimizzazione automatica parametri Prophet (Grid, Random, Bayesian) ✅
- [x] **Sistema Multilingue**: Dashboard e report in 5 lingue (IT, EN, ES, FR, ZH) ✅
- [x] **Cold Start Problem**: Forecasting per nuovi prodotti senza dati storici ✅
- [x] **Anomaly Detection**: Rilevamento outlier integrato (IQR, z-score, isolation forest) ✅

#### 🚧 In Sviluppo (v0.6.0)
- [ ] **LSTM Integration**: Hybrid ARIMA-Deep Learning
- [ ] **Real-time Streaming**: Apache Kafka integration
- [ ] **Cloud Native**: Kubernetes operators

#### 🔮 Future Releases
- [ ] **MLOps Pipeline**: MLflow, DVC, Airflow integration  
- [ ] **Multi-tenancy**: Enterprise deployment features
- [x] **GPU Acceleration**: CUDA support per training veloce ✅ **IMPLEMENTED**

---

### 🤝 **Contributi e Community**

Contributi benvenuti! La libreria è progettata per crescere con la community.

#### 🚀 **Come Contribuire**

```bash
# 1. Fork e clona
git clone https://github.com/tuonome/arima-forecaster.git
cd arima-forecaster

# 2. Setup development environment
uv sync --all-extras
source .venv/bin/activate

# 3. Crea feature branch
git checkout -b feature/nuova-funzionalita

# 4. Sviluppa con testing
uv run pytest tests/ -v
uv run pre-commit run --all-files

# 5. Commit e push
git commit -am 'feat: aggiunge supporto per modelli LSTM'
git push origin feature/nuova-funzionalita

# 6. Apri Pull Request
```

#### 🎯 **Aree di Contribuzione**

| Area | Difficoltà | Skills Richieste |
|------|------------|------------------|
| **Nuovi Modelli** | 🔴🔴🔴 | Statistiche, Matematica |
| **API Endpoints** | 🔴🔴 | FastAPI, Python async |
| **Dashboard Features** | 🔴🔴 | Streamlit, UI/UX |
| **Testing & QA** | 🔴 | Pytest, Test automation |
| **Documentazione** | 🔴 | Technical writing |
| **Performance** | 🔴🔴🔴 | Profiling, Optimization |

#### 📋 **Contribution Guidelines**

- **Code Style**: Black + Ruff formatting
- **Testing**: >90% coverage richiesta
- **Documentation**: Docstrings + type hints
- **Commit Messages**: Conventional commits format
- **Review Process**: Peer review obbligatorio

---

### 📄 **Licenza e Disclaimer**

Questo progetto è rilasciato sotto **Licenza MIT**. Vedi file [LICENSE](LICENSE) per dettagli completi.

**📋 Disclaimer**: Questa libreria è fornita "as is" per scopi educativi e di ricerca. Per utilizzo in produzione, si raccomanda testing approfondito e validazione dei risultati da parte di esperti del dominio.

---

### 👥 **Team e Riconoscimenti**

#### 🏆 **Core Team**
- **Il Tuo Nome** - Architetto e Lead Developer - [@tuonome](https://github.com/tuonome)
- **Contributor 1** - Machine Learning Engineer
- **Contributor 2** - Data Scientist

#### 🌟 **Contributors**
Ringraziamo tutti i [contributors](https://github.com/tuonome/arima-forecaster/graphs/contributors) che hanno reso possibile questo progetto.

#### 🙏 **Special Thanks**
- **Box & Jenkins** per la metodologia ARIMA fondamentale
- **Statsmodels Community** per l'eccellente implementazione statistica
- **FastAPI & Streamlit Teams** per i framework web moderni
- **Optuna Developers** per l'ottimizzazione hyperparameter
- **Open Source Community** per il supporto e feedback continui

---

### 📊 **Metriche Progetto**

![GitHub Stars](https://img.shields.io/github/stars/tuonome/arima-forecaster?style=social)
![GitHub Forks](https://img.shields.io/github/forks/tuonome/arima-forecaster?style=social)
![GitHub Issues](https://img.shields.io/github/issues/tuonome/arima-forecaster)
![GitHub PR](https://img.shields.io/github/issues-pr/tuonome/arima-forecaster)
![Code Coverage](https://img.shields.io/codecov/c/github/tuonome/arima-forecaster)
![PyPI Downloads](https://img.shields.io/pypi/dm/arima-forecaster)

---

### 📞 **Supporto e Community**

#### 💬 **Canali di Supporto**
- 📖 **Documentazione**: Consulta `docs/` per guide dettagliate
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/tuonome/arima-forecaster/issues)
- 💡 **Feature Requests**: [GitHub Discussions](https://github.com/tuonome/arima-forecaster/discussions)
- 📧 **Email**: [arima-forecaster@example.com](mailto:arima-forecaster@example.com)
- 💬 **Discord**: [Community Server](https://discord.gg/arima-forecaster)

#### 🎓 **Risorse Educational**
- 📺 **Video Tutorials**: [YouTube Playlist](https://youtube.com/playlist/arima-tutorials)
- 📝 **Blog Posts**: [Medium Publication](https://medium.com/arima-forecasting)  
- 🎙️ **Podcast**: [Data Science Talks](https://podcast.example.com)
- 📚 **Workshop Materials**: [GitHub Learning](https://github.com/tuonome/arima-workshops)

#### 🌐 **Social Media**
- 🐦 **Twitter**: [@ARIMAForecaster](https://twitter.com/arimaforecaster)
- 💼 **LinkedIn**: [ARIMA Forecasting Group](https://linkedin.com/groups/arimaforecasting)

---

### 🏃‍♂️ **Quick Start Summary**

```bash
# 1️⃣ Install
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/tuonome/arima-forecaster.git && cd arima-forecaster
uv sync --all-extras && source .venv/bin/activate

# 2️⃣ Test
uv run python examples/advanced_forecasting_showcase.py
uv run python examples/sarimax_example.py  # Test SARIMAX con variabili esogene

# 3️⃣ Explore
uv run python test_sarimax_api.py           # Test API completo
uv run python scripts/run_api.py &          # API server
uv run python scripts/run_dashboard.py     # Interactive dashboard

# 4️⃣ Develop
uv run pytest tests/ -v --cov=src/arima_forecaster
```

---

<div align="center">

### ⭐ **Se questo progetto ti è utile, lascia una stella!** ⭐

![Stargazers](https://reporoster.com/stars/tuonome/arima-forecaster)

**Sviluppato con ❤️ per la comunità italiana di Data Science e Time Series Analysis**

*Contribuisci al futuro del forecasting in Italia 🇮🇹*

---

[![Powered by](https://img.shields.io/badge/Powered%20by-Python%203.9+-blue.svg)](https://python.org)
[![Built with](https://img.shields.io/badge/Built%20with-♥-red.svg)](https://github.com/tuonome/arima-forecaster)
[![Made in](https://img.shields.io/badge/Made%20in-Italy-green.svg)](https://github.com/tuonome/arima-forecaster)

</div>