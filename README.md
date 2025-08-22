# ARIMA Forecaster 🚀

## Libreria Avanzata per Forecasting Serie Temporali con Modelli ARIMA, SARIMA e VAR

Una libreria Python professionale e completa per l'analisi, modellazione e previsione di serie temporali utilizzando modelli ARIMA, SARIMA (Seasonal ARIMA) e VAR (Vector Autoregression). Include funzionalità avanzate di Auto-ML, API REST, dashboard interattiva e ottimizzazione automatica dei parametri per applicazioni enterprise-grade.

---

### 🌟 **Nuove Funzionalità Avanzate**

- **🌊 Modelli SARIMA**: Gestione completa della stagionalità con parametri (P,D,Q,s)
- **🌐 Modelli SARIMAX**: Modelli con variabili esogene per incorporare fattori esterni
- **📈 Modelli VAR**: Forecasting multivariato con analisi di causalità e impulse response
- **🤖 Auto-ML**: Ottimizzazione automatica con Optuna, Hyperopt e Scikit-Optimize  
- **🌐 API REST**: Servizi di forecasting production-ready con FastAPI
- **📊 Dashboard Streamlit**: Interfaccia web interattiva per utenti non tecnici
- **📄 Report Quarto**: Generazione report dinamici professionali con analisi automatiche
- **🎯 Ensemble Methods**: Combinazione intelligente di modelli diversi
- **⚡ Ottimizzazione Parallela**: Selezione modelli veloce su hardware multi-core

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
│   ├── core/                       # Modelli ARIMA, SARIMA, VAR e selezione automatica
│   │   ├── arima_model.py         # Implementazione ARIMA base
│   │   ├── sarima_model.py        # Modelli SARIMA con stagionalità
│   │   ├── var_model.py           # Vector Autoregression multivariato
│   │   ├── model_selection.py     # Selezione automatica ARIMA
│   │   └── sarima_selection.py    # Selezione automatica SARIMA
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
│   └── utils/                      # Logging ed eccezioni personalizzate
├── docs/                           # Documentazione completa
│   ├── teoria_arima.md            # Teoria matematica ARIMA
│   ├── teoria_sarima.md           # Teoria matematica SARIMA
│   └── arima_vs_sarima.md         # Confronto dettagliato modelli
├── examples/                       # Script esempio pratici
│   └── advanced_forecasting_showcase.py  # Demo funzionalità avanzate
├── notebooks/                      # Jupyter notebooks per ricerca e sviluppo
│   └── research_and_development.ipynb # Ambiente R&D per sperimentazione algoritmi
├── scripts/                        # Script di utilità
│   ├── run_api.py                 # Lancia API server
│   └── run_dashboard.py           # Lancia dashboard Streamlit
├── tests/                          # Suite test completa
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

# Oppure installazione selettiva
pip install -e ".[api,dashboard,automl]"

# Verifica installazione
python -m pytest tests/ -v
```

#### Verifica Installazione Rapida

```bash
# Test delle funzionalità principali
uv run python examples/advanced_forecasting_showcase.py

# Lancia API (in background)
uv run python scripts/run_api.py &

# Lancia dashboard (nuovo terminale)
uv run python scripts/run_dashboard.py
```

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

#### 3. Forecasting con Variabili Esogene (SARIMAX)

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

#### 5. Auto-ML e Ottimizzazione Avanzata

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

#### 6. API REST - Client Python

```python
import requests
import json

# Training di un modello via API
api_url = "http://localhost:8000"

# Prepara dati per API
data_payload = {
    "data": {
        "timestamps": serie_pulita.index.strftime('%Y-%m-%d').tolist(),
        "values": serie_pulita.values.tolist()
    },
    "model_type": "sarima",
    "auto_select": True
}

# Invia richiesta training
response = requests.post(f"{api_url}/models/train", json=data_payload)
model_info = response.json()
model_id = model_info['model_id']

# Genera forecast via API
forecast_payload = {
    "steps": 12,
    "confidence_level": 0.95,
    "return_intervals": True
}

forecast_response = requests.post(
    f"{api_url}/models/{model_id}/forecast", 
    json=forecast_payload
)
forecast_result = forecast_response.json()
print("Forecast API:", forecast_result['forecast_values'][:3])
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

#### 8. Report Dinamici con Quarto

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

#### 📊 Dashboard Interattiva
- **Data Exploration**: Upload CSV, statistiche, visualizzazioni
- **Model Comparison**: Confronto performance modelli diversi
- **Interactive Plotting**: Grafici Plotly con zoom, filtering
- **Export Results**: Download forecast e report in CSV/PDF

#### 📄 Reporting Dinamico con Quarto
- **Report Automatici**: Template professionali con analisi integrate
- **Multi-Formato**: Export HTML, PDF, DOCX con un comando
- **Analisi Intelligenti**: Interpretazione automatica metriche e risultati
- **Visualizzazioni Embed**: Grafici integrati nei report
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
| **[ARIMA vs SARIMA](docs/arima_vs_sarima.md)** | Confronto dettagliato, scelta del modello, casi d'uso | Pratico |
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
uv run pytest tests/test_sarima.py tests/test_var.py -v

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

#### 🎉 Completate (v0.3.0)
- [x] **Modelli SARIMA**: Stagionalità completa con selezione automatica
- [x] **VAR Multivariato**: Forecasting serie multiple con causalità
- [x] **API REST**: Servizi production-ready con FastAPI
- [x] **Dashboard Streamlit**: Interfaccia web completa
- [x] **Auto-ML**: Ottimizzazione con Optuna, Hyperopt, Scikit-Optimize
- [x] **Ensemble Methods**: Combinazione intelligente modelli
- [x] **Report Quarto**: Generazione automatica report dinamici HTML/PDF/DOCX
- [x] **Documentazione**: Teoria completa ARIMA vs SARIMA

#### 🚧 In Sviluppo (v0.4.0)
- [ ] **SARIMAX**: Modelli con variabili esogene
- [ ] **LSTM Integration**: Hybrid ARIMA-Deep Learning
- [ ] **Real-time Streaming**: Apache Kafka integration
- [ ] **Cloud Native**: Kubernetes operators

#### 🔮 Future Releases
- [ ] **Prophet Integration**: Facebook Prophet models
- [ ] **Anomaly Detection**: Integrated outlier detection
- [ ] **MLOps Pipeline**: MLflow, DVC, Airflow integration  
- [ ] **Multi-tenancy**: Enterprise deployment features

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

# 3️⃣ Explore
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