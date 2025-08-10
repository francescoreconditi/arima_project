# Funzionalità Avanzate di Forecasting ARIMA

Questo documento descrive le funzionalità avanzate aggiunte alla libreria ARIMA Forecaster, fornendo strumenti completi per il forecasting moderno delle serie temporali.

## 🌊 Modelli SARIMA (ARIMA Stagionale)

### Panoramica
I modelli SARIMA (Seasonal AutoRegressive Integrated Moving Average) estendono ARIMA per gestire pattern stagionali nelle serie temporali.

### Caratteristiche Chiave
- **Parametri Stagionali**: (P, D, Q, s) per AR stagionale, differenziazione, MA e periodo
- **Selezione Automatica**: Grid search per parametri stagionali ottimali
- **Decomposizione Stagionale**: Analisi di decomposizione integrata
- **Periodi Stagionali Multipli**: Supporto per diversi cicli stagionali

### Esempio di Utilizzo
```python
from arima_forecaster import SARIMAForecaster, SARIMAModelSelector

# Specifica manuale SARIMA
model = SARIMAForecaster(
    order=(1, 1, 1),           # Non-stagionale (p, d, q)
    seasonal_order=(1, 1, 1, 12)  # Stagionale (P, D, Q, s)
)
model.fit(data)
forecast = model.forecast(steps=12)

# Selezione automatica SARIMA
selector = SARIMAModelSelector(
    seasonal_periods=[12, 4],  # Pattern mensili e trimestrali
    max_models=50
)
selector.search(data)
best_model = selector.get_best_model()
```

### Applicazioni
- **Dati Vendite Mensili**: Stagionalità annuale (s=12)
- **Traffico Giornaliero**: Stagionalità settimanale (s=7)
- **Guadagni Trimestrali**: Stagionalità annuale (s=4)
- **Consumo Energetico Orario**: Pattern giornalieri (s=24) e settimanali (s=168)

## 📈 Modelli Vector Autoregression (VAR)

### Panoramica
I modelli VAR gestiscono serie temporali multivariate dove più variabili si influenzano reciprocamente nel tempo.

### Caratteristiche Chiave
- **Forecasting Multivariato**: Previsione di più variabili correlate simultaneamente
- **Selezione Lag**: Determinazione automatica del lag ottimale
- **Test di Causalità**: Test di causalità di Granger tra variabili
- **Impulse Response**: Analisi delle interazioni tra variabili
- **Test di Cointegrazione**: Analisi delle relazioni a lungo termine

### Esempio di Utilizzo
```python
from arima_forecaster import VARForecaster
import pandas as pd

# Prepara dati multivariati
data = pd.DataFrame({
    'vendite': sales_data,
    'spesa_marketing': marketing_data,
    'indice_competitor': competitor_data
})

# Addestra modello VAR
model = VARForecaster(maxlags=4)
model.fit(data)

# Genera forecast multivariato
forecast = model.forecast(steps=6)
print(forecast['forecast'])  # Previsioni per tutte le variabili

# Analizza relazioni tra variabili
causalità = model.granger_causality('vendite', ['spesa_marketing'])
impulse_resp = model.impulse_response(periods=10)
```

### Analisi Avanzate
- **Funzioni Impulse Response**: Come gli shock in una variabile influenzano le altre
- **Decomposizione Varianza Errore di Previsione**: Contributo di ogni variabile alla varianza della previsione
- **Causalità di Granger**: Test statistici per relazioni causali
- **Cointegrazione**: Relazioni di equilibrio a lungo termine

## 🤖 Ottimizzazione Auto-ML degli Iperparametri

### Panoramica
Algoritmi di ottimizzazione avanzati trovano automaticamente i parametri ottimali del modello utilizzando tecniche all'avanguardia.

### Algoritmi Supportati
- **Optuna**: Tree-structured Parzen Estimator (TPE)
- **Hyperopt**: Ottimizzazione Bayesiana
- **Scikit-Optimize**: Ottimizzazione con Gaussian Process

### Caratteristiche di Ottimizzazione
- **Multi-Obiettivo**: Ottimizza più metriche simultaneamente
- **Cross-Validation**: Validazione consapevole delle serie temporali
- **Early Stopping**: Previene l'overfitting
- **Processamento Parallelo**: Accelera l'ottimizzazione

### Esempio di Utilizzo
```python
from arima_forecaster.automl import ARIMAOptimizer, optimize_model

# Ottimizzazione singolo obiettivo
optimizer = ARIMAOptimizer(objective_metric='aic')
result = optimizer.optimize_optuna(data, n_trials=100)

print(f"Parametri migliori: {result['best_params']}")
print(f"Score migliore: {result['best_score']}")

# Funzione di convenienza per qualsiasi tipo di modello
result = optimize_model(
    model_type='sarima',
    data=data,
    optimizer_type='optuna',
    n_trials=50
)
```

### Obiettivi di Ottimizzazione
- **Criteri Informativi**: AIC, BIC, HQIC
- **Accuratezza Forecast**: MSE, MAE, MAPE
- **Metriche Personalizzate**: Funzioni obiettivo definite dall'utente

## 🎯 Tuning Avanzato degli Iperparametri

### Ottimizzazione Multi-Obiettivo
Ottimizza più obiettivi competitivi simultaneamente utilizzando l'ottimizzazione di Pareto.

```python
from arima_forecaster.automl import HyperparameterTuner

tuner = HyperparameterTuner(
    objective_metrics=['aic', 'bic', 'mse'],
    ensemble_method='weighted_average'
)

result = tuner.multi_objective_optimization('arima', data, n_trials=100)
pareto_front = result['pareto_front']
soluzione_migliore = result['best_solution']
```

### Metodi Ensemble
Crea ensemble di modelli diversi per migliori prestazioni di forecasting.

```python
# Crea ensemble di modelli diversi
ensemble_result = tuner.ensemble_optimization(
    'arima', data, 
    n_models=5, 
    diversity_threshold=0.2
)

# Genera forecast ensemble
forecast = tuner.forecast_ensemble(
    steps=12, 
    method='weighted',
    confidence_level=0.95
)
```

### Ottimizzazione Adattiva
Regola dinamicamente lo spazio di ricerca basandosi sui progressi dell'ottimizzazione.

```python
# Ottimizzazione adattiva con early stopping
adaptive_result = tuner.adaptive_optimization(
    'sarima', data,
    max_iterations=10,
    improvement_threshold=0.01
)
```

## 🌐 API REST per Servizi di Forecasting

### Panoramica
API REST pronta per la produzione per distribuire modelli di forecasting come servizi web.

### Endpoint Principali
- `POST /models/train`: Addestra modelli ARIMA/SARIMA
- `POST /models/train/var`: Addestra modelli VAR
- `POST /models/{id}/forecast`: Genera previsioni
- `POST /models/auto-select`: Selezione automatica del modello
- `GET /models`: Elenca tutti i modelli
- `POST /models/{id}/diagnostics`: Diagnostica del modello

### Esempio di Utilizzo
```bash
# Avvia server API
python scripts/run_api.py --host 0.0.0.0 --port 8000

# Addestra un modello
curl -X POST "http://localhost:8000/models/train" \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "timestamps": ["2023-01-01", "2023-02-01"],
      "values": [100, 105]
    },
    "model_type": "arima",
    "order": {"p": 1, "d": 1, "q": 1}
  }'

# Genera previsioni
curl -X POST "http://localhost:8000/models/{model_id}/forecast" \
  -H "Content-Type: application/json" \
  -d '{
    "steps": 12,
    "confidence_level": 0.95,
    "return_intervals": true
  }'
```

### Caratteristiche API
- **Processamento Asincrono**: Addestramento modelli in background
- **Persistenza Modelli**: Archiviazione e recupero automatico dei modelli
- **Validazione Input**: Validazione richieste/risposte basata su Pydantic
- **Gestione Errori**: Risposte di errore complete
- **Documentazione API**: Docs OpenAPI/Swagger generate automaticamente

## 📊 Dashboard Interattiva Streamlit

### Panoramica
Interfaccia web user-friendly per esplorare dati, addestrare modelli e generare previsioni.

### Caratteristiche Dashboard
1. **Upload Dati**: Caricamento file CSV con mappatura colonne
2. **Esplorazione Dati**: Grafici interattivi e statistiche
3. **Preprocessing**: Valori mancanti, outlier, stazionarità
4. **Addestramento Modelli**: Tutti i tipi di modello con tuning parametri
5. **Forecasting**: Generazione previsioni interattiva
6. **Diagnostica Modelli**: Analisi residui e test

### Utilizzo
```bash
# Lancia dashboard
python scripts/run_dashboard.py

# Accedi a http://localhost:8501
```

### Pagine Dashboard
- **Upload Dati**: Carica e anteprima dati serie temporali
- **Esplorazione Dati**: Visualizza e analizza pattern dei dati
- **Addestramento Modelli**: Addestra e confronta modelli diversi
- **Forecasting**: Genera e visualizza previsioni
- **Diagnostica Modelli**: Valuta performance dei modelli

## 🚀 Installazione e Setup

### Dipendenze
Installa la libreria con tutte le funzionalità avanzate:

```bash
# Installa con tutte le dipendenze opzionali
pip install -e ".[all]"

# Oppure installa gruppi di funzionalità specifici
pip install -e ".[api]"      # Funzionalità API
pip install -e ".[dashboard]" # Funzionalità Dashboard  
pip install -e ".[automl]"    # Funzionalità Auto-ML
```

### Usando UV (Raccomandato)
```bash
# Sincronizza tutte le dipendenze
uv sync --all-extras

# Attiva ambiente
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

## 📈 Benchmark delle Prestazioni

### Velocità Addestramento Modelli
- **ARIMA**: ~0.1-1.0 secondi per modello
- **SARIMA**: ~0.5-5.0 secondi per modello
- **VAR**: ~0.1-2.0 secondi per modello
- **Auto-ML**: ~10-300 secondi per ottimizzazione completa

### Efficienza Ottimizzazione
- **Grid Search**: Prestazioni baseline
- **Optuna TPE**: Convergenza 2-5x più veloce
- **Multi-obiettivo**: 10-50 soluzioni Pareto
- **Ensemble**: 3-7 modelli diversi

### Utilizzo Memoria
- **Singolo Modello**: 1-10 MB
- **Ensemble (5 modelli)**: 5-50 MB
- **Server API**: 50-200 MB memoria base
- **Dashboard**: 100-300 MB inclusa UI

## 🔧 Configurazione Avanzata

### Impostazioni Ottimizzazione
```python
# Configurazione ottimizzazione personalizzata
optimizer = ARIMAOptimizer(
    objective_metric='aic',
    cv_folds=3,
    test_size=0.2,
    n_jobs=4,  # Processamento parallelo
    random_state=42
)

# Impostazioni tuner avanzate
tuner = HyperparameterTuner(
    objective_metrics=['aic', 'bic', 'mse'],
    ensemble_method='pareto',
    meta_learning=True,
    early_stopping_patience=10
)
```

### Configurazione API
```python
# Configurazione API personalizzata
from arima_forecaster.api import create_app

app = create_app(model_storage_path="/path/to/models")

# Esegui con impostazioni personalizzate
import uvicorn
uvicorn.run(app, host="0.0.0.0", port=8080, workers=4)
```

## 🧪 Testing e Validazione

### Test dei Modelli
```bash
# Esegui tutti i test incluse le funzionalità avanzate
uv run pytest tests/ -v --cov=src/arima_forecaster

# Testa moduli specifici
uv run pytest tests/test_sarima.py -v
uv run pytest tests/test_var.py -v
uv run pytest tests/test_automl.py -v
```

### Test API
```bash
# Avvia server di test
python scripts/run_api.py --host localhost --port 8001

# Esegui test API
python -m pytest tests/test_api.py -v
```

## 📚 Esempi e Tutorial

### Esempi Completi
- `examples/advanced_forecasting_showcase.py`: Dimostrazione completa delle funzionalità
- `examples/api_client_example.py`: Utilizzo client API
- `examples/dashboard_demo.py`: Walkthrough funzionalità dashboard
- `examples/automl_tutorial.py`: Guida ottimizzazione Auto-ML

### Notebook Jupyter
- `notebooks/sarima_analysis.ipynb`: Modellazione stagionale
- `notebooks/var_multivariate.ipynb`: Esplorazione modelli VAR
- `notebooks/automl_comparison.ipynb`: Algoritmi di ottimizzazione
- `notebooks/ensemble_forecasting.ipynb`: Metodi ensemble

## 🤝 Contribuire

### Aggiungere Nuove Funzionalità
1. Crea branch feature
2. Implementa con test
3. Aggiorna documentazione
4. Aggiungi esempi
5. Invia pull request

### Estendere Ottimizzatori
```python
class CustomOptimizer(BaseOptimizer):
    def optimize_custom(self, series, **kwargs):
        # Implementa logica di ottimizzazione personalizzata
        pass
```

### Estensioni API
```python
@app.post("/custom-endpoint")
async def custom_forecasting_endpoint(request: CustomRequest):
    # Implementa endpoint API personalizzato
    pass
```

## 📄 Licenza

Questo progetto è rilasciato sotto Licenza MIT - vedi il file LICENSE per i dettagli.

## 🙏 Riconoscimenti

- **Optuna**: Framework di ottimizzazione avanzata
- **FastAPI**: Framework API moderno
- **Streamlit**: Applicazioni web interattive
- **Statsmodels**: Fondamento per modellazione statistica
- **Plotly**: Visualizzazioni interattive

---

Per maggiori informazioni, consulta la documentazione completa o esegui l'esempio showcase:

```bash
python examples/advanced_forecasting_showcase.py
```