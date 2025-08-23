# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⚠️ PROCEDURE DI SICUREZZA OBBLIGATORIE ⚠️

### 🔴 Backup Automatico Pre-Modifica
**PRIMA di qualsiasi modifica a qualsiasi file:**
1. **SEMPRE** copiare il file originale in `C:\Backup\Code\{nome_progetto}\`
2. Mantenere la struttura delle directory originale
3. Rinominare il file aggiungendo in fondo al file la data e ora (es. `file.py.backup_20240427_1530`)
4. **SOLO DOPO** il backup completato, procedere con le modifiche
5. Esempio: `cp src/file.py C:\Backup\Code\arima_project\src\file.py.backup`

### 🔴 Gestione File Grandi (>50KB o errore "token limit")
**Quando appare "superato il numero di token" o file >50KB:**
1. **MAI** modifiche massive su file grandi
2. **SEMPRE** leggere tutto il file a sezioni PRIMA (con Read offset/limit)  
3. **SOLO** modifiche incrementali piccole e verificate (<50 righe)
4. **Test immediato** dopo ogni singola modifica
5. **MAI** sostituire blocchi >50 righe senza mappare completamente il file

### 🔴 Regola d'Oro Inviolabile
```
Un errore = danni ai clienti = soldi persi
SEMPRE: Backup PRIMA → Modifiche piccole → Test subito
```

### 🔴 Procedura di Emergenza
Se qualcosa va storto:
1. **STOP** immediatamente
2. **NON** tentare riparazioni
3. **Avvisare** l'utente del problema
4. Attendere istruzioni per ripristino da backup

### 🔴 Commenti nei Sorgenti
Quando possibile SEMPRE scrivere commenti nei sorgenti in Italiano.

## Panoramica del Progetto

Libreria Python avanzata per forecasting serie temporali con modelli ARIMA, SARIMA e VAR. Include funzionalità enterprise-grade come Auto-ML, API REST, dashboard interattiva e reporting dinamico con Quarto.

## Comandi di Sviluppo Essenziali

### Setup Rapido
```bash
# Installa UV (10x più veloce di pip)
curl -LsSf https://astral.sh/uv/install.sh | sh  # Linux/macOS
# winget install --id=astral-sh.uv  # Windows

# Setup completo con Just
just setup  # Installa dipendenze e configura ambiente

# Oppure manualmente
uv sync --all-extras
```

### Testing
```bash
# Tutti i test
uv run pytest tests/ -v

# Con coverage
uv run pytest tests/ --cov=src/arima_forecaster --cov-report=html

# Test specifici
uv run pytest tests/test_arima_model.py -v
uv run pytest tests/test_sarima_model.py -v  
uv run pytest tests/test_reporting.py -v

# Test paralleli (veloce)
uv run pytest tests/ -v -n auto
```

### Qualità del Codice
```bash
# Formattazione
uv run black src/ tests/ examples/
uv run ruff format src/ tests/ examples/

# Linting
uv run ruff check src/ tests/ examples/
uv run mypy src/arima_forecaster/

# Tutti i controlli
just check  # O: uv run pre-commit run --all-files
```

### Servizi Production
```bash
# API REST (FastAPI)
uv run python scripts/run_api.py
# Swagger UI: http://localhost:8000/docs

# Dashboard Web (Streamlit)  
uv run python scripts/run_dashboard.py
# URL: http://localhost:8501

# Script training
uv run python scripts/train.py --data path/to/data.csv --model sarima

# Script forecasting
uv run python scripts/forecast.py --model path/to/model.pkl --steps 30
```

## Architettura del Codice

### Moduli Principali

#### Modelli Core (`src/arima_forecaster/core/`)
- **ARIMAForecaster** (`arima_model.py`): Modello ARIMA base con parametri (p,d,q)
- **SARIMAForecaster** (`sarima_model.py`): ARIMA stagionale con (P,D,Q,s) 
- **VARForecaster** (`var_model.py`): Vector Autoregression per serie multivariate
- **ARIMAModelSelector** (`model_selection.py`): Grid search automatico per ARIMA
- **SARIMAModelSelector** (`sarima_selection.py`): Selezione automatica SARIMA

#### Data Processing (`src/arima_forecaster/data/`)
- **DataLoader**: Caricamento CSV con validazione automatica
- **TimeSeriesPreprocessor**: Pipeline preprocessing configurabile
  - Gestione valori mancanti: interpolate, drop, forward_fill, backward_fill
  - Rilevamento outlier: IQR, z-score, modified z-score
  - Stazionarietà: difference, log_difference, test ADF/KPSS

#### Valutazione (`src/arima_forecaster/evaluation/`)
- **ModelEvaluator**: 15+ metriche (MAE, RMSE, MAPE, sMAPE, MASE, etc.)
- Diagnostica residui completa (Ljung-Box, Jarque-Bera, ACF/PACF)
- Test statistici e analisi performance

#### Visualizzazione (`src/arima_forecaster/visualization/`)
- **ForecastPlotter**: Grafici forecast con intervalli confidenza
- Dashboard interattivi con decomposizione stagionale
- Analisi residui multi-pannello

#### Reporting (`src/arima_forecaster/reporting/`)
- **QuartoReportGenerator**: Report dinamici HTML/PDF/DOCX
- Template personalizzabili con analisi automatiche
- Comparazione modelli side-by-side

#### Auto-ML (`src/arima_forecaster/automl/`)
- **HyperparameterOptimizer**: Ottimizzazione con Optuna/Hyperopt
- **ModelTuner**: Tuning avanzato multi-obiettivo
- Ensemble methods e stacking

#### API & Dashboard
- **FastAPI REST API** (`src/arima_forecaster/api/`): Endpoints production-ready
- **Streamlit Dashboard** (`src/arima_forecaster/dashboard/`): UI web interattiva

### Pipeline Dati Tipica

1. **Caricamento**: `DataLoader.load_data()` con validazione
2. **Preprocessing**: `TimeSeriesPreprocessor.preprocess_pipeline()`
3. **Selezione Modello**: `ARIMAModelSelector.search()` o manuale
4. **Training**: `model.fit(series)` con metadata
5. **Valutazione**: `ModelEvaluator.evaluate()`
6. **Visualizzazione**: `ForecastPlotter.create_dashboard()`
7. **Reporting**: `QuartoReportGenerator.generate_report()`
8. **Deployment**: API REST o dashboard web

### Pattern Import Consigliati

```python
# Import base
from arima_forecaster import (
    ARIMAForecaster, 
    SARIMAForecaster,
    TimeSeriesPreprocessor,
    ForecastPlotter
)

# Import avanzati
from arima_forecaster.core import ARIMAModelSelector, VARForecaster
from arima_forecaster.evaluation import ModelEvaluator
from arima_forecaster.reporting import QuartoReportGenerator  # Richiede [reports]
from arima_forecaster.automl import HyperparameterOptimizer  # Richiede [automl]
```

### Gestione Errori

```python
from arima_forecaster.utils.exceptions import (
    ModelTrainingError,
    ForecastError, 
    DataProcessingError
)

try:
    model = SARIMAForecaster(order=(1,1,1), seasonal_order=(1,1,1,12))
    model.fit(series)
except ModelTrainingError as e:
    logger.error(f"Training failed: {e}")
```

## Considerazioni Performance

- Usa `validate_input=False` per batch processing dopo validazione iniziale
- Cache preprocessing per training ripetuto (`preprocessor.cache_results=True`)
- Elaborazione parallela: `selector.search(n_jobs=-1)`
- Limita spazio ricerca: `selector.search(max_models=100)`

## Workflow con Just

```bash
just setup       # Setup iniziale ambiente
just test-cov    # Test con coverage
just format      # Formatta codice
just lint        # Controlli qualità
just check       # Tutti i controlli
just examples    # Esegui esempi
just clean       # Pulizia file temporanei
just build       # Build package per distribuzione
```

## Note Implementative

### Modelli Disponibili
- **ARIMA**: Serie univariate non stagionali
- **SARIMA**: Serie con componente stagionale (nuovo)
- **VAR**: Serie temporali multivariate (nuovo)
- **Auto-ARIMA**: Selezione automatica parametri ottimali

### Features Avanzate Verificate
- ✅ SARIMA con decomposizione stagionale automatica
- ✅ VAR con test causalità Granger e impulse response
- ✅ Auto-ML con Optuna, Hyperopt, Scikit-Optimize
- ✅ API REST production-ready con FastAPI
- ✅ Dashboard Streamlit interattiva
- ✅ Reporting Quarto con export multi-formato
- ✅ Ensemble methods e model stacking

### Directory Output
- `outputs/models/`: Modelli serializzati (.pkl)
- `outputs/plots/`: Visualizzazioni (.png, .html)  
- `outputs/reports/`: Report Quarto (.html, .pdf, .docx)
- `logs/`: File di log applicazione

## Dipendenze Chiave

- **statsmodels**: Implementazione modelli ARIMA/SARIMA/VAR
- **pandas/numpy**: Manipolazione dati e calcoli
- **matplotlib/seaborn/plotly**: Visualizzazioni
- **fastapi/uvicorn**: API REST
- **streamlit**: Dashboard web
- **quarto**: Report dinamici (opzionale)
- **optuna/hyperopt**: Auto-ML (opzionale)