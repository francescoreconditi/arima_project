# CLAUDE.md

Questo file fornisce indicazioni a Claude Code (claude.ai/code) per lavorare con il codice in questo repository.

## Panoramica del Progetto

Questa è una libreria completa per il forecasting di serie temporali ARIMA (Autoregressive Integrated Moving Average) sviluppata in Python. Il progetto fornisce un approccio professionale e modulare per l'analisi delle serie temporali, addestramento dei modelli, valutazione e forecasting con documentazione estesa e best practices.

## Struttura del Progetto

```
arima_project/
├── src/arima_forecaster/          # Codice sorgente del package principale
│   ├── core/                      # Modellazione ARIMA di base
│   │   ├── arima_model.py         # Classe principale ARIMAForecaster
│   │   └── model_selection.py     # Selezione automatica del modello
│   ├── data/                      # Gestione dati e preprocessing
│   │   ├── loader.py              # Utilità per caricamento dati
│   │   └── preprocessor.py        # Preprocessing serie temporali
│   ├── evaluation/                # Valutazione modelli e metriche
│   │   └── metrics.py             # Metriche di valutazione complete
│   ├── visualization/             # Grafici e visualizzazione
│   │   └── plotter.py             # Utilità grafiche avanzate
│   └── utils/                     # Utilità e helper
│       ├── logger.py              # Configurazione logging
│       └── exceptions.py          # Eccezioni personalizzate
├── docs/                          # Documentazione
│   ├── teoria_arima.md            # Teoria ARIMA completa
│   └── guida_utente.md            # Guida pratica all'uso
├── examples/                      # Script di esempio
│   ├── forecasting_base.py        # Esempio uso base
│   └── selezione_automatica.py    # Selezione modello avanzata
├── tests/                         # Suite di test completa
├── data/                          # Directory dati
│   └── processed/                 # Dataset processati
├── outputs/                       # Output generati
│   ├── models/                    # Modelli salvati
│   ├── plots/                     # Visualizzazioni generate
│   └── reports/                   # Report Quarto generati
└── config/                        # File di configurazione
```

## Comandi di Sviluppo Comuni

### Configurazione Ambiente

#### Con UV (Raccomandato)
```bash
# Installa uv se non installato
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sincronizza ambiente e installa tutte le dipendenze
uv sync --all-extras

# Attiva ambiente virtuale
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Configura directory logging
mkdir -p logs
```

#### Con pip (Alternativa)
```bash
# Installa dipendenze
pip install -r requirements.txt

# Installa package in modalità sviluppo con tutti gli extra
pip install -e ".[all]"

# Configura directory logging
mkdir -p logs
```

### Testing

#### Con UV
```bash
# Esegui tutti i test
uv run pytest tests/ -v

# Esegui test con coverage
uv run pytest tests/ --cov=src/arima_forecaster --cov-report=html

# Esegui moduli di test specifici
uv run pytest tests/test_arima_model.py -v
uv run pytest tests/test_preprocessing.py -v

# Esegui test per funzionalità specifiche
uv run pytest -k "test_forecast" -v

# Test paralleli per velocità
uv run pytest tests/ -v -n auto
```

#### Con python tradizionale
```bash
python -m pytest tests/ -v
python -m pytest tests/ --cov=src/arima_forecaster --cov-report=html
```

### Qualità del Codice

#### Con UV
```bash
# Formatta codice
uv run black src/ tests/ examples/

# Lint codice (ruff è più veloce di flake8)
uv run ruff check src/ tests/ examples/
uv run ruff format src/ tests/ examples/  # formatting alternativo a black

# Type checking
uv run mypy src/arima_forecaster/

# Pre-commit hooks (tutto insieme)
uv run pre-commit run --all-files
```

#### Comandi tradizionali
```bash
black src/ tests/ examples/
ruff check src/ tests/ examples/
mypy src/arima_forecaster/
```

### Esecuzione Esempi

#### Con UV
```bash
# Esempio forecasting di base
uv run python examples/forecasting_base.py

# Esempio selezione automatica modello
uv run python examples/selezione_automatica.py

# Esempio reporting Quarto (richiede dipendenze reports)
uv run python examples/quarto_reporting.py
```

#### Tradizionale
```bash
python examples/forecasting_base.py
python examples/selezione_automatica.py
python examples/quarto_reporting.py
```

## Architettura del Codice

### Classi Principali e Loro Utilizzo

#### ARIMAForecaster (`src/arima_forecaster/core/arima_model.py`)
Classe principale per operazioni modello ARIMA:
- `ARIMAForecaster(order=(p,d,q))`: Inizializza modello con ordine specifico
- `fit(series)`: Addestra modello su dati serie temporali
- `forecast(steps, confidence_intervals=True)`: Genera previsioni
- `save(path)` / `load(path)`: Persistenza modello
- `get_model_info()`: Informazioni complete del modello

#### ARIMAModelSelector (`src/arima_forecaster/core/model_selection.py`)
Selezione automatica modello usando grid search:
- `search(series, verbose=True)`: Trova ordine ARIMA ottimale
- `get_results_summary(top_n=10)`: Ottieni top modelli performanti
- `plot_selection_results()`: Visualizza processo selezione

#### TimeSeriesPreprocessor (`src/arima_forecaster/data/preprocessor.py`)
Utilità preprocessing complete:
- `preprocess_pipeline()`: Preprocessing completo con tutte le opzioni
- `handle_missing_values()`: Strategie multiple per valori mancanti
- `remove_outliers()`: Vari metodi rilevamento outlier
- `check_stationarity()` / `make_stationary()`: Gestione stazionarietà

#### ModelEvaluator (`src/arima_forecaster/evaluation/metrics.py`)
Capacità valutazione estese:
- `calculate_forecast_metrics()`: 15+ metriche accuratezza forecast
- `evaluate_residuals()`: Diagnostica residui completa
- `generate_evaluation_report()`: Valutazione modello completa

#### ForecastPlotter (`src/arima_forecaster/visualization/plotter.py`)
Strumenti visualizzazione avanzati:
- `plot_forecast()`: Grafici forecast con intervalli confidenza
- `plot_residuals()`: Analisi residui 6-pannelli
- `plot_acf_pacf()`: Analisi autocorrelazione
- `create_dashboard()`: Dashboard forecasting completo

#### QuartoReportGenerator (`src/arima_forecaster/reporting/generator.py`)
Generazione report dinamici con Quarto:
- `generate_model_report()`: Report completo per singolo modello (ARIMA/SARIMA)
- `create_comparison_report()`: Report comparativo tra modelli multipli
- Supporto export HTML, PDF, DOCX
- Template personalizzabili con analisi automatiche

### Caratteristiche Chiave

1. **Gestione Errori Robusta**: Gerarchia eccezioni personalizzate con messaggi dettagliati
2. **Logging Completo**: Logging configurabile attraverso tutto il package
3. **Testing Esteso**: Test unitari per tutte le funzionalità principali
4. **Type Hints**: Annotazioni tipo complete per migliore supporto IDE
5. **Documentazione**: Sia teorica (teoria ARIMA) che pratica (guida utente)
6. **Best Practices**: Segue standard packaging Python e sviluppo

### Architettura Pipeline Dati

1. **Caricamento Dati**: `DataLoader` gestisce file CSV con validazione
2. **Preprocessing**: `TimeSeriesPreprocessor` applica preprocessing configurabile
3. **Selezione Modello**: `ARIMAModelSelector` trova parametri ottimali
4. **Addestramento**: `ARIMAForecaster` addestra modelli con metadata complete
5. **Valutazione**: `ModelEvaluator` fornisce analisi performance dettagliate
6. **Visualizzazione**: `ForecastPlotter` crea grafici pronti per pubblicazione
7. **Reporting**: `QuartoReportGenerator` genera report dinamici e interattivi
8. **Persistenza**: Modelli salvati con metadata per riproducibilità

## Configurazione e Personalizzazione

### Configurazione Logging
```python
from arima_forecaster.utils import setup_logger

logger = setup_logger(
    name='mio_forecaster',
    level='INFO',
    log_file='logs/forecasting.log'
)
```

### Opzioni Preprocessing
- **Valori Mancanti**: 'interpolate', 'drop', 'forward_fill', 'backward_fill'
- **Rilevamento Outlier**: 'iqr', 'zscore', 'modified_zscore'
- **Stazionarietà**: 'difference', 'log_difference'

### Criteri Selezione Modello
- **Criteri Informativi**: 'aic', 'bic', 'hqic'
- **Range Personalizzati**: Range p, d, q configurabili
- **Ottimizzazione Performance**: Supporto elaborazione parallela

## Best Practices di Sviluppo

### Struttura Import
```python
# Classi principali
from arima_forecaster import ARIMAForecaster, SARIMAForecaster, TimeSeriesPreprocessor, ForecastPlotter

# Funzionalità specializzate
from arima_forecaster.core import ARIMAModelSelector, SARIMAModelSelector
from arima_forecaster.evaluation import ModelEvaluator
from arima_forecaster.data import DataLoader

# Reporting (opzionale - richiede dipendenze reports)
from arima_forecaster.reporting import QuartoReportGenerator
```

### Pattern Gestione Errori
```python
from arima_forecaster.utils.exceptions import (
    ModelTrainingError, 
    ForecastError,
    DataProcessingError
)

try:
    model = ARIMAForecaster(order=(1,1,1))
    model.fit(series)
except ModelTrainingError as e:
    logger.error(f"Addestramento fallito: {e}")
    # Gestisci con grazia
```

### Workflow Tipico
1. Carica dati con `DataLoader`
2. Preprocessa con `TimeSeriesPreprocessor`
3. Seleziona modello con `ARIMAModelSelector`/`SARIMAModelSelector` o specifica manualmente
4. Addestra con `ARIMAForecaster`/`SARIMAForecaster`
5. Valuta con `ModelEvaluator`
6. Visualizza con `ForecastPlotter`
7. Genera report con `QuartoReportGenerator` (opzionale)
8. Genera previsioni e salva risultati

### Workflow Reporting Avanzato
1. Addestra modelli multipli (ARIMA, SARIMA, etc.)
2. Crea visualizzazioni con `ForecastPlotter`
3. Genera report individuali con `model.generate_report()`
4. Crea report comparativo con `QuartoReportGenerator.create_comparison_report()`
5. Esporta in formati multipli (HTML, PDF, DOCX)

## Dipendenze Chiave e Loro Ruoli

- **statsmodels**: Implementazione modelli ARIMA, test statistici
- **pandas**: Manipolazione dati serie temporali, gestione datetime
- **numpy**: Calcoli numerici, operazioni array
- **matplotlib/seaborn**: Visualizzazione e grafici
- **scipy**: Funzioni statistiche e test
- **scikit-learn**: Utilità ML aggiuntive e metriche
- **quarto** (opzionale): Generazione report dinamici e pubblicazione
- **jupyter** (opzionale): Supporto notebook per report interattivi

## Considerazioni Performance

- Usa `validate_input=False` per elaborazione batch dopo validazione iniziale
- Cache risultati preprocessing per addestramento modelli ripetuto
- Considera elaborazione parallela per selezione modello su grandi griglie parametri
- Usa parametro `max_models` per limitare spazio ricerca per risultati più veloci

## Testing e Assicurazione Qualità

Il progetto include testing completo:
- **Test Unitari**: Tutte le funzionalità principali testate
- **Test Integrazione**: Testing workflow end-to-end
- **Fixtures**: Dati test realistici con varie caratteristiche
- **Coverage**: Coverage test alta con reporting dettagliato

Esegui test prima di apportare modifiche:
```bash
python -m pytest tests/ -v --cov=src/arima_forecaster
```