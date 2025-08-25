# Test API FastAPI

Questa directory contiene tutti i test per l'API REST FastAPI del progetto ARIMA Forecaster.

## 📁 Struttura Test

```
tests/api/
├── conftest.py                  # Fixture condivise per test API
├── test_health_router.py        # Test endpoint health e status
├── test_training_router.py      # Test training modelli (ARIMA/SARIMA/VAR/Auto-selection)
├── test_forecasting_router.py   # Test generazione previsioni
├── test_models_router.py        # Test gestione modelli (CRUD)
├── test_diagnostics_router.py   # Test diagnostica modelli
├── test_reports_router.py       # Test generazione report
├── test_integration_e2e.py      # Test integrazione end-to-end
├── test_api_comprehensive.py    # Test comprensivi e edge cases
└── README.md                    # Questo file
```

## 🚀 Come Eseguire i Test

### Tutti i Test API
```bash
# Esegui tutti i test API
uv run pytest tests/api/ -v

# Con coverage
uv run pytest tests/api/ --cov=src/arima_forecaster --cov-report=html
```

### Test per Router Specifici
```bash
# Solo health endpoints
uv run pytest tests/api/test_health_router.py -v

# Solo training endpoints  
uv run pytest tests/api/test_training_router.py -v

# Solo forecasting
uv run pytest tests/api/test_forecasting_router.py -v
```

### Test per Funzionalità Specifiche
```bash
# Test marcati come "training"
uv run pytest tests/api/ -m training -v

# Test end-to-end
uv run pytest tests/api/ -m e2e -v

# Test performance
uv run pytest tests/api/ -m performance -v
```

### Test Paralleli (Più Veloce)
```bash
# Esegui test in parallelo
uv run pytest tests/api/ -n auto -v

# Parallelo con coverage
uv run pytest tests/api/ -n auto --cov=src/arima_forecaster
```

## 🏷️ Markers Disponibili

- `@pytest.mark.api` - Test generici API
- `@pytest.mark.health` - Test health check endpoints
- `@pytest.mark.training` - Test training modelli
- `@pytest.mark.forecasting` - Test forecasting
- `@pytest.mark.diagnostics` - Test diagnostica
- `@pytest.mark.reports` - Test report generation
- `@pytest.mark.models` - Test gestione modelli
- `@pytest.mark.e2e` - Test end-to-end integration
- `@pytest.mark.performance` - Test performance
- `@pytest.mark.concurrent` - Test operazioni concorrenti
- `@pytest.mark.edge_case` - Test edge cases
- `@pytest.mark.error_handling` - Test gestione errori

## 📊 Coverage Report

Dopo aver eseguito i test con coverage:

```bash
# Visualizza report HTML
start htmlcov/index.html  # Windows
open htmlcov/index.html   # macOS
xdg-open htmlcov/index.html  # Linux
```

## 🧪 Tipi di Test Implementati

### 1. **Test Health Router**
- Endpoint root (/)
- Health check (/health)
- Performance e stabilità
- Headers e formati risposta

### 2. **Test Training Router**
- Training ARIMA con successo/errori
- Training SARIMA con parametri stagionali
- Training VAR per serie multivariate
- Auto-selezione parametri ottimali
- Validazione dati input
- Training concorrente
- Gestione errori e edge cases

### 3. **Test Forecasting Router**
- Generazione previsioni con intervalli confidenza
- Diversi step e livelli confidenza
- Forecast senza intervalli
- Validazione parametri
- Performance e concorrenza
- Gestione modelli inesistenti

### 4. **Test Models Router**
- Lista modelli (vuota e popolata)
- Recupero info modello specifico
- Eliminazione modelli
- Workflow CRUD completo
- Gestione multipli modelli
- Persistenza metadati

### 5. **Test Diagnostics Router**
- Statistiche residui completa
- Test Ljung-Box autocorrelazione
- Test Jarque-Bera normalità
- Calcolo ACF/PACF
- Metriche performance
- Validazione range valori
- Consistenza risultati

### 6. **Test Reports Router**
- Generazione report HTML/PDF/DOCX
- Configurazione opzioni personalizzate
- Download report (protezione path traversal)
- Workflow generazione completo
- Validazione parametri
- Report concorrenti

### 7. **Test Integration E2E**
- Workflow completo ARIMA (train→forecast→diagnostics→report→delete)
- Workflow SARIMA con componenti stagionali
- Workflow VAR multivariato
- Auto-selezione → produzione
- Recovery da errori
- Operazioni concorrenti
- Test performance sotto carico

### 8. **Test Comprehensive**
- Schema OpenAPI validazione
- Accesso Swagger/ReDoc/Scalar UI
- Validazione JSON e content types
- Headers CORS e sicurezza
- Gestione errori 404/405/422/500
- Limiti e dati estremi
- Consistenza caching
- Metriche e monitoring

## 🛠️ Fixture Condivise (conftest.py)

### Fixture App e Client
- `test_app`: Istanza FastAPI per test
- `client`: TestClient per chiamate API
- `temp_model_dir`: Directory temporanea modelli

### Fixture Dati di Test
- `sample_time_series_data`: Dati serie temporale univariata
- `sample_multivariate_data`: Dati serie multivariate (VAR)
- `sample_arima_request`: Request training ARIMA
- `sample_sarima_request`: Request training SARIMA  
- `sample_var_request`: Request training VAR
- `sample_forecast_request`: Request forecasting
- `sample_auto_select_request`: Request auto-selezione
- `sample_report_request`: Request report generation

### Fixture Modelli Pre-addestrati
- `trained_model_id`: ID modello già addestrato per test

## ⚡ Ottimizzazioni Performance Test

### Test Paralleli
I test API sono progettati per esecuzione parallela sicura:
- Directory temporanee separate per ogni test
- Nessuna condivisione stato globale
- Cleanup automatico risorse

### Test Categorizzati
Usa marker per eseguire solo categorie specifiche:
```bash
# Solo test veloci (esclude e2e e performance)
uv run pytest tests/api/ -m "not e2e and not performance"

# Solo test funzionali (esclude edge cases)
uv run pytest tests/api/ -m "not edge_case"
```

## 🐛 Debug e Troubleshooting

### Verbose Output
```bash
# Massimo dettaglio
uv run pytest tests/api/ -vv -s

# Con output test failures completo
uv run pytest tests/api/ -vv --tb=long
```

### Test Specifico
```bash
# Singolo test
uv run pytest tests/api/test_training_router.py::TestModelTraining::test_train_arima_model_success -vv

# Test che matchano pattern
uv run pytest tests/api/ -k "training and success" -v
```

### Debug Performance
```bash
# Mostra test più lenti
uv run pytest tests/api/ --durations=20

# Profile memory usage (richiede pytest-memray)
uv run pytest tests/api/ --memray -v
```

## 📋 Checklist Test Coverage

✅ **Endpoint Coverage**: Tutti gli endpoint implementati  
✅ **Method Coverage**: GET, POST, DELETE per ogni router  
✅ **Success Cases**: Tutti i casi di successo  
✅ **Error Cases**: 404, 400, 422, 500 errors  
✅ **Validation**: Input validation e sanitization  
✅ **Edge Cases**: Dati limite, parametri estremi  
✅ **Performance**: Tempi risposta e concorrenza  
✅ **Security**: Headers, CORS, path traversal  
✅ **Integration**: Workflow end-to-end completi  
✅ **Documentation**: Schema OpenAPI e UI access  

## 🎯 Metriche Target

- **Coverage**: >90% line coverage
- **Performance**: Health endpoints <100ms, altri <2s  
- **Stability**: Zero test flaky, 100% pass rate
- **Completeness**: Tutti gli endpoint e metodi HTTP testati