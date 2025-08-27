# ARIMA Forecaster API - Guida Completa per Sviluppatori

## Indice
- [Panoramica Generale](#panoramica-generale)
- [Architettura Sistema](#architettura-sistema)
- [Installazione e Setup](#installazione-e-setup)
- [Autenticazione](#autenticazione)
- [Endpoints di Reference](#endpoints-di-reference)
- [Modelli Dati](#modelli-dati)
- [Esempi Pratici](#esempi-pratici)
- [Gestione Errori](#gestione-errori)
- [Performance e Scalabilità](#performance-e-scalabilita)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

---

## Panoramica Generale

ARIMA Forecaster API è un sistema enterprise-grade per time series forecasting che fornisce un'interfaccia REST completa per l'addestramento, valutazione e utilizzo di modelli statistici avanzati.

### 🚀 Caratteristiche Principali

- **Modelli Supportati**: ARIMA, SARIMA, SARIMAX, VAR
- **Auto-ML**: Selezione automatica parametri ottimali
- **Background Processing**: Training asincrono non bloccante
- **Multilingue**: Supporto 5 lingue (IT, EN, ES, FR, ZH)
- **Report Professionali**: HTML, PDF, DOCX con Quarto
- **Diagnostica Avanzata**: Test statistici e metriche complete
- **CORS**: Configurazione cross-origin per frontend

### 🏗️ Stack Tecnologico

- **Framework**: FastAPI 0.104+
- **Engine ML**: Statsmodels, Scikit-learn
- **Database**: File system (models/, reports/)
- **Visualization**: Plotly, Matplotlib
- **Report**: Quarto
- **Async**: Uvicorn ASGI server

---

## Architettura Sistema

### Struttura Modulare

L'API è organizzata in 6 router specializzati:

```
src/arima_forecaster/api/
├── main.py              # App factory e configurazione
├── models.py            # Modelli Pydantic base
├── models_extra.py      # Modelli avanzati
├── services/            # Business logic
│   ├── model_manager.py
│   └── forecast_service.py
└── routers/            # Endpoint controllers
    ├── health.py        # 🏥 Health checks
    ├── training.py      # 🎨 Training modelli
    ├── forecasting.py   # 📈 Generazione previsioni
    ├── models.py        # 📁 Gestione CRUD
    ├── diagnostics.py   # 🔍 Analisi performance
    └── reports.py       # 📄 Report professionali
```

### Dependency Injection Pattern

Ogni router utilizza il pattern DI per servizi condivisi:

```python
def get_services():
    storage_path = Path("models")
    model_manager = ModelManager(storage_path)
    forecast_service = ForecastService(model_manager)
    return model_manager, forecast_service

@router.post("/endpoint")
async def handler(services: tuple = Depends(get_services)):
    model_manager, forecast_service = services
    # Business logic
```

### Background Tasks

Training e report generation utilizzano FastAPI BackgroundTasks:

```python
@router.post("/train")
async def train_model(
    request: ModelTrainingRequest,
    background_tasks: BackgroundTasks
):
    model_id = str(uuid.uuid4())
    background_tasks.add_task(_train_model_background, ...)
    return ModelInfo(model_id=model_id, status="training")
```

---

## Installazione e Setup

### Prerequisiti

- Python 3.9+
- UV package manager (raccomandato)
- Quarto CLI (opzionale, per report)

### Setup Rapido

```bash
# Clone repository
git clone https://github.com/username/arima_project
cd arima_project

# Install con UV (10x più veloce di pip)
curl -LsSf https://astral.sh/uv/install.sh | sh  # Linux/macOS
# winget install --id=astral-sh.uv  # Windows

# Setup ambiente e dipendenze
uv sync --all-extras

# Avvio server
uv run python -m arima_forecaster.api.main
```

### Configurazione Personalizzata

```python
from arima_forecaster.api.main import create_app

app = create_app(
    model_storage_path="/custom/models/path",
    enable_scalar=True,
    production_mode=False
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Variabili Ambiente (Opzionali)

```bash
export MODEL_STORAGE_PATH="/path/to/models"
export ENABLE_SCALAR="true"
export PRODUCTION_MODE="false"
export CORS_ORIGINS="http://localhost:3000,https://yourdomain.com"
```

---

## Autenticazione

**Status Attuale**: L'API è completamente aperta (no auth required)

### Per Ambiente Production

Implementare autenticazione con JWT Token:

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer

security = HTTPBearer()

async def get_current_user(token: str = Depends(security)):
    # Validate JWT token
    # Return user info
    pass

@router.post("/protected-endpoint")
async def protected_handler(current_user = Depends(get_current_user)):
    # Authenticated logic
    pass
```

---

## Endpoints di Reference

### 🏥 Health Router

Base path: `/`

#### `GET /` - Root endpoint
**Scopo**: Informazioni di base sull'API

**Response**:
```json
{
    "message": "ARIMA Forecaster API",
    "version": "1.0.0",
    "docs": "/docs"
}
```

#### `GET /health` - Health Check
**Scopo**: Monitoraggio stato servizio per load balancer

**Response**:
```json
{
    "status": "healthy",
    "timestamp": "2024-08-27T14:30:00.000Z",
    "service": "arima-forecaster-api"
}
```

**Utilizzo in Production**:
```bash
# Kubernetes liveness probe
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
```

### 🎨 Training Router  

Base path: `/models`

#### `POST /models/train` - Addestra Modelli ARIMA/SARIMA/SARIMAX
**Scopo**: Training asincrono modelli univariati

**Request Body**:
```json
{
    "model_type": "sarima",
    "data": {
        "timestamps": ["2024-01-01", "2024-01-02", "..."],
        "values": [100.0, 102.5, 98.7, "..."]
    },
    "order": {"p": 1, "d": 1, "q": 1},
    "seasonal_order": {"P": 1, "D": 1, "Q": 1, "s": 12},
    "exogenous_data": null
}
```

**Response**:
```json
{
    "model_id": "abc123e4-5678-9012-3456-789012345678",
    "model_type": "sarima",
    "status": "training",
    "created_at": "2024-08-27T14:30:00.123456",
    "training_observations": 365,
    "parameters": {},
    "metrics": {}
}
```

**Considerazioni Implementative**:
- **Training asincrono**: Il modello viene addestrato in background
- **Status tracking**: Monitorare stato con `GET /models/{id}`
- **Timeout**: Training può richiedere 30 secondi - 5 minuti
- **Memory usage**: Modelli SARIMA richiedono più RAM

#### `POST /models/train/var` - Addestra Modelli VAR
**Scopo**: Training modelli multivariati per serie correlate

**Request Body**:
```json
{
    "data": {
        "series": [
            {
                "name": "sales",
                "timestamps": ["2024-01-01", "..."],
                "values": [100.0, 102.5, "..."]
            },
            {
                "name": "temperature", 
                "timestamps": ["2024-01-01", "..."],
                "values": [15.2, 16.8, "..."]
            }
        ]
    },
    "max_lags": 10
}
```

**Response**:
```json
{
    "model_id": "var-abc123",
    "model_type": "var",
    "status": "training",
    "created_at": "2024-08-27T14:30:00",
    "variables": ["sales", "temperature"],
    "max_lags": 10,
    "selected_lag_order": null,
    "causality_tests": {}
}
```

**VAR-Specific Considerations**:
- **Multiple series**: Tutte le serie devono avere stessi timestamps
- **Stationarity**: VAR richiede serie stazionarie
- **Causality testing**: Granger causality tests automatici
- **Lag selection**: AIC/BIC optimization per optimal lag order

#### `POST /models/auto-select` - Selezione Automatica Parametri
**Scopo**: Grid search per parametri ottimali ARIMA/SARIMA

**Request Body**:
```json
{
    "data": {
        "timestamps": ["2024-01-01", "..."],
        "values": [100.0, 102.5, "..."]
    },
    "max_p": 3,
    "max_d": 2, 
    "max_q": 3,
    "seasonal": true,
    "seasonal_period": 12,
    "criterion": "aic"
}
```

**Response**:
```json
{
    "best_model": {
        "order": [2, 1, 2],
        "seasonal_order": [1, 1, 1, 12],
        "aic": 1234.56,
        "bic": 1256.78
    },
    "all_models": [
        {"order": [1, 1, 1], "aic": 1245.67},
        {"order": [2, 1, 2], "aic": 1234.56}
    ],
    "search_time_seconds": 45.2
}
```

**Auto-ML Best Practices**:
- **Search space**: Limitare max_p, max_q per evitare overfitting
- **Seasonality detection**: Usare seasonal=true solo se confermato
- **Criterion**: AIC per model selection, BIC per parsimony
- **Timeout**: Auto-selection può richiedere 1-10 minuti

### 📈 Forecasting Router

Base path: `/models`

#### `POST /models/{model_id}/forecast` - Genera Previsioni
**Scopo**: Previsioni future con intervalli di confidenza

**Path Parameters**:
- `model_id`: ID univoco del modello addestrato

**Request Body**:
```json
{
    "steps": 30,
    "confidence_level": 0.95,
    "return_confidence_intervals": true,
    "exogenous_future": null
}
```

**Response**:
```json
{
    "forecast": [105.2, 107.8, 103.5, "..."],
    "timestamps": ["2024-09-01", "2024-09-02", "..."],
    "confidence_intervals": {
        "lower": [100.1, 102.3, 98.2, "..."],
        "upper": [110.3, 113.3, 108.8, "..."]
    },
    "model_id": "abc123e4-5678-9012-3456-789012345678",
    "forecast_steps": 30
}
```

**Forecasting Considerations**:
- **Model dependency**: Il modello deve essere status="completed"
- **Exogenous variables**: SARIMAX richiede future exogenous data
- **Confidence intervals**: Basati su distribuzione normale dei residui
- **Timestamp generation**: Assume frequenza giornaliera default

### 📁 Models Router

Base path: `/models`

#### `GET /models` - Lista Tutti i Modelli
**Response**:
```json
{
    "models": [
        {
            "model_id": "abc123",
            "model_type": "sarima",
            "status": "completed",
            "created_at": "2024-08-27T10:00:00",
            "training_observations": 365,
            "parameters": {
                "order": [1, 1, 1],
                "seasonal_order": [1, 1, 1, 12]
            },
            "metrics": {
                "aic": 1234.56,
                "bic": 1256.78,
                "mape": 5.67
            }
        }
    ],
    "total_count": 1
}
```

#### `GET /models/{model_id}` - Dettagli Modello Specifico
**Response**: Stesso formato ModelInfo del listing

#### `DELETE /models/{model_id}` - Elimina Modello
**Response**:
```json
{
    "message": "Model abc123e4-5678-9012-3456-789012345678 deleted successfully"
}
```

**Model Management Patterns**:
- **Cleanup strategy**: Implementare retention policy automatica
- **Backup**: Backup periodico directory models/
- **Versioning**: Utilizzare semantic versioning per modelli

### 🔍 Diagnostics Router

Base path: `/models`

#### `POST /models/{model_id}/diagnostics` - Analisi Diagnostica
**Response**:
```json
{
    "residuals_stats": {
        "mean": 0.002,
        "std": 1.234,
        "skewness": 0.145,
        "kurtosis": 3.021
    },
    "ljung_box_test": {
        "statistic": 15.234,
        "p_value": 0.432,
        "result": "No autocorrelation detected"
    },
    "jarque_bera_test": {
        "statistic": 2.145,
        "p_value": 0.342,
        "result": "Residuals are normally distributed"
    },
    "acf_values": [1.0, 0.05, -0.02, "..."],
    "pacf_values": [1.0, 0.05, -0.01, "..."],
    "performance_metrics": {
        "mae": 12.34,
        "rmse": 15.67,
        "mape": 5.43,
        "r2": 0.92
    }
}
```

**Statistical Tests Interpretation**:
- **Ljung-Box**: p_value > 0.05 = no autocorrelation (good)
- **Jarque-Bera**: p_value > 0.05 = normal distribution (good)
- **ACF/PACF**: Values near 0 = good model fit
- **MAPE < 10%**: Excellent forecast accuracy
- **R² > 0.8**: Strong model explanatory power

### 📄 Reports Router

Base path: `/`

#### `POST /models/{model_id}/report` - Genera Report
**Request Body**:
```json
{
    "format": "html",
    "include_diagnostics": true,
    "include_forecasts": true,
    "forecast_steps": 30,
    "template": "default"
}
```

**Response**:
```json
{
    "report_id": "report-xyz789",
    "status": "generating",
    "format_type": "html",
    "generation_time": 15.7,
    "file_size_mb": 2.34,
    "download_url": "/reports/sarima_vendite_q4_2024.html"
}
```

#### `GET /reports/{filename}` - Download Report
**Response**: File binario (HTML/PDF/DOCX)

**Content-Type Headers**:
- `.html`: `text/html`
- `.pdf`: `application/pdf`
- `.docx`: `application/vnd.openxmlformats-officedocument.wordprocessingml.document`

---

## Modelli Dati

### Core Models (`models.py`)

#### TimeSeriesData
```python
class TimeSeriesData(BaseModel):
    timestamps: List[str]  # ISO 8601 format
    values: List[float]    # Numeric values
```

#### ModelOrder / SeasonalOrder
```python
class ModelOrder(BaseModel):
    p: int = Field(ge=0, le=5)  # AR order
    d: int = Field(ge=0, le=2)  # Differencing
    q: int = Field(ge=0, le=5)  # MA order

class SeasonalOrder(BaseModel):
    P: int = Field(ge=0, le=2)  # Seasonal AR
    D: int = Field(ge=0, le=1)  # Seasonal differencing
    Q: int = Field(ge=0, le=2)  # Seasonal MA
    s: int = Field(ge=2)        # Seasonal period
```

#### ModelTrainingRequest
```python
class ModelTrainingRequest(BaseModel):
    model_type: Literal["arima", "sarima", "sarimax"]
    data: TimeSeriesData
    order: ModelOrder
    seasonal_order: Optional[SeasonalOrder] = None
    exogenous_data: Optional[ExogenousData] = None
```

### Extended Models (`models_extra.py`)

#### VARTrainingRequest
```python
class VARTrainingRequest(BaseModel):
    data: MultivariateSeries
    max_lags: int = Field(default=10, ge=1, le=50)
```

#### ForecastResponse
```python
class ForecastResponse(BaseModel):
    forecast: List[float]
    timestamps: List[str] 
    confidence_intervals: Optional[Dict[str, List[float]]]
    model_id: str
    forecast_steps: int
```

#### ModelDiagnostics
```python
class ModelDiagnostics(BaseModel):
    residuals_stats: Dict[str, float]
    ljung_box_test: Dict[str, Any]
    jarque_bera_test: Dict[str, Any]
    acf_values: List[float]
    pacf_values: List[float]
    performance_metrics: Dict[str, Optional[float]]
```

### Validation Rules

**Constraints Principali**:
- `timestamps`: Formato ISO 8601 obbligatorio
- `values`: Non possono essere NaN/Infinite
- `p,d,q`: Range 0-5 per prevenire overfitting
- `seasonal_period`: Minimum 2 (per seasonal models)
- `confidence_level`: Range 0.5-0.99
- `forecast_steps`: Maximum 365 giorni

---

## Esempi Pratici

### Workflow Completo End-to-End

```python
import requests
import pandas as pd
from datetime import datetime, timedelta

API_BASE = "http://localhost:8000"

# 1. Preparazione dati
dates = pd.date_range("2023-01-01", "2024-08-01", freq="D")
values = (100 + pd.Series(range(len(dates))) * 0.1 + 
          pd.Series(np.random.normal(0, 2, len(dates)))).tolist()

training_data = {
    "model_type": "sarima",
    "data": {
        "timestamps": [d.strftime("%Y-%m-%d") for d in dates],
        "values": values
    },
    "order": {"p": 1, "d": 1, "q": 1},
    "seasonal_order": {"P": 1, "D": 1, "Q": 1, "s": 7}
}

# 2. Avvio training
response = requests.post(f"{API_BASE}/models/train", json=training_data)
model_info = response.json()
model_id = model_info["model_id"]

# 3. Polling stato training
import time
while True:
    status_response = requests.get(f"{API_BASE}/models/{model_id}")
    status = status_response.json()["status"]
    
    if status == "completed":
        break
    elif status == "failed":
        raise Exception("Training failed")
    
    time.sleep(10)  # Wait 10 seconds

# 4. Generazione previsioni
forecast_request = {
    "steps": 30,
    "confidence_level": 0.95,
    "return_confidence_intervals": True
}

forecast_response = requests.post(
    f"{API_BASE}/models/{model_id}/forecast",
    json=forecast_request
)
forecast_data = forecast_response.json()

# 5. Analisi diagnostica
diagnostics_response = requests.post(f"{API_BASE}/models/{model_id}/diagnostics")
diagnostics = diagnostics_response.json()

print(f"MAPE: {diagnostics['performance_metrics']['mape']:.2f}%")
print(f"R²: {diagnostics['performance_metrics']['r2']:.3f}")

# 6. Generazione report
report_request = {
    "format": "html",
    "include_diagnostics": True,
    "include_forecasts": True,
    "forecast_steps": 30
}

report_response = requests.post(
    f"{API_BASE}/models/{model_id}/report",
    json=report_request
)
report_info = report_response.json()

# 7. Download report (dopo generazione)
import time
time.sleep(20)  # Wait for background generation

download_url = report_info["download_url"]
report_file = requests.get(f"{API_BASE}{download_url}")

with open("forecast_report.html", "wb") as f:
    f.write(report_file.content)
```

### Auto-Selection per Optimized Models

```python
# Grid search per modello ottimale
auto_select_request = {
    "data": {
        "timestamps": timestamps,
        "values": values
    },
    "max_p": 3,
    "max_d": 2,
    "max_q": 3,
    "seasonal": True,
    "seasonal_period": 7,  # Weekly seasonality
    "criterion": "aic"
}

auto_response = requests.post(
    f"{API_BASE}/models/auto-select",
    json=auto_select_request
)

best_params = auto_response.json()["best_model"]

# Usa parametri ottimali per training
optimized_training = {
    "model_type": "sarima",
    "data": {"timestamps": timestamps, "values": values},
    "order": {
        "p": best_params["order"][0],
        "d": best_params["order"][1], 
        "q": best_params["order"][2]
    },
    "seasonal_order": {
        "P": best_params["seasonal_order"][0],
        "D": best_params["seasonal_order"][1],
        "Q": best_params["seasonal_order"][2],
        "s": best_params["seasonal_order"][3]
    }
}
```

### Batch Processing Multiple Series

```python
def train_multiple_models(series_dict):
    """Train models per multiple time series"""
    model_ids = []
    
    for series_name, (timestamps, values) in series_dict.items():
        training_request = {
            "model_type": "arima",
            "data": {
                "timestamps": timestamps,
                "values": values
            },
            "order": {"p": 1, "d": 1, "q": 1}
        }
        
        response = requests.post(f"{API_BASE}/models/train", json=training_request)
        model_id = response.json()["model_id"]
        model_ids.append((series_name, model_id))
    
    return model_ids

# Usage
series_data = {
    "sales": (timestamps_sales, values_sales),
    "inventory": (timestamps_inventory, values_inventory),
    "demand": (timestamps_demand, values_demand)
}

trained_models = train_multiple_models(series_data)
```

### VAR Models per Multivariate Analysis

```python
# Prepare multivariate data
multivariate_request = {
    "data": {
        "series": [
            {
                "name": "sales",
                "timestamps": timestamps,
                "values": sales_values
            },
            {
                "name": "marketing_spend",
                "timestamps": timestamps,
                "values": marketing_values
            },
            {
                "name": "temperature", 
                "timestamps": timestamps,
                "values": temp_values
            }
        ]
    },
    "max_lags": 10
}

var_response = requests.post(
    f"{API_BASE}/models/train/var",
    json=multivariate_request
)

var_model_id = var_response.json()["model_id"]

# VAR predictions include all variables
var_forecast = requests.post(
    f"{API_BASE}/models/{var_model_id}/forecast",
    json={"steps": 14}
)
```

---

## Gestione Errori

### HTTP Status Codes

| Code | Meaning | When |
|------|---------|------|
| 200 | OK | Request successful |  
| 400 | Bad Request | Invalid input data, wrong parameters |
| 404 | Not Found | Model not found, report not available |
| 422 | Unprocessable Entity | Pydantic validation failed |
| 500 | Internal Server Error | Model training/prediction failed |

### Error Response Format

```json
{
    "detail": "Model abc123 not found",
    "error_type": "ModelNotFoundError",
    "timestamp": "2024-08-27T14:30:00Z"
}
```

### Error Handling Patterns

```python
def robust_api_call(url, json_data, max_retries=3):
    """Robust API call con retry logic"""
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=json_data, timeout=30)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            raise
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 500 and attempt < max_retries - 1:
                time.sleep(5)  # Server error retry
                continue
            
            error_detail = e.response.json().get("detail", "Unknown error")
            raise Exception(f"API Error {e.response.status_code}: {error_detail}")

# Usage
try:
    result = robust_api_call(f"{API_BASE}/models/train", training_data)
except Exception as e:
    logger.error(f"Training failed after retries: {e}")
```

### Common Error Scenarios

**Training Failures**:
```python
# Insufficient data
if len(values) < 50:
    raise ValueError("Need at least 50 observations for SARIMA")

# Non-stationary series  
from statsmodels.tsa.stattools import adfuller
def check_stationarity(series):
    result = adfuller(series)
    return result[1] <= 0.05  # p-value check

if not check_stationarity(values):
    # Apply differencing or suggest seasonal_order
    pass
```

**Forecast Failures**:
```python
# Model still training
def wait_for_completion(model_id, timeout=300):
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        status = requests.get(f"{API_BASE}/models/{model_id}").json()
        
        if status["status"] == "completed":
            return True
        elif status["status"] == "failed":
            raise Exception("Model training failed")
        
        time.sleep(10)
    
    raise TimeoutError("Model training timeout")
```

---

## Performance e Scalabilità

### Benchmarks di Performance

**Training Times** (serie 365 giorni):
- ARIMA(1,1,1): ~5-15 seconds
- SARIMA(1,1,1)(1,1,1,12): ~15-45 seconds  
- VAR(3 variables, 10 lags): ~30-90 seconds
- Auto-selection: ~2-10 minutes

**Memory Usage**:
- ARIMA model: ~5-20 MB RAM
- SARIMA model: ~20-100 MB RAM
- VAR model: ~50-200 MB RAM per variable

### Ottimizzazioni

**Database Connection Pooling**:
```python
# Use connection pooling for production
import asyncpg

class DatabasePool:
    def __init__(self):
        self.pool = None
    
    async def create_pool(self):
        self.pool = await asyncpg.create_pool(
            host="localhost",
            database="arima_models",
            min_size=10,
            max_size=20
        )

# Usage in dependency
async def get_db_pool():
    return database_pool.pool
```

**Caching Strategy**:
```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=100)
def get_model_cache(model_id: str):
    """Cache frequently used models"""
    return model_manager.load_model(model_id)

def cache_key(data: dict) -> str:
    """Generate cache key from data"""
    return hashlib.md5(str(sorted(data.items())).encode()).hexdigest()
```

**Background Task Queue**:
```python
# Production: Use Celery instead of BackgroundTasks
from celery import Celery

celery_app = Celery('arima_forecaster')

@celery_app.task
def train_model_task(model_id: str, series_data: dict, params: dict):
    """Celery task for model training"""
    # Training logic
    pass

# In router
@router.post("/models/train")
async def train_model(request: ModelTrainingRequest):
    model_id = str(uuid.uuid4())
    
    # Queue training task
    train_model_task.delay(model_id, request.data.dict(), request.dict())
    
    return ModelInfo(model_id=model_id, status="queued")
```

### Scalabilità Horizontale

**Load Balancer Config** (nginx):
```nginx
upstream arima_api {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;  
    server 127.0.0.1:8002;
}

server {
    listen 80;
    location / {
        proxy_pass http://arima_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    # Health check endpoint
    location /health {
        access_log off;
        proxy_pass http://arima_api/health;
    }
}
```

**Docker Compose per Multi-Instance**:
```yaml
version: '3.8'

services:
  arima-api-1:
    build: .
    ports:
      - "8000:8000"
    environment:
      - WORKER_ID=1
    volumes:
      - ./models:/app/models
      
  arima-api-2:
    build: .
    ports:
      - "8001:8000"
    environment:
      - WORKER_ID=2
    volumes:
      - ./models:/app/models

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
      
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    depends_on:
      - arima-api-1
      - arima-api-2
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
```

---

## Deployment

### Production Checklist

- [ ] **Environment Variables**: Set production configs
- [ ] **Database**: Configure persistent storage  
- [ ] **Monitoring**: Setup health checks e metrics
- [ ] **Security**: Enable HTTPS, authentication
- [ ] **Logging**: Configure structured logging
- [ ] **Backup**: Automated model backup strategy
- [ ] **Scaling**: Load balancer configuration

### Docker Deployment

**Dockerfile**:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-cache

# Copy application
COPY src/ src/

# Create directories
RUN mkdir -p models reports logs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run application
EXPOSE 8000
CMD ["uv", "run", "python", "-m", "arima_forecaster.api.main"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: arima-forecaster-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: arima-api
  template:
    metadata:
      labels:
        app: arima-api
    spec:
      containers:
      - name: arima-api
        image: arima-forecaster:latest
        ports:
        - containerPort: 8000
        env:
        - name: PRODUCTION_MODE
          value: "true"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi" 
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        volumeMounts:
        - name: models-volume
          mountPath: /app/models
      volumes:
      - name: models-volume
        persistentVolumeClaim:
          claimName: arima-models-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: arima-api-service
spec:
  selector:
    app: arima-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Environment Variables

```bash
# Production settings
PRODUCTION_MODE=true
MODEL_STORAGE_PATH=/data/models
REPORT_STORAGE_PATH=/data/reports

# Database (future)
DATABASE_URL=postgresql://user:pass@localhost/arima_db
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=60

# CORS
CORS_ORIGINS=https://yourapp.com,https://dashboard.yourapp.com

# Monitoring
LOG_LEVEL=INFO
SENTRY_DSN=https://your-sentry-dsn
```

---

## Troubleshooting

### Common Issues

#### 1. Training Failures

**Problema**: Model training fails con "convergence not achieved"
```python
# Solution: Adjust parameters
training_request = {
    "model_type": "sarima",
    "data": data,
    "order": {"p": 1, "d": 1, "q": 1},  # Start simple
    "seasonal_order": {"P": 0, "D": 1, "Q": 1, "s": 12}  # Reduce complexity
}
```

**Problema**: "insufficient observations" error
```python
# Solution: Check data length
if len(values) < (p + d + q + P + D + Q) * s:
    raise ValueError("Need more observations for model complexity")
```

#### 2. Memory Issues

**Problema**: "Out of memory" during VAR training
```python
# Solution: Reduce lag order or variables
var_request = {
    "data": {"series": series[:2]},  # Reduce variables
    "max_lags": 5  # Reduce from 10
}
```

#### 3. Forecast Errors

**Problema**: NaN predictions returned
```python
# Check model residuals
diagnostics = requests.post(f"/models/{model_id}/diagnostics").json()
if diagnostics["ljung_box_test"]["p_value"] < 0.05:
    print("Model has autocorrelation issues")
    # Retrain with different parameters
```

#### 4. Performance Issues

**Problema**: Slow API responses
```python
# Monitor endpoint performance
import time

def time_api_call(endpoint, data):
    start = time.time()
    response = requests.post(endpoint, json=data)
    duration = time.time() - start
    
    print(f"API call took {duration:.2f} seconds")
    return response

# Solutions:
# - Use connection pooling
# - Enable response caching
# - Scale horizontally
```

### Debugging Tools

**API Health Monitoring**:
```python
def monitor_api_health():
    """Monitor API health and performance"""
    try:
        start_time = time.time()
        response = requests.get(f"{API_BASE}/health", timeout=5)
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            print(f"✅ API healthy (response time: {response_time:.3f}s)")
        else:
            print(f"❌ API unhealthy: {response.status_code}")
            
    except requests.exceptions.Timeout:
        print("❌ API timeout")
    except Exception as e:
        print(f"❌ API error: {e}")

# Run periodically
import schedule
schedule.every(1).minutes.do(monitor_api_health)
```

**Model Performance Validation**:
```python
def validate_model_performance(model_id, threshold_mape=15.0):
    """Validate model meets performance thresholds"""
    diagnostics = requests.post(f"{API_BASE}/models/{model_id}/diagnostics").json()
    
    mape = diagnostics["performance_metrics"]["mape"]
    r2 = diagnostics["performance_metrics"]["r2"]
    
    if mape > threshold_mape:
        print(f"⚠️ High MAPE: {mape:.2f}% (threshold: {threshold_mape}%)")
        return False
        
    if r2 < 0.7:
        print(f"⚠️ Low R²: {r2:.3f} (threshold: 0.7)")
        return False
        
    print(f"✅ Model performance acceptable (MAPE: {mape:.2f}%, R²: {r2:.3f})")
    return True
```

---

## Appendici

### A. API Reference Summary

| Endpoint | Method | Purpose | Auth Required |
|----------|--------|---------|---------------|
| `/` | GET | API info | No |
| `/health` | GET | Health check | No |
| `/models/train` | POST | Train ARIMA/SARIMA | No |
| `/models/train/var` | POST | Train VAR | No |
| `/models/auto-select` | POST | Parameter optimization | No |
| `/models` | GET | List models | No |
| `/models/{id}` | GET | Model details | No |
| `/models/{id}` | DELETE | Delete model | No |
| `/models/{id}/forecast` | POST | Generate predictions | No |
| `/models/{id}/diagnostics` | POST | Model diagnostics | No |
| `/models/{id}/report` | POST | Generate report | No |
| `/reports/{filename}` | GET | Download report | No |

### B. Performance Benchmarks

**Latency (p95)**:
- Health check: <50ms
- Model listing: <200ms  
- Forecast generation: <2s
- Diagnostics: <5s
- Report generation: 15-30s

**Throughput**:
- Concurrent model training: 5-10 models
- Forecast requests: 100+ req/min
- Report generation: 5-10 reports/hour

### C. Configuration Examples

**Development**:
```python
app = create_app(
    model_storage_path="./dev_models",
    enable_scalar=True,
    production_mode=False
)
```

**Testing**:
```python  
app = create_app(
    model_storage_path="/tmp/test_models",
    enable_scalar=False,
    production_mode=False
)
```

**Production**:
```python
app = create_app(
    model_storage_path="/data/models", 
    enable_scalar=True,
    production_mode=True
)
```

---

## Note Finali

Questa guida copre tutti gli aspetti dell'ARIMA Forecaster API per sviluppatori. Per aggiornamenti e nuove funzionalità, consultare la documentazione Swagger su `/docs` e Scalar su `/scalar`.

**Supporto**:
- Documentazione: `http://localhost:8000/docs`
- GitHub Issues: Repository del progetto
- API Status: `http://localhost:8000/health`

**Versione Documento**: 1.0.0  
**Data**: 27 Agosto 2024  
**Compatibilità API**: v1.1.0+