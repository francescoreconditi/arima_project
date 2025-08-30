# 📚 Guida Completa API ARIMA Forecaster

## 🎯 Panoramica

**ARIMA Forecaster API** è un servizio REST enterprise-grade per analisi e previsione di serie temporali con modelli statistici avanzati.

### Caratteristiche Principali:
- 📊 **Modelli Multipli**: ARIMA, SARIMA, SARIMAX, VAR, Facebook Prophet
- 🤖 **Auto-ML**: Selezione automatica parametri ottimali
- 📈 **Forecasting Avanzato**: Previsioni con intervalli di confidenza
- 🔍 **Diagnostica Completa**: Test statistici e analisi residui
- 📄 **Report Professionali**: Generazione HTML/PDF/DOCX con Quarto
- 🌍 **API RESTful**: JSON in/out con validazione Pydantic
- ⚡ **Processing Asincrono**: Training non-bloccante in background

## 🚀 Quick Start

### Installazione

```bash
# Installa con pip
pip install arima-forecaster[api]

# O con uv (più veloce)
uv sync --extra api
```

### Avvio Server

```python
# Script Python
from arima_forecaster.api.main import create_app
import uvicorn

app = create_app(
    model_storage_path="/path/to/models",
    enable_scalar=True,
    production_mode=False
)

uvicorn.run(app, host="0.0.0.0", port=8000)
```

```bash
# Da terminale
uv run python -m arima_forecaster.api.main

# O con script dedicato
uv run python scripts/run_api.py
```

### URL Documentazione

- **Swagger UI**: http://localhost:8000/docs
- **Scalar UI**: http://localhost:8000/scalar
- **ReDoc**: http://localhost:8000/redoc

## 📋 Riferimento Completo Endpoint

### 1️⃣ Health & Status

#### **GET /** - Root Information
Restituisce informazioni base dell'API.

**Response:**
```json
{
  "message": "ARIMA Forecaster API",
  "version": "1.1.0",
  "docs": "/docs"
}
```

**Esempio cURL:**
```bash
curl http://localhost:8000/
```

---

#### **GET /health** - Health Check
Verifica lo stato del servizio per monitoring e load balancer.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-08-30T10:00:00",
  "service": "arima-forecaster-api"
}
```

**Esempio cURL:**
```bash
curl http://localhost:8000/health
```

---

### 2️⃣ Training Modelli

#### **POST /models/train** - Training ARIMA/SARIMA/SARIMAX
Addestra modelli ARIMA con parametri specificati.

**Request Body:**
```json
{
  "series": [1.2, 2.3, 3.1, 4.5, 5.2, 6.1, 7.3, 8.0, 9.1, 10.2],
  "model_type": "sarima",
  "order": [1, 1, 1],
  "seasonal_order": [1, 1, 1, 12],
  "exog": null,
  "model_id": "sales_sarima_2024",
  "auto_select": false
}
```

**Response:**
```json
{
  "model_id": "sales_sarima_2024",
  "model_type": "sarima",
  "status": "training",
  "created_at": "2024-08-30T10:00:00",
  "parameters": {
    "order": [1, 1, 1],
    "seasonal_order": [1, 1, 1, 12]
  },
  "metrics": {
    "aic": 234.56,
    "bic": 245.67,
    "log_likelihood": -114.28
  }
}
```

**Esempio Python:**
```python
import requests

data = {
    "series": [100, 110, 120, 115, 125, 130, 140, 135, 145, 150],
    "model_type": "arima",
    "order": [2, 1, 2],
    "model_id": "revenue_model"
}

response = requests.post("http://localhost:8000/models/train", json=data)
model_info = response.json()
print(f"Model ID: {model_info['model_id']}")
print(f"Status: {model_info['status']}")
```

---

#### **POST /models/train/var** - Training VAR Multivariato
Addestra modelli VAR per serie temporali multivariate.

**Request Body:**
```json
{
  "series": {
    "sales": [100, 110, 120, 130, 140],
    "marketing": [50, 55, 60, 65, 70],
    "temperature": [20, 22, 25, 23, 21]
  },
  "max_lags": 5,
  "ic": "aic",
  "model_id": "multivariate_var_model"
}
```

**Response:**
```json
{
  "model_id": "multivariate_var_model",
  "model_type": "var",
  "status": "completed",
  "n_series": 3,
  "selected_lag": 2,
  "parameters": {
    "lags": 2,
    "ic": "aic"
  },
  "metrics": {
    "aic": -1234.56,
    "bic": -1200.34,
    "fpe": 0.0012,
    "hqic": -1220.45
  }
}
```

**Esempio Python:**
```python
import pandas as pd
import requests

# Dati multivariati
data = {
    "series": {
        "vendite": list(range(100, 200, 5)),
        "pubblicita": list(range(50, 100, 2.5)),
        "concorrenza": list(range(80, 130, 2.5))
    },
    "max_lags": 10,
    "ic": "bic",
    "model_id": "business_var"
}

response = requests.post("http://localhost:8000/models/train/var", json=data)
print(response.json())
```

---

#### **POST /models/auto-select** - Selezione Automatica Parametri
Trova automaticamente i parametri ottimali per ARIMA/SARIMA.

**Request Body:**
```json
{
  "series": [100, 102, 105, 103, 108, 110, 112, 115, 113, 118],
  "model_type": "sarima",
  "p_range": [0, 3],
  "d_range": [0, 2],
  "q_range": [0, 3],
  "P_range": [0, 2],
  "D_range": [0, 1],
  "Q_range": [0, 2],
  "s": 12,
  "ic": "aic",
  "n_jobs": -1,
  "model_id": "auto_sarima_optimal"
}
```

**Response:**
```json
{
  "best_model_id": "auto_sarima_optimal",
  "best_params": {
    "order": [1, 1, 2],
    "seasonal_order": [1, 0, 1, 12]
  },
  "best_score": 456.78,
  "models_evaluated": 144,
  "search_time": 12.34,
  "all_results": [
    {
      "params": {"order": [1, 1, 2], "seasonal_order": [1, 0, 1, 12]},
      "score": 456.78,
      "converged": true
    }
  ]
}
```

**Esempio Python:**
```python
# Auto-selezione con grid search
auto_request = {
    "series": sales_data,
    "model_type": "arima",
    "p_range": [0, 5],
    "d_range": [0, 2],
    "q_range": [0, 5],
    "ic": "bic",
    "n_jobs": 4,
    "model_id": "best_arima"
}

response = requests.post("http://localhost:8000/models/auto-select", json=auto_request)
best = response.json()
print(f"Parametri ottimali: {best['best_params']}")
print(f"Score: {best['best_score']:.2f}")
print(f"Modelli valutati: {best['models_evaluated']}")
```

---

#### **POST /models/train/prophet** - Training Facebook Prophet
Addestra modelli Prophet con decomposizione trend/stagionalità.

**Request Body:**
```json
{
  "series": [100, 110, 105, 120, 115, 125, 130, 140, 135, 145],
  "dates": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
            "2024-01-06", "2024-01-07", "2024-01-08", "2024-01-09", "2024-01-10"],
  "model_id": "prophet_sales",
  "changepoint_prior_scale": 0.05,
  "seasonality_prior_scale": 10.0,
  "holidays_prior_scale": 10.0,
  "seasonality_mode": "additive",
  "country": "IT",
  "include_holidays": true,
  "yearly_seasonality": "auto",
  "weekly_seasonality": "auto",
  "daily_seasonality": false
}
```

**Response:**
```json
{
  "model_id": "prophet_sales",
  "status": "completed",
  "metrics": {
    "mape": 5.23,
    "mae": 3.45,
    "rmse": 4.12,
    "r2": 0.92
  },
  "components": {
    "trend": "increasing",
    "weekly_seasonality": true,
    "yearly_seasonality": false,
    "holidays_included": 25
  },
  "training_time": 1.23
}
```

**Esempio Python:**
```python
# Prophet con festività italiane
prophet_data = {
    "series": monthly_sales,
    "dates": pd.date_range("2023-01-01", periods=len(monthly_sales), freq="M").tolist(),
    "model_id": "prophet_italy",
    "country": "IT",
    "include_holidays": True,
    "changepoint_prior_scale": 0.1,
    "seasonality_mode": "multiplicative"
}

response = requests.post("http://localhost:8000/models/train/prophet", json=prophet_data)
print(response.json())
```

---

#### **POST /models/train/prophet/auto-select** - Auto-Tuning Prophet
Ottimizzazione automatica parametri Prophet.

**Request Body:**
```json
{
  "series": [100, 110, 120, 130, 125, 135, 140, 150, 145, 155],
  "dates": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01",
            "2024-06-01", "2024-07-01", "2024-08-01", "2024-09-01", "2024-10-01"],
  "model_id": "prophet_auto",
  "optimization_method": "bayesian",
  "n_trials": 50,
  "cv_folds": 3,
  "metric": "mape",
  "param_ranges": {
    "changepoint_prior_scale": [0.001, 0.5],
    "seasonality_prior_scale": [0.01, 10],
    "holidays_prior_scale": [0.01, 10]
  }
}
```

**Response:**
```json
{
  "best_model_id": "prophet_auto",
  "best_params": {
    "changepoint_prior_scale": 0.08,
    "seasonality_prior_scale": 5.2,
    "holidays_prior_scale": 8.3,
    "seasonality_mode": "additive"
  },
  "best_score": 3.45,
  "optimization_history": [
    {"trial": 1, "params": {...}, "score": 5.67},
    {"trial": 2, "params": {...}, "score": 4.23}
  ],
  "total_time": 45.67
}
```

---

### 3️⃣ Forecasting

#### **POST /models/{model_id}/forecast** - Generazione Previsioni
Genera previsioni con modello addestrato.

**Request Body:**
```json
{
  "steps": 30,
  "confidence_level": 0.95,
  "exog": null,
  "return_confidence_intervals": true,
  "return_components": false
}
```

**Response:**
```json
{
  "model_id": "sales_sarima_2024",
  "forecasts": [145.2, 148.5, 152.1, 149.8, 155.3],
  "timestamps": ["2024-09-01", "2024-09-02", "2024-09-03", "2024-09-04", "2024-09-05"],
  "confidence_intervals": {
    "lower": [140.1, 142.3, 145.2, 142.5, 147.1],
    "upper": [150.3, 154.7, 159.0, 157.1, 163.5]
  },
  "metadata": {
    "model_type": "sarima",
    "confidence_level": 0.95,
    "forecast_horizon": 30
  }
}
```

**Esempio Python:**
```python
# Forecast a 30 giorni con intervalli 95%
forecast_request = {
    "steps": 30,
    "confidence_level": 0.95,
    "return_confidence_intervals": True
}

response = requests.post(
    f"http://localhost:8000/models/{model_id}/forecast",
    json=forecast_request
)

forecast = response.json()
predictions = forecast["forecasts"]
lower_bound = forecast["confidence_intervals"]["lower"]
upper_bound = forecast["confidence_intervals"]["upper"]

# Visualizzazione
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(predictions, label="Forecast", color="blue")
plt.fill_between(range(len(predictions)), lower_bound, upper_bound, 
                 alpha=0.3, color="blue", label="95% CI")
plt.xlabel("Giorni")
plt.ylabel("Valore")
plt.title("Previsioni a 30 giorni")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

#### **POST /models/{model_id}/forecast/prophet** - Forecast Prophet con Decomposizione
Previsioni Prophet con componenti trend/stagionalità.

**Request Body:**
```json
{
  "steps": 90,
  "confidence_level": 0.95,
  "return_components": true,
  "include_history": false
}
```

**Response:**
```json
{
  "model_id": "prophet_sales",
  "forecast": {
    "ds": ["2024-11-01", "2024-11-02", "2024-11-03"],
    "yhat": [156.3, 158.2, 160.1],
    "yhat_lower": [150.1, 151.8, 153.5],
    "yhat_upper": [162.5, 164.6, 166.7],
    "trend": [145.0, 145.2, 145.4],
    "weekly": [5.2, 6.8, 8.3],
    "yearly": [6.1, 6.2, 6.4],
    "holidays": [0, 0, 0]
  },
  "components": {
    "trend": {"values": [145.0, 145.2, 145.4], "direction": "increasing"},
    "weekly_seasonality": {"monday": 5.2, "tuesday": 6.8, "wednesday": 8.3},
    "yearly_seasonality": {"january": 6.1, "february": 5.8}
  },
  "metadata": {
    "model_type": "prophet",
    "confidence_level": 0.95,
    "components_included": ["trend", "weekly", "yearly", "holidays"]
  }
}
```

---

### 4️⃣ Gestione Modelli

#### **GET /models** - Lista Modelli
Elenca tutti i modelli addestrati disponibili.

**Response:**
```json
{
  "models": [
    {
      "model_id": "sales_arima_2024",
      "model_type": "arima",
      "status": "completed",
      "created_at": "2024-08-30T10:00:00",
      "metrics": {"aic": 234.56, "bic": 245.67}
    },
    {
      "model_id": "prophet_forecast",
      "model_type": "prophet",
      "status": "completed",
      "created_at": "2024-08-30T11:00:00",
      "metrics": {"mape": 3.45, "rmse": 12.34}
    }
  ],
  "total_count": 2,
  "counts_by_type": {
    "arima": 1,
    "prophet": 1
  }
}
```

**Esempio cURL:**
```bash
curl http://localhost:8000/models
```

---

#### **GET /models/{model_id}** - Dettagli Modello
Recupera informazioni dettagliate su un modello specifico.

**Response:**
```json
{
  "model_id": "sales_sarima_2024",
  "model_type": "sarima",
  "status": "completed",
  "created_at": "2024-08-30T10:00:00",
  "updated_at": "2024-08-30T10:05:00",
  "parameters": {
    "order": [1, 1, 1],
    "seasonal_order": [1, 1, 1, 12],
    "trend": "c"
  },
  "metrics": {
    "aic": 234.56,
    "bic": 245.67,
    "log_likelihood": -114.28,
    "sigma2": 1.23
  },
  "training_info": {
    "n_observations": 120,
    "training_time": 2.34,
    "convergence": true,
    "iterations": 45
  }
}
```

---

#### **DELETE /models/{model_id}** - Elimina Modello
Elimina permanentemente un modello salvato.

**Response:**
```json
{
  "message": "Model sales_arima_2024 deleted successfully"
}
```

**Esempio Python:**
```python
response = requests.delete(f"http://localhost:8000/models/{model_id}")
print(response.json()["message"])
```

---

#### **POST /models/compare** - Confronta Modelli
Confronta le performance di più modelli.

**Request Body:**
```json
{
  "model_ids": ["arima_model_1", "sarima_model_2", "prophet_model_3"]
}
```

**Response:**
```json
{
  "comparison": {
    "arima_model_1": {
      "model_type": "arima",
      "metrics": {"aic": 234.56, "bic": 245.67, "mape": 5.23},
      "rank": 2
    },
    "sarima_model_2": {
      "model_type": "sarima",
      "metrics": {"aic": 220.12, "bic": 235.45, "mape": 4.12},
      "rank": 1
    },
    "prophet_model_3": {
      "model_type": "prophet",
      "metrics": {"mape": 6.78, "rmse": 15.23, "mae": 12.34},
      "rank": 3
    }
  },
  "best_model": "sarima_model_2",
  "ranking_metric": "mape"
}
```

---

### 5️⃣ Diagnostica

#### **POST /models/{model_id}/diagnostics** - Analisi Diagnostica
Esegue analisi diagnostica completa del modello.

**Response:**
```json
{
  "model_id": "sales_sarima_2024",
  "residuals_analysis": {
    "mean": 0.002,
    "std": 1.23,
    "skewness": 0.15,
    "kurtosis": 3.02,
    "min": -3.45,
    "max": 3.67,
    "q1": -0.82,
    "median": 0.01,
    "q3": 0.85
  },
  "statistical_tests": {
    "ljung_box": {
      "statistic": 15.23,
      "p_value": 0.234,
      "conclusion": "No autocorrelation detected"
    },
    "jarque_bera": {
      "statistic": 2.34,
      "p_value": 0.310,
      "conclusion": "Residuals are normally distributed"
    },
    "durbin_watson": {
      "statistic": 1.98,
      "conclusion": "No autocorrelation"
    }
  },
  "acf_pacf": {
    "acf": [1.0, 0.05, -0.02, 0.01],
    "pacf": [1.0, 0.05, -0.03, 0.02],
    "significant_lags": []
  },
  "performance_metrics": {
    "mae": 3.45,
    "rmse": 4.23,
    "mape": 5.67,
    "r2": 0.89,
    "mase": 0.78
  }
}
```

**Esempio Python:**
```python
# Diagnostica completa
response = requests.post(f"http://localhost:8000/models/{model_id}/diagnostics")
diagnostics = response.json()

# Verifica test statistici
ljung_box = diagnostics["statistical_tests"]["ljung_box"]
if ljung_box["p_value"] > 0.05:
    print("✅ Residui non autocorrelati")
else:
    print("⚠️ Possibile autocorrelazione nei residui")

# Verifica normalità
jb_test = diagnostics["statistical_tests"]["jarque_bera"]
if jb_test["p_value"] > 0.05:
    print("✅ Residui normalmente distribuiti")
else:
    print("⚠️ Residui non normali")
```

---

### 6️⃣ Reporting

#### **POST /models/{model_id}/report** - Genera Report
Genera report professionale in HTML/PDF/DOCX.

**Request Body:**
```json
{
  "format": "html",
  "include_diagnostics": true,
  "include_forecast": true,
  "forecast_steps": 30,
  "custom_title": "Report Vendite Q4 2024",
  "author": "Data Science Team",
  "executive_summary": true,
  "technical_appendix": true
}
```

**Response:**
```json
{
  "report_id": "report_12345",
  "status": "generating",
  "model_id": "sales_sarima_2024",
  "format": "html",
  "filename": "report_sales_sarima_2024_20240830.html",
  "download_url": "/reports/report_sales_sarima_2024_20240830.html",
  "estimated_time": 15
}
```

**Esempio Python:**
```python
# Genera report PDF completo
report_request = {
    "format": "pdf",
    "include_diagnostics": True,
    "include_forecast": True,
    "forecast_steps": 90,
    "custom_title": "Analisi Previsionale Vendite 2024",
    "author": "Analytics Team",
    "executive_summary": True
}

response = requests.post(
    f"http://localhost:8000/models/{model_id}/report",
    json=report_request
)

report_info = response.json()
print(f"Report generato: {report_info['filename']}")
print(f"Download: {report_info['download_url']}")

# Download del report
import time
time.sleep(20)  # Attendi generazione

report_response = requests.get(
    f"http://localhost:8000{report_info['download_url']}"
)

with open("report.pdf", "wb") as f:
    f.write(report_response.content)
```

---

#### **GET /reports/{filename}** - Download Report
Scarica un report generato.

**Response:** File binario (HTML/PDF/DOCX)

**Esempio cURL:**
```bash
curl -O http://localhost:8000/reports/report_sales_2024.pdf
```

---

## 🔧 Configurazione Avanzata

### Variabili d'Ambiente

```bash
# .env file
API_HOST=0.0.0.0
API_PORT=8000
MODEL_STORAGE_PATH=/data/models
ENABLE_SCALAR=true
PRODUCTION_MODE=false
LOG_LEVEL=INFO
CORS_ORIGINS=["http://localhost:3000", "https://app.example.com"]
MAX_WORKERS=4
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Installa dipendenze
COPY pyproject.toml .
RUN pip install arima-forecaster[api,gpu]

# Copia codice
COPY . .

# Esponi porta
EXPOSE 8000

# Avvia server
CMD ["uvicorn", "arima_forecaster.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    environment:
      - PRODUCTION_MODE=true
      - MODEL_STORAGE_PATH=/app/models
    restart: unless-stopped
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
      app: arima-forecaster
  template:
    metadata:
      labels:
        app: arima-forecaster
    spec:
      containers:
      - name: api
        image: arima-forecaster:latest
        ports:
        - containerPort: 8000
        env:
        - name: PRODUCTION_MODE
          value: "true"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
---
apiVersion: v1
kind: Service
metadata:
  name: arima-forecaster-service
spec:
  selector:
    app: arima-forecaster
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
```

## 📊 Esempi Completi

### Esempio 1: Pipeline Completa SARIMA

```python
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Base URL API
BASE_URL = "http://localhost:8000"

# 1. Genera dati stagionali di esempio
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=48, freq='M')
trend = np.linspace(100, 200, 48)
seasonal = 10 * np.sin(2 * np.pi * np.arange(48) / 12)
noise = np.random.normal(0, 5, 48)
sales = trend + seasonal + noise

# 2. Auto-selezione parametri ottimali
auto_request = {
    "series": sales.tolist(),
    "model_type": "sarima",
    "p_range": [0, 2],
    "d_range": [0, 1],
    "q_range": [0, 2],
    "P_range": [0, 2],
    "D_range": [0, 1],
    "Q_range": [0, 2],
    "s": 12,
    "ic": "aic",
    "model_id": "sales_auto_sarima"
}

print("🔍 Ricerca parametri ottimali...")
response = requests.post(f"{BASE_URL}/models/auto-select", json=auto_request)
auto_result = response.json()
print(f"✅ Migliori parametri: {auto_result['best_params']}")
print(f"   AIC: {auto_result['best_score']:.2f}")

# 3. Training con parametri ottimali
best_order = auto_result['best_params']['order']
best_seasonal = auto_result['best_params']['seasonal_order']

train_request = {
    "series": sales.tolist(),
    "model_type": "sarima",
    "order": best_order,
    "seasonal_order": best_seasonal,
    "model_id": "sales_final_model"
}

print("\n🎓 Training modello finale...")
response = requests.post(f"{BASE_URL}/models/train", json=train_request)
model_info = response.json()
model_id = model_info['model_id']
print(f"✅ Modello addestrato: {model_id}")

# 4. Attendi completamento training
import time
while True:
    response = requests.get(f"{BASE_URL}/models/{model_id}")
    status = response.json()['status']
    if status == 'completed':
        break
    print(f"   Status: {status}...")
    time.sleep(2)

# 5. Genera previsioni
forecast_request = {
    "steps": 12,
    "confidence_level": 0.95,
    "return_confidence_intervals": True
}

print("\n📈 Generazione previsioni...")
response = requests.post(f"{BASE_URL}/models/{model_id}/forecast", json=forecast_request)
forecast = response.json()

# 6. Diagnostica modello
print("\n🔬 Analisi diagnostica...")
response = requests.post(f"{BASE_URL}/models/{model_id}/diagnostics")
diagnostics = response.json()

print(f"   MAE: {diagnostics['performance_metrics']['mae']:.2f}")
print(f"   MAPE: {diagnostics['performance_metrics']['mape']:.2f}%")
print(f"   R²: {diagnostics['performance_metrics']['r2']:.3f}")

ljung_box = diagnostics['statistical_tests']['ljung_box']
print(f"   Ljung-Box p-value: {ljung_box['p_value']:.3f}")
print(f"   {ljung_box['conclusion']}")

# 7. Genera report
report_request = {
    "format": "html",
    "include_diagnostics": True,
    "include_forecast": True,
    "forecast_steps": 12,
    "custom_title": "Analisi Serie Temporale Vendite",
    "executive_summary": True
}

print("\n📄 Generazione report...")
response = requests.post(f"{BASE_URL}/models/{model_id}/report", json=report_request)
report_info = response.json()
print(f"✅ Report disponibile: {report_info['download_url']}")

# 8. Visualizzazione risultati
plt.figure(figsize=(15, 6))

# Plot dati storici
plt.subplot(1, 2, 1)
plt.plot(dates, sales, label='Dati Storici', color='blue', alpha=0.7)
plt.xlabel('Data')
plt.ylabel('Vendite')
plt.title('Serie Storica')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot previsioni
plt.subplot(1, 2, 2)
future_dates = pd.date_range(dates[-1] + pd.DateOffset(months=1), periods=12, freq='M')
predictions = forecast['forecasts']
lower = forecast['confidence_intervals']['lower']
upper = forecast['confidence_intervals']['upper']

plt.plot(future_dates, predictions, 'r-', label='Previsioni', linewidth=2)
plt.fill_between(future_dates, lower, upper, alpha=0.3, color='red', label='95% CI')
plt.xlabel('Data')
plt.ylabel('Vendite')
plt.title('Previsioni a 12 Mesi')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✅ Pipeline completata con successo!")
```

### Esempio 2: Confronto Multi-Modello

```python
import requests
import json
from tabulate import tabulate

BASE_URL = "http://localhost:8000"

# Dati di esempio
data = [100 + i*5 + np.random.normal(0, 10) for i in range(100)]

# Training di diversi modelli
models = []

# 1. ARIMA semplice
print("Training ARIMA...")
response = requests.post(f"{BASE_URL}/models/train", json={
    "series": data,
    "model_type": "arima",
    "order": [2, 1, 2],
    "model_id": "arima_simple"
})
models.append("arima_simple")

# 2. SARIMA con stagionalità
print("Training SARIMA...")
response = requests.post(f"{BASE_URL}/models/train", json={
    "series": data,
    "model_type": "sarima",
    "order": [1, 1, 1],
    "seasonal_order": [1, 0, 1, 12],
    "model_id": "sarima_seasonal"
})
models.append("sarima_seasonal")

# 3. Prophet
print("Training Prophet...")
dates = pd.date_range('2023-01-01', periods=len(data), freq='D').strftime('%Y-%m-%d').tolist()
response = requests.post(f"{BASE_URL}/models/train/prophet", json={
    "series": data,
    "dates": dates,
    "model_id": "prophet_model",
    "changepoint_prior_scale": 0.05
})
models.append("prophet_model")

# Attendi completamento
import time
time.sleep(10)

# Confronta modelli
print("\nConfronto modelli...")
response = requests.post(f"{BASE_URL}/models/compare", json={"model_ids": models})
comparison = response.json()

# Visualizza risultati in tabella
table_data = []
for model_id, info in comparison['comparison'].items():
    row = [
        model_id,
        info['model_type'],
        info['metrics'].get('mape', 'N/A'),
        info['metrics'].get('rmse', 'N/A'),
        info['metrics'].get('aic', 'N/A'),
        info['rank']
    ]
    table_data.append(row)

headers = ['Model ID', 'Type', 'MAPE', 'RMSE', 'AIC', 'Rank']
print("\n" + tabulate(table_data, headers=headers, tablefmt='grid'))
print(f"\n🏆 Miglior modello: {comparison['best_model']}")
```

### Esempio 3: Monitoring Real-Time con WebSocket

```python
import asyncio
import websockets
import json

async def monitor_training(model_id):
    """Monitora training in real-time via WebSocket."""
    uri = f"ws://localhost:8000/ws/models/{model_id}/status"
    
    async with websockets.connect(uri) as websocket:
        print(f"Connected to model {model_id} status stream")
        
        while True:
            message = await websocket.recv()
            status = json.loads(message)
            
            print(f"Status: {status['status']}")
            print(f"Progress: {status['progress']}%")
            
            if status['status'] in ['completed', 'failed']:
                print(f"Training {status['status']}")
                break
                
            if 'metrics' in status:
                print(f"Current metrics: {status['metrics']}")

# Avvia monitoring
asyncio.run(monitor_training("model_123"))
```

## 🚨 Gestione Errori

### Codici di Stato HTTP

| Codice | Significato | Esempio |
|--------|------------|---------|
| 200 | Success | Operazione completata |
| 201 | Created | Modello creato |
| 400 | Bad Request | Parametri invalidi |
| 404 | Not Found | Modello non trovato |
| 422 | Unprocessable Entity | Dati malformati |
| 500 | Internal Server Error | Errore server |

### Gestione Errori Client

```python
import requests
from requests.exceptions import RequestException

def safe_api_call(url, method="GET", **kwargs):
    """Wrapper sicuro per chiamate API."""
    try:
        if method == "GET":
            response = requests.get(url, **kwargs)
        elif method == "POST":
            response = requests.post(url, **kwargs)
        elif method == "DELETE":
            response = requests.delete(url, **kwargs)
        
        response.raise_for_status()
        return response.json()
        
    except requests.HTTPError as e:
        if e.response.status_code == 404:
            print(f"❌ Risorsa non trovata: {url}")
        elif e.response.status_code == 400:
            print(f"❌ Parametri invalidi: {e.response.json()}")
        elif e.response.status_code == 500:
            print(f"❌ Errore server: {e.response.text}")
        return None
        
    except RequestException as e:
        print(f"❌ Errore connessione: {e}")
        return None

# Utilizzo
result = safe_api_call(
    "http://localhost:8000/models/train",
    method="POST",
    json={"series": [1, 2, 3], "model_type": "arima"}
)
```

## 📈 Performance & Scalabilità

### Best Practices

1. **Batch Processing**
```python
# Processa multiple serie in parallelo
series_batch = {
    "series_1": data1,
    "series_2": data2,
    "series_3": data3
}

for name, data in series_batch.items():
    requests.post(f"{BASE_URL}/models/train", json={
        "series": data,
        "model_type": "arima",
        "model_id": f"batch_{name}"
    })
```

2. **Caching**
```python
# Usa Redis per cache risultati
import redis
import pickle

r = redis.Redis(host='localhost', port=6379, db=0)

def get_cached_forecast(model_id, steps):
    cache_key = f"forecast:{model_id}:{steps}"
    cached = r.get(cache_key)
    
    if cached:
        return pickle.loads(cached)
    
    # Genera nuovo forecast
    response = requests.post(...)
    result = response.json()
    
    # Cache per 1 ora
    r.setex(cache_key, 3600, pickle.dumps(result))
    return result
```

3. **Rate Limiting**
```python
from ratelimit import limits, sleep_and_retry
import requests

# Max 10 chiamate al secondo
@sleep_and_retry
@limits(calls=10, period=1)
def call_api(url, **kwargs):
    return requests.post(url, **kwargs)
```

### Metriche di Performance

| Operazione | Tempo Medio | Note |
|------------|------------|------|
| Training ARIMA (100 obs) | < 1s | Single-threaded |
| Training SARIMA (500 obs) | 2-5s | Con stagionalità |
| Auto-selection (grid 100) | 10-30s | Dipende da n_jobs |
| Prophet training (1000 obs) | 3-8s | Con MCMC |
| Forecast generation | < 100ms | Post-training |
| Report generation | 10-20s | HTML/PDF con Quarto |

## 🛡️ Sicurezza

### Autenticazione JWT

```python
# Configurazione con autenticazione
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Uso negli endpoint
@app.post("/models/train")
async def train_model(
    request: ModelTrainingRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    user = verify_token(credentials)
    # ... resto del codice
```

### Rate Limiting

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100 per minute"]
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/models/train")
@limiter.limit("10 per minute")
async def train_model(request: Request, data: ModelTrainingRequest):
    # ... implementazione
```

## 🤝 Integrazione con Altri Sistemi

### Power BI Integration

```python
# Endpoint per Power BI
@app.get("/powerbi/forecasts/{model_id}")
async def get_powerbi_format(model_id: str, steps: int = 30):
    """Restituisce forecast in formato Power BI."""
    forecast = generate_forecast(model_id, steps)
    
    # Formato tabellare per Power BI
    return {
        "rows": [
            {
                "Date": date,
                "Forecast": value,
                "Lower": lower,
                "Upper": upper
            }
            for date, value, lower, upper in zip(
                forecast["timestamps"],
                forecast["forecasts"],
                forecast["confidence_intervals"]["lower"],
                forecast["confidence_intervals"]["upper"]
            )
        ]
    }
```

### Tableau Integration

```python
# Web Data Connector per Tableau
@app.get("/tableau/wdc")
async def tableau_connector():
    """Genera Web Data Connector per Tableau."""
    return HTMLResponse("""
    <html>
    <head>
        <script src="https://connectors.tableau.com/libs/tableauwdc-2.3.latest.js"></script>
        <script>
        (function() {
            var myConnector = tableau.makeConnector();
            
            myConnector.getSchema = function(schemaCallback) {
                var cols = [
                    {id: "date", dataType: tableau.dataTypeEnum.date},
                    {id: "forecast", dataType: tableau.dataTypeEnum.float},
                    {id: "lower_bound", dataType: tableau.dataTypeEnum.float},
                    {id: "upper_bound", dataType: tableau.dataTypeEnum.float}
                ];
                
                var tableSchema = {
                    id: "forecastFeed",
                    alias: "ARIMA Forecast Data",
                    columns: cols
                };
                
                schemaCallback([tableSchema]);
            };
            
            myConnector.getData = function(table, doneCallback) {
                $.getJSON("/models/latest/forecast?steps=90", function(resp) {
                    table.appendRows(resp.rows);
                    doneCallback();
                });
            };
            
            tableau.registerConnector(myConnector);
        })();
        </script>
    </head>
    </html>
    """)
```

## 📚 Risorse Aggiuntive

### Link Utili
- **Repository GitHub**: https://github.com/your-org/arima-forecaster
- **Documentazione Completa**: https://arima-forecaster.readthedocs.io
- **PyPI Package**: https://pypi.org/project/arima-forecaster
- **Docker Hub**: https://hub.docker.com/r/arima-forecaster/api

### Community & Support
- **Discord**: https://discord.gg/arima-forecaster
- **Stack Overflow Tag**: `arima-forecaster`
- **Email Support**: support@arima-forecaster.com

### Tutorial Video
1. [Getting Started con ARIMA Forecaster API](https://youtube.com/...)
2. [Deploy su Kubernetes](https://youtube.com/...)
3. [Integrazione con Power BI](https://youtube.com/...)

## 📄 Licenza

MIT License - Vedi [LICENSE](LICENSE) per dettagli.

---

**Ultima Modifica**: Agosto 2024  
**Versione Documentazione**: 1.1.0  
**Autore**: ARIMA Forecaster Team