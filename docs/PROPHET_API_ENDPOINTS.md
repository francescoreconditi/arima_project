# 🚀 Prophet API Endpoints - Guida Completa

## 📋 Panoramica

L'API ARIMA Forecaster include supporto completo per Facebook Prophet con endpoint dedicati per training, auto-selection, forecasting avanzato e comparazione modelli.

**Base URL**: `http://localhost:8000`  
**Documentazione Interattiva**: `http://localhost:8000/docs` (Swagger UI)  
**Documentazione Alternativa**: `http://localhost:8000/scalar` (Scalar UI)

---

## 🎯 Endpoints Prophet Disponibili

### 1. **Training Prophet Base**
**`POST /models/train/prophet`**

Addestra un modello Facebook Prophet con parametri specifici.

#### 📋 Request Body
```json
{
    "data": {
        "timestamps": ["2023-01-01", "2023-01-02", "2023-01-03", ...],
        "values": [100.5, 102.3, 98.7, ...]
    },
    "growth": "linear",
    "yearly_seasonality": "auto",
    "weekly_seasonality": true,
    "daily_seasonality": false,
    "seasonality_mode": "additive",
    "country_holidays": "IT",
    "changepoint_prior_scale": 0.05,
    "seasonality_prior_scale": 10.0,
    "holidays_prior_scale": 10.0
}
```

#### 📊 Response
```json
{
    "model_id": "prophet-abc123e4-5678-9012-3456-789012345678",
    "status": "training",
    "message": "Addestramento modello Prophet avviato in background",
    "estimated_time_seconds": 60,
    "endpoint_check": "/models/prophet-abc123e4-5678-9012-3456-789012345678/status"
}
```

#### 🔧 Parametri Prophet
| Parametro | Tipo | Descrizione | Valori | Default |
|-----------|------|-------------|--------|---------|
| `growth` | string | Tipo di crescita | "linear", "logistic", "flat" | "linear" |
| `yearly_seasonality` | string/bool | Stagionalità annuale | true, false, "auto" | "auto" |
| `weekly_seasonality` | string/bool | Stagionalità settimanale | true, false, "auto" | "auto" |
| `daily_seasonality` | string/bool | Stagionalità giornaliera | true, false, "auto" | "auto" |
| `seasonality_mode` | string | Modalità stagionalità | "additive", "multiplicative" | "additive" |
| `country_holidays` | string | Codice paese festività | "IT", "US", "UK", "DE", "FR", "ES" | null |
| `changepoint_prior_scale` | float | Flessibilità trend | 0.001 - 0.5 | 0.05 |
| `seasonality_prior_scale` | float | Forza stagionalità | 0.01 - 10.0 | 10.0 |
| `holidays_prior_scale` | float | Forza holiday effects | 0.01 - 10.0 | 10.0 |

---

### 2. **Auto-Selection Prophet**
**`POST /models/train/prophet/auto-select`**

Selezione automatica dei parametri ottimali Prophet con Grid Search + Cross-Validation.

#### 📋 Request Body
```json
{
    "data": {
        "timestamps": ["2023-01-01", "2023-02-01", "2023-03-01", ...],
        "values": [100, 105, 98, ...]
    },
    "growth_types": ["linear", "logistic"],
    "seasonality_modes": ["additive", "multiplicative"],
    "country_holidays": ["IT", "US", null],
    "max_models": 30,
    "cv_horizon": "30 days"
}
```

#### 📊 Response
```json
{
    "model_id": "prophet-auto-def456",
    "status": "auto_selecting",
    "message": "Selezione automatica Prophet avviata (testando fino a 30 modelli)",
    "estimated_time_seconds": 120,
    "endpoint_check": "/models/prophet-auto-def456/status",
    "search_space": {
        "total_combinations": 12,
        "max_models_tested": 30
    }
}
```

#### ⚙️ Algoritmi di Ottimizzazione
1. **Grid Search**: Esplorazione sistematica spazio parametri
2. **Random Search**: Campionamento casuale efficiente  
3. **Bayesian Optimization**: TPE con Optuna (se disponibile)

#### 🎯 Processo Ottimizzazione
1. **Cross-Validation**: Rolling forecast origin
2. **Metric Selection**: MAPE, MAE, RMSE
3. **Best Model**: Scelta automatica migliore configurazione
4. **Final Training**: Riaddestramento su dati completi

---

### 3. **Lista Modelli Prophet**
**`GET /models/train/prophet/models`**

Elenca tutti i modelli Prophet nel sistema con metadati completi.

#### 📊 Response
```json
{
    "models": [
        {
            "model_id": "prophet-abc123",
            "model_type": "prophet",
            "status": "completed",
            "parameters": {
                "growth": "linear",
                "seasonality_mode": "additive",
                "country_holidays": "IT"
            },
            "metrics": {
                "mape": 8.5,
                "mae": 12.3,
                "rmse": 15.7
            },
            "created_at": "2024-08-28T10:30:00",
            "completed_at": "2024-08-28T10:31:30"
        }
    ],
    "total_count": 1,
    "by_status": {
        "completed": 1,
        "training": 0,
        "failed": 0
    },
    "model_types": {
        "prophet": 1,
        "prophet-auto": 0
    }
}
```

---

### 4. **Forecasting Standard**
**`POST /models/{model_id}/forecast`**

Generazione previsioni standard (compatibile con tutti i modelli).

#### 📋 Request Body
```json
{
    "steps": 30,
    "confidence_level": 0.95,
    "return_intervals": true
}
```

#### 📊 Response
```json
{
    "forecast": [105.2, 107.8, 103.5, ...],
    "timestamps": ["2024-09-01", "2024-09-02", "2024-09-03", ...],
    "confidence_intervals": {
        "lower": [100.1, 102.3, 98.2, ...],
        "upper": [110.3, 113.3, 108.8, ...]
    },
    "model_id": "prophet-abc123",
    "forecast_steps": 30
}
```

---

### 5. **🆕 Forecasting Prophet Avanzato**
**`POST /models/{model_id}/forecast/prophet`**

Forecasting Prophet con decomposizione componenti avanzata.

#### 📋 Request Body
```json
{
    "steps": 30,
    "confidence_level": 0.95,
    "return_intervals": true
}
```

#### 📊 Response Estesa
```json
{
    "forecast": [105.2, 107.8, 103.5, ...],
    "timestamps": ["2024-09-01", "2024-09-02", ...],
    "confidence_intervals": {
        "lower": [100.1, 102.3, ...],
        "upper": [110.3, 113.3, ...]
    },
    "prophet_components": {
        "trend": [102.1, 102.2, 102.3, ...],
        "weekly": [2.1, 4.6, 0.2, ...],
        "yearly": [1.0, 1.0, 1.0, ...],
        "holidays": [0.0, 0.0, 0.0, ...]
    },
    "changepoints": {
        "dates": ["2024-03-15", "2024-06-20"],
        "trend_changes": [-0.5, 1.2]
    },
    "model_id": "prophet-abc123",
    "forecast_steps": 30,
    "model_type": "prophet",
    "decomposition_info": {
        "trend_type": "linear",
        "seasonality_mode": "additive",
        "holidays_included": true
    }
}
```

#### 🔮 Componenti Prophet
- **trend**: Componente di trend puro (linear/logistic)
- **weekly**: Stagionalità settimanale specifica
- **yearly**: Stagionalità annuale (se abilitata)
- **holidays**: Effetti festività (se configurate)
- **seasonal**: Stagionalità combinata
- **changepoints**: Punti di cambio trend automatici

---

### 6. **🆕 Comparazione Modelli**
**`POST /models/compare`**

Confronta performance di più modelli (Prophet vs ARIMA vs SARIMA).

#### 📋 Request Body
```json
{
    "model_ids": [
        "prophet-abc123",
        "arima-def456", 
        "sarima-ghi789"
    ]
}
```

#### 📊 Response Comparativa
```json
{
    "comparison_summary": {
        "best_model": {
            "model_id": "prophet-abc123",
            "model_type": "prophet",
            "overall_score": 0.85,
            "reason": "Miglior bilanciamento accuracy/performance"
        },
        "total_models": 3,
        "comparison_timestamp": "2024-08-28T15:30:00"
    },
    "detailed_comparison": {
        "prophet-abc123": {
            "model_type": "prophet",
            "metrics": {
                "mape": 8.5,
                "mae": 12.3,
                "rmse": 15.7
            },
            "performance": {
                "training_time_seconds": 45.2,
                "prediction_speed_ms": 120,
                "memory_mb": 25.4
            },
            "strengths": [
                "Gestione stagionalità avanzata",
                "Holiday effects nativi",
                "Robusto agli outlier",
                "Interpretabilità trend"
            ],
            "weaknesses": [
                "Training più lento",
                "Maggior uso memoria",
                "Richiede dati lunghi"
            ]
        },
        "arima-def456": {
            "model_type": "arima",
            "metrics": {
                "mape": 12.1,
                "mae": 15.8,
                "rmse": 18.9
            },
            "performance": {
                "training_time_seconds": 12.8,
                "prediction_speed_ms": 45,
                "memory_mb": 8.2
            },
            "strengths": [
                "Training veloce",
                "Memoria ridotta",
                "Teoria statistica solida",
                "Controllo parametri preciso"
            ],
            "weaknesses": [
                "Preprocessing richiesto",
                "Stagionalità complessa",
                "Sensibile a outlier"
            ]
        }
    },
    "recommendations": {
        "best_for_accuracy": "prophet-abc123",
        "best_for_speed": "arima-def456",
        "best_for_interpretability": "arima-def456",
        "best_for_seasonality": "prophet-abc123"
    },
    "scoring_methodology": {
        "accuracy_weight": 0.4,
        "speed_weight": 0.2,
        "memory_weight": 0.15,
        "interpretability_weight": 0.15,
        "robustness_weight": 0.1
    }
}
```

#### 🏆 Metriche di Confronto
- **Accuratezza**: MAE, RMSE, MAPE
- **Velocità**: Training time, prediction speed
- **Memoria**: RAM footprint  
- **Interpretabilità**: Comprensibilità business
- **Robustezza**: Stabilità outlier/missing data

---

## 🔄 Workflow Completo Prophet

### 1. **Training Semplice**
```bash
# Training Prophet base
curl -X POST "http://localhost:8000/models/train/prophet" \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "timestamps": ["2023-01-01", "2023-01-02", "2023-01-03"],
      "values": [100, 105, 98]
    },
    "growth": "linear",
    "country_holidays": "IT"
  }'
```

### 2. **Auto-Selection**
```bash
# Auto-selection parametri ottimali
curl -X POST "http://localhost:8000/models/train/prophet/auto-select" \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "timestamps": ["2023-01-01", "2023-02-01", "2023-03-01"],
      "values": [100, 105, 98]
    },
    "growth_types": ["linear", "logistic"],
    "max_models": 20
  }'
```

### 3. **Forecasting Avanzato**
```bash
# Previsioni con decomposizione
curl -X POST "http://localhost:8000/models/prophet-abc123/forecast/prophet" \
  -H "Content-Type: application/json" \
  -d '{
    "steps": 30,
    "confidence_level": 0.95,
    "return_intervals": true
  }'
```

### 4. **Comparazione Modelli**
```bash
# Confronta Prophet vs ARIMA
curl -X POST "http://localhost:8000/models/compare" \
  -H "Content-Type: application/json" \
  -d '{
    "model_ids": ["prophet-abc123", "arima-def456"]
  }'
```

---

## 📊 Prophet vs ARIMA: Quando Usare Cosa

### ✅ **Usa Prophet Se:**
- ✅ Dati business con stagionalità forte (vendite, traffico web)
- ✅ Effetti festività/eventi importanti per il business  
- ✅ Serie con trend growth e possibile saturazione
- ✅ Team business-oriented senza background statistico
- ✅ Vuoi interpretabilità trend/stagionalità separate
- ✅ Hai >2 anni di dati storici giornalieri
- ✅ Tolleranza per maggior compute cost

### ✅ **Usa ARIMA Se:**
- ✅ Dati finanziari o economici ad alta frequenza
- ✅ Serie principalmente stazionaria con correlazioni lag
- ✅ Serving real-time con latency <100ms
- ✅ Budget computazionale limitato
- ✅ Spiegabilità matematica richiesta dal business
- ✅ Serie IoT/sensor con rumore e pattern brevi
- ✅ Hai esperienza in econometria/statistica

### 🤝 **Usa Ensemble Se:**
- 🤝 High-stakes forecasting (revenue critical)
- 🤝 Vuoi robustezza contro diversi pattern
- 🤝 Hai risorse computazionali abbondanti
- 🤝 Accuracy > interpretability

---

## 🚨 Errori Comuni e Soluzioni

### 1. **Prophet Training Fallito**
```json
{
    "detail": "Facebook Prophet non disponibile. Installa con: pip install prophet"
}
```
**Soluzione**: Installa Prophet con `uv add prophet`

### 2. **Dati Insufficienti**
```json
{
    "detail": "Prophet auto-selection richiede almeno 10 osservazioni"
}
```
**Soluzione**: Fornisci serie più lunghe (raccomandato: >365 giorni)

### 3. **Modello Non Prophet**
```json
{
    "detail": "This endpoint is only for Prophet models. Model type: arima"
}
```
**Soluzione**: Usa endpoint forecast standard per modelli non-Prophet

### 4. **Status Non Completato**
```json
{
    "detail": "Model must be completed for forecasting. Current status: training"
}
```
**Soluzione**: Attendi completamento training o controlla status con `GET /models/{id}`

---

## 🎯 Best Practices API

### 1. **Monitoraggio Status**
```python
import requests
import time

# Avvia training
response = requests.post("/models/train/prophet", json=data)
model_id = response.json()["model_id"]

# Polling status
while True:
    status = requests.get(f"/models/{model_id}").json()["status"]
    if status == "completed":
        break
    elif status == "failed":
        raise Exception("Training failed")
    time.sleep(5)
```

### 2. **Gestione Errori Robusti**
```python
try:
    response = requests.post("/models/train/prophet/auto-select", json=request_data)
    response.raise_for_status()
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 400:
        print("Dati non validi:", e.response.json()["detail"])
    elif e.response.status_code == 500:
        print("Errore server:", e.response.json()["detail"])
```

### 3. **Batch Processing**
```python
# Training multipli paralleli
model_requests = [prophet_config_1, prophet_config_2, prophet_config_3]
model_ids = []

for config in model_requests:
    response = requests.post("/models/train/prophet", json=config)
    model_ids.append(response.json()["model_id"])

# Comparazione finale
comparison = requests.post("/models/compare", json={"model_ids": model_ids})
best_model = comparison.json()["comparison_summary"]["best_model"]
```

---

## 📚 Risorse Aggiuntive

### 📖 **Documentazione**
- **Swagger UI**: http://localhost:8000/docs
- **Scalar UI**: http://localhost:8000/scalar (interfaccia moderna)
- **Prophet vs ARIMA Guide**: `docs/prophet_vs_arima_sarima.md`

### 🧪 **Testing**
- **Health Check**: `GET /health`
- **API Status**: `GET /`
- **Demo Scripts**: `examples/prophet_auto_selection_demo.py`

### 🔧 **Configurazione**
```python
# Avvio API con configurazione custom
from arima_forecaster.api.main import create_app

app = create_app(
    model_storage_path="/custom/models/path",
    enable_scalar=True,
    production_mode=False
)
```

### 📈 **Performance Tuning**
```python
# Prophet ottimizzato per performance
{
    "changepoint_prior_scale": 0.05,  # Meno flessibile = più veloce
    "yearly_seasonality": False,      # Disabilita se non necessario
    "daily_seasonality": False,       # Disabilita se freq > giornaliera
    "mcmc_samples": 0,               # Disabilita MCMC per velocità
    "uncertainty_samples": 100       # Riduci per forecasting veloce
}
```

---

**🚀 Prophet API Integration: PRODUCTION READY!**

*Documentazione aggiornata: 28 Agosto 2024*  
*Versione API: 1.1.0*