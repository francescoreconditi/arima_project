"""
Router per endpoint di training dei modelli.

Gestisce l'addestramento di modelli ARIMA, SARIMA, VAR e selezione automatica.
"""

import uuid
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
import pandas as pd

from arima_forecaster.api.models import (
    ModelTrainingRequest,
    VARTrainingRequest,
    AutoSelectionRequest,
    ModelInfo
)
from arima_forecaster.api.models_extra import (
    AutoSelectionResult,
    VARModelInfo
)
from arima_forecaster.api.services import ModelManager, ForecastService
from arima_forecaster.utils.logger import get_logger

logger = get_logger(__name__)

# Crea router con prefix e tags
router = APIRouter(
    prefix="/models",
    tags=["Training"],
    responses={404: {"description": "Not found"}}
)

"""
🎨 TRAINING ROUTER

Gestisce l'addestramento di tutti i tipi di modelli:

• POST /models/train        - Addestra modelli ARIMA/SARIMA/SARIMAX
• POST /models/train/var    - Addestra modelli VAR multivariati
• POST /models/auto-select  - Selezione automatica parametri ottimali

Caratteristiche:
- Training asincrono in background
- Validazione automatica dati input
- Gestione errori robusta con fallback
- Salvataggio automatico modelli addestrati
- Calcolo metriche performance in tempo reale
"""


# Funzione per dependency injection dei servizi
def get_services():
    """Dependency per ottenere i servizi necessari."""
    from pathlib import Path
    storage_path = Path("models")
    model_manager = ModelManager(storage_path)
    forecast_service = ForecastService(model_manager)
    return model_manager, forecast_service


async def _train_model_background(
    model_manager: ModelManager,
    model_id: str,
    series: pd.Series,
    request: ModelTrainingRequest
):
    """
    Funzione helper per addestrare il modello in background.
    
    Eseguita come task asincrono per non bloccare la risposta HTTP.
    """
    try:
        # Determina il tipo di modello e crea l'istanza appropriata
        if request.model_type == "arima":
            from arima_forecaster import ARIMAForecaster
            order_tuple = (request.order.p, request.order.d, request.order.q)
            model = ARIMAForecaster(order=order_tuple)
        elif request.model_type == "sarima":
            from arima_forecaster import SARIMAForecaster
            order_tuple = (request.order.p, request.order.d, request.order.q)
            seasonal_order_tuple = (request.seasonal_order.P, request.seasonal_order.D, 
                                   request.seasonal_order.Q, request.seasonal_order.s) if request.seasonal_order else None
            model = SARIMAForecaster(
                order=order_tuple,
                seasonal_order=seasonal_order_tuple
            )
        elif request.model_type == "sarimax":
            from arima_forecaster import SARIMAForecaster
            order_tuple = (request.order.p, request.order.d, request.order.q)
            seasonal_order_tuple = (request.seasonal_order.P, request.seasonal_order.D, 
                                   request.seasonal_order.Q, request.seasonal_order.s) if request.seasonal_order else None
            model = SARIMAForecaster(
                order=order_tuple,
                seasonal_order=seasonal_order_tuple
            )
        else:
            raise ValueError(f"Unknown model type: {request.model_type}")
        
        # Esegue l'addestramento del modello
        model.fit(series)
        
        # Calcola le metriche di valutazione
        from arima_forecaster.evaluation import ModelEvaluator
        evaluator = ModelEvaluator()
        
        # Genera previsioni in-sample per calcolare le metriche
        fitted_values = model.fitted_model.fittedvalues if hasattr(model.fitted_model, 'fittedvalues') else None
        if fitted_values is not None and len(fitted_values) > 0:
            # Allinea le serie per il confronto
            common_index = series.index.intersection(fitted_values.index)
            actual = series[common_index]
            predicted = fitted_values[common_index]
            
            # Calcola metriche di errore
            metrics = evaluator.calculate_forecast_metrics(actual, predicted)
        else:
            metrics = {}
        
        # TODO: Implementare save_model nel ModelManager
        # model_manager.save_model(
        #     model_id=model_id,
        #     model=model,
        #     model_type=request.model_type,
        #     metadata={
        #         "order": request.order,
        #         "seasonal_order": request.seasonal_order,
        #         "training_observations": len(series),
        #         "metrics": metrics,
        #         "status": "completed"
        #     }
        # )
        
        logger.info(f"Model {model_id} training completed successfully")
        
    except Exception as e:
        logger.error(f"Model {model_id} training failed: {e}")
        # TODO: Implementare update_model_status o gestione alternativa degli errori
        # model_manager.update_model_status(model_id, "failed", error=str(e))


async def _train_var_background(
    model_manager: ModelManager,
    model_id: str,
    data: pd.DataFrame,
    request: VARTrainingRequest
):
    """
    Funzione helper per addestrare modelli VAR in background.
    """
    try:
        from arima_forecaster.core.var_model import VARForecaster
        
        # Crea e addestra il modello VAR
        model = VARForecaster(max_lags=request.max_lags)
        model.fit(data)
        
        # Salva il modello
        model_manager.save_model(
            model_id=model_id,
            model=model,
            model_type="var",
            metadata={
                "max_lags": request.max_lags,
                "selected_lag_order": model.selected_lag_order,
                "variables": list(data.columns),
                "training_observations": len(data),
                "status": "completed"
            }
        )
        
        logger.info(f"VAR model {model_id} training completed")
        
    except Exception as e:
        logger.error(f"VAR model {model_id} training failed: {e}")
        # TODO: Implementare update_model_status o gestione alternativa degli errori
        # model_manager.update_model_status(model_id, "failed", error=str(e))


@router.post("/train", response_model=ModelInfo)
async def train_model(
    request: ModelTrainingRequest,
    background_tasks: BackgroundTasks,
    services: tuple = Depends(get_services)
):
    """
    Addestra un nuovo modello ARIMA, SARIMA o SARIMAX.
    
    Questo endpoint avvia l'addestramento in background e restituisce immediatamente
    le informazioni iniziali del modello. Il training continua asincronamente.
    
    <h4>Parametri di Ingresso:</h4>
    <table class="table table-striped">
        <tr><th>Nome</th><th>Tipo</th><th>Descrizione</th><th>Richiesto</th></tr>
        <tr><td>request</td><td>ModelTrainingRequest</td><td>Configurazione per l'addestramento del modello</td><td>Sì</td></tr>
    </table>
    
    <h4>Campi del Request Body:</h4>
    <table class="table table-striped">
        <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th><th>Esempio</th></tr>
        <tr><td>model_type</td><td>str</td><td>Tipo di modello: "arima", "sarima", "sarimax"</td><td>"sarima"</td></tr>
        <tr><td>data</td><td>TimeSeriesData</td><td>Dati della serie temporale</td><td>{timestamps: [...], values: [...]}</td></tr>
        <tr><td>order</td><td>tuple</td><td>Parametri (p,d,q) del modello</td><td>[1, 1, 1]</td></tr>
        <tr><td>seasonal_order</td><td>tuple</td><td>Parametri stagionali (P,D,Q,s)</td><td>[1, 1, 1, 12]</td></tr>
        <tr><td>exogenous_data</td><td>ExogenousData</td><td>Variabili esogene (opzionale)</td><td>null</td></tr>
    </table>
    
    <h4>Risposta di Esempio:</h4>
    <pre><code>
    {
        "model_id": "abc123e4-5678-9012-3456-789012345678",
        "model_type": "sarima",
        "status": "training", 
        "created_at": "2024-08-23T22:30:00.123456",
        "training_observations": 365,
        "parameters": {},
        "metrics": {}
    }
    </code></pre>
    
    <h4>Errori Possibili:</h4>
    <ul>
        <li><strong>400</strong>: Dati non validi o parametri incorretti</li>
        <li><strong>500</strong>: Errore interno del server durante l'inizializzazione</li>
    </ul>
    """
    model_manager, _ = services
    
    try:
        # Converte i dati in pandas Series
        timestamps = pd.to_datetime(request.data.timestamps)
        series = pd.Series(request.data.values, index=timestamps)
        
        # Genera ID univoco
        model_id = str(uuid.uuid4())
        
        # Avvia training in background
        background_tasks.add_task(
            _train_model_background,
            model_manager,
            model_id,
            series,
            request
        )
        
        # Restituisce info iniziali
        return ModelInfo(
            model_id=model_id,
            model_type=request.model_type,
            status="training",
            created_at=datetime.now(),
            training_observations=len(series),
            parameters={},
            metrics={}
        )
        
    except Exception as e:
        logger.error(f"Model training request failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/train/var", response_model=VARModelInfo)
async def train_var_model(
    request: VARTrainingRequest,
    background_tasks: BackgroundTasks,
    services: tuple = Depends(get_services)
):
    """
    Addestra un modello VAR (Vector Autoregression) per serie multivariate.
    
    <h4>Parametri Request Body:</h4>
    <table class="table table-striped">
        <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th></tr>
        <tr><td>data</td><td>MultivariateSeries</td><td>Serie temporali multivariate</td></tr>
        <tr><td>max_lags</td><td>int</td><td>Numero massimo di lag da considerare</td></tr>
    </table>
    
    <h4>Risposta di Esempio:</h4>
    <pre><code>
    {
        "model_id": "var-abc123",
        "model_type": "var",
        "status": "training",
        "created_at": "2024-08-23T14:30:00",
        "variables": ["sales", "temperature", "humidity"],
        "max_lags": 10,
        "selected_lag_order": null,
        "causality_tests": {}
    }
    </code></pre>
    """
    model_manager, _ = services
    
    try:
        # Prepara i dati multivariati
        data = {}
        for series in request.data.series:
            timestamps = pd.to_datetime(series.timestamps)
            data[series.name] = pd.Series(series.values, index=timestamps)
        
        df = pd.DataFrame(data)
        
        # Genera ID e avvia training
        model_id = f"var-{uuid.uuid4().hex[:8]}"
        
        background_tasks.add_task(
            _train_var_background,
            model_manager,
            model_id,
            df,
            request
        )
        
        return VARModelInfo(
            model_id=model_id,
            model_type="var",
            status="training",
            created_at=datetime.now(),
            variables=list(df.columns),
            max_lags=request.max_lags,
            selected_lag_order=None,
            causality_tests={}
        )
        
    except Exception as e:
        logger.error(f"VAR model training failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/auto-select", response_model=AutoSelectionResult)
async def auto_select_model(
    request: AutoSelectionRequest,
    services: tuple = Depends(get_services)
):
    """
    Esegue la selezione automatica dei parametri ottimali per modelli ARIMA/SARIMA.
    
    Utilizza grid search o auto-ARIMA per trovare la combinazione ottimale di parametri
    che minimizza il criterio di informazione specificato (AIC/BIC).
    
    <h4>Parametri Request Body:</h4>
    <table class="table table-striped">
        <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th></tr>
        <tr><td>data</td><td>TimeSeriesData</td><td>Dati della serie temporale</td></tr>
        <tr><td>max_p</td><td>int</td><td>Valore massimo per p (default: 3)</td></tr>
        <tr><td>max_d</td><td>int</td><td>Valore massimo per d (default: 2)</td></tr>
        <tr><td>max_q</td><td>int</td><td>Valore massimo per q (default: 3)</td></tr>
        <tr><td>seasonal</td><td>bool</td><td>Se considerare modelli stagionali</td></tr>
        <tr><td>seasonal_period</td><td>int</td><td>Periodo stagionale (es. 12 per dati mensili)</td></tr>
        <tr><td>criterion</td><td>str</td><td>Criterio di selezione: "aic" o "bic"</td></tr>
    </table>
    
    <h4>Risposta di Esempio:</h4>
    <pre><code>
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
    </code></pre>
    """
    _, forecast_service = services
    
    try:
        # Converte i dati
        timestamps = pd.to_datetime(request.data.timestamps)
        series = pd.Series(request.data.values, index=timestamps)
        
        # Esegue la ricerca
        import time
        start_time = time.time()
        
        if request.seasonal:
            from arima_forecaster.core.sarima_selection import SARIMAModelSelector
            selector = SARIMAModelSelector(
                max_p=request.max_p,
                max_d=request.max_d,
                max_q=request.max_q,
                seasonal_period=request.seasonal_period,
                criterion=request.criterion
            )
        else:
            from arima_forecaster.core.model_selection import ARIMAModelSelector
            selector = ARIMAModelSelector(
                max_p=request.max_p,
                max_d=request.max_d,
                max_q=request.max_q,
                criterion=request.criterion
            )
        
        best_model, all_results = selector.search(series)
        
        search_time = time.time() - start_time
        
        # Prepara la risposta
        best_params = best_model.get_params()
        
        return AutoSelectionResult(
            best_model={
                "order": best_params.get("order", []),
                "seasonal_order": best_params.get("seasonal_order", []),
                "aic": best_model.aic if hasattr(best_model, 'aic') else None,
                "bic": best_model.bic if hasattr(best_model, 'bic') else None
            },
            all_models=[
                {
                    "order": result["params"].get("order", []),
                    "seasonal_order": result["params"].get("seasonal_order", []),
                    "aic": result.get("aic"),
                    "bic": result.get("bic")
                }
                for result in all_results[:10]  # Limita ai top 10
            ],
            search_time_seconds=search_time
        )
        
    except Exception as e:
        logger.error(f"Auto-selection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))