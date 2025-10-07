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
    ProphetTrainingRequest,
    ProphetAutoSelectionRequest,
    ModelInfo,
)
from arima_forecaster.api.models_extra import AutoSelectionResult, VARModelInfo
from arima_forecaster.api.services import ModelManager, ForecastService
from arima_forecaster.api.examples import ARIMA_TRAINING_EXAMPLES, TRAINING_RESPONSE_EXAMPLES
from arima_forecaster.utils.logger import get_logger

logger = get_logger(__name__)

# Crea router con prefix e tags
router = APIRouter(
    prefix="/models", tags=["Training"], responses={404: {"description": "Not found"}}
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
    model_manager: ModelManager, model_id: str, series: pd.Series, request: ModelTrainingRequest
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
            seasonal_order_tuple = (
                (
                    request.seasonal_order.P,
                    request.seasonal_order.D,
                    request.seasonal_order.Q,
                    request.seasonal_order.s,
                )
                if request.seasonal_order
                else None
            )
            model = SARIMAForecaster(order=order_tuple, seasonal_order=seasonal_order_tuple)
        elif request.model_type == "sarimax":
            from arima_forecaster import SARIMAForecaster

            order_tuple = (request.order.p, request.order.d, request.order.q)
            seasonal_order_tuple = (
                (
                    request.seasonal_order.P,
                    request.seasonal_order.D,
                    request.seasonal_order.Q,
                    request.seasonal_order.s,
                )
                if request.seasonal_order
                else None
            )
            model = SARIMAForecaster(order=order_tuple, seasonal_order=seasonal_order_tuple)
        elif request.model_type == "prophet":
            try:
                from arima_forecaster.core import ProphetForecaster

                model = ProphetForecaster(
                    growth=request.growth,
                    yearly_seasonality=request.yearly_seasonality,
                    weekly_seasonality=request.weekly_seasonality,
                    daily_seasonality=request.daily_seasonality,
                    seasonality_mode=request.seasonality_mode,
                    country_holidays=request.country_holidays,
                    changepoint_prior_scale=request.changepoint_prior_scale,
                    seasonality_prior_scale=request.seasonality_prior_scale,
                    holidays_prior_scale=request.holidays_prior_scale,
                )
            except ImportError:
                raise HTTPException(
                    status_code=400,
                    detail="Facebook Prophet non disponibile. Installa con: pip install prophet",
                )
        else:
            raise ValueError(f"Unknown model type: {request.model_type}")

        # Esegue l'addestramento del modello
        model.fit(series)

        # Calcola le metriche di valutazione
        from arima_forecaster.evaluation import ModelEvaluator

        evaluator = ModelEvaluator()

        # Genera previsioni in-sample per calcolare le metriche
        fitted_values = (
            model.fitted_model.fittedvalues if hasattr(model.fitted_model, "fittedvalues") else None
        )
        if fitted_values is not None and len(fitted_values) > 0:
            # Allinea le serie per il confronto
            common_index = series.index.intersection(fitted_values.index)
            actual = series[common_index]
            predicted = fitted_values[common_index]

            # Calcola metriche di errore
            metrics = evaluator.calculate_forecast_metrics(actual, predicted)
        else:
            metrics = {}

        # Salva il modello nel registry con metadati completi
        model_manager.save_model(
            model_id=model_id,
            model=model,
            model_type=request.model_type,
            metadata={
                "parameters": {
                    "order": {
                        "p": request.order.p,
                        "d": request.order.d,
                        "q": request.order.q
                    },
                    "seasonal_order": {
                        "p": request.seasonal_order.p,
                        "d": request.seasonal_order.d,
                        "q": request.seasonal_order.q,
                        "P": request.seasonal_order.P,
                        "D": request.seasonal_order.D,
                        "Q": request.seasonal_order.Q,
                        "s": request.seasonal_order.s
                    } if request.seasonal_order else None,
                },
                "training_observations": len(series),
                "metrics": metrics,
                "status": "completed",
                "created_at": datetime.now()
            }
        )

        logger.info(f"Model {model_id} training completed successfully")

    except Exception as e:
        logger.error(f"Model {model_id} training failed: {e}")
        # Aggiorna lo stato del modello a "failed" nel registry
        if model_id in model_manager.model_registry:
            model_manager.model_registry[model_id]["status"] = "failed"
            model_manager.model_registry[model_id]["error"] = str(e)
            model_manager._save_registry()


async def _train_var_background(
    model_manager: ModelManager, model_id: str, data: pd.DataFrame, request: VARTrainingRequest
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
                "status": "completed",
            },
        )

        logger.info(f"VAR model {model_id} training completed")

    except Exception as e:
        logger.error(f"VAR model {model_id} training failed: {e}")
        # TODO: Implementare update_model_status o gestione alternativa degli errori
        # model_manager.update_model_status(model_id, "failed", error=str(e))


@router.post(
    "/train",
    response_model=ModelInfo,
    responses={
        200: {
            "description": "Modello in training",
            "content": {"application/json": {"examples": TRAINING_RESPONSE_EXAMPLES}},
        },
        400: {
            "description": "Dati non validi",
            "content": {
                "application/json": {"example": TRAINING_RESPONSE_EXAMPLES["training_error"]}
            },
        },
    },
)
async def train_model(
    request: ModelTrainingRequest,
    background_tasks: BackgroundTasks,
    services: tuple = Depends(get_services),
):
    """
    Addestra un nuovo modello ARIMA, SARIMA o SARIMAX.

    Questo endpoint avvia l'addestramento in background e restituisce immediatamente
    le informazioni iniziali del modello. Il training continua asincronamente.

    <h4>Parametri di Ingresso:</h4>
    <table >
        <tr><th>Nome</th><th>Tipo</th><th>Descrizione</th><th>Richiesto</th></tr>
        <tr><td>request</td><td>ModelTrainingRequest</td><td>Configurazione per l'addestramento del modello</td><td>Sì</td></tr>
    </table>

    <h4>Campi del Request Body:</h4>
    <table >
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
        background_tasks.add_task(_train_model_background, model_manager, model_id, series, request)

        # Restituisce info iniziali
        return ModelInfo(
            model_id=model_id,
            model_type=request.model_type,
            status="training",
            created_at=datetime.now(),
            training_observations=len(series),
            parameters={},
            metrics={},
        )

    except Exception as e:
        logger.error(f"Model training request failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/train/var", response_model=VARModelInfo)
async def train_var_model(
    request: VARTrainingRequest,
    background_tasks: BackgroundTasks,
    services: tuple = Depends(get_services),
):
    """
    Addestra un modello VAR (Vector Autoregression) per serie multivariate.

    <h4>Parametri Request Body:</h4>
    <table >
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

        background_tasks.add_task(_train_var_background, model_manager, model_id, df, request)

        return VARModelInfo(
            model_id=model_id,
            model_type="var",
            status="training",
            created_at=datetime.now(),
            variables=list(df.columns),
            max_lags=request.max_lags,
            selected_lag_order=None,
            causality_tests={},
        )

    except Exception as e:
        logger.error(f"VAR model training failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/auto-select", response_model=AutoSelectionResult)
async def auto_select_model(request: AutoSelectionRequest, services: tuple = Depends(get_services)):
    """
    Esegue la selezione automatica dei parametri ottimali per modelli ARIMA/SARIMA.

    Utilizza grid search o auto-ARIMA per trovare la combinazione ottimale di parametri
    che minimizza il criterio di informazione specificato (AIC/BIC).

    <h4>Parametri Request Body:</h4>
    <table >
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
                p_range=(0, request.max_p),
                d_range=(0, request.max_d),
                q_range=(0, request.max_q),
                seasonal_periods=[request.seasonal_period],
                information_criterion=request.criterion,
            )
        else:
            from arima_forecaster.core.model_selection import ARIMAModelSelector

            selector = ARIMAModelSelector(
                p_range=(0, request.max_p),
                d_range=(0, request.max_d),
                q_range=(0, request.max_q),
                information_criterion=request.criterion,
            )

        best_order = selector.search(series)

        search_time = time.time() - start_time

        # Accede ai results dalla classe selector
        all_results_data = selector.results if hasattr(selector, "results") else []

        # Prepara il best model
        best_result = all_results_data[0] if all_results_data else None

        return AutoSelectionResult(
            best_model={
                "order": list(best_order) if not request.seasonal else list(best_order[:3]),
                "seasonal_order": list(best_order[3:]) if request.seasonal and len(best_order) > 3 else None,
                "aic": best_result.get(request.criterion) if best_result else None,
                "bic": best_result.get("bic") if best_result else None,
            },
            all_results=[
                {
                    "order": result.get("order", []),
                    "seasonal_order": result.get("seasonal_order"),
                    "aic": result.get("aic"),
                    "bic": result.get("bic"),
                }
                for result in all_results_data[:20]  # Top 20
            ],
            models_tested=len(all_results_data),
            search_time_seconds=search_time,
        )

    except Exception as e:
        logger.error(f"Auto-selection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _train_prophet_background(
    model_manager: ModelManager, model_id: str, series: pd.Series, request: ProphetTrainingRequest
):
    """
    Funzione helper per addestrare modelli Prophet in background.
    """
    try:
        from arima_forecaster.core import ProphetForecaster

        # Crea il modello Prophet con i parametri specificati
        model = ProphetForecaster(
            growth=request.growth,
            yearly_seasonality=request.yearly_seasonality,
            weekly_seasonality=request.weekly_seasonality,
            daily_seasonality=request.daily_seasonality,
            seasonality_mode=request.seasonality_mode,
            country_holidays=request.country_holidays,
            changepoint_prior_scale=request.changepoint_prior_scale,
            seasonality_prior_scale=request.seasonality_prior_scale,
            holidays_prior_scale=request.holidays_prior_scale,
        )

        # Addestra il modello
        model.fit(series)

        # Calcola metriche di performance
        from arima_forecaster.evaluation import ModelEvaluator

        evaluator = ModelEvaluator()

        # Per Prophet, generiamo previsioni in-sample
        in_sample_forecast = model.predict(steps=len(series))
        if len(in_sample_forecast) == len(series):
            metrics = evaluator.calculate_metrics(series, in_sample_forecast)
        else:
            # Fallback se le dimensioni non combaciano
            metrics = {"mae": None, "rmse": None, "mape": None}

        # Salva il modello
        model_manager.save_model(model, model_id)

        # Aggiorna lo stato nel model manager
        model_manager.update_model_status(
            model_id,
            "completed",
            {
                "metrics": metrics,
                "model_params": {
                    "growth": request.growth,
                    "seasonality_mode": request.seasonality_mode,
                    "yearly_seasonality": request.yearly_seasonality,
                    "weekly_seasonality": request.weekly_seasonality,
                    "daily_seasonality": request.daily_seasonality,
                    "country_holidays": request.country_holidays,
                },
                "completed_at": datetime.now().isoformat(),
            },
        )

        logger.info(f"Prophet model {model_id} training completed successfully")

    except Exception as e:
        model_manager.update_model_status(model_id, "failed", {"error": str(e)})
        logger.error(f"Prophet model {model_id} training failed: {e}")


@router.post("/train/prophet", response_model=Dict[str, Any])
async def train_prophet_model(
    request: ProphetTrainingRequest,
    background_tasks: BackgroundTasks,
    services=Depends(get_services),
):
    """
    Addestra un modello Facebook Prophet per forecasting avanzato.

    Facebook Prophet è un modello di forecasting robusto sviluppato da Meta che eccelle
    nella gestione di serie temporali con pattern complessi, trend non lineari e
    stagionalità multiple.

    <h3>Parametri Request Body:</h3>
    <table>
    <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th></tr>
    <tr><td>data</td><td>TimeSeriesData</td><td>Dati della serie temporale</td></tr>
    <tr><td>growth</td><td>str</td><td>Tipo di crescita: "linear", "logistic", "flat" (default: "linear")</td></tr>
    <tr><td>yearly_seasonality</td><td>Union[bool, str, int]</td><td>Stagionalità annuale (default: "auto")</td></tr>
    <tr><td>weekly_seasonality</td><td>Union[bool, str, int]</td><td>Stagionalità settimanale (default: "auto")</td></tr>
    <tr><td>daily_seasonality</td><td>Union[bool, str, int]</td><td>Stagionalità giornaliera (default: "auto")</td></tr>
    <tr><td>seasonality_mode</td><td>str</td><td>Modalità: "additive" o "multiplicative" (default: "additive")</td></tr>
    <tr><td>country_holidays</td><td>str</td><td>Codice paese: IT, US, UK, DE, FR, ES (opzionale)</td></tr>
    <tr><td>changepoint_prior_scale</td><td>float</td><td>Flessibilità trend (default: 0.05)</td></tr>
    <tr><td>seasonality_prior_scale</td><td>float</td><td>Forza stagionalità (default: 10.0)</td></tr>
    <tr><td>holidays_prior_scale</td><td>float</td><td>Impatto festività (default: 10.0)</td></tr>
    </table>

    <h3>Caratteristiche Principali:</h3>
    <ul>
    <li><b>Trend automatico:</b> Rileva automaticamente cambiamenti di trend</li>
    <li><b>Stagionalità multipla:</b> Supporta pattern giornalieri, settimanali e annuali</li>
    <li><b>Gestione festività:</b> Integrazione calendario festività per paese</li>
    <li><b>Robusto agli outlier:</b> Gestisce automaticamente valori anomali</li>
    <li><b>Valori mancanti:</b> Non richiede preprocessing per gap nei dati</li>
    </ul>

    <h3>Vantaggi Prophet:</h3>
    <ul>
    <li>Interpretabile e comprensibile per analisti business</li>
    <li>Gestisce automaticamente trend complessi e changepoints</li>
    <li>Eccellente per serie con forte stagionalità e holiday effects</li>
    <li>Robusto con dati rumorosi e outliers</li>
    <li>Non richiede preprocessing elaborato o stazionarietà</li>
    </ul>
    """
    try:
        model_manager, forecast_service = services

        # Genera un ID univoco per il modello
        model_id = str(uuid.uuid4())

        # Converte i dati in pandas Series
        timestamps = pd.to_datetime(request.data.timestamps)
        series = pd.Series(request.data.values, index=timestamps)

        # Valida la serie temporale
        if series.empty or series.isnull().all():
            raise HTTPException(
                status_code=400,
                detail="La serie temporale non può essere vuota o contenere solo valori nulli",
            )

        # Registra il modello come "in training"
        model_manager.register_model(
            model_id,
            "prophet",
            "training",
            {
                "created_at": datetime.now().isoformat(),
                "parameters": {
                    "growth": request.growth,
                    "seasonality_mode": request.seasonality_mode,
                    "country_holidays": request.country_holidays,
                },
            },
        )

        # Avvia il training in background
        background_tasks.add_task(
            _train_prophet_background, model_manager, model_id, series, request
        )

        logger.info(f"Started Prophet model training with ID: {model_id}")

        return {
            "model_id": model_id,
            "status": "training",
            "message": "Addestramento modello Prophet avviato in background",
            "estimated_time_seconds": 60,  # Prophet è generalmente veloce
            "endpoint_check": f"/models/{model_id}/status",
        }

    except Exception as e:
        logger.error(f"Prophet training request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train/prophet/auto-select")
async def train_prophet_auto_select(
    request: ProphetAutoSelectionRequest,
    background_tasks: BackgroundTasks,
    services=Depends(get_services),
):
    """
    Selezione automatica di parametri ottimali per modelli Prophet.

    Esegue una ricerca sistematica sui parametri Prophet utilizzando cross-validation
    per trovare la configurazione ottimale che minimizza l'errore di previsione.

    <h3>Parametri Request Body:</h3>
    <table>
    <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th></tr>
    <tr><td>data</td><td>TimeSeriesData</td><td>Dati della serie temporale</td></tr>
    <tr><td>growth_types</td><td>List[str]</td><td>Tipi di crescita da testare: ["linear", "logistic", "flat"]</td></tr>
    <tr><td>seasonality_modes</td><td>List[str]</td><td>Modalità stagionalità: ["additive", "multiplicative"]</td></tr>
    <tr><td>country_holidays</td><td>List[Optional[str]]</td><td>Calendari festività: ["IT", "US", "UK", "DE", "FR", "ES", null]</td></tr>
    <tr><td>max_models</td><td>int</td><td>Numero massimo di modelli da testare (default: 30)</td></tr>
    <tr><td>cv_horizon</td><td>str</td><td>Orizzonte cross-validation es. "30 days" (default: "30 days")</td></tr>
    <tr><td>cv_initial</td><td>Optional[str]</td><td>Periodo iniziale training es. "365 days" (opzionale)</td></tr>
    <tr><td>cv_period</td><td>Optional[str]</td><td>Frequenza cutoff es. "180 days" (opzionale)</td></tr>
    <tr><td>metric</td><td>str</td><td>Metrica ottimizzazione: "mape", "mae", "rmse" (default: "mape")</td></tr>
    </table>

    <h3>Processo di Ottimizzazione:</h3>
    <ol>
    <li><b>Grid Search:</b> Testa tutte le combinazioni di parametri specificati</li>
    <li><b>Cross-Validation:</b> Valuta ogni modello con rolling forecast origin</li>
    <li><b>Metric Selection:</b> Sceglie il modello con miglior performance</li>
    <li><b>Final Training:</b> Riaddestra il modello migliore su tutti i dati</li>
    </ol>

    <h3>Metriche di Valutazione:</h3>
    <ul>
    <li><b>MAPE:</b> Mean Absolute Percentage Error (errore percentuale)</li>
    <li><b>MAE:</b> Mean Absolute Error (errore assoluto)</li>
    <li><b>RMSE:</b> Root Mean Square Error (penalizza errori grandi)</li>
    <li><b>Coverage:</b> Copertura degli intervalli di confidenza</li>
    </ul>

    <h3>Performance Attese:</h3>
    <ul>
    <li>Tempo tipico: 2-5 minuti per 20-50 modelli</li>
    <li>Memoria: Ottimizzato per gestire serie lunghe (>10k punti)</li>
    <li>Parallelizzazione: Utilizza tutti i core disponibili</li>
    <li>Early stopping: Interrompe se trova configurazione ottimale</li>
    </ul>
    """
    try:
        model_manager, forecast_service = services

        # Genera un ID univoco per il modello
        model_id = str(uuid.uuid4())

        # Converte i dati in pandas Series
        timestamps = pd.to_datetime(request.data.timestamps)
        series = pd.Series(request.data.values, index=timestamps)

        # Valida la serie temporale
        if series.empty or len(series) < 10:
            raise HTTPException(
                status_code=400, detail="Prophet auto-selection richiede almeno 10 osservazioni"
            )

        # Registra il modello come "auto-selecting"
        model_manager.register_model(
            model_id,
            "prophet-auto",
            "auto_selecting",
            {
                "created_at": datetime.now().isoformat(),
                "search_parameters": {
                    "growth_types": request.growth_types,
                    "seasonality_modes": request.seasonality_modes,
                    "country_holidays": request.country_holidays,
                    "max_models": request.max_models,
                    "cv_horizon": request.cv_horizon,
                },
            },
        )

        # Avvia la selezione automatica in background
        background_tasks.add_task(
            _auto_select_prophet_background, model_manager, model_id, series, request
        )

        logger.info(f"Started Prophet auto-selection with ID: {model_id}")

        estimated_time = max(60, request.max_models * 2)  # Stima 2 secondi per modello

        return {
            "model_id": model_id,
            "status": "auto_selecting",
            "message": f"Selezione automatica Prophet avviata (testando fino a {request.max_models} modelli)",
            "estimated_time_seconds": estimated_time,
            "endpoint_check": f"/models/{model_id}/status",
            "search_space": {
                "total_combinations": len(request.growth_types)
                * len(request.seasonality_modes)
                * len(request.country_holidays),
                "max_models_tested": request.max_models,
            },
        }

    except Exception as e:
        logger.error(f"Prophet auto-selection request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _auto_select_prophet_background(
    model_manager: ModelManager,
    model_id: str,
    series: pd.Series,
    request: ProphetAutoSelectionRequest,
):
    """
    Funzione helper per selezione automatica Prophet in background.
    """
    try:
        from arima_forecaster.core import ProphetModelSelector

        # Crea il selector con i parametri di ricerca
        selector = ProphetModelSelector(
            growth_modes=request.growth_types,  # Fix: growth_modes nel costruttore
            seasonality_modes=request.seasonality_modes,
            max_models=request.max_models,
            cv_horizon=request.cv_horizon,
        )

        # Esegue la ricerca
        import time

        start_time = time.time()

        # Passa country_holidays alla ricerca, non al costruttore
        # TODO: Implementare supporto per lista di country_holidays nel search()
        country_holiday = None
        if request.country_holidays and len(request.country_holidays) > 0:
            country_holiday = request.country_holidays[0]  # Per ora usa solo il primo

        selector.search(series, country_holidays=country_holiday)
        best_model = selector.get_best_model()

        search_time = time.time() - start_time

        if best_model is None:
            raise Exception("Nessun modello Prophet valido trovato durante la ricerca")

        # Salva il modello migliore
        model_manager.save_model(best_model, model_id)

        # Prepara i risultati
        best_params = selector.get_best_params()
        results_summary = selector.get_results_summary(10)

        # Aggiorna lo stato
        model_manager.update_model_status(
            model_id,
            "completed",
            {
                "best_model": best_params,
                "search_results": results_summary.to_dict("records")
                if not results_summary.empty
                else [],
                "search_time_seconds": search_time,
                "total_models_tested": selector.models_tested,
                "completed_at": datetime.now().isoformat(),
            },
        )

        logger.info(f"Prophet auto-selection {model_id} completed in {search_time:.2f}s")

    except Exception as e:
        model_manager.update_model_status(model_id, "failed", {"error": str(e)})
        logger.error(f"Prophet auto-selection {model_id} failed: {e}")


@router.get("/train/prophet/models", response_model=Dict[str, Any])
async def list_prophet_models(services=Depends(get_services)):
    """
    Lista tutti i modelli Prophet disponibili nel sistema.

    Restituisce informazioni dettagliate su tutti i modelli Prophet addestrati,
    inclusi parametri di configurazione, metriche di performance e stato corrente.

    <h3>Struttura Risposta:</h3>
    <table>
    <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th></tr>
    <tr><td>models</td><td>List[Dict]</td><td>Array di modelli Prophet con dettagli completi</td></tr>
    <tr><td>total_count</td><td>int</td><td>Numero totale di modelli Prophet</td></tr>
    <tr><td>by_status</td><td>Dict[str, int]</td><td>Conteggio modelli per stato (completed, training, failed)</td></tr>
    <tr><td>model_types</td><td>Dict[str, int]</td><td>Conteggio per tipo (prophet, prophet-auto)</td></tr>
    </table>

    <h3>Informazioni per Modello:</h3>
    <ul>
    <li><b>model_id:</b> Identificatore univoco del modello</li>
    <li><b>model_type:</b> Tipo specifico (prophet o prophet-auto)</li>
    <li><b>status:</b> Stato corrente (training, completed, failed)</li>
    <li><b>parameters:</b> Configurazione Prophet utilizzata</li>
    <li><b>metrics:</b> Performance del modello (MAPE, MAE, RMSE)</li>
    <li><b>created_at:</b> Timestamp creazione modello</li>
    <li><b>completed_at:</b> Timestamp completamento training</li>
    </ul>

    <h3>Stati del Modello:</h3>
    <ul>
    <li><b>training:</b> Modello in fase di addestramento</li>
    <li><b>auto_selecting:</b> Ricerca parametri ottimali in corso</li>
    <li><b>completed:</b> Training completato con successo</li>
    <li><b>failed:</b> Training fallito con errore</li>
    </ul>
    """
    try:
        model_manager, _ = services

        # Ottiene tutti i modelli Prophet
        all_models = model_manager.list_models()
        prophet_models = [
            model for model in all_models if model.get("model_type", "").startswith("prophet")
        ]

        # Calcola statistiche
        status_counts = {}
        for model in prophet_models:
            status = model.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "models": prophet_models,
            "total_count": len(prophet_models),
            "by_status": status_counts,
            "model_types": {
                "prophet": len([m for m in prophet_models if m.get("model_type") == "prophet"]),
                "prophet-auto": len(
                    [m for m in prophet_models if m.get("model_type") == "prophet-auto"]
                ),
            },
        }

    except Exception as e:
        logger.error(f"List Prophet models failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
