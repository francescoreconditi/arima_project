"""
Router per endpoint di forecasting.

Gestisce la generazione di previsioni dai modelli addestrati.
"""

from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, Depends
import pandas as pd
import numpy as np

from arima_forecaster.api.models import ForecastRequest
from arima_forecaster.api.models_extra import ForecastResponse
from arima_forecaster.api.services import ModelManager, ForecastService
from arima_forecaster.api.examples import FORECAST_REQUEST_EXAMPLES, FORECAST_RESPONSE_EXAMPLES
from arima_forecaster.utils.logger import get_logger

logger = get_logger(__name__)

# Crea router con prefix e tags
router = APIRouter(
    prefix="/models",
    tags=["Forecasting"],
    responses={404: {"description": "Not found"}}
)

"""
📈 FORECASTING ROUTER

Gestisce la generazione di previsioni da modelli addestrati:

• POST /models/{model_id}/forecast        - Genera previsioni future (universale)
• POST /models/{model_id}/forecast/prophet - Previsioni Prophet con decomposizione

Funzionalità:
- Previsioni con intervalli di confidenza personalizzabili
- Supporto variabili esogene per SARIMAX
- Prophet: Decomposizione trend/seasonality/holidays
- Generazione timestamp automatica
- Validazione modello esistente
- Gestione errori predizione con fallback
- Output JSON ottimizzato per visualizzazioni
"""


# Dependency injection dei servizi
def get_services():
    """Dependency per ottenere i servizi necessari."""
    from pathlib import Path
    storage_path = Path("models")
    model_manager = ModelManager(storage_path)
    forecast_service = ForecastService(model_manager)
    return model_manager, forecast_service


@router.post(
    "/{model_id}/forecast",
    responses={
        200: {
            "description": "Previsioni generate con successo",
            "content": {
                "application/json": {
                    "examples": FORECAST_RESPONSE_EXAMPLES
                }
            }
        },
        404: {
            "description": "Modello non trovato",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Modello con ID 'model_123' non trovato"
                    }
                }
            }
        }
    }
)
async def generate_forecast(
    model_id: str,
    request: ForecastRequest,
    services: tuple = Depends(get_services)
) -> ForecastResponse:
    """
    Genera previsioni utilizzando un modello addestrato.
    
    Questo endpoint carica un modello salvato e genera previsioni future
    con intervalli di confidenza opzionali.
    
    <h4>Parametri di Ingresso:</h4>
    <table >
        <tr><th>Nome</th><th>Tipo</th><th>Descrizione</th><th>Richiesto</th></tr>
        <tr><td>model_id</td><td>str</td><td>ID univoco del modello da utilizzare</td><td>Sì</td></tr>
        <tr><td>request</td><td>ForecastRequest</td><td>Parametri per la generazione delle previsioni</td><td>Sì</td></tr>
    </table>
    
    <h4>Campi del Request Body:</h4>
    <table >
        <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th><th>Default</th></tr>
        <tr><td>steps</td><td>int</td><td>Numero di passi futuri da prevedere</td><td>Richiesto</td></tr>
        <tr><td>confidence_level</td><td>float</td><td>Livello di confidenza (es. 0.95 per 95%)</td><td>0.95</td></tr>
        <tr><td>return_confidence_intervals</td><td>bool</td><td>Se restituire gli intervalli di confidenza</td><td>true</td></tr>
        <tr><td>exogenous_future</td><td>ExogenousData</td><td>Variabili esogene future (per SARIMAX)</td><td>null</td></tr>
    </table>
    
    <h4>Esempio di Chiamata:</h4>
    <pre><code>
    curl -X POST "http://localhost:8000/models/{model_id}/forecast" \\
         -H "Content-Type: application/json" \\
         -d '{
           "steps": 30,
           "confidence_level": 0.95,
           "return_confidence_intervals": true
         }'
    </code></pre>
    
    <h4>Risposta:</h4>
    <table >
        <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th></tr>
        <tr><td>forecast</td><td>list[float]</td><td>Valori previsti per ogni passo futuro</td></tr>
        <tr><td>timestamps</td><td>list[str]</td><td>Date/timestamp per ogni previsione</td></tr>
        <tr><td>confidence_intervals</td><td>object</td><td>Intervalli di confidenza (lower/upper)</td></tr>
        <tr><td>model_id</td><td>str</td><td>ID del modello utilizzato</td></tr>
        <tr><td>forecast_steps</td><td>int</td><td>Numero di passi previsti</td></tr>
    </table>
    
    <h4>Esempio di Risposta:</h4>
    <pre><code>
    {
        "forecast": [105.2, 107.8, 103.5, ...],
        "timestamps": ["2024-09-01", "2024-09-02", "2024-09-03", ...],
        "confidence_intervals": {
            "lower": [100.1, 102.3, 98.2, ...],
            "upper": [110.3, 113.3, 108.8, ...]
        },
        "model_id": "abc123e4-5678-9012-3456-789012345678",
        "forecast_steps": 30
    }
    </code></pre>
    
    <h4>Errori Possibili:</h4>
    <ul>
        <li><strong>404</strong>: Modello non trovato</li>
        <li><strong>400</strong>: Parametri di forecast non validi</li>
        <li><strong>500</strong>: Errore durante la generazione delle previsioni</li>
    </ul>
    """
    model_manager, forecast_service = services
    
    try:
        # Verifica che il modello esista
        if not model_manager.model_exists(model_id):
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Carica il modello
        model = model_manager.load_model(model_id)
        
        # Genera le previsioni
        if request.return_confidence_intervals:
            # Genera previsioni con intervalli di confidenza
            forecast_values = model.forecast(
                steps=request.steps,
                confidence_intervals=True,
                confidence_level=request.confidence_level
            )
            
            # Estrai valori e intervalli
            if isinstance(forecast_values, tuple):
                predictions = forecast_values[0]
                lower_bound = forecast_values[1][:, 0]
                upper_bound = forecast_values[1][:, 1]
            else:
                predictions = forecast_values
                # Stima approssimativa degli intervalli se non disponibili
                std_error = np.std(predictions) * 0.1
                z_score = 1.96 if request.confidence_level == 0.95 else 2.576
                lower_bound = predictions - z_score * std_error
                upper_bound = predictions + z_score * std_error
            
            confidence_intervals = {
                "lower": lower_bound.tolist() if hasattr(lower_bound, 'tolist') else list(lower_bound),
                "upper": upper_bound.tolist() if hasattr(upper_bound, 'tolist') else list(upper_bound)
            }
        else:
            # Genera solo le previsioni puntuali
            predictions = model.forecast(steps=request.steps)
            confidence_intervals = None
        
        # Genera timestamps futuri
        # Assume frequenza giornaliera se non specificata
        last_date = pd.Timestamp.now()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=request.steps,
            freq='D'
        )
        timestamps = [date.strftime('%Y-%m-%d') for date in future_dates]
        
        # Converte predictions in lista se necessario
        if hasattr(predictions, 'tolist'):
            forecast_list = predictions.tolist()
        elif isinstance(predictions, pd.Series):
            forecast_list = predictions.values.tolist()
        else:
            forecast_list = list(predictions)
        
        return ForecastResponse(
            forecast=forecast_list,
            timestamps=timestamps,
            confidence_intervals=confidence_intervals,
            model_id=model_id,
            forecast_steps=request.steps
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Forecast generation failed for model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{model_id}/forecast/prophet", response_model=Dict[str, Any])
async def prophet_forecast_with_components(
    model_id: str,
    request: ForecastRequest,
    services: tuple = Depends(get_services)
):
    """
    Genera previsioni Prophet con decomposizione completa delle componenti.
    
    Questo endpoint è specifico per modelli Prophet e fornisce decomposizione dettagliata
    di trend, stagionalità e holiday effects che compongono la previsione finale.
    
    <h3>Parametri Request Body:</h3>
    <table>
    <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th></tr>
    <tr><td>steps</td><td>int</td><td>Numero di periodi da prevedere</td></tr>
    <tr><td>confidence_level</td><td>float</td><td>Livello di confidenza (default: 0.95)</td></tr>
    <tr><td>return_intervals</td><td>bool</td><td>Se restituire intervalli di confidenza (default: true)</td></tr>
    <tr><td>frequency</td><td>str</td><td>Frequenza temporale: "D", "W", "M" (opzionale)</td></tr>
    </table>
    
    <h3>Componenti Prophet nella Risposta:</h3>
    <table>
    <tr><th>Componente</th><th>Tipo</th><th>Descrizione</th></tr>
    <tr><td>forecast</td><td>List[float]</td><td>Previsioni finali complete</td></tr>
    <tr><td>trend</td><td>List[float]</td><td>Componente di trend puro</td></tr>
    <tr><td>seasonal</td><td>List[float]</td><td>Effetti stagionali combinati</td></tr>
    <tr><td>weekly</td><td>List[float]</td><td>Stagionalità settimanale (se presente)</td></tr>
    <tr><td>yearly</td><td>List[float]</td><td>Stagionalità annuale (se presente)</td></tr>
    <tr><td>holidays</td><td>List[float]</td><td>Effetti festività (se configurate)</td></tr>
    <tr><td>changepoints</td><td>Dict</td><td>Date e intensità dei cambio trend</td></tr>
    </table>
    
    <h3>Funzionalità Avanzate:</h3>
    <ul>
    <li><b>Trend Decomposition:</b> Componente di trend lineare/logistico isolato</li>
    <li><b>Seasonality Analysis:</b> Stagionalità weekly/yearly separate</li>
    <li><b>Holiday Effects:</b> Impatto quantificato delle festività</li>
    <li><b>Changepoints:</b> Punti di cambio trend identificati automaticamente</li>
    <li><b>Uncertainty Intervals:</b> Intervalli di confidenza per ogni componente</li>
    </ul>
    
    <h3>Requisiti:</h3>
    <ul>
    <li>Il modello deve essere di tipo "prophet" o "prophet-auto"</li>
    <li>Il modello deve avere status: completed</li>
    <li>Per holiday effects: richiede country_holidays configurato nel training</li>
    </ul>
    """
    model_manager, forecast_service = services
    
    try:
        # Verifica esistenza modello
        if not model_manager.model_exists(model_id):
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Carica il modello
        model = model_manager.load_model(model_id)
        model_info = model_manager.get_model_info(model_id)
        
        # Verifica che sia un modello Prophet
        model_type = model_info.get("model_type", "")
        if not model_type.startswith("prophet"):
            raise HTTPException(
                status_code=400, 
                detail=f"This endpoint is only for Prophet models. Model type: {model_type}"
            )
        
        # Verifica stato completato
        if model_info.get("status") != "completed":
            status = model_info.get("status", "unknown")
            raise HTTPException(
                status_code=400,
                detail=f"Model must be completed for forecasting. Current status: {status}"
            )
        
        # Genera previsioni standard
        if request.return_intervals:
            forecast_result = model.forecast(
                steps=request.steps,
                confidence_intervals=True,
                confidence_level=request.confidence_level
            )
            
            if isinstance(forecast_result, tuple):
                predictions, confidence_dict = forecast_result
                confidence_intervals = {
                    "lower": confidence_dict["lower"].tolist(),
                    "upper": confidence_dict["upper"].tolist()
                }
            else:
                predictions = forecast_result
                confidence_intervals = None
        else:
            predictions = model.forecast(steps=request.steps)
            confidence_intervals = None
        
        # Ottieni decomposizione componenti Prophet
        prophet_model = model.model  # Accesso al modello Prophet sottostante
        
        # Crea future dataframe per Prophet
        future = prophet_model.make_future_dataframe(periods=request.steps)
        forecast_df = prophet_model.predict(future)
        
        # Estrai gli ultimi steps (le previsioni future)
        future_forecast = forecast_df.tail(request.steps)
        
        # Componenti Prophet
        prophet_components = {
            "trend": future_forecast["trend"].tolist(),
            "seasonal": future_forecast.get("seasonal", future_forecast.get("additive_terms", [])).tolist()
        }
        
        # Aggiungi componenti specifiche se presenti
        if "weekly" in future_forecast.columns:
            prophet_components["weekly"] = future_forecast["weekly"].tolist()
        if "yearly" in future_forecast.columns:
            prophet_components["yearly"] = future_forecast["yearly"].tolist()
        if "holidays" in future_forecast.columns:
            prophet_components["holidays"] = future_forecast["holidays"].tolist()
        
        # Informazioni sui changepoints
        changepoints_info = {}
        if hasattr(prophet_model, 'changepoints') and len(prophet_model.changepoints) > 0:
            changepoints_info = {
                "dates": [cp.strftime('%Y-%m-%d') for cp in prophet_model.changepoints],
                "trend_changes": prophet_model.params['delta'].tolist() if hasattr(prophet_model, 'params') else []
            }
        
        # Genera timestamps futuri basati sull'ultimo timestamp del training
        last_training_date = future_forecast.index[-1] if hasattr(future_forecast, 'index') else pd.Timestamp.now()
        future_dates = pd.date_range(
            start=last_training_date + pd.Timedelta(days=1),
            periods=request.steps,
            freq='D'
        )
        timestamps = [date.strftime('%Y-%m-%d') for date in future_dates]
        
        # Risposta completa
        return {
            "forecast": predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions),
            "timestamps": timestamps,
            "confidence_intervals": confidence_intervals,
            "prophet_components": prophet_components,
            "changepoints": changepoints_info,
            "model_id": model_id,
            "forecast_steps": request.steps,
            "model_type": model_type,
            "decomposition_info": {
                "trend_type": model_info.get("parameters", {}).get("growth", "linear"),
                "seasonality_mode": model_info.get("parameters", {}).get("seasonality_mode", "additive"),
                "holidays_included": bool(model_info.get("parameters", {}).get("country_holidays"))
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prophet forecast with components failed for model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Prophet forecast failed: {str(e)}")