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

• POST /models/{model_id}/forecast - Genera previsioni future

Funzionalità:
- Previsioni con intervalli di confidenza personalizzabili
- Supporto variabili esogene per SARIMAX
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


@router.post("/{model_id}/forecast")
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
    <table class="table table-striped">
        <tr><th>Nome</th><th>Tipo</th><th>Descrizione</th><th>Richiesto</th></tr>
        <tr><td>model_id</td><td>str</td><td>ID univoco del modello da utilizzare</td><td>Sì</td></tr>
        <tr><td>request</td><td>ForecastRequest</td><td>Parametri per la generazione delle previsioni</td><td>Sì</td></tr>
    </table>
    
    <h4>Campi del Request Body:</h4>
    <table class="table table-striped">
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
    <table class="table table-striped">
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