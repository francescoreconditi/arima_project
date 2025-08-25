"""
Router per gestione modelli.

Gestisce listing, recupero informazioni e cancellazione modelli.
"""

from typing import List, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends

from arima_forecaster.api.models import ModelInfo, ModelListResponse
from arima_forecaster.api.services import ModelManager, ForecastService
from arima_forecaster.utils.logger import get_logger

logger = get_logger(__name__)

# Crea router con prefix e tags  
router = APIRouter(
    prefix="/models",
    tags=["Models"],
    responses={404: {"description": "Not found"}}
)

"""
📁 MODELS ROUTER

Gestisce il ciclo di vita dei modelli salvati:

• GET /models           - Elenca tutti i modelli disponibili
• GET /models/{id}      - Recupera informazioni modello specifico
• DELETE /models/{id}   - Elimina modello salvato

Caratteristiche:
- CRUD completo per gestione modelli
- Metadati dettagliati (parametri, metriche, stato)
- Filtraggio e ordinamento risultati
- Validazione esistenza modello
- Operazioni batch per gestione multipla
"""


# Dependency injection dei servizi
def get_services():
    """Dependency per ottenere i servizi necessari."""
    from pathlib import Path
    storage_path = Path("models")
    model_manager = ModelManager(storage_path)
    forecast_service = ForecastService(model_manager)
    return model_manager, forecast_service


@router.get("", response_model=ModelListResponse)
async def list_models(
    services: tuple = Depends(get_services)
):
    """
    Elenca tutti i modelli addestrati disponibili.
    
    Restituisce un elenco di tutti i modelli salvati con le loro informazioni di base,
    utile per dashboard e interfacce di gestione.
    
    <h4>Risposta:</h4>
    <table class="table table-striped">
        <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th></tr>
        <tr><td>models</td><td>list[ModelInfo]</td><td>Lista dei modelli disponibili</td></tr>
        <tr><td>total_count</td><td>int</td><td>Numero totale di modelli</td></tr>
    </table>
    
    <h4>Campi di ModelInfo:</h4>
    <table class="table table-striped">
        <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th></tr>
        <tr><td>model_id</td><td>str</td><td>ID univoco del modello</td></tr>
        <tr><td>model_type</td><td>str</td><td>Tipo di modello (arima, sarima, var)</td></tr>
        <tr><td>status</td><td>str</td><td>Stato del modello (completed, training, failed)</td></tr>
        <tr><td>created_at</td><td>datetime</td><td>Data e ora di creazione</td></tr>
        <tr><td>training_observations</td><td>int</td><td>Numero di osservazioni utilizzate per l'addestramento</td></tr>
        <tr><td>parameters</td><td>dict</td><td>Parametri del modello</td></tr>
        <tr><td>metrics</td><td>dict</td><td>Metriche di performance</td></tr>
    </table>
    
    <h4>Esempio di Chiamata:</h4>
    <pre><code>
    curl -X GET "http://localhost:8000/models"
    </code></pre>
    
    <h4>Esempio di Risposta:</h4>
    <pre><code>
    {
        "models": [
            {
                "model_id": "abc123",
                "model_type": "sarima",
                "status": "completed",
                "created_at": "2024-08-23T10:00:00",
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
    </code></pre>
    """
    model_manager, _ = services
    
    try:
        models = model_manager.list_models()
        
        model_infos = []
        for model_data in models:
            model_infos.append(ModelInfo(
                model_id=model_data["model_id"],
                model_type=model_data.get("model_type", "unknown"),
                status=model_data.get("status", "completed"),
                created_at=model_data.get("created_at", datetime.now()),
                training_observations=model_data.get("training_observations", 0),
                parameters=model_data.get("parameters", {}),
                metrics=model_data.get("metrics", {})
            ))
        
        return ModelListResponse(
            models=model_infos,
            total_count=len(model_infos)
        )
        
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{model_id}", response_model=ModelInfo)
async def get_model_info(
    model_id: str,
    services: tuple = Depends(get_services)
):
    """
    Recupera le informazioni dettagliate di un modello specifico.
    
    <h4>Parametri di Ingresso:</h4>
    <table class="table table-striped">
        <tr><th>Nome</th><th>Tipo</th><th>Descrizione</th></tr>
        <tr><td>model_id</td><td>str</td><td>ID univoco del modello</td></tr>
    </table>
    
    <h4>Esempio di Chiamata:</h4>
    <pre><code>
    curl -X GET "http://localhost:8000/models/abc123e4-5678-9012-3456-789012345678"
    </code></pre>
    
    <h4>Esempio di Risposta:</h4>
    <pre><code>
    {
        "model_id": "abc123e4-5678-9012-3456-789012345678",
        "model_type": "sarima",
        "status": "completed",
        "created_at": "2024-08-23T10:00:00",
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
    </code></pre>
    
    <h4>Errori Possibili:</h4>
    <ul>
        <li><strong>404</strong>: Modello non trovato</li>
        <li><strong>500</strong>: Errore interno nel caricamento dei metadati</li>
    </ul>
    """
    model_manager, _ = services
    
    try:
        # Verifica l'esistenza del modello
        if not model_manager.model_exists(model_id):
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Carica i metadati
        metadata = model_manager.get_model_metadata(model_id)
        
        return ModelInfo(
            model_id=model_id,
            model_type=metadata.get("model_type", "unknown"),
            status=metadata.get("status", "completed"),
            created_at=metadata.get("created_at", datetime.now()),
            training_observations=metadata.get("training_observations", 0),
            parameters=metadata.get("parameters", {}),
            metrics=metadata.get("metrics", {})
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model info for {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{model_id}")
async def delete_model(
    model_id: str,
    services: tuple = Depends(get_services)
):
    """
    Elimina un modello salvato.
    
    Rimuove permanentemente il modello e tutti i suoi metadati dal sistema.
    Questa operazione è irreversibile.
    
    <h4>Parametri di Ingresso:</h4>
    <table class="table table-striped">
        <tr><th>Nome</th><th>Tipo</th><th>Descrizione</th></tr>
        <tr><td>model_id</td><td>str</td><td>ID univoco del modello da eliminare</td></tr>
    </table>
    
    <h4>Esempio di Chiamata:</h4>
    <pre><code>
    curl -X DELETE "http://localhost:8000/models/abc123e4-5678-9012-3456-789012345678"
    </code></pre>
    
    <h4>Esempio di Risposta:</h4>
    <pre><code>
    {
        "message": "Model abc123e4-5678-9012-3456-789012345678 deleted successfully"
    }
    </code></pre>
    
    <h4>Errori Possibili:</h4>
    <ul>
        <li><strong>404</strong>: Modello non trovato</li>
        <li><strong>500</strong>: Errore durante l'eliminazione dei file</li>
    </ul>
    """
    model_manager, _ = services
    
    try:
        # Verifica che il modello esista prima della cancellazione
        if not model_manager.model_exists(model_id):
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Elimina il modello
        model_manager.delete_model(model_id)
        
        return {"message": f"Model {model_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))