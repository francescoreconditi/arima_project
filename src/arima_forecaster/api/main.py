"""
Applicazione FastAPI per i servizi di forecasting ARIMA/SARIMA/VAR.

Questo modulo fornisce un'API REST completa per:
- Addestramento di modelli ARIMA, SARIMA, SARIMAX e VAR
- Generazione di previsioni e intervalli di confidenza  
- Selezione automatica dei parametri ottimali
- Diagnostica avanzata dei modelli
- Generazione di report completi
- Gestione del ciclo di vita dei modelli
"""

import uuid
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

# Importa Scalar per documentazione API migliorata
from scalar_fastapi import get_scalar_api_reference

from arima_forecaster.api.models import *
from arima_forecaster.api.services import ModelManager, ForecastService
from arima_forecaster.utils.logger import get_logger


def create_app(
    model_storage_path: Optional[str] = None, 
    enable_scalar: bool = True,
    production_mode: bool = False
) -> FastAPI:
    """
    Crea l'istanza dell'applicazione FastAPI con tutti gli endpoint configurati.
    
    <h4>Parametri di Ingresso:</h4>
    <table class="table table-striped">
        <tr><th>Nome</th><th>Tipo</th><th>Descrizione</th><th>Default</th></tr>
        <tr><td>model_storage_path</td><td>str | None</td><td>Percorso per salvare i modelli addestrati</td><td>None (usa "models")</td></tr>
    </table>
    
    <h4>Valore di Ritorno:</h4>
    <table class="table table-striped">
        <tr><th>Tipo</th><th>Descrizione</th></tr>
        <tr><td>FastAPI</td><td>Istanza dell'applicazione FastAPI configurata con middleware e endpoint</td></tr>
    </table>
    
    <h4>Esempio di Utilizzo:</h4>
    <pre><code>
    # Creare app con percorso personalizzato
    app = create_app(model_storage_path="/custom/models/path")
    
    # Creare app con percorso di default
    app = create_app()
    
    # Avviare il server
    uvicorn.run(app, host="0.0.0.0", port=8000)
    </code></pre>
    
    <h4>Caratteristiche dell'API:</h4>
    - **CORS abilitato**: Permette richieste cross-origin
    - **Documentazione automatica**: Swagger UI su `/docs`
    - **Background tasks**: Per operazioni di lunga durata
    - **Error handling**: Gestione centralizzata degli errori
    - **Logging**: Sistema di log integrato
    """
    
    # Crea l'istanza FastAPI con metadati completi
    app = FastAPI(
        title="ARIMA Forecaster API",
        description="API REST per forecasting di serie temporali con modelli ARIMA, SARIMA, SARIMAX e VAR",
        version="1.0.0",
        docs_url="/docs",      # Documentazione Swagger
        redoc_url="/redoc"     # Documentazione ReDoc alternativa
    )
    
    # Configura il middleware CORS per permettere richieste cross-origin
    # Essenziale per applicazioni web frontend che consumano l'API
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],        # Consente tutte le origini (personalizzare in produzione)
        allow_credentials=True,     # Permette l'invio di cookies/auth headers
        allow_methods=["*"],        # Consente tutti i metodi HTTP
        allow_headers=["*"],        # Permette tutti gli headers
    )
    
    # Inizializza i servizi core dell'applicazione
    storage_path = Path(model_storage_path or "models")  # Percorso per salvare i modelli
    model_manager = ModelManager(storage_path)           # Gestore dei modelli
    forecast_service = ForecastService(model_manager)    # Servizio per le previsioni
    logger = get_logger(__name__)                        # Logger per il debugging
    
    # Configura Scalar UI per documentazione API moderna e interattiva
    if enable_scalar:
        @app.get("/scalar", include_in_schema=False)
        async def scalar_html():
            """Endpoint per servire la documentazione Scalar UI."""
            return get_scalar_api_reference(
                openapi_url=app.openapi_url,
                title="🚀 ARIMA Forecaster API - Documentazione Interattiva"
            )
    
    @app.get("/")
    async def root():
        """
        Endpoint radice dell'API - fornisce informazioni base sui servizi disponibili.
        
        <h4>Risposta:</h4>
        <table class="table table-striped">
            <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th></tr>
            <tr><td>message</td><td>str</td><td>Nome dell'API</td></tr>
            <tr><td>version</td><td>str</td><td>Versione dell'API</td></tr>
            <tr><td>docs</td><td>str</td><td>Percorso alla documentazione Swagger</td></tr>
            <tr><td>health</td><td>str</td><td>Percorso all'endpoint di health check</td></tr>
        </table>
        
        <h4>Esempio di Chiamata:</h4>
        <pre><code>curl -X GET "http://localhost:8000/"</code></pre>
        
        <h4>Risposta di Esempio:</h4>
        <pre><code>{
            "message": "ARIMA Forecaster API",
            "version": "1.0.0", 
            "docs": "/docs",
            "redoc": "/redoc",
            "scalar": "/scalar",
            "health": "/health"
        }</code></pre>
        """
        response = {
            "message": "ARIMA Forecaster API",
            "version": "1.0.0",
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health"
        }
        if enable_scalar:
            response["scalar"] = "/scalar"
        return response
    
    @app.get("/health")
    async def health_check():
        """
        Endpoint per il controllo dello stato di salute dell'API.
        
        Utile per monitoring, load balancer e sistemi di orchestrazione 
        per verificare che l'API sia operativa e responsive.
        
        <h4>Risposta:</h4>
        <table class="table table-striped">
            <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th></tr>
            <tr><td>status</td><td>str</td><td>Stato dell'API ("healthy" se operativa)</td></tr>
            <tr><td>timestamp</td><td>datetime</td><td>Timestamp della risposta</td></tr>
            <tr><td>models_count</td><td>int</td><td>Numero di modelli attualmente salvati</td></tr>
        </table>
        
        <h4>Esempio di Chiamata:</h4>
        <pre><code>curl -X GET "http://localhost:8000/health"</code></pre>
        
        <h4>Risposta di Esempio:</h4>
        <pre><code>{
            "status": "healthy",
            "timestamp": "2024-08-23T22:30:00.123456",
            "models_count": 5
        }</code></pre>
        """
        return {
            "status": "healthy",
            "timestamp": datetime.now(),
            "models_count": len(model_manager.list_models())
        }
    
    @app.post("/models/train", response_model=ModelInfo)
    async def train_model(
        request: ModelTrainingRequest,
        background_tasks: BackgroundTasks
    ):
        """
        Addestra un nuovo modello ARIMA, SARIMA o SARIMAX.
        
        Questo endpoint avvia l'addestramento in background e restituisce immediatamente
        le informazioni iniziali del modello. Il training continua asincronamente.
        
        <h4>Parametri di Ingresso:</h4>
        <table class="table table-striped">
            <tr><th>Nome</th><th>Tipo</th><th>Descrizione</th><th>Richiesto</th></tr>
            <tr><td>request</td><td>ModelTrainingRequest</td><td>Configurazione per l'addestramento del modello</td><td>Sì</td></tr>
            <tr><td>background_tasks</td><td>BackgroundTasks</td><td>Gestore per task in background (automatico)</td><td>No</td></tr>
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
        
        <h4>Risposta:</h4>
        <table class="table table-striped">
            <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th></tr>
            <tr><td>model_id</td><td>str</td><td>ID univoco del modello (UUID)</td></tr>
            <tr><td>model_type</td><td>str</td><td>Tipo di modello addestrato</td></tr>
            <tr><td>status</td><td>str</td><td>Stato del modello ("training", "completed", "failed")</td></tr>
            <tr><td>created_at</td><td>datetime</td><td>Timestamp di creazione</td></tr>
            <tr><td>training_observations</td><td>int</td><td>Numero di osservazioni utilizzate</td></tr>
            <tr><td>parameters</td><td>dict</td><td>Parametri del modello (vuoto durante training)</td></tr>
            <tr><td>metrics</td><td>dict</td><td>Metriche di valutazione (vuoto durante training)</td></tr>
        </table>
        
        <h4>Esempio di Chiamata:</h4>
        <pre><code>
        curl -X POST "http://localhost:8000/models/train" \\
             -H "Content-Type: application/json" \\
             -d '{
               "model_type": "sarima",
               "data": {
                 "timestamps": ["2023-01-01", "2023-01-02", "2023-01-03"],
                 "values": [100, 110, 105]
               },
               "order": [1, 1, 1],
               "seasonal_order": [1, 1, 1, 12]
             }'
        </code></pre>
        
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
        • <strong>400</strong>: Dati non validi o parametri incorretti
        • <strong>500</strong>: Errore interno del server durante l'inizializzazione
        """
        try:
            # Converte i dati della richiesta in una pandas Series
            # Le date vengono parsate automaticamente da pandas
            timestamps = pd.to_datetime(request.data.timestamps)
            series = pd.Series(request.data.values, index=timestamps)
            
            # Genera un ID univoco per il modello usando UUID4
            # Questo garantisce identificazione unica anche in sistemi distribuiti
            model_id = str(uuid.uuid4())
            
            # Avvia l'addestramento del modello come task in background
            # Questo permette di restituire immediatamente una risposta al client
            # mentre il training continua asincronamente
            background_tasks.add_task(
                _train_model_background,
                model_manager,
                model_id,
                series,
                request
            )
            
            # Restituisce le informazioni iniziali del modello
            # Lo status "training" indica che l'addestramento è in corso
            return ModelInfo(
                model_id=model_id,
                model_type=request.model_type,
                status="training",
                created_at=datetime.now(),
                training_observations=len(series),
                parameters={},  # Popolato dopo l'addestramento
                metrics={}      # Popolato dopo l'addestramento
            )
            
        except Exception as e:
            # Log dell'errore per debugging e monitoraggio
            logger.error(f"Model training request failed: {e}")
            # Restituisce un errore HTTP 400 (Bad Request) al client
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.post("/models/train/var", response_model=ModelInfo)
    async def train_var_model(
        request: VARTrainingRequest,
        background_tasks: BackgroundTasks
    ):
        """
        Addestra un modello VAR (Vector Autoregression) per serie temporali multivariate.
        
        I modelli VAR sono ideali per analizzare le relazioni dinamiche tra multiple
        serie temporali correlate, catturando le interdipendenze bidirezionali.
        
        <h4>Parametri di Ingresso:</h4>
        <table class="table table-striped">
            <tr><th>Nome</th><th>Tipo</th><th>Descrizione</th><th>Richiesto</th></tr>
            <tr><td>request</td><td>VARTrainingRequest</td><td>Configurazione per l'addestramento del modello VAR</td><td>Sì</td></tr>
            <tr><td>background_tasks</td><td>BackgroundTasks</td><td>Gestore per task in background (automatico)</td><td>No</td></tr>
        </table>
        
        <h4>Campi del Request Body:</h4>
        <table class="table table-striped">
            <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th><th>Esempio</th></tr>
            <tr><td>data</td><td>MultiVariateTimeSeriesData</td><td>Dati delle serie temporali multivariate</td><td>{timestamps: [...], data: {...}}</td></tr>
            <tr><td>maxlags</td><td>int</td><td>Numero massimo di lag da considerare (None per auto)</td><td>5</td></tr>
            <tr><td>ic</td><td>str</td><td>Criterio informativo: "aic", "bic", "hqic", "fpe"</td><td>"aic"</td></tr>
        </table>
        
        <h4>Risposta:</h4>
        <table class="table table-striped">
            <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th></tr>
            <tr><td>model_id</td><td>str</td><td>ID univoco del modello VAR</td></tr>
            <tr><td>model_type</td><td>str</td><td>"var"</td></tr>
            <tr><td>status</td><td>str</td><td>Stato del modello</td></tr>
            <tr><td>created_at</td><td>datetime</td><td>Timestamp di creazione</td></tr>
            <tr><td>training_observations</td><td>int</td><td>Numero di osservazioni multivariate</td></tr>
            <tr><td>parameters</td><td>dict</td><td>Parametri del modello VAR</td></tr>
            <tr><td>metrics</td><td>dict</td><td>Metriche di valutazione</td></tr>
        </table>
        
        <h4>Esempio di Chiamata:</h4>
        <pre><code>
        curl -X POST "http://localhost:8000/models/train/var" \\
             -H "Content-Type: application/json" \\
             -d '{
               "data": {
                 "timestamps": ["2023-01-01", "2023-01-02"],
                 "data": {
                   "series1": [100, 110],
                   "series2": [50, 55],
                   "series3": [200, 210]
                 }
               },
               "maxlags": 5,
               "ic": "aic"
             }'
        </code></pre>
        
        <h4>Risposta di Esempio:</h4>
        <pre><code>
        {
            "model_id": "def456f7-8901-2345-6789-012345678901",
            "model_type": "var",
            "status": "training",
            "created_at": "2024-08-23T22:35:00.123456", 
            "training_observations": 100,
            "parameters": {"maxlags": 5, "ic": "aic"},
            "metrics": {}
        }
        </code></pre>
        """
        try:
            # Converte i dati della richiesta in un DataFrame pandas
            # Ogni colonna rappresenta una serie temporale diversa
            timestamps = pd.to_datetime(request.data.timestamps)
            data_dict = request.data.data
            
            # Crea DataFrame con index temporale e colonne per ogni serie
            df = pd.DataFrame(data_dict, index=timestamps)
            
            # Genera ID univoco per il modello VAR
            model_id = str(uuid.uuid4())
            
            # Avvia l'addestramento del modello VAR in background
            # I modelli VAR possono richiedere più tempo per convergere
            background_tasks.add_task(
                _train_var_model_background,
                model_manager,
                model_id,
                df,
                request
            )
            
            # Restituisce informazioni iniziali del modello VAR
            return ModelInfo(
                model_id=model_id,
                model_type="var",
                status="training",
                created_at=datetime.now(),
                training_observations=len(df),
                parameters={"maxlags": request.maxlags, "ic": request.ic},
                metrics={}
            )
            
        except Exception as e:
            logger.error(f"VAR model training request failed: {e}")
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.post("/models/auto-select", response_model=AutoSelectionResult)
    async def auto_select_model(request: AutoSelectionRequest):
        """
        Seleziona automaticamente i migliori parametri del modello tramite grid search.
        
        Questo endpoint esegue una ricerca esaustiva sui parametri del modello per trovare
        la combinazione ottimale basata sul criterio informativo specificato (AIC/BIC).
        
        <h4>Parametri di Ingresso:</h4>
        <table class="table table-striped">
            <tr><th>Nome</th><th>Tipo</th><th>Descrizione</th><th>Richiesto</th></tr>
            <tr><td>request</td><td>AutoSelectionRequest</td><td>Configurazione per la selezione automatica</td><td>Sì</td></tr>
        </table>
        
        <h4>Campi del Request Body:</h4>
        <table class="table table-striped">
            <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th><th>Esempio</th></tr>
            <tr><td>data</td><td>TimeSeriesData</td><td>Dati della serie temporale</td><td>{timestamps: [...], values: [...]}</td></tr>
            <tr><td>model_type</td><td>str</td><td>Tipo di modello da ottimizzare</td><td>"sarima"</td></tr>
            <tr><td>max_models</td><td>int</td><td>Numero massimo di modelli da testare</td><td>100</td></tr>
            <tr><td>information_criterion</td><td>str</td><td>Criterio di selezione: "aic" o "bic"</td><td>"aic"</td></tr>
            <tr><td>exogenous_data</td><td>ExogenousData</td><td>Variabili esogene (per SARIMAX)</td><td>null</td></tr>
        </table>
        
        <h4>Risposta:</h4>
        <table class="table table-striped">
            <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th></tr>
            <tr><td>best_model_id</td><td>str</td><td>ID del modello con i migliori parametri</td></tr>
            <tr><td>best_parameters</td><td>dict</td><td>Parametri ottimali trovati</td></tr>
            <tr><td>best_score</td><td>float</td><td>Miglior score del criterio informativo</td></tr>
            <tr><td>all_results</td><td>list</td><td>Risultati di tutti i modelli testati</td></tr>
            <tr><td>selection_time</td><td>float</td><td>Tempo impiegato per la selezione (secondi)</td></tr>
        </table>
        
        <h4>Esempio di Chiamata:</h4>
        <pre><code>
        curl -X POST "http://localhost:8000/models/auto-select" \\
             -H "Content-Type: application/json" \\
             -d '{
               "data": {
                 "timestamps": ["2023-01-01", "2023-01-02", "2023-01-03"],
                 "values": [100, 110, 105]
               },
               "model_type": "sarima",
               "max_models": 50,
               "information_criterion": "aic"
             }'
        </code></pre>
        
        <h4>Risposta di Esempio:</h4>
        <pre><code>
        {
            "best_model_id": "ghi789j0-1234-5678-9012-345678901234",
            "best_parameters": {
                "order": [1, 1, 1],
                "seasonal_order": [1, 1, 1, 12]
            },
            "best_score": 1875.42,
            "all_results": [...],
            "selection_time": 45.7
        }
        </code></pre>
        """
        try:
            start_time = time.time()  # Misura il tempo di esecuzione
            
            # Converte i dati della richiesta in pandas Series
            timestamps = pd.to_datetime(request.data.timestamps)
            series = pd.Series(request.data.values, index=timestamps)
            
            # Prepara i dati esogeni se forniti (per modelli SARIMAX)
            exog_df = None
            if request.exogenous_data:
                exog_df = pd.DataFrame(request.exogenous_data.variables, index=timestamps)
            
            # Esegue la selezione automatica dei parametri
            # Il servizio gestisce la logica di grid search e ottimizzazione
            result = await forecast_service.auto_select_model(
                series=series,
                model_type=request.model_type,
                max_models=request.max_models,
                information_criterion=request.information_criterion,
                exogenous_data=exog_df
            )
            
            # Calcola il tempo totale impiegato
            selection_time = time.time() - start_time
            
            # Restituisce i risultati della selezione automatica
            return AutoSelectionResult(
                best_model_id=result["best_model_id"],
                best_parameters=result["best_parameters"],
                best_score=result["best_score"],
                all_results=result["all_results"],
                selection_time=selection_time
            )
            
        except Exception as e:
            logger.error(f"Auto selection failed: {e}")
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.post("/models/{model_id}/forecast")
    async def forecast(
        model_id: str,
        request: ForecastRequest
    ):
        """
        Genera previsioni future utilizzando un modello addestrato.
        
        Questo endpoint supporta tutti i tipi di modello (ARIMA, SARIMA, SARIMAX, VAR)
        e può generare previsioni con intervalli di confidenza personalizzabili.
        
        <h4>Parametri di Ingresso:</h4>
        <table class="table table-striped">
            <tr><th>Nome</th><th>Tipo</th><th>Descrizione</th><th>Richiesto</th></tr>
            <tr><td>model_id</td><td>str</td><td>ID del modello addestrato (path parameter)</td><td>Sì</td></tr>
            <tr><td>request</td><td>ForecastRequest</td><td>Configurazione per la generazione delle previsioni</td><td>Sì</td></tr>
        </table>
        
        <h4>Campi del Request Body:</h4>
        <table class="table table-striped">
            <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th><th>Esempio</th></tr>
            <tr><td>steps</td><td>int</td><td>Numero di passi futuri da prevedere</td><td>12</td></tr>
            <tr><td>confidence_level</td><td>float</td><td>Livello di confidenza per gli intervalli (0-1)</td><td>0.95</td></tr>
            <tr><td>return_intervals</td><td>bool</td><td>Se includere gli intervalli di confidenza</td><td>true</td></tr>
            <tr><td>exogenous_future</td><td>ExogenousFutureData</td><td>Valori futuri delle variabili esogene</td><td>null</td></tr>
        </table>
        
        <h4>Risposta (varia per tipo di modello):</h4>
        <table class="table table-striped">
            <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th></tr>
            <tr><td>model_id</td><td>str</td><td>ID del modello utilizzato</td></tr>
            <tr><td>model_type</td><td>str</td><td>Tipo di modello</td></tr>
            <tr><td>forecast_values</td><td>list</td><td>Valori delle previsioni</td></tr>
            <tr><td>confidence_intervals</td><td>dict</td><td>Intervalli di confidenza (se richiesti)</td></tr>
            <tr><td>forecast_dates</td><td>list</td><td>Date corrispondenti alle previsioni</td></tr>
        </table>
        
        <h4>Esempio di Chiamata:</h4>
        <pre><code>
        curl -X POST "http://localhost:8000/models/abc123/forecast" \\
             -H "Content-Type: application/json" \\
             -d '{
               "steps": 12,
               "confidence_level": 0.95,
               "return_intervals": true
             }'
        </code></pre>
        
        <h4>Risposta di Esempio:</h4>
        <pre><code>
        {
            "model_id": "abc123e4-5678-9012-3456-789012345678",
            "model_type": "sarima",
            "forecast_values": [105.2, 107.1, 108.9, 110.4],
            "confidence_intervals": {
                "lower": [98.1, 99.8, 101.2, 102.7],
                "upper": [112.3, 114.4, 116.6, 118.1]
            },
            "forecast_dates": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
        }
        </code></pre>
        
        <h4>Note per Modelli SARIMAX:</h4>
        I modelli SARIMAX richiedono i valori futuri delle variabili esogene nel campo 
        `exogenous_future` per generare previsioni accurate.
        """
        try:
            # Verifica che il modello esista nel sistema
            if not model_manager.model_exists(model_id):
                raise HTTPException(status_code=404, detail="Model not found")
            
            # Ottiene le informazioni del modello per determinare il tipo
            model_info = model_manager.get_model_info(model_id)
            
            # Gestisce le previsioni per modelli VAR (logica diversa)
            if model_info["model_type"] == "var":
                # I modelli VAR gestiscono multiple serie temporali
                result = await forecast_service.generate_var_forecast(
                    model_id=model_id,
                    steps=request.steps,
                    confidence_level=request.confidence_level,
                    return_intervals=request.return_intervals
                )
                return result
            else:
                # Gestisce ARIMA, SARIMA, SARIMAX (serie temporali univariate)
                exog_future_df = None
                if request.exogenous_future:
                    # Crea DataFrame per le variabili esogene future
                    # Necessario per i modelli SARIMAX
                    exog_future_df = pd.DataFrame(request.exogenous_future.variables)
                
                # Genera le previsioni utilizzando il servizio appropriato
                result = await forecast_service.generate_forecast(
                    model_id=model_id,
                    steps=request.steps,
                    confidence_level=request.confidence_level,
                    return_intervals=request.return_intervals,
                    exogenous_future=exog_future_df
                )
                return result
                
        except Exception as e:
            logger.error(f"Forecast generation failed: {e}")
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.get("/models", response_model=ModelListResponse)
    async def list_models():
        """
        Elenca tutti i modelli addestrati disponibili nel sistema.
        
        Questo endpoint fornisce una vista d'insieme di tutti i modelli salvati,
        utile per gestire e monitorare il catalogo dei modelli.
        
        <h4>Risposta:</h4>
        <table class="table table-striped">
            <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th></tr>
            <tr><td>models</td><td>list[ModelInfo]</td><td>Lista di informazioni sui modelli</td></tr>
            <tr><td>total</td><td>int</td><td>Numero totale di modelli</td></tr>
        </table>
        
        <h4>Struttura ModelInfo:</h4>
        <table class="table table-striped">
            <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th></tr>
            <tr><td>model_id</td><td>str</td><td>ID univoco del modello</td></tr>
            <tr><td>model_type</td><td>str</td><td>Tipo: "arima", "sarima", "sarimax", "var"</td></tr>
            <tr><td>status</td><td>str</td><td>Stato: "training", "completed", "failed"</td></tr>
            <tr><td>created_at</td><td>datetime</td><td>Timestamp di creazione</td></tr>
            <tr><td>training_observations</td><td>int</td><td>Numero di osservazioni utilizzate</td></tr>
            <tr><td>parameters</td><td>dict</td><td>Parametri del modello</td></tr>
            <tr><td>metrics</td><td>dict</td><td>Metriche di performance</td></tr>
        </table>
        
        <h4>Esempio di Chiamata:</h4>
        <pre><code>
        curl -X GET "http://localhost:8000/models"
        </code></pre>
        
        <h4>Risposta di Esempio:</h4>
        <pre><code>
        {
            "models": [
                {
                    "model_id": "abc123e4-5678-9012-3456-789012345678",
                    "model_type": "sarima",
                    "status": "completed",
                    "created_at": "2024-08-23T22:30:00",
                    "training_observations": 365,
                    "parameters": {"order": [1,1,1], "seasonal_order": [1,1,1,12]},
                    "metrics": {"aic": 1875.42, "bic": 1891.33}
                }
            ],
            "total": 1
        }
        </code></pre>
        """
        try:
            # Ottiene la lista degli ID di tutti i modelli salvati
            models = model_manager.list_models()
            model_infos = []
            
            # Itera su ogni modello per ottenere le informazioni dettagliate
            for model_id in models:
                try:
                    # Carica i metadati del modello
                    info = model_manager.get_model_info(model_id)
                    
                    # Crea un oggetto ModelInfo strutturato
                    model_infos.append(ModelInfo(
                        model_id=model_id,
                        model_type=info.get("model_type", "unknown"),
                        status=info.get("status", "unknown"),
                        created_at=info.get("created_at", datetime.now()),
                        training_observations=info.get("n_observations", 0),
                        parameters=info.get("parameters", {}),
                        metrics=info.get("metrics", {})
                    ))
                except Exception as e:
                    # Log dell'errore ma continua con gli altri modelli
                    # Evita che un modello corrotto blocchi l'intera lista
                    logger.warning(f"Could not load info for model {model_id}: {e}")
                    continue
            
            # Restituisce la lista completa con il conteggio totale
            return ModelListResponse(
                models=model_infos,
                total=len(model_infos)
            )
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/models/{model_id}", response_model=ModelInfo)
    async def get_model_info(model_id: str):
        """
        Ottiene informazioni dettagliate su un modello specifico.
        
        Questo endpoint fornisce tutti i metadati, parametri e metriche
        associati a un modello particolare.
        
        <h4>Parametri di Ingresso:</h4>
        <table class="table table-striped">
            <tr><th>Nome</th><th>Tipo</th><th>Descrizione</th><th>Richiesto</th></tr>
            <tr><td>model_id</td><td>str</td><td>ID del modello (path parameter)</td><td>Sì</td></tr>
        </table>
        
        <h4>Risposta:</h4>
        <table class="table table-striped">
            <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th></tr>
            <tr><td>model_id</td><td>str</td><td>ID del modello richiesto</td></tr>
            <tr><td>model_type</td><td>str</td><td>Tipo di modello</td></tr>
            <tr><td>status</td><td>str</td><td>Stato corrente del modello</td></tr>
            <tr><td>created_at</td><td>datetime</td><td>Timestamp di creazione</td></tr>
            <tr><td>training_observations</td><td>int</td><td>Numero di osservazioni</td></tr>
            <tr><td>parameters</td><td>dict</td><td>Parametri completi del modello</td></tr>
            <tr><td>metrics</td><td>dict</td><td>Metriche di valutazione complete</td></tr>
        </table>
        
        <h4>Esempio di Chiamata:</h4>
        <pre><code>
        curl -X GET "http://localhost:8000/models/abc123e4-5678-9012-3456-789012345678"
        </code></pre>
        
        <h4>Risposta di Esempio:</h4>
        <pre><code>
        {
            "model_id": "abc123e4-5678-9012-3456-789012345678",
            "model_type": "sarima",
            "status": "completed",
            "created_at": "2024-08-23T22:30:00.123456",
            "training_observations": 365,
            "parameters": {
                "order": [1, 1, 1],
                "seasonal_order": [1, 1, 1, 12],
                "trend": "c"
            },
            "metrics": {
                "aic": 1875.42,
                "bic": 1891.33,
                "mae": 2.34,
                "rmse": 3.12,
                "mape": 5.67
            }
        }
        </code></pre>
        
        <h4>Errori Possibili:</h4>
        - **404**: Modello non trovato
        • <strong>500</strong>: Errore interno nel caricamento dei metadati
        """
        try:
            # Verifica l'esistenza del modello
            if not model_manager.model_exists(model_id):
                raise HTTPException(status_code=404, detail="Model not found")
            
            # Carica tutte le informazioni dettagliate del modello
            info = model_manager.get_model_info(model_id)
            
            # Costruisce la risposta strutturata
            return ModelInfo(
                model_id=model_id,
                model_type=info.get("model_type", "unknown"),
                status=info.get("status", "unknown"),
                created_at=info.get("created_at", datetime.now()),
                training_observations=info.get("n_observations", 0),
                parameters=info.get("parameters", {}),
                metrics=info.get("metrics", {})
            )
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.delete("/models/{model_id}")
    async def delete_model(model_id: str):
        """
        Elimina definitivamente un modello dal sistema.
        
        Questa operazione rimuove tutti i file associati al modello
        (file pickle, metadati, report) e non può essere annullata.
        
        <h4>Parametri di Ingresso:</h4>
        <table class="table table-striped">
            <tr><th>Nome</th><th>Tipo</th><th>Descrizione</th><th>Richiesto</th></tr>
            <tr><td>model_id</td><td>str</td><td>ID del modello da eliminare (path parameter)</td><td>Sì</td></tr>
        </table>
        
        <h4>Risposta:</h4>
        <table class="table table-striped">
            <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th></tr>
            <tr><td>message</td><td>str</td><td>Messaggio di conferma dell'eliminazione</td></tr>
        </table>
        
        <h4>Esempio di Chiamata:</h4>
        <pre><code>
        curl -X DELETE "http://localhost:8000/models/abc123e4-5678-9012-3456-789012345678"
        </code></pre>
        
        <h4>Risposta di Esempio:</h4>
        <pre><code>
        {
            "message": "Model abc123e4-5678-9012-3456-789012345678 deleted successfully"
        }
        </code></pre>
        
        <h4>Errori Possibili:</h4>
        - **404**: Modello non trovato
        • <strong>500</strong>: Errore durante l'eliminazione dei file
        """
        try:
            # Verifica che il modello esista prima della cancellazione
            if not model_manager.model_exists(model_id):
                raise HTTPException(status_code=404, detail="Model not found")
            
            # Elimina il modello e tutti i file associati
            # Il ModelManager gestisce la pulizia completa
            model_manager.delete_model(model_id)
            
            # Conferma l'avvenuta cancellazione
            return {"message": f"Model {model_id} deleted successfully"}
            
        except Exception as e:
            logger.error(f"Failed to delete model: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/models/{model_id}/diagnostics", response_model=ModelDiagnostics)
    async def get_model_diagnostics(
        model_id: str,
        request: ModelDiagnosticsRequest
    ):
        """
        Genera analisi diagnostiche avanzate per un modello addestrato.
        
        Le diagnostiche includono analisi dei residui, test statistici, 
        grafici ACF/PACF e test di normalità per valutare la qualità del modello.
        
        <h4>Parametri di Ingresso:</h4>
        <table class="table table-striped">
            <tr><th>Nome</th><th>Tipo</th><th>Descrizione</th><th>Richiesto</th></tr>
            <tr><td>model_id</td><td>str</td><td>ID del modello (path parameter)</td><td>Sì</td></tr>
            <tr><td>request</td><td>ModelDiagnosticsRequest</td><td>Configurazione per le diagnostiche</td><td>Sì</td></tr>
        </table>
        
        <h4>Campi del Request Body:</h4>
        <table class="table table-striped">
            <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th><th>Default</th></tr>
            <tr><td>include_residuals</td><td>bool</td><td>Include analisi dettagliata dei residui</td><td>true</td></tr>
            <tr><td>include_acf_pacf</td><td>bool</td><td>Include grafici ACF e PACF</td><td>true</td></tr>
        </table>
        
        <h4>Risposta:</h4>
        <table class="table table-striped">
            <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th></tr>
            <tr><td>model_id</td><td>str</td><td>ID del modello analizzato</td></tr>
            <tr><td>residuals_analysis</td><td>dict</td><td>Statistiche dei residui</td></tr>
            <tr><td>normality_tests</td><td>dict</td><td>Test di normalità (Jarque-Bera, etc.)</td></tr>
            <tr><td>autocorrelation_tests</td><td>dict</td><td>Test di autocorrelazione (Ljung-Box)</td></tr>
            <tr><td>acf_pacf_plots</td><td>dict</td><td>Dati per grafici ACF/PACF</td></tr>
        </table>
        
        <h4>Esempio di Chiamata:</h4>
        <pre><code>
        curl -X POST "http://localhost:8000/models/abc123/diagnostics" \\
             -H "Content-Type: application/json" \\
             -d '{
               "include_residuals": true,
               "include_acf_pacf": true
             }'
        </code></pre>
        
        <h4>Risposta di Esempio:</h4>
        <pre><code>
        {
            "model_id": "abc123e4-5678-9012-3456-789012345678",
            "residuals_analysis": {
                "mean": 0.0012,
                "std": 2.34,
                "skewness": -0.05,
                "kurtosis": 3.12
            },
            "normality_tests": {
                "jarque_bera": {"statistic": 1.23, "p_value": 0.54},
                "shapiro_wilk": {"statistic": 0.998, "p_value": 0.43}
            },
            "autocorrelation_tests": {
                "ljung_box": {"statistic": 12.4, "p_value": 0.67}
            },
            "acf_pacf_plots": {
                "acf_values": [...],
                "pacf_values": [...],
                "lags": [...]
            }
        }
        </code></pre>
        """
        try:
            # Verifica l'esistenza del modello
            if not model_manager.model_exists(model_id):
                raise HTTPException(status_code=404, detail="Model not found")
            
            # Genera le diagnostiche utilizzando il servizio appropriato
            # Il ForecastService coordina tutte le analisi statistiche
            diagnostics = await forecast_service.generate_diagnostics(
                model_id=model_id,
                include_residuals=request.include_residuals,
                include_acf_pacf=request.include_acf_pacf
            )
            
            return diagnostics
            
        except Exception as e:
            logger.error(f"Failed to generate diagnostics: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/models/{model_id}/report", response_model=ReportGenerationResponse)
    async def generate_model_report(
        model_id: str,
        request: ReportGenerationRequest,
        background_tasks: BackgroundTasks
    ):
        """
        Genera un report completo e professionale per un modello addestrato.
        
        Il report include analisi dettagliate, grafici interattivi, diagnostiche,
        previsioni e raccomandazioni in formato HTML, PDF o DOCX.
        
        <h4>Parametri di Ingresso:</h4>
        <table class="table table-striped">
            <tr><th>Nome</th><th>Tipo</th><th>Descrizione</th><th>Richiesto</th></tr>
            <tr><td>model_id</td><td>str</td><td>ID del modello (path parameter)</td><td>Sì</td></tr>
            <tr><td>request</td><td>ReportGenerationRequest</td><td>Configurazione per il report</td><td>Sì</td></tr>
            <tr><td>background_tasks</td><td>BackgroundTasks</td><td>Gestore task in background (automatico)</td><td>No</td></tr>
        </table>
        
        <h4>Campi del Request Body:</h4>
        <table class="table table-striped">
            <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th><th>Default</th></tr>
            <tr><td>report_title</td><td>str</td><td>Titolo personalizzato del report</td><td>"Model Analysis Report"</td></tr>
            <tr><td>output_filename</td><td>str</td><td>Nome file di output (senza estensione)</td><td>Auto-generato</td></tr>
            <tr><td>format_type</td><td>str</td><td>Formato: "html", "pdf", "docx"</td><td>"html"</td></tr>
            <tr><td>include_diagnostics</td><td>bool</td><td>Include sezione diagnostiche</td><td>true</td></tr>
            <tr><td>include_forecast</td><td>bool</td><td>Include sezione previsioni</td><td>true</td></tr>
            <tr><td>forecast_steps</td><td>int</td><td>Numero di passi di previsione</td><td>12</td></tr>
        </table>
        
        <h4>Risposta:</h4>
        <table class="table table-striped">
            <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th></tr>
            <tr><td>model_id</td><td>str</td><td>ID del modello utilizzato</td></tr>
            <tr><td>report_path</td><td>str</td><td>Percorso completo del file generato</td></tr>
            <tr><td>format_type</td><td>str</td><td>Formato del report generato</td></tr>
            <tr><td>generation_time</td><td>float</td><td>Tempo di generazione in secondi</td></tr>
            <tr><td>file_size_mb</td><td>float</td><td>Dimensione del file in MB</td></tr>
            <tr><td>download_url</td><td>str</td><td>URL per il download del report</td></tr>
        </table>
        
        <h4>Esempio di Chiamata:</h4>
        <pre><code>
        curl -X POST "http://localhost:8000/models/abc123/report" \\
             -H "Content-Type: application/json" \\
             -d '{
               "report_title": "Analisi SARIMA Vendite Q4",
               "output_filename": "sarima_vendite_q4_2024",
               "format_type": "html",
               "include_diagnostics": true,
               "include_forecast": true,
               "forecast_steps": 24
             }'
        </code></pre>
        
        <h4>Risposta di Esempio:</h4>
        <pre><code>
        {
            "model_id": "abc123e4-5678-9012-3456-789012345678",
            "report_path": "/outputs/reports/sarima_vendite_q4_2024.html",
            "format_type": "html",
            "generation_time": 15.7,
            "file_size_mb": 2.34,
            "download_url": "/reports/sarima_vendite_q4_2024.html"
        }
        </code></pre>
        
        <h4>Contenuti del Report:</h4>
        - **Executive Summary**: Sintesi dei risultati principali
        - **Model Overview**: Tipo, parametri e caratteristiche
        - **Performance Metrics**: Tutte le metriche di valutazione
        - **Diagnostics**: Analisi residui e test statistici  
        - **Forecasts**: Previsioni con intervalli di confidenza
        - **Visualizations**: Grafici interattivi e tabelle
        - **Recommendations**: Suggerimenti per miglioramenti
        """
        try:
            # Verifica l'esistenza del modello
            if not model_manager.model_exists(model_id):
                raise HTTPException(status_code=404, detail="Model not found")
            
            start_time = time.time()  # Misura il tempo di generazione
            
            # Genera il report utilizzando il servizio dedicato
            # La generazione può richiedere diversi minuti per report complessi
            report_result = await forecast_service.generate_report(
                model_id=model_id,
                report_title=request.report_title,
                output_filename=request.output_filename,
                format_type=request.format_type,
                include_diagnostics=request.include_diagnostics,
                include_forecast=request.include_forecast,
                forecast_steps=request.forecast_steps
            )
            
            generation_time = time.time() - start_time
            
            # Calcola la dimensione del file generato per statistiche
            report_path = Path(report_result["report_path"])
            file_size_mb = report_path.stat().st_size / (1024 * 1024) if report_path.exists() else None
            
            # Restituisce informazioni complete sul report generato
            return ReportGenerationResponse(
                model_id=model_id,
                report_path=str(report_path),
                format_type=request.format_type,
                generation_time=generation_time,
                file_size_mb=file_size_mb,
                download_url=f"/reports/{report_path.name}"  # URL relativo per il download
            )
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/reports/{filename}")
    async def download_report(filename: str):
        """
        Scarica un file di report generato in precedenza.
        
        Questo endpoint serve i file statici dei report permettendo
        il download diretto tramite browser o client HTTP.
        
        <h4>Parametri di Ingresso:</h4>
        <table class="table table-striped">
            <tr><th>Nome</th><th>Tipo</th><th>Descrizione</th><th>Richiesto</th></tr>
            <tr><td>filename</td><td>str</td><td>Nome del file da scaricare (path parameter)</td><td>Sì</td></tr>
        </table>
        
        <h4>Risposta:</h4>
        <table class="table table-striped">
            <tr><th>Tipo</th><th>Descrizione</th></tr>
            <tr><td>FileResponse</td><td>Stream del file con headers appropriati per il download</td></tr>
        </table>
        
        <h4>Headers di Risposta:</h4>
        <table class="table table-striped">
            <tr><th>Header</th><th>Descrizione</th></tr>
            <tr><td>Content-Disposition</td><td>Indica al browser di scaricare il file</td></tr>
            <tr><td>Content-Type</td><td>application/octet-stream (generico)</td></tr>
            <tr><td>Content-Length</td><td>Dimensione del file in bytes</td></tr>
        </table>
        
        <h4>Esempio di Chiamata:</h4>
        <pre><code>
        curl -X GET "http://localhost:8000/reports/sarima_model_report_20240823.html" \\
             -o "report_locale.html"
        </code></pre>
        
        <h4>Utilizzo dal Browser:</h4>
        </code></pre>
        http://localhost:8000/reports/sarima_model_report_20240823.html
        </code></pre>
        
        <h4>Formati Supportati:</h4>
        - **HTML**: Report interattivi visualizzabili nel browser
        - **PDF**: Report formattati per stampa e condivisione
        - **DOCX**: Report editabili in Microsoft Word
        
        <h4>Errori Possibili:</h4>
        - **404**: File del report non trovato
        • <strong>500</strong>: Errore nell'accesso al file system
        """
        try:
            from fastapi.responses import FileResponse
            
            # Directory dove vengono salvati i report
            # Deve corrispondere al percorso usato durante la generazione
            reports_dir = Path("outputs/reports")
            report_path = reports_dir / filename
            
            # Verifica l'esistenza del file sul filesystem
            if not report_path.exists():
                raise HTTPException(status_code=404, detail="Report file not found")
            
            # Restituisce il file come risposta streaming
            # FastAPI gestisce automaticamente gli headers appropriati
            return FileResponse(
                path=str(report_path),
                filename=filename,
                media_type='application/octet-stream'  # Tipo generico per download
            )
            
        except Exception as e:
            logger.error(f"Failed to download report: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Restituisce l'istanza dell'applicazione FastAPI completamente configurata
    return app


async def _train_model_background(
    model_manager: 'ModelManager',
    model_id: str,
    series: pd.Series,
    request: ModelTrainingRequest
):
    """
    Task in background per l'addestramento di modelli ARIMA/SARIMA/SARIMAX.
    
    Questa funzione viene eseguita asincronamente per evitare di bloccare
    l'API durante l'addestramento di modelli che potrebbero richiedere
    diversi minuti per convergere.
    
    Args:
        model_manager: Istanza del gestore modelli
        model_id: ID univoco del modello da addestrare
        series: Serie temporale per l'addestramento
        request: Configurazione originale della richiesta
    """
    try:
        # Delega l'addestramento al ModelManager
        # Il manager si occupa della serializzazione e salvataggio
        await model_manager.train_model(model_id, series, request)
    except Exception as e:
        logger = get_logger(__name__)
        # Log dell'errore per il debugging
        # Il modello rimarrà nello stato "training" o verrà marcato come "failed"
        logger.error(f"Background model training failed for {model_id}: {e}")


async def _train_var_model_background(
    model_manager: 'ModelManager',
    model_id: str,
    data: pd.DataFrame,
    request: VARTrainingRequest
):
    """
    Task in background per l'addestramento di modelli VAR.
    
    I modelli VAR (Vector Autoregression) gestiscono multiple serie temporali
    simultaneamente e possono richiedere più tempo per l'ottimizzazione.
    
    Args:
        model_manager: Istanza del gestore modelli
        model_id: ID univoco del modello VAR
        data: DataFrame con le serie temporali multivariate
        request: Configurazione originale della richiesta VAR
    """
    try:
        # Delega l'addestramento VAR al ModelManager specializzato
        await model_manager.train_var_model(model_id, data, request)
    except Exception as e:
        logger = get_logger(__name__)
        # Log degli errori per il monitoring e debugging
        logger.error(f"Background VAR model training failed for {model_id}: {e}")


# Crea un'istanza dell'applicazione con configurazione di default
# Questa istanza può essere utilizzata direttamente dai server ASGI
app = create_app()


if __name__ == "__main__":
    # Avvia il server di sviluppo se il modulo viene eseguito direttamente
    # In produzione utilizzare un server ASGI dedicato come Gunicorn o Hypercorn
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)