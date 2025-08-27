"""
Applicazione FastAPI per i servizi di forecasting ARIMA/SARIMA/VAR/Prophet.

Questo modulo fornisce un'API REST completa per:
- Addestramento di modelli ARIMA, SARIMA, SARIMAX, VAR e Prophet
- Generazione di previsioni e intervalli di confidenza
- Selezione automatica dei parametri ottimali
- Diagnostica avanzata dei modelli
- Generazione di report completi
- Gestione del ciclo di vita dei modelli
- Forecasting avanzato con Facebook Prophet
"""

from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Importa Scalar per documentazione API migliorata
from scalar_fastapi import get_scalar_api_reference

# Importa tutti i routers
from arima_forecaster.api.routers import (
    diagnostics_router,
    forecasting_router,
    health_router,
    models_router,
    reports_router,
    training_router,
)
from arima_forecaster.utils.logger import get_logger


def create_app(
    model_storage_path: Optional[str] = None,
    enable_scalar: bool = True,
    production_mode: bool = False,
) -> FastAPI:
    """
    Crea l'istanza dell'applicazione FastAPI con tutti gli endpoint configurati.

    <h4>Parametri di Ingresso:</h4>
    <table class="table table-striped">
        <tr><th>Nome</th><th>Tipo</th><th>Descrizione</th><th>Default</th></tr>
        <tr><td>model_storage_path</td><td>str | None</td><td>Percorso per salvare i modelli addestrati</td><td>None (usa "models")</td></tr>
        <tr><td>enable_scalar</td><td>bool</td><td>Se abilitare Scalar UI per documentazione</td><td>True</td></tr>
        <tr><td>production_mode</td><td>bool</td><td>Se eseguire in modalità produzione</td><td>False</td></tr>
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
    - **Architettura modulare**: Routers separati per funzionalità
    - **CORS abilitato**: Permette richieste cross-origin
    - **Documentazione automatica**: Swagger UI su `/docs`, Scalar su `/scalar`
    - **Background tasks**: Per operazioni di lunga durata
    - **Error handling**: Gestione centralizzata degli errori
    - **Logging**: Sistema di log integrato
    """

    # Definisce le descrizioni per ogni tag/router
    tags_metadata = [
        {
            "name": "Health",
            "description": "🏥 Monitoraggio Stato Servizio<br><br>Endpoint per verificare lo stato e la salute del servizio API:<br><br>• Health checks per sistemi di monitoring<br>• Status checks per load balancer e deployment<br>• Informazioni API di base per client",
        },
        {
            "name": "Training",
            "description": "🎨 Addestramento Modelli Time Series<br><br>Training avanzato di modelli statistici per forecasting:<br><br>• ARIMA/SARIMA/SARIMAX: Modelli univariati con stagionalità<br>• VAR: Modelli multivariati per serie correlate<br>• Auto-ML: Selezione automatica parametri ottimali<br>• Background Processing: Training asincrono non bloccante<br>• Validazione Dati: Controlli automatici qualità input",
        },
        {
            "name": "Forecasting", 
            "description": "📈 Generazione Previsioni<br><br>Creazione di previsioni accurate da modelli addestrati:<br><br>• Previsioni Puntuali: Valori futuri stimati<br>• Intervalli Confidenza: Range di incertezza personalizzabili<br>• Variabili Esogene: Supporto regressori esterni per SARIMAX<br>• Timestamp Automatici: Generazione date future intelligente",
        },
        {
            "name": "Models",
            "description": "📁 Gestione Lifecycle Modelli<br><br>CRUD completo per modelli salvati e loro metadati:<br><br>• Elenco Modelli: Lista tutti i modelli disponibili<br>• Dettagli Modello: Parametri, metriche e stato specifico<br>• Eliminazione: Rimozione sicura modelli non necessari<br>• Metadati: Informazioni training, performance e configurazione",
        },
        {
            "name": "Diagnostics",
            "description": "🔍 Analisi Diagnostica Avanzata<br><br>Validazione statistica e performance dei modelli:<br><br>• Analisi Residui: Statistiche descrittive complete<br>• Test Ljung-Box: Verifica autocorrelazione residui<br>• Test Jarque-Bera: Controllo normalità distribuzione<br>• ACF/PACF: Funzioni di autocorrelazione<br>• Metriche Performance: MAE, RMSE, MAPE, R²",
        },
        {
            "name": "Reports",
            "description": "📄 Report Professionali<br><br>Generazione documentazione completa con Quarto:<br><br>• Formati Multipli: HTML interattivo, PDF stampa, DOCX editabile<br>• Grafici Avanzati: Visualizzazioni Plotly interattive<br>• Executive Summary: Sintesi risultati per management<br>• Analisi Tecnica: Diagnostica e raccomandazioni dettagliate",
        },
    ]
    
    # Crea l'istanza FastAPI con metadati completi
    app = FastAPI(
        title="ARIMA Forecaster API",
        description="""
        🚀 API REST Enterprise per Time Series Forecasting
        
        Sistema professionale per analisi e previsione di serie temporali con modelli statistici avanzati.
        
        📊 **Capacità Complete End-to-End**
        
        Dalla preparazione dati al deployment in produzione, con supporto per modelli ARIMA, SARIMA, VAR e tecniche Auto-ML avanzate.
        """,
        version="1.1.0",
        openapi_tags=tags_metadata,
        docs_url="/docs",  # Documentazione Swagger
        redoc_url="/redoc",  # Documentazione ReDoc alternativa
        contact={"name": "ARIMA Forecaster Team", "email": "support@arima-forecaster.com"},
        license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
    )

    # Configura il middleware CORS per permettere richieste cross-origin
    # Essenziale per applicazioni web frontend che consumano l'API
    cors_origins = (
        ["*"] if not production_mode else ["http://localhost:3000", "https://yourdomain.com"]
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Inizializza il logger
    logger = get_logger(__name__)

    # Configura Scalar UI per documentazione API moderna e interattiva
    if enable_scalar:

        @app.get("/scalar", include_in_schema=False)
        async def scalar_html():
            """Endpoint per servire la documentazione Scalar UI."""
            return get_scalar_api_reference(openapi_url=app.openapi_url, title=app.title)

    # Registra tutti i routers con i loro prefixes
    app.include_router(health_router)  # No prefix per health endpoints
    app.include_router(training_router)  # /models prefix nel router
    app.include_router(forecasting_router)  # /models prefix nel router
    app.include_router(models_router)  # /models prefix nel router
    app.include_router(diagnostics_router)  # /models prefix nel router
    app.include_router(reports_router)  # No prefix per reports

    # Log di startup
    @app.on_event("startup")
    async def startup_event():
        """Eventi da eseguire all'avvio dell'applicazione."""
        logger.info(f"🚀 ARIMA Forecaster API v{app.version} started")
        logger.info(f"📁 Model storage path: {model_storage_path or 'models'}")
        logger.info(f"📊 Scalar UI enabled: {enable_scalar}")
        logger.info(f"🔧 Production mode: {production_mode}")
        logger.info("📚 API documentation available at /docs and /scalar")

    # Log di shutdown
    @app.on_event("shutdown")
    async def shutdown_event():
        """Eventi da eseguire allo shutdown dell'applicazione."""
        logger.info("🛑 ARIMA Forecaster API shutting down")

    return app


# Crea l'istanza di default dell'app
app = create_app()


if __name__ == "__main__":
    """Entry point per esecuzione diretta."""
    import uvicorn

    uvicorn.run(
        "arima_forecaster.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload in development
        log_level="info",
    )
