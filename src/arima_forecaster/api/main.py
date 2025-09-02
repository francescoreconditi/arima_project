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
    inventory_router,
    demand_sensing_router,
    advanced_models_router,
    evaluation_router,
    automl_router,
    visualization_router,
    data_management_router,
    enterprise_router,
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
        {
            "name": "Inventory Management",
            "description": "[CART] Ottimizzazione Inventory Management<br><br>Sistema completo per ottimizzazione scorte enterprise:<br><br>• Classificazione ABC/XYZ: Movement analysis automatico<br>• Slow/Fast Moving Optimization: Strategie differentiate<br>• Safety Stock Dinamico: Calcolo con demand uncertainty<br>• EOQ Optimization: Economic Order Quantity con sconti<br>• Multi-Echelon: Risk pooling e network optimization<br>• Capacity Constraints: Vincoli volume/peso/budget<br>• Bundle/Kitting Analysis: Make-to-Stock vs Assemble-to-Order",
        },
        {
            "name": "Demand Sensing", 
            "description": "[WEATHER] Demand Sensing Avanzato<br><br>Integrazione fattori esterni per forecast accuracy:<br><br>• Weather Integration: Previsioni meteo business-calibrate<br>• Google Trends Analysis: Pattern ricerche e correlazioni<br>• Social Sentiment: Sentiment analysis multi-platform<br>• Economic Indicators: Macro data per forecast context<br>• Calendar Events: Festività e eventi business impact<br>• Ensemble Forecasting: Combinazione multi-source weighted<br>• Sensitivity Analysis: Ottimizzazione pesi automatica",
        },
        {
            "name": "Advanced Models",
            "description": "[CHART] Modelli Avanzati e Comparazioni<br><br>Modelli multivariati e selezione automatica:<br><br>• Vector Autoregression (VAR): Serie interdipendenti<br>• Granger Causality Tests: Relazioni causali tra variabili<br>• Impulse Response Analysis: Shock propagation e policy impact<br>• Model Comparison Framework: Ranking automatico performance<br>• Auto-ML Selection: Selezione ottimale con cross-validation<br>• Grid Search Asincrono: Parameter tuning background jobs",
        },
        {
            "name": "Evaluation & Diagnostics",
            "description": "[MAGNIFY] Valutazione e Diagnostica Avanzata<br><br>Testing approfondito e validazione modelli:<br><br>• Cross-Validation: Time series split e walk-forward<br>• Residual Analysis: Test normalità, autocorrelazione, eteroschedasticità<br>• Model Comparison: Statistical tests di significatività<br>• Performance Metrics: 15+ metriche specializzate forecasting<br>• Diagnostic Plots: QQ-plot, ACF/PACF, residui analysis<br>• Backtesting: Simulazione performance storica realistica",
        },
        {
            "name": "AutoML & Optimization", 
            "description": "[ROBOT] AutoML e Ottimizzazione Automatica<br><br>Selezione e tuning automatico modelli:<br><br>• Optuna Integration: Bayesian optimization hyperparameters<br>• Hyperopt TPE: Tree-structured Parzen Estimator<br>• Scikit-optimize: Gaussian Process optimization<br>• Multi-objective: Ottimizzazione Pareto accuracy vs complexity<br>• Ensemble Methods: Stacking, voting, bagging automatico<br>• Grid Search: Parallel hyperparameter exploration",
        },
        {
            "name": "Visualization & Reporting",
            "description": "[CHART_BAR] Visualizzazione e Reporting Professionale<br><br>Grafici interattivi e report executive:<br><br>• Interactive Plots: Plotly dashboards con drill-down<br>• Executive Dashboards: KPI monitoring real-time<br>• Professional Reports: PDF/HTML multi-formato<br>• Model Comparison: Side-by-side performance visualization<br>• Custom Plots: Visualizzazioni personalizzate avanzate<br>• Alert System: Monitoring anomalie e degradation",
        },
        {
            "name": "Data Management",
            "description": "[DATABASE] Gestione Dati e Preprocessing<br><br>Pipeline completa data management:<br><br>• Data Upload: CSV/Excel/JSON con validazione automatica<br>• Quality Assessment: Scoring qualità multi-dimensionale<br>• Preprocessing Pipeline: Steps configurabili trasformazione<br>• Data Exploration: Analisi esplorativa automatica<br>• Train/Test Split: Metodi appropriati time series<br>• Data Validation: Check consistenza e anomalie",
        },
        {
            "name": "Multi-language & Enterprise",
            "description": "[GLOBE] Enterprise e Multi-lingua<br><br>Funzionalità enterprise production-ready:<br><br>• Multi-language: Traduzioni automatiche 8 lingue<br>• Enterprise Config: Scaling, security, governance<br>• Production Deployment: Blue/green, canary, rollback<br>• Compliance Audit: GDPR, SOC2, ISO27001<br>• Integration Testing: API, database, message queue<br>• Security Audit: Vulnerability scanning e penetration test",
        },
    ]
    
    # Crea l'istanza FastAPI con metadati completi
    app = FastAPI(
        title="ARIMA Forecaster API",
        description="""
        🚀 API REST Enterprise per Time Series Forecasting
        
        Sistema professionale per analisi e previsione di serie temporali con modelli statistici avanzati.
        
        📊 **Capacità Complete End-to-End**
        
        Dalla preparazione dati al deployment in produzione, con supporto per modelli ARIMA, SARIMA, VAR, **Facebook Prophet** e tecniche Auto-ML avanzate.
        
        🆕 **Prophet Integration Completa**
        
        - Training Prophet con parametri customizzati
        - Auto-selection ottimizzazione Bayesian/Grid Search  
        - Forecasting con decomposizione trend/seasonality/holidays
        - Comparazione intelligente Prophet vs ARIMA/SARIMA
        - Gestione festività per 6+ paesi (IT, US, UK, DE, FR, ES)
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
    app.include_router(inventory_router)  # /inventory prefix nel router
    app.include_router(demand_sensing_router)  # /demand-sensing prefix nel router
    app.include_router(advanced_models_router)  # /advanced-models prefix nel router
    app.include_router(evaluation_router)  # /evaluation prefix nel router
    app.include_router(automl_router)  # /automl prefix nel router
    app.include_router(visualization_router)  # /visualization prefix nel router
    app.include_router(data_management_router)  # /data prefix nel router
    app.include_router(enterprise_router)  # /enterprise prefix nel router

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
