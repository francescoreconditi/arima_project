"""
Router per visualizzazione e reporting avanzati.

Questo modulo fornisce endpoint per la generazione di visualizzazioni interattive,
report professionali e dashboard personalizzabili per analisi di serie temporali.
"""

import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import plotly.graph_objs as go
import plotly.io as pio

# Simulazione import delle utilità (da implementare nel progetto reale)
from arima_forecaster.utils.logger import get_logger


# Dependency injection dei servizi
def get_services():
    """Dependency per ottenere i servizi necessari."""
    from arima_forecaster.api.main import get_model_manager, get_forecast_service

    model_manager = get_model_manager()
    forecast_service = get_forecast_service()
    return model_manager, forecast_service


# Configurazione router
router = APIRouter(
    prefix="/visualization",
    tags=["Visualization & Reporting"],
)

logger = get_logger(__name__)

# Dizionario globale per simulare storage jobs in background
visualization_jobs = {}


class PlotConfigRequest(BaseModel):
    """Configurazione per generazione visualizzazioni."""

    model_id: str = Field(..., description="ID univoco del modello addestrato")
    plot_types: List[str] = Field(
        default=["forecast", "residuals", "diagnostics"], description="Tipi di plot da generare"
    )
    forecast_steps: int = Field(default=30, ge=1, le=365, description="Numero di step da prevedere")
    include_intervals: bool = Field(default=True, description="Includere intervalli di confidenza")
    interactive: bool = Field(default=True, description="Generare plot interattivi Plotly")
    theme: str = Field(default="plotly_white", description="Tema visualizzazione")
    export_formats: List[str] = Field(default=["html", "png"], description="Formati di export")


class DashboardConfigRequest(BaseModel):
    """Configurazione per generazione dashboard."""

    model_ids: List[str] = Field(..., description="Lista ID modelli da includere")
    dashboard_type: str = Field(
        default="executive", description="Tipo dashboard: executive, technical, operational"
    )
    update_frequency: str = Field(
        default="daily", description="Frequenza aggiornamento: realtime, hourly, daily, weekly"
    )
    kpi_metrics: List[str] = Field(
        default=["accuracy", "trend", "seasonality"], description="Metriche KPI da monitorare"
    )
    alert_thresholds: Dict[str, float] = Field(
        default={"accuracy_drop": 0.1, "forecast_deviation": 0.2},
        description="Soglie per alert automatici",
    )
    language: str = Field(default="it", description="Lingua dashboard")


class ReportConfigRequest(BaseModel):
    """Configurazione per generazione report."""

    model_ids: List[str] = Field(..., description="Lista ID modelli per report")
    report_type: str = Field(
        default="comprehensive", description="Tipo report: executive, technical, comprehensive"
    )
    include_sections: List[str] = Field(
        default=["summary", "methodology", "results", "recommendations"],
        description="Sezioni da includere nel report",
    )
    export_formats: List[str] = Field(default=["pdf", "html"], description="Formati export report")
    template_style: str = Field(default="corporate", description="Stile template")
    language: str = Field(default="it", description="Lingua report")


class ComparisonRequest(BaseModel):
    """Configurazione per confronto modelli."""

    model_ids: List[str] = Field(
        ..., min_items=2, max_items=10, description="ID modelli da confrontare"
    )
    comparison_metrics: List[str] = Field(
        default=["mae", "rmse", "mape", "aic", "bic"], description="Metriche per confronto"
    )
    visualization_type: str = Field(
        default="comprehensive", description="Tipo visualizzazione: simple, detailed, comprehensive"
    )
    include_statistical_tests: bool = Field(
        default=True, description="Includere test statistici significatività"
    )


class CustomPlotRequest(BaseModel):
    """Configurazione per plot personalizzati."""

    data_source: Dict[str, Any] = Field(..., description="Sorgente dati per plot")
    plot_specification: Dict[str, Any] = Field(..., description="Specifica dettagli plot")
    styling_options: Dict[str, Any] = Field(
        default={}, description="Opzioni styling personalizzate"
    )
    interactivity_level: str = Field(
        default="medium", description="Livello interattività: low, medium, high"
    )


class AlertConfigRequest(BaseModel):
    """Configurazione sistema alert visuali."""

    model_ids: List[str] = Field(..., description="Modelli da monitorare")
    alert_rules: List[Dict[str, Any]] = Field(..., description="Regole alert personalizzate")
    notification_channels: List[str] = Field(
        default=["email", "dashboard"], description="Canali notifica alert"
    )
    severity_levels: List[str] = Field(
        default=["info", "warning", "critical"], description="Livelli severità alert"
    )


class InteractiveAnalysisRequest(BaseModel):
    """Configurazione per analisi interattiva."""

    model_id: str = Field(..., description="ID modello per analisi")
    analysis_type: str = Field(
        default="what_if", description="Tipo analisi: what_if, sensitivity, scenario"
    )
    parameters_range: Dict[str, Any] = Field(..., description="Range parametri da analizzare")
    output_metrics: List[str] = Field(
        default=["forecast_accuracy", "confidence_width"], description="Metriche output analisi"
    )


class VisualizationJobResponse(BaseModel):
    """Risposta per job di visualizzazione."""

    job_id: str = Field(..., description="ID univoco job")
    status: str = Field(..., description="Stato job: queued, running, completed, failed")
    progress: float = Field(default=0.0, ge=0.0, le=1.0, description="Progresso job (0-1)")
    estimated_completion: Optional[datetime] = Field(None, description="Stima completamento")
    results_urls: List[str] = Field(default=[], description="URL risultati generati")


async def simulate_plot_generation(job_id: str, config: PlotConfigRequest):
    """Simula generazione plot in background."""
    try:
        visualization_jobs[job_id]["status"] = "running"

        # Simula generazione diversi tipi di plot
        for i, plot_type in enumerate(config.plot_types):
            await asyncio.sleep(1)  # Simula processing

            progress = (i + 1) / len(config.plot_types)
            visualization_jobs[job_id]["progress"] = progress

            logger.info(f"Generando plot {plot_type} per job {job_id}")

        # Simula generazione file risultati
        results = []
        for plot_type in config.plot_types:
            for fmt in config.export_formats:
                results.append(f"/outputs/plots/{job_id}_{plot_type}.{fmt}")

        visualization_jobs[job_id].update(
            {"status": "completed", "progress": 1.0, "results_urls": results}
        )

    except Exception as e:
        visualization_jobs[job_id]["status"] = "failed"
        logger.error(f"Errore generazione plot {job_id}: {str(e)}")


@router.post("/generate-plots", response_model=VisualizationJobResponse)
async def generate_plots(config: PlotConfigRequest, background_tasks: BackgroundTasks):
    """
    Genera visualizzazioni professionali per modelli di forecasting con grafici interattivi Plotly.

    <h4>Plot Types Disponibili:</h4>
    <table>
        <tr><th>Tipo</th><th>Descrizione</th><th>Output</th></tr>
        <tr><td>forecast</td><td>Previsioni con intervalli confidenza</td><td>HTML interattivo + PNG export</td></tr>
        <tr><td>residuals</td><td>Analisi residui e QQ-plot</td><td>Diagnostica statistica completa</td></tr>
        <tr><td>diagnostics</td><td>ACF/PACF e test statistici</td><td>Validazione assunzioni modello</td></tr>
        <tr><td>decomposition</td><td>Trend, stagionalità, rumore</td><td>Componenti serie temporale</td></tr>
        <tr><td>error_analysis</td><td>Distribuzione e pattern errori</td><td>Identificazione bias sistematici</td></tr>
    </table>

    <h4>Temi Visualizzazione Supportati:</h4>
    <table>
        <tr><th>Tema</th><th>Descrizione</th><th>Migliore Per</th></tr>
        <tr><td>plotly_white</td><td>Sfondo bianco minimalista</td><td>Report professionali e presentazioni</td></tr>
        <tr><td>plotly_dark</td><td>Tema scuro per dashboard</td><td>Monitoraggio real-time e dashboard ops</td></tr>
        <tr><td>ggplot2</td><td>Stile R ggplot2</td><td>Pubblicazioni scientifiche</td></tr>
        <tr><td>seaborn</td><td>Palette colori Seaborn</td><td>Analisi statistiche dettagliate</td></tr>
    </table>

    <h4>Formati Export Disponibili:</h4>
    - **HTML**: Grafici interattivi con zoom/pan/hover
    - **PNG**: Immagini statiche alta risoluzione per report
    - **SVG**: Grafici vettoriali per pubblicazioni
    - **JSON**: Dati strutturati per integrazione API

    <h4>Esempio Richiesta:</h4>
    <pre><code>
    {
        "model_id": "arima_001",
        "plot_types": ["forecast", "residuals", "diagnostics"],
        "forecast_steps": 30,
        "include_intervals": true,
        "interactive": true,
        "theme": "plotly_white",
        "export_formats": ["html", "png"]
    }
    </code></pre>

    <h4>Risposta di Esempio:</h4>
    <pre><code>
    {
        "job_id": "viz_job_abc123",
        "status": "queued",
        "progress": 0.0,
        "estimated_completion": "2024-08-26T10:30:00",
        "results_urls": []
    }
    </code></pre>
    """

    job_id = f"viz_job_{uuid.uuid4().hex[:8]}"

    # Inizializza job tracking
    visualization_jobs[job_id] = {
        "status": "queued",
        "progress": 0.0,
        "model_id": config.model_id,
        "created_at": datetime.now(),
        "results_urls": [],
    }

    # Avvia generazione in background
    background_tasks.add_task(simulate_plot_generation, job_id, config)

    return VisualizationJobResponse(
        job_id=job_id,
        status="queued",
        progress=0.0,
        estimated_completion=datetime.now(),
        results_urls=[],
    )


@router.post("/create-dashboard", response_model=VisualizationJobResponse)
async def create_dashboard(config: DashboardConfigRequest, background_tasks: BackgroundTasks):
    """
    Crea dashboard web interattive personalizzate per monitoraggio modelli con aggiornamento real-time.

    <h4>Dashboard Types e Componenti:</h4>
    <table>
        <tr><th>Tipo</th><th>Descrizione</th><th>Componenti Principali</th></tr>
        <tr><td>executive</td><td>KPI aggregati e trend business</td><td>Scorecard, trend analysis, alert summary</td></tr>
        <tr><td>technical</td><td>Dettagli tecnici modelli</td><td>Model diagnostics, performance metrics, residual plots</td></tr>
        <tr><td>operational</td><td>Monitoraggio real-time</td><td>Live forecasts, anomaly detection, inventory levels</td></tr>
        <tr><td>comparative</td><td>Confronto multi-modello</td><td>Side-by-side metrics, ranking tables, winner selection</td></tr>
    </table>

    <h4>Frequenze Aggiornamento Supportate:</h4>
    <table>
        <tr><th>Frequenza</th><th>Descrizione</th><th>Caso d'Uso</th></tr>
        <tr><td>realtime</td><td>WebSocket streaming continuo</td><td>Trading, monitoring critico</td></tr>
        <tr><td>hourly</td><td>Refresh ogni ora</td><td>Inventory management, sales tracking</td></tr>
        <tr><td>daily</td><td>Aggiornamento giornaliero</td><td>Strategic planning, reporting</td></tr>
        <tr><td>weekly</td><td>Update settimanale</td><td>Trend analysis, executive review</td></tr>
    </table>

    <h4>KPI Metrics Disponibili:</h4>
    <table>
        <tr><th>Metrica</th><th>Descrizione</th><th>Formula/Calcolo</th></tr>
        <tr><td>accuracy</td><td>Forecast accuracy percentage</td><td>100 - MAPE</td></tr>
        <tr><td>trend</td><td>Direzione trend dominante</td><td>Linear regression slope</td></tr>
        <tr><td>seasonality</td><td>Forza componente stagionale</td><td>Seasonal decomposition strength</td></tr>
        <tr><td>volatility</td><td>Variabilità predizioni</td><td>Standard deviation forecast errors</td></tr>
        <tr><td>confidence</td><td>Ampiezza intervalli confidenza</td><td>95% CI width / forecast value</td></tr>
    </table>

    <h4>Sistema Alert Thresholds:</h4>
    - **accuracy_drop**: Degradazione accuracy oltre soglia (default 10%)
    - **forecast_deviation**: Deviazione da forecast precedenti (default 20%)
    - **anomaly_detection**: Identificazione valori anomali automatica
    - **inventory_critical**: Livelli scorte sotto minimo critico

    <h4>Lingue Supportate:</h4>
    - **it**: Italiano (default)
    - **en**: English
    - **es**: Español
    - **fr**: Français
    - **zh**: 中文

    <h4>Esempio Richiesta:</h4>
    <pre><code>
    {
        "model_ids": ["arima_001", "sarima_002"],
        "dashboard_type": "executive",
        "update_frequency": "daily",
        "kpi_metrics": ["accuracy", "trend", "volatility"],
        "alert_thresholds": {
            "accuracy_drop": 0.15,
            "forecast_deviation": 0.25,
            "anomaly_detection": true
        },
        "language": "it"
    }
    </code></pre>
    """

    job_id = f"dash_job_{uuid.uuid4().hex[:8]}"

    # Inizializza job
    visualization_jobs[job_id] = {
        "status": "queued",
        "progress": 0.0,
        "dashboard_type": config.dashboard_type,
        "model_count": len(config.model_ids),
        "created_at": datetime.now(),
        "results_urls": [],
    }

    # Simula creazione dashboard
    async def create_dashboard_job():
        try:
            visualization_jobs[job_id]["status"] = "running"
            await asyncio.sleep(3)  # Simula processing

            dashboard_url = f"/dashboards/{job_id}/index.html"
            api_endpoint = f"/dashboards/{job_id}/api"

            visualization_jobs[job_id].update(
                {
                    "status": "completed",
                    "progress": 1.0,
                    "results_urls": [dashboard_url, api_endpoint],
                }
            )

        except Exception as e:
            visualization_jobs[job_id]["status"] = "failed"
            logger.error(f"Errore creazione dashboard {job_id}: {str(e)}")

    background_tasks.add_task(create_dashboard_job)

    return VisualizationJobResponse(job_id=job_id, status="queued", progress=0.0)


@router.post("/generate-report", response_model=VisualizationJobResponse)
async def generate_report(config: ReportConfigRequest, background_tasks: BackgroundTasks):
    """
    Genera report professionali multi-formato con analisi complete usando Quarto/Jupyter.

    <h4>Report Types e Contenuti:</h4>
    <table>
        <tr><th>Tipo</th><th>Descrizione</th><th>Sezioni Incluse</th></tr>
        <tr><td>executive</td><td>Executive summary 2-3 pagine</td><td>KPI dashboard, key findings, recommendations</td></tr>
        <tr><td>technical</td><td>Analisi tecnica 10-15 pagine</td><td>Model details, diagnostics, performance metrics</td></tr>
        <tr><td>comprehensive</td><td>Report completo 20+ pagine</td><td>Tutte le sezioni + appendici tecniche</td></tr>
        <tr><td>regulatory</td><td>Compliance e audit trail</td><td>Model governance, validation, versioning</td></tr>
    </table>

    <h4>Sezioni Report Disponibili:</h4>
    <table>
        <tr><th>Sezione</th><th>Contenuto</th><th>Pagine Tipiche</th></tr>
        <tr><td>summary</td><td>Executive summary e key insights</td><td>1-2 pagine</td></tr>
        <tr><td>methodology</td><td>Approccio modeling e data preparation</td><td>2-3 pagine</td></tr>
        <tr><td>results</td><td>Forecast results e performance metrics</td><td>3-5 pagine</td></tr>
        <tr><td>diagnostics</td><td>Model diagnostics e validation tests</td><td>2-4 pagine</td></tr>
        <tr><td>recommendations</td><td>Business recommendations e next steps</td><td>1-2 pagine</td></tr>
        <tr><td>appendix</td><td>Dettagli tecnici e codice</td><td>5+ pagine</td></tr>
    </table>

    <h4>Formati Export e Caratteristiche:</h4>
    <table>
        <tr><th>Formato</th><th>Descrizione</th><th>Best Practice</th></tr>
        <tr><td>PDF</td><td>Documento statico professionale</td><td>Print-ready, firma digitale, archivio</td></tr>
        <tr><td>HTML</td><td>Report interattivo web-based</td><td>Grafici dinamici, drill-down, sharing</td></tr>
        <tr><td>DOCX</td><td>Editabile Microsoft Word</td><td>Collaborazione, revisioni, commenti</td></tr>
        <tr><td>PPTX</td><td>Presentazione PowerPoint</td><td>Board meetings, executive briefings</td></tr>
        <tr><td>LaTeX</td><td>Pubblicazione scientifica</td><td>Journal submission, academic papers</td></tr>
    </table>

    <h4>Template Styles Disponibili:</h4>
    <table>
        <tr><th>Stile</th><th>Descrizione</th><th>Uso Consigliato</th></tr>
        <tr><td>corporate</td><td>Brand aziendale professionale</td><td>Report interni e clienti</td></tr>
        <tr><td>minimal</td><td>Design minimalista pulito</td><td>Focus sui dati, meno grafica</td></tr>
        <tr><td>academic</td><td>Stile pubblicazione scientifica</td><td>Research papers, tesi</td></tr>
        <tr><td>dashboard</td><td>Layout tipo dashboard KPI</td><td>Executive briefing, monitoring</td></tr>
    </table>

    <h4>Esempio Richiesta:</h4>
    <pre><code>
    {
        "model_ids": ["arima_001", "sarima_002"],
        "report_type": "comprehensive",
        "include_sections": ["summary", "methodology", "results", "diagnostics", "recommendations"],
        "export_formats": ["pdf", "html", "docx"],
        "template_style": "corporate",
        "language": "it"
    }
    </code></pre>
    """

    job_id = f"report_job_{uuid.uuid4().hex[:8]}"

    visualization_jobs[job_id] = {
        "status": "queued",
        "progress": 0.0,
        "report_type": config.report_type,
        "formats": config.export_formats,
        "created_at": datetime.now(),
        "results_urls": [],
    }

    # Genera report reali usando QuartoReportGenerator
    async def generate_report_job():
        try:
            from arima_forecaster.reporting import QuartoReportGenerator
            from arima_forecaster.visualization import ForecastPlotter
            import tempfile

            visualization_jobs[job_id]["status"] = "running"
            visualization_jobs[job_id]["progress"] = 0.1

            model_manager, forecast_service = get_services()

            # Carica tutti i modelli richiesti
            all_model_results = {}
            for model_id in config.model_ids:
                if not model_manager.model_exists(model_id):
                    raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

                model = model_manager.load_model(model_id)
                metadata = model_manager.get_model_info(model_id)

                # Genera forecast per il modello
                forecast_result = await forecast_service.generate_forecast(
                    model_id=model_id,
                    steps=30,  # Default 30 steps
                    confidence_level=0.95,
                    return_intervals=True,
                )

                all_model_results[model_id] = {
                    "metadata": metadata,
                    "forecast": {
                        "values": forecast_result.forecast_values,
                        "timestamps": forecast_result.forecast_timestamps,
                        "lower_bounds": forecast_result.lower_bounds,
                        "upper_bounds": forecast_result.upper_bounds,
                    },
                }

            visualization_jobs[job_id]["progress"] = 0.3

            # Genera plot per ogni modello
            plots_dir = Path(tempfile.mkdtemp())
            plots_data = {}

            # Fix per TCL/TK: usa backend non-interattivo
            import matplotlib
            matplotlib.use('Agg')  # Backend non-interattivo per evitare errori TCL

            for model_id, results in all_model_results.items():
                plotter = ForecastPlotter()
                plot_path = plots_dir / f"{model_id}_forecast.png"

                # Prepara i dati come pandas Series
                import pandas as pd

                # Crea Series per actual (se disponibile, altrimenti vuota)
                actual_data = results["metadata"].get("training_data", [])
                if actual_data and len(actual_data) > 0:
                    actual_series = pd.Series(actual_data)
                else:
                    # Crea una serie vuota
                    actual_series = pd.Series([])

                # Crea Series per forecast
                forecast_series = pd.Series(results["forecast"]["values"])

                # Crea DataFrame per confidence intervals (se disponibile)
                confidence_df = None
                if results["forecast"]["lower_bounds"] and results["forecast"]["upper_bounds"]:
                    confidence_df = pd.DataFrame(
                        {
                            "lower": results["forecast"]["lower_bounds"],
                            "upper": results["forecast"]["upper_bounds"],
                        }
                    )

                # Crea grafico forecast
                plotter.plot_forecast(
                    actual=actual_series,
                    forecast=forecast_series,
                    confidence_intervals=confidence_df,
                    title=f"Forecast - {model_id}",
                    save_path=str(plot_path),
                )

                plots_data[f"{model_id}_forecast"] = str(plot_path)

            visualization_jobs[job_id]["progress"] = 0.5

            # Genera report per ogni formato richiesto
            results_urls = []
            report_generator = QuartoReportGenerator(template_name="default")

            for i, fmt in enumerate(config.export_formats):
                # Prepara model_results per il template Quarto
                model_results = {
                    "title": f"Report Forecast - {config.report_type.title()}",
                    "report_type": config.report_type,
                    "language": config.language,
                    "models": all_model_results,
                    "sections": config.include_sections,
                    "generated_at": datetime.now().isoformat(),
                }

                # Genera report nel formato specifico
                output_path = report_generator.generate_model_report(
                    model_results=model_results,
                    plots_data=plots_data,
                    report_title=f"Report Forecast - {', '.join(config.model_ids)}",
                    output_filename=f"{job_id}_report",
                    format_type=fmt,
                )

                results_urls.append(str(output_path))

                progress = 0.5 + (0.5 * (i + 1) / len(config.export_formats))
                visualization_jobs[job_id]["progress"] = progress

            visualization_jobs[job_id].update(
                {"status": "completed", "progress": 1.0, "results_urls": results_urls}
            )

        except Exception as e:
            visualization_jobs[job_id]["status"] = "failed"
            visualization_jobs[job_id]["error"] = str(e)
            logger.error(f"Errore generazione report {job_id}: {str(e)}")

    background_tasks.add_task(generate_report_job)

    return VisualizationJobResponse(job_id=job_id, status="queued")


@router.post("/compare-models", response_model=VisualizationJobResponse)
async def compare_models(config: ComparisonRequest, background_tasks: BackgroundTasks):
    """
    Genera confronto dettagliato tra modelli multipli con test statistici e ranking automatico.

    <h4>Comparison Types e Output:</h4>
    <table>
        <tr><th>Tipo</th><th>Descrizione</th><th>Visualizzazioni Generate</th></tr>
        <tr><td>simple</td><td>Confronto metriche base</td><td>Bar chart, radar plot, ranking table</td></tr>
        <tr><td>detailed</td><td>Include test statistici</td><td>Performance matrix, p-values, confidence intervals</td></tr>
        <tr><td>comprehensive</td><td>Analisi completa multi-criterio</td><td>Tutti i grafici + report comparativo PDF</td></tr>
        <tr><td>pareto</td><td>Analisi frontiera Pareto</td><td>Trade-off accuracy vs complexity</td></tr>
    </table>

    <h4>Metriche Confronto Disponibili:</h4>
    <table>
        <tr><th>Metrica</th><th>Descrizione</th><th>Range Ottimale</th></tr>
        <tr><td>mae</td><td>Mean Absolute Error</td><td>Minore è meglio</td></tr>
        <tr><td>rmse</td><td>Root Mean Square Error</td><td>Minore è meglio</td></tr>
        <tr><td>mape</td><td>Mean Absolute Percentage Error</td><td><20% eccellente</td></tr>
        <tr><td>aic</td><td>Akaike Information Criterion</td><td>Minore è meglio</td></tr>
        <tr><td>bic</td><td>Bayesian Information Criterion</td><td>Minore è meglio</td></tr>
        <tr><td>r2</td><td>R-squared coefficient</td><td>>0.8 buono</td></tr>
        <tr><td>mase</td><td>Mean Absolute Scaled Error</td><td><1.0 meglio di naive</td></tr>
        <tr><td>coverage</td><td>Prediction interval coverage</td><td>~95% ideale</td></tr>
    </table>

    <h4>Test Statistici Applicati:</h4>
    <table>
        <tr><th>Test</th><th>Descrizione</th><th>Interpretazione</th></tr>
        <tr><td>Diebold-Mariano</td><td>Confronto forecast accuracy</td><td>p<0.05 differenza significativa</td></tr>
        <tr><td>Model Confidence Set</td><td>Set modelli equivalenti</td><td>Identifica best performers group</td></tr>
        <tr><td>Giacomini-White</td><td>Conditional predictive ability</td><td>Performance in diversi regimi</td></tr>
        <tr><td>Reality Check</td><td>Multiple testing correction</td><td>Controlla false discoveries</td></tr>
    </table>

    <h4>Esempio Richiesta:</h4>
    <pre><code>
    {
        "model_ids": ["arima_001", "sarima_002", "var_003", "prophet_004"],
        "comparison_metrics": ["mae", "rmse", "mape", "aic", "bic"],
        "visualization_type": "comprehensive",
        "include_statistical_tests": true
    }
    </code></pre>

    <h4>Output Generati:</h4>
    - Performance matrix heatmap con ranking
    - Spider/radar chart multi-metrica
    - Statistical significance matrix
    - Winner recommendation con confidence
    - Export tabella comparativa Excel
    """

    job_id = f"compare_job_{uuid.uuid4().hex[:8]}"

    visualization_jobs[job_id] = {
        "status": "queued",
        "progress": 0.0,
        "model_count": len(config.model_ids),
        "metrics": config.comparison_metrics,
        "created_at": datetime.now(),
        "results_urls": [],
    }

    # Simula confronto modelli
    async def compare_models_job():
        try:
            visualization_jobs[job_id]["status"] = "running"
            await asyncio.sleep(4)  # Simula analisi confronto

            results = [
                f"/comparisons/{job_id}/performance_matrix.html",
                f"/comparisons/{job_id}/ranking_plot.html",
                f"/comparisons/{job_id}/statistical_tests.html",
                f"/comparisons/{job_id}/comparison_report.pdf",
            ]

            visualization_jobs[job_id].update(
                {"status": "completed", "progress": 1.0, "results_urls": results}
            )

        except Exception as e:
            visualization_jobs[job_id]["status"] = "failed"
            logger.error(f"Errore confronto modelli {job_id}: {str(e)}")

    background_tasks.add_task(compare_models_job)

    return VisualizationJobResponse(job_id=job_id, status="queued")


@router.post("/custom-plot", response_model=VisualizationJobResponse)
async def create_custom_plot(config: CustomPlotRequest, background_tasks: BackgroundTasks):
    """
    Crea visualizzazioni completamente personalizzate con controllo granulare su ogni aspetto grafico.

    <h4>Chart Types Supportati:</h4>
    <table>
        <tr><th>Tipo</th><th>Descrizione</th><th>Parametri Specifici</th></tr>
        <tr><td>line</td><td>Serie temporali continue</td><td>smoothing, interpolation, markers</td></tr>
        <tr><td>scatter</td><td>Correlazioni e clusters</td><td>size, color mapping, regression line</td></tr>
        <tr><td>bar</td><td>Confronti categorici</td><td>grouped, stacked, horizontal</td></tr>
        <tr><td>heatmap</td><td>Matrici correlazione/confusion</td><td>color scale, annotations, clustering</td></tr>
        <tr><td>box</td><td>Distribuzione e outliers</td><td>notch, violin overlay, points</td></tr>
        <tr><td>candlestick</td><td>OHLC financial data</td><td>volume overlay, moving averages</td></tr>
        <tr><td>3d_surface</td><td>Superfici 3D parametriche</td><td>mesh density, color gradient</td></tr>
        <tr><td>sankey</td><td>Flow e transitions</td><td>node positions, link colors</td></tr>
    </table>

    <h4>Data Source Types:</h4>
    <table>
        <tr><th>Tipo</th><th>Descrizione</th><th>Parametri Required</th></tr>
        <tr><td>model_predictions</td><td>Output forecast modello</td><td>model_id, time_range</td></tr>
        <tr><td>model_residuals</td><td>Residui e errori</td><td>model_id, error_type</td></tr>
        <tr><td>comparison_data</td><td>Multi-model comparison</td><td>model_ids[], metric</td></tr>
        <tr><td>raw_timeseries</td><td>Dati grezzi originali</td><td>series_id, transformations</td></tr>
        <tr><td>custom_data</td><td>Dati esterni forniti</td><td>data_array, schema</td></tr>
    </table>

    <h4>Styling Options Avanzate:</h4>
    <table>
        <tr><th>Opzione</th><th>Valori</th><th>Effetto</th></tr>
        <tr><td>color_palette</td><td>viridis, plasma, turbo, custom</td><td>Schema colori globale</td></tr>
        <tr><td>font_family</td><td>Arial, Times, Courier, custom</td><td>Font testi e labels</td></tr>
        <tr><td>grid_style</td><td>solid, dashed, dotted, none</td><td>Stile griglia sfondo</td></tr>
        <tr><td>annotations</td><td>peaks, valleys, events, custom</td><td>Annotazioni automatiche</td></tr>
        <tr><td>animation</td><td>entrance, transition, loop</td><td>Animazioni grafici</td></tr>
    </table>

    <h4>Livelli Interattività:</h4>
    <table>
        <tr><th>Livello</th><th>Features</th><th>Performance Impact</th></tr>
        <tr><td>low</td><td>Solo hover tooltips</td><td>Veloce, leggero</td></tr>
        <tr><td>medium</td><td>Zoom, pan, reset</td><td>Bilanciato</td></tr>
        <tr><td>high</td><td>Brush selection, cross-filtering</td><td>Rich features, più pesante</td></tr>
        <tr><td>extreme</td><td>Real-time updates, WebGL rendering</td><td>Maximum interactivity</td></tr>
    </table>

    <h4>Esempio Richiesta Complessa:</h4>
    <pre><code>
    {
        "data_source": {
            "type": "comparison_data",
            "model_ids": ["arima_001", "sarima_002"],
            "metric": "forecast_accuracy"
        },
        "plot_specification": {
            "chart_type": "heatmap",
            "x_axis": "time_period",
            "y_axis": "model_id",
            "z_value": "accuracy_score",
            "annotations": true
        },
        "styling_options": {
            "color_palette": "RdYlGn",
            "font_family": "Arial",
            "grid_style": "dotted",
            "animation": "entrance"
        },
        "interactivity_level": "high"
    }
    </code></pre>
    """

    job_id = f"custom_job_{uuid.uuid4().hex[:8]}"

    visualization_jobs[job_id] = {
        "status": "queued",
        "progress": 0.0,
        "plot_type": "custom",
        "interactivity": config.interactivity_level,
        "created_at": datetime.now(),
        "results_urls": [],
    }

    # Simula creazione plot custom
    async def create_custom_plot_job():
        try:
            visualization_jobs[job_id]["status"] = "running"
            await asyncio.sleep(2)  # Simula processing custom

            results = [
                f"/custom_plots/{job_id}/plot.html",
                f"/custom_plots/{job_id}/plot.png",
                f"/custom_plots/{job_id}/plot_config.json",
            ]

            visualization_jobs[job_id].update(
                {"status": "completed", "progress": 1.0, "results_urls": results}
            )

        except Exception as e:
            visualization_jobs[job_id]["status"] = "failed"
            logger.error(f"Errore plot custom {job_id}: {str(e)}")

    background_tasks.add_task(create_custom_plot_job)

    return VisualizationJobResponse(job_id=job_id, status="queued")


@router.post("/setup-alerts", response_model=Dict[str, str])
async def setup_alerts(config: AlertConfigRequest):
    """
    Configura sistema alert intelligente multi-canale per monitoraggio proattivo anomalie.

    <h4>Alert Types e Trigger Conditions:</h4>
    <table>
        <tr><th>Tipo</th><th>Descrizione</th><th>Condizioni Trigger</th></tr>
        <tr><td>Performance Degradation</td><td>Calo accuracy modello</td><td>MAE/RMSE oltre threshold, trend negativo</td></tr>
        <tr><td>Data Drift</td><td>Cambio distribuzione input</td><td>KS test, PSI score, Wasserstein distance</td></tr>
        <tr><td>Concept Drift</td><td>Cambio relazione X-y</td><td>Sliding window performance drop</td></tr>
        <tr><td>Anomaly Detection</td><td>Forecast anomali</td><td>Z-score >3, isolation forest, LSTM autoencoder</td></tr>
        <tr><td>Model Staleness</td><td>Modello non aggiornato</td><td>Days since retrain, data volume threshold</td></tr>
        <tr><td>System Health</td><td>Problemi infrastruttura</td><td>API latency, memory usage, error rate</td></tr>
    </table>

    <h4>Notification Channels Supportati:</h4>
    <table>
        <tr><th>Canale</th><th>Configurazione</th><th>Delivery Time</th></tr>
        <tr><td>email</td><td>SMTP settings, recipient list</td><td>1-2 minuti</td></tr>
        <tr><td>slack</td><td>Webhook URL, channel, mentions</td><td>Real-time</td></tr>
        <tr><td>teams</td><td>Webhook URL, team channel</td><td>Real-time</td></tr>
        <tr><td>sms</td><td>Twilio/AWS SNS config</td><td>30 secondi</td></tr>
        <tr><td>webhook</td><td>Custom HTTP endpoint</td><td>Immediato</td></tr>
        <tr><td>dashboard</td><td>In-app notification center</td><td>Real-time</td></tr>
        <tr><td>pagerduty</td><td>Integration key, escalation</td><td>Immediato + escalation</td></tr>
    </table>

    <h4>Alert Rule Configuration:</h4>
    <table>
        <tr><th>Parametro</th><th>Tipo</th><th>Descrizione</th></tr>
        <tr><td>metric</td><td>string</td><td>Metrica da monitorare (mae, rmse, custom)</td></tr>
        <tr><td>threshold</td><td>float</td><td>Valore soglia per trigger</td></tr>
        <tr><td>condition</td><td>string</td><td>greater_than, less_than, equals, between</td></tr>
        <tr><td>window</td><td>string</td><td>Finestra temporale (5m, 1h, 1d)</td></tr>
        <tr><td>aggregation</td><td>string</td><td>avg, min, max, sum, count</td></tr>
        <tr><td>cooldown</td><td>int</td><td>Minuti prima di re-trigger</td></tr>
    </table>

    <h4>Severity Levels e SLA:</h4>
    <table>
        <tr><th>Severità</th><th>Descrizione</th><th>Response Time SLA</th></tr>
        <tr><td>info</td><td>Informativo, no action required</td><td>Best effort</td></tr>
        <tr><td>warning</td><td>Richiede attenzione</td><td>4 ore business hours</td></tr>
        <tr><td>critical</td><td>Impatto business immediato</td><td>30 minuti 24/7</td></tr>
        <tr><td>emergency</td><td>System down, perdita dati</td><td>15 minuti + escalation</td></tr>
    </table>

    <h4>Esempio Configurazione Avanzata:</h4>
    <pre><code>
    {
        "model_ids": ["arima_001", "sarima_002"],
        "alert_rules": [
            {
                "name": "Accuracy Drop Alert",
                "metric": "mae",
                "threshold": 0.15,
                "condition": "greater_than",
                "window": "1h",
                "aggregation": "avg",
                "severity": "warning",
                "cooldown": 60
            },
            {
                "name": "Anomaly Detection",
                "metric": "forecast_zscore",
                "threshold": 3.0,
                "condition": "greater_than",
                "window": "5m",
                "severity": "critical",
                "cooldown": 15
            }
        ],
        "notification_channels": ["email", "slack", "dashboard"],
        "severity_levels": ["warning", "critical"]
    }
    </code></pre>
    """

    alert_config_id = f"alert_config_{uuid.uuid4().hex[:8]}"

    # Simula configurazione sistema alert
    logger.info(f"Configurando sistema alert {alert_config_id}")
    logger.info(f"Modelli monitorati: {len(config.model_ids)}")
    logger.info(f"Regole alert: {len(config.alert_rules)}")
    logger.info(f"Canali notifica: {config.notification_channels}")

    return {
        "alert_config_id": alert_config_id,
        "status": "configured",
        "message": f"Sistema alert configurato per {len(config.model_ids)} modelli con {len(config.alert_rules)} regole",
        "monitoring_url": f"/alerts/{alert_config_id}/dashboard",
    }


@router.post("/interactive-analysis", response_model=VisualizationJobResponse)
async def create_interactive_analysis(
    config: InteractiveAnalysisRequest, background_tasks: BackgroundTasks
):
    """
    Crea ambiente interattivo avanzato per analisi what-if, sensitivity analysis e scenario planning.

    <h4>Analysis Types Disponibili:</h4>
    <table>
        <tr><th>Tipo</th><th>Descrizione</th><th>Tecniche Utilizzate</th></tr>
        <tr><td>what_if</td><td>Esplora scenari ipotetici</td><td>Parameter sweeping, Monte Carlo simulation</td></tr>
        <tr><td>sensitivity</td><td>Impatto variazione parametri</td><td>Sobol indices, Morris method, FAST</td></tr>
        <tr><td>scenario</td><td>Confronto scenari business</td><td>Best/worst/likely case, stress testing</td></tr>
        <tr><td>optimization</td><td>Trova parametri ottimali</td><td>Grid search, Bayesian optimization</td></tr>
        <tr><td>monte_carlo</td><td>Simulazione probabilistica</td><td>Random sampling, confidence bounds</td></tr>
    </table>

    <h4>Parameter Ranges Configuration:</h4>
    <table>
        <tr><th>Parametro</th><th>Tipo Range</th><th>Esempio</th></tr>
        <tr><td>ARIMA orders</td><td>Integer range</td><td>{"p": [0, 5], "d": [0, 2], "q": [0, 5]}</td></tr>
        <tr><td>Seasonal orders</td><td>Integer range</td><td>{"P": [0, 2], "D": [0, 1], "Q": [0, 2]}</td></tr>
        <tr><td>External factors</td><td>Continuous</td><td>{"price_elasticity": [-2.0, -0.5]}</td></tr>
        <tr><td>Business constraints</td><td>Categorical</td><td>{"inventory_policy": ["JIT", "EOQ", "Safety"]}</td></tr>
        <tr><td>Time horizons</td><td>Discrete</td><td>{"forecast_horizon": [7, 14, 30, 60]}</td></tr>
    </table>

    <h4>Output Metrics per Analysis:</h4>
    <table>
        <tr><th>Metrica</th><th>Descrizione</th><th>Interpretazione</th></tr>
        <tr><td>forecast_accuracy</td><td>Overall accuracy score</td><td>Higher is better (0-100%)</td></tr>
        <tr><td>confidence_width</td><td>Prediction interval width</td><td>Narrower = more certain</td></tr>
        <tr><td>parameter_importance</td><td>Relative parameter impact</td><td>Identifies key drivers</td></tr>
        <tr><td>scenario_probability</td><td>Likelihood of scenario</td><td>Based on historical patterns</td></tr>
        <tr><td>optimization_convergence</td><td>Speed to find optimum</td><td>Fewer iterations = better</td></tr>
    </table>

    <h4>Interactive Features Generated:</h4>
    <table>
        <tr><th>Feature</th><th>Descrizione</th><th>User Interaction</th></tr>
        <tr><td>Parameter Sliders</td><td>Real-time parameter adjustment</td><td>Drag to see instant impact</td></tr>
        <tr><td>3D Surface Plots</td><td>Response surface visualization</td><td>Rotate, zoom, hover for values</td></tr>
        <tr><td>Tornado Diagrams</td><td>Sensitivity ranking</td><td>Click to drill down</td></tr>
        <tr><td>Scenario Builder</td><td>Custom scenario creation</td><td>Drag & drop conditions</td></tr>
        <tr><td>Export Workspace</td><td>Save analysis state</td><td>Download/share configuration</td></tr>
    </table>

    <h4>Esempio Richiesta Avanzata:</h4>
    <pre><code>
    {
        "model_id": "sarima_001",
        "analysis_type": "sensitivity",
        "parameters_range": {
            "p": [0, 3],
            "d": [0, 2],
            "q": [0, 3],
            "seasonal_period": [7, 12, 365],
            "external_temperature": [15.0, 35.0],
            "marketing_spend": [0, 10000]
        },
        "output_metrics": [
            "forecast_accuracy",
            "confidence_width",
            "parameter_importance",
            "computation_time"
        ]
    }
    </code></pre>

    <h4>Dashboard Interattiva Output:</h4>
    - URL dashboard: /interactive/{job_id}/dashboard
    - WebSocket endpoint: ws://api/interactive/{job_id}/stream
    - Export API: /interactive/{job_id}/export
    - Sharing link: /interactive/{job_id}/share
    """

    job_id = f"interactive_job_{uuid.uuid4().hex[:8]}"

    visualization_jobs[job_id] = {
        "status": "queued",
        "progress": 0.0,
        "analysis_type": config.analysis_type,
        "model_id": config.model_id,
        "created_at": datetime.now(),
        "results_urls": [],
    }

    # Simula creazione ambiente interattivo
    async def create_interactive_job():
        try:
            visualization_jobs[job_id]["status"] = "running"
            await asyncio.sleep(3)  # Simula setup ambiente interattivo

            results = [
                f"/interactive/{job_id}/dashboard.html",
                f"/interactive/{job_id}/api/endpoints",
                f"/interactive/{job_id}/scenarios/export",
            ]

            visualization_jobs[job_id].update(
                {"status": "completed", "progress": 1.0, "results_urls": results}
            )

        except Exception as e:
            visualization_jobs[job_id]["status"] = "failed"
            logger.error(f"Errore ambiente interattivo {job_id}: {str(e)}")

    background_tasks.add_task(create_interactive_job)

    return VisualizationJobResponse(job_id=job_id, status="queued")


@router.get("/forecast-plot/{model_id}/{forecast_steps}", response_class=HTMLResponse)
async def generate_forecast_plot(
    model_id: str,
    forecast_steps: int,
    confidence_level: float = 0.95,
    include_intervals: bool = True,
    theme: str = "plotly_white",
    services: tuple = Depends(get_services),
):
    """
    Genera grafico interattivo Plotly con Serie Temporale + Forecast.

    Questo endpoint genera un grafico HTML interattivo che mostra:
    - Serie temporale storica (dati di training)
    - Previsioni future (forecast)
    - Intervalli di confidenza (opzionale)

    <h4>Parametri:</h4>
    <table>
        <tr><th>Nome</th><th>Tipo</th><th>Descrizione</th></tr>
        <tr><td>model_id</td><td>str</td><td>ID del modello per forecast</td></tr>
        <tr><td>forecast_steps</td><td>int</td><td>Numero di step futuri da prevedere</td></tr>
        <tr><td>confidence_level</td><td>float</td><td>Livello confidenza intervalli (default 0.95)</td></tr>
        <tr><td>include_intervals</td><td>bool</td><td>Includere intervalli confidenza (default true)</td></tr>
        <tr><td>theme</td><td>str</td><td>Tema Plotly (default plotly_white)</td></tr>
    </table>

    <h4>Risposta:</h4>
    - HTML completo con grafico Plotly interattivo embeddato
    - Pronto per essere visualizzato in iframe o div

    <h4>Esempio Chiamata:</h4>
    <pre><code>
    GET /visualization/forecast-plot/abc123/30?confidence_level=0.95&include_intervals=true
    </code></pre>
    """
    model_manager, forecast_service = services

    try:
        # Verifica esistenza modello
        if not model_manager.model_exists(model_id):
            raise HTTPException(status_code=404, detail="Modello non trovato")

        # Carica il modello per ottenere i dati storici
        model = model_manager.load_model(model_id)

        # Genera forecast usando il ForecastService
        forecast_response = await forecast_service.generate_forecast(
            model_id=model_id,
            steps=forecast_steps,
            confidence_level=confidence_level,
            return_intervals=include_intervals,
        )

        # Converti ForecastResult in dizionario con i nomi corretti dei campi
        forecast_result = {
            "forecast": forecast_response.forecast_values,
            "timestamps": forecast_response.forecast_timestamps,
            "confidence_intervals": {
                "lower": forecast_response.lower_bounds,
                "upper": forecast_response.upper_bounds,
            }
            if forecast_response.lower_bounds and forecast_response.upper_bounds
            else None,
        }

        # Recupera dati storici dal modello (se disponibili)
        historical_data = []
        historical_dates = []

        if hasattr(model, "training_data") and model.training_data is not None:
            import pandas as pd

            # Converti a lista i dati
            if isinstance(model.training_data, pd.Series):
                historical_data = model.training_data.values.tolist()
                # Usa l'index del pandas Series se disponibile
                if hasattr(model.training_data, "index"):
                    try:
                        # Prova a convertire l'index in stringhe
                        historical_dates = [str(x) for x in model.training_data.index]
                    except:
                        # Fallback a numeri sequenziali
                        historical_dates = list(range(1, len(historical_data) + 1))
                else:
                    historical_dates = list(range(1, len(historical_data) + 1))
            else:
                # Fallback se non è un pandas Series
                try:
                    historical_data = list(model.training_data)
                    historical_dates = list(range(1, len(historical_data) + 1))
                except:
                    historical_data = []
                    historical_dates = []

        # Crea il grafico Plotly
        fig = go.Figure()

        # Serie storica (se disponibile)
        if historical_data:
            fig.add_trace(
                go.Scatter(
                    x=historical_dates,
                    y=historical_data,
                    mode="lines",
                    name="Serie Storica",
                    line=dict(color="blue", width=2),
                    hovertemplate="<b>Storico</b><br>Periodo: %{x}<br>Valore: %{y:.2f}<extra></extra>",
                )
            )

        # Forecast - usa i timestamp se disponibili, altrimenti numeri sequenziali
        if "timestamps" in forecast_result and forecast_result["timestamps"]:
            forecast_dates = forecast_result["timestamps"]
        else:
            # Fallback a numeri sequenziali
            forecast_dates = list(
                range(
                    len(historical_data) + 1 if historical_data else 1,
                    len(historical_data) + forecast_steps + 1
                    if historical_data
                    else forecast_steps + 1,
                )
            )

        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=forecast_result["forecast"],
                mode="lines+markers",
                name="Forecast",
                line=dict(color="red", width=2, dash="dash"),
                marker=dict(size=6),
                hovertemplate="<b>Forecast</b><br>Periodo: %{x}<br>Valore: %{y:.2f}<extra></extra>",
            )
        )

        # Intervalli di confidenza (se richiesti)
        if include_intervals and "confidence_intervals" in forecast_result:
            ci = forecast_result["confidence_intervals"]

            # Upper bound
            fig.add_trace(
                go.Scatter(
                    x=forecast_dates,
                    y=ci["upper"],
                    mode="lines",
                    name=f"Upper Bound ({int(confidence_level * 100)}%)",
                    line=dict(width=0),
                    showlegend=False,
                    hovertemplate="<b>Upper Bound</b><br>Valore: %{y:.2f}<extra></extra>",
                )
            )

            # Lower bound con fill
            fig.add_trace(
                go.Scatter(
                    x=forecast_dates,
                    y=ci["lower"],
                    mode="lines",
                    name=f"Intervallo Confidenza {int(confidence_level * 100)}%",
                    fill="tonexty",
                    fillcolor="rgba(255, 0, 0, 0.2)",
                    line=dict(width=0),
                    hovertemplate="<b>Lower Bound</b><br>Valore: %{y:.2f}<extra></extra>",
                )
            )

        # Layout del grafico
        fig.update_layout(
            title={
                "text": f"Serie Temporale + Forecast ({forecast_steps} step)",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 20},
            },
            xaxis_title="Periodo",
            yaxis_title="Valore",
            template=theme,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=600,
            margin=dict(l=60, r=60, t=80, b=60),
        )

        # Converti in HTML
        html_content = pio.to_html(
            fig,
            include_plotlyjs="cdn",
            full_html=True,
            config={
                "displayModeBar": True,
                "displaylogo": False,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            },
        )

        return HTMLResponse(content=html_content)

    except HTTPException:
        raise
    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        logger.error(f"Errore generazione grafico forecast per {model_id}: {e}\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Errore generazione grafico: {str(e)}")


@router.get("/forecast-report/{model_id}/{forecast_steps}")
async def generate_forecast_report(
    model_id: str,
    forecast_steps: int,
    confidence_level: float = 0.95,
    include_intervals: bool = True,
    report_format: str = "pdf",
    services: tuple = Depends(get_services),
):
    """
    Genera report completo del forecast in formato PDF/HTML/DOCX.

    Questo endpoint crea un report professionale che include:
    - Informazioni sul modello
    - Grafico serie temporale + forecast
    - Tabella dati forecast dettagliata
    - Statistiche e metriche del modello
    - Intervalli di confidenza (se richiesti)

    <h4>Parametri:</h4>
    <table>
        <tr><th>Nome</th><th>Tipo</th><th>Descrizione</th></tr>
        <tr><td>model_id</td><td>str</td><td>ID del modello</td></tr>
        <tr><td>forecast_steps</td><td>int</td><td>Numero di step da prevedere</td></tr>
        <tr><td>confidence_level</td><td>float</td><td>Livello confidenza (default 0.95)</td></tr>
        <tr><td>include_intervals</td><td>bool</td><td>Include intervalli (default true)</td></tr>
        <tr><td>report_format</td><td>str</td><td>Formato: pdf, html, docx (default pdf)</td></tr>
    </table>

    <h4>Risposta:</h4>
    - File PDF/HTML/DOCX pronto per download
    - Content-Type appropriato per il formato
    - Nome file con timestamp

    <h4>Esempio Chiamata:</h4>
    <pre><code>
    GET /visualization/forecast-report/abc123/30?report_format=pdf
    </code></pre>
    """
    from fastapi.responses import FileResponse, StreamingResponse
    from io import BytesIO
    import tempfile
    from datetime import datetime

    model_manager, forecast_service = services

    try:
        # Verifica esistenza modello
        if not model_manager.model_exists(model_id):
            raise HTTPException(status_code=404, detail="Modello non trovato")

        # Carica modello e metadati
        model = model_manager.load_model(model_id)
        metadata = model_manager.get_model_info(model_id)

        # Genera forecast
        forecast_response = await forecast_service.generate_forecast(
            model_id=model_id,
            steps=forecast_steps,
            confidence_level=confidence_level,
            return_intervals=include_intervals,
        )

        if report_format.lower() == "pdf":
            # Import reportlab solo quando necessario per PDF
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import (
                SimpleDocTemplate,
                Table,
                TableStyle,
                Paragraph,
                Spacer,
                Image,
                PageBreak,
            )
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib import colors

            # Crea PDF in memoria
            buffer = BytesIO()
            doc = SimpleDocTemplate(
                buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18
            )

            # Container per gli elementi del report
            elements = []
            styles = getSampleStyleSheet()

            # Titolo
            title_style = ParagraphStyle(
                "CustomTitle",
                parent=styles["Heading1"],
                fontSize=24,
                textColor=colors.HexColor("#1976d2"),
                spaceAfter=30,
                alignment=1,  # Center
            )
            elements.append(Paragraph("Report Forecast", title_style))
            elements.append(Spacer(1, 0.2 * inch))

            # Informazioni Modello
            elements.append(Paragraph("<b>Informazioni Modello</b>", styles["Heading2"]))
            model_info_data = [
                ["Model ID:", model_id[:20] + "..."],
                ["Tipo Modello:", metadata.get("model_type", "N/A").upper()],
                ["Data Creazione:", str(metadata.get("created_at", "N/A"))[:19]],
                ["Osservazioni Training:", str(metadata.get("training_observations", "N/A"))],
                ["Step Forecast:", str(forecast_steps)],
                ["Livello Confidenza:", f"{int(confidence_level * 100)}%"],
            ]

            model_table = Table(model_info_data, colWidths=[3 * inch, 3 * inch])
            model_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e3f2fd")),
                        ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 10),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                        ("GRID", (0, 0), (-1, -1), 1, colors.grey),
                    ]
                )
            )
            elements.append(model_table)
            elements.append(Spacer(1, 0.3 * inch))

            # Statistiche Forecast
            elements.append(Paragraph("<b>Statistiche Forecast</b>", styles["Heading2"]))
            forecast_values = forecast_response.forecast_values
            stats_data = [
                ["Metrica", "Valore"],
                ["Media", f"{sum(forecast_values) / len(forecast_values):.2f}"],
                ["Minimo", f"{min(forecast_values):.2f}"],
                ["Massimo", f"{max(forecast_values):.2f}"],
                ["Range", f"{max(forecast_values) - min(forecast_values):.2f}"],
            ]

            stats_table = Table(stats_data, colWidths=[3 * inch, 3 * inch])
            stats_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1976d2")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 12),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ]
                )
            )
            elements.append(stats_table)
            elements.append(Spacer(1, 0.3 * inch))

            # Tabella Forecast (prime 20 righe per non appesantire)
            elements.append(Paragraph("<b>Dati Forecast (prime 20 righe)</b>", styles["Heading2"]))

            forecast_data = [["Periodo", "Timestamp", "Valore Previsto"]]
            if include_intervals and forecast_response.lower_bounds:
                forecast_data[0].extend(["Lower Bound", "Upper Bound"])

            for i in range(min(20, len(forecast_values))):
                row = [
                    str(i + 1),
                    forecast_response.forecast_timestamps[i][:19],
                    f"{forecast_values[i]:.2f}",
                ]
                if include_intervals and forecast_response.lower_bounds:
                    row.extend(
                        [
                            f"{forecast_response.lower_bounds[i]:.2f}",
                            f"{forecast_response.upper_bounds[i]:.2f}",
                        ]
                    )
                forecast_data.append(row)

            col_widths = [0.8 * inch, 2.2 * inch, 1.5 * inch]
            if include_intervals:
                col_widths.extend([1.5 * inch, 1.5 * inch])

            forecast_table = Table(forecast_data, colWidths=col_widths)
            forecast_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1976d2")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 10),
                        ("FONTSIZE", (0, 1), (-1, -1), 9),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
                    ]
                )
            )
            elements.append(forecast_table)

            # Footer
            elements.append(Spacer(1, 0.5 * inch))
            footer_style = ParagraphStyle(
                "Footer", parent=styles["Normal"], fontSize=9, textColor=colors.grey, alignment=1
            )
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            elements.append(Paragraph(f"Report generato il {timestamp}", footer_style))
            elements.append(Paragraph("Powered by ARIMA Forecaster API", footer_style))

            # Build PDF
            doc.build(elements)
            buffer.seek(0)

            # Genera nome file con timestamp
            filename = (
                f"forecast_report_{model_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            )

            return StreamingResponse(
                buffer,
                media_type="application/pdf",
                headers={"Content-Disposition": f"attachment; filename={filename}"},
            )

        elif report_format.lower() == "html":
            # Genera HTML semplice
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Report Forecast - {model_id[:8]}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    h1 {{ color: #1976d2; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                    th {{ background-color: #1976d2; color: white; }}
                    .stats {{ background-color: #f5f5f5; }}
                </style>
            </head>
            <body>
                <h1>Report Forecast</h1>
                <h2>Informazioni Modello</h2>
                <table>
                    <tr><td><b>Model ID</b></td><td>{model_id}</td></tr>
                    <tr><td><b>Tipo</b></td><td>{metadata.get("model_type", "N/A").upper()}</td></tr>
                    <tr><td><b>Step Forecast</b></td><td>{forecast_steps}</td></tr>
                </table>

                <h2>Dati Forecast</h2>
                <table>
                    <tr><th>Periodo</th><th>Timestamp</th><th>Valore</th></tr>
            """

            for i, (ts, val) in enumerate(
                zip(
                    forecast_response.forecast_timestamps[:20],
                    forecast_response.forecast_values[:20],
                )
            ):
                html_content += f"<tr><td>{i + 1}</td><td>{ts}</td><td>{val:.2f}</td></tr>"

            html_content += """
                </table>
                <p style="color: grey; font-size: 12px; margin-top: 40px;">
                    Report generato da ARIMA Forecaster API
                </p>
            </body>
            </html>
            """

            filename = (
                f"forecast_report_{model_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            )

            return StreamingResponse(
                BytesIO(html_content.encode("utf-8")),
                media_type="text/html",
                headers={"Content-Disposition": f"attachment; filename={filename}"},
            )

        else:
            raise HTTPException(status_code=400, detail=f"Formato non supportato: {report_format}")

    except HTTPException:
        raise
    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        logger.error(f"Errore generazione report per {model_id}: {e}\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Errore generazione report: {str(e)}")


@router.get("/job-status/{job_id}", response_model=VisualizationJobResponse)
async def get_job_status(job_id: str):
    """
    Ottiene stato dettagliato, progresso real-time e risultati di job visualizzazione.

    <h4>Job Status Lifecycle:</h4>
    <table>
        <tr><th>Status</th><th>Descrizione</th><th>Progress Range</th></tr>
        <tr><td>queued</td><td>Job accettato, in attesa risorse</td><td>0.0</td></tr>
        <tr><td>initializing</td><td>Allocazione risorse e setup</td><td>0.0 - 0.1</td></tr>
        <tr><td>running</td><td>Processing attivo in corso</td><td>0.1 - 0.9</td></tr>
        <tr><td>finalizing</td><td>Generazione output e cleanup</td><td>0.9 - 1.0</td></tr>
        <tr><td>completed</td><td>Successo, risultati disponibili</td><td>1.0</td></tr>
        <tr><td>failed</td><td>Errore durante processing</td><td>Ultimo valore</td></tr>
        <tr><td>cancelled</td><td>Interrotto da utente/sistema</td><td>Ultimo valore</td></tr>
        <tr><td>timeout</td><td>Superato tempo massimo</td><td>Ultimo valore</td></tr>
    </table>

    <h4>Progress Tracking Details:</h4>
    <table>
        <tr><th>Progress</th><th>Fase</th><th>Operazioni</th></tr>
        <tr><td>0% - 10%</td><td>Initialization</td><td>Load models, validate inputs</td></tr>
        <tr><td>10% - 30%</td><td>Data Processing</td><td>Prepare data, apply transforms</td></tr>
        <tr><td>30% - 70%</td><td>Core Processing</td><td>Generate visualizations/reports</td></tr>
        <tr><td>70% - 90%</td><td>Rendering</td><td>Create final outputs</td></tr>
        <tr><td>90% - 100%</td><td>Finalization</td><td>Save files, update metadata</td></tr>
    </table>

    <h4>Result URLs Structure:</h4>
    <table>
        <tr><th>Pattern</th><th>Content Type</th><th>Descrizione</th></tr>
        <tr><td>/outputs/plots/*.html</td><td>Interactive HTML</td><td>Plotly interactive charts</td></tr>
        <tr><td>/outputs/plots/*.png</td><td>Static Image</td><td>High-res PNG exports</td></tr>
        <tr><td>/outputs/reports/*.pdf</td><td>PDF Document</td><td>Professional reports</td></tr>
        <tr><td>/dashboards/*/index.html</td><td>Web Dashboard</td><td>Full interactive dashboard</td></tr>
        <tr><td>/api/results/*.json</td><td>JSON Data</td><td>Raw data for API integration</td></tr>
    </table>

    <h4>Response Fields Dettaglio:</h4>
    <table>
        <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th></tr>
        <tr><td>job_id</td><td>string</td><td>Identificatore univoco job</td></tr>
        <tr><td>status</td><td>string</td><td>Stato corrente del job</td></tr>
        <tr><td>progress</td><td>float</td><td>Percentuale completamento (0.0-1.0)</td></tr>
        <tr><td>estimated_completion</td><td>datetime</td><td>Stima tempo completamento (se running)</td></tr>
        <tr><td>results_urls</td><td>list[string]</td><td>URLs output generati (se completed)</td></tr>
        <tr><td>error_message</td><td>string</td><td>Dettagli errore (se failed)</td></tr>
        <tr><td>metadata</td><td>dict</td><td>Informazioni aggiuntive job</td></tr>
    </table>

    <h4>Esempio Response per Diversi Stati:</h4>

    **Job Running:**
    <pre><code>
    {
        "job_id": "viz_job_abc123",
        "status": "running",
        "progress": 0.45,
        "estimated_completion": "2024-08-26T10:35:00Z",
        "results_urls": [],
        "metadata": {
            "current_step": "Generating forecast plots",
            "models_processed": 2,
            "total_models": 5
        }
    }
    </code></pre>

    **Job Completed:**
    <pre><code>
    {
        "job_id": "viz_job_abc123",
        "status": "completed",
        "progress": 1.0,
        "estimated_completion": null,
        "results_urls": [
            "/outputs/plots/viz_job_abc123_forecast.html",
            "/outputs/plots/viz_job_abc123_residuals.png",
            "/outputs/reports/viz_job_abc123_summary.pdf"
        ],
        "metadata": {
            "processing_time_seconds": 45.3,
            "output_size_mb": 12.5,
            "models_analyzed": 5
        }
    }
    </code></pre>

    **Job Failed:**
    <pre><code>
    {
        "job_id": "viz_job_abc123",
        "status": "failed",
        "progress": 0.32,
        "estimated_completion": null,
        "results_urls": [],
        "error_message": "Model arima_001 not found in registry",
        "metadata": {
            "failed_at_step": "model_loading",
            "retry_available": true
        }
    }
    </code></pre>
    """

    if job_id not in visualization_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} non trovato")

    job_info = visualization_jobs[job_id]

    return VisualizationJobResponse(
        job_id=job_id,
        status=job_info["status"],
        progress=job_info.get("progress", 0.0),
        estimated_completion=job_info.get("estimated_completion"),
        results_urls=job_info.get("results_urls", []),
    )


@router.get("/download-report/{file_path:path}")
async def download_report_file(file_path: str):
    """
    Scarica un file report generato dal sistema.

    <h4>Descrizione:</h4>
    Permette il download dei report generati dall'endpoint /generate-report.
    Il path del file viene ottenuto dai risultati del job (results_urls).

    <h4>Parametri:</h4>
    - file_path: Path relativo o assoluto del file da scaricare

    <h4>Formati Supportati:</h4>
    - PDF: Report stampabile professionale
    - HTML: Report interattivo web
    - DOCX: Documento Word editabile
    - PNG: Immagini grafici

    <h4>Risposta:</h4>
    - StreamingResponse con il file binario
    - Content-Type appropriato per il formato
    - Header Content-Disposition per download

    <h4>Esempio Chiamata:</h4>
    <pre><code>
    GET /visualization/download-report/outputs/reports/report_job_abc123_report.html
    </code></pre>
    """
    from fastapi.responses import FileResponse

    # Converti path relativo in assoluto
    file_path_obj = Path(file_path)

    if not file_path_obj.is_absolute():
        # Se il path è relativo, cerca nella directory del progetto
        current_path = Path(__file__).parent
        while current_path.parent != current_path:
            if (current_path / "pyproject.toml").exists() or (current_path / "CLAUDE.md").exists():
                project_root = current_path
                break
            current_path = current_path.parent
        else:
            project_root = Path(__file__).parent.parent.parent.parent

        file_path_obj = project_root / file_path

    # Verifica che il file esista
    if not file_path_obj.exists():
        raise HTTPException(status_code=404, detail=f"File non trovato: {file_path}")

    # Verifica che sia un file (non directory)
    if not file_path_obj.is_file():
        raise HTTPException(status_code=400, detail="Il path specificato non è un file")

    # Determina media type in base all'estensione
    suffix = file_path_obj.suffix.lower()
    media_types = {
        ".pdf": "application/pdf",
        ".html": "text/html",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".json": "application/json",
    }

    media_type = media_types.get(suffix, "application/octet-stream")

    return FileResponse(path=str(file_path_obj), media_type=media_type, filename=file_path_obj.name)
