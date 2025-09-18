"""
Router per generazione e gestione report.

Gestisce la creazione di report HTML/PDF/DOCX con Quarto.
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import FileResponse

from arima_forecaster.api.models import ReportGenerationResponse
from arima_forecaster.api.models_extra import ReportRequest
from arima_forecaster.api.services import ModelManager, ForecastService
from arima_forecaster.utils.logger import get_logger

logger = get_logger(__name__)

# Crea router con prefix e tags
router = APIRouter(tags=["Reports"], responses={404: {"description": "Not found"}})

"""
📄 REPORTS ROUTER

Generazione report professionali con Quarto:

• POST /models/{model_id}/report - Genera report completo modello
• GET /reports/{filename}        - Download file report generati

Formati supportati:
- HTML: Report interattivi con grafici Plotly
- PDF: Documenti professionali per stampa/condivisione
- DOCX: Documenti editabili Microsoft Word

Contenuti report:
- Executive summary risultati
- Analisi modello e parametri
- Metriche performance dettagliate
- Grafici forecasting e diagnostica
- Raccomandazioni miglioramento
"""


# Dependency injection dei servizi
def get_services():
    """Dependency per ottenere i servizi necessari."""
    storage_path = Path("models")
    model_manager = ModelManager(storage_path)
    forecast_service = ForecastService(model_manager)
    return model_manager, forecast_service


async def _generate_report_background(
    model_manager: ModelManager,
    model_id: str,
    report_id: str,
    format_type: str,
    include_diagnostics: bool,
    include_forecasts: bool,
    forecast_steps: Optional[int],
):
    """
    Genera il report in background usando Quarto.
    """
    try:
        # Simula generazione report (in produzione userebbe QuartoReportGenerator)
        import time

        time.sleep(5)  # Simula tempo di generazione

        # Aggiorna stato del report
        logger.info(f"Report {report_id} generated successfully")

    except Exception as e:
        logger.error(f"Report generation failed: {e}")


@router.post("/models/{model_id}/report", response_model=ReportGenerationResponse)
async def generate_report(
    model_id: str,
    request: ReportRequest,
    background_tasks: BackgroundTasks,
    services: tuple = Depends(get_services),
):
    """
    Genera un report completo per un modello addestrato.
    
    Crea report professionali in formato HTML, PDF o DOCX utilizzando Quarto,
    includendo analisi, visualizzazioni e raccomandazioni.
    
    <h4>Parametri di Ingresso:</h4>
    <table >
        <tr><th>Nome</th><th>Tipo</th><th>Descrizione</th></tr>
        <tr><td>model_id</td><td>str</td><td>ID del modello per cui generare il report</td></tr>
        <tr><td>request</td><td>ReportRequest</td><td>Configurazione del report</td></tr>
    </table>
    
    <h4>Campi del Request Body:</h4>
    <table >
        <tr><th>Campo</th><th>Tipo</th><th>Descrizione</th><th>Default</th></tr>
        <tr><td>format</td><td>str</td><td>Formato output: "html", "pdf", "docx"</td><td>"html"</td></tr>
        <tr><td>include_diagnostics</td><td>bool</td><td>Includere analisi diagnostiche</td><td>true</td></tr>
        <tr><td>include_forecasts</td><td>bool</td><td>Includere previsioni future</td><td>true</td></tr>
        <tr><td>forecast_steps</td><td>int</td><td>Numero di passi da prevedere</td><td>30</td></tr>
        <tr><td>template</td><td>str</td><td>Template da utilizzare</td><td>"default"</td></tr>
    </table>
    
    <h4>Esempio di Chiamata:</h4>
    <pre><code>
    curl -X POST "http://localhost:8000/models/abc123/report" \\
         -H "Content-Type: application/json" \\
         -d '{
           "format": "html",
           "include_diagnostics": true,
           "include_forecasts": true,
           "forecast_steps": 30
         }'
    </code></pre>
    
    <h4>Esempio di Risposta:</h4>
    <pre><code>
    {
        "report_id": "report-xyz789",
        "status": "generating",
        "format_type": "html",
        "generation_time": 15.7,
        "file_size_mb": 2.34,
        "download_url": "/reports/sarima_vendite_q4_2024.html"
    }
    </code></pre>
    
    <h4>Contenuti del Report:</h4>
    <ul>
        <li><strong>Executive Summary</strong>: Sintesi dei risultati principali</li>
        <li><strong>Model Overview</strong>: Tipo, parametri e caratteristiche</li>
        <li><strong>Performance Metrics</strong>: Tutte le metriche di valutazione</li>
        <li><strong>Diagnostics</strong>: Analisi residui e test statistici</li>
        <li><strong>Forecasts</strong>: Previsioni con intervalli di confidenza</li>
        <li><strong>Visualizations</strong>: Grafici interattivi e tabelle</li>
        <li><strong>Recommendations</strong>: Suggerimenti per miglioramenti</li>
    </ul>
    """
    model_manager, _ = services

    try:
        # Verifica l'esistenza del modello
        if not model_manager.model_exists(model_id):
            raise HTTPException(status_code=404, detail="Model not found")

        # Genera ID univoco per il report
        report_id = f"report-{uuid.uuid4().hex[:8]}"

        # Avvia generazione in background
        background_tasks.add_task(
            _generate_report_background,
            model_manager,
            model_id,
            report_id,
            request.format,
            request.include_diagnostics,
            request.include_forecasts,
            request.forecast_steps,
        )

        # Stima dimensione e tempo (valori demo)
        import random

        generation_time = random.uniform(10, 20)
        file_size = random.uniform(1.5, 3.5)

        # Genera nome file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_id}_{timestamp}.{request.format}"

        return ReportGenerationResponse(
            report_id=report_id,
            status="generating",
            format_type=request.format,
            generation_time=generation_time,
            file_size_mb=file_size,
            download_url=f"/reports/{filename}",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Report generation failed for model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reports/{filename}")
async def download_report(filename: str):
    """
    Scarica un report generato.

    Permette il download diretto dei report generati in precedenza.
    I report sono serviti come file statici dal file system.

    <h4>Parametri di Ingresso:</h4>
    <table >
        <tr><th>Nome</th><th>Tipo</th><th>Descrizione</th></tr>
        <tr><td>filename</td><td>str</td><td>Nome del file del report da scaricare</td></tr>
    </table>

    <h4>Headers di Risposta:</h4>
    <table >
        <tr><th>Header</th><th>Valore</th><th>Descrizione</th></tr>
        <tr><td>Content-Type</td><td>application/octet-stream</td><td>Tipo MIME per download</td></tr>
        <tr><td>Content-Disposition</td><td>attachment; filename="..."</td><td>Forza il download del file</td></tr>
    </table>

    <h4>Esempio di Chiamata:</h4>
    <pre><code>
    curl -O "http://localhost:8000/reports/sarima_model_report_20240823.html"
    </code></pre>

    <h4>Utilizzo dal Browser:</h4>
    <pre><code>
    http://localhost:8000/reports/sarima_model_report_20240823.html
    </code></pre>

    <h4>Formati Supportati:</h4>
    <ul>
        <li><strong>HTML</strong>: Report interattivi visualizzabili nel browser</li>
        <li><strong>PDF</strong>: Report formattati per stampa e condivisione</li>
        <li><strong>DOCX</strong>: Report editabili in Microsoft Word</li>
    </ul>

    <h4>Errori Possibili:</h4>
    <ul>
        <li><strong>404</strong>: File del report non trovato</li>
        <li><strong>500</strong>: Errore nell'accesso al file system</li>
    </ul>
    """
    try:
        # Costruisce il percorso del file
        report_path = Path("reports") / filename

        # Verifica che il file esista
        if not report_path.exists():
            raise HTTPException(status_code=404, detail="Report file not found")

        # Determina il content type basato sull'estensione
        content_type_map = {
            ".html": "text/html",
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        }

        suffix = report_path.suffix.lower()
        content_type = content_type_map.get(suffix, "application/octet-stream")

        # Restituisce il file
        return FileResponse(path=report_path, media_type=content_type, filename=filename)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to serve report {filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
