"""
Modelli Pydantic aggiuntivi per l'API REST.

Questo modulo contiene modelli creati per supportare i nuovi routers.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class VARModelInfo(BaseModel):
    """
    Informazioni su un modello VAR addestrato.

    <h4>Attributi:</h4>
    - **model_id**: ID univoco del modello
    - **model_type**: Tipo di modello (sempre "var")
    - **status**: Stato del modello
    - **created_at**: Data e ora di creazione
    - **variables**: Lista delle variabili nel modello
    - **max_lags**: Numero massimo di lag considerati
    - **selected_lag_order**: Ordine di lag selezionato automaticamente
    - **causality_tests**: Risultati dei test di causalità di Granger
    """

    model_id: str
    model_type: str = Field(default="var")
    status: str = Field(..., description="Stato: training, completed, failed")
    created_at: datetime
    variables: List[str] = Field(..., description="Nomi delle variabili nel modello")
    max_lags: int = Field(..., description="Numero massimo di lag considerati")
    selected_lag_order: Optional[int] = Field(None, description="Ordine di lag selezionato")
    causality_tests: Dict[str, Any] = Field(default_factory=dict)


class ForecastResponse(BaseModel):
    """
    Risposta contenente le previsioni generate.

    <h4>Attributi:</h4>
    - **forecast**: Lista dei valori previsti
    - **timestamps**: Lista dei timestamp per ogni previsione
    - **confidence_intervals**: Intervalli di confidenza (opzionale)
    - **model_id**: ID del modello utilizzato
    - **forecast_steps**: Numero di passi previsti
    """

    forecast: List[float]
    timestamps: List[str]
    confidence_intervals: Optional[Dict[str, List[float]]] = None
    model_id: str
    forecast_steps: int


class AutoSelectionResult(BaseModel):
    """
    Risultato della selezione automatica dei parametri.

    <h4>Attributi:</h4>
    - **best_model**: Parametri del modello migliore trovato
    - **all_models**: Lista di tutti i modelli testati
    - **search_time_seconds**: Tempo totale di ricerca in secondi
    """

    best_model: Dict[str, Any]
    all_models: List[Dict[str, Any]]
    search_time_seconds: float


class ModelDiagnostics(BaseModel):
    """
    Risultati della diagnostica del modello.

    <h4>Attributi:</h4>
    - **residuals_stats**: Statistiche sui residui
    - **ljung_box_test**: Test di Ljung-Box per autocorrelazione
    - **jarque_bera_test**: Test di Jarque-Bera per normalità
    - **acf_values**: Valori di autocorrelazione
    - **pacf_values**: Valori di autocorrelazione parziale
    - **performance_metrics**: Metriche di performance
    """

    residuals_stats: Dict[str, float]
    ljung_box_test: Dict[str, Any]
    jarque_bera_test: Dict[str, Any]
    acf_values: List[float]
    pacf_values: List[float]
    performance_metrics: Dict[str, float]


class ReportRequest(BaseModel):
    """
    Richiesta per generazione report.

    <h4>Attributi:</h4>
    - **format**: Formato del report (html, pdf, docx)
    - **include_diagnostics**: Se includere diagnostica
    - **include_forecasts**: Se includere previsioni
    - **forecast_steps**: Numero di passi da prevedere
    - **template**: Template da utilizzare
    """

    format: str = Field(default="html", description="Formato: html, pdf, docx")
    include_diagnostics: bool = Field(default=True)
    include_forecasts: bool = Field(default=True)
    forecast_steps: int = Field(default=30)
    template: str = Field(default="default")
