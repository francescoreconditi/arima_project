"""
Modelli Pydantic per validazione richieste/risposte API.
"""

from typing import List, Optional, Dict, Any, Union, Tuple
from pydantic import BaseModel, Field, validator
from datetime import datetime
import pandas as pd
from pathlib import Path


class TimeSeriesData(BaseModel):
    """Dati serie temporali per addestramento modello."""
    
    timestamps: List[str] = Field(..., description="List of timestamp strings")
    values: List[float] = Field(..., description="List of time series values")
    
    @validator('timestamps')
    def validate_timestamps(cls, v):
        if len(v) == 0:
            raise ValueError("Timestamps cannot be empty")
        return v
    
    @validator('values')
    def validate_values(cls, v, values):
        if 'timestamps' in values and len(v) != len(values['timestamps']):
            raise ValueError("Values and timestamps must have the same length")
        if len(v) == 0:
            raise ValueError("Values cannot be empty")
        return v


class MultivariateTimeSeriesData(BaseModel):
    """Dati serie temporali multivariate per modelli VAR."""
    
    timestamps: List[str] = Field(..., description="List of timestamp strings")
    data: Dict[str, List[float]] = Field(..., description="Dictionary of variable names to values")
    
    @validator('data')
    def validate_data(cls, v, values):
        if 'timestamps' in values:
            for var_name, var_values in v.items():
                if len(var_values) != len(values['timestamps']):
                    raise ValueError(f"All variables must have the same length as timestamps")
        if len(v) < 2:
            raise ValueError("Multivariate data must have at least 2 variables")
        return v


class ARIMAOrder(BaseModel):
    """Specifica ordine modello ARIMA."""
    
    p: int = Field(..., ge=0, le=5, description="AR order")
    d: int = Field(..., ge=0, le=2, description="Differencing order")
    q: int = Field(..., ge=0, le=5, description="MA order")


class SARIMAOrder(BaseModel):
    """Specifica ordine modello SARIMA."""
    
    p: int = Field(..., ge=0, le=5, description="AR order")
    d: int = Field(..., ge=0, le=2, description="Differencing order")
    q: int = Field(..., ge=0, le=5, description="MA order")
    P: int = Field(..., ge=0, le=2, description="Seasonal AR order")
    D: int = Field(..., ge=0, le=1, description="Seasonal differencing order")
    Q: int = Field(..., ge=0, le=2, description="Seasonal MA order")
    s: int = Field(..., ge=2, le=365, description="Seasonal period")


class ModelTrainingRequest(BaseModel):
    """Richiesta per addestramento modello."""
    
    data: TimeSeriesData
    model_type: str = Field(..., description="Type of model (arima, sarima)")
    order: Optional[ARIMAOrder] = None
    seasonal_order: Optional[SARIMAOrder] = None
    auto_select: bool = Field(default=False, description="Whether to automatically select model parameters")
    
    @validator('model_type')
    def validate_model_type(cls, v):
        if v not in ['arima', 'sarima']:
            raise ValueError("model_type must be 'arima' or 'sarima'")
        return v
    
    @validator('order')
    def validate_order(cls, v, values):
        if values.get('model_type') == 'arima' and not values.get('auto_select') and v is None:
            raise ValueError("ARIMA order is required when not using auto_select")
        return v
    
    @validator('seasonal_order') 
    def validate_seasonal_order(cls, v, values):
        if values.get('model_type') == 'sarima' and not values.get('auto_select') and v is None:
            raise ValueError("SARIMA seasonal_order is required when not using auto_select")
        return v


class VARTrainingRequest(BaseModel):
    """Richiesta per addestramento modello VAR."""
    
    data: MultivariateTimeSeriesData
    maxlags: Optional[int] = Field(None, ge=1, le=20, description="Maximum lags to consider")
    ic: str = Field(default='aic', description="Information criterion")
    
    @validator('ic')
    def validate_ic(cls, v):
        if v not in ['aic', 'bic', 'hqic', 'fpe']:
            raise ValueError("ic must be one of 'aic', 'bic', 'hqic', 'fpe'")
        return v


class ForecastRequest(BaseModel):
    """Richiesta per generazione previsioni."""
    
    model_id: str = Field(..., description="ID of the trained model")
    steps: int = Field(..., ge=1, le=100, description="Number of forecast steps")
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.99, description="Confidence level")
    return_intervals: bool = Field(default=True, description="Whether to return confidence intervals")


class ModelInfo(BaseModel):
    """Informazioni su modello addestrato."""
    
    model_id: str
    model_type: str
    status: str
    created_at: datetime
    training_observations: int
    parameters: Dict[str, Any]
    metrics: Dict[str, float]


class ForecastResult(BaseModel):
    """Risultato previsione."""
    
    model_id: str
    forecast_timestamps: List[str]
    forecast_values: List[float]
    lower_bounds: Optional[List[float]] = None
    upper_bounds: Optional[List[float]] = None
    confidence_level: Optional[float] = None
    generated_at: datetime


class VARForecastResult(BaseModel):
    """Risultato previsione VAR."""
    
    model_id: str
    forecast_timestamps: List[str]
    forecasts: Dict[str, List[float]]
    lower_bounds: Optional[Dict[str, List[float]]] = None
    upper_bounds: Optional[Dict[str, List[float]]] = None
    confidence_level: Optional[float] = None
    generated_at: datetime


class ErrorResponse(BaseModel):
    """Risposta errore."""
    
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None


class ModelListResponse(BaseModel):
    """Risposta per elenco modelli."""
    
    models: List[ModelInfo]
    total: int


class AutoSelectionRequest(BaseModel):
    """Richiesta per selezione automatica modello."""
    
    data: TimeSeriesData
    model_type: str = Field(..., description="Type of model (arima, sarima)")
    max_models: Optional[int] = Field(default=50, ge=1, le=200, description="Maximum models to test")
    information_criterion: str = Field(default='aic', description="Information criterion")
    
    @validator('model_type')
    def validate_model_type(cls, v):
        if v not in ['arima', 'sarima']:
            raise ValueError("model_type must be 'arima' or 'sarima'")
        return v
    
    @validator('information_criterion')
    def validate_ic(cls, v):
        if v not in ['aic', 'bic', 'hqic']:
            raise ValueError("information_criterion must be 'aic', 'bic', or 'hqic'")
        return v


class AutoSelectionResult(BaseModel):
    """Risultato selezione automatica modello."""
    
    best_model_id: str
    best_parameters: Dict[str, Any]
    best_score: float
    all_results: List[Dict[str, Any]]
    selection_time: float


class ModelDiagnosticsRequest(BaseModel):
    """Richiesta per diagnostica modello."""
    
    model_id: str = Field(..., description="ID of the trained model")
    include_residuals: bool = Field(default=True, description="Include residual analysis")
    include_acf_pacf: bool = Field(default=True, description="Include ACF/PACF plots")


class ModelDiagnostics(BaseModel):
    """Risultati diagnostica modello."""
    
    model_id: str
    residual_stats: Optional[Dict[str, float]] = None
    normality_test: Optional[Dict[str, float]] = None
    ljung_box_test: Optional[Dict[str, float]] = None
    heteroscedasticity_test: Optional[Dict[str, float]] = None
    acf_values: Optional[List[float]] = None
    pacf_values: Optional[List[float]] = None


class ReportGenerationRequest(BaseModel):
    """Richiesta per generazione report modello."""
    
    model_id: str = Field(..., description="ID of the trained model")
    report_title: Optional[str] = Field(default=None, description="Custom title for the report")
    output_filename: Optional[str] = Field(default=None, description="Custom filename for the report")
    format_type: str = Field(default="html", description="Output format (html, pdf, docx)")
    include_diagnostics: bool = Field(default=True, description="Include model diagnostics")
    include_forecast: bool = Field(default=True, description="Include forecast analysis")
    forecast_steps: int = Field(default=12, ge=1, le=100, description="Number of forecast steps")
    
    @validator('format_type')
    def validate_format_type(cls, v):
        if v not in ['html', 'pdf', 'docx']:
            raise ValueError("format_type must be 'html', 'pdf', or 'docx'")
        return v


class ReportGenerationResponse(BaseModel):
    """Risposta per generazione report."""
    
    model_id: str
    report_path: str
    format_type: str
    generation_time: float
    file_size_mb: Optional[float] = None
    download_url: Optional[str] = None