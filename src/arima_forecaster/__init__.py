"""
ARIMA Forecaster Package

Libreria completa per il forecasting di serie temporali usando modelli ARIMA, SARIMA e VAR.
Fornisce strumenti per preprocessing dati, addestramento modelli, valutazione, visualizzazione,
API REST, dashboard interattiva e ottimizzazione automatica dei parametri.
"""

__version__ = "0.3.0"
__author__ = "Il Tuo Nome"

# Core models
from .core.arima_model import ARIMAForecaster
from .core.sarima_model import SARIMAForecaster
from .core.var_model import VARForecaster
from .core.model_selection import ARIMAModelSelector
from .core.sarima_selection import SARIMAModelSelector

# Data handling
from .data.preprocessor import TimeSeriesPreprocessor
from .data.loader import DataLoader

# Evaluation and visualization
from .evaluation.metrics import ModelEvaluator
from .visualization.plotter import ForecastPlotter

# Auto-ML
from .automl.optimizer import ARIMAOptimizer, SARIMAOptimizer, VAROptimizer, optimize_model
from .automl.tuner import HyperparameterTuner

# Reporting (optional import - requires reports extra)
try:
    from .reporting.generator import QuartoReportGenerator
    _has_reporting = True
except ImportError:
    _has_reporting = False

__all__ = [
    # Core models
    "ARIMAForecaster",
    "SARIMAForecaster", 
    "VARForecaster",
    "ARIMAModelSelector",
    "SARIMAModelSelector",
    
    # Data handling
    "TimeSeriesPreprocessor", 
    "DataLoader",
    
    # Evaluation and visualization
    "ModelEvaluator",
    "ForecastPlotter",
    
    # Auto-ML
    "ARIMAOptimizer",
    "SARIMAOptimizer",
    "VAROptimizer",
    "HyperparameterTuner",
    "optimize_model"
]

# Add QuartoReportGenerator to __all__ if available
if _has_reporting:
    __all__.append("QuartoReportGenerator")