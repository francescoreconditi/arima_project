"""
ARIMA Forecaster Package

Libreria completa per il forecasting di serie temporali usando modelli ARIMA.
Fornisce strumenti per preprocessing dati, addestramento modelli, valutazione e visualizzazione.
"""

__version__ = "0.2.0"
__author__ = "Il Tuo Nome"

from .core.arima_model import ARIMAForecaster
from .data.preprocessor import TimeSeriesPreprocessor
from .evaluation.metrics import ModelEvaluator
from .visualization.plotter import ForecastPlotter

__all__ = [
    "ARIMAForecaster",
    "TimeSeriesPreprocessor", 
    "ModelEvaluator",
    "ForecastPlotter"
]