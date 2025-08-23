"""Utility functions and classes."""

from .logger import setup_logger, get_logger
from .exceptions import (
    ARIMAForecasterError,
    DataProcessingError, 
    ModelTrainingError,
    ForecastError
)

try:
    from .preprocessing import ExogenousPreprocessor, validate_exog_data, suggest_preprocessing_method
    _preprocessing_available = True
except ImportError:
    _preprocessing_available = False

__all__ = [
    "setup_logger",
    "get_logger", 
    "ARIMAForecasterError",
    "DataProcessingError",
    "ModelTrainingError", 
    "ForecastError"
]

if _preprocessing_available:
    __all__.extend([
        "ExogenousPreprocessor",
        "validate_exog_data", 
        "suggest_preprocessing_method"
    ])