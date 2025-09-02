"""Core ARIMA modeling functionality."""

from .arima_model import ARIMAForecaster
from .model_selection import ARIMAModelSelector
from .sarima_model import SARIMAForecaster
from .sarima_selection import SARIMAModelSelector
from .sarimax_model import SARIMAXForecaster
from .sarimax_selection import SARIMAXModelSelector
from .var_model import VARForecaster
from .sarimax_auto_selector import SARIMAXAutoSelector
from .prophet_model import ProphetForecaster
from .prophet_selection import ProphetModelSelector
from .intermittent_model import IntermittentForecaster, IntermittentConfig, IntermittentMethod

# GPU-accelerated selectors (optional import)
try:
    from .gpu_model_selector import GPUARIMAModelSelector, GPUSARIMAModelSelector
    _gpu_available = True
except ImportError:
    _gpu_available = False

_base_exports = [
    "ARIMAForecaster", 
    "ARIMAModelSelector", 
    "SARIMAForecaster", 
    "SARIMAModelSelector",
    "SARIMAXForecaster",
    "SARIMAXModelSelector",
    "SARIMAXAutoSelector",  # Advanced Exog Handling
    "VARForecaster",
    "ProphetForecaster",     # Facebook Prophet
    "ProphetModelSelector",   # Prophet Auto-Selection
    "IntermittentForecaster",  # Intermittent Demand (Croston, SBA, TSB)
    "IntermittentConfig",     # Configuration for Intermittent
    "IntermittentMethod"      # Available methods enum
]

# Add GPU selectors if available
_gpu_exports = []
if _gpu_available:
    _gpu_exports = [
        "GPUARIMAModelSelector",   # GPU-accelerated ARIMA
        "GPUSARIMAModelSelector"   # GPU-accelerated SARIMA
    ]

__all__ = _base_exports + _gpu_exports