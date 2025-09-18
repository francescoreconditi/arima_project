"""
Configuration management per ARIMA Forecaster Library.
Gestisce settings globali, GPU configuration e environment variables.
"""

from .settings import ARIMAConfig, get_config
from .gpu_config import GPUBackend, detect_gpu_capability, get_gpu_config

__all__ = ["ARIMAConfig", "GPUBackend", "get_config", "detect_gpu_capability", "get_gpu_config"]
