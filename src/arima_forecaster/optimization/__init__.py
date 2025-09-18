"""
Performance Optimization Module.

Moduli per ottimizzazione performance del sistema ARIMA Forecaster.
Include caching, memory pooling, parallel processing e algoritmi ottimizzati.
"""

from .model_cache import ModelCache, get_model_cache, configure_cache, get_smart_starting_params
from .memory_pool import (
    MemoryPool,
    VectorizedOps,
    get_memory_pool,
    configure_memory_pool,
    ManagedArray,
)
from .benchmark import BenchmarkConfig, BenchmarkResult, DatasetGenerator, PerformanceBenchmark

__all__ = [
    "ModelCache",
    "get_model_cache",
    "configure_cache",
    "get_smart_starting_params",
    "MemoryPool",
    "VectorizedOps",
    "get_memory_pool",
    "configure_memory_pool",
    "ManagedArray",
    "BenchmarkConfig",
    "BenchmarkResult",
    "DatasetGenerator",
    "PerformanceBenchmark",
]
