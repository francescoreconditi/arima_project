"""
Routers per l'API FastAPI.

Questo package contiene i router modulari per organizzare gli endpoint dell'API.
"""

from .health import router as health_router
from .models import router as models_router
from .training import router as training_router
from .forecasting import router as forecasting_router
from .diagnostics import router as diagnostics_router
from .reports import router as reports_router

__all__ = [
    'health_router',
    'models_router', 
    'training_router',
    'forecasting_router',
    'diagnostics_router',
    'reports_router'
]