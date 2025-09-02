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
from .inventory import router as inventory_router
from .demand_sensing import router as demand_sensing_router
from .advanced_models import router as advanced_models_router
from .evaluation import router as evaluation_router
from .automl import router as automl_router
from .visualization import router as visualization_router
from .data_management import router as data_management_router
from .enterprise import router as enterprise_router

__all__ = [
    'health_router',
    'models_router', 
    'training_router',
    'forecasting_router',
    'diagnostics_router',
    'reports_router',
    'inventory_router',
    'demand_sensing_router',
    'advanced_models_router',
    'evaluation_router',
    'automl_router',
    'visualization_router',
    'data_management_router',
    'enterprise_router'
]