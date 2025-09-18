"""REST API for ARIMA forecasting services."""

from .main import create_app
from .models import *

__all__ = ["create_app"]
