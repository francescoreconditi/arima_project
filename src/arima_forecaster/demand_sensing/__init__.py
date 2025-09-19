"""
Modulo Demand Sensing per ARIMA Forecaster.

Integra fattori esterni per migliorare l'accuratezza delle previsioni:
- Dati meteorologici
- Google Trends
- Social media sentiment
- Indicatori economici
- Eventi calendario
"""

from .demand_sensor import (
    DemandSensor,
    ExternalFactor,
    FactorImpact,
    SensingConfig,
    AdjustmentStrategy,
)
from .weather import WeatherIntegration, WeatherImpact
from .trends import GoogleTrendsIntegration, TrendImpact
from .social import SocialSentimentAnalyzer, SentimentImpact
from .economic import EconomicIndicators, EconomicImpact
from .calendar_events import CalendarEvents, EventImpact, Event, EventType
from .ensemble import EnsembleDemandSensor, EnsembleConfig

__all__ = [
    "DemandSensor",
    "ExternalFactor",
    "FactorImpact",
    "SensingConfig",
    "AdjustmentStrategy",
    "WeatherIntegration",
    "WeatherImpact",
    "GoogleTrendsIntegration",
    "TrendImpact",
    "SocialSentimentAnalyzer",
    "SentimentImpact",
    "EconomicIndicators",
    "EconomicImpact",
    "CalendarEvents",
    "EventImpact",
    "Event",
    "EventType",
    "EnsembleDemandSensor",
    "EnsembleConfig",
]
