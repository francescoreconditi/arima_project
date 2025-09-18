"""
Test per modulo demand sensing - Import validation.

Test semplificati per verificare che tutti i moduli si importino correttamente.
"""

import pytest


def test_demand_sensor_import():
    """Test import modulo demand_sensor."""
    from arima_forecaster.demand_sensing.demand_sensor import DemandSensor
    from arima_forecaster.demand_sensing.demand_sensor import FactorType, ImpactLevel

    # Verifica che le classi siano importabili
    assert DemandSensor is not None
    assert FactorType is not None
    assert ImpactLevel is not None


def test_weather_import():
    """Test import modulo weather."""
    from arima_forecaster.demand_sensing.weather import WeatherIntegration
    from arima_forecaster.demand_sensing.weather import WeatherCondition, WeatherImpact

    # Verifica che le classi siano importabili
    assert WeatherIntegration is not None
    assert WeatherCondition is not None
    assert WeatherImpact is not None


def test_trends_import():
    """Test import modulo trends."""
    from arima_forecaster.demand_sensing.trends import GoogleTrendsIntegration
    from arima_forecaster.demand_sensing.trends import TrendData, TrendImpact

    # Verifica che le classi siano importabili
    assert GoogleTrendsIntegration is not None
    assert TrendData is not None
    assert TrendImpact is not None


def test_social_import():
    """Test import modulo social."""
    from arima_forecaster.demand_sensing.social import SocialSentimentAnalyzer
    from arima_forecaster.demand_sensing.social import SentimentScore, SocialPost

    # Verifica che le classi siano importabili
    assert SocialSentimentAnalyzer is not None
    assert SentimentScore is not None
    assert SocialPost is not None


def test_economic_import():
    """Test import modulo economic."""
    from arima_forecaster.demand_sensing.economic import EconomicIndicators
    from arima_forecaster.demand_sensing.economic import EconomicIndicator, EconomicData

    # Verifica che le classi siano importabili
    assert EconomicIndicators is not None
    assert EconomicIndicator is not None
    assert EconomicData is not None


def test_calendar_events_import():
    """Test import modulo calendar_events."""
    from arima_forecaster.demand_sensing.calendar_events import CalendarEvents
    from arima_forecaster.demand_sensing.calendar_events import EventType, Event

    # Verifica che le classi siano importabili
    assert CalendarEvents is not None
    assert EventType is not None
    assert Event is not None


def test_ensemble_import():
    """Test import modulo ensemble."""
    from arima_forecaster.demand_sensing.ensemble import EnsembleDemandSensor
    from arima_forecaster.demand_sensing.ensemble import EnsembleConfig

    # Verifica che le classi siano importabili
    assert EnsembleDemandSensor is not None
    assert EnsembleConfig is not None


def test_demand_sensing_module_import():
    """Test import completo del modulo demand_sensing."""
    import arima_forecaster.demand_sensing

    # Verifica che il modulo si importi
    assert arima_forecaster.demand_sensing is not None


def test_basic_instantiation():
    """Test istanziazione base delle classi principali."""
    from arima_forecaster.demand_sensing.weather import WeatherIntegration
    from arima_forecaster.demand_sensing.trends import GoogleTrendsIntegration
    from arima_forecaster.demand_sensing.economic import EconomicIndicators
    from arima_forecaster.demand_sensing.calendar_events import CalendarEvents
    from arima_forecaster.demand_sensing.ensemble import EnsembleDemandSensor

    # Test istanziazione con parametri minimi
    weather = WeatherIntegration()
    assert weather is not None

    trends = GoogleTrendsIntegration(keywords=["test"])
    assert trends is not None

    econ = EconomicIndicators()
    assert econ is not None

    cal = CalendarEvents()
    assert cal is not None

    ensemble = EnsembleDemandSensor()
    assert ensemble is not None
