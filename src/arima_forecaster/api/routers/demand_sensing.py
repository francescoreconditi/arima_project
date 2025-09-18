"""
Router per endpoint di Demand Sensing e forecasting con fattori esterni.

Gestisce integrazione meteo, trends, sentiment, indicatori economici e calendario
per migliorare accuratezza previsioni con ensemble forecasting.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from pydantic import BaseModel, Field, validator
import pandas as pd
import numpy as np

from arima_forecaster.utils.logger import get_logger

logger = get_logger(__name__)

# Crea router con prefix e tags
router = APIRouter(
    prefix="/demand-sensing", tags=["Demand Sensing"], responses={404: {"description": "Not found"}}
)

"""
DEMAND SENSING ROUTER

Gestisce l'integrazione di fattori esterni per demand sensing avanzato:

• POST /demand-sensing/weather-forecast        - Integrazione previsioni meteo
• POST /demand-sensing/trends-analysis        - Google Trends integration  
• POST /demand-sensing/social-sentiment       - Social sentiment analysis
• POST /demand-sensing/economic-indicators    - Indicatori economici
• POST /demand-sensing/calendar-events        - Eventi calendario business
• POST /demand-sensing/ensemble-forecast      - Previsioni ensemble con fattori esterni
• GET  /demand-sensing/factor-contributions   - Analisi contributi per fonte
• POST /demand-sensing/sensitivity-analysis   - Analisi sensibilità fattori

Caratteristiche:
- Integrazione multi-source (weather, social, economic, calendar)
- Ensemble forecasting con confidence scoring
- Real-time factor adjustment e weighting
- Sensitivity analysis per ottimizzazione pesi
- Historical correlation analysis
"""

# =============================================================================
# MODELLI RICHIESTA E RISPOSTA
# =============================================================================


class WeatherForecastRequest(BaseModel):
    """Richiesta integrazione previsioni meteo."""

    location: str = Field(..., description="Location (città, paese) es: 'Rome,IT'")
    forecast_days: int = Field(7, description="Giorni previsioni meteo")
    weather_sensitivity: Dict[str, float] = Field(
        default={
            "temperature": 0.05,  # 5% impatto per grado
            "precipitation": 0.15,  # 15% impatto pioggia
            "wind_speed": 0.02,  # 2% impatto vento
            "humidity": 0.01,  # 1% impatto umidità
        },
        description="Sensibilità domanda per fattore meteo",
    )
    business_type: Optional[str] = Field(
        "general", description="Tipo business per calibrare sensibilità"
    )


class WeatherFactor(BaseModel):
    """Singolo fattore meteo."""

    date: datetime = Field(..., description="Data previsione")
    temperature: float = Field(..., description="Temperatura °C")
    precipitation: float = Field(..., description="Precipitazioni mm")
    humidity: float = Field(..., description="Umidità %")
    wind_speed: float = Field(..., description="Velocità vento km/h")
    weather_condition: str = Field(..., description="Condizione meteo descrittiva")
    impact_score: float = Field(..., description="Score impatto sulla domanda (-1 a +1)")
    confidence: float = Field(..., description="Confidenza previsione (0-1)")


class WeatherForecastResponse(BaseModel):
    """Risposta previsioni meteo."""

    location: str
    forecast_period: str
    weather_factors: List[WeatherFactor] = Field(..., description="Fattori meteo per periodo")
    business_calibration: Dict[str, float] = Field(
        ..., description="Calibrazione per business type"
    )
    summary_impact: Dict[str, float] = Field(..., description="Impatto aggregato per tipo")
    recommendations: List[str] = Field(..., description="Raccomandazioni operative")


# Google Trends Analysis
class TrendsAnalysisRequest(BaseModel):
    """Richiesta analisi Google Trends."""

    keywords: List[str] = Field(..., description="Keywords da monitorare")
    geo_location: Optional[str] = Field("", description="Location geografica (ES, IT, US)")
    timeframe: str = Field("today 1-m", description="Timeframe trends (today 1-m, today 3-m, etc)")
    category: Optional[int] = Field(0, description="Categoria Google Trends")
    correlation_baseline: Optional[List[float]] = Field(
        None, description="Serie storica per correlazione"
    )


class TrendsFactor(BaseModel):
    """Singolo fattore trends."""

    date: datetime
    keyword: str
    search_volume: float = Field(..., description="Volume ricerche normalizzato 0-100")
    relative_change: float = Field(..., description="Cambio relativo vs periodo precedente")
    trend_direction: str = Field(..., description="Direzione trend (up/down/stable)")
    impact_score: float = Field(..., description="Score impatto domanda")
    confidence: float = Field(..., description="Confidenza dato")


class TrendsAnalysisResponse(BaseModel):
    """Risposta analisi trends."""

    keywords_analyzed: List[str]
    analysis_period: str
    trends_factors: List[TrendsFactor]
    correlation_analysis: Dict[str, float] = Field(
        ..., description="Correlazione keywords vs domanda"
    )
    seasonal_patterns: Dict[str, Any] = Field(..., description="Pattern stagionali identificati")
    forecast_impact: Dict[str, float] = Field(..., description="Impatto forecast per keyword")


# Social Sentiment Analysis
class SocialSentimentRequest(BaseModel):
    """Richiesta analisi social sentiment."""

    brand_keywords: List[str] = Field(..., description="Keywords brand da monitorare")
    product_keywords: List[str] = Field(..., description="Keywords prodotti")
    platforms: List[str] = Field(
        default=["twitter", "instagram", "facebook"], description="Platform sociali"
    )
    sentiment_weights: Dict[str, float] = Field(
        default={"positive": 0.1, "negative": -0.15, "neutral": 0.0},
        description="Pesi impatto sentiment sulla domanda",
    )
    analysis_days: int = Field(7, description="Giorni analisi sentiment")


class SentimentFactor(BaseModel):
    """Singolo fattore sentiment."""

    date: datetime
    platform: str
    keyword: str
    sentiment_score: float = Field(..., description="Score sentiment (-1 negativo, +1 positivo)")
    post_volume: int = Field(..., description="Numero post analizzati")
    engagement_rate: float = Field(..., description="Tasso engagement medio")
    impact_score: float = Field(..., description="Impatto stimato sulla domanda")
    confidence: float = Field(..., description="Confidenza analisi")


class SocialSentimentResponse(BaseModel):
    """Risposta analisi sentiment."""

    analysis_period: str
    platforms_analyzed: List[str]
    sentiment_factors: List[SentimentFactor]
    aggregate_sentiment: Dict[str, float] = Field(
        ..., description="Sentiment aggregato per keyword"
    )
    trend_analysis: Dict[str, str] = Field(..., description="Trend sentiment nel periodo")
    alert_triggers: List[str] = Field(..., description="Alert per sentiment critici")


# Economic Indicators
class EconomicIndicatorsRequest(BaseModel):
    """Richiesta indicatori economici."""

    country_code: str = Field("IT", description="Codice paese (IT, US, DE, etc)")
    indicators: List[str] = Field(
        default=["gdp", "unemployment", "inflation", "consumer_confidence"],
        description="Indicatori da monitorare",
    )
    business_sector: str = Field(
        "consumer_goods", description="Settore business per calibrare sensitivity"
    )
    forecast_horizon: int = Field(30, description="Giorni forecast")


class EconomicFactor(BaseModel):
    """Singolo indicatore economico."""

    date: datetime
    indicator: str = Field(..., description="Nome indicatore")
    value: float = Field(..., description="Valore indicatore")
    change_percent: float = Field(..., description="Cambio percentuale vs periodo precedente")
    impact_score: float = Field(..., description="Score impatto domanda")
    confidence: float = Field(..., description="Confidenza dato")
    data_source: str = Field(..., description="Fonte dato")


class EconomicIndicatorsResponse(BaseModel):
    """Risposta indicatori economici."""

    country: str
    analysis_period: str
    economic_factors: List[EconomicFactor]
    sector_calibration: Dict[str, float] = Field(..., description="Calibrazione per settore")
    leading_indicators: List[str] = Field(..., description="Indicatori leading identificati")
    forecast_impact: Dict[str, float] = Field(..., description="Impatto forecast per indicatore")


# Calendar Events
class CalendarEventsRequest(BaseModel):
    """Richiesta eventi calendario."""

    country_code: str = Field("IT", description="Codice paese per festività")
    custom_events: Optional[List[Dict[str, Any]]] = Field(
        None, description="Eventi custom business"
    )
    event_types: List[str] = Field(
        default=["holidays", "sports", "cultural", "business"],
        description="Tipi eventi da considerare",
    )
    impact_radius_days: int = Field(3, description="Giorni impatto pre/post evento")
    forecast_days: int = Field(30, description="Giorni forecast eventi")


class CalendarEvent(BaseModel):
    """Singolo evento calendario."""

    date: datetime
    event_name: str = Field(..., description="Nome evento")
    event_type: str = Field(..., description="Tipo evento")
    impact_score: float = Field(..., description="Score impatto domanda")
    duration_days: int = Field(..., description="Durata evento giorni")
    pre_impact_days: int = Field(..., description="Giorni pre-impatto")
    post_impact_days: int = Field(..., description="Giorni post-impatto")
    confidence: float = Field(..., description="Confidenza impatto")


class CalendarEventsResponse(BaseModel):
    """Risposta eventi calendario."""

    country: str
    forecast_period: str
    calendar_events: List[CalendarEvent]
    holiday_impact_analysis: Dict[str, float] = Field(..., description="Analisi impatto festività")
    seasonal_adjustments: Dict[str, float] = Field(..., description="Aggiustamenti stagionali")
    high_impact_dates: List[str] = Field(..., description="Date alto impatto identificate")


# Ensemble Forecasting
class EnsembleForecastRequest(BaseModel):
    """Richiesta ensemble forecasting con fattori esterni."""

    base_forecast: List[float] = Field(..., description="Forecast base (ARIMA/SARIMA)")
    forecast_dates: List[str] = Field(..., description="Date forecast")
    weather_factors: Optional[List[WeatherFactor]] = Field(None, description="Fattori meteo")
    trends_factors: Optional[List[TrendsFactor]] = Field(None, description="Fattori trends")
    sentiment_factors: Optional[List[SentimentFactor]] = Field(
        None, description="Fattori sentiment"
    )
    economic_factors: Optional[List[EconomicFactor]] = Field(None, description="Fattori economici")
    calendar_events: Optional[List[CalendarEvent]] = Field(None, description="Eventi calendario")
    ensemble_weights: Optional[Dict[str, float]] = Field(
        default={
            "base": 0.4,
            "weather": 0.2,
            "trends": 0.15,
            "sentiment": 0.1,
            "economic": 0.1,
            "calendar": 0.05,
        },
        description="Pesi ensemble per fonte",
    )


class EnsembleForecastResponse(BaseModel):
    """Risposta ensemble forecasting."""

    forecast_id: str = Field(..., description="ID forecast")
    base_forecast: List[float] = Field(..., description="Forecast originale")
    adjusted_forecast: List[float] = Field(..., description="Forecast aggiustato ensemble")
    adjustment_factors: List[float] = Field(..., description="Fattori aggiustamento per data")
    confidence_intervals: Dict[str, List[float]] = Field(..., description="Intervalli confidenza")
    factor_contributions: Dict[str, List[float]] = Field(..., description="Contributi per fonte")
    total_adjustment: float = Field(..., description="Aggiustamento totale %")
    confidence_score: float = Field(..., description="Score confidenza ensemble")
    details: Dict[str, Any] = Field(..., description="Dettagli fattori e calcoli")


# Factor Contributions Analysis
class FactorContributionsResponse(BaseModel):
    """Risposta analisi contributi fattori."""

    analysis_id: str
    forecast_period: str
    contributions_by_source: Dict[str, Dict[str, float]] = Field(
        ..., description="Contributi per fonte e metrica"
    )
    correlation_matrix: Dict[str, Dict[str, float]] = Field(
        ..., description="Correlazioni tra fattori"
    )
    importance_ranking: List[Dict[str, Any]] = Field(..., description="Ranking importanza fattori")
    optimization_suggestions: List[str] = Field(..., description="Suggerimenti ottimizzazione pesi")


# Sensitivity Analysis
class SensitivityAnalysisRequest(BaseModel):
    """Richiesta analisi sensibilità."""

    historical_forecasts: List[Dict[str, Any]] = Field(
        ..., description="Forecast storici con fattori"
    )
    actual_values: List[float] = Field(..., description="Valori reali per validazione")
    sensitivity_range: float = Field(0.2, description="Range sensibilità (+/- 20%)")
    factors_to_test: List[str] = Field(..., description="Fattori da testare")


class SensitivityAnalysisResponse(BaseModel):
    """Risposta analisi sensibilità."""

    analysis_id: str
    optimal_weights: Dict[str, float] = Field(..., description="Pesi ottimali identificati")
    sensitivity_scores: Dict[str, float] = Field(..., description="Score sensibilità per fattore")
    forecast_improvement: float = Field(..., description="Miglioramento accuracy %")
    weight_adjustments: Dict[str, Dict[str, float]] = Field(
        ..., description="Aggiustamenti pesi raccomandati"
    )
    validation_metrics: Dict[str, float] = Field(..., description="Metriche validazione")


# =============================================================================
# DEPENDENCY INJECTION
# =============================================================================


def get_demand_sensing_services():
    """Dependency per ottenere i servizi demand sensing."""
    try:
        from arima_forecaster.demand_sensing import (
            EnsembleDemandSensor,
            WeatherIntegration,
            GoogleTrendsIntegration,
            SocialSentimentAnalyzer,
            EconomicIndicators,
            CalendarEvents,
        )

        return {
            "ensemble": EnsembleDemandSensor,
            "weather": WeatherIntegration,
            "trends": GoogleTrendsIntegration,
            "sentiment": SocialSentimentAnalyzer,
            "economic": EconomicIndicators,
            "calendar": CalendarEvents,
        }
    except ImportError:
        return None


# =============================================================================
# ENDPOINT IMPLEMENTATIONS
# =============================================================================


@router.post("/weather-forecast", response_model=WeatherForecastResponse)
async def integrate_weather_forecast(
    request: WeatherForecastRequest, services: Optional[Dict] = Depends(get_demand_sensing_services)
):
    """
    Integra previsioni meteo per demand sensing con calibrazioni business-specific.

    <h4>Impatto Meteo per Business Type:</h4>
    <table >
        <tr><th>Business Type</th><th>Fattori Chiave</th><th>Sensibilità</th></tr>
        <tr><td>Food Delivery</td><td>Pioggia: +30%, Caldo: +20%</td><td>Alta</td></tr>
        <tr><td>Fashion Retail</td><td>Temperatura: +15%, Sole: +10%</td><td>Media</td></tr>
        <tr><td>Ice Cream</td><td>Temperatura >25°C: +50%</td><td>Molto Alta</td></tr>
        <tr><td>Umbrella</td><td>Pioggia: +200%, Neve: +100%</td><td>Estrema</td></tr>
        <tr><td>Energy</td><td>Temperatura: +5% per grado</td><td>Lineare</td></tr>
    </table>

    <h4>Esempio Richiesta:</h4>
    <pre><code>
    {
        "location": "Rome,IT",
        "forecast_days": 7,
        "weather_sensitivity": {
            "temperature": 0.05,
            "precipitation": 0.15,
            "wind_speed": 0.02,
            "humidity": 0.01
        },
        "business_type": "food_delivery"
    }
    </code></pre>

    <h4>Risposta di Esempio:</h4>
    <pre><code>
    {
        "location": "Rome,IT",
        "forecast_period": "2024-08-23 to 2024-08-30",
        "weather_factors": [
            {
                "date": "2024-08-24T12:00:00",
                "temperature": 28.5,
                "precipitation": 2.5,
                "humidity": 65.0,
                "wind_speed": 12.0,
                "weather_condition": "Light Rain",
                "impact_score": 0.18,
                "confidence": 0.85
            }
        ],
        "business_calibration": {
            "temperature_multiplier": 1.2,
            "precipitation_boost": 1.8,
            "seasonal_adjustment": 0.95
        },
        "summary_impact": {
            "high_demand_days": 3,
            "average_impact": 0.12,
            "max_daily_impact": 0.25
        },
        "recommendations": [
            "Day 2024-08-24: Rain expected (+18% demand) - increase delivery staff",
            "Day 2024-08-26: Hot weather (30°C) - prepare cold drinks inventory"
        ]
    }
    </code></pre>
    """
    try:
        # Calibrazione business-specific
        business_calibrations = {
            "food_delivery": {
                "temperature_base": 22.0,  # Temperatura neutrale
                "temperature_sensitivity": 0.08,  # 8% per grado
                "precipitation_boost": 1.5,  # 150% boost pioggia
                "wind_sensitivity": 0.05,
                "humidity_impact": 0.02,
            },
            "fashion_retail": {
                "temperature_base": 20.0,
                "temperature_sensitivity": 0.03,
                "precipitation_boost": -0.2,  # Pioggia riduce shopping
                "wind_sensitivity": 0.01,
                "humidity_impact": 0.01,
            },
            "ice_cream": {
                "temperature_base": 15.0,
                "temperature_sensitivity": 0.15,  # Molto sensibile
                "precipitation_boost": -0.8,  # Pioggia dannosa
                "wind_sensitivity": 0.02,
                "humidity_impact": -0.03,  # Umidità riduce appeal
            },
            "general": {
                "temperature_base": 18.0,
                "temperature_sensitivity": 0.02,
                "precipitation_boost": 0.1,
                "wind_sensitivity": 0.01,
                "humidity_impact": 0.01,
            },
        }

        calibration = business_calibrations.get(
            request.business_type, business_calibrations["general"]
        )

        # Genera previsioni meteo demo (in produzione useremmo API reale)
        weather_factors = []
        base_date = datetime.now()

        # Simuliamo dati meteo realistici per il periodo
        np.random.seed(42)  # Per risultati riproducibili

        high_demand_days = 0
        total_impact = 0
        max_daily_impact = 0

        for day in range(request.forecast_days):
            forecast_date = base_date + timedelta(days=day)

            # Genera dati meteo casuali ma realistici
            base_temp = 25.0 + np.random.normal(0, 3)  # Roma estate
            precipitation = max(0, np.random.exponential(2.0))  # Media 2mm
            humidity = 50 + np.random.normal(0, 15)
            wind_speed = max(0, np.random.normal(10, 5))

            # Determina condizione meteo
            if precipitation > 10:
                weather_condition = "Heavy Rain"
                confidence = 0.75
            elif precipitation > 2:
                weather_condition = "Light Rain"
                confidence = 0.80
            elif base_temp > 30:
                weather_condition = "Hot"
                confidence = 0.85
            elif base_temp < 15:
                weather_condition = "Cold"
                confidence = 0.85
            else:
                weather_condition = "Partly Cloudy"
                confidence = 0.90

            # Calcola impatto sulla domanda
            temp_impact = (
                (base_temp - calibration["temperature_base"])
                * calibration["temperature_sensitivity"]
                / 100
            )
            precip_impact = (
                precipitation * calibration["precipitation_boost"] * 0.01
                if precipitation > 0.5
                else 0
            )
            wind_impact = (wind_speed - 10) * calibration["wind_sensitivity"] / 100
            humidity_impact = (humidity - 60) * calibration["humidity_impact"] / 100

            total_impact_score = temp_impact + precip_impact + wind_impact + humidity_impact

            # Limita impact score a range ragionevole
            impact_score = max(-0.5, min(0.5, total_impact_score))

            if impact_score > 0.1:
                high_demand_days += 1

            total_impact += impact_score
            max_daily_impact = max(max_daily_impact, impact_score)

            weather_factors.append(
                WeatherFactor(
                    date=forecast_date,
                    temperature=round(base_temp, 1),
                    precipitation=round(precipitation, 1),
                    humidity=round(humidity, 1),
                    wind_speed=round(wind_speed, 1),
                    weather_condition=weather_condition,
                    impact_score=round(impact_score, 3),
                    confidence=round(confidence, 2),
                )
            )

        # Business calibration response
        business_calibration = {
            "temperature_multiplier": calibration["temperature_sensitivity"]
            / request.weather_sensitivity["temperature"],
            "precipitation_boost": calibration["precipitation_boost"],
            "seasonal_adjustment": 1.0
            if datetime.now().month in [6, 7, 8]
            else 0.9,  # Estate boost
            "business_type_modifier": 1.2 if request.business_type == "food_delivery" else 1.0,
        }

        # Summary impact
        summary_impact = {
            "high_demand_days": high_demand_days,
            "average_impact": round(total_impact / request.forecast_days, 3),
            "max_daily_impact": round(max_daily_impact, 3),
            "total_adjustment_percent": round((total_impact / request.forecast_days) * 100, 1),
        }

        # Genera raccomandazioni operative
        recommendations = []

        for factor in weather_factors:
            if factor.impact_score > 0.15:
                if factor.precipitation > 5:
                    recommendations.append(
                        f"Day {factor.date.strftime('%Y-%m-%d')}: Rain expected ({factor.impact_score:+.0%} demand) - increase delivery staff"
                    )
                elif factor.temperature > 30:
                    recommendations.append(
                        f"Day {factor.date.strftime('%Y-%m-%d')}: Hot weather ({factor.temperature}°C) - prepare cooling products"
                    )
            elif factor.impact_score < -0.1:
                recommendations.append(
                    f"Day {factor.date.strftime('%Y-%m-%d')}: Lower demand expected ({factor.impact_score:+.0%}) - reduce inventory"
                )

        if not recommendations:
            recommendations.append("Weather conditions appear stable - maintain normal operations")

        return WeatherForecastResponse(
            location=request.location,
            forecast_period=f"{base_date.strftime('%Y-%m-%d')} to {(base_date + timedelta(days=request.forecast_days - 1)).strftime('%Y-%m-%d')}",
            weather_factors=weather_factors,
            business_calibration=business_calibration,
            summary_impact=summary_impact,
            recommendations=recommendations,
        )

    except Exception as e:
        logger.error(f"Errore integrazione meteo: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Errore weather forecast: {str(e)}")


@router.post("/trends-analysis", response_model=TrendsAnalysisResponse)
async def analyze_google_trends(
    request: TrendsAnalysisRequest, services: Optional[Dict] = Depends(get_demand_sensing_services)
):
    """
    Analizza Google Trends per identificare pattern di domanda e correlazioni.

    <h4>Google Trends Insights per Business:</h4>
    <table >
        <tr><th>Keyword Pattern</th><th>Interpretazione</th><th>Azione Suggerita</th></tr>
        <tr><td>Crescita costante</td><td>Interesse crescente prodotto/categoria</td><td>Aumenta inventory</td></tr>
        <tr><td>Picco stagionale</td><td>Stagionalità ricorrente</td><td>Pianifica stock stagionale</td></tr>
        <tr><td>Picco anomalo</td><td>Evento virale o trend emergente</td><td>Capitalizza rapidamente</td></tr>
        <tr><td>Declino graduale</td><td>Interesse calante</td><td>Riduci inventory, diversifica</td></tr>
    </table>

    <h4>Esempio Richiesta:</h4>
    <pre><code>
    {
        "keywords": ["pizza delivery", "food delivery", "online ordering"],
        "geo_location": "IT",
        "timeframe": "today 1-m",
        "category": 71,
        "correlation_baseline": [120, 135, 110, 145, 160, 140, 125, 155]
    }
    </code></pre>

    <h4>Risposta di Esempio:</h4>
    <pre><code>
    {
        "keywords_analyzed": ["pizza delivery", "food delivery"],
        "analysis_period": "2024-07-23 to 2024-08-23",
        "trends_factors": [
            {
                "date": "2024-08-20T00:00:00",
                "keyword": "pizza delivery",
                "search_volume": 78.5,
                "relative_change": 0.12,
                "trend_direction": "up",
                "impact_score": 0.08,
                "confidence": 0.85
            }
        ],
        "correlation_analysis": {
            "pizza delivery": 0.72,
            "food delivery": 0.68
        },
        "seasonal_patterns": {
            "weekend_boost": 1.25,
            "evening_peak": 1.40,
            "weather_correlation": 0.45
        },
        "forecast_impact": {
            "pizza delivery": 0.08,
            "food delivery": 0.06
        }
    }
    </code></pre>
    """
    try:
        # Genera dati trends demo realistici
        np.random.seed(42)
        base_date = datetime.now() - timedelta(days=30)

        trends_factors = []
        correlation_scores = {}

        for keyword in request.keywords:
            keyword_correlation = 0
            keyword_factors = []

            # Genera serie temporale per keyword
            for day in range(31):  # Ultimo mese
                trend_date = base_date + timedelta(days=day)

                # Simula search volume con pattern realistici
                base_volume = np.random.normal(60, 15)

                # Aggiungi pattern settimanali (weekend boost per food delivery)
                if trend_date.weekday() >= 5:  # Weekend
                    base_volume *= 1.25

                # Aggiungi trend generale
                growth_trend = day * 0.5  # Crescita graduale
                search_volume = max(0, min(100, base_volume + growth_trend))

                # Calcola cambiamento relativo
                if day > 0:
                    prev_volume = keyword_factors[-1].search_volume
                    relative_change = (
                        (search_volume - prev_volume) / prev_volume if prev_volume > 0 else 0
                    )
                else:
                    relative_change = 0

                # Determina direzione trend
                if relative_change > 0.05:
                    trend_direction = "up"
                elif relative_change < -0.05:
                    trend_direction = "down"
                else:
                    trend_direction = "stable"

                # Calcola impact score (correlazione con domanda)
                impact_score = relative_change * 0.3  # 30% di correlazione base

                # Confidenza basata su volatilità
                volatility = abs(relative_change)
                confidence = max(0.5, 0.9 - volatility * 2)

                factor = TrendsFactor(
                    date=trend_date,
                    keyword=keyword,
                    search_volume=round(search_volume, 1),
                    relative_change=round(relative_change, 3),
                    trend_direction=trend_direction,
                    impact_score=round(impact_score, 3),
                    confidence=round(confidence, 2),
                )

                keyword_factors.append(factor)
                trends_factors.append(factor)

            # Calcola correlazione con baseline se fornita
            if request.correlation_baseline:
                # Prendi ultimi N valori per correlazione
                search_volumes = [
                    f.search_volume for f in keyword_factors[-len(request.correlation_baseline) :]
                ]
                if len(search_volumes) == len(request.correlation_baseline):
                    correlation = np.corrcoef(search_volumes, request.correlation_baseline)[0, 1]
                    correlation_scores[keyword] = round(correlation, 3)
                else:
                    correlation_scores[keyword] = 0.5  # Default neutro
            else:
                # Stima correlazione basata su keyword relevance
                relevance_scores = {
                    "pizza": 0.75,
                    "delivery": 0.70,
                    "food": 0.68,
                    "restaurant": 0.65,
                    "ordering": 0.60,
                }
                keyword_lower = keyword.lower()
                correlation = max(
                    [score for term, score in relevance_scores.items() if term in keyword_lower],
                    default=0.5,
                )
                correlation_scores[keyword] = correlation

        # Analisi pattern stagionali
        weekend_volumes = [f.search_volume for f in trends_factors if f.date.weekday() >= 5]
        weekday_volumes = [f.search_volume for f in trends_factors if f.date.weekday() < 5]

        weekend_boost = (
            np.mean(weekend_volumes) / np.mean(weekday_volumes) if weekday_volumes else 1.0
        )

        seasonal_patterns = {
            "weekend_boost": round(weekend_boost, 2),
            "evening_peak": 1.4,  # Simulato
            "weather_correlation": 0.45,  # Simulato
            "monthly_trend": round(
                np.mean([f.relative_change for f in trends_factors[-7:]]), 3
            ),  # Trend ultima settimana
        }

        # Calcola impatto forecast per keyword
        forecast_impact = {}
        for keyword in request.keywords:
            keyword_factors = [f for f in trends_factors if f.keyword == keyword]
            recent_trend = np.mean(
                [f.impact_score for f in keyword_factors[-7:]]
            )  # Ultima settimana
            forecast_impact[keyword] = round(recent_trend, 3)

        return TrendsAnalysisResponse(
            keywords_analyzed=request.keywords,
            analysis_period=f"{base_date.strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')}",
            trends_factors=trends_factors,
            correlation_analysis=correlation_scores,
            seasonal_patterns=seasonal_patterns,
            forecast_impact=forecast_impact,
        )

    except Exception as e:
        logger.error(f"Errore analisi trends: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Errore trends analysis: {str(e)}")


@router.post("/social-sentiment", response_model=SocialSentimentResponse)
async def analyze_social_sentiment(
    request: SocialSentimentRequest, services: Optional[Dict] = Depends(get_demand_sensing_services)
):
    """
    Analizza sentiment sui social media per previsioni demand-aware.

    <h4>Sentiment Impact su Domanda:</h4>
    <table >
        <tr><th>Sentiment Score</th><th>Interpretazione</th><th>Impatto Domanda</th></tr>
        <tr><td>>0.6</td><td>Molto Positivo</td><td>+10-15% boost</td></tr>
        <tr><td>0.2 to 0.6</td><td>Positivo</td><td>+5-10% boost</td></tr>
        <tr><td>-0.2 to 0.2</td><td>Neutrale</td><td>Nessun impatto</td></tr>
        <tr><td>-0.6 to -0.2</td><td>Negativo</td><td>-5-10% riduzione</td></tr>
        <tr><td><-0.6</td><td>Molto Negativo</td><td>-15-25% riduzione</td></tr>
    </table>

    <h4>Esempio Richiesta:</h4>
    <pre><code>
    {
        "brand_keywords": ["MyBrand", "@MyBrandOfficial"],
        "product_keywords": ["pizza margherita", "delivery service"],
        "platforms": ["twitter", "instagram"],
        "sentiment_weights": {
            "positive": 0.1,
            "negative": -0.15,
            "neutral": 0.0
        },
        "analysis_days": 7
    }
    </code></pre>

    <h4>Risposta di Esempio:</h4>
    <pre><code>
    {
        "analysis_period": "2024-08-16 to 2024-08-23",
        "platforms_analyzed": ["twitter", "instagram"],
        "sentiment_factors": [
            {
                "date": "2024-08-22T00:00:00",
                "platform": "twitter",
                "keyword": "MyBrand",
                "sentiment_score": 0.35,
                "post_volume": 156,
                "engagement_rate": 0.045,
                "impact_score": 0.08,
                "confidence": 0.82
            }
        ],
        "aggregate_sentiment": {
            "MyBrand": 0.28,
            "pizza margherita": 0.15
        },
        "trend_analysis": {
            "MyBrand": "Improving - positive trend last 3 days",
            "delivery service": "Stable - consistent neutral sentiment"
        },
        "alert_triggers": [
            "Negative spike detected for 'delivery service' - investigate customer issues"
        ]
    }
    </code></pre>
    """
    try:
        # Genera dati sentiment demo realistici
        np.random.seed(42)
        base_date = datetime.now() - timedelta(days=request.analysis_days)

        sentiment_factors = []
        all_keywords = request.brand_keywords + request.product_keywords

        for day in range(request.analysis_days):
            analysis_date = base_date + timedelta(days=day)

            for platform in request.platforms:
                for keyword in all_keywords:
                    # Genera sentiment score realistico
                    base_sentiment = np.random.normal(0.1, 0.3)  # Leggermente positivo di default

                    # Aggiungi variabilità per brand vs prodotti
                    if keyword in request.brand_keywords:
                        # Brand tende ad avere sentiment più stabile
                        sentiment_score = np.clip(base_sentiment, -0.8, 0.8)
                    else:
                        # Prodotti possono avere sentiment più volatile
                        sentiment_score = np.clip(base_sentiment * 1.2, -0.9, 0.9)

                    # Volume post variabile per platform
                    platform_multipliers = {
                        "twitter": 3.0,
                        "instagram": 2.0,
                        "facebook": 1.5,
                        "linkedin": 0.5,
                    }
                    base_volume = int(
                        np.random.poisson(50) * platform_multipliers.get(platform, 1.0)
                    )

                    # Engagement rate variabile per sentiment
                    base_engagement = 0.03
                    if sentiment_score > 0.3:
                        engagement_rate = base_engagement * 1.5  # Contenuto positivo più engaging
                    elif sentiment_score < -0.3:
                        engagement_rate = (
                            base_engagement * 1.8
                        )  # Contenuto negativo genera discussioni
                    else:
                        engagement_rate = base_engagement

                    # Calcola impact score basato su sentiment e volume
                    volume_weight = min(1.0, base_volume / 100)  # Normalizza volume
                    sentiment_weight = request.sentiment_weights.get(
                        "positive"
                        if sentiment_score > 0.1
                        else "negative"
                        if sentiment_score < -0.1
                        else "neutral",
                        0.0,
                    )

                    impact_score = sentiment_score * sentiment_weight * volume_weight

                    # Confidenza basata su volume e coerenza sentiment
                    confidence = min(
                        0.95, 0.5 + (volume_weight * 0.3) + (abs(sentiment_score) * 0.15)
                    )

                    sentiment_factors.append(
                        SentimentFactor(
                            date=analysis_date,
                            platform=platform,
                            keyword=keyword,
                            sentiment_score=round(sentiment_score, 3),
                            post_volume=base_volume,
                            engagement_rate=round(engagement_rate, 3),
                            impact_score=round(impact_score, 3),
                            confidence=round(confidence, 2),
                        )
                    )

        # Calcola sentiment aggregato per keyword
        aggregate_sentiment = {}
        for keyword in all_keywords:
            keyword_sentiments = [
                f.sentiment_score for f in sentiment_factors if f.keyword == keyword
            ]
            if keyword_sentiments:
                # Media pesata per volume
                keyword_factors = [f for f in sentiment_factors if f.keyword == keyword]
                total_volume = sum([f.post_volume for f in keyword_factors])
                if total_volume > 0:
                    weighted_sentiment = (
                        sum([f.sentiment_score * f.post_volume for f in keyword_factors])
                        / total_volume
                    )
                else:
                    weighted_sentiment = np.mean(keyword_sentiments)
                aggregate_sentiment[keyword] = round(weighted_sentiment, 3)
            else:
                aggregate_sentiment[keyword] = 0.0

        # Analisi trend sentiment
        trend_analysis = {}
        for keyword in all_keywords:
            keyword_factors = [f for f in sentiment_factors if f.keyword == keyword]
            if len(keyword_factors) >= 3:
                # Confronta prima metà vs seconda metà periodo
                mid_point = len(keyword_factors) // 2
                early_sentiment = np.mean([f.sentiment_score for f in keyword_factors[:mid_point]])
                late_sentiment = np.mean([f.sentiment_score for f in keyword_factors[mid_point:]])

                change = late_sentiment - early_sentiment
                if change > 0.1:
                    trend_analysis[keyword] = "Improving - positive trend in recent period"
                elif change < -0.1:
                    trend_analysis[keyword] = "Declining - negative trend detected"
                else:
                    trend_analysis[keyword] = "Stable - consistent sentiment"
            else:
                trend_analysis[keyword] = "Insufficient data for trend analysis"

        # Generate alert triggers
        alert_triggers = []
        for keyword, sentiment in aggregate_sentiment.items():
            if sentiment < -0.4:
                alert_triggers.append(
                    f"Negative spike detected for '{keyword}' - investigate customer issues"
                )
            elif sentiment > 0.6:
                alert_triggers.append(
                    f"Very positive sentiment for '{keyword}' - consider promotional campaign"
                )

        # Controlla volatilità
        for keyword in all_keywords:
            keyword_sentiments = [
                f.sentiment_score for f in sentiment_factors if f.keyword == keyword
            ]
            if keyword_sentiments and np.std(keyword_sentiments) > 0.4:
                alert_triggers.append(
                    f"High sentiment volatility for '{keyword}' - monitor closely"
                )

        if not alert_triggers:
            alert_triggers.append("No significant sentiment alerts - normal monitoring continues")

        return SocialSentimentResponse(
            analysis_period=f"{base_date.strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')}",
            platforms_analyzed=request.platforms,
            sentiment_factors=sentiment_factors,
            aggregate_sentiment=aggregate_sentiment,
            trend_analysis=trend_analysis,
            alert_triggers=alert_triggers,
        )

    except Exception as e:
        logger.error(f"Errore analisi sentiment: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Errore social sentiment: {str(e)}")


@router.post("/economic-indicators", response_model=EconomicIndicatorsResponse)
async def analyze_economic_indicators(
    request: EconomicIndicatorsRequest,
    services: Optional[Dict] = Depends(get_demand_sensing_services),
):
    """
    Analizza indicatori economici per forecasting macro-economico della domanda.

    <h4>Indicatori Economici per Settore:</h4>
    <table >
        <tr><th>Settore</th><th>Indicatori Chiave</th><th>Sensibilità</th></tr>
        <tr><td>Consumer Goods</td><td>Consumer Confidence, Unemployment</td><td>Alta</td></tr>
        <tr><td>Luxury</td><td>GDP Growth, Stock Market</td><td>Molto Alta</td></tr>
        <tr><td>Essential Goods</td><td>Inflation, Wage Growth</td><td>Bassa</td></tr>
        <tr><td>Real Estate</td><td>Interest Rates, Employment</td><td>Estrema</td></tr>
        <tr><td>Technology</td><td>Business Investment, Innovation Index</td><td>Media</td></tr>
    </table>

    <h4>Esempio Richiesta:</h4>
    <pre><code>
    {
        "country_code": "IT",
        "indicators": ["gdp", "unemployment", "inflation", "consumer_confidence"],
        "business_sector": "consumer_goods",
        "forecast_horizon": 30
    }
    </code></pre>

    <h4>Risposta di Esempio:</h4>
    <pre><code>
    {
        "country": "IT",
        "analysis_period": "2024-08-01 to 2024-08-30",
        "economic_factors": [
            {
                "date": "2024-08-20T00:00:00",
                "indicator": "consumer_confidence",
                "value": 102.5,
                "change_percent": 2.1,
                "impact_score": 0.085,
                "confidence": 0.90,
                "data_source": "ISTAT"
            }
        ],
        "sector_calibration": {
            "gdp_sensitivity": 0.15,
            "unemployment_sensitivity": -0.08,
            "confidence_multiplier": 1.2
        },
        "leading_indicators": ["consumer_confidence", "unemployment"],
        "forecast_impact": {
            "consumer_confidence": 0.085,
            "gdp": 0.045
        }
    }
    </code></pre>
    """
    try:
        # Calibrazioni settoriali
        sector_calibrations = {
            "consumer_goods": {
                "gdp_sensitivity": 0.12,
                "unemployment_sensitivity": -0.08,
                "inflation_sensitivity": -0.05,
                "confidence_sensitivity": 0.15,
            },
            "luxury": {
                "gdp_sensitivity": 0.25,
                "unemployment_sensitivity": -0.15,
                "inflation_sensitivity": -0.08,
                "confidence_sensitivity": 0.20,
            },
            "essential": {
                "gdp_sensitivity": 0.05,
                "unemployment_sensitivity": -0.03,
                "inflation_sensitivity": -0.12,  # Più sensibile all'inflazione
                "confidence_sensitivity": 0.08,
            },
            "technology": {
                "gdp_sensitivity": 0.18,
                "unemployment_sensitivity": -0.06,
                "inflation_sensitivity": -0.04,
                "confidence_sensitivity": 0.12,
            },
        }

        calibration = sector_calibrations.get(
            request.business_sector, sector_calibrations["consumer_goods"]
        )

        # Genera dati economici demo realistici
        np.random.seed(42)
        base_date = datetime.now() - timedelta(days=30)

        economic_factors = []

        # Valori base per indicatori (Italia 2024)
        indicator_baselines = {
            "gdp": {"value": 101.2, "volatility": 0.5, "source": "ISTAT"},
            "unemployment": {"value": 7.8, "volatility": 0.2, "source": "ISTAT"},
            "inflation": {"value": 1.9, "volatility": 0.3, "source": "ISTAT"},
            "consumer_confidence": {"value": 100.5, "volatility": 2.0, "source": "ISTAT"},
            "retail_sales": {"value": 102.8, "volatility": 1.5, "source": "ISTAT"},
            "industrial_production": {"value": 99.2, "volatility": 1.8, "source": "ISTAT"},
        }

        # Genera serie temporali per indicatori
        for indicator in request.indicators:
            if indicator not in indicator_baselines:
                continue

            baseline = indicator_baselines[indicator]
            previous_value = baseline["value"]

            for week in range(5):  # 5 settimane
                factor_date = base_date + timedelta(weeks=week)

                # Simula cambiamento realistico
                change = np.random.normal(0, baseline["volatility"])
                new_value = previous_value + change

                # Calcola change percent
                change_percent = (
                    (new_value - previous_value) / previous_value * 100
                    if previous_value != 0
                    else 0
                )

                # Calcola impact score basato su calibrazione settoriale
                if indicator == "gdp":
                    impact_score = change_percent * calibration["gdp_sensitivity"] / 100
                elif indicator == "unemployment":
                    impact_score = change_percent * calibration["unemployment_sensitivity"] / 100
                elif indicator == "inflation":
                    impact_score = change_percent * calibration["inflation_sensitivity"] / 100
                elif indicator == "consumer_confidence":
                    impact_score = change_percent * calibration["confidence_sensitivity"] / 100
                else:
                    impact_score = change_percent * 0.05 / 100  # Default sensitivity

                # Confidenza basata su source e volatilità
                confidence = 0.95 if baseline["source"] == "ISTAT" else 0.85
                if abs(change_percent) > baseline["volatility"] * 2:  # Cambio anomalo
                    confidence *= 0.8

                economic_factors.append(
                    EconomicFactor(
                        date=factor_date,
                        indicator=indicator,
                        value=round(new_value, 2),
                        change_percent=round(change_percent, 2),
                        impact_score=round(impact_score, 4),
                        confidence=round(confidence, 2),
                        data_source=baseline["source"],
                    )
                )

                previous_value = new_value

        # Identifica leading indicators (quelli con maggiore correlazione)
        leading_indicators = []
        for indicator in request.indicators:
            indicator_factors = [f for f in economic_factors if f.indicator == indicator]
            avg_impact = np.mean([abs(f.impact_score) for f in indicator_factors])
            if avg_impact > 0.01:  # Soglia significatività
                leading_indicators.append(indicator)

        # Se nessuno supera la soglia, prendi i top 2
        if not leading_indicators and economic_factors:
            impact_by_indicator = {}
            for indicator in request.indicators:
                indicator_factors = [f for f in economic_factors if f.indicator == indicator]
                if indicator_factors:
                    impact_by_indicator[indicator] = np.mean(
                        [abs(f.impact_score) for f in indicator_factors]
                    )

            leading_indicators = sorted(
                impact_by_indicator.keys(), key=lambda x: impact_by_indicator[x], reverse=True
            )[:2]

        # Calcola forecast impact per indicatore
        forecast_impact = {}
        for indicator in request.indicators:
            recent_factors = [f for f in economic_factors if f.indicator == indicator][
                -2:
            ]  # Ultimi 2 punti
            if recent_factors:
                forecast_impact[indicator] = round(
                    np.mean([f.impact_score for f in recent_factors]), 4
                )
            else:
                forecast_impact[indicator] = 0.0

        return EconomicIndicatorsResponse(
            country=request.country_code,
            analysis_period=f"{base_date.strftime('%Y-%m-%d')} to {(base_date + timedelta(days=35)).strftime('%Y-%m-%d')}",
            economic_factors=economic_factors,
            sector_calibration={
                f"{k.replace('_sensitivity', '')}_sensitivity": v for k, v in calibration.items()
            },
            leading_indicators=leading_indicators,
            forecast_impact=forecast_impact,
        )

    except Exception as e:
        logger.error(f"Errore analisi indicatori economici: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Errore economic indicators: {str(e)}")


@router.post("/calendar-events", response_model=CalendarEventsResponse)
async def analyze_calendar_events(
    request: CalendarEventsRequest, services: Optional[Dict] = Depends(get_demand_sensing_services)
):
    """
    Analizza eventi calendario per identificare impatti su domanda (festività, eventi sportivi, etc).

    <h4>Tipi Eventi e Impatto Tipico:</h4>
    <table >
        <tr><th>Tipo Evento</th><th>Durata Impatto</th><th>Variazione Domanda</th></tr>
        <tr><td>Festività Nazionali</td><td>3-5 giorni</td><td>-20% a +50% settore-specifico</td></tr>
        <tr><td>Eventi Sportivi Major</td><td>1-3 giorni</td><td>+15-30% food/beverage</td></tr>
        <tr><td>Black Friday/Sales</td><td>1-7 giorni</td><td>+100-300% retail</td></tr>
        <tr><td>Concerti/Festival</td><td>1-2 giorni</td><td>+25-50% hospitality locale</td></tr>
        <tr><td>Meteo Estremo</td><td>1-3 giorni</td><td>-30-80% mobilità dipendente</td></tr>
    </table>

    <h4>Esempio Richiesta:</h4>
    <pre><code>
    {
        "country_code": "IT",
        "custom_events": [
            {
                "name": "Champions League Final",
                "date": "2024-06-01",
                "impact_score": 0.25,
                "duration_days": 1
            }
        ],
        "event_types": ["holidays", "sports", "cultural"],
        "impact_radius_days": 3,
        "forecast_days": 30
    }
    </code></pre>

    <h4>Risposta di Esempio:</h4>
    <pre><code>
    {
        "country": "IT",
        "forecast_period": "2024-08-23 to 2024-09-23",
        "calendar_events": [
            {
                "date": "2024-08-15T00:00:00",
                "event_name": "Ferragosto",
                "event_type": "holiday",
                "impact_score": -0.35,
                "duration_days": 1,
                "pre_impact_days": 2,
                "post_impact_days": 1,
                "confidence": 0.95
            }
        ],
        "holiday_impact_analysis": {
            "total_holiday_days": 3,
            "avg_impact_per_holiday": -0.25,
            "business_closure_days": 1
        },
        "seasonal_adjustments": {
            "august_vacation_factor": 0.75,
            "back_to_school_boost": 1.15
        },
        "high_impact_dates": ["2024-08-15", "2024-09-01"]
    }
    </code></pre>
    """
    try:
        # Database eventi per paese (semplificato)
        country_holidays = {
            "IT": [
                {"name": "Ferragosto", "date": "08-15", "impact": -0.35, "type": "holiday"},
                {"name": "Ognissanti", "date": "11-01", "impact": -0.20, "type": "holiday"},
                {"name": "Immacolata", "date": "12-08", "impact": -0.15, "type": "holiday"},
                {"name": "Natale", "date": "12-25", "impact": -0.50, "type": "holiday"},
                {"name": "Santo Stefano", "date": "12-26", "impact": -0.40, "type": "holiday"},
                {"name": "Capodanno", "date": "01-01", "impact": -0.45, "type": "holiday"},
                {"name": "Epifania", "date": "01-06", "impact": -0.25, "type": "holiday"},
                {"name": "Festa Repubblica", "date": "06-02", "impact": -0.20, "type": "holiday"},
            ],
            "US": [
                {"name": "Thanksgiving", "date": "11-25", "impact": 0.30, "type": "holiday"},
                {"name": "Black Friday", "date": "11-26", "impact": 1.50, "type": "business"},
                {"name": "Christmas", "date": "12-25", "impact": -0.30, "type": "holiday"},
                {"name": "New Year", "date": "01-01", "impact": -0.40, "type": "holiday"},
            ],
            "DE": [
                {"name": "Oktoberfest", "date": "09-16", "impact": 0.40, "type": "cultural"},
                {"name": "Christmas Markets", "date": "12-01", "impact": 0.20, "type": "cultural"},
            ],
        }

        # Eventi sportivi ricorrenti
        sports_events = [
            {"name": "Champions League Final", "month": 6, "impact": 0.25, "duration": 1},
            {"name": "World Cup Final", "month": 7, "impact": 0.40, "duration": 1},
            {"name": "Super Bowl", "month": 2, "impact": 0.35, "duration": 1},
            {"name": "Olympics Opening", "month": 8, "impact": 0.20, "duration": 16},
        ]

        base_date = datetime.now()
        calendar_events = []

        # Processa festività del paese
        holidays = country_holidays.get(request.country_code, [])
        current_year = base_date.year

        for holiday in holidays:
            # Converte data da MM-DD a datetime dell'anno corrente/prossimo
            month, day = map(int, holiday["date"].split("-"))
            event_date = datetime(current_year, month, day)

            # Se data già passata, usa anno prossimo
            if event_date < base_date:
                event_date = datetime(current_year + 1, month, day)

            # Controlla se rientra nel forecast period
            if event_date <= base_date + timedelta(days=request.forecast_days):
                calendar_events.append(
                    CalendarEvent(
                        date=event_date,
                        event_name=holiday["name"],
                        event_type=holiday["type"],
                        impact_score=holiday["impact"],
                        duration_days=1,
                        pre_impact_days=request.impact_radius_days,
                        post_impact_days=request.impact_radius_days // 2,
                        confidence=0.95,  # Alta confidenza per festività ufficiali
                    )
                )

        # Aggiungi eventi custom se forniti
        if request.custom_events:
            for custom_event in request.custom_events:
                event_date = datetime.fromisoformat(
                    custom_event["date"].replace("Z", "+00:00")
                ).replace(tzinfo=None)
                if base_date <= event_date <= base_date + timedelta(days=request.forecast_days):
                    calendar_events.append(
                        CalendarEvent(
                            date=event_date,
                            event_name=custom_event["name"],
                            event_type=custom_event.get("type", "custom"),
                            impact_score=custom_event.get("impact_score", 0.1),
                            duration_days=custom_event.get("duration_days", 1),
                            pre_impact_days=request.impact_radius_days,
                            post_impact_days=request.impact_radius_days // 2,
                            confidence=0.80,  # Confidenza media per eventi custom
                        )
                    )

        # Aggiungi eventi sportivi se richiesti e nel periodo
        if "sports" in request.event_types:
            for sport_event in sports_events:
                # Controlla se evento cade nel periodo forecast
                if sport_event["month"] in range(base_date.month, base_date.month + 2):
                    # Stima data (semplificata)
                    event_date = base_date.replace(month=sport_event["month"], day=15)
                    if event_date >= base_date and event_date <= base_date + timedelta(
                        days=request.forecast_days
                    ):
                        calendar_events.append(
                            CalendarEvent(
                                date=event_date,
                                event_name=sport_event["name"],
                                event_type="sports",
                                impact_score=sport_event["impact"],
                                duration_days=sport_event["duration"],
                                pre_impact_days=1,
                                post_impact_days=1,
                                confidence=0.70,  # Confidenza media - date possono variare
                            )
                        )

        # Analisi impatto festività
        holiday_events = [e for e in calendar_events if e.event_type == "holiday"]
        holiday_impact_analysis = {
            "total_holiday_days": len(holiday_events),
            "avg_impact_per_holiday": round(np.mean([e.impact_score for e in holiday_events]), 3)
            if holiday_events
            else 0.0,
            "business_closure_days": len([e for e in holiday_events if e.impact_score < -0.3]),
            "positive_impact_holidays": len([e for e in holiday_events if e.impact_score > 0]),
        }

        # Aggiustamenti stagionali basati su mese
        current_month = base_date.month
        seasonal_adjustments = {}

        if current_month in [6, 7, 8]:  # Estate
            seasonal_adjustments["summer_vacation_factor"] = 0.80
            seasonal_adjustments["tourism_boost"] = 1.25
        elif current_month in [9, 10]:  # Autunno
            seasonal_adjustments["back_to_school_boost"] = 1.15
            seasonal_adjustments["autumn_adjustment"] = 1.05
        elif current_month in [11, 12]:  # Inverno/Natale
            seasonal_adjustments["holiday_shopping_boost"] = 1.40
            seasonal_adjustments["winter_adjustment"] = 0.95
        else:  # Primavera
            seasonal_adjustments["spring_revival"] = 1.10
            seasonal_adjustments["post_holiday_normalization"] = 1.00

        # Identifica date alto impatto
        high_impact_dates = []
        for event in calendar_events:
            if abs(event.impact_score) > 0.15:  # Soglia significatività
                high_impact_dates.append(event.date.strftime("%Y-%m-%d"))

        # Ordina eventi per data
        calendar_events.sort(key=lambda x: x.date)

        return CalendarEventsResponse(
            country=request.country_code,
            forecast_period=f"{base_date.strftime('%Y-%m-%d')} to {(base_date + timedelta(days=request.forecast_days)).strftime('%Y-%m-%d')}",
            calendar_events=calendar_events,
            holiday_impact_analysis=holiday_impact_analysis,
            seasonal_adjustments=seasonal_adjustments,
            high_impact_dates=high_impact_dates,
        )

    except Exception as e:
        logger.error(f"Errore analisi eventi calendario: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Errore calendar events: {str(e)}")


@router.post("/ensemble-forecast", response_model=EnsembleForecastResponse)
async def generate_ensemble_forecast(
    request: EnsembleForecastRequest,
    services: Optional[Dict] = Depends(get_demand_sensing_services),
):
    """
    Genera previsioni ensemble combinando forecast base con fattori esterni multi-source.

    <h4>Algoritmo Ensemble Weighting:</h4>
    <table >
        <tr><th>Fonte</th><th>Peso Default</th><th>Condizioni Boost</th></tr>
        <tr><td>Base Forecast</td><td>40%</td><td>Alta accuratezza storica</td></tr>
        <tr><td>Weather</td><td>20%</td><td>Business weather-sensitive</td></tr>
        <tr><td>Trends</td><td>15%</td><td>Alta correlazione keyword-domanda</td></tr>
        <tr><td>Sentiment</td><td>10%</td><td>Brand-focused, alta volatilità</td></tr>
        <tr><td>Economic</td><td>10%</td><td>Luxury goods, economic sensitivity</td></tr>
        <tr><td>Calendar</td><td>5%</td><td>Eventi high-impact identificati</td></tr>
    </table>

    <h4>Esempio Richiesta:</h4>
    <pre><code>
    {
        "base_forecast": [120, 125, 118, 135, 142, 138, 145],
        "forecast_dates": ["2024-08-24", "2024-08-25", "2024-08-26", "2024-08-27", "2024-08-28", "2024-08-29", "2024-08-30"],
        "weather_factors": [
            {
                "date": "2024-08-24T12:00:00",
                "impact_score": 0.15,
                "confidence": 0.85
            }
        ],
        "ensemble_weights": {
            "base": 0.4,
            "weather": 0.25,
            "trends": 0.15,
            "sentiment": 0.1,
            "economic": 0.05,
            "calendar": 0.05
        }
    }
    </code></pre>

    <h4>Risposta di Esempio:</h4>
    <pre><code>
    {
        "forecast_id": "ens-pqr678",
        "base_forecast": [120, 125, 118, 135, 142, 138, 145],
        "adjusted_forecast": [138.2, 127.5, 122.1, 140.8, 145.6, 141.2, 149.3],
        "adjustment_factors": [1.152, 1.020, 1.035, 1.043, 1.025, 1.023, 1.030],
        "confidence_intervals": {
            "lower_95": [125.1, 115.4, 110.2, 127.5, 131.8, 127.9, 135.1],
            "upper_95": [151.3, 139.6, 134.0, 154.1, 159.4, 154.5, 163.5]
        },
        "factor_contributions": {
            "weather": [18.2, 2.5, 4.1, 5.8, 3.6, 3.2, 4.3],
            "trends": [8.1, 1.2, 2.8, 3.2, 2.1, 1.8, 2.9]
        },
        "total_adjustment": 0.045,
        "confidence_score": 0.82,
        "details": {
            "dominant_factors": ["weather", "trends"],
            "low_confidence_days": 2,
            "max_adjustment_day": "2024-08-24"
        }
    }
    </code></pre>
    """
    try:
        forecast_id = f"ens-{uuid.uuid4().hex[:8]}"

        if len(request.base_forecast) != len(request.forecast_dates):
            raise HTTPException(
                status_code=400,
                detail="Base forecast e forecast dates devono avere stessa lunghezza",
            )

        # Converte date string in datetime
        forecast_dates_dt = [datetime.fromisoformat(date) for date in request.forecast_dates]

        # Inizializza adjustment factors e contributions
        adjustment_factors = [1.0] * len(request.base_forecast)
        factor_contributions = {
            "weather": [0.0] * len(request.base_forecast),
            "trends": [0.0] * len(request.base_forecast),
            "sentiment": [0.0] * len(request.base_forecast),
            "economic": [0.0] * len(request.base_forecast),
            "calendar": [0.0] * len(request.base_forecast),
        }

        # Processa fattori meteo
        if request.weather_factors:
            weather_weight = request.ensemble_weights.get("weather", 0.2)
            for weather_factor in request.weather_factors:
                # Trova date corrispondenti
                weather_date = weather_factor.date.date()
                for i, forecast_date in enumerate(forecast_dates_dt):
                    if forecast_date.date() == weather_date:
                        impact = (
                            weather_factor.impact_score * weather_factor.confidence * weather_weight
                        )
                        adjustment_factors[i] += impact
                        factor_contributions["weather"][i] = impact * request.base_forecast[i]

        # Processa fattori trends
        if request.trends_factors:
            trends_weight = request.ensemble_weights.get("trends", 0.15)
            for trend_factor in request.trends_factors:
                trend_date = trend_factor.date.date()
                for i, forecast_date in enumerate(forecast_dates_dt):
                    if forecast_date.date() == trend_date:
                        impact = trend_factor.impact_score * trend_factor.confidence * trends_weight
                        adjustment_factors[i] += impact
                        factor_contributions["trends"][i] = impact * request.base_forecast[i]

        # Processa fattori sentiment
        if request.sentiment_factors:
            sentiment_weight = request.ensemble_weights.get("sentiment", 0.1)
            # Aggrega sentiment per data
            daily_sentiment = {}
            for sentiment_factor in request.sentiment_factors:
                date_key = sentiment_factor.date.date()
                if date_key not in daily_sentiment:
                    daily_sentiment[date_key] = []
                daily_sentiment[date_key].append(
                    sentiment_factor.impact_score * sentiment_factor.confidence
                )

            for date_key, impacts in daily_sentiment.items():
                avg_impact = np.mean(impacts)
                for i, forecast_date in enumerate(forecast_dates_dt):
                    if forecast_date.date() == date_key:
                        impact = avg_impact * sentiment_weight
                        adjustment_factors[i] += impact
                        factor_contributions["sentiment"][i] = impact * request.base_forecast[i]

        # Processa fattori economici
        if request.economic_factors:
            economic_weight = request.ensemble_weights.get("economic", 0.1)
            # Fattori economici hanno impatto distribuito nel tempo
            for economic_factor in request.economic_factors:
                impact = economic_factor.impact_score * economic_factor.confidence * economic_weight
                # Applica impatto a tutti i giorni con decay
                for i in range(len(request.base_forecast)):
                    days_diff = abs(
                        (forecast_dates_dt[i].date() - economic_factor.date.date()).days
                    )
                    decay_factor = max(0.1, 1.0 - (days_diff * 0.05))  # Decay 5% per giorno
                    distributed_impact = impact * decay_factor
                    adjustment_factors[i] += distributed_impact
                    factor_contributions["economic"][i] += (
                        distributed_impact * request.base_forecast[i]
                    )

        # Processa eventi calendario
        if request.calendar_events:
            calendar_weight = request.ensemble_weights.get("calendar", 0.05)
            for calendar_event in request.calendar_events:
                event_impact = (
                    calendar_event.impact_score * calendar_event.confidence * calendar_weight
                )

                # Applica impatto su periodo evento + pre/post impact
                event_start = calendar_event.date.date() - timedelta(
                    days=calendar_event.pre_impact_days
                )
                event_end = calendar_event.date.date() + timedelta(
                    days=calendar_event.post_impact_days
                )

                for i, forecast_date in enumerate(forecast_dates_dt):
                    if event_start <= forecast_date.date() <= event_end:
                        # Impatto massimo nel giorno evento, decay nei giorni adiacenti
                        if forecast_date.date() == calendar_event.date.date():
                            impact = event_impact
                        else:
                            days_from_event = abs(
                                (forecast_date.date() - calendar_event.date.date()).days
                            )
                            impact = event_impact * (1.0 - days_from_event * 0.2)

                        adjustment_factors[i] += impact
                        factor_contributions["calendar"][i] += impact * request.base_forecast[i]

        # Calcola forecast aggiustato
        adjusted_forecast = [
            request.base_forecast[i] * adjustment_factors[i]
            for i in range(len(request.base_forecast))
        ]

        # Calcola confidence intervals
        base_std = np.std(request.base_forecast)
        confidence_intervals = {}

        for i, adj_forecast in enumerate(adjusted_forecast):
            # Aumenta uncertainty per aggiustamenti grandi
            uncertainty_multiplier = 1.0 + abs(adjustment_factors[i] - 1.0)
            interval_width = 1.96 * base_std * uncertainty_multiplier  # 95% CI

            if "lower_95" not in confidence_intervals:
                confidence_intervals["lower_95"] = []
                confidence_intervals["upper_95"] = []

            confidence_intervals["lower_95"].append(round(max(0, adj_forecast - interval_width), 1))
            confidence_intervals["upper_95"].append(round(adj_forecast + interval_width, 1))

        # Calcola metriche aggregate
        total_base_forecast = sum(request.base_forecast)
        total_adjusted_forecast = sum(adjusted_forecast)
        total_adjustment = (
            (total_adjusted_forecast - total_base_forecast) / total_base_forecast
            if total_base_forecast > 0
            else 0
        )

        # Calcola confidence score
        # Basato su consistency dei fattori e coverage
        factor_consistency_scores = []

        for factor_name, contributions in factor_contributions.items():
            if any(c != 0 for c in contributions):
                factor_std = np.std([c for c in contributions if c != 0])
                factor_mean = np.mean([abs(c) for c in contributions if c != 0])
                consistency = 1.0 - min(1.0, factor_std / max(0.01, factor_mean))
                factor_consistency_scores.append(consistency)

        confidence_score = np.mean(factor_consistency_scores) if factor_consistency_scores else 0.5

        # Penalizza per aggiustamenti estremi
        max_adjustment = max([abs(af - 1.0) for af in adjustment_factors])
        if max_adjustment > 0.3:  # Aggiustamento >30%
            confidence_score *= 0.8

        # Dettagli analysis
        dominant_factors = []
        total_contributions = {}

        for factor_name, contributions in factor_contributions.items():
            total_contribution = sum([abs(c) for c in contributions])
            total_contributions[factor_name] = total_contribution

        # Ordina fattori per contributo
        sorted_factors = sorted(total_contributions.items(), key=lambda x: x[1], reverse=True)
        dominant_factors = [
            factor for factor, contribution in sorted_factors[:2] if contribution > 0
        ]

        low_confidence_days = len([af for af in adjustment_factors if abs(af - 1.0) > 0.2])
        max_adjustment_idx = adjustment_factors.index(
            max(adjustment_factors, key=lambda x: abs(x - 1.0))
        )
        max_adjustment_date = request.forecast_dates[max_adjustment_idx]

        details = {
            "dominant_factors": dominant_factors,
            "low_confidence_days": low_confidence_days,
            "max_adjustment_day": max_adjustment_date,
            "total_factors_used": len([f for f, c in total_contributions.items() if c > 0]),
            "avg_daily_adjustment": round(np.mean([abs(af - 1.0) for af in adjustment_factors]), 3),
        }

        # Arrotonda risultati per output pulito
        adjusted_forecast = [round(f, 1) for f in adjusted_forecast]
        adjustment_factors = [round(af, 3) for af in adjustment_factors]

        for factor_name in factor_contributions:
            factor_contributions[factor_name] = [
                round(c, 1) for c in factor_contributions[factor_name]
            ]

        return EnsembleForecastResponse(
            forecast_id=forecast_id,
            base_forecast=request.base_forecast,
            adjusted_forecast=adjusted_forecast,
            adjustment_factors=adjustment_factors,
            confidence_intervals=confidence_intervals,
            factor_contributions=factor_contributions,
            total_adjustment=round(total_adjustment, 3),
            confidence_score=round(confidence_score, 2),
            details=details,
        )

    except Exception as e:
        logger.error(f"Errore ensemble forecast: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Errore ensemble forecast: {str(e)}")


@router.get("/factor-contributions/{forecast_id}", response_model=FactorContributionsResponse)
async def analyze_factor_contributions(
    forecast_id: str,
    period_days: int = Query(30, description="Giorni periodo analisi"),
    services: Optional[Dict] = Depends(get_demand_sensing_services),
):
    """
    Analizza contributi storici dei fattori per ottimizzazione pesi ensemble.

    <h4>Analisi Contributi Fattori:</h4>
    <table >
        <tr><th>Metrica</th><th>Descrizione</th><th>Utilizzo</th></tr>
        <tr><td>Contribution %</td><td>% contributo al miglioramento forecast</td><td>Peso relativo fattori</td></tr>
        <tr><td>Correlation Matrix</td><td>Correlazioni tra fattori</td><td>Ridondanza detection</td></tr>
        <tr><td>Importance Ranking</td><td>Ranking importanza per business</td><td>Focus prioritario</td></tr>
        <tr><td>Optimization Suggestions</td><td>Raccomandazioni pesi</td><td>Tuning algoritmo</td></tr>
    </table>

    <h4>Risposta di Esempio:</h4>
    <pre><code>
    {
        "analysis_id": "contrib-stu901",
        "forecast_period": "2024-07-24 to 2024-08-23",
        "contributions_by_source": {
            "weather": {"accuracy_improvement": 0.15, "frequency": 0.8, "avg_magnitude": 0.12},
            "trends": {"accuracy_improvement": 0.08, "frequency": 0.6, "avg_magnitude": 0.05}
        },
        "correlation_matrix": {
            "weather": {"trends": 0.25, "sentiment": -0.10},
            "trends": {"sentiment": 0.45, "economic": 0.30}
        },
        "importance_ranking": [
            {"factor": "weather", "importance_score": 0.85, "business_impact": "High"},
            {"factor": "trends", "importance_score": 0.72, "business_impact": "Medium"}
        ],
        "optimization_suggestions": [
            "Increase weather weight to 0.25 (+0.05) - consistently high impact",
            "Consider reducing sentiment weight to 0.08 (-0.02) - low recent correlation"
        ]
    }
    </code></pre>
    """
    try:
        analysis_id = f"contrib-{uuid.uuid4().hex[:8]}"

        # Simula analisi contributi storica (in produzione attingeremmo da database)
        np.random.seed(42)
        base_date = datetime.now() - timedelta(days=period_days)

        # Simula contributi storici per fattore
        factors = ["weather", "trends", "sentiment", "economic", "calendar"]
        contributions_by_source = {}

        for factor in factors:
            # Simula metriche di performance storica
            if factor == "weather":
                accuracy_improvement = 0.15
                frequency = 0.80  # 80% giorni ha impatto meteo
                avg_magnitude = 0.12
            elif factor == "trends":
                accuracy_improvement = 0.08
                frequency = 0.60
                avg_magnitude = 0.05
            elif factor == "sentiment":
                accuracy_improvement = 0.04
                frequency = 0.40
                avg_magnitude = 0.03
            elif factor == "economic":
                accuracy_improvement = 0.06
                frequency = 0.20  # Meno frequente ma impact duraturo
                avg_magnitude = 0.08
            else:  # calendar
                accuracy_improvement = 0.12
                frequency = 0.15  # Pochi giorni ma impact alto
                avg_magnitude = 0.25

            contributions_by_source[factor] = {
                "accuracy_improvement": accuracy_improvement,
                "frequency": frequency,
                "avg_magnitude": avg_magnitude,
                "total_contribution": round(accuracy_improvement * frequency, 3),
            }

        # Calcola correlation matrix tra fattori
        correlation_matrix = {}
        for i, factor1 in enumerate(factors):
            correlation_matrix[factor1] = {}
            for j, factor2 in enumerate(factors):
                if i != j:
                    # Simula correlazioni realistiche
                    if (factor1, factor2) in [("weather", "trends"), ("trends", "weather")]:
                        correlation = 0.25  # Meteo-trends correlazione moderata
                    elif (factor1, factor2) in [("trends", "sentiment"), ("sentiment", "trends")]:
                        correlation = 0.45  # Trends-sentiment alta correlazione
                    elif (factor1, factor2) in [
                        ("economic", "sentiment"),
                        ("sentiment", "economic"),
                    ]:
                        correlation = 0.35  # Economic-sentiment moderata
                    elif (factor1, factor2) in [("weather", "sentiment"), ("sentiment", "weather")]:
                        correlation = -0.10  # Meteo-sentiment lievemente negativa
                    else:
                        correlation = np.random.uniform(-0.1, 0.2)  # Correlazioni casuali basse

                    correlation_matrix[factor1][factor2] = round(correlation, 3)

        # Calcola importance ranking
        importance_ranking = []
        for factor in factors:
            contrib = contributions_by_source[factor]

            # Score combinato: accuracy * frequency * magnitude
            importance_score = (
                contrib["accuracy_improvement"] * contrib["frequency"] * contrib["avg_magnitude"]
            )

            # Determina business impact level
            if importance_score > 0.08:
                business_impact = "High"
            elif importance_score > 0.03:
                business_impact = "Medium"
            else:
                business_impact = "Low"

            importance_ranking.append(
                {
                    "factor": factor,
                    "importance_score": round(importance_score, 3),
                    "business_impact": business_impact,
                    "rank": 0,  # Sarà assegnato dopo sorting
                }
            )

        # Ordina per importance
        importance_ranking.sort(key=lambda x: x["importance_score"], reverse=True)
        for i, item in enumerate(importance_ranking):
            item["rank"] = i + 1

        # Genera optimization suggestions
        optimization_suggestions = []

        # Top performer - suggerisce aumento peso
        top_factor = importance_ranking[0]
        if top_factor["importance_score"] > 0.08:
            optimization_suggestions.append(
                f"Increase {top_factor['factor']} weight to 0.25 (+0.05) - consistently high impact"
            )

        # Low performer - suggerisce riduzione peso
        low_performers = [item for item in importance_ranking if item["importance_score"] < 0.02]
        for performer in low_performers:
            optimization_suggestions.append(
                f"Consider reducing {performer['factor']} weight to 0.08 (-0.02) - low recent correlation"
            )

        # High correlation - suggerisce consolidamento
        high_correlations = []
        for factor1, correlations in correlation_matrix.items():
            for factor2, corr in correlations.items():
                if corr > 0.4:
                    high_correlations.append((factor1, factor2, corr))

        for factor1, factor2, corr in high_correlations:
            optimization_suggestions.append(
                f"High correlation between {factor1} and {factor2} ({corr:.2f}) - consider composite factor"
            )

        # Business-specific suggestions
        if contributions_by_source["weather"]["frequency"] > 0.7:
            optimization_suggestions.append(
                "Weather factor active 70%+ days - consider weather-specialized model"
            )

        if not optimization_suggestions:
            optimization_suggestions.append(
                "Current ensemble weights appear well-balanced - no major adjustments needed"
            )

        return FactorContributionsResponse(
            analysis_id=analysis_id,
            forecast_period=f"{base_date.strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')}",
            contributions_by_source=contributions_by_source,
            correlation_matrix=correlation_matrix,
            importance_ranking=importance_ranking,
            optimization_suggestions=optimization_suggestions,
        )

    except Exception as e:
        logger.error(f"Errore analisi contributi: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Errore factor contributions: {str(e)}")


@router.post("/sensitivity-analysis", response_model=SensitivityAnalysisResponse)
async def perform_sensitivity_analysis(
    request: SensitivityAnalysisRequest,
    services: Optional[Dict] = Depends(get_demand_sensing_services),
):
    """
    Esegue analisi sensibilità per ottimizzare pesi ensemble e identificare parametri critici.

    <h4>Metodologia Sensitivity Analysis:</h4>
    <table >
        <tr><th>Fase</th><th>Descrizione</th><th>Output</th></tr>
        <tr><td>Parameter Sweep</td><td>Test pesi nel range ±20%</td><td>Performance grid</td></tr>
        <tr><td>Cross Validation</td><td>Valida su historical data</td><td>Accuracy metrics</td></tr>
        <tr><td>Optimization</td><td>Identifica pesi ottimali</td><td>Best weights configuration</td></tr>
        <tr><td>Robustness Test</td><td>Testa stabilità configurazione</td><td>Confidence intervals</td></tr>
    </table>

    <h4>Esempio Richiesta:</h4>
    <pre><code>
    {
        "historical_forecasts": [
            {
                "base_forecast": [120, 125, 118],
                "weather_impact": [0.1, 0.05, 0.15],
                "trends_impact": [0.02, 0.08, 0.04],
                "ensemble_weights": {"weather": 0.2, "trends": 0.15}
            }
        ],
        "actual_values": [138, 130, 142],
        "sensitivity_range": 0.2,
        "factors_to_test": ["weather", "trends", "sentiment"]
    }
    </code></pre>

    <h4>Risposta di Esempio:</h4>
    <pre><code>
    {
        "analysis_id": "sens-vwx234",
        "optimal_weights": {
            "weather": 0.25,
            "trends": 0.12,
            "sentiment": 0.08
        },
        "sensitivity_scores": {
            "weather": 0.85,
            "trends": 0.62,
            "sentiment": 0.40
        },
        "forecast_improvement": 0.18,
        "weight_adjustments": {
            "weather": {"from": 0.20, "to": 0.25, "improvement": 0.12},
            "trends": {"from": 0.15, "to": 0.12, "improvement": 0.05}
        },
        "validation_metrics": {
            "mae_improvement": 0.15,
            "rmse_improvement": 0.22,
            "mape_improvement": 0.18
        }
    }
    </code></pre>
    """
    try:
        analysis_id = f"sens-{uuid.uuid4().hex[:8]}"

        if len(request.historical_forecasts) == 0:
            raise HTTPException(
                status_code=400, detail="Servono almeno alcuni forecast storici per l'analisi"
            )

        # Simula sensitivity analysis (in produzione useremmo algoritmo di ottimizzazione)
        np.random.seed(42)

        # Pesi correnti (estratti dai forecast storici)
        current_weights = {}
        if request.historical_forecasts:
            first_forecast = request.historical_forecasts[0]
            ensemble_weights = first_forecast.get("ensemble_weights", {})
            for factor in request.factors_to_test:
                current_weights[factor] = ensemble_weights.get(factor, 0.1)

        # Testa variazioni pesi nel sensitivity range
        best_weights = {}
        sensitivity_scores = {}
        weight_adjustments = {}

        for factor in request.factors_to_test:
            current_weight = current_weights.get(factor, 0.1)

            # Test range di pesi
            min_weight = max(0.01, current_weight * (1 - request.sensitivity_range))
            max_weight = min(0.5, current_weight * (1 + request.sensitivity_range))

            best_performance = 0
            best_weight = current_weight

            # Simula test di diversi pesi
            for test_weight in np.linspace(min_weight, max_weight, 10):
                # Simula performance con questo peso
                # In produzione: ricalcoleremmo forecast con nuovo peso e misureremmo accuracy

                if factor == "weather":
                    # Weather tende a performare meglio con pesi più alti
                    performance = 0.5 + (test_weight - 0.1) * 2.0 + np.random.normal(0, 0.05)
                elif factor == "trends":
                    # Trends ha optimum intorno a 0.12
                    performance = 0.6 - abs(test_weight - 0.12) * 3.0 + np.random.normal(0, 0.05)
                elif factor == "sentiment":
                    # Sentiment ha performance decrescente con peso alto
                    performance = 0.4 + (0.15 - test_weight) * 1.5 + np.random.normal(0, 0.05)
                else:
                    # Fattori generic
                    performance = 0.5 + np.random.normal(0, 0.1)

                performance = max(0, min(1, performance))  # Clamp 0-1

                if performance > best_performance:
                    best_performance = performance
                    best_weight = test_weight

            best_weights[factor] = round(best_weight, 3)
            sensitivity_scores[factor] = round(best_performance, 3)

            # Calcola adjustment
            weight_change = best_weight - current_weight
            improvement = (best_performance - 0.5) * 0.3  # Converti in % improvement

            weight_adjustments[factor] = {
                "from": round(current_weight, 3),
                "to": round(best_weight, 3),
                "change": round(weight_change, 3),
                "improvement": round(improvement, 3),
            }

        # Calcola forecast improvement aggregato
        improvements = [adj["improvement"] for adj in weight_adjustments.values()]
        forecast_improvement = round(np.mean(improvements), 3)

        # Simula validation metrics
        # In produzione: applicheremmo pesi ottimali ai dati storici
        base_mae_improvement = forecast_improvement * 0.8
        base_rmse_improvement = forecast_improvement * 1.2
        base_mape_improvement = forecast_improvement * 1.0

        validation_metrics = {
            "mae_improvement": round(base_mae_improvement, 3),
            "rmse_improvement": round(base_rmse_improvement, 3),
            "mape_improvement": round(base_mape_improvement, 3),
            "total_forecasts_tested": len(request.historical_forecasts),
            "avg_accuracy_gain": round(forecast_improvement, 3),
        }

        return SensitivityAnalysisResponse(
            analysis_id=analysis_id,
            optimal_weights=best_weights,
            sensitivity_scores=sensitivity_scores,
            forecast_improvement=round(forecast_improvement, 3),
            weight_adjustments=weight_adjustments,
            validation_metrics=validation_metrics,
        )

    except Exception as e:
        logger.error(f"Errore sensitivity analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Errore sensitivity analysis: {str(e)}")
