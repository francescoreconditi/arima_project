"""
Integrazione indicatori economici per Demand Sensing.

Monitora indicatori macro-economici per prevedere impatti su domanda.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from enum import Enum

from .demand_sensor import ExternalFactor, FactorType
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class EconomicIndicator(str, Enum):
    """Tipi di indicatori economici."""

    GDP = "gdp"  # PIL
    INFLATION = "inflation"  # Inflazione
    UNEMPLOYMENT = "unemployment"  # Disoccupazione
    CONSUMER_CONFIDENCE = "consumer_confidence"  # Fiducia consumatori
    RETAIL_SALES = "retail_sales"  # Vendite retail
    INTEREST_RATE = "interest_rate"  # Tasso interesse
    EXCHANGE_RATE = "exchange_rate"  # Tasso cambio
    OIL_PRICE = "oil_price"  # Prezzo petrolio
    STOCK_INDEX = "stock_index"  # Indice borsa


class EconomicData(BaseModel):
    """Dato economico."""

    indicator: EconomicIndicator
    value: float
    date: datetime
    previous_value: Optional[float] = None
    change_percent: Optional[float] = None
    forecast: Optional[float] = None
    source: str = "demo"


class EconomicImpact(BaseModel):
    """Configurazione impatto indicatori economici."""

    # Impatto per indicatore (elasticità domanda)
    indicator_elasticity: Dict[str, float] = Field(
        default_factory=lambda: {
            "gdp": 0.8,  # +1% PIL = +0.8% domanda
            "inflation": -0.5,  # +1% inflazione = -0.5% domanda
            "unemployment": -0.6,  # +1% disoccupazione = -0.6% domanda
            "consumer_confidence": 1.2,  # Forte correlazione
            "retail_sales": 0.9,
            "interest_rate": -0.3,  # Tassi alti riducono consumi
            "exchange_rate": 0.2,  # Impatto su import/export
            "oil_price": -0.2,  # Costi trasporto
            "stock_index": 0.4,  # Effetto ricchezza
        }
    )

    # Lag temporale per indicatore (giorni)
    indicator_lag: Dict[str, int] = Field(
        default_factory=lambda: {
            "gdp": 90,  # Impatto dopo 3 mesi
            "inflation": 30,
            "unemployment": 60,
            "consumer_confidence": 15,
            "retail_sales": 7,
            "interest_rate": 45,
            "exchange_rate": 7,
            "oil_price": 14,
            "stock_index": 3,
        }
    )

    # Peso per settore
    sector_sensitivity: Dict[str, Dict[str, float]] = Field(
        default_factory=lambda: {
            "luxury": {
                "gdp": 1.5,
                "consumer_confidence": 2.0,
                "stock_index": 1.8,
                "unemployment": -1.5,
            },
            "essential": {
                "gdp": 0.3,
                "inflation": -0.2,
                "unemployment": -0.3,
                "consumer_confidence": 0.5,
            },
            "technology": {
                "gdp": 1.0,
                "interest_rate": -0.5,
                "stock_index": 1.2,
                "consumer_confidence": 1.3,
            },
            "retail": {
                "retail_sales": 1.5,
                "consumer_confidence": 1.0,
                "unemployment": -0.8,
                "inflation": -0.6,
            },
            "automotive": {
                "gdp": 1.2,
                "interest_rate": -1.0,
                "oil_price": -0.5,
                "consumer_confidence": 1.5,
            },
            "default": {
                "gdp": 0.8,
                "inflation": -0.4,
                "unemployment": -0.5,
                "consumer_confidence": 0.8,
            },
        }
    )


class EconomicIndicators:
    """
    Gestore indicatori economici per demand sensing.
    """

    def __init__(
        self,
        country: str = "IT",
        sector: str = "default",
        impact_config: Optional[EconomicImpact] = None,
        api_key: Optional[str] = None,
    ):
        """
        Inizializza gestore indicatori.

        Args:
            country: Codice paese (IT, US, etc.)
            sector: Settore economico
            impact_config: Configurazione impatti
            api_key: API key per servizi dati economici
        """
        self.country = country
        self.sector = sector
        self.impact_config = impact_config or EconomicImpact()
        self.api_key = api_key

        # Cache dati
        self._cache = {}

        # Valori baseline per paese (demo)
        self.baseline_values = {
            "IT": {
                "gdp": 1.5,  # Crescita annuale %
                "inflation": 2.5,
                "unemployment": 8.5,
                "consumer_confidence": 105,
                "retail_sales": 2.0,
                "interest_rate": 4.5,
                "exchange_rate": 1.08,  # EUR/USD
                "oil_price": 85,
                "stock_index": 28000,  # FTSE MIB
            },
            "US": {
                "gdp": 2.5,
                "inflation": 3.5,
                "unemployment": 3.8,
                "consumer_confidence": 110,
                "retail_sales": 3.0,
                "interest_rate": 5.5,
                "exchange_rate": 1.0,
                "oil_price": 85,
                "stock_index": 35000,  # S&P 500
            },
            "default": {
                "gdp": 2.0,
                "inflation": 3.0,
                "unemployment": 6.0,
                "consumer_confidence": 100,
                "retail_sales": 2.5,
                "interest_rate": 4.0,
                "exchange_rate": 1.0,
                "oil_price": 80,
                "stock_index": 10000,
            },
        }

        logger.info(f"EconomicIndicators inizializzato per {country}, settore {sector}")

    def fetch_indicators(
        self, indicators: List[EconomicIndicator] = None, use_demo_data: bool = True
    ) -> List[EconomicData]:
        """
        Recupera indicatori economici.

        Args:
            indicators: Indicatori da recuperare
            use_demo_data: Usa dati demo

        Returns:
            Lista dati economici
        """
        if indicators is None:
            indicators = [
                EconomicIndicator.GDP,
                EconomicIndicator.INFLATION,
                EconomicIndicator.CONSUMER_CONFIDENCE,
                EconomicIndicator.UNEMPLOYMENT,
            ]

        if use_demo_data or not self.api_key:
            return self._generate_demo_indicators(indicators)

        # Qui andrebbero chiamate API reali come:
        # - FRED (Federal Reserve Economic Data)
        # - World Bank API
        # - IMF Data API
        # - Trading Economics API

        logger.warning("API economiche non configurate. Uso dati demo.")
        return self._generate_demo_indicators(indicators)

    def _generate_demo_indicators(self, indicators: List[EconomicIndicator]) -> List[EconomicData]:
        """
        Genera indicatori demo realistici.

        Args:
            indicators: Indicatori richiesti

        Returns:
            Lista dati demo
        """
        data = []
        np.random.seed(42)

        baseline = self.baseline_values.get(self.country, self.baseline_values["default"])

        for indicator in indicators:
            base_value = baseline.get(indicator.value, 100)

            # Simula variazione recente
            change = np.random.uniform(-0.1, 0.1)  # +/-10%
            current_value = base_value * (1 + change)

            # Simula trend
            if indicator == EconomicIndicator.GDP:
                # PIL tende a crescere
                current_value = abs(current_value)
                forecast = current_value * 1.02
            elif indicator == EconomicIndicator.INFLATION:
                # Inflazione volatile
                forecast = current_value * np.random.uniform(0.9, 1.1)
            elif indicator == EconomicIndicator.UNEMPLOYMENT:
                # Disoccupazione ciclica
                forecast = current_value * np.random.uniform(0.95, 1.05)
            else:
                forecast = current_value * np.random.uniform(0.98, 1.02)

            data.append(
                EconomicData(
                    indicator=indicator,
                    value=round(current_value, 2),
                    date=datetime.now(),
                    previous_value=round(base_value, 2),
                    change_percent=round(change * 100, 2),
                    forecast=round(forecast, 2),
                    source="demo",
                )
            )

        logger.info(f"Generati {len(data)} indicatori economici demo")
        return data

    def calculate_economic_impact(
        self, indicators: List[EconomicData], forecast_horizon: int = 30
    ) -> List[ExternalFactor]:
        """
        Calcola impatto economico sulla domanda.

        Args:
            indicators: Dati economici
            forecast_horizon: Giorni previsione

        Returns:
            Lista fattori esterni economici
        """
        factors = []

        # Sensibilità settore
        sector_sens = self.impact_config.sector_sensitivity.get(
            self.sector, self.impact_config.sector_sensitivity["default"]
        )

        for day in range(forecast_horizon):
            forecast_date = datetime.now() + timedelta(days=day)

            # Calcola impatto aggregato
            total_impact = 0
            weighted_confidence = 0
            impact_details = []

            for data in indicators:
                indicator_name = data.indicator.value

                # Elasticità base
                base_elasticity = self.impact_config.indicator_elasticity.get(indicator_name, 0.5)

                # Aggiusta per settore
                sector_adjustment = sector_sens.get(indicator_name, 1.0)
                elasticity = base_elasticity * sector_adjustment

                # Calcola cambio percentuale
                if data.previous_value and data.previous_value != 0:
                    change_pct = (data.value - data.previous_value) / data.previous_value
                else:
                    change_pct = 0

                # Calcola impatto con lag
                lag_days = self.impact_config.indicator_lag.get(indicator_name, 30)
                lag_factor = max(0, 1 - (day / lag_days)) if day < lag_days else 0

                # Impatto indicatore
                indicator_impact = change_pct * elasticity * lag_factor
                total_impact += indicator_impact

                # Confidenza basata su forecast accuracy (simulata)
                if data.forecast:
                    forecast_error = abs(data.forecast - data.value) / data.value
                    confidence = max(0.3, 1 - forecast_error)
                else:
                    confidence = 0.5

                weighted_confidence += confidence * abs(indicator_impact)

                impact_details.append(
                    {
                        "indicator": indicator_name,
                        "value": data.value,
                        "change": change_pct,
                        "impact": indicator_impact,
                    }
                )

            # Normalizza confidenza
            if total_impact != 0:
                avg_confidence = weighted_confidence / abs(total_impact)
            else:
                avg_confidence = 0.5

            # Limita impatto totale
            total_impact = max(-0.3, min(0.3, total_impact))

            # Crea fattore
            factor = ExternalFactor(
                name=f"Economic_{forecast_date.strftime('%Y-%m-%d')}",
                type=FactorType.ECONOMIC,
                value=sum(d.value for d in indicators) / len(indicators),
                impact=total_impact,
                confidence=avg_confidence,
                timestamp=forecast_date,
                metadata={
                    "country": self.country,
                    "sector": self.sector,
                    "indicators": impact_details,
                    "dominant_factor": max(impact_details, key=lambda x: abs(x["impact"]))[
                        "indicator"
                    ],
                },
            )

            factors.append(factor)

            logger.debug(
                f"Economic {forecast_date}: impact={total_impact:.2%}, "
                f"confidence={avg_confidence:.2f}"
            )

        return factors

    def get_economic_outlook(self, indicators: List[EconomicData]) -> Dict[str, str]:
        """
        Genera outlook economico qualitativo.

        Args:
            indicators: Dati economici

        Returns:
            Outlook per indicatore
        """
        outlook = {}

        for data in indicators:
            name = data.indicator.value

            # Analizza trend
            if data.previous_value:
                change = data.value - data.previous_value
                change_pct = (change / data.previous_value) * 100
            else:
                change = 0
                change_pct = 0

            # Genera outlook testuale
            if name == "gdp":
                if change_pct > 2:
                    outlook[name] = "Forte crescita economica"
                elif change_pct > 0:
                    outlook[name] = "Crescita moderata"
                elif change_pct > -2:
                    outlook[name] = "Stagnazione"
                else:
                    outlook[name] = "Recessione"

            elif name == "inflation":
                if data.value > 5:
                    outlook[name] = "Inflazione elevata - pressione su prezzi"
                elif data.value > 3:
                    outlook[name] = "Inflazione moderata"
                elif data.value > 0:
                    outlook[name] = "Inflazione sotto controllo"
                else:
                    outlook[name] = "Rischio deflazione"

            elif name == "unemployment":
                if data.value < 4:
                    outlook[name] = "Piena occupazione"
                elif data.value < 7:
                    outlook[name] = "Disoccupazione normale"
                elif data.value < 10:
                    outlook[name] = "Disoccupazione elevata"
                else:
                    outlook[name] = "Crisi occupazionale"

            elif name == "consumer_confidence":
                if data.value > 110:
                    outlook[name] = "Fiducia molto alta - forte domanda"
                elif data.value > 100:
                    outlook[name] = "Fiducia positiva"
                elif data.value > 90:
                    outlook[name] = "Fiducia in calo"
                else:
                    outlook[name] = "Pessimismo consumatori"

            else:
                if change_pct > 5:
                    outlook[name] = "Forte crescita"
                elif change_pct > 0:
                    outlook[name] = "Crescita moderata"
                elif change_pct > -5:
                    outlook[name] = "Stabile"
                else:
                    outlook[name] = "In calo"

        return outlook

    def get_recession_probability(self, indicators: List[EconomicData]) -> float:
        """
        Calcola probabilità recessione.

        Args:
            indicators: Dati economici

        Returns:
            Probabilità 0-1
        """
        signals = 0
        weights = 0

        for data in indicators:
            if data.indicator == EconomicIndicator.GDP:
                if data.value < 0:
                    signals += 0.3
                weights += 0.3

            elif data.indicator == EconomicIndicator.UNEMPLOYMENT:
                if data.change_percent and data.change_percent > 10:
                    signals += 0.2
                weights += 0.2

            elif data.indicator == EconomicIndicator.CONSUMER_CONFIDENCE:
                if data.value < 90:
                    signals += 0.2
                weights += 0.2

            elif data.indicator == EconomicIndicator.RETAIL_SALES:
                if data.value < 0:
                    signals += 0.15
                weights += 0.15

            elif data.indicator == EconomicIndicator.INTEREST_RATE:
                if data.value > 6:
                    signals += 0.15
                weights += 0.15

        if weights > 0:
            probability = signals / weights
        else:
            probability = 0.1  # Default bassa

        return min(1.0, max(0.0, probability))
