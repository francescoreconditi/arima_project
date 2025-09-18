"""
Integrazione con Google Trends per Demand Sensing.

Analizza trend di ricerca per prevedere la domanda.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
import time

from .demand_sensor import ExternalFactor, FactorType
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

# Importazione opzionale pytrends
try:
    from pytrends.request import TrendReq

    PYTRENDS_AVAILABLE = True
except ImportError:
    PYTRENDS_AVAILABLE = False
    logger.warning("pytrends non installato. Usando dati demo per Google Trends.")


class TrendData(BaseModel):
    """Dati di trend da Google Trends."""

    keyword: str
    date: datetime
    interest: int  # 0-100 dove 100 è il picco di interesse
    related_queries: List[str] = Field(default_factory=list)
    rising_queries: List[str] = Field(default_factory=list)
    region: str = "IT"


class TrendImpact(BaseModel):
    """Configurazione impatto trend su domanda."""

    # Soglie di interesse
    low_interest_threshold: int = Field(30, description="Sotto = basso interesse")
    high_interest_threshold: int = Field(70, description="Sopra = alto interesse")
    spike_threshold: int = Field(85, description="Sopra = spike di interesse")

    # Moltiplicatori impatto
    impact_multiplier: float = Field(
        0.005, description="Impatto per punto percentuale di interesse"
    )
    spike_multiplier: float = Field(2.0, description="Moltiplicatore aggiuntivo per spike")

    # Lag temporale (giorni tra ricerca e acquisto)
    purchase_lag_days: int = Field(3, description="Giorni medi tra ricerca e acquisto")

    # Peso per categoria
    category_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "tech_product": 1.5,  # Molto influenzato da ricerche
            "fashion": 1.2,
            "food": 0.5,  # Meno influenzato
            "service": 0.8,
            "event_ticket": 2.0,  # Altamente correlato
            "default": 1.0,
        }
    )


class GoogleTrendsIntegration:
    """
    Integrazione con Google Trends per demand sensing.
    """

    def __init__(
        self,
        keywords: List[str],
        geo: str = "IT",
        language: str = "it-IT",
        impact_config: Optional[TrendImpact] = None,
        use_cache: bool = True,
    ):
        """
        Inizializza integrazione Google Trends.

        Args:
            keywords: Parole chiave da monitorare
            geo: Codice geografico (IT, US, etc.)
            language: Lingua per l'interfaccia
            impact_config: Configurazione impatti
            use_cache: Usa cache per ridurre richieste
        """
        self.keywords = keywords
        self.geo = geo
        self.language = language
        self.impact_config = impact_config or TrendImpact()
        self.use_cache = use_cache

        # Inizializza client pytrends se disponibile
        if PYTRENDS_AVAILABLE:
            self.pytrends = TrendReq(hl=language, tz=360, geo=geo)
        else:
            self.pytrends = None

        # Cache risultati
        self._cache = {}
        self._cache_expiry = {}

        # Dati storici demo per categorie
        self.demo_patterns = {
            "tech_product": self._generate_tech_pattern,
            "fashion": self._generate_fashion_pattern,
            "food": self._generate_food_pattern,
            "event_ticket": self._generate_event_pattern,
            "default": self._generate_default_pattern,
        }

        logger.info(f"GoogleTrendsIntegration inizializzata per {len(keywords)} keywords")

    def fetch_trends(
        self, timeframe: str = "today 3-m", use_demo_data: bool = False
    ) -> pd.DataFrame:
        """
        Recupera dati trend da Google.

        Args:
            timeframe: Periodo temporale (es. "today 3-m", "2024-01-01 2024-12-31")
            use_demo_data: Usa dati demo invece di API reale

        Returns:
            DataFrame con interesse nel tempo
        """
        # Controlla cache
        cache_key = f"{','.join(self.keywords)}_{timeframe}"
        if self.use_cache and cache_key in self._cache:
            if datetime.now() < self._cache_expiry[cache_key]:
                logger.info("Usando dati trend da cache")
                return self._cache[cache_key]

        if use_demo_data or not PYTRENDS_AVAILABLE:
            df = self._generate_demo_trends(timeframe)
        else:
            try:
                # Costruisci payload
                self.pytrends.build_payload(self.keywords, timeframe=timeframe, geo=self.geo)

                # Recupera interesse nel tempo
                df = self.pytrends.interest_over_time()

                # Rimuovi colonna isPartial se presente
                if "isPartial" in df.columns:
                    df = df.drop("isPartial", axis=1)

                logger.info(f"Recuperati {len(df)} punti dati da Google Trends")

                # Aggiungi ritardo per evitare rate limiting
                time.sleep(1)

            except Exception as e:
                logger.warning(f"Errore Google Trends API: {e}. Uso dati demo.")
                df = self._generate_demo_trends(timeframe)

        # Aggiorna cache
        if self.use_cache:
            self._cache[cache_key] = df
            self._cache_expiry[cache_key] = datetime.now() + timedelta(hours=6)

        return df

    def _generate_demo_trends(self, timeframe: str) -> pd.DataFrame:
        """
        Genera dati trend demo realistici.

        Args:
            timeframe: Periodo temporale

        Returns:
            DataFrame con trend simulati
        """
        # Determina periodo
        if "today" in timeframe:
            if "3-m" in timeframe:
                days = 90
            elif "1-m" in timeframe:
                days = 30
            else:
                days = 7
            start_date = datetime.now() - timedelta(days=days)
            end_date = datetime.now()
        else:
            # Parse date custom (semplificato)
            days = 90
            start_date = datetime.now() - timedelta(days=days)
            end_date = datetime.now()

        # Genera date range
        dates = pd.date_range(start=start_date, end=end_date, freq="D")

        # Genera trend per ogni keyword
        data = {}
        for keyword in self.keywords:
            # Pattern base con stagionalità e trend
            base_interest = 50
            trend = np.random.uniform(-0.2, 0.3)  # Trend generale

            values = []
            for i, date in enumerate(dates):
                # Componente trend
                trend_value = base_interest + (i * trend)

                # Componente stagionale settimanale
                day_of_week = date.dayofweek
                if day_of_week in [5, 6]:  # Weekend
                    seasonal = 10
                else:
                    seasonal = 0

                # Componente casuale
                random_component = np.random.normal(0, 10)

                # Eventi speciali (spike occasionali)
                spike = 0
                if np.random.random() < 0.05:  # 5% probabilità spike
                    spike = np.random.uniform(20, 40)

                # Combina componenti
                value = trend_value + seasonal + random_component + spike

                # Limita tra 0 e 100
                value = max(0, min(100, value))
                values.append(int(value))

            data[keyword] = values

        df = pd.DataFrame(data, index=dates)
        logger.info(f"Generati {len(df)} punti dati trend demo")

        return df

    def _generate_tech_pattern(self, days: int) -> List[int]:
        """Pattern per prodotti tech (picchi su lanci)."""
        values = []
        base = 40
        for i in range(days):
            # Simula lancio prodotto ogni 30 giorni
            if i % 30 < 7:
                value = base + np.random.uniform(30, 50)
            else:
                value = base + np.random.uniform(-10, 10)
            values.append(int(max(0, min(100, value))))
        return values

    def _generate_fashion_pattern(self, days: int) -> List[int]:
        """Pattern per moda (stagionale)."""
        values = []
        for i in range(days):
            # Picchi cambio stagione
            month = (datetime.now() - timedelta(days=days - i)).month
            if month in [3, 4, 9, 10]:  # Cambio stagione
                base = 70
            else:
                base = 40
            value = base + np.random.uniform(-15, 15)
            values.append(int(max(0, min(100, value))))
        return values

    def _generate_food_pattern(self, days: int) -> List[int]:
        """Pattern per cibo (stabile con picchi weekend)."""
        values = []
        for i in range(days):
            date = datetime.now() - timedelta(days=days - i)
            base = 50
            # Weekend boost
            if date.weekday() in [4, 5, 6]:
                base += 15
            value = base + np.random.uniform(-10, 10)
            values.append(int(max(0, min(100, value))))
        return values

    def _generate_event_pattern(self, days: int) -> List[int]:
        """Pattern per eventi (spike prima di date specifiche)."""
        values = []
        for i in range(days):
            # Simula evento ogni 20 giorni
            days_to_event = i % 20
            if days_to_event < 7:
                # Crescita esponenziale vicino all'evento
                value = 30 + (7 - days_to_event) * 10
            else:
                value = 20
            value += np.random.uniform(-5, 5)
            values.append(int(max(0, min(100, value))))
        return values

    def _generate_default_pattern(self, days: int) -> List[int]:
        """Pattern generico."""
        return [int(50 + np.random.uniform(-20, 20)) for _ in range(days)]

    def calculate_trend_impact(
        self, trend_data: pd.DataFrame, product_category: str = "default", forecast_horizon: int = 7
    ) -> List[ExternalFactor]:
        """
        Calcola l'impatto dei trend sulla domanda.

        Args:
            trend_data: Dati trend da Google
            product_category: Categoria prodotto
            forecast_horizon: Giorni di previsione

        Returns:
            Lista di fattori esterni trend
        """
        factors = []

        # Weight per categoria
        category_weight = self.impact_config.category_weights.get(
            product_category, self.impact_config.category_weights["default"]
        )

        # Calcola media mobile per smoothing
        if len(trend_data) > 7:
            trend_smooth = trend_data.rolling(window=7, min_periods=1).mean()
        else:
            trend_smooth = trend_data

        # Calcola trend recente
        if len(trend_smooth) >= 14:
            recent_trend = trend_smooth.iloc[-7:].mean()
            previous_trend = trend_smooth.iloc[-14:-7].mean()
        else:
            recent_trend = trend_smooth.mean()
            previous_trend = recent_trend

        # Per ogni giorno di forecast
        for day in range(forecast_horizon):
            date = datetime.now() + timedelta(days=day + self.impact_config.purchase_lag_days)

            # Media interesse per tutti i keywords
            avg_interest = (
                recent_trend.mean() if isinstance(recent_trend, pd.Series) else recent_trend
            )

            # Calcola cambio trend
            if previous_trend.mean() != 0:
                trend_change = (avg_interest - previous_trend.mean()) / previous_trend.mean()
            else:
                trend_change = 0

            # Calcola impatto base
            if avg_interest < self.impact_config.low_interest_threshold:
                base_impact = -0.1  # Basso interesse = domanda ridotta
            elif avg_interest > self.impact_config.spike_threshold:
                base_impact = 0.3  # Spike = forte aumento domanda
                base_impact *= self.impact_config.spike_multiplier
            elif avg_interest > self.impact_config.high_interest_threshold:
                base_impact = 0.15  # Alto interesse
            else:
                base_impact = 0  # Interesse normale

            # Aggiusta per trend change
            trend_impact = trend_change * 0.5

            # Combina impatti
            total_impact = (base_impact + trend_impact) * category_weight

            # Limita impatto
            total_impact = max(-0.4, min(0.4, total_impact))

            # Calcola confidenza (decresce con distanza temporale)
            confidence = max(0.3, 0.9 - (day * 0.05))

            # Aggiusta confidenza per volatilità dei dati
            if len(trend_data) > 1:
                volatility = trend_data.std().mean() / 50  # Normalizza su 100
                confidence *= 1 - volatility * 0.3  # Riduci per alta volatilità

            # Crea fattore
            factor = ExternalFactor(
                name=f"Trend_{date.strftime('%Y-%m-%d')}",
                type=FactorType.TRENDS,
                value=float(avg_interest),
                impact=total_impact,
                confidence=confidence,
                timestamp=date,
                metadata={
                    "keywords": self.keywords,
                    "average_interest": float(avg_interest),
                    "trend_change": float(trend_change),
                    "product_category": product_category,
                    "geo": self.geo,
                },
            )

            factors.append(factor)

            logger.debug(
                f"Trend {date}: interest={avg_interest:.0f}, "
                f"change={trend_change:.2%}, impact={total_impact:.2%}"
            )

        return factors

    def get_related_queries(self, use_demo: bool = False) -> Dict[str, List[str]]:
        """
        Recupera query correlate per espandere keywords.

        Args:
            use_demo: Usa dati demo

        Returns:
            Dict con query correlate per keyword
        """
        related = {}

        if use_demo or not PYTRENDS_AVAILABLE:
            # Dati demo
            demo_related = {
                "smartphone": ["iphone 15", "samsung galaxy", "xiaomi", "offerte smartphone"],
                "giacca invernale": [
                    "piumino",
                    "cappotto donna",
                    "giubbotto uomo",
                    "saldi giacche",
                ],
                "pizza": [
                    "pizza domicilio",
                    "pizzeria vicino",
                    "pizza napoletana",
                    "pizza margherita",
                ],
                "concerto": ["biglietti concerto", "eventi milano", "concerti 2024", "ticketone"],
                "default": ["prodotto simile", "alternativa", "offerta", "recensioni"],
            }

            for keyword in self.keywords:
                related[keyword] = demo_related.get(keyword, demo_related["default"])
        else:
            try:
                for keyword in self.keywords:
                    self.pytrends.build_payload([keyword], geo=self.geo)
                    queries = self.pytrends.related_queries()

                    if keyword in queries and queries[keyword]["rising"] is not None:
                        related[keyword] = queries[keyword]["rising"]["query"].tolist()[:5]
                    else:
                        related[keyword] = []

                    time.sleep(1)  # Rate limiting

            except Exception as e:
                logger.warning(f"Errore recupero query correlate: {e}")
                return self.get_related_queries(use_demo=True)

        return related

    def detect_trending_topics(self, threshold: int = 80) -> List[Tuple[str, int]]:
        """
        Identifica topic in forte trend.

        Args:
            threshold: Soglia interesse per considerare trending

        Returns:
            Lista di (keyword, interesse) per topic trending
        """
        trend_data = self.fetch_trends("today 1-m", use_demo_data=not PYTRENDS_AVAILABLE)

        trending = []
        for keyword in self.keywords:
            if keyword in trend_data.columns:
                recent_interest = trend_data[keyword].iloc[-7:].mean()
                if recent_interest >= threshold:
                    trending.append((keyword, int(recent_interest)))

        # Ordina per interesse
        trending.sort(key=lambda x: x[1], reverse=True)

        return trending
