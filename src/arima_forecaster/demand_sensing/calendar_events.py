"""
Gestione eventi calendario per Demand Sensing.

Monitora festività, eventi sportivi, fiere e altri eventi che impattano domanda.
"""

import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from enum import Enum

from .demand_sensor import ExternalFactor, FactorType
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class EventType(str, Enum):
    """Tipi di eventi calendario."""

    HOLIDAY = "holiday"  # Festività nazionale
    RELIGIOUS = "religious"  # Festività religiosa
    SPORT = "sport"  # Evento sportivo
    CONCERT = "concert"  # Concerto/spettacolo
    FAIR = "fair"  # Fiera/expo
    SEASONAL = "seasonal"  # Evento stagionale (es. saldi)
    LOCAL = "local"  # Evento locale
    BLACKFRIDAY = "blackfriday"  # Black Friday/Cyber Monday
    CUSTOM = "custom"  # Evento personalizzato


class Event(BaseModel):
    """Evento calendario."""

    name: str
    date: datetime
    type: EventType
    duration_days: int = 1
    impact_radius_days: int = Field(7, description="Giorni prima/dopo con impatto")
    expected_impact: float = Field(0.1, description="Impatto atteso su domanda (-1 a 1)")
    confidence: float = Field(0.8, description="Confidenza impatto")
    recurring: bool = False
    metadata: Dict = Field(default_factory=dict)


class EventImpact(BaseModel):
    """Configurazione impatto eventi su domanda."""

    # Moltiplicatori per tipo evento
    event_multipliers: Dict[str, float] = Field(
        default_factory=lambda: {
            "holiday": 0.20,  # +20% festività
            "religious": 0.15,  # +15% feste religiose
            "sport": 0.10,  # +10% eventi sportivi
            "concert": 0.08,
            "fair": 0.12,
            "seasonal": 0.25,  # Saldi molto impattanti
            "local": 0.05,
            "blackfriday": 0.40,  # Black Friday boom
            "custom": 0.10,
        }
    )

    # Pattern temporale impatto (giorni prima/dopo)
    impact_pattern: Dict[str, List[float]] = Field(
        default_factory=lambda: {
            "holiday": [0.3, 0.5, 0.8, 1.0, 0.6, 0.3, 0.1],  # Picco il giorno
            "blackfriday": [0.4, 0.6, 0.8, 1.0, 1.0, 0.7, 0.3],  # Prolungato
            "seasonal": [0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 0.8],  # Crescente
            "default": [0.2, 0.4, 0.7, 1.0, 0.5, 0.2, 0.1],
        }
    )

    # Categorie prodotto più impattate
    category_sensitivity: Dict[str, Dict[str, float]] = Field(
        default_factory=lambda: {
            "gift": {"holiday": 2.0, "religious": 1.8, "seasonal": 1.5},
            "food": {"holiday": 1.5, "religious": 1.3, "sport": 1.2},
            "electronics": {"blackfriday": 2.5, "seasonal": 1.8},
            "clothing": {"seasonal": 2.0, "blackfriday": 1.8},
            "travel": {"holiday": 2.5, "seasonal": 1.5},
            "entertainment": {"sport": 2.0, "concert": 2.5},
            "default": {"holiday": 1.2, "seasonal": 1.3},
        }
    )


class CalendarEvents:
    """
    Gestore eventi calendario per demand sensing.
    """

    def __init__(
        self,
        country: str = "IT",
        product_category: str = "default",
        impact_config: Optional[EventImpact] = None,
        custom_events: Optional[List[Event]] = None,
    ):
        """
        Inizializza gestore eventi.

        Args:
            country: Codice paese per festività
            product_category: Categoria prodotto
            impact_config: Configurazione impatti
            custom_events: Eventi personalizzati
        """
        self.country = country
        self.product_category = product_category
        self.impact_config = impact_config or EventImpact()
        self.custom_events = custom_events or []

        # Inizializza calendario festività
        self._init_holidays()

        logger.info(f"CalendarEvents inizializzato per {country}")

    def _init_holidays(self):
        """Inizializza festività per paese."""

        # Festività Italia 2024-2025
        self.holidays = {
            "IT": [
                Event(
                    name="Capodanno",
                    date=datetime(2025, 1, 1),
                    type=EventType.HOLIDAY,
                    impact_radius_days=3,
                    expected_impact=0.15,
                ),
                Event(
                    name="Epifania",
                    date=datetime(2025, 1, 6),
                    type=EventType.RELIGIOUS,
                    impact_radius_days=2,
                    expected_impact=0.10,
                ),
                Event(
                    name="Pasqua",
                    date=datetime(2025, 4, 20),
                    type=EventType.RELIGIOUS,
                    impact_radius_days=5,
                    expected_impact=0.20,
                ),
                Event(
                    name="Pasquetta",
                    date=datetime(2025, 4, 21),
                    type=EventType.HOLIDAY,
                    impact_radius_days=2,
                    expected_impact=0.12,
                ),
                Event(
                    name="Festa della Liberazione",
                    date=datetime(2025, 4, 25),
                    type=EventType.HOLIDAY,
                    impact_radius_days=2,
                    expected_impact=0.08,
                ),
                Event(
                    name="Festa del Lavoro",
                    date=datetime(2025, 5, 1),
                    type=EventType.HOLIDAY,
                    impact_radius_days=2,
                    expected_impact=0.08,
                ),
                Event(
                    name="Festa della Repubblica",
                    date=datetime(2025, 6, 2),
                    type=EventType.HOLIDAY,
                    impact_radius_days=2,
                    expected_impact=0.08,
                ),
                Event(
                    name="Ferragosto",
                    date=datetime(2025, 8, 15),
                    type=EventType.HOLIDAY,
                    impact_radius_days=7,
                    expected_impact=0.25,
                ),
                Event(
                    name="Ognissanti",
                    date=datetime(2025, 11, 1),
                    type=EventType.RELIGIOUS,
                    impact_radius_days=2,
                    expected_impact=0.08,
                ),
                Event(
                    name="Immacolata",
                    date=datetime(2025, 12, 8),
                    type=EventType.RELIGIOUS,
                    impact_radius_days=3,
                    expected_impact=0.15,
                ),
                Event(
                    name="Natale",
                    date=datetime(2024, 12, 25),
                    type=EventType.RELIGIOUS,
                    impact_radius_days=10,
                    expected_impact=0.35,
                ),
                Event(
                    name="Santo Stefano",
                    date=datetime(2024, 12, 26),
                    type=EventType.HOLIDAY,
                    impact_radius_days=2,
                    expected_impact=0.15,
                ),
                Event(
                    name="San Silvestro",
                    date=datetime(2024, 12, 31),
                    type=EventType.HOLIDAY,
                    impact_radius_days=3,
                    expected_impact=0.20,
                ),
                # Eventi commerciali
                Event(
                    name="Black Friday",
                    date=datetime(2024, 11, 29),
                    type=EventType.BLACKFRIDAY,
                    impact_radius_days=5,
                    expected_impact=0.40,
                ),
                Event(
                    name="Cyber Monday",
                    date=datetime(2024, 12, 2),
                    type=EventType.BLACKFRIDAY,
                    impact_radius_days=2,
                    expected_impact=0.30,
                ),
                Event(
                    name="Saldi Invernali",
                    date=datetime(2025, 1, 5),
                    type=EventType.SEASONAL,
                    duration_days=30,
                    impact_radius_days=5,
                    expected_impact=0.25,
                ),
                Event(
                    name="Saldi Estivi",
                    date=datetime(2025, 7, 1),
                    type=EventType.SEASONAL,
                    duration_days=30,
                    impact_radius_days=5,
                    expected_impact=0.25,
                ),
                Event(
                    name="San Valentino",
                    date=datetime(2025, 2, 14),
                    type=EventType.HOLIDAY,
                    impact_radius_days=5,
                    expected_impact=0.15,
                ),
                Event(
                    name="Festa della Mamma",
                    date=datetime(2025, 5, 11),
                    type=EventType.HOLIDAY,
                    impact_radius_days=5,
                    expected_impact=0.12,
                ),
                Event(
                    name="Festa del Papà",
                    date=datetime(2025, 3, 19),
                    type=EventType.HOLIDAY,
                    impact_radius_days=3,
                    expected_impact=0.10,
                ),
            ],
            "US": [
                Event(
                    name="New Year",
                    date=datetime(2025, 1, 1),
                    type=EventType.HOLIDAY,
                    expected_impact=0.15,
                ),
                Event(
                    name="MLK Day",
                    date=datetime(2025, 1, 20),
                    type=EventType.HOLIDAY,
                    expected_impact=0.05,
                ),
                Event(
                    name="Presidents Day",
                    date=datetime(2025, 2, 17),
                    type=EventType.HOLIDAY,
                    expected_impact=0.08,
                ),
                Event(
                    name="Easter",
                    date=datetime(2025, 4, 20),
                    type=EventType.RELIGIOUS,
                    expected_impact=0.15,
                ),
                Event(
                    name="Memorial Day",
                    date=datetime(2025, 5, 26),
                    type=EventType.HOLIDAY,
                    expected_impact=0.12,
                ),
                Event(
                    name="Independence Day",
                    date=datetime(2025, 7, 4),
                    type=EventType.HOLIDAY,
                    expected_impact=0.20,
                ),
                Event(
                    name="Labor Day",
                    date=datetime(2025, 9, 1),
                    type=EventType.HOLIDAY,
                    expected_impact=0.10,
                ),
                Event(
                    name="Thanksgiving",
                    date=datetime(2024, 11, 28),
                    type=EventType.HOLIDAY,
                    impact_radius_days=5,
                    expected_impact=0.30,
                ),
                Event(
                    name="Black Friday",
                    date=datetime(2024, 11, 29),
                    type=EventType.BLACKFRIDAY,
                    impact_radius_days=5,
                    expected_impact=0.45,
                ),
                Event(
                    name="Christmas",
                    date=datetime(2024, 12, 25),
                    type=EventType.RELIGIOUS,
                    impact_radius_days=10,
                    expected_impact=0.35,
                ),
            ],
            "default": [
                Event(
                    name="New Year",
                    date=datetime(2025, 1, 1),
                    type=EventType.HOLIDAY,
                    expected_impact=0.15,
                ),
                Event(
                    name="Easter",
                    date=datetime(2025, 4, 20),
                    type=EventType.RELIGIOUS,
                    expected_impact=0.15,
                ),
                Event(
                    name="Christmas",
                    date=datetime(2024, 12, 25),
                    type=EventType.RELIGIOUS,
                    expected_impact=0.30,
                ),
            ],
        }

    def get_events(
        self,
        start_date: datetime,
        end_date: datetime,
        event_types: Optional[List[EventType]] = None,
    ) -> List[Event]:
        """
        Recupera eventi in un periodo.

        Args:
            start_date: Data inizio
            end_date: Data fine
            event_types: Tipi eventi da includere

        Returns:
            Lista eventi nel periodo
        """
        # Eventi paese
        country_events = self.holidays.get(self.country, self.holidays["default"])

        # Combina con eventi custom
        all_events = country_events + self.custom_events

        # Filtra per periodo
        events_in_period = []
        for event in all_events:
            # Calcola range impatto evento
            event_start = event.date - timedelta(days=event.impact_radius_days)
            event_end = event.date + timedelta(days=event.duration_days + event.impact_radius_days)

            # Verifica overlap con periodo richiesto
            if event_end >= start_date and event_start <= end_date:
                if event_types is None or event.type in event_types:
                    events_in_period.append(event)

        # Ordina per data
        events_in_period.sort(key=lambda x: x.date)

        logger.info(f"Trovati {len(events_in_period)} eventi nel periodo")
        return events_in_period

    def calculate_event_impact(
        self, events: List[Event], forecast_horizon: int = 30, base_date: Optional[datetime] = None
    ) -> List[ExternalFactor]:
        """
        Calcola impatto eventi sulla domanda.

        Args:
            events: Eventi da analizzare
            forecast_horizon: Giorni previsione
            base_date: Data base per forecast

        Returns:
            Lista fattori esterni eventi
        """
        if base_date is None:
            base_date = datetime.now()

        factors = []

        # Sensibilità categoria prodotto
        category_sens = self.impact_config.category_sensitivity.get(
            self.product_category, self.impact_config.category_sensitivity["default"]
        )

        # Per ogni giorno del forecast
        for day_offset in range(forecast_horizon):
            forecast_date = base_date + timedelta(days=day_offset)

            # Calcola impatto aggregato di tutti gli eventi
            daily_impact = 0
            daily_confidence = 0
            impacting_events = []

            for event in events:
                # Calcola distanza temporale dall'evento
                days_to_event = (event.date.date() - forecast_date.date()).days
                days_from_event = -days_to_event

                # Verifica se siamo nel raggio di impatto
                if abs(days_to_event) <= event.impact_radius_days:
                    # Pattern temporale impatto
                    pattern = self.impact_config.impact_pattern.get(
                        event.type.value, self.impact_config.impact_pattern["default"]
                    )

                    # Indice nel pattern
                    pattern_idx = days_from_event + event.impact_radius_days
                    if 0 <= pattern_idx < len(pattern):
                        pattern_weight = pattern[pattern_idx]
                    else:
                        pattern_weight = 0.1

                    # Moltiplicatore tipo evento
                    type_mult = self.impact_config.event_multipliers.get(event.type.value, 0.1)

                    # Sensibilità categoria
                    cat_mult = category_sens.get(event.type.value, 1.0)

                    # Calcola impatto evento
                    event_impact = event.expected_impact * pattern_weight * type_mult * cat_mult

                    daily_impact += event_impact
                    daily_confidence = max(daily_confidence, event.confidence)

                    impacting_events.append(
                        {
                            "name": event.name,
                            "type": event.type.value,
                            "days_to": days_to_event,
                            "impact": event_impact,
                        }
                    )

            # Se ci sono eventi impattanti, crea fattore
            if impacting_events:
                # Limita impatto totale
                daily_impact = max(-0.5, min(0.5, daily_impact))

                # Media pesata confidenza
                if len(impacting_events) > 1:
                    # Riduci confidenza per eventi multipli
                    daily_confidence *= 0.9

                factor = ExternalFactor(
                    name=f"Calendar_{forecast_date.strftime('%Y-%m-%d')}",
                    type=FactorType.CALENDAR,
                    value=len(impacting_events),  # Numero eventi
                    impact=daily_impact,
                    confidence=daily_confidence,
                    timestamp=forecast_date,
                    metadata={
                        "events": impacting_events,
                        "dominant_event": max(impacting_events, key=lambda x: abs(x["impact"]))[
                            "name"
                        ],
                        "product_category": self.product_category,
                    },
                )

                factors.append(factor)

                logger.debug(
                    f"Calendar {forecast_date}: {len(impacting_events)} eventi, "
                    f"impact={daily_impact:.2%}"
                )

        return factors

    def add_custom_event(
        self,
        name: str,
        date: datetime,
        event_type: EventType = EventType.CUSTOM,
        expected_impact: float = 0.1,
        **kwargs,
    ) -> None:
        """
        Aggiunge evento personalizzato.

        Args:
            name: Nome evento
            date: Data evento
            event_type: Tipo evento
            expected_impact: Impatto atteso
            **kwargs: Altri parametri Event
        """
        event = Event(
            name=name, date=date, type=event_type, expected_impact=expected_impact, **kwargs
        )

        self.custom_events.append(event)
        logger.info(f"Aggiunto evento custom: {name} il {date}")

    def get_next_major_event(
        self, from_date: Optional[datetime] = None, min_impact: float = 0.15
    ) -> Optional[Event]:
        """
        Trova prossimo evento maggiore.

        Args:
            from_date: Data da cui cercare
            min_impact: Impatto minimo per considerare "major"

        Returns:
            Prossimo evento major o None
        """
        if from_date is None:
            from_date = datetime.now()

        # Cerca nei prossimi 365 giorni
        events = self.get_events(from_date, from_date + timedelta(days=365))

        # Filtra per impatto
        major_events = [e for e in events if abs(e.expected_impact) >= min_impact]

        if major_events:
            return major_events[0]  # Già ordinati per data

        return None

    def get_seasonal_factor(self, target_date: datetime) -> float:
        """
        Calcola fattore stagionale base.

        Args:
            target_date: Data target

        Returns:
            Moltiplicatore stagionale
        """
        month = target_date.month

        # Pattern stagionale generico
        seasonal_patterns = {
            1: 0.9,  # Gennaio post-feste
            2: 0.85,  # Febbraio basso
            3: 0.95,  # Marzo ripresa
            4: 1.0,  # Aprile normale
            5: 1.05,  # Maggio buono
            6: 1.1,  # Giugno pre-estate
            7: 1.15,  # Luglio estate
            8: 1.1,  # Agosto estate/ferie
            9: 1.0,  # Settembre ripresa
            10: 1.05,  # Ottobre
            11: 1.15,  # Novembre pre-festività
            12: 1.25,  # Dicembre festività
        }

        return seasonal_patterns.get(month, 1.0)
