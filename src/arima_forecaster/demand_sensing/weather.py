"""
Integrazione con dati meteorologici per Demand Sensing.

Supporta OpenWeatherMap API e dati meteo simulati per demo.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
import requests
import json
from functools import lru_cache

from .demand_sensor import ExternalFactor, FactorType
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class WeatherCondition(BaseModel):
    """Condizione meteorologica."""
    
    date: datetime
    temperature: float  # Celsius
    humidity: float  # Percentuale
    precipitation: float  # mm
    wind_speed: float  # km/h
    condition: str  # clear, clouds, rain, snow
    feels_like: float  # Temperatura percepita
    pressure: float  # hPa
    uv_index: Optional[float] = None


class WeatherImpact(BaseModel):
    """Configurazione impatto meteo su domanda."""
    
    # Impatto temperatura (per grado Celsius dalla media)
    temp_impact_per_degree: float = Field(
        0.02,
        description="Impatto % per grado di differenza dalla media"
    )
    
    # Soglie temperatura
    cold_threshold: float = Field(10.0, description="Soglia freddo (C)")
    hot_threshold: float = Field(25.0, description="Soglia caldo (C)")
    optimal_temp: float = Field(20.0, description="Temperatura ottimale (C)")
    
    # Impatto precipitazioni
    rain_impact: float = Field(-0.15, description="Impatto pioggia")
    heavy_rain_impact: float = Field(-0.30, description="Impatto pioggia forte")
    snow_impact: float = Field(-0.40, description="Impatto neve")
    
    # Impatto per categoria prodotto
    product_sensitivity: Dict[str, float] = Field(
        default_factory=lambda: {
            'beverage_cold': 2.0,  # Molto sensibile al caldo
            'beverage_hot': -1.5,  # Inverso per bevande calde
            'ice_cream': 3.0,  # Altissima sensibilità
            'umbrella': -2.0,  # Vendite inverse (pioggia aumenta)
            'clothing_summer': 1.5,
            'clothing_winter': -1.5,
            'outdoor_equipment': 1.0,
            'default': 0.5
        }
    )


class WeatherIntegration:
    """
    Integrazione con servizi meteo per demand sensing.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        location: str = "Milan,IT",
        impact_config: Optional[WeatherImpact] = None,
        use_cache: bool = True
    ):
        """
        Inizializza integrazione meteo.
        
        Args:
            api_key: API key per OpenWeatherMap (opzionale per demo)
            location: Località per previsioni
            impact_config: Configurazione impatti
            use_cache: Usa cache per ridurre chiamate API
        """
        self.api_key = api_key
        self.location = location
        self.impact_config = impact_config or WeatherImpact()
        self.use_cache = use_cache
        
        # URL base OpenWeatherMap
        self.base_url = "https://api.openweathermap.org/data/2.5"
        
        # Cache per dati storici
        self._cache = {}
        
        # Media storica per località (demo)
        self.historical_avg = {
            'Milan,IT': {'temp': 15, 'rain_days': 80},
            'Rome,IT': {'temp': 17, 'rain_days': 75},
            'London,UK': {'temp': 11, 'rain_days': 150},
            'Madrid,ES': {'temp': 16, 'rain_days': 60},
            'Paris,FR': {'temp': 12, 'rain_days': 110},
            'Berlin,DE': {'temp': 10, 'rain_days': 100},
            'default': {'temp': 15, 'rain_days': 90}
        }
        
        logger.info(f"WeatherIntegration inizializzata per {location}")
    
    @lru_cache(maxsize=128)
    def fetch_forecast(
        self,
        days_ahead: int = 7,
        use_demo_data: bool = False
    ) -> List[WeatherCondition]:
        """
        Recupera previsioni meteo.
        
        Args:
            days_ahead: Giorni di previsione
            use_demo_data: Usa dati demo invece di API reale
            
        Returns:
            Lista di condizioni meteo
        """
        if use_demo_data or not self.api_key:
            return self._generate_demo_forecast(days_ahead)
        
        try:
            # Chiamata API OpenWeatherMap
            url = f"{self.base_url}/forecast/daily"
            params = {
                'q': self.location,
                'appid': self.api_key,
                'units': 'metric',
                'cnt': days_ahead
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            conditions = []
            
            for day in data['list']:
                condition = WeatherCondition(
                    date=datetime.fromtimestamp(day['dt']),
                    temperature=day['temp']['day'],
                    feels_like=day['feels_like']['day'],
                    humidity=day['humidity'],
                    precipitation=day.get('rain', 0) + day.get('snow', 0),
                    wind_speed=day['speed'] * 3.6,  # m/s to km/h
                    condition=day['weather'][0]['main'].lower(),
                    pressure=day['pressure'],
                    uv_index=day.get('uvi')
                )
                conditions.append(condition)
            
            logger.info(f"Recuperate {len(conditions)} previsioni meteo da API")
            return conditions
            
        except Exception as e:
            logger.warning(f"Errore API meteo: {e}. Uso dati demo.")
            return self._generate_demo_forecast(days_ahead)
    
    def _generate_demo_forecast(self, days: int) -> List[WeatherCondition]:
        """
        Genera previsioni demo realistiche.
        
        Args:
            days: Numero di giorni
            
        Returns:
            Lista condizioni meteo simulate
        """
        np.random.seed(42)  # Per riproducibilità
        
        # Media storica per location
        hist = self.historical_avg.get(
            self.location,
            self.historical_avg['default']
        )
        
        conditions = []
        base_date = datetime.now()
        
        # Stagionalità (semplificata)
        month = base_date.month
        if month in [12, 1, 2]:  # Inverno
            temp_adjust = -5
            rain_prob = 0.3
        elif month in [3, 4, 5]:  # Primavera
            temp_adjust = 0
            rain_prob = 0.25
        elif month in [6, 7, 8]:  # Estate
            temp_adjust = 8
            rain_prob = 0.15
        else:  # Autunno
            temp_adjust = -2
            rain_prob = 0.35
        
        for i in range(days):
            date = base_date + timedelta(days=i)
            
            # Temperatura con variazione casuale e trend
            temp = hist['temp'] + temp_adjust + np.random.normal(0, 3)
            
            # Condizioni
            rand = np.random.random()
            if rand < rain_prob:
                if temp < 2:
                    condition = 'snow'
                    precip = np.random.uniform(5, 20)
                else:
                    condition = 'rain'
                    precip = np.random.uniform(2, 15)
            elif rand < rain_prob + 0.3:
                condition = 'clouds'
                precip = 0
            else:
                condition = 'clear'
                precip = 0
            
            conditions.append(WeatherCondition(
                date=date,
                temperature=round(temp, 1),
                feels_like=round(temp + np.random.uniform(-2, 2), 1),
                humidity=np.random.uniform(40, 90),
                precipitation=round(precip, 1),
                wind_speed=round(np.random.uniform(5, 25), 1),
                condition=condition,
                pressure=round(np.random.uniform(990, 1030)),
                uv_index=max(0, round(np.random.uniform(0, 11))) if condition == 'clear' else 0
            ))
        
        logger.info(f"Generate {len(conditions)} previsioni meteo demo")
        return conditions
    
    def calculate_weather_impact(
        self,
        conditions: List[WeatherCondition],
        product_category: str = 'default',
        base_demand: Optional[float] = None
    ) -> List[ExternalFactor]:
        """
        Calcola l'impatto delle condizioni meteo sulla domanda.
        
        Args:
            conditions: Condizioni meteo
            product_category: Categoria prodotto
            base_demand: Domanda base (per calibrazione)
            
        Returns:
            Lista di fattori esterni meteo
        """
        factors = []
        
        # Sensitivity per categoria
        sensitivity = self.impact_config.product_sensitivity.get(
            product_category,
            self.impact_config.product_sensitivity['default']
        )
        
        # Media storica
        hist_temp = self.historical_avg.get(
            self.location,
            self.historical_avg['default']
        )['temp']
        
        for condition in conditions:
            # Calcola impatto temperatura
            temp_diff = condition.temperature - hist_temp
            temp_impact = (
                temp_diff * 
                self.impact_config.temp_impact_per_degree * 
                sensitivity
            ) / 100
            
            # Impatto condizioni
            condition_impact = 0
            if condition.condition == 'rain':
                if condition.precipitation > 10:
                    condition_impact = self.impact_config.heavy_rain_impact
                else:
                    condition_impact = self.impact_config.rain_impact
            elif condition.condition == 'snow':
                condition_impact = self.impact_config.snow_impact
            
            # Aggiusta per categoria (es. ombrelli vendono di più con pioggia)
            if product_category == 'umbrella' and condition.condition in ['rain', 'snow']:
                condition_impact = abs(condition_impact) * 2
            
            # Combina impatti
            total_impact = temp_impact + condition_impact
            
            # Limita impatto totale
            total_impact = max(-0.5, min(0.5, total_impact))
            
            # Calcola confidenza basata su distanza temporale
            days_ahead = (condition.date - datetime.now()).days
            confidence = max(0.3, 1.0 - (days_ahead * 0.1))
            
            # Crea fattore
            factor = ExternalFactor(
                name=f"Meteo_{condition.date.strftime('%Y-%m-%d')}",
                type=FactorType.WEATHER,
                value=condition.temperature,
                impact=total_impact,
                confidence=confidence,
                timestamp=condition.date,
                metadata={
                    'temperature': condition.temperature,
                    'condition': condition.condition,
                    'precipitation': condition.precipitation,
                    'humidity': condition.humidity,
                    'product_category': product_category,
                    'location': self.location
                }
            )
            
            factors.append(factor)
            
            logger.debug(
                f"Meteo {condition.date}: temp={condition.temperature}°C, "
                f"condition={condition.condition}, impact={total_impact:.2%}"
            )
        
        return factors
    
    def get_extreme_weather_alerts(
        self,
        conditions: List[WeatherCondition]
    ) -> List[str]:
        """
        Identifica condizioni meteo estreme.
        
        Args:
            conditions: Condizioni meteo
            
        Returns:
            Lista di alert
        """
        alerts = []
        
        for condition in conditions:
            # Temperature estreme
            if condition.temperature < self.impact_config.cold_threshold - 5:
                alerts.append(
                    f"FREDDO ESTREMO previsto per {condition.date.strftime('%d/%m')}: "
                    f"{condition.temperature}°C"
                )
            elif condition.temperature > self.impact_config.hot_threshold + 10:
                alerts.append(
                    f"CALDO ESTREMO previsto per {condition.date.strftime('%d/%m')}: "
                    f"{condition.temperature}°C"
                )
            
            # Precipitazioni intense
            if condition.precipitation > 30:
                alerts.append(
                    f"PRECIPITAZIONI INTENSE previste per {condition.date.strftime('%d/%m')}: "
                    f"{condition.precipitation}mm"
                )
            
            # Vento forte
            if condition.wind_speed > 50:
                alerts.append(
                    f"VENTO FORTE previsto per {condition.date.strftime('%d/%m')}: "
                    f"{condition.wind_speed}km/h"
                )
        
        return alerts
    
    def get_seasonal_pattern(
        self,
        historical_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Analizza pattern stagionali meteo-domanda.
        
        Args:
            historical_data: Dati storici (opzionale)
            
        Returns:
            Pattern stagionali
        """
        patterns = {
            'spring_effect': 0.05,  # +5% in primavera
            'summer_effect': 0.15,  # +15% in estate
            'autumn_effect': -0.05,  # -5% in autunno
            'winter_effect': -0.10,  # -10% in inverno
            'weekend_effect': 0.08,  # +8% weekend
            'holiday_effect': 0.12   # +12% festivi
        }
        
        # Se abbiamo dati storici, affina i pattern
        if historical_data is not None:
            # Analisi stagionale sui dati storici
            # (implementazione semplificata)
            pass
        
        return patterns