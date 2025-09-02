"""
Modulo principale per Demand Sensing - Integrazione fattori esterni nelle previsioni.

Questo modulo coordina l'integrazione di vari fattori esterni per migliorare
l'accuratezza delle previsioni di domanda.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum
from dataclasses import dataclass
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, validator
import warnings

from ..core.arima_model import ARIMAForecaster
from ..core.sarima_model import SARIMAForecaster
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class AdjustmentStrategy(str, Enum):
    """Strategie di aggiustamento per i fattori esterni."""
    
    ADDITIVE = "additive"  # Aggiustamento additivo
    MULTIPLICATIVE = "multiplicative"  # Aggiustamento moltiplicativo
    HYBRID = "hybrid"  # Combinazione additiva e moltiplicativa
    WEIGHTED = "weighted"  # Media pesata
    MACHINE_LEARNING = "ml"  # Modello ML per combinare fattori


class FactorType(str, Enum):
    """Tipi di fattori esterni."""
    
    WEATHER = "weather"
    TRENDS = "trends"
    SOCIAL = "social"
    ECONOMIC = "economic"
    CALENDAR = "calendar"
    COMPETITOR = "competitor"
    CUSTOM = "custom"


class ImpactLevel(str, Enum):
    """Livelli di impatto dei fattori esterni."""
    
    VERY_LOW = "very_low"  # < 2% impatto
    LOW = "low"  # 2-5% impatto
    MEDIUM = "medium"  # 5-10% impatto
    HIGH = "high"  # 10-20% impatto
    VERY_HIGH = "very_high"  # > 20% impatto


class ExternalFactor(BaseModel):
    """Modello per un fattore esterno."""
    
    name: str = Field(..., description="Nome del fattore")
    type: FactorType = Field(..., description="Tipo di fattore")
    value: float = Field(..., description="Valore corrente del fattore")
    impact: float = Field(0.0, description="Impatto stimato (-1 a 1)")
    confidence: float = Field(0.5, description="Confidenza nell'impatto (0-1)")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('impact')
    def validate_impact(cls, v):
        if not -1 <= v <= 1:
            raise ValueError("Impact deve essere tra -1 e 1")
        return v
    
    @validator('confidence')
    def validate_confidence(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Confidence deve essere tra 0 e 1")
        return v


class FactorImpact(BaseModel):
    """Risultato dell'analisi di impatto di un fattore."""
    
    factor: ExternalFactor
    original_forecast: float
    adjusted_forecast: float
    adjustment_amount: float
    adjustment_percentage: float
    impact_level: ImpactLevel
    explanation: str
    
    @property
    def is_significant(self) -> bool:
        """Verifica se l'impatto è significativo (>5%)."""
        return abs(self.adjustment_percentage) > 5.0


class SensingConfig(BaseModel):
    """Configurazione per il Demand Sensing."""
    
    strategy: AdjustmentStrategy = Field(
        AdjustmentStrategy.WEIGHTED,
        description="Strategia di aggiustamento"
    )
    max_adjustment: float = Field(
        0.3,
        description="Massimo aggiustamento permesso (30%)"
    )
    min_confidence: float = Field(
        0.3,
        description="Confidenza minima per applicare aggiustamento"
    )
    enable_weather: bool = Field(True, description="Abilita integrazione meteo")
    enable_trends: bool = Field(True, description="Abilita Google Trends")
    enable_social: bool = Field(True, description="Abilita social sentiment")
    enable_economic: bool = Field(True, description="Abilita indicatori economici")
    enable_calendar: bool = Field(True, description="Abilita eventi calendario")
    historical_weight: float = Field(
        0.7,
        description="Peso dato alle previsioni storiche vs fattori esterni"
    )
    learning_rate: float = Field(
        0.1,
        description="Tasso di apprendimento per aggiustamenti adattivi"
    )
    
    @validator('max_adjustment')
    def validate_max_adjustment(cls, v):
        if not 0 < v <= 1:
            raise ValueError("max_adjustment deve essere tra 0 e 1")
        return v


@dataclass
class SensingResult:
    """Risultato del demand sensing."""
    
    original_forecast: pd.Series
    adjusted_forecast: pd.Series
    factors_applied: List[FactorImpact]
    total_adjustment: float
    confidence_score: float
    recommendations: List[str]
    metadata: Dict[str, Any]
    
    @property
    def improvement_percentage(self) -> float:
        """Calcola il miglioramento percentuale medio."""
        if len(self.original_forecast) == 0:
            return 0.0
        avg_original = self.original_forecast.mean()
        avg_adjusted = self.adjusted_forecast.mean()
        if avg_original == 0:
            return 0.0
        return ((avg_adjusted - avg_original) / avg_original) * 100
    
    def to_dataframe(self) -> pd.DataFrame:
        """Converte i risultati in DataFrame."""
        df = pd.DataFrame({
            'original': self.original_forecast,
            'adjusted': self.adjusted_forecast,
            'adjustment': self.adjusted_forecast - self.original_forecast,
            'adjustment_pct': ((self.adjusted_forecast - self.original_forecast) / 
                              self.original_forecast * 100)
        })
        return df


class DemandSensor:
    """
    Classe principale per Demand Sensing.
    
    Integra vari fattori esterni per migliorare le previsioni di domanda.
    """
    
    def __init__(
        self,
        base_model: Optional[Union[ARIMAForecaster, SARIMAForecaster]] = None,
        config: Optional[SensingConfig] = None
    ):
        """
        Inizializza il Demand Sensor.
        
        Args:
            base_model: Modello di forecasting base
            config: Configurazione del sensing
        """
        self.base_model = base_model
        self.config = config or SensingConfig()
        self.factors: List[ExternalFactor] = []
        self.impact_history: List[FactorImpact] = []
        self.learning_cache: Dict[str, float] = {}
        
        # Inizializza integrazioni
        self._init_integrations()
        
        logger.info(f"DemandSensor inizializzato con strategia {self.config.strategy}")
    
    def _init_integrations(self):
        """Inizializza le integrazioni con servizi esterni."""
        self.integrations = {}
        
        # Le integrazioni verranno caricate on-demand
        # per evitare dipendenze obbligatorie
        self.integration_status = {
            'weather': self.config.enable_weather,
            'trends': self.config.enable_trends,
            'social': self.config.enable_social,
            'economic': self.config.enable_economic,
            'calendar': self.config.enable_calendar
        }
    
    def add_factor(self, factor: ExternalFactor) -> None:
        """
        Aggiunge un fattore esterno.
        
        Args:
            factor: Fattore esterno da aggiungere
        """
        # Valida il fattore
        if factor.confidence < self.config.min_confidence:
            logger.warning(
                f"Fattore {factor.name} ha confidenza {factor.confidence} "
                f"sotto la soglia minima {self.config.min_confidence}"
            )
        
        self.factors.append(factor)
        logger.info(f"Aggiunto fattore {factor.name} con impatto {factor.impact}")
    
    def clear_factors(self) -> None:
        """Rimuove tutti i fattori esterni."""
        self.factors = []
        logger.info("Tutti i fattori esterni rimossi")
    
    def calculate_impact(
        self,
        factor: ExternalFactor,
        base_forecast: float
    ) -> FactorImpact:
        """
        Calcola l'impatto di un singolo fattore.
        
        Args:
            factor: Fattore da analizzare
            base_forecast: Previsione base
            
        Returns:
            Impatto del fattore
        """
        # Calcola aggiustamento basato su strategia
        if self.config.strategy == AdjustmentStrategy.MULTIPLICATIVE:
            adjustment_factor = 1 + (factor.impact * factor.confidence)
            adjusted_value = base_forecast * adjustment_factor
        
        elif self.config.strategy == AdjustmentStrategy.ADDITIVE:
            adjustment_amount = base_forecast * factor.impact * factor.confidence
            adjusted_value = base_forecast + adjustment_amount
        
        elif self.config.strategy == AdjustmentStrategy.WEIGHTED:
            # Media pesata tra impatto e forecast base
            weight = factor.confidence * (1 - self.config.historical_weight)
            adjustment_factor = 1 + (factor.impact * weight)
            adjusted_value = base_forecast * adjustment_factor
        
        else:  # HYBRID
            # Combinazione di additivo e moltiplicativo
            mult_adjustment = 1 + (factor.impact * factor.confidence * 0.5)
            add_adjustment = base_forecast * factor.impact * factor.confidence * 0.5
            adjusted_value = (base_forecast * mult_adjustment) + add_adjustment
        
        # Applica limiti di aggiustamento
        max_change = base_forecast * self.config.max_adjustment
        if abs(adjusted_value - base_forecast) > max_change:
            if adjusted_value > base_forecast:
                adjusted_value = base_forecast + max_change
            else:
                adjusted_value = base_forecast - max_change
        
        # Calcola metriche
        adjustment_amount = adjusted_value - base_forecast
        adjustment_percentage = (adjustment_amount / base_forecast * 100) if base_forecast != 0 else 0
        
        # Determina livello di impatto
        abs_pct = abs(adjustment_percentage)
        if abs_pct < 2:
            impact_level = ImpactLevel.VERY_LOW
        elif abs_pct < 5:
            impact_level = ImpactLevel.LOW
        elif abs_pct < 10:
            impact_level = ImpactLevel.MEDIUM
        elif abs_pct < 20:
            impact_level = ImpactLevel.HIGH
        else:
            impact_level = ImpactLevel.VERY_HIGH
        
        # Genera spiegazione
        direction = "aumenta" if adjustment_amount > 0 else "diminuisce"
        explanation = (
            f"Il fattore '{factor.name}' ({factor.type.value}) "
            f"{direction} la previsione del {abs(adjustment_percentage):.1f}% "
            f"(confidenza: {factor.confidence:.0%})"
        )
        
        return FactorImpact(
            factor=factor,
            original_forecast=base_forecast,
            adjusted_forecast=adjusted_value,
            adjustment_amount=adjustment_amount,
            adjustment_percentage=adjustment_percentage,
            impact_level=impact_level,
            explanation=explanation
        )
    
    def combine_impacts(
        self,
        impacts: List[FactorImpact],
        base_forecast: float
    ) -> float:
        """
        Combina multipli impatti in un aggiustamento finale.
        
        Args:
            impacts: Lista di impatti da combinare
            base_forecast: Previsione base
            
        Returns:
            Valore aggiustato finale
        """
        if not impacts:
            return base_forecast
        
        if self.config.strategy == AdjustmentStrategy.WEIGHTED:
            # Media pesata basata su confidenza
            total_confidence = sum(i.factor.confidence for i in impacts)
            if total_confidence == 0:
                return base_forecast
            
            weighted_adjustment = sum(
                i.adjustment_amount * i.factor.confidence 
                for i in impacts
            ) / total_confidence
            
            return base_forecast + weighted_adjustment
        
        elif self.config.strategy == AdjustmentStrategy.MULTIPLICATIVE:
            # Prodotto dei fattori moltiplicativi
            combined_factor = 1.0
            for impact in impacts:
                factor_mult = impact.adjusted_forecast / impact.original_forecast
                combined_factor *= factor_mult
            
            return base_forecast * combined_factor
        
        else:
            # Somma degli aggiustamenti con cap
            total_adjustment = sum(i.adjustment_amount for i in impacts)
            max_adjustment = base_forecast * self.config.max_adjustment
            
            if abs(total_adjustment) > max_adjustment:
                total_adjustment = max_adjustment if total_adjustment > 0 else -max_adjustment
            
            return base_forecast + total_adjustment
    
    def sense(
        self,
        base_forecast: Union[pd.Series, np.ndarray, List[float]],
        start_date: Optional[datetime] = None,
        fetch_external: bool = True
    ) -> SensingResult:
        """
        Applica demand sensing alle previsioni base.
        
        Args:
            base_forecast: Previsioni base da aggiustare
            start_date: Data di inizio previsioni
            fetch_external: Se True, recupera dati esterni automaticamente
            
        Returns:
            Risultato del sensing con previsioni aggiustate
        """
        # Converti a Series se necessario
        if not isinstance(base_forecast, pd.Series):
            base_forecast = pd.Series(base_forecast)
        
        if start_date is None:
            start_date = datetime.now()
        
        # Fetch fattori esterni se richiesto
        if fetch_external:
            self._fetch_external_factors(start_date, len(base_forecast))
        
        # Calcola impatti per ogni punto di previsione
        adjusted_values = []
        all_impacts = []
        
        for i, value in enumerate(base_forecast):
            # Calcola impatti per questo punto temporale
            point_impacts = []
            for factor in self.factors:
                impact = self.calculate_impact(factor, value)
                point_impacts.append(impact)
            
            # Combina impatti
            adjusted_value = self.combine_impacts(point_impacts, value)
            adjusted_values.append(adjusted_value)
            all_impacts.extend(point_impacts)
        
        # Crea serie aggiustata
        adjusted_forecast = pd.Series(adjusted_values, index=base_forecast.index)
        
        # Calcola metriche aggregate
        total_adjustment = (adjusted_forecast.sum() - base_forecast.sum()) / base_forecast.sum()
        avg_confidence = np.mean([f.confidence for f in self.factors]) if self.factors else 0
        
        # Genera raccomandazioni
        recommendations = self._generate_recommendations(all_impacts)
        
        # Aggiorna history
        self.impact_history.extend(all_impacts)
        
        # Metadata
        metadata = {
            'factors_count': len(self.factors),
            'strategy': self.config.strategy.value,
            'max_adjustment': self.config.max_adjustment,
            'significant_impacts': sum(1 for i in all_impacts if i.is_significant)
        }
        
        return SensingResult(
            original_forecast=base_forecast,
            adjusted_forecast=adjusted_forecast,
            factors_applied=all_impacts[:10],  # Top 10 impatti
            total_adjustment=total_adjustment,
            confidence_score=avg_confidence,
            recommendations=recommendations,
            metadata=metadata
        )
    
    def _fetch_external_factors(
        self,
        start_date: datetime,
        horizon: int
    ) -> None:
        """
        Recupera fattori esterni dai vari servizi.
        
        Args:
            start_date: Data inizio
            horizon: Orizzonte previsionale in giorni
        """
        logger.info(f"Recupero fattori esterni per {horizon} giorni da {start_date}")
        
        # Questo sarà implementato dalle sottoclassi specifiche
        # (WeatherIntegration, GoogleTrendsIntegration, etc.)
        pass
    
    def _generate_recommendations(
        self,
        impacts: List[FactorImpact]
    ) -> List[str]:
        """
        Genera raccomandazioni basate sugli impatti.
        
        Args:
            impacts: Lista di impatti analizzati
            
        Returns:
            Lista di raccomandazioni
        """
        recommendations = []
        
        # Analizza impatti significativi
        significant = [i for i in impacts if i.is_significant]
        
        if significant:
            # Top positive impact
            positive = [i for i in significant if i.adjustment_amount > 0]
            if positive:
                top_positive = max(positive, key=lambda x: x.adjustment_amount)
                recommendations.append(
                    f"Prepararsi per aumento domanda del {top_positive.adjustment_percentage:.1f}% "
                    f"dovuto a: {top_positive.factor.name}"
                )
            
            # Top negative impact
            negative = [i for i in significant if i.adjustment_amount < 0]
            if negative:
                top_negative = min(negative, key=lambda x: x.adjustment_amount)
                recommendations.append(
                    f"Attenzione: possibile calo domanda del {abs(top_negative.adjustment_percentage):.1f}% "
                    f"dovuto a: {top_negative.factor.name}"
                )
        
        # Analizza per tipo
        by_type = {}
        for impact in impacts:
            factor_type = impact.factor.type.value
            if factor_type not in by_type:
                by_type[factor_type] = []
            by_type[factor_type].append(impact)
        
        # Raccomandazioni per tipo
        if 'weather' in by_type and len(by_type['weather']) > 0:
            avg_weather = np.mean([i.adjustment_percentage for i in by_type['weather']])
            if abs(avg_weather) > 5:
                recommendations.append(
                    f"Fattori meteo indicano variazione del {avg_weather:.1f}% - "
                    "considerare stock buffer aggiuntivo"
                )
        
        if 'calendar' in by_type and len(by_type['calendar']) > 0:
            recommendations.append(
                "Eventi calendario rilevati - verificare disponibilità personale e logistica"
            )
        
        # Raccomandazione su confidenza
        avg_confidence = np.mean([i.factor.confidence for i in impacts]) if impacts else 0
        if avg_confidence < 0.5:
            recommendations.append(
                "Bassa confidenza nei fattori esterni - mantenere approccio conservativo"
            )
        
        return recommendations[:5]  # Max 5 raccomandazioni
    
    def learn_from_actuals(
        self,
        predicted: pd.Series,
        actuals: pd.Series,
        factors_used: List[ExternalFactor]
    ) -> None:
        """
        Apprende dai risultati effettivi per migliorare gli aggiustamenti futuri.
        
        Args:
            predicted: Valori previsti
            actuals: Valori effettivi
            factors_used: Fattori utilizzati per le previsioni
        """
        if len(predicted) != len(actuals):
            raise ValueError("Predicted e actuals devono avere la stessa lunghezza")
        
        # Calcola errore
        error = (actuals - predicted).mean()
        relative_error = error / actuals.mean() if actuals.mean() != 0 else 0
        
        # Aggiorna learning cache per ogni fattore
        for factor in factors_used:
            key = f"{factor.type.value}_{factor.name}"
            
            # Aggiorna peso basato su performance
            current_weight = self.learning_cache.get(key, factor.impact)
            
            # Se l'errore è nella direzione prevista dal fattore, aumenta peso
            # Altrimenti diminuisci
            if (factor.impact > 0 and relative_error > 0) or \
               (factor.impact < 0 and relative_error < 0):
                # Previsione nella direzione giusta
                new_weight = current_weight * (1 + self.config.learning_rate)
            else:
                # Previsione nella direzione sbagliata
                new_weight = current_weight * (1 - self.config.learning_rate)
            
            # Limita peso tra -1 e 1
            new_weight = max(-1, min(1, new_weight))
            self.learning_cache[key] = new_weight
            
            logger.info(
                f"Aggiornato peso per {key}: {current_weight:.3f} -> {new_weight:.3f}"
            )
    
    def get_factor_importance(self) -> pd.DataFrame:
        """
        Calcola l'importanza relativa dei vari fattori.
        
        Returns:
            DataFrame con importanza dei fattori
        """
        if not self.impact_history:
            return pd.DataFrame()
        
        # Aggrega per fattore
        importance = {}
        for impact in self.impact_history:
            key = f"{impact.factor.type.value}_{impact.factor.name}"
            if key not in importance:
                importance[key] = {
                    'count': 0,
                    'avg_impact': 0,
                    'avg_confidence': 0,
                    'total_adjustment': 0
                }
            
            importance[key]['count'] += 1
            importance[key]['avg_impact'] += abs(impact.adjustment_percentage)
            importance[key]['avg_confidence'] += impact.factor.confidence
            importance[key]['total_adjustment'] += abs(impact.adjustment_amount)
        
        # Calcola medie
        for key in importance:
            importance[key]['avg_impact'] /= importance[key]['count']
            importance[key]['avg_confidence'] /= importance[key]['count']
        
        # Converti a DataFrame
        df = pd.DataFrame.from_dict(importance, orient='index')
        df['importance_score'] = (
            df['avg_impact'] * df['avg_confidence'] * np.log1p(df['count'])
        )
        
        return df.sort_values('importance_score', ascending=False)