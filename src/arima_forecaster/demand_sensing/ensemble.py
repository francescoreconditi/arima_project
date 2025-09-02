"""
Ensemble Demand Sensor - Combina tutti i fattori esterni in modo intelligente.

Orchestratore principale che integra tutti i moduli di demand sensing.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path

from .demand_sensor import (
    DemandSensor, 
    ExternalFactor, 
    SensingConfig,
    SensingResult,
    AdjustmentStrategy
)
from .weather import WeatherIntegration, WeatherImpact
from .trends import GoogleTrendsIntegration, TrendImpact
from .social import SocialSentimentAnalyzer, SentimentImpact
from .economic import EconomicIndicators, EconomicImpact, EconomicIndicator
from .calendar_events import CalendarEvents, EventImpact, EventType

from ..core.arima_model import ARIMAForecaster
from ..core.sarima_model import SARIMAForecaster
from ..utils.logger import setup_logger
from ..visualization.plotter import ForecastPlotter

logger = setup_logger(__name__)


@dataclass
class EnsembleConfig:
    """Configurazione per Ensemble Demand Sensor."""
    
    # Pesi per fonte
    source_weights: Dict[str, float] = None
    
    # Abilitazione moduli
    enable_weather: bool = True
    enable_trends: bool = True
    enable_social: bool = True
    enable_economic: bool = True
    enable_calendar: bool = True
    
    # Strategia combinazione
    combination_strategy: str = "weighted_average"  # weighted_average, max, voting
    
    # Parametri learning
    enable_learning: bool = True
    learning_rate: float = 0.1
    
    # Limiti aggiustamento
    max_total_adjustment: float = 0.4  # Max 40% aggiustamento totale
    min_sources_for_adjustment: int = 2  # Minimo 2 fonti concordi
    
    # Cache e performance
    use_cache: bool = True
    cache_ttl_hours: int = 6
    
    def __post_init__(self):
        if self.source_weights is None:
            self.source_weights = {
                'weather': 0.20,
                'trends': 0.25,
                'social': 0.15,
                'economic': 0.25,
                'calendar': 0.15
            }


class EnsembleDemandSensor:
    """
    Orchestratore principale per Demand Sensing integrato.
    
    Combina intelligentemente tutti i fattori esterni per migliorare
    le previsioni di domanda.
    """
    
    def __init__(
        self,
        base_model: Optional[Union[ARIMAForecaster, SARIMAForecaster]] = None,
        product_category: str = "default",
        location: str = "Milan,IT",
        config: Optional[EnsembleConfig] = None
    ):
        """
        Inizializza Ensemble Demand Sensor.
        
        Args:
            base_model: Modello forecasting base
            product_category: Categoria prodotto
            location: Località geografica
            config: Configurazione ensemble
        """
        self.base_model = base_model
        self.product_category = product_category
        self.location = location
        self.config = config or EnsembleConfig()
        
        # Inizializza moduli
        self._init_modules()
        
        # Cache risultati
        self._cache = {}
        self._cache_timestamps = {}
        
        # Storia per learning
        self.prediction_history = []
        self.actual_history = []
        
        logger.info(
            f"EnsembleDemandSensor inizializzato per {product_category} in {location}"
        )
    
    def _init_modules(self):
        """Inizializza tutti i moduli di sensing."""
        
        # Weather
        if self.config.enable_weather:
            self.weather = WeatherIntegration(
                location=self.location,
                impact_config=WeatherImpact()
            )
        else:
            self.weather = None
        
        # Google Trends
        if self.config.enable_trends:
            # Keywords basate su categoria prodotto
            keywords = self._get_category_keywords()
            self.trends = GoogleTrendsIntegration(
                keywords=keywords,
                geo=self.location.split(',')[1] if ',' in self.location else 'IT'
            )
        else:
            self.trends = None
        
        # Social Sentiment
        if self.config.enable_social:
            brand_keywords = self._get_brand_keywords()
            self.social = SocialSentimentAnalyzer(
                brand_keywords=brand_keywords,
                product_keywords=self._get_category_keywords()
            )
        else:
            self.social = None
        
        # Economic Indicators
        if self.config.enable_economic:
            country = self.location.split(',')[1] if ',' in self.location else 'IT'
            self.economic = EconomicIndicators(
                country=country,
                sector=self._get_economic_sector()
            )
        else:
            self.economic = None
        
        # Calendar Events
        if self.config.enable_calendar:
            country = self.location.split(',')[1] if ',' in self.location else 'IT'
            self.calendar = CalendarEvents(
                country=country,
                product_category=self.product_category
            )
        else:
            self.calendar = None
        
        # Base demand sensor per orchestrazione
        self.base_sensor = DemandSensor(
            base_model=self.base_model,
            config=SensingConfig(
                strategy=AdjustmentStrategy.WEIGHTED,
                max_adjustment=self.config.max_total_adjustment
            )
        )
    
    def _get_category_keywords(self) -> List[str]:
        """Ottieni keywords per categoria prodotto."""
        
        keywords_map = {
            'electronics': ['smartphone', 'laptop', 'tablet', 'tv'],
            'clothing': ['moda', 'abbigliamento', 'vestiti', 'scarpe'],
            'food': ['cibo', 'alimentari', 'spesa', 'ristorante'],
            'medical': ['farmaco', 'medicina', 'salute', 'ospedale'],
            'automotive': ['auto', 'macchina', 'ricambi', 'benzina'],
            'default': ['prodotto', 'acquisto', 'offerta', 'shopping']
        }
        
        return keywords_map.get(self.product_category, keywords_map['default'])
    
    def _get_brand_keywords(self) -> List[str]:
        """Ottieni brand keywords."""
        # In produzione questi verrebbero da configurazione
        return ['brand', 'azienda', 'negozio']
    
    def _get_economic_sector(self) -> str:
        """Mappa categoria prodotto a settore economico."""
        
        sector_map = {
            'electronics': 'technology',
            'clothing': 'retail',
            'food': 'essential',
            'medical': 'essential',
            'automotive': 'automotive',
            'luxury': 'luxury',
            'default': 'retail'
        }
        
        return sector_map.get(self.product_category, 'default')
    
    def collect_all_factors(
        self,
        forecast_horizon: int = 30,
        use_demo_data: bool = True
    ) -> Dict[str, List[ExternalFactor]]:
        """
        Raccoglie tutti i fattori esterni da tutte le fonti.
        
        Args:
            forecast_horizon: Giorni di previsione
            use_demo_data: Usa dati demo invece di API reali
            
        Returns:
            Dizionario con fattori per fonte
        """
        all_factors = {}
        
        # Weather factors
        if self.weather:
            try:
                conditions = self.weather.fetch_forecast(
                    days_ahead=forecast_horizon,
                    use_demo_data=use_demo_data
                )
                weather_factors = self.weather.calculate_weather_impact(
                    conditions,
                    product_category=self.product_category
                )
                all_factors['weather'] = weather_factors
                logger.info(f"Raccolti {len(weather_factors)} fattori meteo")
            except Exception as e:
                logger.warning(f"Errore raccolta fattori meteo: {e}")
                all_factors['weather'] = []
        
        # Trends factors
        if self.trends:
            try:
                trend_data = self.trends.fetch_trends(
                    timeframe="today 1-m",
                    use_demo_data=use_demo_data
                )
                trend_factors = self.trends.calculate_trend_impact(
                    trend_data,
                    product_category=self.product_category,
                    forecast_horizon=forecast_horizon
                )
                all_factors['trends'] = trend_factors
                logger.info(f"Raccolti {len(trend_factors)} fattori trend")
            except Exception as e:
                logger.warning(f"Errore raccolta fattori trend: {e}")
                all_factors['trends'] = []
        
        # Social factors
        if self.social:
            try:
                posts = self.social.fetch_social_posts(
                    days_back=7,
                    use_demo_data=use_demo_data
                )
                social_factors = self.social.calculate_social_impact(
                    posts,
                    forecast_horizon=forecast_horizon
                )
                all_factors['social'] = social_factors
                logger.info(f"Raccolti {len(social_factors)} fattori social")
            except Exception as e:
                logger.warning(f"Errore raccolta fattori social: {e}")
                all_factors['social'] = []
        
        # Economic factors
        if self.economic:
            try:
                indicators = self.economic.fetch_indicators(
                    use_demo_data=use_demo_data
                )
                economic_factors = self.economic.calculate_economic_impact(
                    indicators,
                    forecast_horizon=forecast_horizon
                )
                all_factors['economic'] = economic_factors
                logger.info(f"Raccolti {len(economic_factors)} fattori economici")
            except Exception as e:
                logger.warning(f"Errore raccolta fattori economici: {e}")
                all_factors['economic'] = []
        
        # Calendar factors
        if self.calendar:
            try:
                events = self.calendar.get_events(
                    start_date=datetime.now(),
                    end_date=datetime.now() + timedelta(days=forecast_horizon)
                )
                calendar_factors = self.calendar.calculate_event_impact(
                    events,
                    forecast_horizon=forecast_horizon
                )
                all_factors['calendar'] = calendar_factors
                logger.info(f"Raccolti {len(calendar_factors)} fattori calendario")
            except Exception as e:
                logger.warning(f"Errore raccolta fattori calendario: {e}")
                all_factors['calendar'] = []
        
        return all_factors
    
    def combine_factors(
        self,
        factors_by_source: Dict[str, List[ExternalFactor]]
    ) -> List[ExternalFactor]:
        """
        Combina intelligentemente fattori da diverse fonti.
        
        Args:
            factors_by_source: Fattori organizzati per fonte
            
        Returns:
            Lista fattori combinati
        """
        combined_factors = []
        
        # Raggruppa per data
        factors_by_date = {}
        
        for source, factors in factors_by_source.items():
            source_weight = self.config.source_weights.get(source, 0.2)
            
            for factor in factors:
                date_key = factor.timestamp.date()
                
                if date_key not in factors_by_date:
                    factors_by_date[date_key] = []
                
                # Applica peso fonte
                weighted_factor = ExternalFactor(
                    name=f"{source}_{factor.name}",
                    type=factor.type,
                    value=factor.value,
                    impact=factor.impact * source_weight,
                    confidence=factor.confidence,
                    timestamp=factor.timestamp,
                    metadata={**factor.metadata, 'source': source}
                )
                
                factors_by_date[date_key].append(weighted_factor)
        
        # Combina per data usando strategia configurata
        for date, date_factors in factors_by_date.items():
            if len(date_factors) < self.config.min_sources_for_adjustment:
                # Non abbastanza fonti concordi, skip
                continue
            
            if self.config.combination_strategy == "weighted_average":
                # Media pesata
                total_impact = sum(f.impact for f in date_factors)
                avg_confidence = np.mean([f.confidence for f in date_factors])
                
            elif self.config.combination_strategy == "max":
                # Prendi impatto massimo (più conservativo)
                max_factor = max(date_factors, key=lambda x: abs(x.impact))
                total_impact = max_factor.impact
                avg_confidence = max_factor.confidence
                
            elif self.config.combination_strategy == "voting":
                # Voting: maggioranza decide direzione
                positive = sum(1 for f in date_factors if f.impact > 0)
                negative = sum(1 for f in date_factors if f.impact < 0)
                
                if positive > negative:
                    total_impact = np.mean([f.impact for f in date_factors if f.impact > 0])
                elif negative > positive:
                    total_impact = np.mean([f.impact for f in date_factors if f.impact < 0])
                else:
                    total_impact = 0
                
                avg_confidence = np.mean([f.confidence for f in date_factors])
            
            else:
                total_impact = np.mean([f.impact for f in date_factors])
                avg_confidence = np.mean([f.confidence for f in date_factors])
            
            # Limita impatto totale
            total_impact = max(
                -self.config.max_total_adjustment,
                min(self.config.max_total_adjustment, total_impact)
            )
            
            # Crea fattore combinato
            combined_factor = ExternalFactor(
                name=f"Ensemble_{date}",
                type=date_factors[0].type,  # Usa tipo del primo
                value=len(date_factors),  # Numero fonti
                impact=total_impact,
                confidence=avg_confidence,
                timestamp=datetime.combine(date, datetime.min.time()),
                metadata={
                    'sources_count': len(date_factors),
                    'sources': [f.metadata.get('source', 'unknown') for f in date_factors],
                    'strategy': self.config.combination_strategy
                }
            )
            
            combined_factors.append(combined_factor)
        
        # Ordina per data
        combined_factors.sort(key=lambda x: x.timestamp)
        
        logger.info(f"Combinati fattori in {len(combined_factors)} giorni")
        return combined_factors
    
    def sense(
        self,
        base_forecast: Union[pd.Series, np.ndarray, List[float]],
        use_demo_data: bool = True,
        return_details: bool = False
    ) -> Union[SensingResult, Tuple[SensingResult, Dict]]:
        """
        Applica ensemble demand sensing alle previsioni.
        
        Args:
            base_forecast: Previsioni base da aggiustare
            use_demo_data: Usa dati demo
            return_details: Ritorna dettagli aggiuntivi
            
        Returns:
            Risultato sensing (e dettagli se richiesto)
        """
        # Converti a Series
        if not isinstance(base_forecast, pd.Series):
            base_forecast = pd.Series(base_forecast)
        
        forecast_horizon = len(base_forecast)
        
        # Raccogli tutti i fattori
        all_factors = self.collect_all_factors(
            forecast_horizon=forecast_horizon,
            use_demo_data=use_demo_data
        )
        
        # Combina fattori
        combined_factors = self.combine_factors(all_factors)
        
        # Applica fattori usando base sensor
        self.base_sensor.clear_factors()
        for factor in combined_factors:
            self.base_sensor.add_factor(factor)
        
        # Ottieni risultato sensing
        result = self.base_sensor.sense(
            base_forecast,
            fetch_external=False  # Già raccolti
        )
        
        # Aggiungi raccomandazioni ensemble
        ensemble_recommendations = self._generate_ensemble_recommendations(
            all_factors,
            result
        )
        result.recommendations.extend(ensemble_recommendations)
        
        if return_details:
            details = {
                'factors_by_source': all_factors,
                'combined_factors': combined_factors,
                'source_contributions': self._calculate_source_contributions(all_factors)
            }
            return result, details
        
        return result
    
    def _calculate_source_contributions(
        self,
        factors_by_source: Dict[str, List[ExternalFactor]]
    ) -> pd.DataFrame:
        """
        Calcola contributo di ogni fonte all'aggiustamento.
        
        Args:
            factors_by_source: Fattori per fonte
            
        Returns:
            DataFrame con contributi
        """
        contributions = []
        
        for source, factors in factors_by_source.items():
            if factors:
                avg_impact = np.mean([abs(f.impact) for f in factors])
                avg_confidence = np.mean([f.confidence for f in factors])
                weight = self.config.source_weights.get(source, 0.2)
                
                contributions.append({
                    'source': source,
                    'factors_count': len(factors),
                    'avg_impact': avg_impact,
                    'avg_confidence': avg_confidence,
                    'weight': weight,
                    'contribution': avg_impact * weight * avg_confidence
                })
        
        df = pd.DataFrame(contributions)
        if not df.empty:
            df['contribution_pct'] = df['contribution'] / df['contribution'].sum() * 100
            df = df.sort_values('contribution', ascending=False)
        
        return df
    
    def _generate_ensemble_recommendations(
        self,
        all_factors: Dict[str, List[ExternalFactor]],
        result: SensingResult
    ) -> List[str]:
        """
        Genera raccomandazioni basate su analisi ensemble.
        
        Args:
            all_factors: Tutti i fattori raccolti
            result: Risultato sensing
            
        Returns:
            Lista raccomandazioni
        """
        recommendations = []
        
        # Analizza concordanza fonti
        positive_sources = []
        negative_sources = []
        
        for source, factors in all_factors.items():
            if factors:
                avg_impact = np.mean([f.impact for f in factors])
                if avg_impact > 0.05:
                    positive_sources.append(source)
                elif avg_impact < -0.05:
                    negative_sources.append(source)
        
        # Raccomandazioni su concordanza
        if len(positive_sources) >= 3:
            recommendations.append(
                f"FORTE SEGNALE POSITIVO: {len(positive_sources)} fonti concordano "
                f"su aumento domanda ({', '.join(positive_sources)})"
            )
        
        if len(negative_sources) >= 3:
            recommendations.append(
                f"ATTENZIONE: {len(negative_sources)} fonti indicano "
                f"calo domanda ({', '.join(negative_sources)})"
            )
        
        # Conflitti tra fonti
        if positive_sources and negative_sources:
            recommendations.append(
                "Segnali contrastanti tra fonti - mantenere flessibilità operativa"
            )
        
        # Analisi adjustment totale
        total_adj = result.total_adjustment
        if abs(total_adj) > 0.2:
            direction = "aumento" if total_adj > 0 else "diminuzione"
            recommendations.append(
                f"Aggiustamento significativo: {direction} del {abs(total_adj):.1%}"
            )
        
        return recommendations[:3]  # Max 3 raccomandazioni ensemble
    
    def learn_from_actuals(
        self,
        predicted: pd.Series,
        actuals: pd.Series
    ) -> None:
        """
        Apprende dai risultati per migliorare pesi fonti.
        
        Args:
            predicted: Valori previsti (post-sensing)
            actuals: Valori effettivi
        """
        if not self.config.enable_learning:
            return
        
        # Calcola errore
        mae = np.mean(np.abs(predicted - actuals))
        mape = np.mean(np.abs((actuals - predicted) / actuals)) * 100
        
        # Salva in storia
        self.prediction_history.append(predicted)
        self.actual_history.append(actuals)
        
        # Se abbiamo abbastanza storia, aggiusta pesi
        if len(self.prediction_history) >= 5:
            # Analizza quale fonte ha performato meglio
            # (implementazione semplificata)
            
            # Per ora aggiusta leggermente verso fonti più affidabili
            if mape < 10:  # Performance buona
                # Mantieni pesi
                pass
            elif mape < 20:  # Performance media
                # Riduci leggermente pesi fonti meno affidabili
                self.config.source_weights['social'] *= 0.95
                self.config.source_weights['trends'] *= 0.98
            else:  # Performance scarsa
                # Riduci pesi significativamente
                for source in self.config.source_weights:
                    self.config.source_weights[source] *= 0.9
            
            # Rinormalizza pesi
            total = sum(self.config.source_weights.values())
            for source in self.config.source_weights:
                self.config.source_weights[source] /= total
            
            logger.info(f"Aggiornati pesi fonti. MAPE: {mape:.1f}%")
    
    def plot_sensing_analysis(
        self,
        result: SensingResult,
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualizza analisi demand sensing.
        
        Args:
            result: Risultato sensing
            save_path: Path per salvare plot
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Original vs Adjusted Forecast
        ax1 = axes[0, 0]
        ax1.plot(result.original_forecast.index, result.original_forecast.values, 
                label='Original', linewidth=2)
        ax1.plot(result.adjusted_forecast.index, result.adjusted_forecast.values,
                label='Adjusted', linewidth=2, linestyle='--')
        ax1.fill_between(result.adjusted_forecast.index,
                        result.original_forecast.values,
                        result.adjusted_forecast.values,
                        alpha=0.3)
        ax1.set_title('Forecast Adjustment')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Adjustment Percentage
        ax2 = axes[0, 1]
        adjustment_pct = ((result.adjusted_forecast - result.original_forecast) / 
                         result.original_forecast * 100)
        ax2.bar(range(len(adjustment_pct)), adjustment_pct.values,
               color=['green' if x > 0 else 'red' for x in adjustment_pct.values])
        ax2.set_title('Daily Adjustment %')
        ax2.set_xlabel('Day')
        ax2.set_ylabel('Adjustment %')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Factor Impacts
        ax3 = axes[1, 0]
        if result.factors_applied:
            impacts = [f.adjustment_percentage for f in result.factors_applied[:10]]
            names = [f.factor.name[:15] for f in result.factors_applied[:10]]
            colors = ['green' if x > 0 else 'red' for x in impacts]
            ax3.barh(names, impacts, color=colors)
            ax3.set_title('Top 10 Factor Impacts')
            ax3.set_xlabel('Impact %')
        
        # Plot 4: Confidence Scores
        ax4 = axes[1, 1]
        if result.factors_applied:
            confidences = [f.factor.confidence for f in result.factors_applied[:10]]
            names = [f.factor.name[:15] for f in result.factors_applied[:10]]
            ax4.barh(names, confidences)
            ax4.set_title('Factor Confidence Scores')
            ax4.set_xlabel('Confidence')
            ax4.set_xlim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100)
            logger.info(f"Plot salvato in {save_path}")
        
        plt.show()