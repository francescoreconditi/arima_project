"""
Modelli di Forecasting per Domanda Intermittente (Intermittent Demand)
Implementazione di Croston's Method, SBA, TSB per spare parts e prodotti a bassa rotazione

Autore: Claude Code
Data: 2025-09-02
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from enum import Enum
from dataclasses import dataclass
import warnings
from scipy import stats
from pydantic import BaseModel, Field, field_validator

# Import base classes
from arima_forecaster.utils.logger import get_logger
from arima_forecaster.utils.exceptions import ModelTrainingError, ForecastError

logger = get_logger(__name__)


class IntermittentMethod(str, Enum):
    """Metodi disponibili per forecasting domanda intermittente"""
    CROSTON = "croston"
    SBA = "sba"  # Syntetos-Boylan Approximation
    TSB = "tsb"  # Teunter-Syntetos-Babai
    SBJ = "sbj"  # Shale-Boylan-Johnston (variant)
    ADAPTIVE_CROSTON = "adaptive_croston"  # Adaptive smoothing parameter


@dataclass
class IntermittentPattern:
    """Analisi pattern domanda intermittente"""
    adi: float  # Average Demand Interval (tempo medio tra ordini)
    cv2: float  # Squared Coefficient of Variation
    demand_periods: int  # Numero periodi con domanda > 0
    zero_periods: int  # Numero periodi con domanda = 0
    intermittence: float  # Percentuale periodi zero
    lumpiness: float  # Variabilità dimensione ordini
    classification: str  # Smooth, Intermittent, Lumpy, Erratic


class IntermittentConfig(BaseModel):
    """Configurazione per modelli Intermittent Demand"""
    method: IntermittentMethod = IntermittentMethod.CROSTON
    alpha: float = Field(0.1, ge=0.0, le=1.0, description="Smoothing parameter")
    initial_level: Optional[float] = Field(None, description="Initial demand level")
    initial_interval: Optional[float] = Field(None, description="Initial interval")
    bias_correction: bool = Field(True, description="Apply bias correction (SBA)")
    optimize_alpha: bool = Field(False, description="Auto-optimize alpha")
    min_demand_periods: int = Field(2, description="Min periods with demand")
    
    @field_validator('alpha')
    def validate_alpha(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Alpha must be between 0 and 1")
        return v


class IntermittentForecaster:
    """
    Forecaster per Domanda Intermittente (Spare Parts, Slow Movers)
    
    Implementa:
    - Croston's Method (1972)
    - SBA - Syntetos-Boylan Approximation (2005)
    - TSB - Teunter-Syntetos-Babai (2011)
    
    Ideale per:
    - Ricambi auto/aerospace
    - Farmaci specialistici
    - Prodotti stagionali estremi
    - Componentistica industriale
    """
    
    def __init__(self, config: Optional[IntermittentConfig] = None):
        """
        Inizializza forecaster per domanda intermittente
        
        Args:
            config: Configurazione modello
        """
        self.config = config or IntermittentConfig()
        self.is_fitted = False
        
        # State variables
        self.demand_level_ = None
        self.interval_level_ = None
        self.forecast_ = None
        self.pattern_ = None
        self.history_ = None
        
        # Performance metrics
        self.mse_ = None
        self.bias_ = None
        
    def analyze_pattern(self, data: Union[pd.Series, np.ndarray]) -> IntermittentPattern:
        """
        Analizza il pattern di domanda per classificazione
        
        Args:
            data: Serie storica domanda
            
        Returns:
            IntermittentPattern con analisi dettagliata
        """
        if isinstance(data, pd.Series):
            values = data.values
        else:
            values = np.array(data)
            
        # Calcola metriche base
        non_zero = values[values > 0]
        demand_periods = len(non_zero)
        zero_periods = len(values) - demand_periods
        
        # ADI - Average Demand Interval
        if demand_periods > 1:
            intervals = []
            last_demand_idx = -1
            for i, v in enumerate(values):
                if v > 0:
                    if last_demand_idx >= 0:
                        intervals.append(i - last_demand_idx)
                    last_demand_idx = i
            adi = np.mean(intervals) if intervals else len(values) / demand_periods
        else:
            adi = len(values)
            
        # CV² - Squared Coefficient of Variation
        if demand_periods > 0 and np.mean(non_zero) > 0:
            cv2 = (np.std(non_zero) / np.mean(non_zero)) ** 2
        else:
            cv2 = 0
            
        # Intermittence e Lumpiness
        intermittence = zero_periods / len(values)
        lumpiness = cv2 * intermittence if intermittence > 0 else 0
        
        # Classificazione (Syntetos et al. 2005)
        if adi < 1.32 and cv2 < 0.49:
            classification = "Smooth"
        elif adi >= 1.32 and cv2 < 0.49:
            classification = "Intermittent"
        elif adi < 1.32 and cv2 >= 0.49:
            classification = "Erratic"
        else:
            classification = "Lumpy"
            
        return IntermittentPattern(
            adi=adi,
            cv2=cv2,
            demand_periods=demand_periods,
            zero_periods=zero_periods,
            intermittence=intermittence,
            lumpiness=lumpiness,
            classification=classification
        )
        
    def fit(self, data: Union[pd.Series, np.ndarray]) -> 'IntermittentForecaster':
        """
        Addestra il modello su dati storici
        
        Args:
            data: Serie storica domanda (può contenere molti zeri)
            
        Returns:
            self per chaining
        """
        # Converti in numpy array
        if isinstance(data, pd.Series):
            self.history_ = data.copy()
            values = data.values
        else:
            values = np.array(data)
            self.history_ = pd.Series(values)
            
        # Analizza pattern
        self.pattern_ = self.analyze_pattern(values)
        
        # Verifica dati sufficienti
        if self.pattern_.demand_periods < self.config.min_demand_periods:
            raise ModelTrainingError(
                f"Insufficient demand periods: {self.pattern_.demand_periods} < "
                f"{self.config.min_demand_periods}"
            )
            
        # Ottimizza alpha se richiesto
        if self.config.optimize_alpha:
            self.config.alpha = self._optimize_alpha(values)
            
        # Fit basato sul metodo scelto
        if self.config.method == IntermittentMethod.CROSTON:
            self._fit_croston(values)
        elif self.config.method == IntermittentMethod.SBA:
            self._fit_sba(values)
        elif self.config.method == IntermittentMethod.TSB:
            self._fit_tsb(values)
        elif self.config.method == IntermittentMethod.ADAPTIVE_CROSTON:
            self._fit_adaptive_croston(values)
        else:
            raise ValueError(f"Unknown method: {self.config.method}")
            
        self.is_fitted = True
        logger.info(f"Fitted {self.config.method.value} model on {len(values)} periods")
        logger.info(f"Pattern: {self.pattern_.classification}, ADI={self.pattern_.adi:.2f}")
        
        return self
        
    def _fit_croston(self, values: np.ndarray):
        """Implementazione Croston's Method originale"""
        alpha = self.config.alpha
        
        # Inizializzazione
        demand_est = self.config.initial_level or np.mean(values[values > 0])
        interval_est = self.config.initial_interval or self.pattern_.adi
        
        # Arrays per tracking
        demand_estimates = []
        interval_estimates = []
        periods_since_demand = 0
        
        for value in values:
            if value > 0:  # Domanda presente
                # Aggiorna stime
                demand_est = alpha * value + (1 - alpha) * demand_est
                interval_est = alpha * (periods_since_demand + 1) + (1 - alpha) * interval_est
                periods_since_demand = 0
            else:
                periods_since_demand += 1
                
            demand_estimates.append(demand_est)
            interval_estimates.append(interval_est)
            
        # Salva stati finali
        self.demand_level_ = demand_est
        self.interval_level_ = max(interval_est, 1.0)  # Evita divisione per zero
        
        # Forecast = demand / interval
        self.forecast_ = self.demand_level_ / self.interval_level_
        
    def _fit_sba(self, values: np.ndarray):
        """SBA - Syntetos-Boylan Approximation con bias correction"""
        # Prima fit Croston base
        self._fit_croston(values)
        
        # Applica correzione bias SBA
        if self.config.bias_correction:
            # Fattore correzione: 1 - alpha/2
            bias_factor = 1 - self.config.alpha / 2
            self.forecast_ = bias_factor * (self.demand_level_ / self.interval_level_)
            
    def _fit_tsb(self, values: np.ndarray):
        """TSB - Teunter-Syntetos-Babai con probability-based forecast"""
        alpha = self.config.alpha
        
        # Inizializzazione
        demand_est = self.config.initial_level or np.mean(values[values > 0])
        prob_est = 1.0 / (self.config.initial_interval or self.pattern_.adi)
        
        for value in values:
            # Aggiorna probabilità (sempre)
            prob_est = alpha * (1 if value > 0 else 0) + (1 - alpha) * prob_est
            
            # Aggiorna demand level solo se c'è domanda
            if value > 0:
                demand_est = alpha * value + (1 - alpha) * demand_est
                
        # Salva stati
        self.demand_level_ = demand_est
        self.interval_level_ = 1.0 / max(prob_est, 0.001)  # Converti prob in interval
        
        # TSB forecast = demand * probability
        self.forecast_ = demand_est * prob_est
        
    def _fit_adaptive_croston(self, values: np.ndarray):
        """Croston con alpha adattivo basato su errore"""
        # Inizializza con alpha fisso
        alpha = self.config.alpha
        demand_est = np.mean(values[values > 0])
        interval_est = self.pattern_.adi
        
        # Parametri adattivi
        alpha_min, alpha_max = 0.01, 0.3
        adjustment_rate = 0.01
        
        periods_since_demand = 0
        errors = []
        
        for value in values:
            # Calcola forecast corrente
            forecast = demand_est / max(interval_est, 1.0)
            
            if value > 0:
                # Calcola errore e aggiusta alpha
                error = abs(value - forecast)
                errors.append(error)
                
                if len(errors) > 2:
                    # Se errore aumenta, aumenta alpha (più reattivo)
                    if errors[-1] > errors[-2]:
                        alpha = min(alpha + adjustment_rate, alpha_max)
                    else:
                        alpha = max(alpha - adjustment_rate, alpha_min)
                
                # Aggiorna stime
                demand_est = alpha * value + (1 - alpha) * demand_est
                interval_est = alpha * (periods_since_demand + 1) + (1 - alpha) * interval_est
                periods_since_demand = 0
            else:
                periods_since_demand += 1
                
        self.demand_level_ = demand_est
        self.interval_level_ = max(interval_est, 1.0)
        self.forecast_ = self.demand_level_ / self.interval_level_
        
    def _optimize_alpha(self, values: np.ndarray) -> float:
        """
        Ottimizza alpha minimizzando MSE con grid search
        
        Args:
            values: Dati storici
            
        Returns:
            Alpha ottimale
        """
        best_alpha = 0.1
        best_mse = float('inf')
        
        for alpha in np.arange(0.01, 0.31, 0.01):
            self.config.alpha = alpha
            self._fit_croston(values)
            
            # Calcola MSE su one-step-ahead
            mse = self._calculate_mse(values)
            
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha
                
        logger.info(f"Optimized alpha: {best_alpha:.3f} (MSE: {best_mse:.3f})")
        return best_alpha
        
    def _calculate_mse(self, values: np.ndarray) -> float:
        """Calcola Mean Squared Error per validazione"""
        if len(values) < 10:  # Minimo più alto per intermittent
            return float('inf')
            
        errors = []
        
        # Usa configurazione senza optimize_alpha per evitare ricorsione
        temp_config = IntermittentConfig(
            method=self.config.method,
            alpha=self.config.alpha,
            optimize_alpha=False  # Importante: evita ricorsione
        )
        
        # Rolling origin evaluation
        min_train_size = 20  # Minimo dati per training
        for i in range(min_train_size, len(values), 5):  # Step di 5 per velocità
            train_data = values[:i]
            
            # Verifica periodi con domanda sufficienti
            if np.sum(train_data > 0) < self.config.min_demand_periods:
                continue
                
            try:
                temp_forecaster = IntermittentForecaster(temp_config)
                temp_forecaster.fit(train_data)
                forecast = temp_forecaster.forecast(1)
                actual = values[i]
                errors.append((actual - forecast[0]) ** 2)
            except:
                continue  # Skip errori di fitting
            
        return np.mean(errors) if errors else float('inf')
        
    def forecast(self, steps: int = 1) -> np.ndarray:
        """
        Genera forecast per periodi futuri
        
        Args:
            steps: Numero periodi da prevedere
            
        Returns:
            Array con previsioni
        """
        if not self.is_fitted:
            raise ForecastError("Model must be fitted before forecasting")
            
        # Per intermittent demand, forecast è costante
        return np.full(steps, self.forecast_)
        
    def predict_intervals(self, 
                         steps: int = 1,
                         confidence: float = 0.95) -> Dict[str, np.ndarray]:
        """
        Calcola intervalli di confidenza per forecast
        
        Args:
            steps: Numero periodi
            confidence: Livello confidenza (es. 0.95)
            
        Returns:
            Dict con 'forecast', 'lower', 'upper'
        """
        if not self.is_fitted:
            raise ForecastError("Model must be fitted before forecasting")
            
        forecast = self.forecast(steps)
        
        # Stima deviazione standard da dati storici
        non_zero = self.history_.values[self.history_.values > 0]
        if len(non_zero) > 1:
            std = np.std(non_zero)
        else:
            std = forecast[0] * 0.5  # Fallback conservativo
            
        # Calcola intervalli usando distribuzione normale
        z_score = stats.norm.ppf((1 + confidence) / 2)
        margin = z_score * std
        
        return {
            'forecast': forecast,
            'lower': np.maximum(0, forecast - margin),  # Non negativi
            'upper': forecast + margin
        }
        
    def calculate_safety_stock(self,
                              lead_time: int,
                              service_level: float = 0.95) -> float:
        """
        Calcola safety stock per livello servizio target
        
        Args:
            lead_time: Lead time in periodi
            service_level: Livello servizio target (es. 0.95)
            
        Returns:
            Quantità safety stock suggerita
        """
        if not self.is_fitted:
            raise ForecastError("Model must be fitted first")
            
        # Forecast domanda durante lead time
        lead_time_demand = self.forecast_ * lead_time
        
        # Stima variabilità
        non_zero = self.history_.values[self.history_.values > 0]
        if len(non_zero) > 1:
            demand_std = np.std(non_zero)
            interval_std = np.std([self.interval_level_])  # Simplified
            
            # Formula safety stock per intermittent demand
            # SS = z * sqrt(LT * σ_d² + d² * σ_LT²)
            z_score = stats.norm.ppf(service_level)
            variance = lead_time * demand_std**2 + self.demand_level_**2 * interval_std**2
            safety_stock = z_score * np.sqrt(variance)
        else:
            # Fallback conservativo
            safety_stock = lead_time_demand * 0.5
            
        return max(0, safety_stock)
        
    def calculate_reorder_point(self,
                               lead_time: int,
                               service_level: float = 0.95) -> float:
        """
        Calcola punto di riordino ottimale
        
        Args:
            lead_time: Lead time fornitore
            service_level: Livello servizio target
            
        Returns:
            Reorder point quantity
        """
        lead_time_demand = self.forecast_ * lead_time
        safety_stock = self.calculate_safety_stock(lead_time, service_level)
        return lead_time_demand + safety_stock
        
    def get_metrics(self) -> Dict[str, Any]:
        """
        Ritorna metriche performance e diagnostica
        
        Returns:
            Dict con metriche dettagliate
        """
        if not self.is_fitted:
            return {}
            
        return {
            'method': self.config.method.value,
            'alpha': self.config.alpha,
            'forecast': float(self.forecast_),
            'demand_level': float(self.demand_level_),
            'interval_level': float(self.interval_level_),
            'pattern': {
                'classification': self.pattern_.classification,
                'adi': float(self.pattern_.adi),
                'cv2': float(self.pattern_.cv2),
                'intermittence': float(self.pattern_.intermittence),
                'lumpiness': float(self.pattern_.lumpiness)
            }
        }
        
    def __repr__(self) -> str:
        if self.is_fitted:
            return (f"IntermittentForecaster(method={self.config.method.value}, "
                   f"pattern={self.pattern_.classification}, "
                   f"forecast={self.forecast_:.3f})")
        return f"IntermittentForecaster(method={self.config.method.value}, fitted=False)"