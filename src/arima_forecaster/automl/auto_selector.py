"""
AutoML Forecasting Engine
One-click automatic model selection with intelligent pattern detection

Autore: Claude Code
Data: 2025-09-02
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import warnings
from scipy import stats
from sklearn.metrics import mean_absolute_error
import logging

from arima_forecaster.utils.logger import get_logger
from arima_forecaster.utils.exceptions import ModelTrainingError

# Import all available models
from arima_forecaster.core.arima_model import ARIMAForecaster
from arima_forecaster.core.sarima_model import SARIMAForecaster  
from arima_forecaster.core.sarimax_model import SARIMAXForecaster
from arima_forecaster.core.var_model import VARForecaster
from arima_forecaster.core.intermittent_model import IntermittentForecaster, IntermittentPattern
from arima_forecaster.data.preprocessor import TimeSeriesPreprocessor
from arima_forecaster.evaluation.metrics import ModelEvaluator
from arima_forecaster.evaluation.intermittent_metrics import IntermittentEvaluator

# Try Prophet import
try:
    from arima_forecaster.core.prophet_model import ProphetForecaster
    _has_prophet = True
except ImportError:
    _has_prophet = False

logger = get_logger(__name__)


class DataType(str, Enum):
    """Tipologie di serie temporali identificabili"""
    REGULAR = "regular"           # Serie regolare senza gap
    INTERMITTENT = "intermittent" # Domanda sporadica (spare parts)
    SEASONAL = "seasonal"         # Con stagionalità marcata  
    TRENDING = "trending"         # Con trend chiaro
    MULTIVARIATE = "multivariate" # Multiple series correlate
    HIGH_FREQUENCY = "high_freq"  # Dati intra-day
    VOLATILE = "volatile"         # Alta variabilità, possibili outlier


class ModelType(str, Enum):
    """Modelli disponibili per selezione"""
    ARIMA = "ARIMA"
    SARIMA = "SARIMA" 
    SARIMAX = "SARIMAX"
    PROPHET = "Prophet"
    VAR = "VAR"
    INTERMITTENT = "Intermittent"
    ENSEMBLE = "Ensemble"


@dataclass
class PatternAnalysis:
    """Risultato analisi pattern serie temporale"""
    data_type: DataType
    characteristics: Dict[str, float]
    recommendations: List[ModelType] 
    confidence_score: float
    explanation: str
    warnings: List[str]
    

@dataclass
class ModelResult:
    """Risultato training e valutazione singolo modello"""
    model_type: ModelType
    model_instance: Any
    accuracy_score: float
    performance_metrics: Dict[str, float]
    training_time: float
    confidence: float
    error_message: Optional[str] = None


@dataclass
class AutoMLExplanation:
    """Spiegazione dettagliata scelta modello"""
    recommended_model: ModelType
    confidence_score: float
    why_chosen: str
    pattern_detected: str
    business_recommendation: str
    alternative_models: List[Dict[str, Any]]
    risk_assessment: str


class SeriesPatternDetector:
    """
    Engine avanzato per detection pattern serie temporali
    """
    
    def __init__(self):
        self.intermittent_analyzer = IntermittentForecaster()
        
    def analyze_series(self, data: Union[pd.Series, np.ndarray]) -> PatternAnalysis:
        """
        Analisi completa pattern serie temporale
        
        Args:
            data: Serie temporale da analizzare
            
        Returns:
            PatternAnalysis con raccomandazioni
        """
        if isinstance(data, pd.Series):
            values = data.values
            has_datetime_index = isinstance(data.index, pd.DatetimeIndex)
        else:
            values = np.array(data)
            has_datetime_index = False
            
        characteristics = {}
        warnings_list = []
        
        # 1. Basic statistics
        characteristics['length'] = len(values)
        characteristics['mean'] = np.mean(values)
        characteristics['std'] = np.std(values)
        characteristics['cv'] = characteristics['std'] / characteristics['mean'] if characteristics['mean'] != 0 else 0
        
        # 2. Missing values and zeros
        characteristics['missing_ratio'] = np.sum(np.isnan(values)) / len(values)
        characteristics['zero_ratio'] = np.sum(values == 0) / len(values)
        
        # 3. Trend detection (linear regression slope)
        x = np.arange(len(values))
        if len(values) > 2:
            slope, _, r_value, p_value, _ = stats.linregress(x, values)
            characteristics['trend_slope'] = slope
            characteristics['trend_r2'] = r_value ** 2
            characteristics['trend_significance'] = p_value
        else:
            characteristics['trend_slope'] = 0
            characteristics['trend_r2'] = 0
            characteristics['trend_significance'] = 1.0
            
        # 4. Seasonality detection (multiple frequencies)
        characteristics.update(self._detect_seasonality(values))
        
        # 5. Intermittency analysis
        intermittent_pattern = self.intermittent_analyzer.analyze_pattern(values)
        characteristics['intermittency'] = intermittent_pattern.intermittence
        characteristics['adi'] = intermittent_pattern.adi
        characteristics['cv2'] = intermittent_pattern.cv2
        characteristics['intermittent_classification'] = intermittent_pattern.classification
        
        # 6. Volatility and outliers
        characteristics.update(self._analyze_volatility(values))
        
        # 7. Stationarity test
        characteristics.update(self._test_stationarity(values))
        
        # 8. Data quality checks
        if characteristics['missing_ratio'] > 0.1:
            warnings_list.append(f"High missing values: {characteristics['missing_ratio']:.1%}")
            
        if characteristics['length'] < 50:
            warnings_list.append(f"Short series: {characteristics['length']} observations")
            
        # 9. Pattern classification e raccomandazioni
        data_type, recommendations, confidence, explanation = self._classify_pattern(characteristics)
        
        return PatternAnalysis(
            data_type=data_type,
            characteristics=characteristics,
            recommendations=recommendations,
            confidence_score=confidence,
            explanation=explanation,
            warnings=warnings_list
        )
        
    def _detect_seasonality(self, values: np.ndarray) -> Dict[str, float]:
        """Rileva stagionalità a multiple frequenze"""
        if len(values) < 24:  # Troppo corto
            return {
                'seasonality_strength': 0.0,
                'dominant_period': 0,
                'seasonal_significance': 1.0
            }
            
        # Test multiple periods comuni
        periods_to_test = [7, 12, 24, 30, 52, 365]  # weekly, monthly, etc
        max_period = min(len(values) // 3, max(periods_to_test))
        periods_to_test = [p for p in periods_to_test if p <= max_period]
        
        best_strength = 0
        best_period = 0
        
        for period in periods_to_test:
            if len(values) >= 2 * period:
                # Calcola autocorrelazione al lag stagionale
                try:
                    # Semplice test: correlazione tra valore e valore period steps prima
                    seasonal_values = values[period:]
                    lagged_values = values[:-period]
                    
                    if len(seasonal_values) > 0 and len(lagged_values) > 0:
                        correlation = np.corrcoef(seasonal_values, lagged_values)[0, 1]
                        if not np.isnan(correlation) and abs(correlation) > best_strength:
                            best_strength = abs(correlation)
                            best_period = period
                except:
                    continue
                    
        return {
            'seasonality_strength': best_strength,
            'dominant_period': best_period,
            'seasonal_significance': 1 - best_strength  # p-value approssimato
        }
        
    def _analyze_volatility(self, values: np.ndarray) -> Dict[str, float]:
        """Analizza volatilità e outlier"""
        if len(values) < 10:
            return {'volatility': 0, 'outlier_ratio': 0}
            
        # Volatility come rolling std
        window = min(30, len(values) // 4)
        if window >= 2:
            rolling_std = pd.Series(values).rolling(window).std()
            volatility = np.nanmean(rolling_std)
        else:
            volatility = np.std(values)
            
        # Outlier detection con IQR
        q25, q75 = np.percentile(values, [25, 75])
        iqr = q75 - q25
        if iqr > 0:
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            outliers = np.sum((values < lower_bound) | (values > upper_bound))
            outlier_ratio = outliers / len(values)
        else:
            outlier_ratio = 0
            
        return {
            'volatility': volatility,
            'outlier_ratio': outlier_ratio
        }
        
    def _test_stationarity(self, values: np.ndarray) -> Dict[str, float]:
        """Test stazionarietà semplificato"""
        if len(values) < 20:
            return {'is_stationary': 0.5, 'stationarity_confidence': 0.0}
            
        # Test semplice: confronta mean/std di prima e seconda metà
        mid = len(values) // 2
        first_half = values[:mid]
        second_half = values[mid:]
        
        # T-test per differenza medie
        try:
            t_stat, p_value = stats.ttest_ind(first_half, second_half)
            mean_stable = p_value > 0.05  # No differenza significativa
        except:
            mean_stable = True
            p_value = 0.5
            
        # F-test per differenza varianze (semplificato)
        var1, var2 = np.var(first_half), np.var(second_half)
        var_stable = abs(var1 - var2) / max(var1, var2, 1e-10) < 0.5
        
        is_stationary = mean_stable and var_stable
        confidence = 1 - p_value if is_stationary else p_value
        
        return {
            'is_stationary': 1.0 if is_stationary else 0.0,
            'stationarity_confidence': confidence
        }
        
    def _classify_pattern(self, char: Dict[str, float]) -> Tuple[DataType, List[ModelType], float, str]:
        """Classifica pattern e raccomanda modelli"""
        
        # Decision tree per classification
        
        # 1. Check Intermittent Demand
        if char['zero_ratio'] > 0.3 and char['intermittency'] > 0.3:
            return (
                DataType.INTERMITTENT,
                [ModelType.INTERMITTENT],
                0.9,
                f"Domanda intermittente rilevata ({char['zero_ratio']:.1%} zeri, ADI={char['adi']:.1f})"
            )
            
        # 2. Check Strong Seasonality  
        if char['seasonality_strength'] > 0.6 and char['dominant_period'] > 0:
            if char['trend_r2'] > 0.3:
                models = [ModelType.PROPHET, ModelType.SARIMA] if _has_prophet else [ModelType.SARIMA]
                return (
                    DataType.SEASONAL,
                    models,
                    0.85,
                    f"Stagionalità forte (period={char['dominant_period']}, strength={char['seasonality_strength']:.2f}) con trend"
                )
            else:
                return (
                    DataType.SEASONAL,
                    [ModelType.SARIMA],
                    0.8,
                    f"Stagionalità detectata (period={char['dominant_period']})"
                )
                
        # 3. Check Strong Trend
        if char['trend_r2'] > 0.5 and char['trend_significance'] < 0.05:
            models = [ModelType.PROPHET, ModelType.ARIMA] if _has_prophet else [ModelType.ARIMA]
            return (
                DataType.TRENDING,
                models,
                0.8,
                f"Trend significativo rilevato (R²={char['trend_r2']:.2f})"
            )
            
        # 4. Check High Volatility
        if char['outlier_ratio'] > 0.1 or char['cv'] > 2.0:
            models = [ModelType.PROPHET] if _has_prophet else [ModelType.ARIMA]
            return (
                DataType.VOLATILE,
                models,
                0.7,
                f"Serie volatile (CV={char['cv']:.1f}, outliers={char['outlier_ratio']:.1%})"
            )
            
        # 5. Check Short Series
        if char['length'] < 100:
            return (
                DataType.REGULAR,
                [ModelType.ARIMA],
                0.6,
                f"Serie breve ({char['length']} obs), ARIMA semplice consigliato"
            )
            
        # 6. Default: Regular series
        return (
            DataType.REGULAR,
            [ModelType.ARIMA, ModelType.SARIMA],
            0.7,
            "Serie regolare senza pattern particolari"
        )


class AutoForecastSelector:
    """
    AutoML Engine per selezione automatica modello ottimale
    
    One-click forecasting con spiegazioni intelligenti
    """
    
    def __init__(self, 
                 validation_split: float = 0.2,
                 max_models_to_try: int = 5,
                 timeout_per_model: float = 60.0,
                 verbose: bool = True):
        """
        Inizializza AutoML selector
        
        Args:
            validation_split: % dati per test
            max_models_to_try: Massimo modelli da testare
            timeout_per_model: Timeout training per modello (secondi)  
            verbose: Output dettagliato
        """
        self.validation_split = validation_split
        self.max_models_to_try = max_models_to_try
        self.timeout_per_model = timeout_per_model
        self.verbose = verbose
        
        self.detector = SeriesPatternDetector()
        self.preprocessor = TimeSeriesPreprocessor()
        
        # Storage results
        self.pattern_analysis_ = None
        self.model_results_ = []
        self.best_model_ = None
        self.explanation_ = None
        
    def fit(self, 
            data: Union[pd.Series, np.ndarray],
            exog: Optional[Union[pd.DataFrame, np.ndarray]] = None) -> Tuple[Any, AutoMLExplanation]:
        """
        AutoML main method: analizza, seleziona e addestra modello ottimale
        
        Args:
            data: Serie temporale
            exog: Variabili esogene opzionali
            
        Returns:
            Tuple(best_model, explanation)
        """
        if self.verbose:
            print("[AutoML] Forecasting Engine Starting...")
            
        # 1. Pattern Analysis
        if self.verbose:
            print("[ANALYSIS] Analyzing data patterns...")
            
        self.pattern_analysis_ = self.detector.analyze_series(data)
        
        if self.verbose:
            print(f"   Detected: {self.pattern_analysis_.data_type.value}")
            print(f"   Confidence: {self.pattern_analysis_.confidence_score:.1%}")
            
        # 2. Data Preprocessing
        if self.verbose:
            print("[PREP] Preprocessing data...")
            
        try:
            clean_data, metadata = self.preprocessor.preprocess_pipeline(
                data,
                handle_missing='interpolate',
                test_stationarity=True
            )
        except Exception as e:
            logger.warning(f"Preprocessing failed, using raw data: {e}")
            clean_data = data if isinstance(data, pd.Series) else pd.Series(data)
            metadata = {}
            
        # 3. Train/Test Split
        split_idx = int(len(clean_data) * (1 - self.validation_split))
        train_data = clean_data[:split_idx]
        test_data = clean_data[split_idx:]
        
        if self.verbose:
            print(f"   Train: {len(train_data)} | Test: {len(test_data)}")
            
        # 4. Model Selection & Training
        if self.verbose:
            print("[TRAINING] Testing candidate models...")
            
        self.model_results_ = self._train_candidate_models(
            train_data, test_data, exog
        )
        
        # 5. Select Best Model
        self.best_model_ = self._select_best_model()
        
        # 6. Generate Explanation
        self.explanation_ = self._generate_explanation()
        
        if self.verbose:
            print(f"[SUCCESS] Best model: {self.explanation_.recommended_model}")
            print(f"   Confidence: {self.explanation_.confidence_score:.1%}")
            print(f"   Reason: {self.explanation_.why_chosen}")
            
        return self.best_model_.model_instance, self.explanation_
        
    def _train_candidate_models(self, 
                               train_data: pd.Series,
                               test_data: pd.Series,
                               exog: Optional[np.ndarray] = None) -> List[ModelResult]:
        """Addestra modelli candidati basandosi su raccomandazioni"""
        
        results = []
        recommended_models = self.pattern_analysis_.recommendations[:self.max_models_to_try]
        
        for model_type in recommended_models:
            if self.verbose:
                print(f"   Training {model_type.value}...")
                
            try:
                result = self._train_single_model(model_type, train_data, test_data, exog)
                if result:
                    results.append(result)
                    if self.verbose:
                        print(f"     [OK] Score: {result.accuracy_score:.3f}")
            except Exception as e:
                if self.verbose:
                    print(f"     [FAIL] {str(e)}")
                logger.warning(f"{model_type.value} training failed: {e}")
                
        return results
        
    def _train_single_model(self,
                           model_type: ModelType,
                           train_data: pd.Series,
                           test_data: pd.Series,
                           exog: Optional[np.ndarray] = None) -> Optional[ModelResult]:
        """Addestra singolo modello e valuta performance"""
        
        import time
        start_time = time.time()
        
        try:
            # Initialize model
            if model_type == ModelType.ARIMA:
                model = ARIMAForecaster(order=(1,1,1))
                
            elif model_type == ModelType.SARIMA:
                # Auto-detect seasonal period  
                season_period = self.pattern_analysis_.characteristics.get('dominant_period', 12)
                if season_period == 0:
                    season_period = 12
                model = SARIMAForecaster(
                    order=(1,1,1),
                    seasonal_order=(1,1,1,season_period)
                )
                
            elif model_type == ModelType.SARIMAX and exog is not None:
                model = SARIMAXForecaster(order=(1,1,1))
                
            elif model_type == ModelType.INTERMITTENT:
                from arima_forecaster.core.intermittent_model import IntermittentConfig, IntermittentMethod
                config = IntermittentConfig(method=IntermittentMethod.SBA, optimize_alpha=True)
                model = IntermittentForecaster(config)
                
            elif model_type == ModelType.PROPHET and _has_prophet:
                model = ProphetForecaster(
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    country_holidays='IT'
                )
                
            else:
                return None
                
            # Train model
            if model_type == ModelType.SARIMAX and exog is not None:
                # Split exog too
                split_idx = len(train_data)
                train_exog = exog[:split_idx]
                test_exog = exog[split_idx:split_idx + len(test_data)]
                model.fit(train_data, exog=train_exog)
                forecast = model.forecast(len(test_data), exog=test_exog)
            else:
                model.fit(train_data)
                forecast = model.forecast(len(test_data))
                
            training_time = time.time() - start_time
            
            # Evaluate performance
            if model_type == ModelType.INTERMITTENT:
                evaluator = IntermittentEvaluator()
                metrics = evaluator.evaluate(test_data.values, forecast)
                accuracy_score = 1.0 / max(metrics.mase, 0.1)  # Convert MASE to score
                performance_metrics = {
                    'mase': metrics.mase,
                    'fill_rate': metrics.fill_rate,
                    'service_level': metrics.achieved_service_level
                }
            else:
                # Simple evaluation using MAE
                mae = mean_absolute_error(test_data.values, forecast)
                accuracy_score = 1.0 / (1.0 + mae / max(test_data.mean(), 1))
                performance_metrics = {
                    'mae': mae,
                    'mape': 100 * mae / max(test_data.mean(), 1)
                }
                
            # Calculate confidence based on multiple factors
            confidence = self._calculate_model_confidence(
                accuracy_score, training_time, model_type
            )
            
            return ModelResult(
                model_type=model_type,
                model_instance=model,
                accuracy_score=accuracy_score,
                performance_metrics=performance_metrics,
                training_time=training_time,
                confidence=confidence
            )
            
        except Exception as e:
            return ModelResult(
                model_type=model_type,
                model_instance=None,
                accuracy_score=0.0,
                performance_metrics={},
                training_time=time.time() - start_time,
                confidence=0.0,
                error_message=str(e)
            )
            
    def _calculate_model_confidence(self, 
                                   accuracy_score: float,
                                   training_time: float,
                                   model_type: ModelType) -> float:
        """Calcola confidence score per modello"""
        
        # Base confidence from accuracy
        confidence = min(accuracy_score, 1.0)
        
        # Penalty for excessive training time
        if training_time > 30:
            confidence *= 0.9
        elif training_time > 60:
            confidence *= 0.8
            
        # Bonus for recommended model type
        pattern_recommendations = self.pattern_analysis_.recommendations
        if model_type in pattern_recommendations:
            position = pattern_recommendations.index(model_type)
            bonus = 0.1 * (len(pattern_recommendations) - position) / len(pattern_recommendations)
            confidence += bonus
            
        return min(confidence, 1.0)
        
    def _select_best_model(self) -> ModelResult:
        """Seleziona modello migliore basandosi su score e confidence"""
        
        if not self.model_results_:
            raise ModelTrainingError("No models trained successfully")
            
        # Filter successful models
        successful_models = [r for r in self.model_results_ if r.model_instance is not None]
        
        if not successful_models:
            raise ModelTrainingError("All models failed to train")
            
        # Score combinato: accuracy + confidence
        for result in successful_models:
            result.combined_score = (result.accuracy_score * 0.7 + result.confidence * 0.3)
            
        # Sort by combined score
        successful_models.sort(key=lambda x: x.combined_score, reverse=True)
        
        return successful_models[0]
        
    def _generate_explanation(self) -> AutoMLExplanation:
        """Genera spiegazione dettagliata scelta modello"""
        
        pattern = self.pattern_analysis_
        best = self.best_model_
        
        # Why chosen
        why_chosen = f"Modello {best.model_type.value} selezionato per "
        
        if pattern.data_type == DataType.INTERMITTENT:
            why_chosen += f"domanda intermittente (zeri: {pattern.characteristics['zero_ratio']:.1%})"
        elif pattern.data_type == DataType.SEASONAL:
            period = pattern.characteristics['dominant_period']
            why_chosen += f"forte stagionalità (periodo={period})"
        elif pattern.data_type == DataType.TRENDING:
            r2 = pattern.characteristics['trend_r2']
            why_chosen += f"trend significativo (R²={r2:.2f})"
        else:
            why_chosen += "pattern regolare identificato"
            
        # Business recommendation
        if best.model_type == ModelType.INTERMITTENT:
            business_rec = f"Gestire come spare part. Safety stock raccomandato basato su service level 95%."
        elif pattern.characteristics['seasonality_strength'] > 0.5:
            business_rec = f"Pianificare per stagionalità periodo {pattern.characteristics['dominant_period']}."
        elif pattern.characteristics['trend_slope'] > 0:
            business_rec = "Trend crescente: considerare aumento capacità."
        else:
            business_rec = "Domanda stabile: focus su efficienza operativa."
            
        # Alternative models
        alternatives = []
        for result in self.model_results_[1:4]:  # Top 3 alternatives
            if result.model_instance:
                alternatives.append({
                    'model': result.model_type.value,
                    'score': result.accuracy_score,
                    'reason': f"Score: {result.accuracy_score:.3f}"
                })
                
        # Risk assessment
        if best.confidence < 0.7:
            risk = "ALTO: Confidence bassa, monitorare performance"
        elif pattern.characteristics['length'] < 100:
            risk = "MEDIO: Serie breve, potrebbero servire più dati"
        elif pattern.characteristics['outlier_ratio'] > 0.1:
            risk = "MEDIO: Presenza outlier, validation continua necessaria"
        else:
            risk = "BASSO: Modello stabile e affidabile"
            
        return AutoMLExplanation(
            recommended_model=best.model_type,
            confidence_score=best.confidence,
            why_chosen=why_chosen,
            pattern_detected=pattern.explanation,
            business_recommendation=business_rec,
            alternative_models=alternatives,
            risk_assessment=risk
        )
        
    def get_model_comparison(self) -> pd.DataFrame:
        """Ritorna DataFrame con confronto tutti i modelli testati"""
        
        if not self.model_results_:
            return pd.DataFrame()
            
        data = []
        for result in self.model_results_:
            row = {
                'Model': result.model_type.value,
                'Accuracy_Score': result.accuracy_score,
                'Confidence': result.confidence,
                'Training_Time': result.training_time,
                'Status': 'Success' if result.model_instance else 'Failed'
            }
            
            # Add model-specific metrics
            if 'mase' in result.performance_metrics:
                row['MASE'] = result.performance_metrics['mase']
            if 'mape' in result.performance_metrics:
                row['MAPE'] = result.performance_metrics['mape']
                
            data.append(row)
            
        df = pd.DataFrame(data)
        return df.sort_values('Accuracy_Score', ascending=False) if not df.empty else df
        
    def __repr__(self) -> str:
        if self.best_model_:
            return f"AutoForecastSelector(best_model={self.best_model_.model_type.value}, confidence={self.best_model_.confidence:.1%})"
        return "AutoForecastSelector(not_fitted=True)"