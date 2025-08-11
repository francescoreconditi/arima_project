"""
SARIMA model implementation with seasonal support.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from ..utils.logger import get_logger
from ..utils.exceptions import ModelTrainingError, ForecastError


class SARIMAForecaster:
    """
    SARIMA forecaster with seasonal support.
    """
    
    def __init__(
        self, 
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12),
        trend: Optional[str] = None
    ):
        """
        Initialize SARIMA forecaster.
        
        Args:
            order: Non-seasonal ARIMA order (p, d, q)
            seasonal_order: Seasonal ARIMA order (P, D, Q, s)
            trend: Trend parameter ('n', 'c', 't', 'ct')
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.model = None
        self.fitted_model = None
        self.training_data = None
        self.training_metadata = {}
        self.logger = get_logger(__name__)
        
    def fit(
        self, 
        series: pd.Series,
        validate_input: bool = True,
        **fit_kwargs
    ) -> 'SARIMAForecaster':
        """
        Fit SARIMA model to time series data.
        
        Args:
            series: Time series data to fit
            validate_input: Whether to validate input data
            **fit_kwargs: Additional arguments for model fitting
            
        Returns:
            Self for method chaining
            
        Raises:
            ModelTrainingError: If model training fails
        """
        try:
            self.logger.info(
                f"Fitting SARIMA{self.order}x{self.seasonal_order} model "
                f"to {len(series)} observations"
            )
            
            if validate_input:
                self._validate_series(series)
                self._validate_seasonal_parameters(series)
            
            # Store training data and metadata
            self.training_data = series.copy()
            self.training_metadata = {
                'training_start': series.index.min(),
                'training_end': series.index.max(), 
                'training_observations': len(series),
                'order': self.order,
                'seasonal_order': self.seasonal_order,
                'trend': self.trend
            }
            
            # Create and fit model
            self.model = SARIMAX(
                series, 
                order=self.order,
                seasonal_order=self.seasonal_order,
                trend=self.trend
            )
            self.fitted_model = self.model.fit(**fit_kwargs)
            
            # Log model summary
            self.logger.info("SARIMA model fitted successfully")
            self.logger.info(f"AIC: {self.fitted_model.aic:.2f}")
            self.logger.info(f"BIC: {self.fitted_model.bic:.2f}")
            
            return self
            
        except Exception as e:
            self.logger.error(f"SARIMA model fitting failed: {e}")
            raise ModelTrainingError(f"Impossibile addestrare il modello SARIMA: {e}")
    
    def forecast(
        self, 
        steps: int,
        confidence_intervals: bool = True,
        alpha: float = 0.05,
        return_conf_int: bool = False
    ) -> Union[pd.Series, Tuple[pd.Series, pd.DataFrame]]:
        """
        Generate forecasts from fitted SARIMA model.
        
        Args:
            steps: Number of steps to forecast
            confidence_intervals: Whether to calculate confidence intervals
            alpha: Alpha level for confidence intervals (1-alpha = confidence level)
            return_conf_int: Whether to return confidence intervals
            
        Returns:
            Forecast series, optionally with confidence intervals
            
        Raises:
            ForecastError: If forecasting fails
        """
        try:
            if self.fitted_model is None:
                raise ForecastError("SARIMA model must be fitted before forecasting")
            
            self.logger.info(f"Generating {steps}-step SARIMA forecast")
            
            # Generate forecast
            forecast_result = self.fitted_model.get_forecast(steps=steps, alpha=alpha)
            forecast_values = forecast_result.predicted_mean
            
            if confidence_intervals:
                conf_int = forecast_result.conf_int()
            else:
                conf_int = None
            
            # Create forecast index
            last_date = self.training_data.index[-1]
            if isinstance(last_date, pd.Timestamp):
                freq = pd.infer_freq(self.training_data.index)
                if freq:
                    try:
                        # Convert string frequency to DateOffset and add to timestamp
                        freq_offset = pd.tseries.frequencies.to_offset(freq)
                        forecast_index = pd.date_range(
                            start=last_date + freq_offset,
                            periods=steps,
                            freq=freq
                        )
                    except Exception:
                        # Fallback: use daily frequency
                        forecast_index = pd.date_range(
                            start=last_date + pd.Timedelta(days=1),
                            periods=steps,
                            freq='D'
                        )
                else:
                    # Fallback: use daily frequency if no frequency can be inferred
                    forecast_index = pd.date_range(
                        start=last_date + pd.Timedelta(days=1),
                        periods=steps,
                        freq='D'
                    )
            else:
                forecast_index = range(len(self.training_data), len(self.training_data) + steps)
            
            forecast_series = pd.Series(forecast_values, index=forecast_index, name='forecast')
            
            self.logger.info(
                f"SARIMA forecast generated: {forecast_series.iloc[0]:.2f} to {forecast_series.iloc[-1]:.2f}"
            )
            
            if return_conf_int and conf_int is not None:
                conf_int.index = forecast_index
                return forecast_series, conf_int
            else:
                return forecast_series
                
        except Exception as e:
            self.logger.error(f"SARIMA forecasting failed: {e}")
            raise ForecastError(f"Impossibile generare il forecast SARIMA: {e}")
    
    def predict(
        self, 
        start: Optional[Union[int, str, pd.Timestamp]] = None,
        end: Optional[Union[int, str, pd.Timestamp]] = None,
        dynamic: bool = False
    ) -> pd.Series:
        """
        Generate in-sample and out-of-sample predictions.
        
        Args:
            start: Start of prediction period
            end: End of prediction period  
            dynamic: Whether to use dynamic prediction
            
        Returns:
            Series of predictions
        """
        try:
            if self.fitted_model is None:
                raise ForecastError("SARIMA model must be fitted before prediction")
            
            predictions = self.fitted_model.predict(start=start, end=end, dynamic=dynamic)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"SARIMA prediction failed: {e}")
            raise ForecastError(f"Impossibile generare le predizioni SARIMA: {e}")
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save fitted SARIMA model to disk.
        
        Args:
            filepath: Path to save model
        """
        try:
            if self.fitted_model is None:
                raise ModelTrainingError("Nessun modello SARIMA addestrato da salvare")
            
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Save using statsmodels built-in method
            self.fitted_model.save(str(filepath))
            
            # Also save metadata
            metadata_path = filepath.with_suffix('.metadata.pkl')
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'order': self.order,
                    'seasonal_order': self.seasonal_order,
                    'trend': self.trend,
                    'training_metadata': self.training_metadata
                }, f)
            
            self.logger.info(f"SARIMA model saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save SARIMA model: {e}")
            raise ModelTrainingError(f"Impossibile salvare il modello SARIMA: {e}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'SARIMAForecaster':
        """
        Load fitted SARIMA model from disk.
        
        Args:
            filepath: Path to saved model
            
        Returns:
            Loaded SARIMAForecaster instance
        """
        try:
            filepath = Path(filepath)
            
            # Load the fitted model
            fitted_model = SARIMAXResults.load(str(filepath))
            
            # Load metadata if available
            metadata_path = filepath.with_suffix('.metadata.pkl')
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                order = metadata.get('order', (1, 1, 1))
                seasonal_order = metadata.get('seasonal_order', (1, 1, 1, 12))
                trend = metadata.get('trend', None)
                training_metadata = metadata.get('training_metadata', {})
            else:
                order = (1, 1, 1)  # Default order
                seasonal_order = (1, 1, 1, 12)  # Default seasonal order
                trend = None
                training_metadata = {}
            
            # Create instance and populate
            instance = cls(order=order, seasonal_order=seasonal_order, trend=trend)
            instance.fitted_model = fitted_model
            instance.training_metadata = training_metadata
            
            instance.logger.info(f"SARIMA model loaded from {filepath}")
            
            return instance
            
        except Exception as e:
            logger = get_logger(__name__)
            logger.error(f"Impossibile caricare il modello SARIMA: {e}")
            raise ModelTrainingError(f"Impossibile caricare il modello SARIMA: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive SARIMA model information.
        
        Returns:
            Dictionary with model information
        """
        if self.fitted_model is None:
            return {'status': 'not_fitted'}
        
        info = {
            'status': 'fitted',
            'model_type': 'SARIMA',
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'trend': self.trend,
            'aic': self.fitted_model.aic,
            'bic': self.fitted_model.bic,
            'hqic': self.fitted_model.hqic,
            'llf': self.fitted_model.llf,
            'n_observations': self.fitted_model.nobs,
            'params': dict(self.fitted_model.params),
            'training_metadata': self.training_metadata
        }
        
        return info
    
    def get_seasonal_decomposition(self) -> Dict[str, pd.Series]:
        """
        Get seasonal decomposition of the fitted model.
        
        Returns:
            Dictionary with decomposition components
        """
        if self.fitted_model is None:
            raise ForecastError("SARIMA model must be fitted before decomposition")
        
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            decomposition = seasonal_decompose(
                self.training_data, 
                model='additive',
                period=self.seasonal_order[3]  # seasonal period
            )
            
            return {
                'observed': decomposition.observed,
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid
            }
            
        except Exception as e:
            self.logger.error(f"Seasonal decomposition failed: {e}")
            raise ForecastError(f"Impossibile eseguire la decomposizione stagionale: {e}")
    
    def _validate_series(self, series: pd.Series) -> None:
        """
        Validate input time series.
        
        Args:
            series: Series to validate
            
        Raises:
            ModelTrainingError: If validation fails
        """
        if not isinstance(series, pd.Series):
            raise ModelTrainingError("Input must be a pandas Series")
        
        if len(series) == 0:
            raise ModelTrainingError("Series cannot be empty")
        
        if series.isnull().all():
            raise ModelTrainingError("Series cannot be all NaN")
        
        if len(series) < 10:
            self.logger.warning("Series has fewer than 10 observations, model may be unreliable")
        
        if series.isnull().any():
            missing_pct = series.isnull().sum() / len(series) * 100
            self.logger.warning(f"Series contains {missing_pct:.1f}% missing values")
    
    def _validate_seasonal_parameters(self, series: pd.Series) -> None:
        """
        Validate seasonal parameters against the data.
        
        Args:
            series: Series to validate against
            
        Raises:
            ModelTrainingError: If validation fails
        """
        seasonal_period = self.seasonal_order[3]
        
        if seasonal_period <= 1:
            raise ModelTrainingError("Seasonal period must be greater than 1")
        
        if len(series) < 2 * seasonal_period:
            self.logger.warning(
                f"Series length ({len(series)}) is less than 2 seasonal periods "
                f"({2 * seasonal_period}). Model may be unreliable."
            )
        
        # Check if seasonal period makes sense for the data frequency
        if hasattr(series.index, 'freq') and series.index.freq is not None:
            freq = series.index.freq
            if 'D' in str(freq) and seasonal_period not in [7, 30, 365]:
                self.logger.warning(
                    f"Daily data with seasonal period {seasonal_period} may not be appropriate. "
                    "Consider 7 (weekly), 30 (monthly), or 365 (yearly)."
                )
            elif 'M' in str(freq) and seasonal_period != 12:
                self.logger.warning(
                    f"Monthly data with seasonal period {seasonal_period} may not be appropriate. "
                    "Consider 12 (yearly)."
                )