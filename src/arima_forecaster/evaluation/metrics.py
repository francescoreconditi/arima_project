"""
Comprehensive evaluation metrics for time series forecasting models.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple
from scipy import stats
from ..utils.logger import get_logger
from ..utils.exceptions import ValidationError


class ModelEvaluator:
    """
    Comprehensive evaluation metrics and diagnostics for ARIMA models.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
    def calculate_forecast_metrics(
        self,
        actual: Union[pd.Series, np.ndarray],
        predicted: Union[pd.Series, np.ndarray],
        return_all: bool = True
    ) -> Dict[str, float]:
        """
        Calculate comprehensive forecast accuracy metrics.
        
        Args:
            actual: Actual observed values
            predicted: Predicted/forecast values
            return_all: Whether to return all metrics or just main ones
            
        Returns:
            Dictionary with calculated metrics
        """
        try:
            # Convert to numpy arrays for calculation
            actual = np.asarray(actual)
            predicted = np.asarray(predicted)
            
            # Validate inputs
            if len(actual) != len(predicted):
                raise ValidationError("Actual and predicted arrays must have same length")
            
            if len(actual) == 0:
                raise ValidationError("Arrays cannot be empty")
            
            # Remove any infinite or NaN values
            mask = np.isfinite(actual) & np.isfinite(predicted)
            actual = actual[mask]
            predicted = predicted[mask]
            
            if len(actual) == 0:
                raise ValidationError("No valid data points after removing NaN/infinite values")
            
            # Calculate errors
            errors = actual - predicted
            abs_errors = np.abs(errors)
            squared_errors = errors ** 2
            
            # Basic metrics
            metrics = {
                'mae': np.mean(abs_errors),  # Mean Absolute Error
                'mse': np.mean(squared_errors),  # Mean Squared Error
                'rmse': np.sqrt(np.mean(squared_errors)),  # Root Mean Squared Error
                'mape': self._calculate_mape(actual, predicted),  # Mean Absolute Percentage Error
                'smape': self._calculate_smape(actual, predicted),  # Symmetric MAPE
                'r_squared': self._calculate_r_squared(actual, predicted),  # R-squared
                'correlation': np.corrcoef(actual, predicted)[0, 1] if len(actual) > 1 else 0
            }
            
            if return_all:
                # Additional metrics
                additional_metrics = {
                    'mean_error': np.mean(errors),  # Mean Error (bias)
                    'median_ae': np.median(abs_errors),  # Median Absolute Error
                    'max_error': np.max(abs_errors),  # Maximum Error
                    'min_error': np.min(abs_errors),  # Minimum Error
                    'std_error': np.std(errors),  # Standard deviation of errors
                    'iqr_error': np.percentile(abs_errors, 75) - np.percentile(abs_errors, 25),  # IQR of errors
                    'q90_error': np.percentile(abs_errors, 90),  # 90th percentile of absolute errors
                    'q95_error': np.percentile(abs_errors, 95),  # 95th percentile of absolute errors
                    'theil_u': self._calculate_theil_u(actual, predicted),  # Theil's U statistic
                    'mase': self._calculate_mase(actual, predicted),  # Mean Absolute Scaled Error
                }
                metrics.update(additional_metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating forecast metrics: {e}")
            raise ValidationError(f"Failed to calculate metrics: {e}")
    
    def evaluate_residuals(
        self,
        residuals: Union[pd.Series, np.ndarray],
        fitted_values: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive residual analysis.
        
        Args:
            residuals: Model residuals
            fitted_values: Optional fitted values for additional tests
            
        Returns:
            Dictionary with residual diagnostic results
        """
        try:
            residuals = np.asarray(residuals)
            
            # Remove any infinite or NaN values
            clean_residuals = residuals[np.isfinite(residuals)]
            
            if len(clean_residuals) == 0:
                raise ValidationError("No valid residuals after cleaning")
            
            diagnostics = {
                'n_observations': len(clean_residuals),
                'mean': np.mean(clean_residuals),
                'std': np.std(clean_residuals),
                'min': np.min(clean_residuals),
                'max': np.max(clean_residuals),
                'skewness': stats.skew(clean_residuals),
                'kurtosis': stats.kurtosis(clean_residuals),
                'jarque_bera_test': self._jarque_bera_test(clean_residuals),
                'ljung_box_test': self._ljung_box_test(clean_residuals),
                'runs_test': self._runs_test(clean_residuals),
                'durbin_watson': self._durbin_watson_test(clean_residuals)
            }
            
            # Additional tests if fitted values provided
            if fitted_values is not None:
                fitted_values = np.asarray(fitted_values)
                if len(fitted_values) == len(residuals):
                    clean_fitted = fitted_values[np.isfinite(residuals)]
                    diagnostics['breusch_pagan_test'] = self._breusch_pagan_test(clean_residuals, clean_fitted)
            
            return diagnostics
            
        except Exception as e:
            self.logger.error(f"Error in residual analysis: {e}")
            raise ValidationError(f"Failed to analyze residuals: {e}")
    
    def _calculate_mape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        # Avoid division by zero
        nonzero_mask = actual != 0
        if not nonzero_mask.any():
            return float('inf')
        
        mape = np.mean(np.abs((actual[nonzero_mask] - predicted[nonzero_mask]) / actual[nonzero_mask])) * 100
        return mape
    
    def _calculate_smape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error."""
        denominator = (np.abs(actual) + np.abs(predicted)) / 2
        mask = denominator != 0
        
        if not mask.any():
            return 0.0
        
        smape = np.mean(np.abs(actual[mask] - predicted[mask]) / denominator[mask]) * 100
        return smape
    
    def _calculate_r_squared(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate R-squared (coefficient of determination)."""
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        
        r_squared = 1 - (ss_res / ss_tot)
        return r_squared
    
    def _calculate_theil_u(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Theil's U statistic."""
        if len(actual) < 2:
            return float('nan')
        
        # Theil's U2 statistic
        numerator = np.sqrt(np.mean((predicted[1:] - actual[1:]) ** 2))
        denominator = np.sqrt(np.mean((actual[1:] - actual[:-1]) ** 2))
        
        if denominator == 0:
            return float('inf') if numerator > 0 else 0.0
        
        return numerator / denominator
    
    def _calculate_mase(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Mean Absolute Scaled Error."""
        if len(actual) < 2:
            return float('nan')
        
        # Scale by naive forecast (seasonal naive if seasonal pattern detected)
        mae_forecast = np.mean(np.abs(actual - predicted))
        mae_naive = np.mean(np.abs(actual[1:] - actual[:-1]))  # Simple naive forecast
        
        if mae_naive == 0:
            return float('inf') if mae_forecast > 0 else 0.0
        
        return mae_forecast / mae_naive
    
    def _jarque_bera_test(self, residuals: np.ndarray) -> Dict[str, float]:
        """Jarque-Bera test for normality."""
        try:
            statistic, p_value = stats.jarque_bera(residuals)
            return {
                'statistic': float(statistic),
                'p_value': float(p_value),
                'is_normal': p_value > 0.05
            }
        except:
            return {'statistic': float('nan'), 'p_value': float('nan'), 'is_normal': False}
    
    def _ljung_box_test(self, residuals: np.ndarray, lags: int = 10) -> Dict[str, float]:
        """Ljung-Box test for autocorrelation."""
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            
            # Adjust lags if necessary
            max_lags = min(lags, len(residuals) // 4)
            if max_lags < 1:
                return {'statistic': float('nan'), 'p_value': float('nan'), 'is_white_noise': False}
            
            result = acorr_ljungbox(residuals, lags=max_lags, return_df=False)
            statistic = result[0][-1] if hasattr(result[0], '__iter__') else result[0]
            p_value = result[1][-1] if hasattr(result[1], '__iter__') else result[1]
            
            return {
                'statistic': float(statistic),
                'p_value': float(p_value),
                'is_white_noise': p_value > 0.05
            }
        except:
            return {'statistic': float('nan'), 'p_value': float('nan'), 'is_white_noise': False}
    
    def _runs_test(self, residuals: np.ndarray) -> Dict[str, Any]:
        """Runs test for randomness."""
        try:
            median = np.median(residuals)
            runs = np.diff([x > median for x in residuals]).sum() + 1
            n1 = sum(x > median for x in residuals)
            n2 = len(residuals) - n1
            
            if n1 == 0 or n2 == 0:
                return {'runs': runs, 'expected_runs': float('nan'), 'is_random': False}
            
            expected_runs = 2 * n1 * n2 / (n1 + n2) + 1
            variance = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
            
            if variance <= 0:
                return {'runs': runs, 'expected_runs': expected_runs, 'is_random': False}
            
            z_score = (runs - expected_runs) / np.sqrt(variance)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            
            return {
                'runs': runs,
                'expected_runs': expected_runs,
                'z_score': z_score,
                'p_value': p_value,
                'is_random': p_value > 0.05
            }
        except:
            return {'runs': float('nan'), 'expected_runs': float('nan'), 'is_random': False}
    
    def _durbin_watson_test(self, residuals: np.ndarray) -> float:
        """Durbin-Watson test statistic for autocorrelation."""
        try:
            diff_residuals = np.diff(residuals)
            dw_stat = np.sum(diff_residuals ** 2) / np.sum(residuals ** 2)
            return float(dw_stat)
        except:
            return float('nan')
    
    def _breusch_pagan_test(self, residuals: np.ndarray, fitted_values: np.ndarray) -> Dict[str, float]:
        """Breusch-Pagan test for heteroscedasticity."""
        try:
            from scipy.stats import chi2
            
            # Regression of squared residuals on fitted values
            n = len(residuals)
            squared_residuals = residuals ** 2
            
            # Simple linear regression
            X = np.column_stack([np.ones(n), fitted_values])
            
            # Avoid singular matrix
            if np.linalg.matrix_rank(X) < X.shape[1]:
                return {'statistic': float('nan'), 'p_value': float('nan'), 'is_homoscedastic': False}
            
            beta = np.linalg.lstsq(X, squared_residuals, rcond=None)[0]
            predicted_sq_res = X @ beta
            
            # Calculate R-squared
            ss_res = np.sum((squared_residuals - predicted_sq_res) ** 2)
            ss_tot = np.sum((squared_residuals - np.mean(squared_residuals)) ** 2)
            
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Test statistic
            lm_stat = n * r_squared
            p_value = 1 - chi2.cdf(lm_stat, df=1)
            
            return {
                'statistic': float(lm_stat),
                'p_value': float(p_value),
                'is_homoscedastic': p_value > 0.05
            }
        except:
            return {'statistic': float('nan'), 'p_value': float('nan'), 'is_homoscedastic': False}
    
    def generate_evaluation_report(
        self,
        actual: Union[pd.Series, np.ndarray],
        predicted: Union[pd.Series, np.ndarray],
        residuals: Optional[Union[pd.Series, np.ndarray]] = None,
        model_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            residuals: Optional residuals for diagnostic tests
            model_info: Optional model information
            
        Returns:
            Complete evaluation report
        """
        report = {
            'forecast_metrics': self.calculate_forecast_metrics(actual, predicted),
            'model_info': model_info or {},
            'evaluation_timestamp': pd.Timestamp.now()
        }
        
        if residuals is not None:
            report['residual_diagnostics'] = self.evaluate_residuals(residuals, predicted)
        
        # Add interpretation
        report['interpretation'] = self._interpret_metrics(report['forecast_metrics'])
        
        return report
    
    def _interpret_metrics(self, metrics: Dict[str, float]) -> Dict[str, str]:
        """Provide interpretation of metrics."""
        interpretation = {}
        
        # MAPE interpretation
        mape = metrics.get('mape', float('inf'))
        if mape < 10:
            interpretation['mape'] = "Excellent forecast accuracy"
        elif mape < 20:
            interpretation['mape'] = "Good forecast accuracy"
        elif mape < 50:
            interpretation['mape'] = "Reasonable forecast accuracy"
        else:
            interpretation['mape'] = "Poor forecast accuracy"
        
        # R-squared interpretation
        r_squared = metrics.get('r_squared', 0)
        if r_squared > 0.9:
            interpretation['r_squared'] = "Excellent model fit"
        elif r_squared > 0.7:
            interpretation['r_squared'] = "Good model fit"
        elif r_squared > 0.5:
            interpretation['r_squared'] = "Moderate model fit"
        else:
            interpretation['r_squared'] = "Poor model fit"
        
        return interpretation