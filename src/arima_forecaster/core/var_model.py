"""
Vector Autoregression (VAR) model for multivariate time series forecasting.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.var_model import VARResults
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from ..utils.logger import get_logger
from ..utils.exceptions import ModelTrainingError, ForecastError


class VARForecaster:
    """
    Vector Autoregression forecaster for multivariate time series.
    """
    
    def __init__(self, maxlags: Optional[int] = None, ic: str = 'aic'):
        """
        Initialize VAR forecaster.
        
        Args:
            maxlags: Maximum number of lags to consider
            ic: Information criterion for lag selection ('aic', 'bic', 'hqic', 'fpe')
        """
        self.maxlags = maxlags
        self.ic = ic
        self.model = None
        self.fitted_model = None
        self.training_data = None
        self.training_metadata = {}
        self.selected_lag = None
        self.logger = get_logger(__name__)
        
        if self.ic not in ['aic', 'bic', 'hqic', 'fpe']:
            raise ValueError("ic must be one of 'aic', 'bic', 'hqic', 'fpe'")
    
    def fit(
        self, 
        data: pd.DataFrame,
        validate_input: bool = True,
        trend: str = 'c',
        **fit_kwargs
    ) -> 'VARForecaster':
        """
        Fit VAR model to multivariate time series data.
        
        Args:
            data: DataFrame with multiple time series columns
            validate_input: Whether to validate input data
            trend: Trend parameter ('c', 'ct', 'ctt', 'n')
            **fit_kwargs: Additional arguments for model fitting
            
        Returns:
            Self for method chaining
            
        Raises:
            ModelTrainingError: If model training fails
        """
        try:
            self.logger.info(f"Fitting VAR model to {data.shape[0]} observations with {data.shape[1]} variables")
            
            if validate_input:
                self._validate_data(data)
            
            # Store training data and metadata
            self.training_data = data.copy()
            self.training_metadata = {
                'training_start': data.index.min(),
                'training_end': data.index.max(), 
                'training_observations': len(data),
                'n_variables': data.shape[1],
                'variable_names': list(data.columns),
                'maxlags': self.maxlags,
                'ic': self.ic,
                'trend': trend
            }
            
            # Create VAR model
            self.model = VAR(data)
            
            # Select optimal lag if not specified
            if self.maxlags is None:
                # Use automatic lag selection
                max_lag_test = min(12, len(data) // 4)  # Conservative default
                lag_selection = self.model.select_order(maxlags=max_lag_test)
                self.selected_lag = getattr(lag_selection, self.ic)
                self.logger.info(f"Auto-selected lag order: {self.selected_lag} using {self.ic.upper()}")
            else:
                self.selected_lag = self.maxlags
                self.logger.info(f"Using specified lag order: {self.selected_lag}")
            
            # Fit the model
            self.fitted_model = self.model.fit(
                maxlags=self.selected_lag,
                trend=trend,
                **fit_kwargs
            )
            
            # Log model summary
            self.logger.info("VAR model fitted successfully")
            self.logger.info(f"Lag order: {self.fitted_model.k_ar}")
            self.logger.info(f"AIC: {self.fitted_model.aic:.2f}")
            self.logger.info(f"BIC: {self.fitted_model.bic:.2f}")
            
            return self
            
        except Exception as e:
            self.logger.error(f"VAR model fitting failed: {e}")
            raise ModelTrainingError(f"Failed to fit VAR model: {e}")
    
    def forecast(
        self, 
        steps: int,
        alpha: float = 0.05
    ) -> Dict[str, Union[pd.DataFrame, pd.Panel]]:
        """
        Generate forecasts from fitted VAR model.
        
        Args:
            steps: Number of steps to forecast
            alpha: Alpha level for confidence intervals
            
        Returns:
            Dictionary containing forecasts and confidence intervals
            
        Raises:
            ForecastError: If forecasting fails
        """
        try:
            if self.fitted_model is None:
                raise ForecastError("VAR model must be fitted before forecasting")
            
            self.logger.info(f"Generating {steps}-step VAR forecast")
            
            # Generate forecast
            forecast_result = self.fitted_model.forecast(
                y=self.training_data.values[-self.fitted_model.k_ar:],
                steps=steps
            )
            
            # Create forecast index
            last_date = self.training_data.index[-1]
            if isinstance(last_date, pd.Timestamp):
                forecast_index = pd.date_range(
                    start=last_date + pd.infer_freq(self.training_data.index),
                    periods=steps,
                    freq=pd.infer_freq(self.training_data.index)
                )
            else:
                forecast_index = range(len(self.training_data), len(self.training_data) + steps)
            
            # Create forecast DataFrame
            forecast_df = pd.DataFrame(
                forecast_result,
                index=forecast_index,
                columns=self.training_data.columns
            )
            
            # Get forecast confidence intervals
            conf_int = self.fitted_model.forecast_interval(
                y=self.training_data.values[-self.fitted_model.k_ar:],
                steps=steps,
                alpha=alpha
            )
            
            # Create confidence interval DataFrames
            lower_bounds = pd.DataFrame(
                conf_int[:, :, 0],
                index=forecast_index,
                columns=self.training_data.columns
            )
            
            upper_bounds = pd.DataFrame(
                conf_int[:, :, 1],
                index=forecast_index,
                columns=self.training_data.columns
            )
            
            self.logger.info("VAR forecast generated successfully")
            
            return {
                'forecast': forecast_df,
                'lower_bounds': lower_bounds,
                'upper_bounds': upper_bounds,
                'confidence_level': 1 - alpha
            }
                
        except Exception as e:
            self.logger.error(f"VAR forecasting failed: {e}")
            raise ForecastError(f"Failed to generate VAR forecast: {e}")
    
    def impulse_response(
        self, 
        periods: int = 20,
        impulse: Optional[str] = None,
        response: Optional[str] = None,
        orthogonalized: bool = True
    ) -> pd.DataFrame:
        """
        Calculate impulse response functions.
        
        Args:
            periods: Number of periods for impulse response
            impulse: Variable to apply impulse to (None for all)
            response: Variable to measure response from (None for all)
            orthogonalized: Whether to use orthogonalized impulses
            
        Returns:
            DataFrame with impulse response functions
        """
        try:
            if self.fitted_model is None:
                raise ForecastError("VAR model must be fitted before impulse response analysis")
            
            irf = self.fitted_model.irf(periods=periods)
            
            if orthogonalized:
                irf_data = irf.orth_irfs
            else:
                irf_data = irf.irfs
            
            # Create MultiIndex for columns (impulse -> response)
            variables = self.training_data.columns
            columns = pd.MultiIndex.from_product(
                [variables, variables],
                names=['impulse', 'response']
            )
            
            # Reshape data for DataFrame
            n_vars = len(variables)
            reshaped_data = irf_data.reshape(periods, n_vars * n_vars)
            
            irf_df = pd.DataFrame(
                reshaped_data,
                columns=columns,
                index=range(periods)
            )
            
            # Filter by specific impulse/response if requested
            if impulse is not None and response is not None:
                return irf_df[(impulse, response)]
            elif impulse is not None:
                return irf_df[impulse]
            elif response is not None:
                return irf_df.xs(response, axis=1, level='response')
            else:
                return irf_df
                
        except Exception as e:
            self.logger.error(f"Impulse response analysis failed: {e}")
            raise ForecastError(f"Failed to calculate impulse response: {e}")
    
    def forecast_error_variance_decomposition(
        self, 
        periods: int = 20,
        normalize: bool = True
    ) -> pd.DataFrame:
        """
        Calculate forecast error variance decomposition.
        
        Args:
            periods: Number of periods for FEVD
            normalize: Whether to normalize to percentages
            
        Returns:
            DataFrame with variance decomposition
        """
        try:
            if self.fitted_model is None:
                raise ForecastError("VAR model must be fitted before FEVD analysis")
            
            fevd = self.fitted_model.fevd(periods=periods)
            
            # Create MultiIndex for columns (variable -> shock source)
            variables = self.training_data.columns
            columns = pd.MultiIndex.from_product(
                [variables, variables],
                names=['variable', 'shock_source']
            )
            
            # Reshape data
            n_vars = len(variables)
            reshaped_data = fevd.decomp.reshape(periods, n_vars * n_vars)
            
            fevd_df = pd.DataFrame(
                reshaped_data,
                columns=columns,
                index=range(1, periods + 1)
            )
            
            if normalize:
                # Convert to percentages
                for var in variables:
                    fevd_df[var] = fevd_df[var].div(fevd_df[var].sum(axis=1), axis=0) * 100
            
            return fevd_df
                
        except Exception as e:
            self.logger.error(f"FEVD analysis failed: {e}")
            raise ForecastError(f"Failed to calculate FEVD: {e}")
    
    def granger_causality(
        self, 
        caused_variable: str,
        causing_variables: Optional[List[str]] = None,
        maxlag: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform Granger causality test.
        
        Args:
            caused_variable: Variable being caused
            causing_variables: Variables potentially causing (None for all others)
            maxlag: Maximum lag to test (None uses model lag)
            
        Returns:
            Dictionary with test results
        """
        try:
            if self.fitted_model is None:
                raise ForecastError("VAR model must be fitted before Granger causality test")
            
            if causing_variables is None:
                causing_variables = [col for col in self.training_data.columns if col != caused_variable]
            
            if maxlag is None:
                maxlag = self.fitted_model.k_ar
            
            results = {}
            for causing_var in causing_variables:
                test_result = self.fitted_model.test_causality(
                    caused=caused_variable,
                    causing=causing_var,
                    kind='f'
                )
                
                results[f"{causing_var} -> {caused_variable}"] = {
                    'test_statistic': test_result.test_statistic,
                    'p_value': test_result.pvalue,
                    'critical_value': test_result.critical_value,
                    'conclusion': 'reject' if test_result.pvalue < 0.05 else 'fail_to_reject'
                }
            
            return results
                
        except Exception as e:
            self.logger.error(f"Granger causality test failed: {e}")
            raise ForecastError(f"Failed to perform Granger causality test: {e}")
    
    def cointegration_test(self, test_type: str = 'johansen') -> Dict[str, Any]:
        """
        Test for cointegration among variables.
        
        Args:
            test_type: Type of cointegration test ('johansen')
            
        Returns:
            Dictionary with test results
        """
        try:
            if self.fitted_model is None:
                raise ForecastError("VAR model must be fitted before cointegration test")
            
            if test_type == 'johansen':
                # Johansen cointegration test
                result = coint_johansen(self.training_data.values, det_order=0, k_ar_diff=1)
                
                return {
                    'test_type': 'johansen',
                    'trace_statistics': result.lr1,
                    'max_eigenvalue_statistics': result.lr2,
                    'critical_values_trace_90': result.cvt[:, 0],
                    'critical_values_trace_95': result.cvt[:, 1],
                    'critical_values_trace_99': result.cvt[:, 2],
                    'critical_values_maxeig_90': result.cvm[:, 0],
                    'critical_values_maxeig_95': result.cvm[:, 1],
                    'critical_values_maxeig_99': result.cvm[:, 2],
                    'eigenvalues': result.eig,
                    'eigenvectors': result.evec
                }
            else:
                raise ValueError(f"Unsupported test type: {test_type}")
                
        except Exception as e:
            self.logger.error(f"Cointegration test failed: {e}")
            raise ForecastError(f"Failed to perform cointegration test: {e}")
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save fitted VAR model to disk.
        
        Args:
            filepath: Path to save model
        """
        try:
            if self.fitted_model is None:
                raise ModelTrainingError("No fitted VAR model to save")
            
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Save using statsmodels built-in method
            self.fitted_model.save(str(filepath))
            
            # Also save metadata
            metadata_path = filepath.with_suffix('.metadata.pkl')
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'maxlags': self.maxlags,
                    'ic': self.ic,
                    'selected_lag': self.selected_lag,
                    'training_metadata': self.training_metadata
                }, f)
            
            self.logger.info(f"VAR model saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save VAR model: {e}")
            raise ModelTrainingError(f"Failed to save VAR model: {e}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'VARForecaster':
        """
        Load fitted VAR model from disk.
        
        Args:
            filepath: Path to saved model
            
        Returns:
            Loaded VARForecaster instance
        """
        try:
            filepath = Path(filepath)
            
            # Load the fitted model
            fitted_model = VARResults.load(str(filepath))
            
            # Load metadata if available
            metadata_path = filepath.with_suffix('.metadata.pkl')
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                maxlags = metadata.get('maxlags', None)
                ic = metadata.get('ic', 'aic')
                selected_lag = metadata.get('selected_lag', None)
                training_metadata = metadata.get('training_metadata', {})
            else:
                maxlags = None
                ic = 'aic'
                selected_lag = None
                training_metadata = {}
            
            # Create instance and populate
            instance = cls(maxlags=maxlags, ic=ic)
            instance.fitted_model = fitted_model
            instance.selected_lag = selected_lag
            instance.training_metadata = training_metadata
            
            instance.logger.info(f"VAR model loaded from {filepath}")
            
            return instance
            
        except Exception as e:
            logger = get_logger(__name__)
            logger.error(f"Failed to load VAR model: {e}")
            raise ModelTrainingError(f"Failed to load VAR model: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive VAR model information.
        
        Returns:
            Dictionary with model information
        """
        if self.fitted_model is None:
            return {'status': 'not_fitted'}
        
        info = {
            'status': 'fitted',
            'model_type': 'VAR',
            'lag_order': self.fitted_model.k_ar,
            'n_variables': len(self.fitted_model.names),
            'variable_names': self.fitted_model.names,
            'aic': self.fitted_model.aic,
            'bic': self.fitted_model.bic,
            'hqic': self.fitted_model.hqic,
            'fpe': self.fitted_model.fpe,
            'n_observations': self.fitted_model.nobs,
            'training_metadata': self.training_metadata
        }
        
        return info
    
    def _validate_data(self, data: pd.DataFrame) -> None:
        """
        Validate input multivariate data.
        
        Args:
            data: DataFrame to validate
            
        Raises:
            ModelTrainingError: If validation fails
        """
        if not isinstance(data, pd.DataFrame):
            raise ModelTrainingError("Input must be a pandas DataFrame")
        
        if data.empty:
            raise ModelTrainingError("DataFrame cannot be empty")
        
        if data.shape[1] < 2:
            raise ModelTrainingError("VAR requires at least 2 variables")
        
        if data.isnull().all().any():
            raise ModelTrainingError("Some variables are entirely NaN")
        
        if len(data) < 20:
            self.logger.warning("Data has fewer than 20 observations, model may be unreliable")
        
        if data.isnull().any().any():
            missing_info = data.isnull().sum()
            self.logger.warning(f"Data contains missing values:\n{missing_info[missing_info > 0]}")
    
    def check_stationarity(self) -> Dict[str, Dict[str, Any]]:
        """
        Check stationarity of all variables using ADF test.
        
        Returns:
            Dictionary with stationarity test results for each variable
        """
        if self.training_data is None:
            raise ModelTrainingError("No training data available for stationarity check")
        
        results = {}
        
        for col in self.training_data.columns:
            try:
                series = self.training_data[col].dropna()
                adf_result = adfuller(series, autolag='AIC')
                
                results[col] = {
                    'adf_statistic': adf_result[0],
                    'p_value': adf_result[1],
                    'critical_values': adf_result[4],
                    'is_stationary': adf_result[1] < 0.05,
                    'recommendation': 'stationary' if adf_result[1] < 0.05 else 'non-stationary (consider differencing)'
                }
                
            except Exception as e:
                results[col] = {
                    'error': str(e),
                    'is_stationary': None,
                    'recommendation': 'test_failed'
                }
        
        return results