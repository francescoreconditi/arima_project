"""
Time series preprocessing utilities.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any, Union
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from ..utils.logger import get_logger
from ..utils.exceptions import DataProcessingError


class TimeSeriesPreprocessor:
    """
    Comprehensive time series preprocessing utilities.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.preprocessing_steps = []
    
    def handle_missing_values(
        self, 
        series: pd.Series, 
        method: str = 'interpolate'
    ) -> pd.Series:
        """
        Handle missing values in time series.
        
        Args:
            series: Input time series
            method: Method to handle missing values ('drop', 'interpolate', 'forward_fill', 'backward_fill')
            
        Returns:
            Series with missing values handled
        """
        self.logger.info(f"Gestione valori mancanti usando metodo: {method}")
        
        if method == 'drop':
            result = series.dropna()
        elif method == 'interpolate':
            result = series.interpolate(method='time')
        elif method == 'forward_fill':
            result = series.fillna(method='ffill')
        elif method == 'backward_fill':
            result = series.fillna(method='bfill')
        else:
            raise DataProcessingError(f"Metodo valori mancanti sconosciuto: {method}")
        
        self.preprocessing_steps.append(f"Missing values handled: {method}")
        self.logger.info(f"Valori mancanti gestiti: {series.isnull().sum()} -> {result.isnull().sum()}")
        
        return result
    
    def remove_outliers(
        self, 
        series: pd.Series, 
        method: str = 'iqr',
        threshold: float = 3.0
    ) -> pd.Series:
        """
        Remove outliers from time series.
        
        Args:
            series: Input time series
            method: Method for outlier detection ('iqr', 'zscore', 'modified_zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            Series with outliers removed
        """
        self.logger.info(f"Rimozione outliers usando metodo: {method}")
        
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            mask = (series >= lower_bound) & (series <= upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(series.dropna()))
            mask = z_scores < threshold
            
        elif method == 'modified_zscore':
            median = series.median()
            mad = np.median(np.abs(series - median))
            modified_z_scores = 0.6745 * (series - median) / mad
            mask = np.abs(modified_z_scores) < threshold
            
        else:
            raise DataProcessingError(f"Metodo rilevamento outlier sconosciuto: {method}")
        
        outliers_removed = (~mask).sum()
        result = series[mask]
        
        self.preprocessing_steps.append(f"Outliers removed: {outliers_removed} using {method}")
        self.logger.info(f"Outliers rimossi: {outliers_removed}")
        
        return result
    
    def check_stationarity(
        self, 
        series: pd.Series,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Check if time series is stationary using Augmented Dickey-Fuller test.
        
        Args:
            series: Input time series
            alpha: Significance level
            
        Returns:
            Dictionary with stationarity test results
        """
        self.logger.info("Verifica stazionarietà usando test ADF")
        
        try:
            # Remove any infinite or NaN values for the test
            clean_series = series.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(clean_series) < 10:
                raise DataProcessingError("Dati insufficienti per test di stazionarietà")
            
            adf_result = adfuller(clean_series)
            
            results = {
                'is_stationary': adf_result[1] < alpha,
                'p_value': adf_result[1],
                'adf_statistic': adf_result[0],
                'critical_values': adf_result[4],
                'used_lag': adf_result[2],
                'n_observations': adf_result[3]
            }
            
            status = "stazionaria" if results['is_stationary'] else "non-stazionaria"
            self.logger.info(f"Serie è {status} (p-value: {results['p_value']:.4f})")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Errore nel test di stazionarietà: {e}")
            raise DataProcessingError(f"Test di stazionarietà fallito: {e}")
    
    def make_stationary(
        self, 
        series: pd.Series,
        method: str = 'difference',
        max_diff: int = 2
    ) -> Tuple[pd.Series, int]:
        """
        Transform series to make it stationary.
        
        Args:
            series: Input time series
            method: Method for making stationary ('difference', 'log_difference')
            max_diff: Maximum number of differences to apply
            
        Returns:
            Tuple of (stationary series, number of differences applied)
        """
        self.logger.info(f"Rendendo serie stazionaria usando metodo: {method}")
        
        current_series = series.copy()
        n_diff = 0
        
        for i in range(max_diff + 1):
            stationarity_test = self.check_stationarity(current_series)
            
            if stationarity_test['is_stationary']:
                break
            
            if i == max_diff:
                self.logger.warning(f"Serie ancora non stazionaria dopo {max_diff} differenze")
                break
            
            if method == 'difference':
                current_series = current_series.diff().dropna()
            elif method == 'log_difference':
                if (current_series <= 0).any():
                    self.logger.warning("Impossibile applicare trasformazione logaritmica a valori non positivi, uso differenza regolare")
                    current_series = current_series.diff().dropna()
                else:
                    current_series = np.log(current_series).diff().dropna()
            else:
                raise DataProcessingError(f"Metodo di stazionarietà sconosciuto: {method}")
            
            n_diff += 1
            
        self.preprocessing_steps.append(f"Made stationary: {n_diff} differences applied")
        self.logger.info(f"Applicate {n_diff} differenze per ottenere stazionarietà")
        
        return current_series, n_diff
    
    def preprocess_pipeline(
        self,
        series: pd.Series,
        handle_missing: bool = True,
        missing_method: str = 'interpolate',
        remove_outliers_flag: bool = False,
        outlier_method: str = 'iqr',
        make_stationary_flag: bool = True,
        stationarity_method: str = 'difference'
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Complete preprocessing pipeline.
        
        Args:
            series: Input time series
            handle_missing: Whether to handle missing values
            missing_method: Method for handling missing values
            remove_outliers_flag: Whether to remove outliers
            outlier_method: Method for outlier removal
            make_stationary_flag: Whether to make series stationary
            stationarity_method: Method for making stationary
            
        Returns:
            Tuple of (processed series, preprocessing metadata)
        """
        self.logger.info("Avvio pipeline preprocessing")
        self.preprocessing_steps = []
        
        result = series.copy()
        metadata = {
            'original_length': len(series),
            'original_missing': series.isnull().sum()
        }
        
        # Handle missing values
        if handle_missing and result.isnull().any():
            result = self.handle_missing_values(result, missing_method)
        
        # Remove outliers
        if remove_outliers_flag:
            original_length = len(result)
            result = self.remove_outliers(result, outlier_method)
            metadata['outliers_removed'] = original_length - len(result)
        
        # Make stationary
        if make_stationary_flag:
            result, n_diff = self.make_stationary(result, stationarity_method)
            metadata['differencing_order'] = n_diff
        
        metadata['final_length'] = len(result)
        metadata['preprocessing_steps'] = self.preprocessing_steps.copy()
        
        self.logger.info(f"Preprocessing completato: {metadata['original_length']} -> {metadata['final_length']} osservazioni")
        
        return result, metadata