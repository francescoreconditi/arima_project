"""
Tests for time series preprocessing functionality.
"""

import pytest
import pandas as pd
import numpy as np

from arima_forecaster.data.preprocessor import TimeSeriesPreprocessor
from arima_forecaster.utils.exceptions import DataProcessingError


class TestTimeSeriesPreprocessor:
    """Test cases for TimeSeriesPreprocessor class."""
    
    def test_init(self):
        """Test preprocessor initialization."""
        preprocessor = TimeSeriesPreprocessor()
        assert preprocessor.preprocessing_steps == []
    
    def test_handle_missing_values_interpolate(self, preprocessor, sample_time_series_with_missing):
        """Test handling missing values by interpolation."""
        original_length = len(sample_time_series_with_missing)
        
        result = preprocessor.handle_missing_values(sample_time_series_with_missing, method='interpolate')
        
        assert len(result) == original_length
        assert not result.isnull().any()
        assert 'Missing values handled: interpolate' in preprocessor.preprocessing_steps
    
    def test_check_stationarity_stationary(self, preprocessor, sample_stationary_series):
        """Test stationarity check on stationary series."""
        result = preprocessor.check_stationarity(sample_stationary_series)
        
        assert isinstance(result, dict)
        assert 'is_stationary' in result
        assert 'p_value' in result
        assert 'adf_statistic' in result
        assert 'critical_values' in result
    
    def test_make_stationary_difference(self, preprocessor, sample_non_stationary_series):
        """Test making series stationary using differencing."""
        stationary_series, n_diff = preprocessor.make_stationary(
            sample_non_stationary_series, method='difference'
        )
        
        assert n_diff >= 1
        assert n_diff <= 2  # Should not need more than 2 differences
        assert len(stationary_series) < len(sample_non_stationary_series)
    
    def test_preprocess_pipeline_full(self, preprocessor, sample_time_series_with_missing):
        """Test complete preprocessing pipeline."""
        processed_series, metadata = preprocessor.preprocess_pipeline(
            sample_time_series_with_missing,
            handle_missing=True,
            missing_method='interpolate',
            remove_outliers_flag=True,
            outlier_method='iqr',
            make_stationary_flag=True,
            stationarity_method='difference'
        )
        
        assert isinstance(processed_series, pd.Series)
        assert isinstance(metadata, dict)
        
        # Check metadata
        assert 'original_length' in metadata
        assert 'original_missing' in metadata
        assert 'final_length' in metadata
        assert 'preprocessing_steps' in metadata
        
        # Should have no missing values
        assert not processed_series.isnull().any()
        
        # Should have applied some preprocessing
        assert len(metadata['preprocessing_steps']) > 0