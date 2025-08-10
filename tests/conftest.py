"""
Pytest configuration and shared fixtures.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to Python path for testing
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from arima_forecaster import ARIMAForecaster, TimeSeriesPreprocessor


@pytest.fixture
def sample_time_series():
    """Create a sample time series for testing."""
    np.random.seed(42)
    
    # Create date range
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    
    # Generate AR(1) process
    values = np.zeros(100)
    values[0] = np.random.normal(0, 1)
    
    for i in range(1, 100):
        values[i] = 0.7 * values[i-1] + np.random.normal(0, 1)
    
    # Add trend
    trend = np.linspace(10, 20, 100)
    values += trend
    
    return pd.Series(values, index=dates, name='test_series')


@pytest.fixture
def sample_time_series_with_missing():
    """Create a sample time series with missing values."""
    series = sample_time_series()
    
    # Introduce missing values
    missing_indices = np.random.choice(len(series), size=5, replace=False)
    series.iloc[missing_indices] = np.nan
    
    return series


@pytest.fixture
def sample_stationary_series():
    """Create a stationary time series."""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    
    # Generate white noise
    values = np.random.normal(0, 1, 100)
    
    return pd.Series(values, index=dates, name='stationary_series')


@pytest.fixture
def sample_non_stationary_series():
    """Create a non-stationary time series (random walk)."""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    
    # Generate random walk
    innovations = np.random.normal(0, 1, 100)
    values = np.cumsum(innovations)
    
    return pd.Series(values, index=dates, name='non_stationary_series')


@pytest.fixture 
def sample_seasonal_series():
    """Create a time series with seasonal pattern."""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=120, freq='M')
    
    # Generate seasonal pattern
    t = np.arange(120)
    seasonal = 10 * np.sin(2 * np.pi * t / 12)  # Annual seasonality
    trend = 0.1 * t
    noise = np.random.normal(0, 2, 120)
    
    values = 100 + trend + seasonal + noise
    
    return pd.Series(values, index=dates, name='seasonal_series')


@pytest.fixture
def preprocessor():
    """Create a TimeSeriesPreprocessor instance."""
    return TimeSeriesPreprocessor()


@pytest.fixture
def arima_model():
    """Create an ARIMA model instance."""
    return ARIMAForecaster(order=(1, 1, 1))


@pytest.fixture
def fitted_model(arima_model, sample_time_series):
    """Create a fitted ARIMA model."""
    # Use differenced data for fitting
    diff_series = sample_time_series.diff().dropna()
    arima_model.fit(diff_series)
    return arima_model


@pytest.fixture
def test_data_dir():
    """Get test data directory."""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary directory for test outputs."""
    return tmp_path