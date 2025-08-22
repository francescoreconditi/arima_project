"""
Configurazione Pytest e fixture condivise.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Aggiungi src al path Python per i test
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from arima_forecaster import ARIMAForecaster, TimeSeriesPreprocessor


@pytest.fixture
def sample_time_series():
    """Crea una serie temporale di esempio per i test."""
    np.random.seed(42)
    
    # Crea intervallo di date
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    
    # Genera processo AR(1)
    values = np.zeros(100)
    values[0] = np.random.normal(0, 1)
    
    for i in range(1, 100):
        values[i] = 0.7 * values[i-1] + np.random.normal(0, 1)
    
    # Aggiungi trend
    trend = np.linspace(10, 20, 100)
    values += trend
    
    return pd.Series(values, index=dates, name='test_series')


@pytest.fixture
def sample_time_series_with_missing():
    """Crea una serie temporale di esempio con valori mancanti."""
    series = sample_time_series()
    
    # Introduce valori mancanti
    missing_indices = np.random.choice(len(series), size=5, replace=False)
    series.iloc[missing_indices] = np.nan
    
    return series


@pytest.fixture
def sample_stationary_series():
    """Crea una serie temporale stazionaria."""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    
    # Genera rumore bianco
    values = np.random.normal(0, 1, 100)
    
    return pd.Series(values, index=dates, name='stationary_series')


@pytest.fixture
def sample_non_stationary_series():
    """Crea una serie temporale non stazionaria (passeggiata casuale)."""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    
    # Genera passeggiata casuale
    innovations = np.random.normal(0, 1, 100)
    values = np.cumsum(innovations)
    
    return pd.Series(values, index=dates, name='non_stationary_series')


@pytest.fixture 
def sample_seasonal_series():
    """Crea una serie temporale con pattern stagionale."""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=120, freq='M')
    
    # Genera pattern stagionale
    t = np.arange(120)
    seasonal = 10 * np.sin(2 * np.pi * t / 12)  # Stagionalità annuale
    trend = 0.1 * t
    noise = np.random.normal(0, 2, 120)
    
    values = 100 + trend + seasonal + noise
    
    return pd.Series(values, index=dates, name='seasonal_series')


@pytest.fixture
def preprocessor():
    """Crea un'istanza di TimeSeriesPreprocessor."""
    return TimeSeriesPreprocessor()


@pytest.fixture
def arima_model():
    """Crea un'istanza del modello ARIMA."""
    return ARIMAForecaster(order=(1, 1, 1))


@pytest.fixture
def fitted_model(arima_model, sample_time_series):
    """Crea un modello ARIMA addestrato."""
    # Usa dati differenziati per l'addestramento
    diff_series = sample_time_series.diff().dropna()
    arima_model.fit(diff_series)
    return arima_model


@pytest.fixture
def test_data_dir():
    """Ottieni directory dei dati di test."""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def temp_dir(tmp_path):
    """Crea directory temporanea per output dei test."""
    return tmp_path