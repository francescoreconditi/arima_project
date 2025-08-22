"""
Test per la funzionalità core del modello ARIMA.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from arima_forecaster.core.arima_model import ARIMAForecaster
from arima_forecaster.utils.exceptions import ModelTrainingError, ForecastError


class TestARIMAForecaster:
    """Casi di test per la classe ARIMAForecaster."""
    
    def test_init(self):
        """Test dell'inizializzazione del modello ARIMA."""
        model = ARIMAForecaster(order=(2, 1, 2))
        assert model.order == (2, 1, 2)
        assert model.model is None
        assert model.fitted_model is None
        assert model.training_data is None
    
    def test_init_default_order(self):
        """Test dell'inizializzazione con ordine predefinito."""
        model = ARIMAForecaster()
        assert model.order == (1, 1, 1)
    
    def test_fit_valid_series(self, arima_model, sample_stationary_series):
        """Test dell'addestramento del modello con serie valida."""
        arima_model.fit(sample_stationary_series)
        
        assert arima_model.fitted_model is not None
        assert arima_model.training_data is not None
        assert len(arima_model.training_data) == len(sample_stationary_series)
        assert 'training_start' in arima_model.training_metadata
        assert 'training_end' in arima_model.training_metadata
        assert 'order' in arima_model.training_metadata
    
    def test_fit_with_validation(self, arima_model, sample_stationary_series):
        """Test dell'addestramento con validazione dell'input."""
        arima_model.fit(sample_stationary_series, validate_input=True)
        assert arima_model.fitted_model is not None
    
    def test_fit_invalid_input_type(self, arima_model):
        """Test dell'addestramento con tipo di input non valido."""
        with pytest.raises(ModelTrainingError, match="Input must be a pandas Series"):
            arima_model.fit([1, 2, 3, 4, 5])
    
    def test_fit_empty_series(self, arima_model):
        """Test dell'addestramento con serie vuota."""
        empty_series = pd.Series([], dtype=float)
        with pytest.raises(ModelTrainingError, match="Series cannot be empty"):
            arima_model.fit(empty_series)
    
    def test_fit_all_nan_series(self, arima_model):
        """Test dell'addestramento con serie contenente solo NaN."""
        nan_series = pd.Series([np.nan] * 10)
        with pytest.raises(ModelTrainingError, match="Series cannot be all NaN"):
            arima_model.fit(nan_series)
    
    def test_forecast_unfitted_model(self, arima_model):
        """Test del forecasting con modello non addestrato."""
        with pytest.raises(ForecastError, match="Model must be fitted before forecasting"):
            arima_model.forecast(steps=5)
    
    def test_forecast_basic(self, fitted_model):
        """Test della funzionalità di forecasting di base."""
        forecast = fitted_model.forecast(steps=5)
        
        assert isinstance(forecast, pd.Series)
        assert len(forecast) == 5
        assert forecast.name == 'forecast'
        assert not forecast.isnull().any()
    
    def test_forecast_with_confidence_intervals(self, fitted_model):
        """Test del forecasting con intervalli di confidenza."""
        forecast, conf_int = fitted_model.forecast(
            steps=5,
            confidence_intervals=True,
            return_conf_int=True
        )
        
        assert isinstance(forecast, pd.Series)
        assert isinstance(conf_int, pd.DataFrame)
        assert len(forecast) == len(conf_int) == 5
        assert conf_int.shape[1] == 2  # Limiti inferiore e superiore
    
    def test_forecast_different_alpha(self, fitted_model):
        """Test del forecasting con diversi livelli di confidenza."""
        _, conf_int_95 = fitted_model.forecast(
            steps=3, alpha=0.05, return_conf_int=True
        )
        _, conf_int_90 = fitted_model.forecast(
            steps=3, alpha=0.10, return_conf_int=True
        )
        
        # IC 95% dovrebbe essere più ampio di IC 90%
        width_95 = (conf_int_95.iloc[:, 1] - conf_int_95.iloc[:, 0]).mean()
        width_90 = (conf_int_90.iloc[:, 1] - conf_int_90.iloc[:, 0]).mean()
        assert width_95 > width_90
    
    def test_predict_unfitted_model(self, arima_model):
        """Test della predizione con modello non addestrato."""
        with pytest.raises(ForecastError, match="Model must be fitted before prediction"):
            arima_model.predict(start=0, end=5)
    
    def test_predict_basic(self, fitted_model):
        """Test della funzionalità di predizione di base."""
        predictions = fitted_model.predict(start=0, end=4)
        
        assert isinstance(predictions, pd.Series)
        assert len(predictions) == 5
    
    def test_save_unfitted_model(self, arima_model, temp_dir):
        """Test del salvataggio di modello non addestrato."""
        model_path = temp_dir / "test_model.pkl"
        
        with pytest.raises(ModelTrainingError, match="No fitted model to save"):
            arima_model.save(model_path)
    
    def test_save_and_load_model(self, fitted_model, temp_dir):
        """Test del salvataggio e caricamento del modello."""
        model_path = temp_dir / "test_model.pkl"
        
        # Salva modello
        fitted_model.save(model_path)
        assert model_path.exists()
        assert (model_path.parent / (model_path.stem + ".metadata.pkl")).exists()
        
        # Carica modello
        loaded_model = ARIMAForecaster.load(model_path)
        
        assert loaded_model.order == fitted_model.order
        assert loaded_model.fitted_model is not None
        assert loaded_model.training_metadata == fitted_model.training_metadata
    
    def test_load_nonexistent_model(self):
        """Test del caricamento di modello inesistente."""
        with pytest.raises(ModelTrainingError, match="Failed to load model"):
            ARIMAForecaster.load("nonexistent_model.pkl")
    
    def test_get_model_info_unfitted(self, arima_model):
        """Test dell'ottenimento di informazioni da modello non addestrato."""
        info = arima_model.get_model_info()
        assert info == {'status': 'not_fitted'}
    
    def test_get_model_info_fitted(self, fitted_model):
        """Test dell'ottenimento di informazioni da modello addestrato."""
        info = fitted_model.get_model_info()
        
        assert info['status'] == 'fitted'
        assert info['order'] == fitted_model.order
        assert 'aic' in info
        assert 'bic' in info
        assert 'hqic' in info
        assert 'llf' in info
        assert 'n_observations' in info
        assert 'params' in info
        assert 'training_metadata' in info
    
    def test_fit_with_short_series_warning(self, arima_model, caplog):
        """Test che l'addestramento con serie corta produce avviso."""
        short_series = pd.Series([1, 2, 3, 4, 5])  # Solo 5 osservazioni
        
        arima_model.fit(short_series, validate_input=True)
        
        # Verifica se l'avviso è stato registrato
        assert "fewer than 10 observations" in caplog.text
    
    def test_fit_with_missing_values_warning(self, arima_model, sample_time_series_with_missing, caplog):
        """Test che l'addestramento con valori mancanti produce avviso."""
        arima_model.fit(sample_time_series_with_missing, validate_input=True)
        
        # Verifica se l'avviso è stato registrato
        assert "missing values" in caplog.text
    
    def test_forecast_index_creation_datetime(self, arima_model):
        """Test della creazione indice forecast con indice datetime."""
        # Crea serie con indice datetime
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        series = pd.Series(np.random.randn(50), index=dates)
        
        arima_model.fit(series.diff().dropna())
        forecast = arima_model.forecast(steps=5)
        
        # Verifica che il forecast abbia un indice datetime appropriato
        assert isinstance(forecast.index, pd.DatetimeIndex)
        assert len(forecast.index) == 5
    
    def test_forecast_index_creation_numeric(self, arima_model):
        """Test della creazione indice forecast con indice numerico."""
        # Crea serie con indice numerico
        series = pd.Series(np.random.randn(50), index=range(50))
        
        arima_model.fit(series.diff().dropna())
        forecast = arima_model.forecast(steps=5)
        
        # Verifica che il forecast abbia un indice numerico appropriato
        assert len(forecast.index) == 5