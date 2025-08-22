"""
Test suite per il modello SARIMAX.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path

# Assicuriamoci che il package sia trovabile
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from arima_forecaster.core.sarimax_model import SARIMAXForecaster
from arima_forecaster.utils.exceptions import ModelTrainingError, ForecastError


class TestSARIMAXForecaster:
    """Test suite per SARIMAXForecaster."""
    
    @pytest.fixture
    def sample_data(self):
        """Genera dati di test con variabili esogene."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        
        # Serie target
        trend = np.arange(100) * 0.1
        seasonal = 2 * np.sin(2 * np.pi * np.arange(100) / 7)
        noise = np.random.normal(0, 0.5, 100)
        series = pd.Series(trend + seasonal + noise, index=dates, name='target')
        
        # Variabili esogene
        exog = pd.DataFrame({
            'temp': 20 + 5 * np.sin(2 * np.pi * np.arange(100) / 365) + np.random.normal(0, 1, 100),
            'marketing': 1000 + 10 * np.arange(100) + np.random.normal(0, 50, 100)
        }, index=dates)
        
        return series, exog
    
    @pytest.fixture
    def simple_forecaster(self):
        """Crea un forecaster SARIMAX base."""
        return SARIMAXForecaster(
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 7),
            exog_names=['temp', 'marketing']
        )
    
    def test_init(self):
        """Test inizializzazione modello SARIMAX."""
        forecaster = SARIMAXForecaster(
            order=(2, 1, 1),
            seasonal_order=(1, 0, 1, 12),
            exog_names=['var1', 'var2'],
            trend='c'
        )
        
        assert forecaster.order == (2, 1, 1)
        assert forecaster.seasonal_order == (1, 0, 1, 12)
        assert forecaster.exog_names == ['var1', 'var2']
        assert forecaster.trend == 'c'
        assert forecaster.fitted_model is None
        assert forecaster.training_data is None
        assert forecaster.training_exog is None
    
    def test_init_defaults(self):
        """Test inizializzazione con valori default."""
        forecaster = SARIMAXForecaster()
        
        assert forecaster.order == (1, 1, 1)
        assert forecaster.seasonal_order == (1, 1, 1, 12)
        assert forecaster.exog_names == []
        assert forecaster.trend is None
    
    def test_fit_with_exog(self, simple_forecaster, sample_data):
        """Test addestramento con variabili esogene."""
        series, exog = sample_data
        
        result = simple_forecaster.fit(series, exog=exog)
        
        # Verifica concatenamento
        assert result is simple_forecaster
        
        # Verifica stato modello
        assert simple_forecaster.fitted_model is not None
        assert simple_forecaster.training_data is not None
        assert simple_forecaster.training_exog is not None
        assert len(simple_forecaster.training_data) == len(series)
        assert simple_forecaster.training_exog.shape == exog.shape
        
        # Verifica metadati
        metadata = simple_forecaster.training_metadata
        assert metadata['training_observations'] == len(series)
        assert metadata['order'] == (1, 1, 1)
        assert metadata['seasonal_order'] == (1, 1, 1, 7)
        assert metadata['exog_names'] == ['temp', 'marketing']
        assert metadata['n_exog'] == 2
    
    def test_fit_without_exog(self):
        """Test addestramento senza variabili esogene (dovrebbe funzionare come SARIMA)."""
        forecaster = SARIMAXForecaster(order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
        
        # Serie semplice
        series = pd.Series(np.random.randn(50), name='test')
        
        result = forecaster.fit(series, exog=None)
        
        assert result is forecaster
        assert forecaster.fitted_model is not None
        assert forecaster.training_exog is None
        assert forecaster.training_metadata['n_exog'] == 0
    
    def test_fit_invalid_series(self, simple_forecaster):
        """Test addestramento con serie non valida."""
        # Serie vuota
        empty_series = pd.Series([], dtype=float)
        with pytest.raises(ModelTrainingError, match="La serie non può essere vuota"):
            simple_forecaster.fit(empty_series)
        
        # Serie con tutti NaN
        nan_series = pd.Series([np.nan, np.nan, np.nan])
        with pytest.raises(ModelTrainingError, match="La serie non può essere tutta NaN"):
            simple_forecaster.fit(nan_series)
        
        # Input non Series
        with pytest.raises(ModelTrainingError, match="L'input deve essere una pandas Series"):
            simple_forecaster.fit([1, 2, 3])
    
    def test_fit_invalid_exog(self, simple_forecaster, sample_data):
        """Test addestramento con variabili esogene non valide."""
        series, _ = sample_data
        
        # Exog con lunghezza diversa
        wrong_exog = pd.DataFrame({'temp': [1, 2, 3]})
        with pytest.raises(ModelTrainingError, match="devono avere la stessa lunghezza"):
            simple_forecaster.fit(series, exog=wrong_exog)
        
        # Exog non DataFrame
        with pytest.raises(ModelTrainingError, match="devono essere un pandas DataFrame"):
            simple_forecaster.fit(series, exog=[1, 2, 3])
        
        # Exog vuoto
        empty_exog = pd.DataFrame()
        with pytest.raises(ModelTrainingError, match="non può essere vuoto"):
            simple_forecaster.fit(series, exog=empty_exog)
    
    def test_forecast_with_exog(self, simple_forecaster, sample_data):
        """Test previsioni con variabili esogene."""
        series, exog = sample_data
        simple_forecaster.fit(series, exog=exog)
        
        # Variabili esogene future
        exog_future = pd.DataFrame({
            'temp': [22, 23, 21],
            'marketing': [1200, 1300, 1400]
        })
        
        forecast = simple_forecaster.forecast(
            steps=3,
            exog_future=exog_future
        )
        
        assert isinstance(forecast, pd.Series)
        assert len(forecast) == 3
        assert forecast.name == 'forecast'
        assert not forecast.isnull().any()
    
    def test_forecast_with_confidence_intervals(self, simple_forecaster, sample_data):
        """Test previsioni con intervalli di confidenza."""
        series, exog = sample_data
        simple_forecaster.fit(series, exog=exog)
        
        exog_future = pd.DataFrame({
            'temp': [22, 23, 21],
            'marketing': [1200, 1300, 1400]
        })
        
        forecast, conf_int = simple_forecaster.forecast(
            steps=3,
            exog_future=exog_future,
            confidence_intervals=True,
            return_conf_int=True
        )
        
        assert isinstance(forecast, pd.Series)
        assert isinstance(conf_int, pd.DataFrame)
        assert len(forecast) == len(conf_int) == 3
        assert conf_int.shape[1] == 2  # Lower e upper bounds
    
    def test_forecast_without_exog_required(self, simple_forecaster, sample_data):
        """Test previsioni quando sono richieste variabili esogene ma non fornite."""
        series, exog = sample_data
        simple_forecaster.fit(series, exog=exog)
        
        with pytest.raises(ForecastError, match="Il modello richiede variabili esogene"):
            simple_forecaster.forecast(steps=3, exog_future=None)
    
    def test_forecast_invalid_exog_future(self, simple_forecaster, sample_data):
        """Test previsioni con variabili esogene future non valide."""
        series, exog = sample_data
        simple_forecaster.fit(series, exog=exog)
        
        # Numero sbagliato di righe
        wrong_exog = pd.DataFrame({'temp': [22], 'marketing': [1200]})
        with pytest.raises(ForecastError, match="devono avere lo stesso numero di righe"):
            simple_forecaster.forecast(steps=3, exog_future=wrong_exog)
        
        # Colonne mancanti
        incomplete_exog = pd.DataFrame({'temp': [22, 23, 21]})
        with pytest.raises(ForecastError, match="Variabili mancanti"):
            simple_forecaster.forecast(steps=3, exog_future=incomplete_exog)
    
    def test_forecast_not_fitted(self, simple_forecaster):
        """Test previsioni senza modello addestrato."""
        with pytest.raises(ForecastError, match="deve essere addestrato prima della previsione"):
            simple_forecaster.forecast(steps=3)
    
    def test_predict(self, simple_forecaster, sample_data):
        """Test predizioni in-sample."""
        series, exog = sample_data
        simple_forecaster.fit(series, exog=exog)
        
        predictions = simple_forecaster.predict()
        
        assert isinstance(predictions, pd.Series)
        assert len(predictions) == len(series)
    
    def test_predict_with_exog(self, simple_forecaster, sample_data):
        """Test predizioni con variabili esogene specifiche."""
        series, exog = sample_data
        simple_forecaster.fit(series, exog=exog)
        
        # Usa subset delle variabili esogene
        exog_subset = exog.iloc[:10]
        predictions = simple_forecaster.predict(
            start=0, 
            end=9, 
            exog=exog_subset
        )
        
        assert isinstance(predictions, pd.Series)
        assert len(predictions) == 10
    
    def test_save_and_load(self, simple_forecaster, sample_data):
        """Test salvataggio e caricamento modello."""
        series, exog = sample_data
        simple_forecaster.fit(series, exog=exog)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "sarimax_model.pkl"
            
            # Salva
            simple_forecaster.save(save_path)
            assert save_path.exists()
            
            # Verifica che i file metadata e exog siano stati creati
            metadata_path = save_path.with_suffix('.metadata.pkl')
            exog_path = save_path.with_suffix('.exog.pkl')
            assert metadata_path.exists()
            assert exog_path.exists()
            
            # Carica
            loaded_forecaster = SARIMAXForecaster.load(save_path)
            
            # Verifica parametri
            assert loaded_forecaster.order == simple_forecaster.order
            assert loaded_forecaster.seasonal_order == simple_forecaster.seasonal_order
            assert loaded_forecaster.exog_names == simple_forecaster.exog_names
            assert loaded_forecaster.trend == simple_forecaster.trend
            
            # Verifica che i metadati siano caricati
            assert loaded_forecaster.training_metadata == simple_forecaster.training_metadata
            
            # Verifica che le variabili esogene siano caricate
            pd.testing.assert_frame_equal(
                loaded_forecaster.training_exog,
                simple_forecaster.training_exog
            )
    
    def test_save_not_fitted(self, simple_forecaster):
        """Test salvataggio modello non addestrato."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "model.pkl"
            with pytest.raises(ModelTrainingError, match="Nessun modello SARIMAX addestrato da salvare"):
                simple_forecaster.save(save_path)
    
    def test_get_model_info(self, simple_forecaster, sample_data):
        """Test informazioni modello."""
        # Modello non addestrato
        info = simple_forecaster.get_model_info()
        assert info == {'status': 'not_fitted'}
        
        # Modello addestrato
        series, exog = sample_data
        simple_forecaster.fit(series, exog=exog)
        
        info = simple_forecaster.get_model_info()
        assert info['status'] == 'fitted'
        assert info['model_type'] == 'SARIMAX'
        assert info['order'] == (1, 1, 1)
        assert info['seasonal_order'] == (1, 1, 1, 7)
        assert info['exog_names'] == ['temp', 'marketing']
        assert info['n_exog'] == 2
        assert 'aic' in info
        assert 'bic' in info
        assert 'params' in info
        assert 'exog_params' in info
    
    def test_get_exog_importance(self, simple_forecaster, sample_data):
        """Test analisi importanza variabili esogene."""
        series, exog = sample_data
        simple_forecaster.fit(series, exog=exog)
        
        importance = simple_forecaster.get_exog_importance()
        
        assert isinstance(importance, pd.DataFrame)
        assert len(importance) == 2  # Due variabili esogene
        assert 'variable' in importance.columns
        assert 'coefficient' in importance.columns
        assert 'pvalue' in importance.columns
        assert 'significant' in importance.columns
        
        # Verifica variabili
        assert set(importance['variable']) == {'temp', 'marketing'}
    
    def test_get_exog_importance_not_fitted(self, simple_forecaster):
        """Test analisi importanza senza modello addestrato."""
        with pytest.raises(ForecastError, match="deve essere addestrato"):
            simple_forecaster.get_exog_importance()
    
    def test_get_exog_importance_no_exog(self):
        """Test analisi importanza senza variabili esogene."""
        forecaster = SARIMAXForecaster(order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
        series = pd.Series(np.random.randn(50))
        forecaster.fit(series)
        
        importance = forecaster.get_exog_importance()
        
        assert isinstance(importance, pd.DataFrame)
        assert importance.empty
    
    def test_get_seasonal_decomposition(self, simple_forecaster, sample_data):
        """Test decomposizione stagionale."""
        series, exog = sample_data
        simple_forecaster.fit(series, exog=exog)
        
        decomposition = simple_forecaster.get_seasonal_decomposition()
        
        assert isinstance(decomposition, dict)
        expected_keys = {'observed', 'trend', 'seasonal', 'residual'}
        assert set(decomposition.keys()) == expected_keys
        
        for component in decomposition.values():
            assert isinstance(component, pd.Series)
    
    def test_get_seasonal_decomposition_not_fitted(self, simple_forecaster):
        """Test decomposizione senza modello addestrato."""
        with pytest.raises(ForecastError, match="deve essere addestrato"):
            simple_forecaster.get_seasonal_decomposition()
    
    @patch('arima_forecaster.core.sarimax_model.SARIMAX')
    def test_fit_training_failure(self, mock_sarimax, simple_forecaster, sample_data):
        """Test gestione errori durante addestramento."""
        series, exog = sample_data
        
        # Simula errore in statsmodels
        mock_sarimax.side_effect = Exception("Statsmodels error")
        
        with pytest.raises(ModelTrainingError, match="Impossibile addestrare il modello SARIMAX"):
            simple_forecaster.fit(series, exog=exog)
    
    def test_seasonal_parameter_validation(self, sample_data):
        """Test validazione parametri stagionali."""
        series, exog = sample_data
        
        # Periodo stagionale <= 1
        forecaster = SARIMAXForecaster(
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 1)  # Periodo = 1
        )
        with pytest.raises(ModelTrainingError, match="deve essere maggiore di 1"):
            forecaster.fit(series, exog=exog)
        
        # Serie troppo corta per periodo stagionale
        short_series = series[:5]
        short_exog = exog[:5]
        forecaster = SARIMAXForecaster(
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 20)  # Periodo = 20 > lunghezza serie
        )
        # Dovrebbe solo dare un warning, non errore
        forecaster.fit(short_series, exog=short_exog)
    
    def test_generate_report_functionality(self, simple_forecaster, sample_data):
        """Test funzionalità di base della generazione report (senza dependencies)."""
        series, exog = sample_data
        simple_forecaster.fit(series, exog=exog)
        
        # Test che il metodo esista e abbia i parametri corretti
        assert hasattr(simple_forecaster, 'generate_report')
        
        # Test con modulo non disponibile (dovrebbe dare ImportError)
        with patch('arima_forecaster.core.sarimax_model.QuartoReportGenerator', side_effect=ImportError):
            with pytest.raises(ForecastError, match="Moduli di reporting non disponibili"):
                simple_forecaster.generate_report()


class TestSARIMAXIntegration:
    """Test di integrazione per SARIMAX."""
    
    def test_complete_workflow(self):
        """Test workflow completo SARIMAX."""
        # Genera dati realistici
        np.random.seed(123)
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        
        # Variabili esogene
        temperature = 20 + 10 * np.sin(2 * np.pi * np.arange(200) / 365) + np.random.normal(0, 2, 200)
        marketing = 1000 + np.random.normal(0, 200, 200)
        exog = pd.DataFrame({
            'temperature': temperature,
            'marketing': marketing
        }, index=dates)
        
        # Serie target influenzata dalle esogene
        target = (1000 + 0.5 * temperature + 0.01 * marketing + 
                 50 * np.sin(2 * np.pi * np.arange(200) / 7) + 
                 np.random.normal(0, 20, 200))
        series = pd.Series(target, index=dates, name='sales')
        
        # Split train/test
        train_size = 150
        series_train = series[:train_size]
        series_test = series[train_size:]
        exog_train = exog[:train_size]
        exog_test = exog[train_size:]
        
        # 1. Addestramento
        model = SARIMAXForecaster(
            order=(1, 1, 1),
            seasonal_order=(1, 0, 1, 7),
            exog_names=['temperature', 'marketing']
        )
        model.fit(series_train, exog=exog_train)
        
        # 2. Valutazione in-sample
        in_sample_pred = model.predict()
        assert len(in_sample_pred) == len(series_train)
        
        # 3. Previsioni out-of-sample
        forecast_steps = 20
        exog_future = exog_test[:forecast_steps]
        forecast = model.forecast(forecast_steps, exog_future=exog_future)
        assert len(forecast) == forecast_steps
        
        # 4. Analisi importanza
        importance = model.get_exog_importance()
        assert len(importance) == 2
        assert set(importance['variable']) == {'temperature', 'marketing'}
        
        # 5. Informazioni modello
        info = model.get_model_info()
        assert info['status'] == 'fitted'
        assert info['n_exog'] == 2
        assert 'aic' in info
        
        # 6. Decomposizione stagionale
        decomposition = model.get_seasonal_decomposition()
        assert 'seasonal' in decomposition
        
        print(f"✅ Test integrazione SARIMAX completato")
        print(f"   Dati: {len(series)} osservazioni, {len(exog.columns)} variabili esogene")
        print(f"   Modello: SARIMAX{model.order}x{model.seasonal_order}")
        print(f"   AIC: {info['aic']:.2f}")
        print(f"   Previsioni generate: {len(forecast)}")


if __name__ == "__main__":
    # Esegui test base se chiamato direttamente
    pytest.main([__file__, "-v"])