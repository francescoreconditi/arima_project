"""
Test per la funzionalità di preprocessing delle serie temporali.
"""

import pytest
import pandas as pd
import numpy as np

from arima_forecaster.data.preprocessor import TimeSeriesPreprocessor
from arima_forecaster.utils.exceptions import DataProcessingError


class TestTimeSeriesPreprocessor:
    """Casi di test per la classe TimeSeriesPreprocessor."""

    def test_init(self):
        """Test dell'inizializzazione del preprocessor."""
        preprocessor = TimeSeriesPreprocessor()
        assert preprocessor.preprocessing_steps == []

    def test_handle_missing_values_interpolate(self, preprocessor, sample_time_series_with_missing):
        """Test della gestione dei valori mancanti tramite interpolazione."""
        original_length = len(sample_time_series_with_missing)

        result = preprocessor.handle_missing_values(
            sample_time_series_with_missing, method="interpolate"
        )

        assert len(result) == original_length
        assert not result.isnull().any()
        assert "Missing values handled: interpolate" in preprocessor.preprocessing_steps

    def test_check_stationarity_stationary(self, preprocessor, sample_stationary_series):
        """Test del controllo di stazionarietà su serie stazionaria."""
        result = preprocessor.check_stationarity(sample_stationary_series)

        assert isinstance(result, dict)
        assert "is_stationary" in result
        assert "p_value" in result
        assert "adf_statistic" in result
        assert "critical_values" in result

    def test_make_stationary_difference(self, preprocessor, sample_non_stationary_series):
        """Test del rendere stazionaria una serie usando differenziazione."""
        stationary_series, n_diff = preprocessor.make_stationary(
            sample_non_stationary_series, method="difference"
        )

        assert n_diff >= 1
        assert n_diff <= 2  # Non dovrebbe richiedere più di 2 differenziazioni
        assert len(stationary_series) < len(sample_non_stationary_series)

    def test_preprocess_pipeline_full(self, preprocessor, sample_time_series_with_missing):
        """Test della pipeline completa di preprocessing."""
        processed_series, metadata = preprocessor.preprocess_pipeline(
            sample_time_series_with_missing,
            handle_missing=True,
            missing_method="interpolate",
            remove_outliers_flag=True,
            outlier_method="iqr",
            make_stationary_flag=True,
            stationarity_method="difference",
        )

        assert isinstance(processed_series, pd.Series)
        assert isinstance(metadata, dict)

        # Verifica metadata
        assert "original_length" in metadata
        assert "original_missing" in metadata
        assert "final_length" in metadata
        assert "preprocessing_steps" in metadata

        # Non dovrebbe avere valori mancanti
        assert not processed_series.isnull().any()

        # Dovrebbe aver applicato del preprocessing
        assert len(metadata["preprocessing_steps"]) > 0
