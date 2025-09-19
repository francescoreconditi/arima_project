# ============================================
# FILE DI TEST CLI
# Creato da: Claude Code
# Data: 15 Settembre 2025
# Scopo: Test per CLI module
# ============================================

"""
Test per CLI module.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from io import StringIO
import argparse
from click.testing import CliRunner

from arima_forecaster.cli import (
    main,
    setup_cli_logger,
    load_config,
    train_command,
    forecast_command,
    evaluate_command,
    version_command,
)
from arima_forecaster import __version__


class TestCLIBasics:
    """Test funzionalità base CLI."""

    def test_setup_cli_logger(self):
        """Test setup logger CLI."""
        setup_cli_logger(verbose=False)
        setup_cli_logger(verbose=True)

    def test_load_config_no_file(self):
        """Test caricamento config senza file."""
        config = load_config()
        assert config == {}

    def test_load_config_nonexistent_file(self):
        """Test caricamento config file inesistente."""
        config = load_config("nonexistent.json")
        assert config == {}

    def test_load_config_valid_file(self):
        """Test caricamento config valido."""
        test_config = {
            "arima": {"order": [2, 1, 2]},
            "sarima": {"order": [1, 1, 1], "seasonal_order": [1, 1, 1, 12]},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_config, f)
            config_path = f.name

        try:
            config = load_config(config_path)
            assert config == test_config
        finally:
            Path(config_path).unlink()


class TestVersionCommand:
    """Test comando version."""

    @patch("arima_forecaster.cli.console.print")
    def test_version_command(self, mock_print):
        """Test comando version."""
        args = Mock()
        version_command(args)
        mock_print.assert_called_once_with(f"[green]ARIMA Forecaster v{__version__}[/green]")


class TestTrainCommand:
    """Test comando train."""

    @patch("arima_forecaster.cli.DataLoader")
    @patch("arima_forecaster.cli.TimeSeriesPreprocessor")
    @patch("arima_forecaster.cli.ARIMAForecaster")
    @patch("arima_forecaster.cli.console")
    def test_train_command_arima_success(
        self, mock_console, mock_arima, mock_preprocessor, mock_loader
    ):
        """Test training ARIMA con successo."""
        # Setup mocks
        mock_data = Mock()
        mock_loader.return_value.load_data.return_value = mock_data
        mock_preprocessor.return_value.preprocess_pipeline.return_value = mock_data
        mock_model = Mock()
        mock_arima.return_value = mock_model

        # Setup args
        args = Mock()
        args.data = "test.csv"
        args.model = "arima"
        args.output = "model.pkl"
        args.preprocess = True
        args.evaluate = False
        args.config = None
        args.verbose = False

        # Test
        train_command(args)

        # Verifica chiamate
        mock_loader.assert_called_once()
        mock_loader.return_value.load_data.assert_called_once_with("test.csv")
        mock_preprocessor.assert_called_once()
        mock_arima.assert_called_once()
        mock_model.fit.assert_called_once_with(mock_data)
        mock_model.save_model.assert_called_once()

    @patch("arima_forecaster.cli.DataLoader")
    @patch("arima_forecaster.cli.SARIMAForecaster")
    @patch("arima_forecaster.cli.console")
    def test_train_command_sarima_success(self, mock_console, mock_sarima, mock_loader):
        """Test training SARIMA con successo."""
        # Setup mocks
        mock_data = Mock()
        mock_loader.return_value.load_data.return_value = mock_data
        mock_model = Mock()
        mock_sarima.return_value = mock_model

        # Setup args
        args = Mock()
        args.data = "test.csv"
        args.model = "sarima"
        args.output = "model.pkl"
        args.preprocess = False
        args.evaluate = False
        args.config = None
        args.verbose = False

        # Test
        train_command(args)

        # Verifica chiamate
        mock_sarima.assert_called_once()
        mock_model.fit.assert_called_once_with(mock_data)

    @patch("arima_forecaster.cli.console")
    def test_train_command_unsupported_model(self, mock_console):
        """Test training con modello non supportato."""
        args = Mock()
        args.model = "unsupported"
        args.verbose = False

        train_command(args)

        # Verifica messaggio errore
        mock_console.print.assert_called_with("[red]❌ Modello unsupported non supportato[/red]")


class TestForecastCommand:
    """Test comando forecast."""

    @patch("arima_forecaster.cli.ARIMAForecaster")
    @patch("arima_forecaster.cli.console")
    @patch("pandas.Series")
    def test_forecast_command_success(self, mock_series, mock_console, mock_arima):
        """Test forecasting con successo."""
        # Setup mocks
        mock_model = Mock()
        mock_forecast = [1, 2, 3, 4, 5]
        mock_model.forecast.return_value = mock_forecast
        mock_arima.load_model.return_value = mock_model

        # Setup args
        args = Mock()
        args.model = "arima"
        args.model_path = "model.pkl"
        args.steps = 5
        args.output = None
        args.verbose = False

        # Test
        forecast_command(args)

        # Verifica chiamate
        mock_arima.load_model.assert_called_once_with("model.pkl")
        mock_model.forecast.assert_called_once_with(steps=5)

    @patch("arima_forecaster.cli.console")
    def test_forecast_command_unsupported_model(self, mock_console):
        """Test forecasting con modello non supportato."""
        args = Mock()
        args.model = "unsupported"
        args.verbose = False

        forecast_command(args)

        # Verifica messaggio errore
        mock_console.print.assert_called_with("[red]❌ Modello unsupported non supportato[/red]")


class TestEvaluateCommand:
    """Test comando evaluate."""

    @patch("arima_forecaster.cli.DataLoader")
    @patch("arima_forecaster.cli.ARIMAForecaster")
    @patch("arima_forecaster.cli.ModelEvaluator")
    @patch("arima_forecaster.cli.console")
    def test_evaluate_command_success(self, mock_console, mock_evaluator, mock_arima, mock_loader):
        """Test valutazione con successo."""
        # Setup mocks
        mock_data = Mock()
        mock_loader.return_value.load_data.return_value = mock_data
        mock_model = Mock()
        mock_arima.load_model.return_value = mock_model
        mock_metrics = {"mae": 0.5, "rmse": 0.7}
        mock_evaluator.return_value.evaluate_model.return_value = mock_metrics

        # Setup args
        args = Mock()
        args.data = "test.csv"
        args.model = "arima"
        args.model_path = "model.pkl"
        args.output = None
        args.verbose = False

        # Test
        evaluate_command(args)

        # Verifica chiamate
        mock_loader.assert_called_once()
        mock_arima.load_model.assert_called_once_with("model.pkl")
        mock_evaluator.assert_called_once()


class TestCLIIntegration:
    """Test integrazione CLI completa."""

    @patch("sys.argv", ["arima-forecast", "version"])
    @patch("arima_forecaster.cli.console.print")
    def test_main_version(self, mock_print):
        """Test main con comando version."""
        main()
        mock_print.assert_called_with(f"[green]ARIMA Forecaster v{__version__}[/green]")

    @patch("sys.argv", ["arima-forecast"])
    @patch("argparse.ArgumentParser.print_help")
    def test_main_no_command(self, mock_help):
        """Test main senza comando."""
        main()
        mock_help.assert_called_once()

    @patch("sys.argv", ["arima-forecast", "--help"])
    def test_main_help(self):
        """Test main con help."""
        with pytest.raises(SystemExit):
            main()


class TestCLIErrorHandling:
    """Test gestione errori CLI."""

    @patch("arima_forecaster.cli.DataLoader")
    @patch("arima_forecaster.cli.console")
    @patch("sys.exit")
    def test_train_command_exception(self, mock_exit, mock_console, mock_loader):
        """Test gestione eccezione in train."""
        # Setup exception
        mock_loader.side_effect = Exception("Test error")

        args = Mock()
        args.data = "test.csv"
        args.model = "arima"
        args.verbose = False

        train_command(args)

        # Verifica gestione errore
        mock_console.print.assert_called()
        mock_exit.assert_called_once_with(1)

    @patch("arima_forecaster.cli.ARIMAForecaster")
    @patch("arima_forecaster.cli.console")
    @patch("sys.exit")
    def test_forecast_command_exception(self, mock_exit, mock_console, mock_arima):
        """Test gestione eccezione in forecast."""
        # Setup exception
        mock_arima.load_model.side_effect = Exception("Test error")

        args = Mock()
        args.model = "arima"
        args.model_path = "model.pkl"
        args.verbose = False

        forecast_command(args)

        # Verifica gestione errore
        mock_console.print.assert_called()
        mock_exit.assert_called_once_with(1)


class TestCLIOutputFormats:
    """Test formati output CLI."""

    @patch("arima_forecaster.cli.ARIMAForecaster")
    @patch("arima_forecaster.cli.console")
    @patch("pandas.Series")
    def test_forecast_csv_output(self, mock_series, mock_console, mock_arima):
        """Test output forecast in CSV."""
        # Setup mocks
        mock_model = Mock()
        mock_forecast = Mock()
        mock_forecast.to_csv = Mock()
        mock_model.forecast.return_value = mock_forecast
        mock_arima.load_model.return_value = mock_model

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            output_path = f.name

        try:
            args = Mock()
            args.model = "arima"
            args.model_path = "model.pkl"
            args.steps = 5
            args.output = output_path
            args.verbose = False

            forecast_command(args)

            # Verifica salvataggio
            mock_forecast.to_csv.assert_called_once()

        finally:
            Path(output_path).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__])
