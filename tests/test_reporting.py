"""
Test suite per le funzionalità di reporting Quarto.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from arima_forecaster.core.arima_model import ARIMAForecaster
from arima_forecaster.core.sarima_model import SARIMAForecaster
from arima_forecaster.utils.exceptions import ForecastError, ModelTrainingError

# Test per import condizionale
try:
    from arima_forecaster.reporting.generator import QuartoReportGenerator
    HAS_REPORTING = True
except ImportError:
    HAS_REPORTING = False


class TestQuartoReportGenerator:
    """Test per QuartoReportGenerator class."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Crea directory temporanea per i test."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_model_results(self):
        """Risultati modello di esempio per i test."""
        return {
            'model_type': 'ARIMA',
            'order': (2, 1, 2),
            'model_info': {
                'aic': 245.67,
                'bic': 258.89,
                'n_observations': 100,
                'status': 'fitted'
            },
            'metrics': {
                'mae': 2.34,
                'rmse': 3.45,
                'mape': 5.67,
                'r2_score': 0.89
            },
            'training_data': {
                'start_date': '2020-01-01',
                'end_date': '2023-12-31',
                'observations': 100
            },
            'python_version': '1.24.0',
            'environment': 'ARIMA Forecaster Library'
        }
    
    @pytest.fixture
    def sample_plots_data(self, temp_output_dir):
        """Dati plot di esempio per i test."""
        plots_dir = temp_output_dir / "plots"
        plots_dir.mkdir()
        
        # Crea file plot fittizi
        plot1 = plots_dir / "forecast.png"
        plot2 = plots_dir / "residuals.png"
        plot1.write_text("fake plot content")
        plot2.write_text("fake plot content")
        
        return {
            'forecast': str(plot1),
            'residuals': str(plot2)
        }
    
    @pytest.mark.skipif(not HAS_REPORTING, reason="Reporting dependencies not available")
    def test_init(self, temp_output_dir):
        """Test inizializzazione QuartoReportGenerator."""
        generator = QuartoReportGenerator(output_dir=temp_output_dir)
        
        assert generator.output_dir == temp_output_dir
        assert temp_output_dir.exists()
    
    @pytest.mark.skipif(not HAS_REPORTING, reason="Reporting dependencies not available")
    def test_make_json_serializable(self, temp_output_dir):
        """Test conversione oggetti per JSON serialization."""
        generator = QuartoReportGenerator(output_dir=temp_output_dir)
        
        # Test numpy array
        np_array = np.array([1, 2, 3])
        result = generator._make_json_serializable(np_array)
        assert result == [1, 2, 3]
        
        # Test numpy scalar
        np_scalar = np.float64(3.14)
        result = generator._make_json_serializable(np_scalar)
        assert result == 3.14
        
        # Test pandas Series
        series = pd.Series([1, 2, 3])
        result = generator._make_json_serializable(series)
        assert result == [1, 2, 3]
        
        # Test nested dictionary
        nested = {
            'array': np.array([1, 2, 3]),
            'scalar': np.float64(2.5),
            'normal': "string"
        }
        result = generator._make_json_serializable(nested)
        expected = {
            'array': [1, 2, 3],
            'scalar': 2.5,
            'normal': "string"
        }
        assert result == expected
    
    @pytest.mark.skipif(not HAS_REPORTING, reason="Reporting dependencies not available")
    def test_create_quarto_document(self, temp_output_dir, sample_model_results, sample_plots_data):
        """Test creazione documento Quarto."""
        generator = QuartoReportGenerator(output_dir=temp_output_dir)
        
        report_dir = temp_output_dir / "test_report"
        report_dir.mkdir()
        
        qmd_path = generator._create_quarto_document(
            model_results=sample_model_results,
            plots_data=sample_plots_data,
            title="Test Report",
            report_dir=report_dir
        )
        
        # Verifica creazione file
        assert qmd_path.exists()
        assert qmd_path.name == "report.qmd"
        
        # Verifica contenuto
        content = qmd_path.read_text(encoding='utf-8')
        assert "Test Report" in content
        assert "ARIMA" in content
        assert "AIC" in content
        
        # Verifica creazione metadata JSON
        metadata_path = report_dir / "model_results.json"
        assert metadata_path.exists()
    
    @pytest.mark.skipif(not HAS_REPORTING, reason="Reporting dependencies not available") 
    def test_generate_qmd_content(self, temp_output_dir, sample_model_results):
        """Test generazione contenuto Quarto markdown."""
        generator = QuartoReportGenerator(output_dir=temp_output_dir)
        
        results_path = temp_output_dir / "test_results.json"
        
        content = generator._generate_qmd_content(
            title="Test Analysis",
            model_results=sample_model_results,
            plot_files={'forecast': 'forecast.png'},
            results_path=results_path
        )
        
        # Verifica elementi chiave nel contenuto
        assert "Test Analysis" in content
        assert "ARIMA" in content
        assert "Parametri del Modello" in content
        assert "Performance del Modello" in content
        assert "Raccomandazioni" in content
        assert "forecast.png" in content
    
    @pytest.mark.skipif(not HAS_REPORTING, reason="Reporting dependencies not available")
    @patch('subprocess.run')
    def test_render_report_success(self, mock_subprocess, temp_output_dir):
        """Test rendering report con successo."""
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stderr = ""
        
        generator = QuartoReportGenerator(output_dir=temp_output_dir)
        
        qmd_path = temp_output_dir / "test.qmd"
        qmd_path.write_text("# Test content")
        
        output_path = generator._render_report(qmd_path, "html", "test_output")
        
        expected_path = temp_output_dir / "test_output.html"
        assert output_path == expected_path
        
        # Verifica chiamata subprocess
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args[0][0]
        assert "quarto" in call_args
        assert "render" in call_args
        assert "--to" in call_args
        assert "html" in call_args
    
    @pytest.mark.skipif(not HAS_REPORTING, reason="Reporting dependencies not available")
    @patch('subprocess.run')
    def test_render_report_failure(self, mock_subprocess, temp_output_dir):
        """Test gestione errore rendering."""
        mock_subprocess.return_value.returncode = 1
        mock_subprocess.return_value.stderr = "Render error"
        
        generator = QuartoReportGenerator(output_dir=temp_output_dir)
        
        qmd_path = temp_output_dir / "test.qmd"
        qmd_path.write_text("# Test content")
        
        with pytest.raises(ForecastError, match="Quarto rendering failed"):
            generator._render_report(qmd_path, "html", "test_output")
    
    @pytest.mark.skipif(not HAS_REPORTING, reason="Reporting dependencies not available")
    @patch('subprocess.run')
    def test_generate_model_report_full(self, mock_subprocess, temp_output_dir, 
                                       sample_model_results, sample_plots_data):
        """Test generazione report completo."""
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stderr = ""
        
        generator = QuartoReportGenerator(output_dir=temp_output_dir)
        
        report_path = generator.generate_model_report(
            model_results=sample_model_results,
            plots_data=sample_plots_data,
            report_title="Complete Test Report",
            output_filename="complete_test",
            format_type="html"
        )
        
        # Verifica path di output
        expected_path = temp_output_dir / "complete_test.html"
        assert report_path == expected_path
        
        # Verifica creazione directory report
        report_dir = temp_output_dir / "complete_test_files"
        assert report_dir.exists()
    
    @pytest.mark.skipif(not HAS_REPORTING, reason="Reporting dependencies not available")
    def test_create_comparison_report(self, temp_output_dir):
        """Test creazione report comparativo."""
        generator = QuartoReportGenerator(output_dir=temp_output_dir)
        
        models_results = {
            'ARIMA(1,1,1)': {
                'model_type': 'ARIMA',
                'order': (1, 1, 1),
                'metrics': {'aic': 250.0, 'rmse': 3.2}
            },
            'ARIMA(2,1,2)': {
                'model_type': 'ARIMA', 
                'order': (2, 1, 2),
                'metrics': {'aic': 245.0, 'rmse': 2.8}
            }
        }
        
        with patch('subprocess.run') as mock_subprocess:
            mock_subprocess.return_value.returncode = 0
            mock_subprocess.return_value.stderr = ""
            
            report_path = generator.create_comparison_report(
                models_results=models_results,
                report_title="Model Comparison",
                output_filename="comparison_test",
                format_type="html"
            )
            
            expected_path = temp_output_dir / "comparison_test.html"
            assert report_path == expected_path


class TestARIMAForecasterReporting:
    """Test per funzionalità reporting integrate in ARIMAForecaster."""
    
    @pytest.fixture
    def sample_data(self):
        """Dati di esempio per i test."""
        dates = pd.date_range('2020-01-01', periods=50, freq='M')
        values = 100 + np.cumsum(np.random.normal(0, 5, len(dates)))
        return pd.Series(values, index=dates, name='test_data')
    
    @pytest.fixture
    def fitted_arima_model(self, sample_data):
        """Modello ARIMA addestrato per i test."""
        model = ARIMAForecaster(order=(1, 1, 1))
        model.fit(sample_data)
        return model
    
    def test_generate_report_not_fitted(self):
        """Test errore se modello non addestrato."""
        model = ARIMAForecaster(order=(1, 1, 1))
        
        with pytest.raises(ModelTrainingError, match="deve essere addestrato"):
            model.generate_report()
    
    @pytest.mark.skipif(not HAS_REPORTING, reason="Reporting dependencies not available")
    @patch('arima_forecaster.reporting.generator.QuartoReportGenerator.generate_model_report')
    def test_generate_report_success(self, mock_generate, fitted_arima_model):
        """Test generazione report con successo."""
        mock_generate.return_value = Path("/fake/report.html")
        
        report_path = fitted_arima_model.generate_report(
            report_title="Test ARIMA Report",
            format_type="html"
        )
        
        assert report_path == Path("/fake/report.html")
        mock_generate.assert_called_once()
        
        # Verifica argomenti chiamata
        call_args = mock_generate.call_args
        model_results = call_args[1]['model_results']
        
        assert model_results['model_type'] == 'ARIMA'
        assert model_results['order'] == (1, 1, 1)
        assert 'model_info' in model_results
        assert 'metrics' in model_results
    
    def test_generate_report_missing_dependencies(self, fitted_arima_model):
        """Test gestione dipendenze mancanti."""
        with patch.dict('sys.modules', {'arima_forecaster.reporting': None}):
            with pytest.raises(ForecastError, match="Moduli di reporting non disponibili"):
                fitted_arima_model.generate_report()
    
    @pytest.mark.skipif(not HAS_REPORTING, reason="Reporting dependencies not available")
    def test_generate_report_with_plots(self, fitted_arima_model):
        """Test generazione report con plot."""
        with patch('arima_forecaster.reporting.generator.QuartoReportGenerator.generate_model_report') as mock_generate:
            mock_generate.return_value = Path("/fake/report.html")
            
            plots_data = {
                'forecast': '/fake/forecast.png',
                'residuals': '/fake/residuals.png'
            }
            
            fitted_arima_model.generate_report(
                plots_data=plots_data,
                include_diagnostics=True,
                include_forecast=True,
                forecast_steps=24
            )
            
            # Verifica plots_data passati correttamente
            call_args = mock_generate.call_args
            assert call_args[1]['plots_data'] == plots_data


class TestSARIMAForecasterReporting:
    """Test per funzionalità reporting integrate in SARIMAForecaster."""
    
    @pytest.fixture
    def sample_seasonal_data(self):
        """Dati stagionali di esempio."""
        dates = pd.date_range('2020-01-01', periods=60, freq='M')
        trend = np.linspace(100, 150, len(dates))
        seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
        values = trend + seasonal + np.random.normal(0, 3, len(dates))
        return pd.Series(values, index=dates, name='seasonal_data')
    
    @pytest.fixture
    def fitted_sarima_model(self, sample_seasonal_data):
        """Modello SARIMA addestrato per i test."""
        model = SARIMAForecaster(
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12)
        )
        model.fit(sample_seasonal_data)
        return model
    
    @pytest.mark.skipif(not HAS_REPORTING, reason="Reporting dependencies not available")
    @patch('arima_forecaster.reporting.generator.QuartoReportGenerator.generate_model_report')
    def test_sarima_generate_report(self, mock_generate, fitted_sarima_model):
        """Test generazione report SARIMA."""
        mock_generate.return_value = Path("/fake/sarima_report.html")
        
        report_path = fitted_sarima_model.generate_report(
            report_title="Test SARIMA Report",
            include_seasonal_decomposition=True
        )
        
        assert report_path == Path("/fake/sarima_report.html")
        
        # Verifica argomenti specifici SARIMA
        call_args = mock_generate.call_args
        model_results = call_args[1]['model_results']
        
        assert model_results['model_type'] == 'SARIMA'
        assert model_results['order'] == (1, 1, 1)
        assert model_results['seasonal_order'] == (1, 1, 1, 12)


class TestReportingIntegration:
    """Test integrazione completa sistema reporting."""
    
    @pytest.mark.skipif(not HAS_REPORTING, reason="Reporting dependencies not available")
    def test_import_reporting_module(self):
        """Test import modulo reporting."""
        from arima_forecaster.reporting import QuartoReportGenerator
        
        generator = QuartoReportGenerator()
        assert generator is not None
    
    def test_import_without_dependencies(self):
        """Test import graceful senza dipendenze."""
        # Simula assenza dipendenze
        with patch.dict('sys.modules', {'quarto': None}):
            with pytest.raises(ImportError):
                from arima_forecaster.reporting import QuartoReportGenerator
    
    @pytest.mark.skipif(not HAS_REPORTING, reason="Reporting dependencies not available")
    def test_multiple_format_export(self, temp_dir=None):
        """Test export in formati multipli."""
        if temp_dir is None:
            temp_dir = Path(tempfile.mkdtemp())
        
        generator = QuartoReportGenerator(output_dir=temp_dir)
        
        sample_results = {
            'model_type': 'ARIMA',
            'order': (1, 1, 1),
            'metrics': {'aic': 245.67}
        }
        
        # Mock subprocess per evitare chiamate Quarto reali
        with patch('subprocess.run') as mock_subprocess:
            mock_subprocess.return_value.returncode = 0
            mock_subprocess.return_value.stderr = ""
            
            # Test HTML
            html_path = generator.generate_model_report(
                model_results=sample_results,
                output_filename="test_html",
                format_type="html"
            )
            assert html_path.suffix == ".html"
            
            # Test PDF (simulato)
            pdf_path = generator.generate_model_report(
                model_results=sample_results,
                output_filename="test_pdf", 
                format_type="pdf"
            )
            assert pdf_path.suffix == ".pdf"
        
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__])