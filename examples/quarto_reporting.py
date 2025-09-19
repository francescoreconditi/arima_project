#!/usr/bin/env python3
"""
Esempio completo di utilizzo delle funzionalità di reporting Quarto
per modelli ARIMA e SARIMA.

Questo script mostra come:
1. Caricare e preprocessare i dati
2. Addestrare modelli ARIMA e SARIMA
3. Generare visualizzazioni
4. Creare report Quarto dinamici
5. Esportare in diversi formati
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Import della libreria ARIMA Forecaster
from arima_forecaster import (
    ARIMAForecaster,
    SARIMAForecaster,
    TimeSeriesPreprocessor,
    DataLoader,
    ForecastPlotter,
    ModelEvaluator,
)

# Import opzionale per reporting
try:
    from arima_forecaster.reporting import QuartoReportGenerator

    HAS_REPORTING = True
    print("[OK] Moduli di reporting Quarto disponibili")
except ImportError:
    HAS_REPORTING = False
    print(
        "[WARN] Moduli di reporting non disponibili. Installare con: pip install 'arima-forecaster[reports]'"
    )


def load_sample_data() -> pd.Series:
    """Carica o genera dati di esempio per il forecasting."""
    try:
        # Prova a caricare dati esistenti
        loader = DataLoader()
        data_path = Path("data/processed/sample_data.csv")
        if data_path.exists():
            data = loader.load_csv(data_path, date_column="date", value_column="value")
            print(f"✓ Caricati {len(data)} punti dati da {data_path}")
            return data
    except:
        pass

    # Genera dati di esempio se non disponibili
    print("Generazione dati di esempio...")
    dates = pd.date_range("2020-01-01", periods=100, freq="M")
    trend = np.linspace(100, 150, len(dates))
    seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
    noise = np.random.normal(0, 5, len(dates))
    values = trend + seasonal + noise

    data = pd.Series(values, index=dates, name="sales")
    print(f"✓ Generati {len(data)} punti dati sintetici")
    return data


def preprocess_data(data: pd.Series) -> pd.Series:
    """Preprocessing dei dati."""
    print("\n=== Preprocessing Dati ===")

    preprocessor = TimeSeriesPreprocessor()

    # Applica preprocessing pipeline
    processed_data = preprocessor.preprocess_pipeline(
        data, missing_strategy="interpolate", outlier_method="iqr", stationarity_method="auto"
    )

    print(f"✓ Dati preprocessati: {len(processed_data)} osservazioni")
    return processed_data


def train_arima_model(data: pd.Series) -> ARIMAForecaster:
    """Addestra modello ARIMA."""
    print("\n=== Addestramento Modello ARIMA ===")

    # Crea e addestra modello
    model = ARIMAForecaster(order=(2, 1, 2))
    model.fit(data)

    # Mostra informazioni modello
    info = model.get_model_info()
    print(f"✓ Modello ARIMA{model.order} addestrato")
    print(f"  - AIC: {info['aic']:.2f}")
    print(f"  - BIC: {info['bic']:.2f}")

    return model


def train_sarima_model(data: pd.Series) -> SARIMAForecaster:
    """Addestra modello SARIMA."""
    print("\n=== Addestramento Modello SARIMA ===")

    # Crea e addestra modello
    model = SARIMAForecaster(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model.fit(data)

    # Mostra informazioni modello
    info = model.get_model_info()
    print(f"✓ Modello SARIMA{model.order}x{model.seasonal_order} addestrato")
    print(f"  - AIC: {info['aic']:.2f}")
    print(f"  - BIC: {info['bic']:.2f}")

    return model


def create_visualizations(
    arima_model: ARIMAForecaster, sarima_model: SARIMAForecaster, data: pd.Series
) -> dict:
    """Crea visualizzazioni per i modelli."""
    print("\n=== Creazione Visualizzazioni ===")

    plotter = ForecastPlotter()
    plots_dir = Path("outputs/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_files = {}

    try:
        # Plot forecast ARIMA
        arima_forecast = arima_model.forecast(steps=12, confidence_intervals=True)
        forecast_data = (
            arima_forecast["forecast"] if isinstance(arima_forecast, dict) else arima_forecast
        )
        conf_int = (
            arima_forecast.get("confidence_intervals") if isinstance(arima_forecast, dict) else None
        )

        arima_plot_path = plots_dir / "arima_forecast.png"
        plotter.plot_forecast(
            actual=data,
            forecast=forecast_data,
            confidence_intervals=conf_int,
            title="Previsione ARIMA",
            save_path=str(arima_plot_path),
        )
        plot_files["arima_forecast"] = str(arima_plot_path)
        print(f"✓ Grafico previsione ARIMA salvato in {arima_plot_path}")

        # Plot forecast SARIMA
        sarima_forecast = sarima_model.forecast(steps=12, confidence_intervals=True)
        forecast_data = (
            sarima_forecast["forecast"] if isinstance(sarima_forecast, dict) else sarima_forecast
        )
        conf_int = (
            sarima_forecast.get("confidence_intervals")
            if isinstance(sarima_forecast, dict)
            else None
        )

        sarima_plot_path = plots_dir / "sarima_forecast.png"
        plotter.plot_forecast(
            actual=data,
            forecast=forecast_data,
            confidence_intervals=conf_int,
            title="Previsione SARIMA",
            save_path=str(sarima_plot_path),
        )
        plot_files["sarima_forecast"] = str(sarima_plot_path)
        print(f"✓ Grafico previsione SARIMA salvato in {sarima_plot_path}")

        # Plot diagnostici ARIMA
        arima_residuals_path = plots_dir / "arima_residuals.png"
        plotter.plot_residuals(
            residuals=arima_model.fitted_model.resid,
            title="Diagnostici Residui ARIMA",
            save_path=str(arima_residuals_path),
        )
        plot_files["arima_residuals"] = str(arima_residuals_path)
        print(f"✓ Plot residui ARIMA salvato in {arima_residuals_path}")

        # Plot diagnostici SARIMA
        sarima_residuals_path = plots_dir / "sarima_residuals.png"
        plotter.plot_residuals(
            residuals=sarima_model.fitted_model.resid,
            title="Diagnostici Residui SARIMA",
            save_path=str(sarima_residuals_path),
        )
        plot_files["sarima_residuals"] = str(sarima_residuals_path)
        print(f"✓ Plot residui SARIMA salvato in {sarima_residuals_path}")

    except Exception as e:
        print(f"⚠ Errore durante la creazione delle visualizzazioni: {e}")

    return plot_files


def generate_individual_reports(
    arima_model: ARIMAForecaster, sarima_model: SARIMAForecaster, plot_files: dict
):
    """Genera report individuali per ogni modello."""
    if not HAS_REPORTING:
        print("⚠ Reporting non disponibile - skip generazione report")
        return

    print("\n=== Generazione Report Individuali ===")

    reports_dir = Path("outputs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Report ARIMA
    try:
        arima_plots = {k: v for k, v in plot_files.items() if "arima" in k}
        arima_report_path = arima_model.generate_report(
            plots_data=arima_plots,
            report_title="Analisi Completa Modello ARIMA",
            output_filename="arima_analysis",
            format_type="html",
            include_diagnostics=True,
            include_forecast=True,
            forecast_steps=24,
        )
        print(f"✓ Report ARIMA generato: {arima_report_path}")
    except Exception as e:
        print(f"⚠ Errore generazione report ARIMA: {e}")

    # Report SARIMA
    try:
        sarima_plots = {k: v for k, v in plot_files.items() if "sarima" in k}
        sarima_report_path = sarima_model.generate_report(
            plots_data=sarima_plots,
            report_title="Analisi Completa Modello SARIMA",
            output_filename="sarima_analysis",
            format_type="html",
            include_diagnostics=True,
            include_forecast=True,
            include_seasonal_decomposition=True,
            forecast_steps=24,
        )
        print(f"✓ Report SARIMA generato: {sarima_report_path}")
    except Exception as e:
        print(f"⚠ Errore generazione report SARIMA: {e}")


def generate_comparison_report(
    arima_model: ARIMAForecaster, sarima_model: SARIMAForecaster, data: pd.Series, plot_files: dict
):
    """Genera report comparativo tra i modelli."""
    if not HAS_REPORTING:
        print("⚠ Reporting non disponibile - skip report comparativo")
        return

    print("\n=== Generazione Report Comparativo ===")

    try:
        # Raccogli risultati di entrambi i modelli
        evaluator = ModelEvaluator()

        # Metriche ARIMA
        arima_predictions = arima_model.predict()
        arima_metrics = evaluator.calculate_forecast_metrics(data, arima_predictions)
        arima_results = {
            "model_type": "ARIMA",
            "order": arima_model.order,
            "model_info": arima_model.get_model_info(),
            "metrics": arima_metrics,
            "training_data": {
                "observations": len(data),
                "start_date": str(data.index.min()),
                "end_date": str(data.index.max()),
            },
        }

        # Metriche SARIMA
        sarima_predictions = sarima_model.predict()
        sarima_metrics = evaluator.calculate_forecast_metrics(data, sarima_predictions)
        sarima_results = {
            "model_type": "SARIMA",
            "order": sarima_model.order,
            "seasonal_order": sarima_model.seasonal_order,
            "model_info": sarima_model.get_model_info(),
            "metrics": sarima_metrics,
            "training_data": {
                "observations": len(data),
                "start_date": str(data.index.min()),
                "end_date": str(data.index.max()),
            },
        }

        # Genera report comparativo
        generator = QuartoReportGenerator()
        comparison_path = generator.create_comparison_report(
            models_results={
                "ARIMA(2,1,2)": arima_results,
                "SARIMA(1,1,1)x(1,1,1,12)": sarima_results,
            },
            report_title="Confronto Modelli ARIMA vs SARIMA",
            output_filename="models_comparison",
            format_type="html",
        )

        print(f"✓ Report comparativo generato: {comparison_path}")

    except Exception as e:
        print(f"⚠ Errore generazione report comparativo: {e}")


def export_reports_multiple_formats():
    """Esempio di esportazione in formati multipli."""
    if not HAS_REPORTING:
        return

    print("\n=== Esportazione Formati Multipli ===")

    try:
        # Genera dati di esempio rapidi
        dates = pd.date_range("2020-01-01", periods=50, freq="M")
        values = 100 + np.cumsum(np.random.normal(0, 5, len(dates)))
        data = pd.Series(values, index=dates, name="example")

        # Modello veloce
        model = ARIMAForecaster(order=(1, 1, 1))
        model.fit(data)

        # Export HTML
        html_path = model.generate_report(
            report_title="Test Export HTML", output_filename="export_test_html", format_type="html"
        )
        print(f"✓ Report HTML: {html_path}")

        # Export PDF (richiede LaTeX)
        try:
            pdf_path = model.generate_report(
                report_title="Test Export PDF", output_filename="export_test_pdf", format_type="pdf"
            )
            print(f"✓ Report PDF: {pdf_path}")
        except Exception as e:
            print(f"⚠ Export PDF fallito (LaTeX richiesto): {e}")

        # Export DOCX (richiede pandoc)
        try:
            docx_path = model.generate_report(
                report_title="Test Export DOCX",
                output_filename="export_test_docx",
                format_type="docx",
            )
            print(f"✓ Report DOCX: {docx_path}")
        except Exception as e:
            print(f"⚠ Export DOCX fallito (pandoc richiesto): {e}")

    except Exception as e:
        print(f"⚠ Errore durante export multipli: {e}")


def main():
    """Funzione principale per l'esempio di reporting."""
    print("=== Esempio Reporting Quarto per Modelli ARIMA/SARIMA ===\n")

    # 1. Carica e preprocessa dati
    data = load_sample_data()
    processed_data = preprocess_data(data)

    # 2. Addestra modelli
    arima_model = train_arima_model(processed_data)
    sarima_model = train_sarima_model(processed_data)

    # 3. Crea visualizzazioni
    plot_files = create_visualizations(arima_model, sarima_model, processed_data)

    # 4. Genera report individuali
    generate_individual_reports(arima_model, sarima_model, plot_files)

    # 5. Genera report comparativo
    generate_comparison_report(arima_model, sarima_model, processed_data, plot_files)

    # 6. Test export formati multipli
    export_reports_multiple_formats()

    print("\n=== Riepilogo ===")
    print("✓ Modelli ARIMA e SARIMA addestrati con successo")
    print("✓ Visualizzazioni create")
    if HAS_REPORTING:
        print("✓ Report Quarto generati in outputs/reports/")
        print("✓ Visualizza i report aprendo i file HTML in un browser")
        print("\nNext Steps:")
        print("1. Esamina i report generati")
        print("2. Personalizza i template Quarto se necessario")
        print("3. Integra nella tua pipeline di forecasting")
    else:
        print("⚠ Installa le dipendenze di reporting per funzionalità complete")
        print("   pip install 'arima-forecaster[reports]'")


if __name__ == "__main__":
    main()
