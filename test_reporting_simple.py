#!/usr/bin/env python3
"""Simple test of Quarto reporting system."""

import pandas as pd
import numpy as np
from pathlib import Path

# Set matplotlib backend to avoid tkinter issues
import matplotlib
matplotlib.use('Agg')

print("=== Test Sistema Reporting Completo ===")

# 1. Generate synthetic monthly sales data
dates = pd.date_range('2020-01-01', periods=36, freq='ME')
trend = np.linspace(100, 130, len(dates))
seasonal = 15 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
noise = np.random.normal(0, 5, len(dates))
values = trend + seasonal + noise
sales_data = pd.Series(values, index=dates, name='vendite')

print(f"1. Dati generati: {len(sales_data)} punti mensili")
print(f"   Periodo: {sales_data.index.min().strftime('%Y-%m')} - {sales_data.index.max().strftime('%Y-%m')}")

# 2. Train ARIMA model
from arima_forecaster import ARIMAForecaster
model = ARIMAForecaster(order=(1, 1, 1))
model.fit(sales_data)
print("2. Modello ARIMA addestrato")

# 3. Create some visualizations
from arima_forecaster.visualization import ForecastPlotter
plotter = ForecastPlotter()

# Generate forecasts
forecast_result = model.forecast(steps=12, confidence_intervals=True)
print("3. Forecast generato per 12 mesi")

# Create plots
plots_dir = Path("outputs/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
plots_dir.mkdir(parents=True, exist_ok=True)

plots_data = {}

try:
    # Forecast plot - handle different forecast result formats
    if isinstance(forecast_result, dict) and 'forecast' in forecast_result:
        forecast_series = forecast_result['forecast']
        confidence_intervals = forecast_result.get('confidence_intervals')
        
        # Convert confidence intervals format if needed
        if confidence_intervals and isinstance(confidence_intervals, dict):
            # Convert to DataFrame format expected by plotter
            import pandas as pd
            conf_df = pd.DataFrame({
                'lower': confidence_intervals['lower'],
                'upper': confidence_intervals['upper']
            })
            confidence_intervals = conf_df
    else:
        forecast_series = forecast_result
        confidence_intervals = None
    
    # Forecast plot
    forecast_path = plots_dir / "test_forecast.png"
    plotter.plot_forecast(
        actual=sales_data,
        forecast=forecast_series,
        confidence_intervals=confidence_intervals,
        title="Forecast Vendite - Test",
        save_path=str(forecast_path)
    )
    plots_data['forecast'] = str(forecast_path)
    print(f"4. Plot forecast creato: {forecast_path}")

    # Residuals plot
    residuals_path = plots_dir / "test_residuals.png"
    plotter.plot_residuals(
        residuals=model.fitted_model.resid,
        title="Diagnostici Residui - Test",
        save_path=str(residuals_path)
    )
    plots_data['residuals'] = str(residuals_path)
    print(f"5. Plot residui creato: {residuals_path}")

except Exception as e:
    print(f"Errore creazione plot: {e}")
    import traceback
    traceback.print_exc()
    plots_data = {}

# 4. Generate Quarto report
print("\n--- Generazione Report Quarto ---")

try:
    report_path = model.generate_report(
        plots_data=plots_data,
        report_title="Test Report Sistema Completo",
        output_filename="test_sistema_completo",
        format_type="html",
        include_diagnostics=True,
        include_forecast=True,
        forecast_steps=12
    )
    
    print(f"6. Report HTML generato: {report_path}")
    
    if Path(report_path).exists():
        file_size = Path(report_path).stat().st_size
        print(f"   Dimensione file: {file_size:,} bytes")
        print(f"   Apri in browser: file:///{report_path.as_posix()}")
    else:
        print("   ERRORE: File report non trovato!")
        
except Exception as e:
    print(f"ERRORE generazione report: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Test Completato ===")
print("Se tutto è andato bene, dovresti avere:")
print("- outputs/plots/test_forecast.png")
print("- outputs/plots/test_residuals.png") 
print("- outputs/reports/test_sistema_completo.html")
print("\nApri il file HTML nel browser per vedere il report completo!")