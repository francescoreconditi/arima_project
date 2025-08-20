#!/usr/bin/env python3
"""Quick test of Quarto reporting functionality."""

import pandas as pd
import numpy as np
from pathlib import Path

print("=== Test Reporting Quarto ===")

# Generate sample data
dates = pd.date_range('2020-01-01', periods=100, freq='M')
trend = np.linspace(100, 150, len(dates))
seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
noise = np.random.normal(0, 5, len(dates))
values = trend + seasonal + noise
data = pd.Series(values, index=dates, name='sales')

print(f"Generated {len(data)} data points")

# Test ARIMA model
from arima_forecaster import ARIMAForecaster
model = ARIMAForecaster(order=(1, 1, 1))
model.fit(data)
print("ARIMA model trained successfully")

# Test reporting functionality
try:
    report_path = model.generate_report(
        report_title="Test Quarto Report",
        output_filename="test_arima_report",
        format_type="html",
        include_diagnostics=True,
        include_forecast=True,
        forecast_steps=12
    )
    print(f"Report generated successfully: {report_path}")
    
    if Path(report_path).exists():
        print(f"Report file exists and size: {Path(report_path).stat().st_size} bytes")
    else:
        print("Warning: Report path returned but file not found")
        
except ImportError as e:
    print(f"Reporting not available: {e}")
except Exception as e:
    print(f"Error generating report: {e}")
    import traceback
    traceback.print_exc()

print("Test completed.")