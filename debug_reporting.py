#!/usr/bin/env python3
"""Debug Quarto reporting step by step."""

import pandas as pd
import numpy as np
from pathlib import Path
from arima_forecaster import ARIMAForecaster
from arima_forecaster.reporting import QuartoReportGenerator

print("=== Debug Reporting Step by Step ===")

# Generate sample data
dates = pd.date_range('2020-01-01', periods=50, freq='ME')  # Use ME instead of deprecated M
values = 100 + np.cumsum(np.random.normal(0, 3, len(dates)))
data = pd.Series(values, index=dates, name='test_data')

print(f"1. Generated {len(data)} data points")

# Train model
model = ARIMAForecaster(order=(1, 1, 1))
model.fit(data)
print("2. Model trained")

# Test QuartoReportGenerator directly
generator = QuartoReportGenerator(output_dir="outputs/reports")
print(f"3. Generator created, output dir: {generator.output_dir}")

# Prepare model results manually
model_results = {
    'model_type': 'ARIMA',
    'order': model.order,
    'model_info': model.get_model_info(),
    'metrics': {'mae': 2.5, 'rmse': 3.2, 'aic': 245.0},
    'training_data': {
        'observations': len(data),
        'start_date': str(data.index.min()),
        'end_date': str(data.index.max())
    },
    'python_version': '3.11.13',
    'environment': 'Test Environment'
}

print("4. Model results prepared")

# Create report directory
report_dir = Path("outputs/reports/debug_test_files")
report_dir.mkdir(parents=True, exist_ok=True)
print(f"5. Report directory created: {report_dir}")

# Try to create QMD document
try:
    qmd_path = generator._create_quarto_document(
        model_results=model_results,
        plots_data=None,
        title="Debug Test Report",
        report_dir=report_dir
    )
    print(f"6. QMD document created: {qmd_path}")
    
    # Check if file exists and show content preview
    if qmd_path.exists():
        file_size = qmd_path.stat().st_size
        print(f"   File exists, size: {file_size} bytes")
        
        # Show first few lines
        with open(qmd_path, 'r', encoding='utf-8') as f:
            content = f.read()[:500]
            print(f"   Content preview:\n{content}...")
    else:
        print("   ERROR: QMD file was not created!")
        
except Exception as e:
    print(f"6. ERROR creating QMD: {e}")
    import traceback
    traceback.print_exc()

print("Debug completed.")