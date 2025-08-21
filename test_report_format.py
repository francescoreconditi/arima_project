# File creato da Claude Code per test temporaneo
# Data: 2025-08-21
# Scopo: Testare il nuovo formato tabellare nei report
# NOTA: Questo file può essere eliminato dopo l'uso

import numpy as np
import pandas as pd
import sys
import os

# Force UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from arima_forecaster import ARIMAForecaster

def main():
    print("Testing report generation with new table format...")
    
    # Generate simple test data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    values = 100 + np.cumsum(np.random.normal(0, 1, 100))
    ts = pd.Series(values, index=dates, name='test_series')
    
    # Fit simple ARIMA model
    print("Fitting ARIMA model...")
    model = ARIMAForecaster(order=(1, 1, 1))
    model.fit(ts)
    
    # Generate report
    print("Generating report...")
    try:
        report_path = model.generate_report(
            report_title="Test Report - Table Format",
            output_filename="test_table_format_report",
            format_type="html",
            include_diagnostics=True,
            include_forecast=True,
            forecast_steps=10
        )
        print(f"Report generated successfully: {report_path}")
        print("\nPlease check the report to verify that:")
        print("1. The 'Configurazione Completa' section shows formatted tables")
        print("2. No raw JSON is displayed")
        print("3. Data is organized in categories with proper formatting")
    except Exception as e:
        print(f"Error generating report: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()