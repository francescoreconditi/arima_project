# File creato da Claude Code per test temporaneo
# Data: 2025-08-21
# Scopo: Test semplificato della nuova funzionalità di generazione report nella dashboard
# NOTA: Questo file può essere eliminato dopo l'uso

"""
Test semplificato per verificare la funzionalità di generazione report nella dashboard.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    print("Starting Dashboard Report Generation Test")
    print("=" * 50)
    
    try:
        from arima_forecaster import ARIMAForecaster
        
        # 1. Generate sample data
        print("\n1. Generating sample data...")
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        values = 100 + np.cumsum(np.random.normal(0, 2, 100))
        ts = pd.Series(values, index=dates, name='dashboard_test_data')
        print(f"   Generated {len(ts)} data points")
        
        # 2. Train model
        print("\n2. Training ARIMA model...")
        model = ARIMAForecaster(order=(1, 1, 1))
        model.fit(ts)
        model_info = model.get_model_info()
        print(f"   Model trained - AIC: {model_info.get('aic', 0):.2f}")
        
        # 3. Generate forecast
        print("\n3. Generating forecast...")
        forecast_result = model.forecast(steps=10, confidence_intervals=True)
        
        if isinstance(forecast_result, dict):
            forecast = forecast_result['forecast']
            print(f"   Forecast generated - {len(forecast)} steps")
        else:
            forecast = forecast_result
            print(f"   Basic forecast generated - {len(forecast)} steps")
        
        # 4. Generate HTML report
        print("\n4. Generating HTML report...")
        report_path = model.generate_report(
            plots_data=None,  # Let the system handle plots
            report_title="Dashboard Test Report",
            output_filename="dashboard_test_report",
            format_type="html",
            include_diagnostics=True,
            include_forecast=True,
            forecast_steps=10
        )
        
        # Check if report was created
        if Path(report_path).exists():
            file_size = Path(report_path).stat().st_size / 1024  # KB
            print(f"   HTML report generated: {report_path}")
            print(f"   File size: {file_size:.1f} KB")
            
            print(f"\nSUCCESS: Dashboard report generation test PASSED!")
            print(f"Reports saved in: outputs/reports/")
            print(f"The dashboard Report Generation page should work correctly.")
            
            return True
        else:
            print(f"   ERROR: Report file not found at: {report_path}")
            return False
            
    except Exception as e:
        print(f"ERROR: Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    print("\n" + "=" * 50)
    if success:
        print("ALL TESTS PASSED")
        print("\nTo test in the dashboard:")
        print("1. Run: streamlit run src/arima_forecaster/dashboard/main.py")
        print("2. Upload data or use sample data")
        print("3. Train a model")
        print("4. Generate forecast (optional)")
        print("5. Go to 'Report Generation' page")
        print("6. Configure and generate report")
    else:
        print("TESTS FAILED")
        print("Check the error messages above for debugging.")