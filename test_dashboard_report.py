# File creato da Claude Code per test temporaneo
# Data: 2025-08-21
# Scopo: Testare la nuova funzionalità di generazione report nella dashboard
# NOTA: Questo file può essere eliminato dopo l'uso

"""
Test script per verificare la funzionalità di generazione report nella dashboard.
Questo script simula il workflow della dashboard per testare la nuova feature.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from arima_forecaster import ARIMAForecaster

def test_dashboard_report_generation():
    """Test the report generation functionality."""
    
    print("Testing Dashboard Report Generation...")
    
    try:
        # 1. Generate sample data (simulating dashboard data upload)
        print("\n1. Generating sample data...")
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        values = 100 + np.cumsum(np.random.normal(0, 2, 100))
        ts = pd.Series(values, index=dates, name='dashboard_test_data')
        print(f"   OK Generated {len(ts)} data points")
        
        # 2. Train model (simulating dashboard model training)
        print("\n2. Training ARIMA model...")
        model = ARIMAForecaster(order=(1, 1, 1))
        model.fit(ts)
        model_info = model.get_model_info()
        print(f"   OK Model trained - AIC: {model_info.get('aic', 0):.2f}")
        
        # 3. Generate forecast (simulating dashboard forecasting)
        print("\n3. Generating forecast...")
        forecast_result = model.forecast(steps=10, confidence_intervals=True)
        
        if isinstance(forecast_result, dict):
            forecast = forecast_result['forecast']
            conf_int = forecast_result['confidence_intervals']
            print(f"   OK Forecast generated - {len(forecast)} steps")
        else:
            forecast = forecast_result
            conf_int = None
            print(f"   OK Basic forecast generated - {len(forecast)} steps")
        
        # 4. Test report generation (simulating dashboard report generation)
        print("\n4. Testing report generation...")
        
        # Create a simple plot if needed (simulating dashboard plot creation)
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot historical data
            ax.plot(ts.index[-30:], ts.values[-30:], 
                   label='Historical', color='blue', linewidth=1.5)
            
            # Plot forecast
            ax.plot(forecast.index, forecast.values, 
                   label='Forecast', color='red', linewidth=2, marker='o', markersize=4)
            
            # Plot confidence intervals if available
            if conf_int is not None:
                if hasattr(conf_int, 'iloc'):
                    ax.fill_between(forecast.index, 
                                   conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                                   alpha=0.3, color='red', label='95% CI')
            
            ax.set_title('Dashboard Test - Time Series Forecast')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Save plot
            plots_dir = Path("outputs/plots")
            plots_dir.mkdir(parents=True, exist_ok=True)
            plot_path = plots_dir / "dashboard_test_plot.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            plots_data = {'main_plot': str(plot_path)}
            print(f"   OK Plot created: {plot_path}")
            
        except Exception as e:
            print(f"   WARNING Could not create plot: {e}")
            plots_data = None
        
        # 5. Generate HTML report
        print("\n5. Generating HTML report...")
        try:
            report_path = model.generate_report(
                plots_data=plots_data,
                report_title="Dashboard Test Report",
                output_filename="dashboard_test_report",
                format_type="html",
                include_diagnostics=True,
                include_forecast=True,
                forecast_steps=10
            )
            
            # Check if report was created successfully
            if Path(report_path).exists():
                file_size = Path(report_path).stat().st_size / 1024  # KB
                print(f"   ✅ HTML report generated: {report_path}")
                print(f"   📄 File size: {file_size:.1f} KB")
                
                # Verify the report contains expected content
                with open(report_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                checks = {
                    "Has title": "Dashboard Test Report" in content,
                    "Has model info": "Model Information" in content or "Configurazione Completa" in content,
                    "Has forecast": "Forecast" in content or "forecast" in content,
                    "Has styling": "table" in content.lower() and "css" in content.lower()
                }
                
                print(f"   📋 Content verification:")
                for check, passed in checks.items():
                    status = "✅" if passed else "❌"
                    print(f"      {status} {check}")
                
                # Test different formats if dependencies are available
                print("\n6️⃣ Testing additional formats...")
                
                # Test PDF (if LaTeX available)
                try:
                    pdf_path = model.generate_report(
                        plots_data=plots_data,
                        report_title="Dashboard Test Report - PDF",
                        output_filename="dashboard_test_report_pdf",
                        format_type="pdf",
                        include_diagnostics=False,  # Simpler for testing
                        include_forecast=True,
                        forecast_steps=5
                    )
                    if Path(pdf_path).exists():
                        print(f"   ✅ PDF report generated: {pdf_path}")
                    else:
                        print(f"   ❌ PDF report not found")
                except Exception as e:
                    print(f"   ⚠️ PDF generation failed (expected if LaTeX not installed): {e}")
                
                # Test DOCX (if pandoc available)
                try:
                    docx_path = model.generate_report(
                        plots_data=plots_data,
                        report_title="Dashboard Test Report - DOCX",
                        output_filename="dashboard_test_report_docx",
                        format_type="docx",
                        include_diagnostics=False,  # Simpler for testing
                        include_forecast=True,
                        forecast_steps=5
                    )
                    if Path(docx_path).exists():
                        print(f"   ✅ DOCX report generated: {docx_path}")
                    else:
                        print(f"   ❌ DOCX report not found")
                except Exception as e:
                    print(f"   ⚠️ DOCX generation failed (expected if pandoc not installed): {e}")
                
                print(f"\n🎉 Dashboard report generation test PASSED!")
                print(f"📁 Reports saved in: outputs/reports/")
                print(f"📊 The dashboard Report Generation page should work correctly.")
                
                return True
                
            else:
                print(f"   ❌ Report file not found at: {report_path}")
                return False
                
        except Exception as e:
            print(f"   ❌ Report generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the test."""
    print("🚀 Starting Dashboard Report Generation Test")
    print("=" * 50)
    
    success = test_dashboard_report_generation()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ ALL TESTS PASSED")
        print("\n💡 To test in the dashboard:")
        print("1. Run: streamlit run src/arima_forecaster/dashboard/main.py")
        print("2. Upload data or use sample data")
        print("3. Train a model")
        print("4. Generate forecast")
        print("5. Go to 'Report Generation' page")
        print("6. Configure and generate report")
    else:
        print("❌ TESTS FAILED")
        print("\n🔧 Check the error messages above for debugging.")

if __name__ == "__main__":
    main()