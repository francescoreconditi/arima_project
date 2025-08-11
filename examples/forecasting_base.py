"""
Esempio di forecasting ARIMA base.

Questo esempio dimostra:
1. Caricamento e preprocessing di dati serie temporali
2. Addestramento di un modello ARIMA
3. Generazione previsioni
4. Visualizzazione risultati

Esecuzione:
    uv run python examples/forecasting_base.py
    oppure: python examples/forecasting_base.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Windows
import matplotlib.pyplot as plt
from pathlib import Path

# Aggiungi src al path per gli import
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arima_forecaster import ARIMAForecaster, TimeSeriesPreprocessor, ForecastPlotter
from arima_forecaster.data import DataLoader
from arima_forecaster.utils import setup_logger

def generate_sample_data():
    """Generate sample time series data for demonstration."""
    np.random.seed(42)
    
    # Create date range
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='M')
    
    # Generate trend component
    trend = np.linspace(100, 200, len(dates))
    
    # Generate seasonal component (annual seasonality)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
    
    # Generate noise
    noise = np.random.normal(0, 10, len(dates))
    
    # Combine components
    values = trend + seasonal + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'sales': values
    }).set_index('date')
    
    return df

def main():
    """Main forecasting example."""
    
    # Setup logging
    logger = setup_logger('basic_forecasting', level='INFO')
    logger.info("Starting basic forecasting example")
    
    try:
        # Step 1: Generate or load data
        logger.info("Generating sample data...")
        df = generate_sample_data()
        
        # Save sample data for future use
        output_dir = Path(__file__).parent.parent / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / "sample_sales_data.csv")
        logger.info(f"Sample data saved to {output_dir / 'sample_sales_data.csv'}")
        
        # Extract the time series
        series = df['sales']
        logger.info(f"Loaded time series with {len(series)} observations")
        logger.info(f"Date range: {series.index.min()} to {series.index.max()}")
        
        # Step 2: Data preprocessing
        logger.info("Preprocessing data...")
        preprocessor = TimeSeriesPreprocessor()
        
        # Check stationarity
        stationarity_test = preprocessor.check_stationarity(series)
        logger.info(f"Original series is stationary: {stationarity_test['is_stationary']} (p-value: {stationarity_test['p_value']:.4f})")
        
        # Apply preprocessing pipeline
        processed_series, metadata = preprocessor.preprocess_pipeline(
            series,
            handle_missing=True,
            missing_method='interpolate',
            remove_outliers_flag=False,  # Keep all data for this example
            make_stationary_flag=True,
            stationarity_method='difference'
        )
        
        logger.info(f"Preprocessing complete: {len(series)} -> {len(processed_series)} observations")
        logger.info(f"Preprocessing steps: {metadata['preprocessing_steps']}")
        
        # Step 3: Split data for evaluation
        train_size = int(len(processed_series) * 0.8)
        train_data = processed_series[:train_size]
        test_data = processed_series[train_size:]
        
        logger.info(f"Train set: {len(train_data)} observations")
        logger.info(f"Test set: {len(test_data)} observations")
        
        # Step 4: Train ARIMA model
        logger.info("Training ARIMA model...")
        model = ARIMAForecaster(order=(2, 1, 1))  # ARIMA(2,1,1)
        model.fit(train_data)
        
        # Get model information
        model_info = model.get_model_info()
        logger.info(f"Model trained successfully")
        logger.info(f"AIC: {model_info['aic']:.2f}")
        logger.info(f"BIC: {model_info['bic']:.2f}")
        
        # Step 5: Generate forecasts
        logger.info("Generating forecasts...")
        
        # In-sample forecast (fitted values)
        fitted_values = model.predict(start=0, end=len(train_data)-1)
        
        # Out-of-sample forecast
        forecast_steps = len(test_data)
        forecast, conf_int = model.forecast(
            steps=forecast_steps,
            confidence_intervals=True,
            return_conf_int=True
        )
        
        logger.info(f"Generated {len(forecast)} forecasts")
        
        # Step 6: Evaluate model performance
        from arima_forecaster.evaluation import ModelEvaluator
        
        evaluator = ModelEvaluator()
        
        # Calculate forecast metrics
        metrics = evaluator.calculate_forecast_metrics(test_data, forecast)
        logger.info("Forecast Performance Metrics:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {metric.upper()}: {value:.4f}")
        
        # Residual analysis
        residuals = model.fitted_model.resid
        residual_diagnostics = evaluator.evaluate_residuals(residuals)
        
        logger.info("Residual Diagnostics:")
        jb_test = residual_diagnostics['jarque_bera_test']
        logger.info(f"  Jarque-Bera test (normality): p-value = {jb_test['p_value']:.4f}")
        
        lb_test = residual_diagnostics['ljung_box_test']
        logger.info(f"  Ljung-Box test (autocorrelation): p-value = {lb_test['p_value']:.4f}")
        
        # Step 7: Visualization
        logger.info("Creating visualizations...")
        
        plotter = ForecastPlotter()
        
        # Create output directory for plots
        plots_dir = Path(__file__).parent.parent / "outputs" / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot 1: Forecast with confidence intervals
        fig1 = plotter.plot_forecast(
            actual=train_data,
            forecast=forecast,
            confidence_intervals=conf_int,
            title="ARIMA Forecast with 95% Confidence Intervals",
            save_path=plots_dir / "forecast_with_ci.png"
        )
        
        # Plot 2: Residual analysis
        fig2 = plotter.plot_residuals(
            residuals=residuals,
            fitted_values=fitted_values,
            title="Residual Analysis",
            save_path=plots_dir / "residual_analysis.png"
        )
        
        # Plot 3: ACF/PACF of original series
        fig3 = plotter.plot_acf_pacf(
            series=series,
            lags=24,
            title="ACF/PACF of Original Series",
            save_path=plots_dir / "acf_pacf_original.png"
        )
        
        # Plot 4: Complete dashboard
        fig4 = plotter.create_dashboard(
            actual=train_data,
            forecast=forecast,
            residuals=residuals,
            confidence_intervals=conf_int,
            metrics=metrics,
            title="Forecasting Dashboard",
            save_path=plots_dir / "dashboard.png"
        )
        
        logger.info(f"Plots saved to {plots_dir}")
        
        # Step 8: Save model
        model_dir = Path(__file__).parent.parent / "outputs" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / "basic_arima_model.pkl"
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Step 9: Generate future forecasts
        logger.info("Generating future forecasts...")
        
        # Retrain on all available data
        full_model = ARIMAForecaster(order=(2, 1, 1))
        full_model.fit(processed_series)
        
        # Generate 12-month ahead forecast
        future_forecast, future_conf_int = full_model.forecast(
            steps=12,
            confidence_intervals=True,
            return_conf_int=True
        )
        
        # Plot future forecast
        fig5 = plotter.plot_forecast(
            actual=processed_series.tail(36),  # Show last 3 years + forecast
            forecast=future_forecast,
            confidence_intervals=future_conf_int,
            title="12-Month Future Forecast",
            save_path=plots_dir / "future_forecast.png"
        )
        
        logger.info("Future forecast:")
        for i, (f, lower, upper) in enumerate(zip(future_forecast, 
                                                  future_conf_int.iloc[:, 0], 
                                                  future_conf_int.iloc[:, 1])):
            logger.info(f"  Month {i+1}: {f:.2f} [{lower:.2f}, {upper:.2f}]")
        
        logger.info("Basic forecasting example completed successfully!")
        
        # Display plots (optional - comment out if running headless)
        # plt.show()  # Disabled for Windows compatibility
    print("Plot saved as 'outputs/plots/forecasting_base.png'")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise

if __name__ == "__main__":
    main()