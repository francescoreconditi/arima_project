"""
Esempio selezione automatica modello ARIMA.

Questo esempio dimostra:
1. Caricamento e preprocessing dati
2. Selezione automatica modello usando grid search
3. Confronto modelli e valutazione
4. Visualizzazione avanzata

Esecuzione:
    uv run python examples/selezione_automatica.py
    oppure: python examples/selezione_automatica.py
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
from arima_forecaster.core import ARIMAModelSelector
from arima_forecaster.evaluation import ModelEvaluator
from arima_forecaster.utils import setup_logger
from utils import get_plots_path, get_models_path, get_reports_path

def generate_complex_data():
    """Generate more complex time series data with multiple patterns."""
    np.random.seed(123)
    
    # Create date range (5 years of monthly data)
    dates = pd.date_range(start='2019-01-01', end='2023-12-31', freq='M')
    
    # Generate multiple components
    n = len(dates)
    t = np.arange(n)
    
    # Trend component (with change point)
    trend = np.where(t < n//2, 100 + 2*t, 100 + 2*(n//2) + 1*(t - n//2))
    
    # Seasonal component (annual)
    seasonal_annual = 30 * np.sin(2 * np.pi * t / 12)
    
    # Cyclical component (longer cycle)
    cyclical = 15 * np.sin(2 * np.pi * t / 36)  # 3-year cycle
    
    # AR(1) component for additional autocorrelation
    ar_component = np.zeros(n)
    ar_component[0] = np.random.normal(0, 5)
    for i in range(1, n):
        ar_component[i] = 0.6 * ar_component[i-1] + np.random.normal(0, 5)
    
    # Random walk component
    random_walk = np.cumsum(np.random.normal(0, 2, n))
    
    # Combine all components
    values = trend + seasonal_annual + cyclical + ar_component + random_walk
    
    # Add some outliers
    outlier_indices = np.random.choice(n, size=3, replace=False)
    values[outlier_indices] += np.random.choice([-50, 50], size=3)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'value': values
    }).set_index('date')
    
    return df

def main():
    """Main automatic model selection example."""
    
    # Setup logging
    logger = setup_logger('auto_model_selection', level='INFO')
    logger.info("Starting automatic model selection example")
    
    try:
        # Step 1: Generate complex data
        logger.info("Generating complex time series data...")
        df = generate_complex_data()
        
        # Save data
        output_dir = Path(__file__).parent.parent / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / "complex_sample_data.csv")
        
        series = df['value']
        logger.info(f"Generated series with {len(series)} observations")
        
        # Step 2: Data exploration and preprocessing
        logger.info("Exploring and preprocessing data...")
        
        preprocessor = TimeSeriesPreprocessor()
        
        # Initial exploration
        logger.info(f"Series statistics:")
        logger.info(f"  Mean: {series.mean():.2f}")
        logger.info(f"  Std: {series.std():.2f}")
        logger.info(f"  Min: {series.min():.2f}")
        logger.info(f"  Max: {series.max():.2f}")
        logger.info(f"  Missing values: {series.isnull().sum()}")
        
        # Check for outliers
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)).sum()
        logger.info(f"  Outliers (IQR method): {outliers}")
        
        # Preprocessing with outlier removal
        processed_series, metadata = preprocessor.preprocess_pipeline(
            series,
            handle_missing=True,
            missing_method='interpolate',
            remove_outliers_flag=True,
            outlier_method='iqr',
            make_stationary_flag=True,
            stationarity_method='difference'
        )
        
        logger.info(f"Preprocessing applied: {metadata['preprocessing_steps']}")
        logger.info(f"Final series length: {len(processed_series)}")
        if 'outliers_removed' in metadata:
            logger.info(f"Outliers removed: {metadata['outliers_removed']}")
        if 'differencing_order' in metadata:
            logger.info(f"Differencing applied: {metadata['differencing_order']}")
        
        # Step 3: Train/test split
        train_size = int(len(processed_series) * 0.8)
        train_data = processed_series[:train_size]
        test_data = processed_series[train_size:]
        
        logger.info(f"Train set: {len(train_data)} observations")
        logger.info(f"Test set: {len(test_data)} observations")
        
        # Step 4: Automatic model selection
        logger.info("Starting automatic model selection...")
        
        try:
            # Create model selector
            selector = ARIMAModelSelector(
                p_range=(0, 3),
                d_range=(0, 2), 
                q_range=(0, 3),
                information_criterion='aic'
            )
            
            # Perform grid search
            best_order = selector.search(train_data, verbose=True, max_models=30)
            logger.info(f"Best model found: ARIMA{best_order}")
            
        except Exception as e:
            logger.warning(f"Automatic model selection failed: {e}")
            logger.info("Using fallback model ARIMA(1,1,1)")
            best_order = (1, 1, 1)
            
            # Create a dummy selector result for consistency
            selector = None
        
        # Get detailed results
        if selector is not None:
            results_df = selector.get_results_summary(top_n=10)
            logger.info("Top 10 models by AIC:")
            logger.info(results_df.to_string())
            
            best_info = selector.get_best_model_info()
            logger.info(f"Best model details:")
            logger.info(f"  AIC: {best_info['aic']:.2f}")
            logger.info(f"  BIC: {best_info['bic']:.2f}")
            logger.info(f"  HQIC: {best_info['hqic']:.2f}")
            logger.info(f"  Parameters: {best_info['parameters']}")
        else:
            logger.info("Using fallback model, no detailed comparison available")
        
        # Step 5: Compare multiple models
        logger.info("Training and comparing top models...")
        
        evaluator = ModelEvaluator()
        model_comparison = {}
        
        if selector is not None:
            # Test top 3 models
            top_models = results_df.head(3)
            
            for idx, row in top_models.iterrows():
                order = row['order']
                logger.info(f"Evaluating ARIMA{order}...")
                
                try:
                    # Train model
                    model = ARIMAForecaster(order=order)
                    model.fit(train_data)
                    
                    # Generate forecasts
                    forecast = model.forecast(steps=len(test_data))
                    
                    # Calculate metrics
                    metrics = evaluator.calculate_forecast_metrics(test_data, forecast)
                    
                    # Store results
                    model_name = f"ARIMA{order}"
                    model_comparison[model_name] = {
                        'aic': row['aic'],
                        'bic': row['bic'],
                        'rmse': metrics['rmse'],
                        'mae': metrics['mae'],
                        'mape': metrics['mape'],
                        'r_squared': metrics['r_squared']
                    }
                    
                    logger.info(f"  RMSE: {metrics['rmse']:.4f}")
                    logger.info(f"  MAE: {metrics['mae']:.4f}")
                    logger.info(f"  MAPE: {metrics['mape']:.2f}%")
                    
                except Exception as e:
                    logger.warning(f"Failed to evaluate ARIMA{order}: {e}")
        else:
            logger.info("Skipping model comparison due to fallback mode")
        
        # Step 6: Train final model with best parameters
        logger.info("Training final model...")
        
        final_model = ARIMAForecaster(order=best_order)
        final_model.fit(train_data)
        
        # Generate forecasts
        forecast, conf_int = final_model.forecast(
            steps=len(test_data),
            confidence_intervals=True,
            return_conf_int=True
        )
        
        # Step 7: Comprehensive evaluation
        logger.info("Performing comprehensive evaluation...")
        
        # Get residuals
        residuals = final_model.fitted_model.resid
        fitted_values = final_model.predict(start=0, end=len(train_data)-1)
        
        # Generate evaluation report
        final_metrics = evaluator.calculate_forecast_metrics(test_data, forecast)
        evaluation_report = evaluator.generate_evaluation_report(
            actual=test_data,
            predicted=forecast,
            residuals=residuals,
            model_info=final_model.get_model_info()
        )
        
        logger.info("Final Model Performance:")
        for metric, value in final_metrics.items():
            if isinstance(value, float):
                logger.info(f"  {metric.upper()}: {value:.4f}")
        
        logger.info("Model Interpretation:")
        for metric, interpretation in evaluation_report['interpretation'].items():
            logger.info(f"  {metric}: {interpretation}")
        
        # Step 8: Advanced visualizations
        logger.info("Creating advanced visualizations...")
        
        plotter = ForecastPlotter()
        plots_dir = Path(__file__).parent.parent / "outputs" / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot 1: Model selection results
        if selector is not None:
            try:
                selector.plot_selection_results(figsize=(15, 10))
                plt.savefig(plots_dir / "model_selection_results.png", dpi=300, bbox_inches='tight')
                logger.info("Model selection plots created")
            except Exception as e:
                logger.warning(f"Could not create selection plots: {e}")
        else:
            logger.info("Skipping selection plots due to fallback mode")
        
        # Plot 2: Model comparison
        if model_comparison:
            fig2 = plotter.plot_model_comparison(
                results=model_comparison,
                metric='rmse',
                title='Model Comparison by RMSE',
                save_path=plots_dir / "model_comparison_rmse.png"
            )
        else:
            logger.info("Skipping model comparison plot due to empty results")
        
        # Plot 3: Comprehensive dashboard
        fig3 = plotter.create_dashboard(
            actual=train_data,
            forecast=forecast,
            residuals=residuals,
            confidence_intervals=conf_int,
            metrics=final_metrics,
            title=f"Best Model Dashboard - ARIMA{best_order}",
            save_path=plots_dir / "best_model_dashboard.png"
        )
        
        # Plot 4: Original vs preprocessed data comparison
        fig4, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Original data
        axes[0].plot(series.index, series.values, color='blue', linewidth=2)
        axes[0].set_title('Original Time Series')
        axes[0].set_ylabel('Value')
        axes[0].grid(True, alpha=0.3)
        
        # Preprocessed data
        axes[1].plot(processed_series.index, processed_series.values, color='red', linewidth=2)
        axes[1].set_title('Preprocessed Time Series (Stationary)')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Value')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "data_preprocessing_comparison.png", dpi=300, bbox_inches='tight')
        
        # Step 9: Save results
        logger.info("Saving results...")
        
        # Save final model
        model_dir = Path(__file__).parent.parent / "outputs" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        final_model.save(model_dir / f"best_arima_model_{best_order}.pkl")
        

    # Genera report Quarto
    logger.info("Generazione report Quarto...")
    try:
        # Get the plot filename if it exists
        plot_files = {}
        # Try to find the most recent plot file
        if 'plot_path' in locals():
            plot_files['main_plot'] = str(plot_path)
        elif 'plt' in locals():
            # If we have a matplotlib figure, save it temporarily
            temp_plot = get_plots_path('temp_report_plot.png')
            plt.savefig(temp_plot, dpi=300, bbox_inches='tight')
            plot_files['analysis_plot'] = str(temp_plot)
        
        report_path = model.generate_report(
            plots_data=plot_files if plot_files else None,
            report_title="Selezione Automatica Analysis",
            output_filename="selezione_automatica_report",
            format_type="html",
            include_diagnostics=True,
            include_forecast=True,
            forecast_steps=12
        )
        logger.info(f"Report HTML generato: {report_path}")
        print(f"Report HTML salvato in: {report_path}")
    except Exception as e:
        logger.warning(f"Impossibile generare report: {e}")
        print(f"Report non generato: {e}")
        # Save model comparison results
        if model_comparison:
            comparison_df = pd.DataFrame(model_comparison).T
            comparison_df.to_csv(model_dir / "model_comparison_results.csv")
            logger.info("Model comparison results saved")
        else:
            logger.info("No model comparison results to save (fallback mode)")
        
        # Save detailed results
        if selector is not None:
            results_summary = {
                'best_model': {
                    'order': best_order,
                    'aic': best_info['aic'],
                    'bic': best_info['bic'],
                    'performance': final_metrics
                },
                'preprocessing': metadata,
                'evaluation': evaluation_report
            }
        else:
            results_summary = {
                'best_model': {
                    'order': best_order,
                    'aic': 'N/A (fallback mode)',
                    'bic': 'N/A (fallback mode)', 
                    'performance': final_metrics
                },
                'preprocessing': metadata,
                'evaluation': evaluation_report
            }
        
        import json
        
        # Create a simplified version for JSON serialization
        simplified_summary = {
            'best_model': {
                'order': best_order,
                'aic': results_summary['best_model']['aic'],
                'bic': results_summary['best_model']['bic'],
                'performance': dict(final_metrics)  # Convert to simple dict
            },
            'preprocessing': dict(metadata) if isinstance(metadata, dict) else str(metadata)
        }
        
        with open(model_dir / "modeling_results.json", 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, (np.integer, int)):
                    return int(obj)
                elif isinstance(obj, (np.floating, float)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.Timestamp):
                    return obj.isoformat()
                elif isinstance(obj, (pd.Series, pd.DataFrame)):
                    return str(obj)  # Convert pandas objects to string
                elif hasattr(obj, '__dict__'):
                    return str(obj)  # Convert complex objects to string
                return obj
            
            json.dump(simplified_summary, f, indent=2, default=convert_numpy)
        
        logger.info(f"Results saved to {model_dir}")
        
        # Step 10: Future forecasting with best model
        logger.info("Generating future forecasts...")
        
        # Retrain on all data
        final_full_model = ARIMAForecaster(order=best_order)
        final_full_model.fit(processed_series)
        
        # 24-month ahead forecast
        future_forecast, future_conf_int = final_full_model.forecast(
            steps=24,
            confidence_intervals=True,
            return_conf_int=True
        )
        
        # Plot future forecast
        fig5 = plotter.plot_forecast(
            actual=processed_series.tail(48),  # Last 4 years + forecast
            forecast=future_forecast,
            confidence_intervals=future_conf_int,
            title="24-Month Future Forecast with Best Model",
            save_path=plots_dir / "future_forecast_best_model.png"
        )
        
        logger.info("Future forecast summary:")
        logger.info(f"  Next 6 months average: {future_forecast[:6].mean():.2f}")
        logger.info(f"  Next 12 months average: {future_forecast[:12].mean():.2f}")
        logger.info(f"  Full 24 months average: {future_forecast.mean():.2f}")
        
        logger.info("Automatic model selection example completed successfully!")
        logger.info(f"Best model: ARIMA{best_order}")
        logger.info(f"Test RMSE: {final_metrics['rmse']:.4f}")
        logger.info(f"Test MAPE: {final_metrics['mape']:.2f}%")
        print("Plot saved as 'outputs/plots/selezione_automatica.png'")
        
        # Display plots (optional)
        # plt.show()  # Disabled for Windows compatibility
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise

if __name__ == "__main__":
    main()