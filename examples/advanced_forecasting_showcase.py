"""
Comprehensive showcase of advanced ARIMA forecasting features.

This script demonstrates:
- SARIMA models with seasonal support
- Multivariate VAR models
- Auto-ML hyperparameter optimization
- Ensemble forecasting
- Model comparison and evaluation
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Windows
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arima_forecaster import (
from utils import get_plots_path, get_models_path, get_reports_path
    ARIMAForecaster, SARIMAForecaster, VARForecaster,
    ARIMAModelSelector, SARIMAModelSelector,
    TimeSeriesPreprocessor, ModelEvaluator,
    ARIMAOptimizer, SARIMAOptimizer, VAROptimizer,
    HyperparameterTuner, optimize_model
)


def generate_sample_data():
    """Generate sample time series data for demonstration."""
    print("📊 Generating sample datasets...")
    
    # Univariate time series with trend, seasonality, and noise
    dates = pd.date_range('2018-01-01', '2023-12-31', freq='M')
    n = len(dates)
    
    # Base trend
    trend = np.linspace(100, 200, n)
    
    # Seasonal component (yearly cycle)
    seasonal = 15 * np.sin(2 * np.pi * np.arange(n) / 12)
    
    # Random walk component
    np.random.seed(42)
    random_walk = np.cumsum(np.random.normal(0, 2, n))
    
    # Noise
    noise = np.random.normal(0, 3, n)
    
    # Combine components
    univariate_series = pd.Series(
        trend + seasonal + random_walk + noise,
        index=dates,
        name='sales'
    )
    
    # Multivariate data (e.g., sales, marketing spend, competitor index)
    marketing_spend = 50 + 0.3 * trend + 5 * np.sin(2 * np.pi * np.arange(n) / 12) + np.random.normal(0, 5, n)
    competitor_index = 80 + 0.1 * trend - 3 * np.sin(2 * np.pi * np.arange(n) / 12) + np.random.normal(0, 4, n)
    
    multivariate_data = pd.DataFrame({
        'sales': univariate_series.values,
        'marketing_spend': marketing_spend,
        'competitor_index': competitor_index
    }, index=dates)
    
    print(f"✅ Generated univariate series: {len(univariate_series)} observations")
    print(f"✅ Generated multivariate data: {multivariate_data.shape[0]} × {multivariate_data.shape[1]}")
    
    return univariate_series, multivariate_data


def demonstrate_sarima_models(data):
    """Demonstrate SARIMA models with seasonal support."""
    print("\n🌊 SARIMA Models with Seasonal Support")
    print("=" * 50)
    
    # Manual SARIMA specification
    print("\n1. Manual SARIMA Model (1,1,1)(1,1,1,12)")
    sarima_manual = SARIMAForecaster(
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12)
    )
    sarima_manual.fit(data)
    
    model_info = sarima_manual.get_model_info()
    print(f"   AIC: {model_info['aic']:.2f}")
    print(f"   BIC: {model_info['bic']:.2f}")
    
    # Generate forecast
    forecast, conf_int = sarima_manual.forecast(steps=12, return_conf_int=True)
    print(f"   12-month forecast range: {forecast.min():.2f} to {forecast.max():.2f}")
    
    # Automatic SARIMA selection
    print("\n2. Automatic SARIMA Model Selection")
    sarima_selector = SARIMAModelSelector(
        p_range=(0, 2),
        d_range=(0, 2),
        q_range=(0, 2),
        P_range=(0, 1),
        D_range=(0, 1),
        Q_range=(0, 1),
        seasonal_periods=[12],
        max_models=20
    )
    
    sarima_selector.search(data, verbose=False)
    best_sarima = sarima_selector.get_best_model()
    
    print(f"   Best SARIMA order: {sarima_selector.best_order}")
    print(f"   Best seasonal order: {sarima_selector.best_seasonal_order}")
    
    best_info = best_sarima.get_model_info()
    print(f"   Best AIC: {best_info['aic']:.2f}")
    
    # Seasonal decomposition
    print("\n3. Seasonal Decomposition")
    decomposition = best_sarima.get_seasonal_decomposition()
    print(f"   Trend component range: {decomposition['trend'].min():.2f} to {decomposition['trend'].max():.2f}")
    print(f"   Seasonal component range: {decomposition['seasonal'].min():.2f} to {decomposition['seasonal'].max():.2f}")
    
    return best_sarima


def demonstrate_var_models(data):
    """Demonstrate Vector Autoregression (VAR) models."""
    print("\n📈 Vector Autoregression (VAR) Models")
    print("=" * 50)
    
    # First fit the VAR model to check stationarity
    print("\n1. Stationarity Analysis")
    var_model = VARForecaster(maxlags=4)
    var_model.fit(data)  # Fit first so we can check stationarity
    
    stationarity_results = var_model.check_stationarity()
    
    for variable, result in stationarity_results.items():
        status = "✅" if result.get('is_stationary', False) else "❌"
        print(f"   {variable}: {status} (p-value: {result.get('p_value', 'N/A'):.4f})")
    
    # Make data stationary if needed
    stationary_data = data.diff().dropna()  # First difference
    
    # Fit VAR model with automatic lag selection on stationary data
    print("\n2. VAR Model with Automatic Lag Selection")
    var_model = VARForecaster(ic='aic')
    var_model.fit(stationary_data)
    
    model_info = var_model.get_model_info()
    print(f"   Selected lag order: {model_info['lag_order']}")
    print(f"   AIC: {model_info['aic']:.2f}")
    print(f"   BIC: {model_info['bic']:.2f}")
    
    # Generate VAR forecast
    print("\n3. VAR Forecasting")
    var_forecast = var_model.forecast(steps=6, alpha=0.05)
    
    forecast_df = var_forecast['forecast']
    print(f"   6-period forecast generated for {len(forecast_df.columns)} variables")
    
    for var in forecast_df.columns:
        forecast_range = f"{forecast_df[var].min():.2f} to {forecast_df[var].max():.2f}"
        print(f"   {var} forecast range: {forecast_range}")
    
    # Impulse Response Functions
    print("\n4. Impulse Response Analysis")
    try:
        irf = var_model.impulse_response(periods=10)
        print(f"   IRF calculated for {irf.shape[1]} variable combinations")
        
        # Example: Sales response to marketing spend shock
        if 'sales' in data.columns and 'marketing_spend' in data.columns:
            sales_response = var_model.impulse_response(
                periods=10,
                impulse='marketing_spend',
                response='sales'
            )
            max_response = sales_response.max()
            print(f"   Max sales response to marketing shock: {max_response:.4f}")
        
    except Exception as e:
        print(f"   IRF analysis failed: {e}")
    
    # Granger Causality Test
    print("\n5. Granger Causality Tests")
    try:
        if 'sales' in data.columns and len(data.columns) > 1:
            other_vars = [col for col in data.columns if col != 'sales']
            causality_results = var_model.granger_causality('sales', other_vars)
            
            for test, result in causality_results.items():
                significance = "✅" if result['p_value'] < 0.05 else "❌"
                print(f"   {test}: {significance} (p-value: {result['p_value']:.4f})")
                
    except Exception as e:
        print(f"   Granger causality tests failed: {e}")
    
    return var_model


def demonstrate_automl_optimization(data):
    """Demonstrate Auto-ML hyperparameter optimization."""
    print("\n🤖 Auto-ML Hyperparameter Optimization")
    print("=" * 50)
    
    # Single-objective optimization with different algorithms
    print("\n1. Single-Objective Optimization Comparison")
    
    algorithms = ['optuna']  # Focus on Optuna as it's most likely to be available
    results = {}
    
    for algorithm in algorithms:
        try:
            print(f"\n   Testing {algorithm.upper()}:")
            
            if algorithm == 'optuna':
                optimizer = ARIMAOptimizer(objective_metric='aic')
                result = optimizer.optimize_optuna(data, n_trials=20)
            
            results[algorithm] = result
            print(f"   ✅ Best parameters: {result['best_params']}")
            print(f"   ✅ Best score: {result['best_score']:.2f}")
            print(f"   ✅ Trials completed: {result.get('n_trials', 'N/A')}")
            
        except ImportError as e:
            print(f"   ❌ {algorithm} not available: {e}")
        except Exception as e:
            print(f"   ❌ {algorithm} failed: {e}")
    
    # Multi-objective optimization
    print("\n2. Multi-Objective Optimization")
    try:
        tuner = HyperparameterTuner(
            objective_metrics=['aic', 'bic'],
            ensemble_method='weighted_average'
        )
        
        multi_result = tuner.multi_objective_optimization(
            'arima', data, n_trials=15
        )
        
        print(f"   ✅ Pareto-optimal solutions found: {multi_result['n_pareto_solutions']}")
        
        if multi_result['best_solution']:
            best_params = multi_result['best_solution']['params']
            print(f"   ✅ Best compromise solution: {best_params}")
            
            # Show scores for multiple objectives
            scores = multi_result['best_solution']['scores']
            for metric, score in scores.items():
                print(f"      {metric.upper()}: {score:.2f}")
        
    except Exception as e:
        print(f"   ❌ Multi-objective optimization failed: {e}")
    
    # Ensemble optimization
    print("\n3. Ensemble Model Creation")
    try:
        ensemble_result = tuner.ensemble_optimization(
            'arima', data, n_models=3, diversity_threshold=0.2
        )
        
        n_models = ensemble_result['n_models']
        diversity = ensemble_result['diversity_metrics']
        
        print(f"   ✅ Ensemble created with {n_models} models")
        print(f"   ✅ Average diversity: {diversity['avg_diversity']:.3f}")
        print(f"   ✅ Ensemble score: {ensemble_result['ensemble_score']:.2f}")
        
        # Generate ensemble forecast
        ensemble_forecast = tuner.forecast_ensemble(steps=6, method='weighted')
        print(f"   ✅ Ensemble forecast generated (6 steps)")
        print(f"      Forecast range: {ensemble_forecast['forecast'].min():.2f} to {ensemble_forecast['forecast'].max():.2f}")
        
    except Exception as e:
        print(f"   ❌ Ensemble optimization failed: {e}")
    
    # Adaptive optimization
    print("\n4. Adaptive Optimization")
    try:
        adaptive_result = tuner.adaptive_optimization(
            'arima', data, max_iterations=5, improvement_threshold=0.01
        )
        
        iterations = adaptive_result['total_iterations']
        final_score = adaptive_result['final_score']
        converged = adaptive_result['converged']
        
        print(f"   ✅ Completed {iterations} iterations")
        print(f"   ✅ Final score: {final_score:.2f}")
        print(f"   ✅ Converged: {'Yes' if converged else 'No'}")
        
        if adaptive_result['best_iteration']:
            best_iter = adaptive_result['best_iteration']
            print(f"   ✅ Best iteration: {best_iter['iteration']} (score: {best_iter['score']:.2f})")
        
    except Exception as e:
        print(f"   ❌ Adaptive optimization failed: {e}")
    
    return results


def comprehensive_model_comparison(data):
    """Compare all model types comprehensively."""
    print("\n🏆 Comprehensive Model Comparison")
    print("=" * 50)
    
    models = {}
    
    # Prepare train/test split
    split_point = int(0.8 * len(data))
    train_data = data[:split_point]
    test_data = data[split_point:]
    
    print(f"\nTraining on {len(train_data)} observations, testing on {len(test_data)} observations")
    
    # 1. Basic ARIMA
    print("\n1. Basic ARIMA Model")
    try:
        arima = ARIMAForecaster(order=(2, 1, 2))
        arima.fit(train_data)
        
        arima_forecast = arima.forecast(steps=len(test_data), confidence_intervals=False)
        arima_mse = np.mean((arima_forecast[:len(test_data)] - test_data[:len(arima_forecast)]) ** 2)
        
        models['ARIMA'] = {
            'model': arima,
            'forecast': arima_forecast,
            'mse': arima_mse,
            'aic': arima.get_model_info()['aic']
        }
        print(f"   ✅ MSE: {arima_mse:.2f}, AIC: {models['ARIMA']['aic']:.2f}")
        
    except Exception as e:
        print(f"   ❌ ARIMA failed: {e}")
    
    # 2. SARIMA Model
    print("\n2. SARIMA Model")
    try:
        sarima = SARIMAForecaster(
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12)
        )
        sarima.fit(train_data)
        
        sarima_forecast = sarima.forecast(steps=len(test_data), confidence_intervals=False)
        sarima_mse = np.mean((sarima_forecast[:len(test_data)] - test_data[:len(sarima_forecast)]) ** 2)
        
        models['SARIMA'] = {
            'model': sarima,
            'forecast': sarima_forecast,
            'mse': sarima_mse,
            'aic': sarima.get_model_info()['aic']
        }
        print(f"   ✅ MSE: {sarima_mse:.2f}, AIC: {models['SARIMA']['aic']:.2f}")
        
    except Exception as e:
        print(f"   ❌ SARIMA failed: {e}")
    
    # 3. Auto-optimized ARIMA
    print("\n3. Auto-Optimized ARIMA")
    try:
        optimizer = ARIMAOptimizer(objective_metric='aic')
        opt_result = optimizer.optimize_optuna(train_data, n_trials=15)
        
        opt_model = optimizer.best_model
        opt_forecast = opt_model.forecast(steps=len(test_data), confidence_intervals=False)
        opt_mse = np.mean((opt_forecast[:len(test_data)] - test_data[:len(opt_forecast)]) ** 2)
        
        models['Auto-ARIMA'] = {
            'model': opt_model,
            'forecast': opt_forecast,
            'mse': opt_mse,
            'aic': opt_result['best_score']
        }
        print(f"   ✅ MSE: {opt_mse:.2f}, AIC: {models['Auto-ARIMA']['aic']:.2f}")
        print(f"      Best parameters: {opt_result['best_params']}")
        
    except Exception as e:
        print(f"   ❌ Auto-ARIMA failed: {e}")
    
    # Model ranking
    print("\n🏅 Model Ranking (by MSE):")
    if models:
        ranked_models = sorted(models.items(), key=lambda x: x[1]['mse'])
        
        for i, (name, info) in enumerate(ranked_models, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            print(f"   {medal} {name}: MSE={info['mse']:.2f}, AIC={info['aic']:.2f}")
    
    return models


def create_visualization_dashboard(data, models):
    """Create comprehensive visualization dashboard."""
    print("\n📊 Creating Visualization Dashboard")
    print("=" * 50)
    
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('ARIMA Forecasting Advanced Features Dashboard', fontsize=16, fontweight='bold')
        
        # Plot 1: Original data with trend
        ax1 = axes[0, 0]
        ax1.plot(data.index, data.values, 'b-', alpha=0.7, label='Original Data')
        ax1.set_title('Time Series Data')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Model comparison (if models available)
        ax2 = axes[0, 1]
        if models:
            # Show last 24 months of data + forecasts
            recent_data = data[-24:]
            ax2.plot(recent_data.index, recent_data.values, 'k-', label='Actual', linewidth=2)
            
            colors = ['red', 'green', 'orange', 'purple']
            for i, (name, info) in enumerate(models.items()):
                if 'forecast' in info:
                    forecast = info['forecast']
                    # Create forecast index
                    last_date = data.index[-1]
                    freq = pd.infer_freq(data.index)
                    forecast_index = pd.date_range(
                        start=last_date + pd.tseries.frequencies.to_offset(freq),
                        periods=len(forecast),
                        freq=freq
                    )
                    
                    color = colors[i % len(colors)]
                    ax2.plot(forecast_index[:len(forecast)], forecast.values[:len(forecast_index)], 
                            '--', color=color, label=f'{name} Forecast', alpha=0.8)
        
        ax2.set_title('Model Forecasts Comparison')
        ax2.set_ylabel('Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Seasonal decomposition (if available)
        ax3 = axes[1, 0]
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            decomposition = seasonal_decompose(data, model='additive', period=12)
            ax3.plot(decomposition.seasonal[-48:], 'g-', label='Seasonal Component')
            ax3.set_title('Seasonal Component (Last 4 Years)')
            ax3.set_ylabel('Seasonal Effect')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        except Exception:
            ax3.text(0.5, 0.5, 'Seasonal decomposition\nnot available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Seasonal Analysis')
        
        # Plot 4: Model performance metrics
        ax4 = axes[1, 1]
        if models:
            model_names = list(models.keys())
            mse_values = [models[name]['mse'] for name in model_names]
            aic_values = [models[name]['aic'] for name in model_names]
            
            # Normalize values for comparison
            mse_norm = np.array(mse_values) / max(mse_values)
            aic_norm = np.array(aic_values) / max(aic_values)
            
            x = np.arange(len(model_names))
            width = 0.35
            
            ax4.bar(x - width/2, mse_norm, width, label='MSE (normalized)', alpha=0.7)
            ax4.bar(x + width/2, aic_norm, width, label='AIC (normalized)', alpha=0.7)
            
            ax4.set_xlabel('Models')
            ax4.set_ylabel('Normalized Score')
            ax4.set_title('Model Performance Comparison')
            ax4.set_xticks(x)
            ax4.set_xticklabels(model_names, rotation=45)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No model results\navailable', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Model Performance')
        
        plt.tight_layout()
        
        # Save the plot
        output_path = Path(__file__).parent.parent / "outputs" / "plots"
        output_path.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(output_path / "advanced_forecasting_dashboard.png", 
                   dpi=300, bbox_inches='tight')
        print(f"✅ Dashboard saved to: {output_path / 'advanced_forecasting_dashboard.png'}")
        print("Grafico salvato come 'outputs/plots/advanced_forecast_showcase.png'")
        
        # plt.show()  # Disabled for Windows compatibility
        
    except ImportError:
        print("❌ Matplotlib non disponibile per la visualizzazione")
    except Exception as e:
        print(f"❌ Visualizzazione fallita: {e}")


def main():
    """Main demonstration function."""
    print("🚀 Dimostrazione Funzionalità Avanzate di ARIMA Forecaster")
    print("=" * 60)
    print("Questo script dimostra le funzionalità avanzate aggiunte alla libreria ARIMA:")
    print("• Modelli SARIMA con supporto stagionale")
    print("• Modelli VAR multivariati")  
    print("• Ottimizzazione iperparametri Auto-ML")
    print("• Metodi ensemble e tuning avanzato")
    print("• Confronto modelli completo")
    print("=" * 60)
    
    # Generate sample data
    univariate_data, multivariate_data = generate_sample_data()
    
    # Demonstrate SARIMA models
    best_sarima = demonstrate_sarima_models(univariate_data)
    
    # Demonstrate VAR models
    var_model = demonstrate_var_models(multivariate_data)
    
    # Demonstrate Auto-ML optimization
    automl_results = demonstrate_automl_optimization(univariate_data)
    
    # Comprehensive model comparison
    model_comparison = comprehensive_model_comparison(univariate_data)
    
    # Create visualization dashboard
    create_visualization_dashboard(univariate_data, model_comparison)
    
    print("\n✅ Advanced Features Showcase Complete!")
    print("\nNext steps:")
    print("• Run the API server: python scripts/run_api.py")
    print("• Launch the dashboard: python scripts/run_dashboard.py")
    print("• Check the generated plots in outputs/plots/")
    
    return {
        'univariate_data': univariate_data,
        'multivariate_data': multivariate_data,
        'best_sarima': best_sarima,
        'var_model': var_model,
        'automl_results': automl_results,
        'model_comparison': model_comparison
    }


if __name__ == "__main__":
    results = main()