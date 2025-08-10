# Advanced ARIMA Forecasting Features

This document describes the advanced features added to the ARIMA Forecaster library, providing comprehensive tools for modern time series forecasting.

## 🌊 SARIMA Models (Seasonal ARIMA)

### Overview
SARIMA (Seasonal AutoRegressive Integrated Moving Average) models extend ARIMA to handle seasonal patterns in time series data.

### Key Features
- **Seasonal Parameters**: (P, D, Q, s) for seasonal AR, differencing, MA, and period
- **Automatic Selection**: Grid search for optimal seasonal parameters  
- **Seasonal Decomposition**: Built-in decomposition analysis
- **Multiple Seasonal Periods**: Support for different seasonal cycles

### Usage Example
```python
from arima_forecaster import SARIMAForecaster, SARIMAModelSelector

# Manual SARIMA specification
model = SARIMAForecaster(
    order=(1, 1, 1),           # Non-seasonal (p, d, q)
    seasonal_order=(1, 1, 1, 12)  # Seasonal (P, D, Q, s)
)
model.fit(data)
forecast = model.forecast(steps=12)

# Automatic SARIMA selection
selector = SARIMAModelSelector(
    seasonal_periods=[12, 4],  # Monthly and quarterly patterns
    max_models=50
)
selector.search(data)
best_model = selector.get_best_model()
```

### Applications
- **Monthly Sales Data**: Yearly seasonality (s=12)
- **Daily Traffic**: Weekly seasonality (s=7)
- **Quarterly Earnings**: Yearly seasonality (s=4)
- **Hourly Energy Consumption**: Daily (s=24) and weekly (s=168) patterns

## 📈 Vector Autoregression (VAR) Models

### Overview
VAR models handle multivariate time series where multiple variables influence each other over time.

### Key Features
- **Multivariate Forecasting**: Forecast multiple related variables simultaneously
- **Lag Selection**: Automatic optimal lag determination
- **Causality Testing**: Granger causality tests between variables
- **Impulse Response**: Analysis of variable interactions
- **Cointegration Tests**: Long-term relationship analysis

### Usage Example
```python
from arima_forecaster import VARForecaster
import pandas as pd

# Prepare multivariate data
data = pd.DataFrame({
    'sales': sales_data,
    'marketing_spend': marketing_data,
    'competitor_index': competitor_data
})

# Fit VAR model
model = VARForecaster(maxlags=4)
model.fit(data)

# Generate multivariate forecast
forecast = model.forecast(steps=6)
print(forecast['forecast'])  # Forecasts for all variables

# Analyze variable relationships
causality = model.granger_causality('sales', ['marketing_spend'])
impulse_resp = model.impulse_response(periods=10)
```

### Advanced Analysis
- **Impulse Response Functions**: How shocks in one variable affect others
- **Forecast Error Variance Decomposition**: Contribution of each variable to forecast variance
- **Granger Causality**: Statistical tests for causal relationships
- **Cointegration**: Long-term equilibrium relationships

## 🤖 Auto-ML Hyperparameter Optimization

### Overview
Advanced optimization algorithms automatically find optimal model parameters using state-of-the-art techniques.

### Supported Algorithms
- **Optuna**: Tree-structured Parzen Estimator (TPE)
- **Hyperopt**: Bayesian optimization
- **Scikit-Optimize**: Gaussian Process optimization

### Optimization Features
- **Multi-Objective**: Optimize multiple metrics simultaneously
- **Cross-Validation**: Time series-aware validation
- **Early Stopping**: Prevent overfitting
- **Parallel Processing**: Speed up optimization

### Usage Example
```python
from arima_forecaster.automl import ARIMAOptimizer, optimize_model

# Single-objective optimization
optimizer = ARIMAOptimizer(objective_metric='aic')
result = optimizer.optimize_optuna(data, n_trials=100)

print(f"Best parameters: {result['best_params']}")
print(f"Best score: {result['best_score']}")

# Convenience function for any model type
result = optimize_model(
    model_type='sarima',
    data=data,
    optimizer_type='optuna',
    n_trials=50
)
```

### Optimization Objectives
- **Information Criteria**: AIC, BIC, HQIC
- **Forecast Accuracy**: MSE, MAE, MAPE
- **Custom Metrics**: User-defined objective functions

## 🎯 Advanced Hyperparameter Tuning

### Multi-Objective Optimization
Optimize multiple competing objectives simultaneously using Pareto optimization.

```python
from arima_forecaster.automl import HyperparameterTuner

tuner = HyperparameterTuner(
    objective_metrics=['aic', 'bic', 'mse'],
    ensemble_method='weighted_average'
)

result = tuner.multi_objective_optimization('arima', data, n_trials=100)
pareto_front = result['pareto_front']
best_solution = result['best_solution']
```

### Ensemble Methods
Create diverse model ensembles for improved forecasting performance.

```python
# Create ensemble of diverse models
ensemble_result = tuner.ensemble_optimization(
    'arima', data, 
    n_models=5, 
    diversity_threshold=0.2
)

# Generate ensemble forecast
forecast = tuner.forecast_ensemble(
    steps=12, 
    method='weighted',
    confidence_level=0.95
)
```

### Adaptive Optimization
Dynamically adjust search space based on optimization progress.

```python
# Adaptive optimization with early stopping
adaptive_result = tuner.adaptive_optimization(
    'sarima', data,
    max_iterations=10,
    improvement_threshold=0.01
)
```

## 🌐 REST API for Forecasting Services

### Overview
Production-ready REST API for deploying forecasting models as web services.

### Key Endpoints
- `POST /models/train`: Train ARIMA/SARIMA models
- `POST /models/train/var`: Train VAR models  
- `POST /models/{id}/forecast`: Generate forecasts
- `POST /models/auto-select`: Automatic model selection
- `GET /models`: List all models
- `POST /models/{id}/diagnostics`: Model diagnostics

### Usage Example
```bash
# Start API server
python scripts/run_api.py --host 0.0.0.0 --port 8000

# Train a model
curl -X POST "http://localhost:8000/models/train" \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "timestamps": ["2023-01-01", "2023-02-01"],
      "values": [100, 105]
    },
    "model_type": "arima",
    "order": {"p": 1, "d": 1, "q": 1}
  }'

# Generate forecast
curl -X POST "http://localhost:8000/models/{model_id}/forecast" \
  -H "Content-Type: application/json" \
  -d '{
    "steps": 12,
    "confidence_level": 0.95,
    "return_intervals": true
  }'
```

### API Features
- **Async Processing**: Background model training
- **Model Persistence**: Automatic model storage and retrieval
- **Input Validation**: Pydantic-based request/response validation
- **Error Handling**: Comprehensive error responses
- **API Documentation**: Auto-generated OpenAPI/Swagger docs

## 📊 Interactive Streamlit Dashboard

### Overview
User-friendly web interface for exploring data, training models, and generating forecasts.

### Dashboard Features
1. **Data Upload**: CSV file upload with column mapping
2. **Data Exploration**: Interactive plots and statistics
3. **Preprocessing**: Missing values, outliers, stationarity
4. **Model Training**: All model types with parameter tuning
5. **Forecasting**: Interactive forecast generation
6. **Model Diagnostics**: Residual analysis and tests

### Usage
```bash
# Launch dashboard
python scripts/run_dashboard.py

# Access at http://localhost:8501
```

### Dashboard Pages
- **Data Upload**: Load and preview time series data
- **Data Exploration**: Visualize and analyze data patterns
- **Model Training**: Train and compare different models
- **Forecasting**: Generate and visualize predictions
- **Model Diagnostics**: Evaluate model performance

## 🚀 Installation and Setup

### Dependencies
Install the library with all advanced features:

```bash
# Install with all optional dependencies
pip install -e ".[all]"

# Or install specific feature groups
pip install -e ".[api]"      # API features
pip install -e ".[dashboard]" # Dashboard features  
pip install -e ".[automl]"    # Auto-ML features
```

### Using UV (Recommended)
```bash
# Sync all dependencies
uv sync --all-extras

# Activate environment
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

## 📈 Performance Benchmarks

### Model Training Speed
- **ARIMA**: ~0.1-1.0 seconds per model
- **SARIMA**: ~0.5-5.0 seconds per model
- **VAR**: ~0.1-2.0 seconds per model
- **Auto-ML**: ~10-300 seconds for full optimization

### Optimization Efficiency
- **Grid Search**: Baseline performance
- **Optuna TPE**: 2-5x faster convergence
- **Multi-objective**: 10-50 Pareto solutions
- **Ensemble**: 3-7 diverse models

### Memory Usage
- **Single Model**: 1-10 MB
- **Ensemble (5 models)**: 5-50 MB
- **API Server**: 50-200 MB base memory
- **Dashboard**: 100-300 MB including UI

## 🔧 Advanced Configuration

### Optimization Settings
```python
# Custom optimization configuration
optimizer = ARIMAOptimizer(
    objective_metric='aic',
    cv_folds=3,
    test_size=0.2,
    n_jobs=4,  # Parallel processing
    random_state=42
)

# Advanced tuner settings
tuner = HyperparameterTuner(
    objective_metrics=['aic', 'bic', 'mse'],
    ensemble_method='pareto',
    meta_learning=True,
    early_stopping_patience=10
)
```

### API Configuration
```python
# Custom API configuration
from arima_forecaster.api import create_app

app = create_app(model_storage_path="/path/to/models")

# Run with custom settings
import uvicorn
uvicorn.run(app, host="0.0.0.0", port=8080, workers=4)
```

## 🧪 Testing and Validation

### Model Testing
```bash
# Run all tests including advanced features
uv run pytest tests/ -v --cov=src/arima_forecaster

# Test specific modules
uv run pytest tests/test_sarima.py -v
uv run pytest tests/test_var.py -v
uv run pytest tests/test_automl.py -v
```

### API Testing
```bash
# Start test server
python scripts/run_api.py --host localhost --port 8001

# Run API tests
python -m pytest tests/test_api.py -v
```

## 📚 Examples and Tutorials

### Complete Examples
- `examples/advanced_forecasting_showcase.py`: Comprehensive feature demonstration
- `examples/api_client_example.py`: API client usage
- `examples/dashboard_demo.py`: Dashboard features walkthrough
- `examples/automl_tutorial.py`: Auto-ML optimization guide

### Jupyter Notebooks
- `notebooks/sarima_analysis.ipynb`: Seasonal modeling
- `notebooks/var_multivariate.ipynb`: VAR model exploration  
- `notebooks/automl_comparison.ipynb`: Optimization algorithms
- `notebooks/ensemble_forecasting.ipynb`: Ensemble methods

## 🤝 Contributing

### Adding New Features
1. Create feature branch
2. Implement with tests
3. Update documentation
4. Add examples
5. Submit pull request

### Extending Optimizers
```python
class CustomOptimizer(BaseOptimizer):
    def optimize_custom(self, series, **kwargs):
        # Implement custom optimization logic
        pass
```

### API Extensions
```python
@app.post("/custom-endpoint")
async def custom_forecasting_endpoint(request: CustomRequest):
    # Implement custom API endpoint
    pass
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Optuna**: Advanced optimization framework
- **FastAPI**: Modern API framework
- **Streamlit**: Interactive web applications
- **Statsmodels**: Statistical modeling foundation
- **Plotly**: Interactive visualizations

---

For more information, see the full documentation or run the showcase example:

```bash
python examples/advanced_forecasting_showcase.py
```