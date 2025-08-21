"""
Service classes for API functionality.
"""

import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

from .models import *
from ..core import ARIMAForecaster, SARIMAForecaster, VARForecaster
from ..core import ARIMAModelSelector, SARIMAModelSelector
from ..evaluation.metrics import ModelEvaluator
from ..utils.logger import get_logger
from ..utils.exceptions import ModelTrainingError, ForecastError


class ModelManager:
    """
    Manages trained models storage and retrieval.
    """
    
    def __init__(self, storage_path: Path):
        """
        Initialize model manager.
        
        Args:
            storage_path: Path to store models
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__)
        
        # In-memory model registry
        self.model_registry: Dict[str, Dict[str, Any]] = {}
        
        # Load existing models
        self._load_existing_models()
    
    def _load_existing_models(self):
        """Load existing models from disk."""
        try:
            registry_path = self.storage_path / "registry.json"
            if registry_path.exists():
                with open(registry_path, 'r') as f:
                    self.model_registry = json.load(f)
                self.logger.info(f"Loaded {len(self.model_registry)} models from registry")
        except Exception as e:
            self.logger.warning(f"Failed to load model registry: {e}")
            self.model_registry = {}
    
    def _save_registry(self):
        """Save model registry to disk."""
        try:
            registry_path = self.storage_path / "registry.json"
            # Convert datetime objects to strings for JSON serialization
            registry_copy = {}
            for model_id, info in self.model_registry.items():
                registry_copy[model_id] = info.copy()
                if 'created_at' in registry_copy[model_id]:
                    registry_copy[model_id]['created_at'] = registry_copy[model_id]['created_at'].isoformat()
            
            with open(registry_path, 'w') as f:
                json.dump(registry_copy, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save model registry: {e}")
    
    async def train_model(
        self, 
        model_id: str, 
        series: pd.Series, 
        request: ModelTrainingRequest
    ):
        """
        Train an ARIMA or SARIMA model.
        
        Args:
            model_id: Unique model identifier
            series: Time series data
            request: Training request parameters
        """
        try:
            self.logger.info(f"Training {request.model_type.upper()} model {model_id}")
            
            # Update status to training
            self.model_registry[model_id] = {
                "model_type": request.model_type,
                "status": "training",
                "created_at": datetime.now(),
                "n_observations": len(series),
                "parameters": {},
                "metrics": {}
            }
            self._save_registry()
            
            model = None
            
            if request.auto_select:
                # Automatic parameter selection
                if request.model_type == "arima":
                    selector = ARIMAModelSelector()
                    selector.search(series)
                    model = selector.get_best_model()
                    params = {"order": selector.best_order}
                elif request.model_type == "sarima":
                    selector = SARIMAModelSelector()
                    selector.search(series)
                    model = selector.get_best_model()
                    params = {
                        "order": selector.best_order,
                        "seasonal_order": selector.best_seasonal_order
                    }
            else:
                # Manual parameter specification
                if request.model_type == "arima":
                    order = (request.order.p, request.order.d, request.order.q)
                    model = ARIMAForecaster(order=order)
                    model.fit(series)
                    params = {"order": order}
                elif request.model_type == "sarima":
                    order = (request.order.p, request.order.d, request.order.q)
                    seasonal_order = (
                        request.seasonal_order.P,
                        request.seasonal_order.D,
                        request.seasonal_order.Q,
                        request.seasonal_order.s
                    )
                    model = SARIMAForecaster(order=order, seasonal_order=seasonal_order)
                    model.fit(series)
                    params = {"order": order, "seasonal_order": seasonal_order}
            
            if model is None:
                raise ModelTrainingError("Failed to create model")
            
            # Save model to disk
            model_path = self.storage_path / f"{model_id}.pkl"
            model.save(model_path)
            
            # Get model metrics
            model_info = model.get_model_info()
            metrics = {
                "aic": model_info.get("aic", 0),
                "bic": model_info.get("bic", 0),
                "hqic": model_info.get("hqic", 0)
            }
            
            # Update registry
            self.model_registry[model_id].update({
                "status": "trained",
                "parameters": params,
                "metrics": metrics,
                "model_path": str(model_path)
            })
            self._save_registry()
            
            self.logger.info(f"Model {model_id} trained successfully")
            
        except Exception as e:
            # Update status to failed
            if model_id in self.model_registry:
                self.model_registry[model_id]["status"] = "failed"
                self.model_registry[model_id]["error"] = str(e)
                self._save_registry()
            
            self.logger.error(f"Model training failed for {model_id}: {e}")
            raise
    
    async def train_var_model(
        self,
        model_id: str,
        data: pd.DataFrame,
        request: VARTrainingRequest
    ):
        """
        Train a VAR model.
        
        Args:
            model_id: Unique model identifier
            data: Multivariate time series data
            request: VAR training request parameters
        """
        try:
            self.logger.info(f"Training VAR model {model_id}")
            
            # Update status to training
            self.model_registry[model_id] = {
                "model_type": "var",
                "status": "training",
                "created_at": datetime.now(),
                "n_observations": len(data),
                "n_variables": data.shape[1],
                "variable_names": list(data.columns),
                "parameters": {"maxlags": request.maxlags, "ic": request.ic},
                "metrics": {}
            }
            self._save_registry()
            
            # Train VAR model
            model = VARForecaster(maxlags=request.maxlags, ic=request.ic)
            model.fit(data)
            
            # Save model to disk
            model_path = self.storage_path / f"{model_id}.pkl"
            model.save(model_path)
            
            # Get model metrics
            model_info = model.get_model_info()
            metrics = {
                "aic": model_info.get("aic", 0),
                "bic": model_info.get("bic", 0),
                "hqic": model_info.get("hqic", 0),
                "fpe": model_info.get("fpe", 0)
            }
            
            # Update registry
            self.model_registry[model_id].update({
                "status": "trained",
                "metrics": metrics,
                "model_path": str(model_path),
                "selected_lag": model_info.get("lag_order", 0)
            })
            self._save_registry()
            
            self.logger.info(f"VAR model {model_id} trained successfully")
            
        except Exception as e:
            # Update status to failed
            if model_id in self.model_registry:
                self.model_registry[model_id]["status"] = "failed"
                self.model_registry[model_id]["error"] = str(e)
                self._save_registry()
            
            self.logger.error(f"VAR model training failed for {model_id}: {e}")
            raise
    
    def load_model(self, model_id: str):
        """
        Load a trained model from disk.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Loaded model instance
        """
        try:
            if model_id not in self.model_registry:
                raise ValueError(f"Model {model_id} not found in registry")
            
            model_info = self.model_registry[model_id]
            model_path = Path(model_info["model_path"])
            
            if not model_path.exists():
                raise ValueError(f"Model file not found: {model_path}")
            
            model_type = model_info["model_type"]
            
            if model_type == "arima":
                return ARIMAForecaster.load(model_path)
            elif model_type == "sarima":
                return SARIMAForecaster.load(model_path)
            elif model_type == "var":
                return VARForecaster.load(model_path)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
        except Exception as e:
            self.logger.error(f"Failed to load model {model_id}: {e}")
            raise
    
    def model_exists(self, model_id: str) -> bool:
        """Check if a model exists."""
        return model_id in self.model_registry
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get model information."""
        if model_id not in self.model_registry:
            raise ValueError(f"Model {model_id} not found")
        return self.model_registry[model_id].copy()
    
    def list_models(self) -> List[str]:
        """List all model IDs."""
        return list(self.model_registry.keys())
    
    def delete_model(self, model_id: str):
        """Delete a model."""
        try:
            if model_id not in self.model_registry:
                raise ValueError(f"Model {model_id} not found")
            
            # Delete model file
            model_info = self.model_registry[model_id]
            if "model_path" in model_info:
                model_path = Path(model_info["model_path"])
                if model_path.exists():
                    model_path.unlink()
                
                # Also delete metadata file
                metadata_path = model_path.with_suffix('.metadata.pkl')
                if metadata_path.exists():
                    metadata_path.unlink()
            
            # Remove from registry
            del self.model_registry[model_id]
            self._save_registry()
            
            self.logger.info(f"Model {model_id} deleted")
            
        except Exception as e:
            self.logger.error(f"Failed to delete model {model_id}: {e}")
            raise


class ForecastService:
    """
    Service for generating forecasts and model diagnostics.
    """
    
    def __init__(self, model_manager: ModelManager):
        """
        Initialize forecast service.
        
        Args:
            model_manager: Model manager instance
        """
        self.model_manager = model_manager
        self.logger = get_logger(__name__)
    
    async def generate_forecast(
        self,
        model_id: str,
        steps: int,
        confidence_level: float = 0.95,
        return_intervals: bool = True
    ) -> ForecastResult:
        """
        Generate forecast from ARIMA/SARIMA model.
        
        Args:
            model_id: Model identifier
            steps: Number of forecast steps
            confidence_level: Confidence level for intervals
            return_intervals: Whether to return confidence intervals
            
        Returns:
            Forecast result
        """
        try:
            # Load model
            model = self.model_manager.load_model(model_id)
            
            # Generate forecast
            alpha = 1 - confidence_level
            if return_intervals:
                forecast, conf_int = model.forecast(
                    steps=steps,
                    alpha=alpha,
                    return_conf_int=True
                )
            else:
                forecast = model.forecast(steps=steps, confidence_intervals=False)
                conf_int = None
            
            # Convert to lists
            forecast_timestamps = [str(ts) for ts in forecast.index]
            forecast_values = forecast.tolist()
            
            lower_bounds = None
            upper_bounds = None
            if conf_int is not None:
                lower_bounds = conf_int.iloc[:, 0].tolist()
                upper_bounds = conf_int.iloc[:, 1].tolist()
            
            return ForecastResult(
                model_id=model_id,
                forecast_timestamps=forecast_timestamps,
                forecast_values=forecast_values,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                confidence_level=confidence_level if return_intervals else None,
                generated_at=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Forecast generation failed: {e}")
            raise ForecastError(f"Failed to generate forecast: {e}")
    
    async def generate_var_forecast(
        self,
        model_id: str,
        steps: int,
        confidence_level: float = 0.95,
        return_intervals: bool = True
    ) -> VARForecastResult:
        """
        Generate forecast from VAR model.
        
        Args:
            model_id: Model identifier
            steps: Number of forecast steps
            confidence_level: Confidence level for intervals
            return_intervals: Whether to return confidence intervals
            
        Returns:
            VAR forecast result
        """
        try:
            # Load model
            model = self.model_manager.load_model(model_id)
            
            # Generate forecast
            alpha = 1 - confidence_level
            forecast_result = model.forecast(steps=steps, alpha=alpha)
            
            # Extract forecasts
            forecast_df = forecast_result['forecast']
            forecast_timestamps = [str(ts) for ts in forecast_df.index]
            
            forecasts = {}
            lower_bounds = None
            upper_bounds = None
            
            for col in forecast_df.columns:
                forecasts[col] = forecast_df[col].tolist()
            
            if return_intervals:
                lower_df = forecast_result['lower_bounds']
                upper_df = forecast_result['upper_bounds']
                
                lower_bounds = {}
                upper_bounds = {}
                
                for col in forecast_df.columns:
                    lower_bounds[col] = lower_df[col].tolist()
                    upper_bounds[col] = upper_df[col].tolist()
            
            return VARForecastResult(
                model_id=model_id,
                forecast_timestamps=forecast_timestamps,
                forecasts=forecasts,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                confidence_level=confidence_level if return_intervals else None,
                generated_at=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"VAR forecast generation failed: {e}")
            raise ForecastError(f"Failed to generate VAR forecast: {e}")
    
    async def auto_select_model(
        self,
        series: pd.Series,
        model_type: str,
        max_models: int = 50,
        information_criterion: str = 'aic'
    ) -> Dict[str, Any]:
        """
        Automatically select best model parameters.
        
        Args:
            series: Time series data
            model_type: Type of model ('arima' or 'sarima')
            max_models: Maximum models to test
            information_criterion: Criterion for selection
            
        Returns:
            Selection results
        """
        try:
            if model_type == "arima":
                selector = ARIMAModelSelector(
                    information_criterion=information_criterion,
                    max_models=max_models
                )
            elif model_type == "sarima":
                selector = SARIMAModelSelector(
                    information_criterion=information_criterion,
                    max_models=max_models
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Perform selection
            selector.search(series)
            
            # Get best model and save it
            best_model = selector.get_best_model()
            if best_model is None:
                raise ForecastError("No suitable model found")
            
            # Generate model ID and save
            import uuid
            model_id = str(uuid.uuid4())
            
            model_path = self.model_manager.storage_path / f"{model_id}.pkl"
            best_model.save(model_path)
            
            # Get results
            results_df = selector.get_results_summary()
            all_results = results_df.to_dict('records')
            
            # Update model registry
            best_info = best_model.get_model_info()
            self.model_manager.model_registry[model_id] = {
                "model_type": model_type,
                "status": "trained",
                "created_at": datetime.now(),
                "n_observations": len(series),
                "parameters": {
                    "order": selector.best_order,
                    "seasonal_order": getattr(selector, 'best_seasonal_order', None)
                },
                "metrics": {
                    "aic": best_info.get("aic", 0),
                    "bic": best_info.get("bic", 0),
                    "hqic": best_info.get("hqic", 0)
                },
                "model_path": str(model_path)
            }
            self.model_manager._save_registry()
            
            return {
                "best_model_id": model_id,
                "best_parameters": {
                    "order": selector.best_order,
                    "seasonal_order": getattr(selector, 'best_seasonal_order', None)
                },
                "best_score": getattr(best_info, information_criterion),
                "all_results": all_results
            }
            
        except Exception as e:
            self.logger.error(f"Auto model selection failed: {e}")
            raise ForecastError(f"Auto model selection failed: {e}")
    
    async def generate_diagnostics(
        self,
        model_id: str,
        include_residuals: bool = True,
        include_acf_pacf: bool = True
    ) -> ModelDiagnostics:
        """
        Generate model diagnostics.
        
        Args:
            model_id: Model identifier
            include_residuals: Include residual analysis
            include_acf_pacf: Include ACF/PACF analysis
            
        Returns:
            Model diagnostics
        """
        try:
            # Load model
            model = self.model_manager.load_model(model_id)
            
            # Get model info
            model_info = self.model_manager.get_model_info(model_id)
            
            if model_info["model_type"] == "var":
                # VAR models don't have standard residual diagnostics
                return ModelDiagnostics(model_id=model_id)
            
            diagnostics = ModelDiagnostics(model_id=model_id)
            
            if include_residuals and hasattr(model, 'fitted_model'):
                # Get residuals
                residuals = model.fitted_model.resid
                
                # Basic residual statistics
                diagnostics.residual_stats = {
                    "mean": float(residuals.mean()),
                    "std": float(residuals.std()),
                    "min": float(residuals.min()),
                    "max": float(residuals.max()),
                    "skewness": float(residuals.skew()) if hasattr(residuals, 'skew') else 0,
                    "kurtosis": float(residuals.kurtosis()) if hasattr(residuals, 'kurtosis') else 0
                }
                
                # Statistical tests (simplified)
                try:
                    from scipy import stats
                    
                    # Normality test
                    stat, p_value = stats.jarque_bera(residuals.dropna())
                    diagnostics.normality_test = {
                        "test_statistic": float(stat),
                        "p_value": float(p_value),
                        "is_normal": p_value > 0.05
                    }
                    
                except ImportError:
                    pass
            
            if include_acf_pacf:
                try:
                    from statsmodels.tsa.stattools import acf, pacf
                    
                    # Get training data for ACF/PACF
                    if hasattr(model, 'training_data'):
                        series = model.training_data
                        
                        # Calculate ACF and PACF
                        acf_vals = acf(series, nlags=min(20, len(series)//4))
                        pacf_vals = pacf(series, nlags=min(20, len(series)//4))
                        
                        diagnostics.acf_values = acf_vals.tolist()
                        diagnostics.pacf_values = pacf_vals.tolist()
                        
                except Exception:
                    pass
            
            return diagnostics
            
        except Exception as e:
            self.logger.error(f"Diagnostics generation failed: {e}")
            raise ForecastError(f"Failed to generate diagnostics: {e}")
    
    async def generate_report(
        self,
        model_id: str,
        report_title: Optional[str] = None,
        output_filename: Optional[str] = None,
        format_type: str = "html",
        include_diagnostics: bool = True,
        include_forecast: bool = True,
        forecast_steps: int = 12
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive report for a trained model.
        
        Args:
            model_id: ID of the trained model
            report_title: Custom title for the report
            output_filename: Custom filename for the report
            format_type: Output format (html, pdf, docx)
            include_diagnostics: Whether to include model diagnostics
            include_forecast: Whether to include forecast analysis
            forecast_steps: Number of forecast steps
            
        Returns:
            Dictionary with report information
        """
        try:
            # Load model
            model = self.model_manager.load_model(model_id)
            
            # Set default title and filename
            if report_title is None:
                model_info = self.model_manager.get_model_info(model_id)
                model_type = model_info.get("model_type", "ARIMA").upper()
                report_title = f"{model_type} Model Analysis Report"
            
            if output_filename is None:
                timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"dashboard_report_{model_id[:8]}_{timestamp}"
            
            # Generate any plots if needed (optional - can be None for basic reports)
            plots_data = None
            
            # Create visualization if forecast is requested
            if include_forecast and hasattr(model, 'forecast'):
                try:
                    # Generate forecast for visualization
                    forecast_result = model.forecast(
                        steps=forecast_steps,
                        confidence_intervals=True
                    )
                    
                    # We could create a plot here and save it, but for simplicity
                    # we'll let the report generator handle the visualization
                    plots_data = {}
                    
                except Exception as e:
                    self.logger.warning(f"Could not generate forecast for report: {e}")
            
            # Generate the report using the model's built-in report generation
            report_path = model.generate_report(
                plots_data=plots_data,
                report_title=report_title,
                output_filename=output_filename,
                format_type=format_type,
                include_diagnostics=include_diagnostics,
                include_forecast=include_forecast,
                forecast_steps=forecast_steps
            )
            
            return {
                "model_id": model_id,
                "report_path": str(report_path),
                "format_type": format_type,
                "report_title": report_title
            }
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            raise ForecastError(f"Failed to generate report: {e}")