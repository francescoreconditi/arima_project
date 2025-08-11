"""
FastAPI application for ARIMA forecasting services.
"""

import uuid
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from arima_forecaster.api.models import *
from arima_forecaster.api.services import ModelManager, ForecastService
from arima_forecaster.utils.logger import get_logger


def create_app(model_storage_path: Optional[str] = None) -> FastAPI:
    """
    Create FastAPI application.
    
    Args:
        model_storage_path: Path to store trained models
        
    Returns:
        FastAPI application instance
    """
    
    app = FastAPI(
        title="ARIMA Forecaster API",
        description="REST API for ARIMA time series forecasting",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize services
    storage_path = Path(model_storage_path or "models")
    model_manager = ModelManager(storage_path)
    forecast_service = ForecastService(model_manager)
    logger = get_logger(__name__)
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "ARIMA Forecaster API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health"
        }
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": datetime.now(),
            "models_count": len(model_manager.list_models())
        }
    
    @app.post("/models/train", response_model=ModelInfo)
    async def train_model(
        request: ModelTrainingRequest,
        background_tasks: BackgroundTasks
    ):
        """
        Train an ARIMA or SARIMA model.
        """
        try:
            # Convert request data to pandas Series
            timestamps = pd.to_datetime(request.data.timestamps)
            series = pd.Series(request.data.values, index=timestamps)
            
            # Generate model ID
            model_id = str(uuid.uuid4())
            
            # Train model in background
            background_tasks.add_task(
                _train_model_background,
                model_manager,
                model_id,
                series,
                request
            )
            
            # Return initial model info
            return ModelInfo(
                model_id=model_id,
                model_type=request.model_type,
                status="training",
                created_at=datetime.now(),
                training_observations=len(series),
                parameters={},
                metrics={}
            )
            
        except Exception as e:
            logger.error(f"Model training request failed: {e}")
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.post("/models/train/var", response_model=ModelInfo)
    async def train_var_model(
        request: VARTrainingRequest,
        background_tasks: BackgroundTasks
    ):
        """
        Train a VAR model for multivariate time series.
        """
        try:
            # Convert request data to pandas DataFrame
            timestamps = pd.to_datetime(request.data.timestamps)
            data_dict = request.data.data
            
            df = pd.DataFrame(data_dict, index=timestamps)
            
            # Generate model ID
            model_id = str(uuid.uuid4())
            
            # Train model in background
            background_tasks.add_task(
                _train_var_model_background,
                model_manager,
                model_id,
                df,
                request
            )
            
            # Return initial model info
            return ModelInfo(
                model_id=model_id,
                model_type="var",
                status="training",
                created_at=datetime.now(),
                training_observations=len(df),
                parameters={"maxlags": request.maxlags, "ic": request.ic},
                metrics={}
            )
            
        except Exception as e:
            logger.error(f"VAR model training request failed: {e}")
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.post("/models/auto-select", response_model=AutoSelectionResult)
    async def auto_select_model(request: AutoSelectionRequest):
        """
        Automatically select best model parameters.
        """
        try:
            start_time = time.time()
            
            # Convert request data to pandas Series
            timestamps = pd.to_datetime(request.data.timestamps)
            series = pd.Series(request.data.values, index=timestamps)
            
            # Perform auto selection
            result = await forecast_service.auto_select_model(
                series=series,
                model_type=request.model_type,
                max_models=request.max_models,
                information_criterion=request.information_criterion
            )
            
            selection_time = time.time() - start_time
            
            return AutoSelectionResult(
                best_model_id=result["best_model_id"],
                best_parameters=result["best_parameters"],
                best_score=result["best_score"],
                all_results=result["all_results"],
                selection_time=selection_time
            )
            
        except Exception as e:
            logger.error(f"Auto selection failed: {e}")
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.post("/models/{model_id}/forecast")
    async def forecast(
        model_id: str,
        request: ForecastRequest
    ):
        """
        Generate forecasts from a trained model.
        """
        try:
            if not model_manager.model_exists(model_id):
                raise HTTPException(status_code=404, detail="Model not found")
            
            # Get model info to determine type
            model_info = model_manager.get_model_info(model_id)
            
            if model_info["model_type"] == "var":
                # VAR model forecast
                result = await forecast_service.generate_var_forecast(
                    model_id=model_id,
                    steps=request.steps,
                    confidence_level=request.confidence_level,
                    return_intervals=request.return_intervals
                )
                return result
            else:
                # ARIMA/SARIMA forecast
                result = await forecast_service.generate_forecast(
                    model_id=model_id,
                    steps=request.steps,
                    confidence_level=request.confidence_level,
                    return_intervals=request.return_intervals
                )
                return result
                
        except Exception as e:
            logger.error(f"Forecast generation failed: {e}")
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.get("/models", response_model=ModelListResponse)
    async def list_models():
        """
        List all trained models.
        """
        try:
            models = model_manager.list_models()
            model_infos = []
            
            for model_id in models:
                try:
                    info = model_manager.get_model_info(model_id)
                    model_infos.append(ModelInfo(
                        model_id=model_id,
                        model_type=info.get("model_type", "unknown"),
                        status=info.get("status", "unknown"),
                        created_at=info.get("created_at", datetime.now()),
                        training_observations=info.get("n_observations", 0),
                        parameters=info.get("parameters", {}),
                        metrics=info.get("metrics", {})
                    ))
                except Exception as e:
                    logger.warning(f"Could not load info for model {model_id}: {e}")
                    continue
            
            return ModelListResponse(
                models=model_infos,
                total=len(model_infos)
            )
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/models/{model_id}", response_model=ModelInfo)
    async def get_model_info(model_id: str):
        """
        Get information about a specific model.
        """
        try:
            if not model_manager.model_exists(model_id):
                raise HTTPException(status_code=404, detail="Model not found")
            
            info = model_manager.get_model_info(model_id)
            
            return ModelInfo(
                model_id=model_id,
                model_type=info.get("model_type", "unknown"),
                status=info.get("status", "unknown"),
                created_at=info.get("created_at", datetime.now()),
                training_observations=info.get("n_observations", 0),
                parameters=info.get("parameters", {}),
                metrics=info.get("metrics", {})
            )
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.delete("/models/{model_id}")
    async def delete_model(model_id: str):
        """
        Delete a trained model.
        """
        try:
            if not model_manager.model_exists(model_id):
                raise HTTPException(status_code=404, detail="Model not found")
            
            model_manager.delete_model(model_id)
            
            return {"message": f"Model {model_id} deleted successfully"}
            
        except Exception as e:
            logger.error(f"Failed to delete model: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/models/{model_id}/diagnostics", response_model=ModelDiagnostics)
    async def get_model_diagnostics(
        model_id: str,
        request: ModelDiagnosticsRequest
    ):
        """
        Get diagnostic information for a model.
        """
        try:
            if not model_manager.model_exists(model_id):
                raise HTTPException(status_code=404, detail="Model not found")
            
            diagnostics = await forecast_service.generate_diagnostics(
                model_id=model_id,
                include_residuals=request.include_residuals,
                include_acf_pacf=request.include_acf_pacf
            )
            
            return diagnostics
            
        except Exception as e:
            logger.error(f"Failed to generate diagnostics: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


async def _train_model_background(
    model_manager: 'ModelManager',
    model_id: str,
    series: pd.Series,
    request: ModelTrainingRequest
):
    """Background task for model training."""
    try:
        await model_manager.train_model(model_id, series, request)
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Background model training failed for {model_id}: {e}")


async def _train_var_model_background(
    model_manager: 'ModelManager',
    model_id: str,
    data: pd.DataFrame,
    request: VARTrainingRequest
):
    """Background task for VAR model training."""
    try:
        await model_manager.train_var_model(model_id, data, request)
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Background VAR model training failed for {model_id}: {e}")


# Create default app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)