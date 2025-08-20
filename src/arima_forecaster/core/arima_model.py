"""
Core ARIMA model implementation with enhanced functionality.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima.model import ARIMAResults
from ..utils.logger import get_logger
from ..utils.exceptions import ModelTrainingError, ForecastError


class ARIMAForecaster:
    """
    Enhanced ARIMA forecaster with comprehensive functionality.
    """
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1)):
        """
        Initialize ARIMA forecaster.
        
        Args:
            order: ARIMA order (p, d, q)
        """
        self.order = order
        self.model = None
        self.fitted_model = None
        self.training_data = None
        self.training_metadata = {}
        self.logger = get_logger(__name__)
        
    def fit(
        self, 
        series: pd.Series,
        validate_input: bool = True,
        **fit_kwargs
    ) -> 'ARIMAForecaster':
        """
        Fit ARIMA model to time series data.
        
        Args:
            series: Time series data to fit
            validate_input: Whether to validate input data
            **fit_kwargs: Additional arguments for model fitting
            
        Returns:
            Self for method chaining
            
        Raises:
            ModelTrainingError: If model training fails
        """
        try:
            self.logger.info(f"Fitting ARIMA{self.order} model to {len(series)} observations")
            
            if validate_input:
                self._validate_series(series)
            
            # Store training data and metadata
            self.training_data = series.copy()
            self.training_metadata = {
                'training_start': series.index.min(),
                'training_end': series.index.max(), 
                'training_observations': len(series),
                'order': self.order
            }
            
            # Create and fit model
            self.model = ARIMA(series, order=self.order)
            self.fitted_model = self.model.fit(**fit_kwargs)
            
            # Log model summary
            self.logger.info("Modello ARIMA addestrato con successo")
            self.logger.info(f"AIC: {self.fitted_model.aic:.2f}")
            self.logger.info(f"BIC: {self.fitted_model.bic:.2f}")
            
            return self
            
        except Exception as e:
            self.logger.error(f"Addestramento modello fallito: {e}")
            raise ModelTrainingError(f"Impossibile addestrare il modello ARIMA: {e}")
    
    def forecast(
        self, 
        steps: int,
        confidence_intervals: bool = True,
        alpha: float = 0.05,
        return_conf_int: bool = False
    ) -> Union[pd.Series, Tuple[pd.Series, pd.DataFrame], Dict[str, Union[pd.Series, Dict[str, pd.Series]]]]:
        """
        Generate forecasts from fitted model.
        
        Args:
            steps: Number of steps to forecast
            confidence_intervals: Whether to calculate confidence intervals
            alpha: Alpha level for confidence intervals (1-alpha = confidence level)
            return_conf_int: Whether to return confidence intervals
            
        Returns:
            Forecast series, optionally with confidence intervals
            
        Raises:
            ForecastError: If forecasting fails
        """
        try:
            if self.fitted_model is None:
                raise ForecastError("Il modello deve essere addestrato prima del forecasting")
            
            self.logger.info(f"Generazione forecast a {steps} passi")
            
            # Generate forecast
            if confidence_intervals:
                forecast_result = self.fitted_model.forecast(
                    steps=steps, 
                    alpha=alpha
                )
                forecast_values = forecast_result
                conf_int = self.fitted_model.get_forecast(steps=steps, alpha=alpha).conf_int()
            else:
                forecast_values = self.fitted_model.forecast(steps=steps)
                conf_int = None
            
            # Create forecast index
            last_date = self.training_data.index[-1]
            if isinstance(last_date, pd.Timestamp):
                freq = pd.infer_freq(self.training_data.index)
                if freq is None:
                    # Fallback: calculate frequency from first two dates
                    freq = self.training_data.index[1] - self.training_data.index[0]
                    forecast_index = pd.date_range(
                        start=last_date + freq,
                        periods=steps,
                        freq=freq
                    )
                else:
                    forecast_index = pd.date_range(
                        start=last_date,
                        periods=steps + 1,
                        freq=freq
                    )[1:]  # Skip first element to avoid overlap
            else:
                forecast_index = range(len(self.training_data), len(self.training_data) + steps)
            
            forecast_series = pd.Series(forecast_values, index=forecast_index, name='forecast')
            
            self.logger.info(f"Forecast generato: {forecast_series.iloc[0]:.2f} a {forecast_series.iloc[-1]:.2f}")
            
            # Return format depends on parameters
            if confidence_intervals and conf_int is not None:
                conf_int.index = forecast_index
                if return_conf_int:
                    return forecast_series, conf_int
                else:
                    # Return dictionary format for compatibility with examples
                    return {
                        'forecast': forecast_series,
                        'confidence_intervals': {
                            'lower': conf_int.iloc[:, 0],
                            'upper': conf_int.iloc[:, 1]
                        }
                    }
            else:
                return forecast_series
                
        except Exception as e:
            self.logger.error(f"Forecasting fallito: {e}")
            raise ForecastError(f"Impossibile generare il forecast: {e}")
    
    def predict(
        self, 
        start: Optional[Union[int, str, pd.Timestamp]] = None,
        end: Optional[Union[int, str, pd.Timestamp]] = None,
        dynamic: bool = False
    ) -> pd.Series:
        """
        Generate in-sample and out-of-sample predictions.
        
        Args:
            start: Start of prediction period
            end: End of prediction period  
            dynamic: Whether to use dynamic prediction
            
        Returns:
            Series of predictions
        """
        try:
            if self.fitted_model is None:
                raise ForecastError("Il modello deve essere addestrato prima della predizione")
            
            predictions = self.fitted_model.predict(start=start, end=end, dynamic=dynamic)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Predizione fallita: {e}")
            raise ForecastError(f"Impossibile generare le predizioni: {e}")
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save fitted model to disk.
        
        Args:
            filepath: Path to save model
        """
        try:
            if self.fitted_model is None:
                raise ModelTrainingError("Nessun modello addestrato da salvare")
            
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Save using statsmodels built-in method
            self.fitted_model.save(str(filepath))
            
            # Also save metadata
            metadata_path = filepath.with_suffix('.metadata.pkl')
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'order': self.order,
                    'training_metadata': self.training_metadata
                }, f)
            
            self.logger.info(f"Modello salvato in {filepath}")
            
        except Exception as e:
            self.logger.error(f"Impossibile salvare il modello: {e}")
            raise ModelTrainingError(f"Impossibile salvare il modello: {e}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'ARIMAForecaster':
        """
        Load fitted model from disk.
        
        Args:
            filepath: Path to saved model
            
        Returns:
            Loaded ARIMAForecaster instance
        """
        try:
            filepath = Path(filepath)
            
            # Load the fitted model
            fitted_model = ARIMAResults.load(str(filepath))
            
            # Load metadata if available
            metadata_path = filepath.with_suffix('.metadata.pkl')
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                order = metadata.get('order', (1, 1, 1))
                training_metadata = metadata.get('training_metadata', {})
            else:
                order = (1, 1, 1)  # Default order
                training_metadata = {}
            
            # Create instance and populate
            instance = cls(order=order)
            instance.fitted_model = fitted_model
            instance.training_metadata = training_metadata
            
            instance.logger.info(f"Modello caricato da {filepath}")
            
            return instance
            
        except Exception as e:
            logger = get_logger(__name__)
            logger.error(f"Impossibile caricare il modello: {e}")
            raise ModelTrainingError(f"Impossibile caricare il modello: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Dictionary with model information
        """
        if self.fitted_model is None:
            return {'status': 'not_fitted'}
        
        info = {
            'status': 'fitted',
            'order': self.order,
            'aic': self.fitted_model.aic,
            'bic': self.fitted_model.bic,
            'hqic': self.fitted_model.hqic,
            'llf': self.fitted_model.llf,
            'n_observations': self.fitted_model.nobs,
            'params': self.fitted_model.params.to_dict() if hasattr(self.fitted_model.params, 'to_dict') else dict(self.fitted_model.params),
            'training_metadata': self.training_metadata
        }
        
        return info
    
    def _validate_series(self, series: pd.Series) -> None:
        """
        Validate input time series.
        
        Args:
            series: Series to validate
            
        Raises:
            ModelTrainingError: If validation fails
        """
        if not isinstance(series, pd.Series):
            raise ModelTrainingError("L'input deve essere una pandas Series")
        
        if len(series) == 0:
            raise ModelTrainingError("La serie non può essere vuota")
        
        if series.isnull().all():
            raise ModelTrainingError("La serie non può contenere solo valori NaN")
        
        if len(series) < 10:
            self.logger.warning("La serie ha meno di 10 osservazioni, il modello potrebbe essere inaffidabile")
        
        if series.isnull().any():
            missing_pct = series.isnull().sum() / len(series) * 100
            self.logger.warning(f"La serie contiene {missing_pct:.1f}% di valori mancanti")
    
    def generate_report(
        self,
        plots_data: Optional[Dict[str, str]] = None,
        report_title: str = None,
        output_filename: str = None,
        format_type: str = "html",
        include_diagnostics: bool = True,
        include_forecast: bool = True,
        forecast_steps: int = 12
    ) -> Path:
        """
        Generate comprehensive Quarto report for the model analysis.
        
        Args:
            plots_data: Dictionary with plot file paths {'plot_name': 'path/to/plot.png'}
            report_title: Custom title for the report
            output_filename: Custom filename for the report
            format_type: Output format ('html', 'pdf', 'docx')
            include_diagnostics: Whether to include model diagnostics
            include_forecast: Whether to include forecast analysis
            forecast_steps: Number of steps to forecast for the report
            
        Returns:
            Path to generated report
            
        Raises:
            ModelTrainingError: If model is not fitted
            ForecastError: If report generation fails
        """
        try:
            from ..reporting import QuartoReportGenerator
            from ..evaluation.metrics import ModelEvaluator
            
            if self.fitted_model is None:
                raise ModelTrainingError("Il modello deve essere addestrato prima di generare il report")
            
            # Set default title
            if report_title is None:
                report_title = f"Analisi Modello ARIMA{self.order}"
            
            # Collect model results
            model_results = {
                'model_type': 'ARIMA',
                'order': self.order,
                'model_info': self.get_model_info(),
                'training_data': {
                    'start_date': str(self.training_metadata.get('training_start', 'N/A')),
                    'end_date': str(self.training_metadata.get('training_end', 'N/A')),
                    'observations': self.training_metadata.get('training_observations', 0)
                }
            }
            
            # Add metrics if training data is available
            if self.training_data is not None and len(self.training_data) > 0:
                evaluator = ModelEvaluator()
                
                # Get in-sample predictions for evaluation
                predictions = self.predict()
                
                # Calculate metrics
                if len(predictions) == len(self.training_data):
                    metrics = evaluator.calculate_forecast_metrics(
                        actual=self.training_data,
                        predicted=predictions
                    )
                    model_results['metrics'] = metrics
                
                # Add diagnostics if requested
                if include_diagnostics:
                    try:
                        diagnostics = evaluator.evaluate_residuals(
                            residuals=self.fitted_model.resid
                        )
                        model_results['diagnostics'] = diagnostics
                    except Exception as e:
                        self.logger.warning(f"Non è stato possibile calcolare i diagnostici: {e}")
            
            # Add forecast if requested
            if include_forecast:
                try:
                    forecast_result = self.forecast(
                        steps=forecast_steps,
                        confidence_intervals=True
                    )
                    if isinstance(forecast_result, dict):
                        model_results['forecast'] = {
                            'steps': forecast_steps,
                            'values': forecast_result['forecast'].tolist(),
                            'confidence_intervals': {
                                'lower': forecast_result['confidence_intervals']['lower'].tolist(),
                                'upper': forecast_result['confidence_intervals']['upper'].tolist()
                            },
                            'index': forecast_result['forecast'].index.astype(str).tolist()
                        }
                    else:
                        model_results['forecast'] = {
                            'steps': forecast_steps,
                            'values': forecast_result.tolist(),
                            'index': forecast_result.index.astype(str).tolist()
                        }
                except Exception as e:
                    self.logger.warning(f"Non è stato possibile generare il forecast per il report: {e}")
            
            # Add technical information
            model_results.update({
                'python_version': f"{np.__version__}",
                'environment': 'ARIMA Forecaster Library',
                'generation_timestamp': pd.Timestamp.now().isoformat()
            })
            
            # Generate report
            generator = QuartoReportGenerator()
            report_path = generator.generate_model_report(
                model_results=model_results,
                plots_data=plots_data,
                report_title=report_title,
                output_filename=output_filename,
                format_type=format_type
            )
            
            return report_path
            
        except ImportError as e:
            raise ForecastError(
                "Moduli di reporting non disponibili. Installa le dipendenze con: pip install 'arima-forecaster[reports]'"
            )
        except Exception as e:
            self.logger.error(f"Generazione report fallita: {e}")
            raise ForecastError(f"Impossibile generare il report: {e}")