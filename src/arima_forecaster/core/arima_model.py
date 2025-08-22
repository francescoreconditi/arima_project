"""
Implementazione del modello ARIMA core con funzionalità avanzate.
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
    Previsore ARIMA avanzato con funzionalità complete.
    """
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1)):
        """
        Inizializza il previsore ARIMA.
        
        Args:
            order: Ordine ARIMA (p, d, q)
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
        Addestra il modello ARIMA sui dati delle serie temporali.
        
        Args:
            series: Dati delle serie temporali da addestrare
            validate_input: Se validare i dati di input
            **fit_kwargs: Argomenti aggiuntivi per l'addestramento del modello
            
        Returns:
            Self per concatenamento dei metodi
            
        Raises:
            ModelTrainingError: Se l'addestramento del modello fallisce
        """
        try:
            self.logger.info(f"Fitting ARIMA{self.order} model to {len(series)} observations")
            
            if validate_input:
                self._validate_series(series)
            
            # Memorizza i dati di addestramento e i metadati
            self.training_data = series.copy()
            self.training_metadata = {
                'training_start': series.index.min(),
                'training_end': series.index.max(), 
                'training_observations': len(series),
                'order': self.order
            }
            
            # Crea e addestra il modello
            self.model = ARIMA(series, order=self.order)
            self.fitted_model = self.model.fit(**fit_kwargs)
            
            # Registra il riepilogo del modello
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
        Genera previsioni dal modello addestrato.
        
        Args:
            steps: Numero di passi da prevedere
            confidence_intervals: Se calcolare gli intervalli di confidenza
            alpha: Livello alpha per gli intervalli di confidenza (1-alpha = livello di confidenza)
            return_conf_int: Se restituire gli intervalli di confidenza
            
        Returns:
            Serie di previsioni, opzionalmente con intervalli di confidenza
            
        Raises:
            ForecastError: Se la previsione fallisce
        """
        try:
            if self.fitted_model is None:
                raise ForecastError("Il modello deve essere addestrato prima del forecasting")
            
            self.logger.info(f"Generazione forecast a {steps} passi")
            
            # Genera previsione
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
            
            # Crea indice di previsione
            last_date = self.training_data.index[-1]
            if isinstance(last_date, pd.Timestamp):
                freq = pd.infer_freq(self.training_data.index)
                if freq is None:
                    # Fallback: calcola la frequenza dalle prime due date
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
                    )[1:]  # Salta il primo elemento per evitare sovrapposizioni
            else:
                forecast_index = range(len(self.training_data), len(self.training_data) + steps)
            
            forecast_series = pd.Series(forecast_values, index=forecast_index, name='forecast')
            
            self.logger.info(f"Forecast generato: {forecast_series.iloc[0]:.2f} a {forecast_series.iloc[-1]:.2f}")
            
            # Il formato di ritorno dipende dai parametri
            if confidence_intervals and conf_int is not None:
                conf_int.index = forecast_index
                if return_conf_int:
                    return forecast_series, conf_int
                else:
                    # Restituisce formato dizionario per compatibilità con gli esempi
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
        Genera predizioni in-sample e out-of-sample.
        
        Args:
            start: Inizio del periodo di predizione
            end: Fine del periodo di predizione  
            dynamic: Se usare la predizione dinamica
            
        Returns:
            Serie di predizioni
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
        Salva il modello addestrato su disco.
        
        Args:
            filepath: Percorso dove salvare il modello
        """
        try:
            if self.fitted_model is None:
                raise ModelTrainingError("Nessun modello addestrato da salvare")
            
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Salva usando il metodo built-in di statsmodels
            self.fitted_model.save(str(filepath))
            
            # Salva anche i metadati
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
        Carica il modello addestrato dal disco.
        
        Args:
            filepath: Percorso del modello salvato
            
        Returns:
            Istanza ARIMAForecaster caricata
        """
        try:
            filepath = Path(filepath)
            
            # Carica il modello addestrato
            fitted_model = ARIMAResults.load(str(filepath))
            
            # Carica i metadati se disponibili
            metadata_path = filepath.with_suffix('.metadata.pkl')
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                order = metadata.get('order', (1, 1, 1))
                training_metadata = metadata.get('training_metadata', {})
            else:
                order = (1, 1, 1)  # Ordine di default
                training_metadata = {}
            
            # Crea istanza e popola
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
        Ottiene informazioni complete sul modello.
        
        Returns:
            Dizionario con informazioni sul modello
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
        Valida la serie temporale di input.
        
        Args:
            series: Serie da validare
            
        Raises:
            ModelTrainingError: Se la validazione fallisce
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
        Genera report Quarto completo per l'analisi del modello.
        
        Args:
            plots_data: Dizionario con percorsi dei file di plot {'nome_plot': 'percorso/plot.png'}
            report_title: Titolo personalizzato per il report
            output_filename: Nome file personalizzato per il report
            format_type: Formato di output ('html', 'pdf', 'docx')
            include_diagnostics: Se includere i diagnostici del modello
            include_forecast: Se includere l'analisi delle previsioni
            forecast_steps: Numero di passi da prevedere per il report
            
        Returns:
            Percorso del report generato
            
        Raises:
            ModelTrainingError: Se il modello non è addestrato
            ForecastError: Se la generazione del report fallisce
        """
        try:
            from ..reporting import QuartoReportGenerator
            from ..evaluation.metrics import ModelEvaluator
            
            if self.fitted_model is None:
                raise ModelTrainingError("Il modello deve essere addestrato prima di generare il report")
            
            # Imposta titolo di default
            if report_title is None:
                report_title = f"Analisi Modello ARIMA{self.order}"
            
            # Raccoglie i risultati del modello
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
            
            # Aggiungi metriche se i dati di addestramento sono disponibili
            if self.training_data is not None and len(self.training_data) > 0:
                evaluator = ModelEvaluator()
                
                # Ottiene predizioni in-sample per la valutazione
                predictions = self.predict()
                
                # Calcola metriche
                if len(predictions) == len(self.training_data):
                    metrics = evaluator.calculate_forecast_metrics(
                        actual=self.training_data,
                        predicted=predictions
                    )
                    model_results['metrics'] = metrics
                
                # Aggiungi diagnostici se richiesto
                if include_diagnostics:
                    try:
                        diagnostics = evaluator.evaluate_residuals(
                            residuals=self.fitted_model.resid
                        )
                        model_results['diagnostics'] = diagnostics
                    except Exception as e:
                        self.logger.warning(f"Non è stato possibile calcolare i diagnostici: {e}")
            
            # Aggiungi previsione se richiesto
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
            
            # Aggiungi informazioni tecniche
            model_results.update({
                'python_version': f"{np.__version__}",
                'environment': 'ARIMA Forecaster Library',
                'generation_timestamp': pd.Timestamp.now().isoformat()
            })
            
            # Genera report
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