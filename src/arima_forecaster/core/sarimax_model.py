"""
Implementazione del modello SARIMAX con supporto per variabili esogene.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union, List
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from ..utils.logger import get_logger
from ..utils.exceptions import ModelTrainingError, ForecastError


class SARIMAXForecaster:
    """
    Previsore SARIMAX con supporto stagionale e variabili esogene.
    
    SARIMAX estende SARIMA includendo variabili esterne (eXogenous) che possono
    aiutare a spiegare la serie temporale target:
    - Variabili economiche (PIL, inflazione)  
    - Variabili meteorologiche (temperatura, precipitazioni)
    - Variabili di marketing (spesa pubblicitaria, promozioni)
    - Altre serie temporali correlate
    """
    
    def __init__(
        self, 
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12),
        exog_names: Optional[List[str]] = None,
        trend: Optional[str] = None
    ):
        """
        Inizializza il previsore SARIMAX.
        
        Args:
            order: Ordine ARIMA non stagionale (p, d, q)
            seasonal_order: Ordine ARIMA stagionale (P, D, Q, s)
            exog_names: Lista dei nomi delle variabili esogene
            trend: Parametro di trend ('n', 'c', 't', 'ct')
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.exog_names = exog_names or []
        self.trend = trend
        self.model = None
        self.fitted_model = None
        self.training_data = None
        self.training_exog = None
        self.training_metadata = {}
        self.logger = get_logger(__name__)
        
    def fit(
        self, 
        series: pd.Series,
        exog: Optional[pd.DataFrame] = None,
        validate_input: bool = True,
        **fit_kwargs
    ) -> 'SARIMAXForecaster':
        """
        Addestra il modello SARIMAX sui dati delle serie temporali e variabili esogene.
        
        Args:
            series: Dati delle serie temporali da addestrare
            exog: DataFrame con variabili esogene (stesse righe di series)
            validate_input: Se validare i dati di input
            **fit_kwargs: Argomenti aggiuntivi per l'addestramento del modello
            
        Returns:
            Self per concatenamento dei metodi
            
        Raises:
            ModelTrainingError: Se l'addestramento del modello fallisce
        """
        try:
            exog_info = f" con {exog.shape[1] if exog is not None else 0} variabili esogene" if exog is not None else " senza variabili esogene"
            self.logger.info(
                f"Fitting SARIMAX{self.order}x{self.seasonal_order} model "
                f"a {len(series)} osservazioni{exog_info}"
            )
            
            if validate_input:
                self._validate_series(series)
                if exog is not None:
                    self._validate_exog(series, exog)
                self._validate_seasonal_parameters(series)
            
            # Memorizza i dati di addestramento e i metadati
            self.training_data = series.copy()
            self.training_exog = exog.copy() if exog is not None else None
            
            # Aggiorna nomi variabili esogene se fornite
            if exog is not None and self.exog_names is None:
                self.exog_names = list(exog.columns)
            
            self.training_metadata = {
                'training_start': series.index.min(),
                'training_end': series.index.max(), 
                'training_observations': len(series),
                'order': self.order,
                'seasonal_order': self.seasonal_order,
                'trend': self.trend,
                'exog_names': self.exog_names,
                'n_exog': len(self.exog_names) if self.exog_names else 0
            }
            
            # Crea e addestra il modello
            self.model = SARIMAX(
                series, 
                exog=exog,
                order=self.order,
                seasonal_order=self.seasonal_order,
                trend=self.trend
            )
            self.fitted_model = self.model.fit(**fit_kwargs)
            
            # Registra il riepilogo del modello
            self.logger.info("Modello SARIMAX addestrato con successo")
            self.logger.info(f"AIC: {self.fitted_model.aic:.2f}")
            self.logger.info(f"BIC: {self.fitted_model.bic:.2f}")
            if self.exog_names:
                self.logger.info(f"Variabili esogene utilizzate: {', '.join(self.exog_names)}")
            
            return self
            
        except Exception as e:
            self.logger.error(f"Addestramento modello SARIMAX fallito: {e}")
            raise ModelTrainingError(f"Impossibile addestrare il modello SARIMAX: {e}")
    
    def forecast(
        self, 
        steps: int,
        exog_future: Optional[pd.DataFrame] = None,
        confidence_intervals: bool = True,
        alpha: float = 0.05,
        return_conf_int: bool = False
    ) -> Union[pd.Series, Tuple[pd.Series, pd.DataFrame]]:
        """
        Genera previsioni dal modello SARIMAX addestrato.
        
        Args:
            steps: Numero di passaggi da prevedere
            exog_future: DataFrame con valori futuri delle variabili esogene (steps righe)
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
                raise ForecastError("Il modello SARIMAX deve essere addestrato prima della previsione")
            
            # Valida variabili esogene se necessarie
            if self.exog_names and exog_future is None:
                raise ForecastError(
                    f"Il modello richiede variabili esogene per la previsione: {', '.join(self.exog_names)}"
                )
            
            if exog_future is not None:
                self._validate_exog_future(exog_future, steps)
            
            self.logger.info(f"Generazione previsione SARIMAX a {steps} passaggi")
            
            # Genera previsione
            forecast_result = self.fitted_model.get_forecast(steps=steps, exog=exog_future, alpha=alpha)
            forecast_values = forecast_result.predicted_mean
            
            if confidence_intervals:
                conf_int = forecast_result.conf_int()
            else:
                conf_int = None
            
            # Crea indice previsione
            last_date = self.training_data.index[-1]
            if isinstance(last_date, pd.Timestamp):
                freq = pd.infer_freq(self.training_data.index)
                if freq:
                    try:
                        # Converte frequenza stringa in DateOffset e aggiunge al timestamp
                        freq_offset = pd.tseries.frequencies.to_offset(freq)
                        forecast_index = pd.date_range(
                            start=last_date + freq_offset,
                            periods=steps,
                            freq=freq
                        )
                    except Exception:
                        # Fallback: usa frequenza giornaliera
                        forecast_index = pd.date_range(
                            start=last_date + pd.Timedelta(days=1),
                            periods=steps,
                            freq='D'
                        )
                else:
                    # Fallback: usa frequenza giornaliera se non può essere inferita
                    forecast_index = pd.date_range(
                        start=last_date + pd.Timedelta(days=1),
                        periods=steps,
                        freq='D'
                    )
            else:
                forecast_index = range(len(self.training_data), len(self.training_data) + steps)
            
            forecast_series = pd.Series(forecast_values, index=forecast_index, name='forecast')
            
            self.logger.info(
                f"Previsione SARIMAX generata: {forecast_series.iloc[0]:.2f} a {forecast_series.iloc[-1]:.2f}"
            )
            
            if return_conf_int and conf_int is not None:
                conf_int.index = forecast_index
                return forecast_series, conf_int
            else:
                return forecast_series
                
        except Exception as e:
            self.logger.error(f"Previsione SARIMAX fallita: {e}")
            raise ForecastError(f"Impossibile generare il forecast SARIMAX: {e}")
    
    def predict(
        self, 
        start: Optional[Union[int, str, pd.Timestamp]] = None,
        end: Optional[Union[int, str, pd.Timestamp]] = None,
        exog: Optional[pd.DataFrame] = None,
        dynamic: bool = False
    ) -> pd.Series:
        """
        Genera predizioni in-sample e out-of-sample.
        
        Args:
            start: Inizio del periodo di predizione
            end: Fine del periodo di predizione  
            exog: Variabili esogene per il periodo di predizione
            dynamic: Se usare predizione dinamica
            
        Returns:
            Serie di predizioni
        """
        try:
            if self.fitted_model is None:
                raise ForecastError("Il modello SARIMAX deve essere addestrato prima della predizione")
            
            predictions = self.fitted_model.predict(start=start, end=end, exog=exog, dynamic=dynamic)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Predizione SARIMAX fallita: {e}")
            raise ForecastError(f"Impossibile generare le predizioni SARIMAX: {e}")
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Salva il modello SARIMAX addestrato su disco.
        
        Args:
            filepath: Percorso per salvare il modello
        """
        try:
            if self.fitted_model is None:
                raise ModelTrainingError("Nessun modello SARIMAX addestrato da salvare")
            
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Salva usando il metodo integrato di statsmodels
            self.fitted_model.save(str(filepath))
            
            # Salva anche i metadati
            metadata_path = filepath.with_suffix('.metadata.pkl')
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'order': self.order,
                    'seasonal_order': self.seasonal_order,
                    'exog_names': self.exog_names,
                    'trend': self.trend,
                    'training_metadata': self.training_metadata
                }, f)
            
            # Salva i dati di addestramento esogeni se presenti
            if self.training_exog is not None:
                exog_path = filepath.with_suffix('.exog.pkl')
                with open(exog_path, 'wb') as f:
                    pickle.dump(self.training_exog, f)
            
            self.logger.info(f"Modello SARIMAX salvato in {filepath}")
            
        except Exception as e:
            self.logger.error(f"Impossibile salvare il modello SARIMAX: {e}")
            raise ModelTrainingError(f"Impossibile salvare il modello SARIMAX: {e}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'SARIMAXForecaster':
        """
        Carica modello SARIMAX addestrato da disco.
        
        Args:
            filepath: Percorso del modello salvato
            
        Returns:
            Istanza SARIMAXForecaster caricata
        """
        try:
            filepath = Path(filepath)
            
            # Carica il modello addestrato
            fitted_model = SARIMAXResults.load(str(filepath))
            
            # Carica metadati se disponibili
            metadata_path = filepath.with_suffix('.metadata.pkl')
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                order = metadata.get('order', (1, 1, 1))
                seasonal_order = metadata.get('seasonal_order', (1, 1, 1, 12))
                exog_names = metadata.get('exog_names', [])
                trend = metadata.get('trend', None)
                training_metadata = metadata.get('training_metadata', {})
            else:
                order = (1, 1, 1)  # Ordine predefinito
                seasonal_order = (1, 1, 1, 12)  # Ordine stagionale predefinito
                exog_names = []
                trend = None
                training_metadata = {}
            
            # Carica dati esogeni se disponibili
            training_exog = None
            exog_path = filepath.with_suffix('.exog.pkl')
            if exog_path.exists():
                with open(exog_path, 'rb') as f:
                    training_exog = pickle.load(f)
            
            # Crea istanza e popola
            instance = cls(order=order, seasonal_order=seasonal_order, exog_names=exog_names, trend=trend)
            instance.fitted_model = fitted_model
            instance.training_exog = training_exog
            instance.training_metadata = training_metadata
            
            instance.logger.info(f"Modello SARIMAX caricato da {filepath}")
            
            return instance
            
        except Exception as e:
            logger = get_logger(__name__)
            logger.error(f"Impossibile caricare il modello SARIMAX: {e}")
            raise ModelTrainingError(f"Impossibile caricare il modello SARIMAX: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Ottieni informazioni complete del modello SARIMAX.
        
        Returns:
            Dizionario con informazioni del modello
        """
        if self.fitted_model is None:
            return {'status': 'not_fitted'}
        
        info = {
            'status': 'fitted',
            'model_type': 'SARIMAX',
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'trend': self.trend,
            'exog_names': self.exog_names,
            'n_exog': len(self.exog_names),
            'aic': self.fitted_model.aic,
            'bic': self.fitted_model.bic,
            'hqic': self.fitted_model.hqic,
            'llf': self.fitted_model.llf,
            'n_observations': self.fitted_model.nobs,
            'params': dict(self.fitted_model.params),
            'training_metadata': self.training_metadata
        }
        
        # Aggiungi informazioni sui parametri delle variabili esogene se presenti
        if self.exog_names and hasattr(self.fitted_model, 'params'):
            exog_params = {}
            for name in self.exog_names:
                if name in self.fitted_model.params.index:
                    exog_params[name] = {
                        'coefficient': float(self.fitted_model.params[name]),
                        'pvalue': float(self.fitted_model.pvalues[name]) if hasattr(self.fitted_model, 'pvalues') else None,
                        'tvalue': float(self.fitted_model.tvalues[name]) if hasattr(self.fitted_model, 'tvalues') else None
                    }
            info['exog_params'] = exog_params
        
        return info
    
    def get_seasonal_decomposition(self) -> Dict[str, pd.Series]:
        """
        Ottieni decomposizione stagionale del modello addestrato.
        
        Returns:
            Dizionario con componenti di decomposizione
        """
        if self.fitted_model is None:
            raise ForecastError("Il modello SARIMAX deve essere addestrato prima della decomposizione")
        
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            decomposition = seasonal_decompose(
                self.training_data, 
                model='additive',
                period=self.seasonal_order[3]  # periodo stagionale
            )
            
            return {
                'observed': decomposition.observed,
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid
            }
            
        except Exception as e:
            self.logger.error(f"Decomposizione stagionale fallita: {e}")
            raise ForecastError(f"Impossibile eseguire la decomposizione stagionale: {e}")
    
    def get_exog_importance(self) -> pd.DataFrame:
        """
        Ottieni l'importanza delle variabili esogene nel modello.
        
        Returns:
            DataFrame con coefficienti, p-values e significatività delle variabili esogene
        """
        if self.fitted_model is None:
            raise ForecastError("Il modello SARIMAX deve essere addestrato per analizzare le variabili esogene")
        
        if not self.exog_names:
            return pd.DataFrame(columns=['coefficient', 'pvalue', 'tvalue', 'significant'])
        
        try:
            importance_data = []
            
            for name in self.exog_names:
                if name in self.fitted_model.params.index:
                    coeff = float(self.fitted_model.params[name])
                    pvalue = float(self.fitted_model.pvalues[name]) if hasattr(self.fitted_model, 'pvalues') else None
                    tvalue = float(self.fitted_model.tvalues[name]) if hasattr(self.fitted_model, 'tvalues') else None
                    significant = pvalue < 0.05 if pvalue is not None else None
                    
                    importance_data.append({
                        'variable': name,
                        'coefficient': coeff,
                        'pvalue': pvalue,
                        'tvalue': tvalue,
                        'significant': significant,
                        'abs_coefficient': abs(coeff)
                    })
            
            df = pd.DataFrame(importance_data)
            if not df.empty:
                df = df.sort_values('abs_coefficient', ascending=False)
                df = df.drop('abs_coefficient', axis=1)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Analisi importanza variabili esogene fallita: {e}")
            raise ForecastError(f"Impossibile analizzare le variabili esogene: {e}")
    
    def _validate_series(self, series: pd.Series) -> None:
        """
        Valida serie temporale di input.
        
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
            raise ModelTrainingError("La serie non può essere tutta NaN")
        
        if len(series) < 10:
            self.logger.warning("La serie ha meno di 10 osservazioni, il modello potrebbe essere inaffidabile")
        
        if series.isnull().any():
            missing_pct = series.isnull().sum() / len(series) * 100
            self.logger.warning(f"La serie contiene {missing_pct:.1f}% valori mancanti")
    
    def _validate_exog(self, series: pd.Series, exog: pd.DataFrame) -> None:
        """
        Valida variabili esogene per l'addestramento.
        
        Args:
            series: Serie temporale target
            exog: DataFrame delle variabili esogene
            
        Raises:
            ModelTrainingError: Se la validazione fallisce
        """
        if not isinstance(exog, pd.DataFrame):
            raise ModelTrainingError("Le variabili esogene devono essere un pandas DataFrame")
        
        if len(exog) != len(series):
            raise ModelTrainingError(
                f"Le variabili esogene ({len(exog)} righe) devono avere la stessa lunghezza "
                f"della serie target ({len(series)} righe)"
            )
        
        if exog.empty:
            raise ModelTrainingError("Il DataFrame delle variabili esogene non può essere vuoto")
        
        # Controlla valori mancanti
        missing_cols = exog.columns[exog.isnull().any()].tolist()
        if missing_cols:
            self.logger.warning(f"Variabili esogene con valori mancanti: {missing_cols}")
        
        # Controlla variabili numeriche
        non_numeric_cols = exog.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            self.logger.warning(f"Variabili esogene non numeriche (potrebbero causare problemi): {non_numeric_cols}")
        
        # Aggiorna nomi variabili se non specificati
        if self.exog_names is None or len(self.exog_names) == 0:
            self.exog_names = list(exog.columns)
        elif set(self.exog_names) != set(exog.columns):
            self.logger.warning(
                f"Nomi variabili specificati ({self.exog_names}) diversi da colonne DataFrame ({list(exog.columns)})"
            )
    
    def _validate_exog_future(self, exog_future: pd.DataFrame, steps: int) -> None:
        """
        Valida variabili esogene per la previsione.
        
        Args:
            exog_future: DataFrame con valori futuri delle variabili esogene
            steps: Numero di passi di previsione
            
        Raises:
            ForecastError: Se la validazione fallisce
        """
        if not isinstance(exog_future, pd.DataFrame):
            raise ForecastError("Le variabili esogene future devono essere un pandas DataFrame")
        
        if len(exog_future) != steps:
            raise ForecastError(
                f"Le variabili esogene future ({len(exog_future)} righe) devono avere "
                f"lo stesso numero di righe dei passi di previsione ({steps})"
            )
        
        # Verifica che le colonne corrispondano
        if set(exog_future.columns) != set(self.exog_names):
            missing_cols = set(self.exog_names) - set(exog_future.columns)
            extra_cols = set(exog_future.columns) - set(self.exog_names)
            error_msg = []
            if missing_cols:
                error_msg.append(f"Variabili mancanti: {missing_cols}")
            if extra_cols:
                error_msg.append(f"Variabili extra: {extra_cols}")
            raise ForecastError(f"Variabili esogene non corrispondenti. {' '.join(error_msg)}")
        
        # Controlla valori mancanti
        if exog_future.isnull().any().any():
            raise ForecastError("Le variabili esogene future non possono contenere valori mancanti")
    
    def _validate_seasonal_parameters(self, series: pd.Series) -> None:
        """
        Valida parametri stagionali contro i dati.
        
        Args:
            series: Serie su cui validare
            
        Raises:
            ModelTrainingError: Se la validazione fallisce
        """
        seasonal_period = self.seasonal_order[3]
        
        if seasonal_period <= 1:
            raise ModelTrainingError("Il periodo stagionale deve essere maggiore di 1")
        
        if len(series) < 2 * seasonal_period:
            self.logger.warning(
                f"Lunghezza serie ({len(series)}) è inferiore a 2 periodi stagionali "
                f"({2 * seasonal_period}). Il modello potrebbe essere inaffidabile."
            )
        
        # Controlla se il periodo stagionale ha senso per la frequenza dei dati
        if hasattr(series.index, 'freq') and series.index.freq is not None:
            freq = series.index.freq
            if 'D' in str(freq) and seasonal_period not in [7, 30, 365]:
                self.logger.warning(
                    f"Dati giornalieri con periodo stagionale {seasonal_period} potrebbero non essere appropriati. "
                    "Considera 7 (settimanale), 30 (mensile), o 365 (annuale)."
                )
            elif 'M' in str(freq) and seasonal_period != 12:
                self.logger.warning(
                    f"Dati mensili con periodo stagionale {seasonal_period} potrebbero non essere appropriati. "
                    "Considera 12 (annuale)."
                )
    
    def generate_report(
        self,
        plots_data: Optional[Dict[str, str]] = None,
        report_title: str = None,
        output_filename: str = None,
        format_type: str = "html",
        include_diagnostics: bool = True,
        include_forecast: bool = True,
        forecast_steps: int = 12,
        include_seasonal_decomposition: bool = True,
        include_exog_analysis: bool = True,
        exog_future: Optional[pd.DataFrame] = None,
        precomputed_forecast: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Generate comprehensive Quarto report for the SARIMAX model analysis.
        
        Args:
            plots_data: Dictionary with plot file paths {'plot_name': 'path/to/plot.png'}
            report_title: Custom title for the report
            output_filename: Custom filename for the report
            format_type: Output format ('html', 'pdf', 'docx')
            include_diagnostics: Whether to include model diagnostics
            include_forecast: Whether to include forecast analysis
            forecast_steps: Number of steps to forecast for the report
            include_seasonal_decomposition: Whether to include seasonal decomposition
            include_exog_analysis: Whether to include exogenous variables analysis
            
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
                exog_info = f" con {len(self.exog_names)} variabili esogene" if self.exog_names else ""
                report_title = f"Analisi Modello SARIMAX{self.order}x{self.seasonal_order}{exog_info}"
            
            # Collect model results
            model_results = {
                'model_type': 'SARIMAX',
                'order': self.order,
                'seasonal_order': self.seasonal_order,
                'trend': self.trend,
                'exog_names': self.exog_names,
                'n_exog': len(self.exog_names),
                'model_info': self.get_model_info(),
                'training_data': {
                    'start_date': str(self.training_metadata.get('training_start', 'N/A')),
                    'end_date': str(self.training_metadata.get('training_end', 'N/A')),
                    'observations': self.training_metadata.get('training_observations', 0)
                }
            }
            
            # Add exogenous variables analysis if requested
            if include_exog_analysis and self.exog_names:
                try:
                    exog_importance = self.get_exog_importance()
                    if not exog_importance.empty:
                        model_results['exog_analysis'] = {
                            'importance_table': exog_importance.to_dict('records'),
                            'significant_vars': exog_importance[exog_importance['significant'] == True]['variable'].tolist(),
                            'n_significant': int(exog_importance['significant'].sum()) if 'significant' in exog_importance else 0
                        }
                except Exception as e:
                    self.logger.warning(f"Non è stato possibile includere l'analisi delle variabili esogene: {e}")
            
            # Add seasonal decomposition if requested
            if include_seasonal_decomposition and self.training_data is not None:
                try:
                    decomposition = self.get_seasonal_decomposition()
                    if decomposition is not None:
                        model_results['seasonal_decomposition'] = {
                            'trend_mean': float(np.nanmean(decomposition['trend'].dropna())),
                            'seasonal_amplitude': float(np.nanstd(decomposition['seasonal'].dropna())),
                            'residual_variance': float(np.nanvar(decomposition['residual'].dropna()))
                        }
                except Exception as e:
                    self.logger.warning(f"Non è stato possibile includere la decomposizione stagionale: {e}")
            
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
                    # Usa forecast precompilato se disponibile
                    if precomputed_forecast:
                        self.logger.info("Usando forecast precompilato per il report")
                        forecast_series = precomputed_forecast.get('forecast')
                        conf_int = precomputed_forecast.get('conf_int')
                        confidence_level = precomputed_forecast.get('confidence_level', 0.95)
                        
                        if forecast_series is not None:
                            model_results['forecast'] = {
                                'steps': len(forecast_series),
                                'values': forecast_series.tolist(),
                                'index': forecast_series.index.astype(str).tolist(),
                                'confidence_level': confidence_level
                            }
                            
                            if conf_int is not None:
                                model_results['forecast']['confidence_intervals'] = {
                                    'lower': conf_int.iloc[:, 0].tolist(),
                                    'upper': conf_int.iloc[:, 1].tolist()
                                }
                        
                    # Altrimenti genera nuovo forecast se possibile
                    elif self.exog_names and exog_future is None:
                        self.logger.warning("Forecast nel report non generato: SARIMAX richiede variabili esogene future")
                        model_results['forecast_note'] = "Forecast non disponibile: necessarie variabili esogene future"
                        
                    # Genera nuovo forecast
                    else:
                        kwargs = {'steps': forecast_steps, 'confidence_intervals': True}
                        if self.exog_names and exog_future is not None:
                            kwargs['exog_future'] = exog_future
                            
                        forecast_result = self.forecast(**kwargs)
                        
                        if isinstance(forecast_result, tuple):
                            forecast_series, conf_int = forecast_result
                            model_results['forecast'] = {
                                'steps': forecast_steps,
                                'values': forecast_series.tolist(),
                                'confidence_intervals': {
                                    'lower': conf_int.iloc[:, 0].tolist(),
                                    'upper': conf_int.iloc[:, 1].tolist()
                                },
                                'index': forecast_series.index.astype(str).tolist()
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