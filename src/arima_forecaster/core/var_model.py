"""
Modello Vector Autoregression (VAR) per la previsione di serie temporali multivariate.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.var_model import VARResults
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from ..utils.logger import get_logger
from ..utils.exceptions import ModelTrainingError, ForecastError


class VARForecaster:
    """
    Previsore Vector Autoregression per serie temporali multivariate.
    """
    
    def __init__(self, maxlags: Optional[int] = None, ic: str = 'aic'):
        """
        Inizializza il previsore VAR.
        
        Args:
            maxlags: Numero massimo di lag da considerare
            ic: Criterio di informazione per la selezione lag ('aic', 'bic', 'hqic', 'fpe')
        """
        self.maxlags = maxlags
        self.ic = ic
        self.model = None
        self.fitted_model = None
        self.training_data = None
        self.training_metadata = {}
        self.selected_lag = None
        self.logger = get_logger(__name__)
        
        if self.ic not in ['aic', 'bic', 'hqic', 'fpe']:
            raise ValueError("ic deve essere uno tra 'aic', 'bic', 'hqic', 'fpe'")
    
    def fit(
        self, 
        data: pd.DataFrame,
        validate_input: bool = True,
        trend: str = 'c',
        **fit_kwargs
    ) -> 'VARForecaster':
        """
        Addestra il modello VAR sui dati di serie temporali multivariate.
        
        Args:
            data: DataFrame con colonne di serie temporali multiple
            validate_input: Se validare i dati di input
            trend: Parametro di trend ('c', 'ct', 'ctt', 'n')
            **fit_kwargs: Argomenti aggiuntivi per l'addestramento del modello
            
        Returns:
            Self per concatenamento dei metodi
            
        Raises:
            ModelTrainingError: Se l'addestramento del modello fallisce
        """
        try:
            self.logger.info(f"Addestramento modello VAR su {data.shape[0]} osservazioni con {data.shape[1]} variabili")
            
            if validate_input:
                self._validate_data(data)
            
            # Memorizza dati di addestramento e metadati
            self.training_data = data.copy()
            self.training_metadata = {
                'training_start': data.index.min(),
                'training_end': data.index.max(), 
                'training_observations': len(data),
                'n_variables': data.shape[1],
                'variable_names': list(data.columns),
                'maxlags': self.maxlags,
                'ic': self.ic,
                'trend': trend
            }
            
            # Crea modello VAR
            self.model = VAR(data)
            
            # Seleziona lag ottimale se non specificato
            if self.maxlags is None:
                # Usa selezione automatica del lag
                max_lag_test = min(12, len(data) // 4)  # Default conservativo
                lag_selection = self.model.select_order(maxlags=max_lag_test)
                self.selected_lag = getattr(lag_selection, self.ic)
                self.logger.info(f"Ordine lag auto-selezionato: {self.selected_lag} usando {self.ic.upper()}")
            else:
                self.selected_lag = self.maxlags
                self.logger.info(f"Utilizzo ordine lag specificato: {self.selected_lag}")
            
            # Addestra il modello
            self.fitted_model = self.model.fit(
                maxlags=self.selected_lag,
                trend=trend,
                **fit_kwargs
            )
            
            # Registra riepilogo del modello
            self.logger.info("Modello VAR addestrato con successo")
            self.logger.info(f"Lag order: {self.fitted_model.k_ar}")
            self.logger.info(f"AIC: {self.fitted_model.aic:.2f}")
            self.logger.info(f"BIC: {self.fitted_model.bic:.2f}")
            
            return self
            
        except Exception as e:
            self.logger.error(f"Addestramento modello VAR fallito: {e}")
            raise ModelTrainingError(f"Impossibile addestrare il modello VAR: {e}")
    
    def forecast(
        self, 
        steps: int,
        alpha: float = 0.05
    ) -> Dict[str, pd.DataFrame]:
        """
        Genera previsioni dal modello VAR addestrato.
        
        Args:
            steps: Numero di passaggi da prevedere
            alpha: Livello alpha per gli intervalli di confidenza
            
        Returns:
            Dizionario contenente previsioni e intervalli di confidenza
            
        Raises:
            ForecastError: Se la previsione fallisce
        """
        try:
            if self.fitted_model is None:
                raise ForecastError("Il modello VAR deve essere addestrato prima della previsione")
            
            self.logger.info(f"Generazione previsione VAR a {steps} passaggi")
            
            # Genera previsione
            forecast_result = self.fitted_model.forecast(
                y=self.training_data.values[-self.fitted_model.k_ar:],
                steps=steps
            )
            
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
            
            # Crea DataFrame delle previsioni
            forecast_df = pd.DataFrame(
                forecast_result,
                index=forecast_index,
                columns=self.training_data.columns
            )
            
            # Ottieni intervalli di confidenza delle previsioni
            try:
                conf_int = self.fitted_model.forecast_interval(
                    y=self.training_data.values[-self.fitted_model.k_ar:],
                    steps=steps,
                    alpha=alpha
                )
                
                # Gestisce diversi formati di intervalli di confidenza
                if isinstance(conf_int, tuple) and len(conf_int) == 2:
                    # Caso: conf_int è una tupla di (inferiore, superiore)
                    lower_bounds = pd.DataFrame(
                        conf_int[0],
                        index=forecast_index,
                        columns=self.training_data.columns
                    )
                    upper_bounds = pd.DataFrame(
                        conf_int[1],
                        index=forecast_index,
                        columns=self.training_data.columns
                    )
                elif hasattr(conf_int, 'shape') and len(conf_int.shape) == 3:
                    # Caso: conf_int è un array 3D
                    lower_bounds = pd.DataFrame(
                        conf_int[:, :, 0],
                        index=forecast_index,
                        columns=self.training_data.columns
                    )
                    upper_bounds = pd.DataFrame(
                        conf_int[:, :, 1],
                        index=forecast_index,
                        columns=self.training_data.columns
                    )
                else:
                    # Fallback: crea intervalli di confidenza semplici
                    forecast_std = forecast_df.std()
                    multiplier = 1.96  # CI approssimativo al 95%
                    lower_bounds = forecast_df - multiplier * forecast_std
                    upper_bounds = forecast_df + multiplier * forecast_std
                    
            except Exception as e:
                self.logger.warning(f"Impossibile generare intervalli di confidenza: {e}")
                # Fallback: crea intervalli di confidenza semplici basati sulla varianza delle previsioni
                forecast_std = forecast_df.std()
                multiplier = 1.96  # CI approssimativo al 95%
                lower_bounds = forecast_df - multiplier * forecast_std
                upper_bounds = forecast_df + multiplier * forecast_std
            
            self.logger.info("Previsione VAR generata con successo")
            
            return {
                'forecast': forecast_df,
                'lower_bounds': lower_bounds,
                'upper_bounds': upper_bounds,
                'confidence_level': 1 - alpha
            }
                
        except Exception as e:
            self.logger.error(f"Previsione VAR fallita: {e}")
            raise ForecastError(f"Impossibile generare il forecast VAR: {e}")
    
    def impulse_response(
        self, 
        periods: int = 20,
        impulse: Optional[str] = None,
        response: Optional[str] = None,
        orthogonalized: bool = True
    ) -> pd.DataFrame:
        """
        Calcola le funzioni di risposta agli impulsi.
        
        Args:
            periods: Numero di periodi per la risposta agli impulsi
            impulse: Variabile a cui applicare l'impulso (None per tutte)
            response: Variabile da cui misurare la risposta (None per tutte)
            orthogonalized: Se usare impulsi ortogonalizzati
            
        Returns:
            DataFrame con le funzioni di risposta agli impulsi
        """
        try:
            if self.fitted_model is None:
                raise ForecastError("Il modello VAR deve essere addestrato prima dell'analisi della risposta agli impulsi")
            
            irf = self.fitted_model.irf(periods=periods)
            
            if orthogonalized:
                irf_data = irf.orth_irfs
            else:
                irf_data = irf.irfs
            
            # Crea MultiIndex per colonne (impulso -> risposta)
            variables = self.training_data.columns
            columns = pd.MultiIndex.from_product(
                [variables, variables],
                names=['impulse', 'response']
            )
            
            # Rimodella i dati per DataFrame
            n_vars = len(variables)
            reshaped_data = irf_data.reshape(periods, n_vars * n_vars)
            
            irf_df = pd.DataFrame(
                reshaped_data,
                columns=columns,
                index=range(periods)
            )
            
            # Filtra per impulso/risposta specifici se richiesto
            if impulse is not None and response is not None:
                return irf_df[(impulse, response)]
            elif impulse is not None:
                return irf_df[impulse]
            elif response is not None:
                return irf_df.xs(response, axis=1, level='response')
            else:
                return irf_df
                
        except Exception as e:
            self.logger.error(f"Analisi risposta agli impulsi fallita: {e}")
            raise ForecastError(f"Impossibile calcolare la risposta agli impulsi: {e}")
    
    def forecast_error_variance_decomposition(
        self, 
        periods: int = 20,
        normalize: bool = True
    ) -> pd.DataFrame:
        """
        Calcola la decomposizione della varianza dell'errore di previsione.
        
        Args:
            periods: Numero di periodi per FEVD
            normalize: Se normalizzare a percentuali
            
        Returns:
            DataFrame con decomposizione della varianza
        """
        try:
            if self.fitted_model is None:
                raise ForecastError("Il modello VAR deve essere addestrato prima dell'analisi FEVD")
            
            fevd = self.fitted_model.fevd(periods=periods)
            
            # Crea MultiIndex per colonne (variabile -> fonte shock)
            variables = self.training_data.columns
            columns = pd.MultiIndex.from_product(
                [variables, variables],
                names=['variable', 'shock_source']
            )
            
            # Rimodella dati
            n_vars = len(variables)
            reshaped_data = fevd.decomp.reshape(periods, n_vars * n_vars)
            
            fevd_df = pd.DataFrame(
                reshaped_data,
                columns=columns,
                index=range(1, periods + 1)
            )
            
            if normalize:
                # Converte a percentuali
                for var in variables:
                    fevd_df[var] = fevd_df[var].div(fevd_df[var].sum(axis=1), axis=0) * 100
            
            return fevd_df
                
        except Exception as e:
            self.logger.error(f"Analisi FEVD fallita: {e}")
            raise ForecastError(f"Impossibile calcolare il FEVD: {e}")
    
    def granger_causality(
        self, 
        caused_variable: str,
        causing_variables: Optional[List[str]] = None,
        maxlag: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Esegue il test di causalità di Granger.
        
        Args:
            caused_variable: Variabile causata
            causing_variables: Variabili potenzialmente causanti (None per tutte le altre)
            maxlag: Lag massimo da testare (None usa il lag del modello)
            
        Returns:
            Dizionario con risultati del test
        """
        try:
            if self.fitted_model is None:
                raise ForecastError("Il modello VAR deve essere addestrato prima del test di causalità di Granger")
            
            if causing_variables is None:
                causing_variables = [col for col in self.training_data.columns if col != caused_variable]
            
            if maxlag is None:
                maxlag = self.fitted_model.k_ar
            
            results = {}
            for causing_var in causing_variables:
                test_result = self.fitted_model.test_causality(
                    caused=caused_variable,
                    causing=causing_var,
                    kind='f'
                )
                
                results[f"{causing_var} -> {caused_variable}"] = {
                    'test_statistic': test_result.test_statistic,
                    'p_value': test_result.pvalue,
                    'critical_value': test_result.critical_value,
                    'conclusion': 'reject' if test_result.pvalue < 0.05 else 'fail_to_reject'
                }
            
            return results
                
        except Exception as e:
            self.logger.error(f"Test di causalità di Granger fallito: {e}")
            raise ForecastError(f"Impossibile eseguire il test di causalità di Granger: {e}")
    
    def cointegration_test(self, test_type: str = 'johansen') -> Dict[str, Any]:
        """
        Testa la cointegrazione tra variabili.
        
        Args:
            test_type: Tipo di test di cointegrazione ('johansen')
            
        Returns:
            Dizionario con risultati del test
        """
        try:
            if self.fitted_model is None:
                raise ForecastError("Il modello VAR deve essere addestrato prima del test di cointegrazione")
            
            if test_type == 'johansen':
                # Test di cointegrazione di Johansen
                result = coint_johansen(self.training_data.values, det_order=0, k_ar_diff=1)
                
                return {
                    'test_type': 'johansen',
                    'trace_statistics': result.lr1,
                    'max_eigenvalue_statistics': result.lr2,
                    'critical_values_trace_90': result.cvt[:, 0],
                    'critical_values_trace_95': result.cvt[:, 1],
                    'critical_values_trace_99': result.cvt[:, 2],
                    'critical_values_maxeig_90': result.cvm[:, 0],
                    'critical_values_maxeig_95': result.cvm[:, 1],
                    'critical_values_maxeig_99': result.cvm[:, 2],
                    'eigenvalues': result.eig,
                    'eigenvectors': result.evec
                }
            else:
                raise ValueError(f"Tipo di test non supportato: {test_type}")
                
        except Exception as e:
            self.logger.error(f"Test di cointegrazione fallito: {e}")
            raise ForecastError(f"Impossibile eseguire il test di cointegrazione: {e}")
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Salva il modello VAR addestrato su disco.
        
        Args:
            filepath: Percorso per salvare il modello
        """
        try:
            if self.fitted_model is None:
                raise ModelTrainingError("Nessun modello VAR addestrato da salvare")
            
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Salva usando il metodo integrato di statsmodels
            self.fitted_model.save(str(filepath))
            
            # Salva anche i metadati
            metadata_path = filepath.with_suffix('.metadata.pkl')
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'maxlags': self.maxlags,
                    'ic': self.ic,
                    'selected_lag': self.selected_lag,
                    'training_metadata': self.training_metadata
                }, f)
            
            self.logger.info(f"VAR model saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Impossibile salvare il modello VAR: {e}")
            raise ModelTrainingError(f"Impossibile salvare il modello VAR: {e}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'VARForecaster':
        """
        Carica modello VAR addestrato da disco.
        
        Args:
            filepath: Percorso del modello salvato
            
        Returns:
            Istanza VARForecaster caricata
        """
        try:
            filepath = Path(filepath)
            
            # Carica il modello addestrato
            fitted_model = VARResults.load(str(filepath))
            
            # Carica metadati se disponibili
            metadata_path = filepath.with_suffix('.metadata.pkl')
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                maxlags = metadata.get('maxlags', None)
                ic = metadata.get('ic', 'aic')
                selected_lag = metadata.get('selected_lag', None)
                training_metadata = metadata.get('training_metadata', {})
            else:
                maxlags = None
                ic = 'aic'
                selected_lag = None
                training_metadata = {}
            
            # Crea istanza e popola
            instance = cls(maxlags=maxlags, ic=ic)
            instance.fitted_model = fitted_model
            instance.selected_lag = selected_lag
            instance.training_metadata = training_metadata
            
            instance.logger.info(f"VAR model loaded from {filepath}")
            
            return instance
            
        except Exception as e:
            logger = get_logger(__name__)
            logger.error(f"Impossibile caricare il modello VAR: {e}")
            raise ModelTrainingError(f"Impossibile caricare il modello VAR: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Ottieni informazioni complete del modello VAR.
        
        Returns:
            Dizionario con informazioni del modello
        """
        if self.fitted_model is None:
            return {'status': 'not_fitted'}
        
        info = {
            'status': 'fitted',
            'model_type': 'VAR',
            'lag_order': self.fitted_model.k_ar,
            'n_variables': len(self.fitted_model.names),
            'variable_names': self.fitted_model.names,
            'aic': self.fitted_model.aic,
            'bic': self.fitted_model.bic,
            'hqic': self.fitted_model.hqic,
            'fpe': self.fitted_model.fpe,
            'n_observations': self.fitted_model.nobs,
            'training_metadata': self.training_metadata
        }
        
        return info
    
    def _validate_data(self, data: pd.DataFrame) -> None:
        """
        Valida dati multivariati di input.
        
        Args:
            data: DataFrame da validare
            
        Raises:
            ModelTrainingError: Se la validazione fallisce
        """
        if not isinstance(data, pd.DataFrame):
            raise ModelTrainingError("L'input deve essere un pandas DataFrame")
        
        if data.empty:
            raise ModelTrainingError("Il DataFrame non può essere vuoto")
        
        if data.shape[1] < 2:
            raise ModelTrainingError("VAR richiede almeno 2 variabili")
        
        if data.isnull().all().any():
            raise ModelTrainingError("Alcune variabili sono interamente NaN")
        
        if len(data) < 20:
            self.logger.warning("I dati hanno meno di 20 osservazioni, il modello potrebbe essere inaffidabile")
        
        if data.isnull().any().any():
            missing_info = data.isnull().sum()
            self.logger.warning(f"I dati contengono valori mancanti:\n{missing_info[missing_info > 0]}")
    
    def check_stationarity(self) -> Dict[str, Dict[str, Any]]:
        """
        Controlla la stazionarietà di tutte le variabili usando il test ADF.
        
        Returns:
            Dizionario con risultati del test di stazionarietà per ogni variabile
        """
        if self.training_data is None:
            raise ModelTrainingError("Nessun dato di addestramento disponibile per il controllo di stazionarietà")
        
        results = {}
        
        for col in self.training_data.columns:
            try:
                series = self.training_data[col].dropna()
                adf_result = adfuller(series, autolag='AIC')
                
                results[col] = {
                    'adf_statistic': adf_result[0],
                    'p_value': adf_result[1],
                    'critical_values': adf_result[4],
                    'is_stationary': adf_result[1] < 0.05,
                    'recommendation': 'stazionaria' if adf_result[1] < 0.05 else 'non-stazionaria (considera differenziazione)'
                }
                
            except Exception as e:
                results[col] = {
                    'error': str(e),
                    'is_stationary': None,
                    'recommendation': 'test_fallito'
                }
        
        return results