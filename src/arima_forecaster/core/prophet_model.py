"""
Implementazione Facebook Prophet per il forecasting di serie temporali.

Questo modulo fornisce la classe ProphetForecaster che integra Facebook Prophet
nel framework ARIMA Forecaster, offrendo capacità avanzate per:

- Gestione automatica della stagionalità multipla
- Rilevamento changepoints automatico  
- Gestione holiday e eventi speciali
- Regressori esterni semplificati
- Trend non-lineari con saturazione
"""

import logging
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.make_holidays import make_holidays_df
from sklearn.metrics import mean_absolute_error, mean_squared_error

from arima_forecaster.utils.exceptions import (
    DataProcessingError,
    ForecastError,  
    ModelTrainingError,
)
from arima_forecaster.utils.logger import get_logger

logger = get_logger(__name__)

# Suppress Prophet logging by default
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)


class ProphetForecaster:
    """
    Forecaster basato su Facebook Prophet con funzionalità avanzate.
    
    Prophet è particolarmente efficace per:
    - Serie con stagionalità multiple (giornaliera, settimanale, annuale)
    - Trend non-lineari con changepoints
    - Gestione automatica di holiday ed eventi
    - Dati con molti valori mancanti
    - Series con forti effetti stagionali
    
    Attributes:
        model: Il modello Prophet sottostante
        is_fitted: Se il modello è stato addestrato
        last_series: Ultima serie utilizzata per training (per diagnostiche)
        forecast_result: Ultimo risultato di forecast
        country_holidays: Paese per holiday automatici
    """
    
    def __init__(
        self,
        growth: str = 'linear',
        changepoints: Optional[List[str]] = None,
        n_changepoints: int = 25,
        changepoint_range: float = 0.8,
        yearly_seasonality: Union[bool, str, int] = 'auto',
        weekly_seasonality: Union[bool, str, int] = 'auto', 
        daily_seasonality: Union[bool, str, int] = 'auto',
        holidays: Optional[pd.DataFrame] = None,
        seasonality_mode: str = 'additive',
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0,
        changepoint_prior_scale: float = 0.05,
        mcmc_samples: int = 0,
        interval_width: float = 0.95,
        uncertainty_samples: int = 1000,
        country_holidays: Optional[str] = None,
        **kwargs
    ):
        """
        Inizializza ProphetForecaster.
        
        Args:
            growth: Tipo di crescita - 'linear' o 'logistic'
            changepoints: Date specifiche per changepoints
            n_changepoints: Numero changepoints automatici
            changepoint_range: Range per posizionamento changepoints
            yearly_seasonality: Stagionalità annuale (True/False/'auto'/int)
            weekly_seasonality: Stagionalità settimanale 
            daily_seasonality: Stagionalità giornaliera
            holidays: DataFrame con holiday personalizzati
            seasonality_mode: 'additive' o 'multiplicative' 
            seasonality_prior_scale: Prior scale per stagionalità
            holidays_prior_scale: Prior scale per holidays
            changepoint_prior_scale: Prior scale per changepoints
            mcmc_samples: Campioni MCMC (0 = MAP estimation)
            interval_width: Ampiezza intervalli confidenza
            uncertainty_samples: Campioni per incertezza
            country_holidays: Codice paese per holidays ('IT', 'US', 'UK', etc.)
            **kwargs: Altri parametri Prophet
        """
        self.growth = growth
        self.changepoints = changepoints
        self.n_changepoints = n_changepoints  
        self.changepoint_range = changepoint_range
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.holidays = holidays
        self.seasonality_mode = seasonality_mode
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self.changepoint_prior_scale = changepoint_prior_scale
        self.mcmc_samples = mcmc_samples
        self.interval_width = interval_width
        self.uncertainty_samples = uncertainty_samples
        self.country_holidays = country_holidays
        self.kwargs = kwargs
        
        # Stato interno
        self.model: Optional[Prophet] = None
        self.is_fitted: bool = False
        self.last_series: Optional[pd.Series] = None
        self.forecast_result: Optional[pd.DataFrame] = None
        self._regressors: List[str] = []
        
        # Inizializza modello
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Inizializza il modello Prophet con i parametri specificati."""
        try:
            # Prepara holidays automatici per paese
            holidays_df = None
            if self.country_holidays:
                holidays_df = self._get_country_holidays(self.country_holidays)
                if self.holidays is not None:
                    holidays_df = pd.concat([holidays_df, self.holidays], ignore_index=True)
            elif self.holidays is not None:
                holidays_df = self.holidays
            
            # Crea modello Prophet
            self.model = Prophet(
                growth=self.growth,
                changepoints=self.changepoints,
                n_changepoints=self.n_changepoints,
                changepoint_range=self.changepoint_range,
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=self.daily_seasonality,
                holidays=holidays_df,
                seasonality_mode=self.seasonality_mode,
                seasonality_prior_scale=self.seasonality_prior_scale,
                holidays_prior_scale=self.holidays_prior_scale,
                changepoint_prior_scale=self.changepoint_prior_scale,
                mcmc_samples=self.mcmc_samples,
                interval_width=self.interval_width,
                uncertainty_samples=self.uncertainty_samples,
                **self.kwargs
            )
            
            logger.info(f"Prophet model initialized with {self.country_holidays or 'no'} country holidays")
            
        except Exception as e:
            raise ModelTrainingError(f"Error initializing Prophet model: {e}")
    
    def _get_country_holidays(self, country: str) -> pd.DataFrame:
        """
        Genera holidays per il paese specificato.
        
        Args:
            country: Codice paese ('IT', 'US', 'UK', 'DE', 'FR', 'ES', 'CN', etc.)
            
        Returns:
            DataFrame con holidays del paese
        """
        try:
            # Genera holidays per 10 anni (5 passati + 5 futuri)
            current_year = datetime.now().year
            years = range(current_year - 5, current_year + 6)
            
            holidays_df = make_holidays_df(
                year_list=list(years),
                country=country
            )
            
            if len(holidays_df) == 0:
                logger.warning(f"No holidays found for country {country}")
                return pd.DataFrame(columns=['ds', 'holiday'])
            
            logger.info(f"Loaded {len(holidays_df)} holidays for {country}")
            return holidays_df
            
        except Exception as e:
            logger.warning(f"Could not load holidays for {country}: {e}")
            return pd.DataFrame(columns=['ds', 'holiday'])
    
    def add_regressor(
        self, 
        name: str, 
        prior_scale: Optional[float] = None,
        standardize: Union[bool, str] = 'auto',
        mode: Optional[str] = None
    ) -> None:
        """
        Aggiunge un regressore esterno al modello.
        
        Args:
            name: Nome del regressore
            prior_scale: Prior scale per il regressore
            standardize: Se standardizzare ('auto', True, False)
            mode: Modalità ('additive' o 'multiplicative')
        """
        if self.is_fitted:
            raise ModelTrainingError("Cannot add regressors after model is fitted")
        
        try:
            self.model.add_regressor(
                name=name,
                prior_scale=prior_scale,
                standardize=standardize,
                mode=mode
            )
            self._regressors.append(name)
            logger.info(f"Added regressor: {name}")
            
        except Exception as e:
            raise ModelTrainingError(f"Error adding regressor {name}: {e}")
    
    def add_seasonality(
        self,
        name: str,
        period: float,
        fourier_order: int,
        prior_scale: Optional[float] = None,
        mode: Optional[str] = None,
        condition_name: Optional[str] = None
    ) -> None:
        """
        Aggiunge una stagionalità personalizzata.
        
        Args:
            name: Nome della stagionalità
            period: Periodo in giorni (es. 365.25 per annuale)
            fourier_order: Numero di termini Fourier
            prior_scale: Prior scale
            mode: 'additive' o 'multiplicative'
            condition_name: Nome condizione per stagionalità condizionale
        """
        if self.is_fitted:
            raise ModelTrainingError("Cannot add seasonality after model is fitted")
        
        try:
            self.model.add_seasonality(
                name=name,
                period=period,
                fourier_order=fourier_order,
                prior_scale=prior_scale,
                mode=mode,
                condition_name=condition_name
            )
            logger.info(f"Added seasonality: {name} (period={period}, fourier_order={fourier_order})")
            
        except Exception as e:
            raise ModelTrainingError(f"Error adding seasonality {name}: {e}")
    
    def fit(
        self,
        series: pd.Series,
        exog: Optional[pd.DataFrame] = None,
        validate_input: bool = True
    ) -> 'ProphetForecaster':
        """
        Addestra il modello Prophet sui dati.
        
        Args:
            series: Serie temporale con DatetimeIndex
            exog: Variabili esogene (regressori)
            validate_input: Se validare i dati di input
            
        Returns:
            Self per method chaining
        """
        if validate_input:
            series, exog = self._validate_input_data(series, exog)
        
        self.last_series = series.copy()
        
        try:
            # Prepara dati per Prophet (ds, y format)
            prophet_data = self._prepare_prophet_data(series, exog)
            
            # Verifica regressori
            if exog is not None:
                missing_regressors = set(self._regressors) - set(exog.columns)
                if missing_regressors:
                    raise ModelTrainingError(f"Missing regressors in exog data: {missing_regressors}")
            
            # Supprime warnings Prophet durante training
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model.fit(prophet_data)
            
            self.is_fitted = True
            logger.info(f"Prophet model fitted on {len(series)} observations")
            
            return self
            
        except Exception as e:
            raise ModelTrainingError(f"Error fitting Prophet model: {e}")
    
    def _validate_input_data(
        self, 
        series: pd.Series, 
        exog: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
        """Valida i dati di input."""
        
        # Validazione serie principale
        if not isinstance(series, pd.Series):
            raise DataProcessingError("Input series must be pandas Series")
        
        if not isinstance(series.index, pd.DatetimeIndex):
            raise DataProcessingError("Series must have DatetimeIndex")
        
        if len(series) < 10:
            raise DataProcessingError("Need at least 10 observations for Prophet")
        
        if series.isna().sum() > len(series) * 0.5:
            raise DataProcessingError("Series has too many missing values (>50%)")
        
        # Validazione variabili esogene
        if exog is not None:
            if not isinstance(exog, pd.DataFrame):
                raise DataProcessingError("Exogenous data must be DataFrame")
            
            if not isinstance(exog.index, pd.DatetimeIndex):
                raise DataProcessingError("Exogenous data must have DatetimeIndex")
            
            if len(exog) != len(series):
                raise DataProcessingError("Exogenous data length must match series length")
        
        # Rimuove valori infiniti
        series = series.replace([np.inf, -np.inf], np.nan)
        if exog is not None:
            exog = exog.replace([np.inf, -np.inf], np.nan)
        
        return series, exog
    
    def _prepare_prophet_data(
        self, 
        series: pd.Series, 
        exog: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Prepara i dati nel formato richiesto da Prophet."""
        
        # Crea DataFrame base con ds e y
        data = pd.DataFrame({
            'ds': series.index,
            'y': series.values
        })
        
        # Aggiungi regressori se presenti
        if exog is not None:
            for col in exog.columns:
                data[col] = exog[col].values
        
        # Rimuove righe con NaN in y
        data = data.dropna(subset=['y'])
        
        return data
    
    def forecast(
        self,
        steps: int,
        exog_future: Optional[pd.DataFrame] = None,
        confidence_intervals: bool = True,
        confidence_level: float = 0.95
    ) -> Union[pd.Series, Tuple[pd.Series, pd.DataFrame]]:
        """
        Genera previsioni future.
        
        Args:
            steps: Numero di passi futuri da prevedere
            exog_future: Variabili esogene future (richieste se usati regressori)
            confidence_intervals: Se restituire intervalli di confidenza
            confidence_level: Livello di confidenza (0.8-0.99)
            
        Returns:
            Se confidence_intervals=False: Serie con previsioni
            Se confidence_intervals=True: Tupla (previsioni, intervalli_confidenza)
        """
        if not self.is_fitted:
            raise ForecastError("Model must be fitted before forecasting")
        
        try:
            # Crea future dataframe
            future = self._make_future_dataframe(steps, exog_future)
            
            # Genera previsioni
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                forecast_df = self.model.predict(future)
            
            self.forecast_result = forecast_df
            
            # Estrai previsioni future (solo le nuove, non fit values)
            future_predictions = forecast_df.tail(steps)
            
            # Crea serie previsioni
            forecast_series = pd.Series(
                future_predictions['yhat'].values,
                index=future_predictions['ds'].values,
                name='forecast'
            )
            
            if confidence_intervals:
                # Crea intervalli di confidenza
                lower_col = f'yhat_lower'
                upper_col = f'yhat_upper'
                
                intervals_df = pd.DataFrame({
                    'lower': future_predictions[lower_col].values,
                    'upper': future_predictions[upper_col].values
                }, index=forecast_series.index)
                
                return forecast_series, intervals_df
            
            return forecast_series
            
        except Exception as e:
            raise ForecastError(f"Error generating forecasts: {e}")
    
    def _make_future_dataframe(
        self, 
        steps: int, 
        exog_future: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Crea DataFrame per previsioni future."""
        
        # Crea future dataframe base
        future = self.model.make_future_dataframe(periods=steps)
        
        # Aggiungi regressori futuri se necessari
        if self._regressors:
            if exog_future is None:
                raise ForecastError(
                    f"Future values required for regressors: {self._regressors}"
                )
            
            # Verifica che abbiamo abbastanza dati futuri
            if len(exog_future) < steps:
                raise ForecastError(
                    f"Need {steps} future values for regressors, got {len(exog_future)}"
                )
            
            # Aggiungi solo le colonne dei regressori registrati
            for regressor in self._regressors:
                if regressor not in exog_future.columns:
                    raise ForecastError(f"Missing future values for regressor: {regressor}")
                
                # Riempi valori storici (se mancanti) + futuri  
                historical_values = self.last_series.index
                future_start_idx = len(historical_values)
                
                # Valori storici (se disponibili) + valori futuri
                regressor_values = np.full(len(future), np.nan)
                
                # Riempi valori futuri
                future_values = exog_future[regressor].iloc[:steps].values
                regressor_values[future_start_idx:] = future_values
                
                # Riempi valori storici se disponibili nei dati originali
                if hasattr(self, '_original_exog') and self._original_exog is not None:
                    if regressor in self._original_exog.columns:
                        hist_vals = self._original_exog[regressor].values
                        regressor_values[:len(hist_vals)] = hist_vals
                
                future[regressor] = regressor_values
        
        return future
    
    def predict(self, start: Optional[int] = None, end: Optional[int] = None) -> pd.Series:
        """
        Genera predizioni in-sample (fitted values).
        
        Args:
            start: Indice di inizio (ignorato, per compatibilità)
            end: Indice di fine (ignorato, per compatibilità)
            
        Returns:
            Serie con valori predetti in-sample
        """
        if not self.is_fitted:
            raise ForecastError("Model must be fitted before prediction")
        
        try:
            # Usa ultimo forecast o rigenera per serie completa
            if self.forecast_result is not None:
                # Estrai solo fitted values (non future predictions)
                fitted_length = len(self.last_series)
                fitted_result = self.forecast_result.head(fitted_length)
                
                return pd.Series(
                    fitted_result['yhat'].values,
                    index=self.last_series.index,
                    name='fitted'
                )
            else:
                # Rigenera prediction per serie completa
                future = self.model.make_future_dataframe(periods=0)  # Solo dati storici
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    forecast_df = self.model.predict(future)
                
                return pd.Series(
                    forecast_df['yhat'].values,
                    index=self.last_series.index,
                    name='fitted'
                )
                
        except Exception as e:
            raise ForecastError(f"Error generating in-sample predictions: {e}")
    
    def get_components(self) -> pd.DataFrame:
        """
        Restituisce i componenti del modello (trend, stagionalità, holidays).
        
        Returns:
            DataFrame con componenti decomposte
        """
        if not self.is_fitted or self.forecast_result is None:
            raise ForecastError("Model must be fitted and forecasted to get components")
        
        component_cols = ['trend']
        
        # Aggiungi componenti stagionali se presenti
        if 'yearly' in self.forecast_result.columns:
            component_cols.append('yearly')
        if 'weekly' in self.forecast_result.columns:  
            component_cols.append('weekly')
        if 'daily' in self.forecast_result.columns:
            component_cols.append('daily')
        
        # Aggiungi holidays se presente
        if 'holidays' in self.forecast_result.columns:
            component_cols.append('holidays')
        
        # Aggiungi regressori
        for regressor in self._regressors:
            if regressor in self.forecast_result.columns:
                component_cols.append(regressor)
        
        components_df = self.forecast_result[component_cols].copy()
        components_df.index = self.forecast_result['ds']
        
        return components_df
    
    def get_changepoints(self) -> pd.DataFrame:
        """
        Restituisce informazioni sui changepoints rilevati.
        
        Returns:
            DataFrame con changepoints e loro impatti
        """
        if not self.is_fitted:
            raise ForecastError("Model must be fitted to get changepoints")
        
        try:
            changepoints_df = pd.DataFrame({
                'ds': self.model.changepoints,
                'delta': self.model.params['delta'].mean(axis=0)  # Media posteriore
            })
            
            # Ordina per data
            changepoints_df = changepoints_df.sort_values('ds')
            
            # Aggiungi significatività (delta assoluto > soglia)
            changepoints_df['significant'] = np.abs(changepoints_df['delta']) > 0.01
            
            return changepoints_df
            
        except Exception as e:
            logger.warning(f"Could not extract changepoints: {e}")
            return pd.DataFrame(columns=['ds', 'delta', 'significant'])
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Restituisce informazioni dettagliate sul modello.
        
        Returns:
            Dizionario con informazioni del modello
        """
        if not self.is_fitted:
            return {
                'fitted': False,
                'model_type': 'Facebook Prophet',
                'parameters': self._get_parameters_dict()
            }
        
        # Calcola metriche base se possibile
        metrics = {}
        if self.last_series is not None:
            try:
                fitted_values = self.predict()
                mae = mean_absolute_error(self.last_series, fitted_values)
                mse = mean_squared_error(self.last_series, fitted_values)
                
                metrics = {
                    'mae': mae,
                    'rmse': np.sqrt(mse),
                    'mape': np.mean(np.abs((self.last_series - fitted_values) / self.last_series)) * 100
                }
            except Exception:
                metrics = {'error': 'Could not calculate metrics'}
        
        info = {
            'fitted': True,
            'model_type': 'Facebook Prophet',
            'parameters': self._get_parameters_dict(),
            'training_observations': len(self.last_series) if self.last_series is not None else 0,
            'regressors': self._regressors,
            'metrics': metrics,
            'changepoints_detected': len(self.model.changepoints) if hasattr(self.model, 'changepoints') else 0
        }
        
        return info
    
    def _get_parameters_dict(self) -> Dict[str, Any]:
        """Restituisce dizionario con parametri del modello."""
        return {
            'growth': self.growth,
            'n_changepoints': self.n_changepoints,
            'changepoint_range': self.changepoint_range,
            'yearly_seasonality': self.yearly_seasonality,
            'weekly_seasonality': self.weekly_seasonality,
            'daily_seasonality': self.daily_seasonality,
            'seasonality_mode': self.seasonality_mode,
            'seasonality_prior_scale': self.seasonality_prior_scale,
            'holidays_prior_scale': self.holidays_prior_scale,
            'changepoint_prior_scale': self.changepoint_prior_scale,
            'interval_width': self.interval_width,
            'country_holidays': self.country_holidays
        }
    
    @property
    def residuals(self) -> Optional[pd.Series]:
        """
        Calcola i residui del modello (actual - fitted).
        
        Returns:
            Serie con residui o None se modello non fittato
        """
        if not self.is_fitted or self.last_series is None:
            return None
        
        try:
            fitted_values = self.predict()
            residuals = self.last_series - fitted_values
            return residuals
        except Exception:
            return None
    
    def __repr__(self) -> str:
        """Rappresentazione string del modello."""
        if not self.is_fitted:
            return "ProphetForecaster(not fitted)"
        
        params = []
        if self.yearly_seasonality != 'auto':
            params.append(f"yearly={self.yearly_seasonality}")
        if self.weekly_seasonality != 'auto':
            params.append(f"weekly={self.weekly_seasonality}")
        if self.daily_seasonality != 'auto':
            params.append(f"daily={self.daily_seasonality}")
        if self._regressors:
            params.append(f"regressors={len(self._regressors)}")
        if self.country_holidays:
            params.append(f"holidays={self.country_holidays}")
        
        params_str = ", ".join(params) if params else "default"
        n_obs = len(self.last_series) if self.last_series is not None else 0
        
        return f"ProphetForecaster({params_str}, n_obs={n_obs})"