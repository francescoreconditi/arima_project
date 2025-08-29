"""
Advanced Exogenous Variables Handling per modelli SARIMAX.
Sistema di selezione automatica delle feature con validazione robusta.
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union, List
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.feature_selection import f_regression, SelectKBest, RFE
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import logging

from .sarima_model import SARIMAForecaster
from ..utils.logger import get_logger
from ..utils.exceptions import ModelTrainingError, ForecastError
from ..utils.preprocessing import ExogenousPreprocessor, validate_exog_data


class SARIMAXAutoSelector(SARIMAForecaster):
    """
    SARIMAX Forecaster con selezione automatica feature esogene.
    
    Estende SARIMAForecaster aggiungendo:
    - Selezione automatica feature rilevanti 
    - Preprocessing intelligente variabili esogene
    - Validazione robusta e diagnostica
    - Feature engineering automatico
    """
    
    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12),
        trend: Optional[str] = None,
        # Parametri Advanced Exog Handling
        max_features: Optional[int] = 10,
        selection_method: str = 'stepwise',
        feature_engineering: List[str] = ['lags'],
        preprocessing_method: str = 'auto',
        validation_split: float = 0.2,
        min_feature_importance: float = 0.01
    ):
        """
        Inizializza SARIMAX Auto Selector.
        
        Args:
            order: Ordine ARIMA non stagionale (p, d, q)
            seasonal_order: Ordine ARIMA stagionale (P, D, Q, s)
            trend: Parametro di trend ('n', 'c', 't', 'ct')
            max_features: Numero massimo feature da selezionare
            selection_method: Metodo selezione ('stepwise', 'lasso', 'elastic_net', 'f_test')
            feature_engineering: Lista tecniche feature engineering ('lags', 'differences', 'interactions')
            preprocessing_method: Metodo preprocessing ('auto', 'robust', 'standard', 'minmax', 'none')
            validation_split: Frazione dati per validazione
            min_feature_importance: Soglia minima importanza feature
        """
        # Inizializza classe base
        super().__init__(order=order, seasonal_order=seasonal_order, trend=trend)
        
        # Parametri Advanced Exog Handling
        self.max_features = max_features
        self.selection_method = selection_method
        self.feature_engineering = feature_engineering
        self.preprocessing_method = preprocessing_method
        self.validation_split = validation_split
        self.min_feature_importance = min_feature_importance
        
        # Componenti interni
        self.preprocessor = None
        self.feature_selector = None
        self.selected_features = []
        self.feature_importance = {}
        self.original_features = []
        self.engineered_features = []
        
        # Validazione parametri
        self._validate_parameters()
        
        # Override logger per classe specifica
        self.logger = get_logger(f"{__name__}.SARIMAXAutoSelector")
    
    def fit_with_exog(
        self,
        series: pd.Series,
        exog: pd.DataFrame,
        validate_input: bool = True,
        **fit_kwargs
    ) -> 'SARIMAXAutoSelector':
        """
        Addestra SARIMAX con selezione automatica feature esogene.
        
        Args:
            series: Serie temporale target
            exog: DataFrame variabili esogene
            validate_input: Se validare input
            **fit_kwargs: Parametri aggiuntivi per SARIMAX.fit()
            
        Returns:
            Self per method chaining
            
        Raises:
            ModelTrainingError: Se training fallisce
        """
        try:
            self.logger.info(
                f"Iniziando training SARIMAX{self.order}x{self.seasonal_order} "
                f"con {len(exog.columns)} variabili esogene"
            )
            
            if validate_input:
                self._validate_inputs(series, exog)
            
            # Memorizza dati originali (verrà aggiornato dopo feature engineering)
            self.original_features = list(exog.columns)
            
            # Step 1: Feature Engineering
            exog_engineered, series_aligned = self._engineer_features(exog, series)
            self.logger.info(f"Feature engineering: {len(exog.columns)} → {len(exog_engineered.columns)} feature")
            self.logger.info(f"Serie allineata: {len(series)} → {len(series_aligned)} obs")
            
            # Memorizza dati allineati dopo feature engineering
            self.training_data = series_aligned.copy()
            
            # Step 2: Preprocessing
            exog_processed = self._preprocess_features(exog_engineered)
            
            # Step 3: Feature Selection
            exog_selected, selected_indices = self._select_features(exog_processed, series_aligned)
            self.logger.info(f"Feature selection: {len(exog_processed.columns)} → {len(exog_selected.columns)} feature")
            
            # Step 4: Verifica allineamento finale prima del training
            self.logger.info(f"Verifica allineamento finale:")
            self.logger.info(f"Serie: {len(series_aligned)} obs, index: {series_aligned.index[0]} to {series_aligned.index[-1]}")
            self.logger.info(f"Exog: {len(exog_selected)} obs, index: {exog_selected.index[0]} to {exog_selected.index[-1]}")
            self.logger.info(f"Indici allineati: {series_aligned.index.equals(exog_selected.index)}")
            
            if not series_aligned.index.equals(exog_selected.index):
                self.logger.error("ERRORE: Indici non allineati prima del training SARIMAX!")
                self.logger.error(f"Serie index dtype: {series_aligned.index.dtype}, Exog index dtype: {exog_selected.index.dtype}")
                raise ModelTrainingError(
                    "Indici non allineati dopo preprocessing. "
                    f"Serie: {len(series_aligned)} obs, Exog: {len(exog_selected)} obs"
                )
            
            # Step 4: Training SARIMAX con feature selezionate e serie allineata
            self.model = SARIMAX(
                series_aligned,
                exog=exog_selected,
                order=self.order,
                seasonal_order=self.seasonal_order,
                trend=self.trend
            )
            
            # Training con gestione warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self.fitted_model = self.model.fit(**fit_kwargs)
            
            # Step 5: Calcola importanza feature
            self._calculate_feature_importance(exog_selected)
            
            # Step 6: Aggiorna metadati
            self._update_training_metadata(series, exog, exog_selected)
            
            # Log finale
            self.logger.info("SARIMAX Auto Selector training completato")
            self.logger.info(f"Feature selezionate: {self.selected_features}")
            self.logger.info(f"AIC: {self.fitted_model.aic:.2f}, BIC: {self.fitted_model.bic:.2f}")
            
            return self
            
        except Exception as e:
            self.logger.error(f"Training SARIMAX Auto Selector fallito: {e}")
            raise ModelTrainingError(f"Impossibile addestrare SARIMAX Auto Selector: {e}")
    
    def forecast_with_exog(
        self,
        steps: int,
        exog: Optional[pd.DataFrame] = None,
        confidence_intervals: bool = True,
        alpha: float = 0.05
    ) -> Union[pd.Series, Tuple[pd.Series, pd.DataFrame]]:
        """
        Genera previsioni con variabili esogene.
        
        Args:
            steps: Numero step da prevedere
            exog: Variabili esogene per forecast (opzionale)
            confidence_intervals: Se calcolare intervalli confidenza
            alpha: Livello alpha per intervalli confidenza
            
        Returns:
            Serie previsioni, opzionalmente con intervalli confidenza
        """
        try:
            if self.fitted_model is None:
                raise ForecastError("Modello deve essere addestrato prima del forecast")
                
            self.logger.info(f"Generazione forecast SARIMAX a {steps} step")
            
            # Prepara variabili esogene per forecast
            if exog is not None:
                # Applica stesso preprocessing e selezione del training
                exog_processed = self._prepare_forecast_exog(exog, steps)
                if exog_processed is None or len(exog_processed) == 0:
                    self.logger.error("Preparazione exog forecast fallita - nessuna feature valida")
                    raise ForecastError("Impossibile preparare variabili esogene per forecast")
            else:
                # Per modelli SARIMAX con exog, dobbiamo fornire valori anche per forecast
                # Se non abbiamo exog, creiamo valori zero
                if self.selected_features:
                    self.logger.warning("Creazione exog nulle per forecast - modello addestrato con exog")
                    exog_processed = pd.DataFrame(
                        np.zeros((steps, len(self.selected_features))),
                        columns=self.selected_features
                    )
                else:
                    exog_processed = None
                    self.logger.info("Nessuna variabile esogena richiesta per forecast")
            
            # Genera forecast
            forecast_result = self.fitted_model.get_forecast(
                steps=steps,
                exog=exog_processed,
                alpha=alpha
            )
            
            forecast_values = forecast_result.predicted_mean
            
            # Crea indice temporale
            forecast_index = self._create_forecast_index(steps)
            forecast_series = pd.Series(forecast_values, index=forecast_index, name='forecast')
            
            if confidence_intervals:
                conf_int = forecast_result.conf_int()
                conf_int.index = forecast_index
                
                self.logger.info(
                    f"Forecast SARIMAX completato: {forecast_series.iloc[0]:.2f} → {forecast_series.iloc[-1]:.2f}"
                )
                
                return forecast_series, conf_int
            else:
                return forecast_series
                
        except Exception as e:
            self.logger.error(f"Forecast SARIMAX fallito: {e}")
            raise ForecastError(f"Impossibile generare forecast SARIMAX: {e}")
    
    def _engineer_features(self, exog: pd.DataFrame, series: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Applica feature engineering alle variabili esogene."""
        result = exog.copy()
        original_cols = list(exog.columns)
        
        if 'lags' in self.feature_engineering:
            # Aggiungi lag delle variabili esogene
            for col in original_cols:
                # Lag 1 e 2 per catturare dipendenze temporali
                result[f"{col}_lag1"] = result[col].shift(1)
                result[f"{col}_lag2"] = result[col].shift(2)
            
            self.logger.debug("Aggiunti lag delle variabili esogene")
        
        if 'differences' in self.feature_engineering:
            # Aggiungi differenze per catturare trend
            for col in original_cols:
                result[f"{col}_diff"] = result[col].diff()
                result[f"{col}_pct_change"] = result[col].pct_change()
            
            self.logger.debug("Aggiunte differenze e variazioni percentuali")
        
        if 'interactions' in self.feature_engineering:
            # Aggiungi interazioni tra prime 2-3 variabili (evita esplosione dimensionale)
            interaction_cols = original_cols[:min(3, len(original_cols))]
            if len(interaction_cols) >= 2:
                for i, col1 in enumerate(interaction_cols):
                    for col2 in interaction_cols[i+1:]:
                        result[f"{col1}_x_{col2}"] = result[col1] * result[col2]
            
            self.logger.debug("Aggiunte interazioni tra variabili")
        
        # Rimuovi NaN generati da lags/diff (dropna allinea automaticamente gli indici)
        result = result.dropna()
        
        # Allinea serie temporale ai dati processati
        series_aligned = series.loc[result.index]
        
        self.engineered_features = [col for col in result.columns if col not in original_cols]
        
        return result, series_aligned
    
    def _preprocess_features(self, exog: pd.DataFrame) -> pd.DataFrame:
        """Applica preprocessing alle feature."""
        if self.preprocessing_method == 'auto':
            from ..utils.preprocessing import suggest_preprocessing_method
            method = suggest_preprocessing_method(exog)
        else:
            method = self.preprocessing_method
        
        self.preprocessor = ExogenousPreprocessor(
            method=method,
            handle_outliers=True
        )
        
        exog_processed = self.preprocessor.fit_transform(exog)
        
        self.logger.info(f"Preprocessing applicato: metodo={method}")
        
        return exog_processed
    
    def _select_features(self, exog: pd.DataFrame, series: pd.Series) -> Tuple[pd.DataFrame, List[int]]:
        """Seleziona le feature più rilevanti."""
        if len(exog.columns) <= (self.max_features or 10):
            # Se abbiamo già poche feature, usale tutte
            self.selected_features = list(exog.columns)
            return exog, list(range(len(exog.columns)))
        
        # Allinea serie temporale con exog
        series_aligned = series.loc[exog.index]
        
        if self.selection_method == 'stepwise':
            selected_features, indices = self._stepwise_selection(exog, series_aligned)
        elif self.selection_method == 'lasso':
            selected_features, indices = self._lasso_selection(exog, series_aligned)
        elif self.selection_method == 'elastic_net':
            selected_features, indices = self._elastic_net_selection(exog, series_aligned)
        elif self.selection_method == 'f_test':
            selected_features, indices = self._f_test_selection(exog, series_aligned)
        else:
            raise ValueError(f"Metodo selezione non supportato: {self.selection_method}")
        
        self.selected_features = selected_features
        exog_selected = exog[selected_features]
        
        return exog_selected, indices
    
    def _stepwise_selection(self, exog: pd.DataFrame, series: pd.Series) -> Tuple[List[str], List[int]]:
        """Selezione stepwise basata su AIC."""
        remaining_features = list(exog.columns)
        selected_features = []
        best_aic = float('inf')
        
        max_iterations = min(self.max_features or len(remaining_features), len(remaining_features))
        
        for iteration in range(max_iterations):
            best_feature = None
            best_aic_iter = float('inf')
            
            # Forward selection: testa aggiunta di ogni feature rimanente
            for feature in remaining_features:
                test_features = selected_features + [feature]
                
                try:
                    # Test modello con feature candidate
                    model = SARIMAX(
                        series,
                        exog=exog[test_features],
                        order=self.order,
                        seasonal_order=self.seasonal_order,
                        trend=self.trend
                    )
                    
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        fitted = model.fit(disp=False)
                    
                    if fitted.aic < best_aic_iter:
                        best_aic_iter = fitted.aic
                        best_feature = feature
                        
                except Exception:
                    # Skippa feature che causano problemi
                    continue
            
            # Aggiunge feature se migliora AIC
            if best_feature and best_aic_iter < best_aic:
                selected_features.append(best_feature)
                remaining_features.remove(best_feature)
                best_aic = best_aic_iter
                
                self.logger.debug(f"Stepwise iteration {iteration+1}: aggiunta '{best_feature}' (AIC: {best_aic:.2f})")
            else:
                # Nessun miglioramento, ferma selezione
                break
        
        # Trova indici delle feature selezionate
        indices = [exog.columns.get_loc(feat) for feat in selected_features]
        
        return selected_features, indices
    
    def _lasso_selection(self, exog: pd.DataFrame, series: pd.Series) -> Tuple[List[str], List[int]]:
        """Selezione con Lasso regression."""
        # Standardizza features per Lasso
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(exog.values)
        
        # Cross-validated Lasso
        lasso = LassoCV(cv=5, random_state=42, max_iter=1000)
        lasso.fit(X_scaled, series.values)
        
        # Seleziona feature con coefficienti non-zero
        selected_mask = np.abs(lasso.coef_) > self.min_feature_importance
        selected_features = exog.columns[selected_mask].tolist()
        
        # Limita numero feature
        if self.max_features and len(selected_features) > self.max_features:
            # Ordina per importanza assoluta coefficiente
            importance_scores = np.abs(lasso.coef_[selected_mask])
            top_indices = np.argsort(importance_scores)[-self.max_features:]
            selected_features = [selected_features[i] for i in top_indices]
        
        indices = [exog.columns.get_loc(feat) for feat in selected_features]
        
        self.logger.debug(f"Lasso selection: {len(selected_features)} feature selezionate (alpha: {lasso.alpha_:.6f})")
        
        return selected_features, indices
    
    def _elastic_net_selection(self, exog: pd.DataFrame, series: pd.Series) -> Tuple[List[str], List[int]]:
        """Selezione con Elastic Net."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(exog.values)
        
        # Cross-validated Elastic Net
        elastic = ElasticNetCV(cv=5, random_state=42, max_iter=1000)
        elastic.fit(X_scaled, series.values)
        
        selected_mask = np.abs(elastic.coef_) > self.min_feature_importance
        selected_features = exog.columns[selected_mask].tolist()
        
        if self.max_features and len(selected_features) > self.max_features:
            importance_scores = np.abs(elastic.coef_[selected_mask])
            top_indices = np.argsort(importance_scores)[-self.max_features:]
            selected_features = [selected_features[i] for i in top_indices]
        
        indices = [exog.columns.get_loc(feat) for feat in selected_features]
        
        self.logger.debug(f"Elastic Net selection: {len(selected_features)} feature (alpha: {elastic.alpha_:.6f}, l1_ratio: {elastic.l1_ratio_:.3f})")
        
        return selected_features, indices
    
    def _f_test_selection(self, exog: pd.DataFrame, series: pd.Series) -> Tuple[List[str], List[int]]:
        """Selezione basata su F-test statistico."""
        k = min(self.max_features or 10, len(exog.columns))
        
        selector = SelectKBest(score_func=f_regression, k=k)
        X_selected = selector.fit_transform(exog.values, series.values)
        
        selected_indices = selector.get_support(indices=True)
        selected_features = exog.columns[selected_indices].tolist()
        
        self.logger.debug(f"F-test selection: {len(selected_features)} feature con maggiore F-statistic")
        
        return selected_features, list(selected_indices)
    
    def _calculate_feature_importance(self, exog_selected: pd.DataFrame) -> None:
        """Calcola importanza delle feature selezionate."""
        if self.fitted_model is None:
            return
        
        try:
            # Importanza basata sui coefficienti del modello
            if hasattr(self.fitted_model, 'params'):
                params = self.fitted_model.params
                
                # I parametri esogeni sono dopo quelli SARIMA
                n_sarima_params = len(self.order) + len(self.seasonal_order) - 1
                if self.trend:
                    n_sarima_params += len(self.trend)
                
                exog_params = params.iloc[n_sarima_params:]
                
                for i, feature in enumerate(exog_selected.columns):
                    if i < len(exog_params):
                        self.feature_importance[feature] = abs(exog_params.iloc[i])
                
                self.logger.debug(f"Feature importance calcolata per {len(self.feature_importance)} feature")
                
        except Exception as e:
            self.logger.warning(f"Impossibile calcolare feature importance: {e}")
    
    def _prepare_forecast_exog(self, exog: pd.DataFrame, steps: int) -> Optional[pd.DataFrame]:
        """Prepara variabili esogene per il forecast."""
        if exog is None:
            return None
        
        try:
            # 1. Applica stesso feature engineering
            if self.feature_engineering:
                # Create a dummy series for feature engineering
                dummy_series = pd.Series(index=exog.index, dtype=float)
                exog_engineered, _ = self._engineer_features(exog, dummy_series)
            else:
                exog_engineered = exog.copy()
            
            # 2. Applica preprocessing se disponibile
            if self.preprocessor and self.preprocessor.is_fitted:
                exog_processed = self.preprocessor.transform(exog_engineered)
            else:
                exog_processed = exog_engineered
            
            # 3. Seleziona solo feature usate nel training
            available_features = [f for f in self.selected_features if f in exog_processed.columns]
            
            if len(available_features) != len(self.selected_features):
                missing = set(self.selected_features) - set(available_features)
                self.logger.warning(f"Feature mancanti per forecast: {missing}")
            
            if available_features:
                exog_final = exog_processed[available_features]
                
                # 4. Assicura lunghezza corretta per forecast
                if len(exog_final) < steps:
                    # Estendi usando l'ultimo valore (forward fill)
                    n_missing = steps - len(exog_final)
                    last_row = exog_final.iloc[-1:]
                    
                    # Crea nuovo index numerico per le righe estese
                    extension_data = []
                    for i in range(n_missing):
                        extension_data.append(last_row.iloc[0].copy())
                    
                    extension = pd.DataFrame(extension_data, columns=exog_final.columns)
                    # Concatena e resetta l'index
                    exog_final = pd.concat([exog_final, extension], ignore_index=True)
                    
                elif len(exog_final) > steps:
                    exog_final = exog_final.iloc[:steps].copy()
                
                # Reset index to ensure numeric index for statsmodels (0, 1, 2, ...)
                exog_final = exog_final.reset_index(drop=True)
                
                self.logger.info(f"Preparazione exog forecast completata: {len(exog_final)} righe, {len(available_features)} features")
                return exog_final
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Errore preparazione exog forecast: {e}")
            return None
    
    def _validate_inputs(self, series: pd.Series, exog: pd.DataFrame) -> None:
        """Valida input per training."""
        # Validazione serie temporale (dalla classe base)
        self._validate_series(series)
        
        # Validazione variabili esogene
        is_valid, error_msg = validate_exog_data(exog, len(series))
        if not is_valid:
            raise ModelTrainingError(f"Validazione exog fallita: {error_msg}")
        
        # Validazione allineamento indici
        if not series.index.equals(exog.index):
            self.logger.warning("Indici serie e exog non perfettamente allineati - applicando allineamento automatico")
            raise ModelTrainingError(
                "Indici serie temporale e variabili esogene non sono allineati. "
                "Assicurarsi che abbiano lo stesso indice temporale."
            )
    
    def _validate_parameters(self) -> None:
        """Valida parametri inizializzazione."""
        valid_selection_methods = ['stepwise', 'lasso', 'elastic_net', 'f_test']
        if self.selection_method not in valid_selection_methods:
            raise ValueError(f"selection_method deve essere uno di: {valid_selection_methods}")
        
        valid_preprocessing_methods = ['auto', 'robust', 'standard', 'minmax', 'none']
        if self.preprocessing_method not in valid_preprocessing_methods:
            raise ValueError(f"preprocessing_method deve essere uno di: {valid_preprocessing_methods}")
        
        valid_feature_engineering = ['lags', 'differences', 'interactions']
        if not all(method in valid_feature_engineering for method in self.feature_engineering):
            raise ValueError(f"feature_engineering deve contenere elementi di: {valid_feature_engineering}")
        
        if self.max_features is not None and self.max_features < 1:
            raise ValueError("max_features deve essere >= 1")
        
        if not 0 < self.validation_split < 1:
            raise ValueError("validation_split deve essere tra 0 e 1")
    
    def _update_training_metadata(self, series: pd.Series, exog_original: pd.DataFrame, exog_selected: pd.DataFrame) -> None:
        """Aggiorna metadati training."""
        self.training_metadata.update({
            'n_original_exog': len(exog_original.columns),
            'n_engineered_features': len(self.engineered_features),
            'n_selected_features': len(self.selected_features),
            'original_features': self.original_features,
            'engineered_features': self.engineered_features,
            'selected_features': self.selected_features,
            'selection_method': self.selection_method,
            'preprocessing_method': self.preprocessing_method,
            'feature_engineering': self.feature_engineering,
            'feature_importance': self.feature_importance
        })
    
    def _create_forecast_index(self, steps: int) -> pd.Index:
        """Crea indice temporale per forecast."""
        last_date = self.training_data.index[-1]
        
        if isinstance(last_date, pd.Timestamp):
            freq = pd.infer_freq(self.training_data.index)
            if freq:
                try:
                    freq_offset = pd.tseries.frequencies.to_offset(freq)
                    forecast_index = pd.date_range(
                        start=last_date + freq_offset,
                        periods=steps,
                        freq=freq
                    )
                except Exception:
                    forecast_index = pd.date_range(
                        start=last_date + pd.Timedelta(days=1),
                        periods=steps,
                        freq='D'
                    )
            else:
                forecast_index = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=steps,
                    freq='D'
                )
        else:
            forecast_index = range(len(self.training_data), len(self.training_data) + steps)
        
        return forecast_index
    
    def get_feature_analysis(self) -> Dict[str, Any]:
        """
        Restituisce analisi completa delle feature selezionate.
        
        Returns:
            Dizionario con analisi feature
        """
        if not self.selected_features:
            return {'status': 'no_features_selected'}
        
        analysis = {
            'status': 'features_selected',
            'selection_summary': {
                'original_features': len(self.original_features),
                'engineered_features': len(self.engineered_features),
                'total_features': len(self.original_features) + len(self.engineered_features),
                'selected_features': len(self.selected_features),
                'selection_ratio': len(self.selected_features) / (len(self.original_features) + len(self.engineered_features))
            },
            'feature_details': {
                'original_features': self.original_features,
                'engineered_features': self.engineered_features,
                'selected_features': self.selected_features,
                'feature_importance': self.feature_importance
            },
            'configuration': {
                'selection_method': self.selection_method,
                'preprocessing_method': self.preprocessing_method,
                'feature_engineering': self.feature_engineering,
                'max_features': self.max_features
            }
        }
        
        # Aggiungi statistiche preprocessing se disponibili
        if self.preprocessor and hasattr(self.preprocessor, 'get_stats'):
            analysis['preprocessing_stats'] = self.preprocessor.get_stats()
        
        return analysis
    
    def get_model_info(self) -> Dict[str, Any]:
        """Estende get_model_info della classe base con info exog."""
        base_info = super().get_model_info()
        
        if base_info['status'] == 'fitted':
            base_info.update({
                'exog_features': {
                    'n_original': len(self.original_features) if self.original_features else 0,
                    'n_selected': len(self.selected_features) if self.selected_features else 0,
                    'selected_features': self.selected_features,
                    'selection_method': self.selection_method
                },
                'feature_importance': self.feature_importance
            })
        
        return base_info