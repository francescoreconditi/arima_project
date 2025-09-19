"""
Implementazione del modello ARIMA core con funzionalità avanzate.
"""

import pandas as pd
import numpy as np
import pickle
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima.model import ARIMAResults
from ..utils.logger import get_logger
from ..utils.exceptions import ModelTrainingError, ForecastError
from ..optimization import (
    get_model_cache,
    get_smart_starting_params,
    get_memory_pool,
    VectorizedOps,
    ManagedArray,
)


class ARIMAForecaster:
    """
    Previsore ARIMA avanzato con funzionalità complete.
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        use_cache: bool = True,
        use_smart_params: bool = True,
        use_memory_pool: bool = True,
        use_vectorized_ops: bool = True,
    ):
        """
        Inizializza il previsore ARIMA con ottimizzazioni performance.

        Args:
            order: Ordine ARIMA (p, d, q)
            use_cache: Se utilizzare model caching per speedup
            use_smart_params: Se utilizzare parametri starting intelligenti
            use_memory_pool: Se utilizzare memory pooling per ridurre GC overhead
            use_vectorized_ops: Se utilizzare operazioni vettorizzate ottimizzate
        """
        self.order = order
        self.use_cache = use_cache
        self.use_smart_params = use_smart_params
        self.use_memory_pool = use_memory_pool
        self.use_vectorized_ops = use_vectorized_ops
        self.model = None
        self.fitted_model = None
        self.training_data = None
        self.training_metadata = {}

        # Performance tracking
        self.memory_pool = get_memory_pool() if use_memory_pool else None
        self.logger = get_logger(__name__)

        # Performance tracking
        self.fit_time = 0.0
        self.cache_used = False

    def fit(
        self, series: pd.Series, validate_input: bool = True, **fit_kwargs
    ) -> "ARIMAForecaster":
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
            fit_start_time = time.time()
            self.logger.info(f"Fitting ARIMA{self.order} model to {len(series)} observations")

            if validate_input:
                self._validate_series(series)

            # PREPROCESSING OTTIMIZZATO con memory pool
            preprocessed_series = self._preprocess_with_memory_pool(series)

            # ANALISI SERIE OTTIMIZZATA con operazioni vettorizzate
            series_analysis = self._analyze_series_optimized(preprocessed_series)

            # Memorizza i dati di addestramento e i metadati
            self.training_data = preprocessed_series.copy()
            self.training_metadata = {
                "training_start": preprocessed_series.index.min(),
                "training_end": preprocessed_series.index.max(),
                "training_observations": len(preprocessed_series),
                "order": self.order,
                "series_analysis": series_analysis,
                "optimization_used": {
                    "memory_pool": self.use_memory_pool,
                    "vectorized_ops": self.use_vectorized_ops,
                    "preprocessing_applied": preprocessed_series is not series,
                },
            }

            # 1. CHECK CACHE se abilitato (usa dati preprocessati)
            if self.use_cache:
                cache = get_model_cache()
                cached_model = cache.get(preprocessed_series, self.order)

                if cached_model is not None:
                    self.fitted_model = cached_model
                    self.cache_used = True
                    self.fit_time = time.time() - fit_start_time

                    # Log cache hit
                    cache_stats = cache.get_stats()
                    self.logger.info(
                        f"CACHE HIT - Modello caricato da cache (hit_rate: {cache_stats['hit_rate']:.1%})"
                    )
                    self.logger.info(f"AIC: {self.fitted_model.aic:.2f}")
                    self.logger.info(f"BIC: {self.fitted_model.bic:.2f}")

                    return self

            # 2. TRAINING CON SMART PARAMETERS se cache miss (usa dati preprocessati)
            self.model = ARIMA(preprocessed_series, order=self.order)

            # Smart starting parameters (usa analisi serie ottimizzata)
            if self.use_smart_params:
                smart_params = self._get_smart_fit_params_from_analysis(series_analysis)
                fit_kwargs = {**smart_params, **fit_kwargs}  # fit_kwargs ha precedenza

                self.logger.debug(f"Using smart parameters from optimized analysis: {smart_params}")

            # 3. FIT del modello
            training_start = time.time()
            self.fitted_model = self.model.fit(**fit_kwargs)
            training_time = time.time() - training_start

            # 4. STORE IN CACHE se abilitato
            if self.use_cache:
                cache = get_model_cache()
                cache.store(
                    data=preprocessed_series,
                    order=self.order,
                    model=self.fitted_model,
                    fit_time=training_time,
                    metadata={
                        "aic": self.fitted_model.aic,
                        "bic": self.fitted_model.bic,
                        "observations": len(preprocessed_series),
                        "optimization_used": self.training_metadata.get("optimization_used", {}),
                    },
                )

            self.fit_time = time.time() - fit_start_time
            self.cache_used = False

            # Log risultati
            self.logger.info(
                f"Modello ARIMA addestrato con successo (fit_time: {self.fit_time:.3f}s)"
            )
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
        return_conf_int: bool = False,
    ) -> Union[
        pd.Series, Tuple[pd.Series, pd.DataFrame], Dict[str, Union[pd.Series, Dict[str, pd.Series]]]
    ]:
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
                forecast_result = self.fitted_model.forecast(steps=steps, alpha=alpha)
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
                    forecast_index = pd.date_range(start=last_date + freq, periods=steps, freq=freq)
                else:
                    forecast_index = pd.date_range(start=last_date, periods=steps + 1, freq=freq)[
                        1:
                    ]  # Salta il primo elemento per evitare sovrapposizioni
            else:
                forecast_index = range(len(self.training_data), len(self.training_data) + steps)

            forecast_series = pd.Series(forecast_values, index=forecast_index, name="forecast")

            self.logger.info(
                f"Forecast generato: {forecast_series.iloc[0]:.2f} a {forecast_series.iloc[-1]:.2f}"
            )

            # Il formato di ritorno dipende dai parametri
            if confidence_intervals and conf_int is not None:
                conf_int.index = forecast_index
                if return_conf_int:
                    return forecast_series, conf_int
                else:
                    # Restituisce formato dizionario per compatibilità con gli esempi
                    return {
                        "forecast": forecast_series,
                        "confidence_intervals": {
                            "lower": conf_int.iloc[:, 0],
                            "upper": conf_int.iloc[:, 1],
                        },
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
        dynamic: bool = False,
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
            metadata_path = filepath.with_suffix(".metadata.pkl")
            with open(metadata_path, "wb") as f:
                pickle.dump({"order": self.order, "training_metadata": self.training_metadata}, f)

            self.logger.info(f"Modello salvato in {filepath}")

        except Exception as e:
            self.logger.error(f"Impossibile salvare il modello: {e}")
            raise ModelTrainingError(f"Impossibile salvare il modello: {e}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "ARIMAForecaster":
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
            metadata_path = filepath.with_suffix(".metadata.pkl")
            if metadata_path.exists():
                with open(metadata_path, "rb") as f:
                    metadata = pickle.load(f)
                order = metadata.get("order", (1, 1, 1))
                training_metadata = metadata.get("training_metadata", {})
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
            return {"status": "not_fitted"}

        info = {
            "status": "fitted",
            "order": self.order,
            "aic": self.fitted_model.aic,
            "bic": self.fitted_model.bic,
            "hqic": self.fitted_model.hqic,
            "llf": self.fitted_model.llf,
            "n_observations": self.fitted_model.nobs,
            "params": self.fitted_model.params.to_dict()
            if hasattr(self.fitted_model.params, "to_dict")
            else dict(self.fitted_model.params),
            "training_metadata": self.training_metadata,
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
            self.logger.warning(
                "La serie ha meno di 10 osservazioni, il modello potrebbe essere inaffidabile"
            )

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
        forecast_steps: int = 12,
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
                raise ModelTrainingError(
                    "Il modello deve essere addestrato prima di generare il report"
                )

            # Imposta titolo di default
            if report_title is None:
                report_title = f"Analisi Modello ARIMA{self.order}"

            # Raccoglie i risultati del modello
            model_results = {
                "model_type": "ARIMA",
                "order": self.order,
                "model_info": self.get_model_info(),
                "training_data": {
                    "start_date": str(self.training_metadata.get("training_start", "N/A")),
                    "end_date": str(self.training_metadata.get("training_end", "N/A")),
                    "observations": self.training_metadata.get("training_observations", 0),
                },
            }

            # Aggiungi metriche se i dati di addestramento sono disponibili
            if self.training_data is not None and len(self.training_data) > 0:
                evaluator = ModelEvaluator()

                # Ottiene predizioni in-sample per la valutazione
                predictions = self.predict()

                # Calcola metriche
                if len(predictions) == len(self.training_data):
                    metrics = evaluator.calculate_forecast_metrics(
                        actual=self.training_data, predicted=predictions
                    )
                    model_results["metrics"] = metrics

                # Aggiungi diagnostici se richiesto
                if include_diagnostics:
                    try:
                        diagnostics = evaluator.evaluate_residuals(
                            residuals=self.fitted_model.resid
                        )
                        model_results["diagnostics"] = diagnostics
                    except Exception as e:
                        self.logger.warning(f"Non è stato possibile calcolare i diagnostici: {e}")

            # Aggiungi previsione se richiesto
            if include_forecast:
                try:
                    forecast_result = self.forecast(steps=forecast_steps, confidence_intervals=True)
                    if isinstance(forecast_result, dict):
                        model_results["forecast"] = {
                            "steps": forecast_steps,
                            "values": forecast_result["forecast"].tolist(),
                            "confidence_intervals": {
                                "lower": forecast_result["confidence_intervals"]["lower"].tolist(),
                                "upper": forecast_result["confidence_intervals"]["upper"].tolist(),
                            },
                            "index": forecast_result["forecast"].index.astype(str).tolist(),
                        }
                    else:
                        model_results["forecast"] = {
                            "steps": forecast_steps,
                            "values": forecast_result.tolist(),
                            "index": forecast_result.index.astype(str).tolist(),
                        }
                except Exception as e:
                    self.logger.warning(
                        f"Non è stato possibile generare il forecast per il report: {e}"
                    )

            # Aggiungi informazioni tecniche
            model_results.update(
                {
                    "python_version": f"{np.__version__}",
                    "environment": "ARIMA Forecaster Library",
                    "generation_timestamp": pd.Timestamp.now().isoformat(),
                }
            )

            # Genera report
            generator = QuartoReportGenerator()
            report_path = generator.generate_model_report(
                model_results=model_results,
                plots_data=plots_data,
                report_title=report_title,
                output_filename=output_filename,
                format_type=format_type,
            )

            return report_path

        except ImportError as e:
            raise ForecastError(
                "Moduli di reporting non disponibili. Installa le dipendenze con: pip install 'arima-forecaster[reports]'"
            )
        except Exception as e:
            self.logger.error(f"Generazione report fallita: {e}")
            raise ForecastError(f"Impossibile generare il report: {e}")

    def _get_smart_fit_params(self, series: pd.Series) -> Dict[str, Any]:
        """
        Calcola parametri intelligenti di fitting basati sulle caratteristiche dei dati.

        Analizza la serie per determinare starting parameters ottimali per il solver,
        riducendo il numero di iterazioni necessarie per la convergenza.

        Args:
            series: Serie temporale da analizzare

        Returns:
            Dizionario con parametri ottimizzati per statsmodels ARIMA fit
        """
        try:
            # Analisi caratteristiche dati
            data_length = len(series)

            # Calcola volatilità (CV = std/mean)
            cv = series.std() / abs(series.mean()) if series.mean() != 0 else 1.0
            if cv < 0.1:
                volatility_level = "low"
            elif cv < 0.3:
                volatility_level = "medium"
            else:
                volatility_level = "high"

            # Rilevamento trend (pendenza regressione lineare)
            x = np.arange(len(series))
            trend_slope = np.polyfit(x, series.values, 1)[0]
            has_trend = abs(trend_slope) > (series.std() / data_length)

            # Rilevamento stagionalità (autocorr lag 12 significativa)
            has_seasonality = False
            if data_length > 24:
                try:
                    autocorr_12 = series.autocorr(12)
                    has_seasonality = abs(autocorr_12) > 0.3
                except:
                    has_seasonality = False

            # Ottieni parametri smart dal modulo optimization
            smart_params = get_smart_starting_params(
                data_length=data_length,
                has_trend=has_trend,
                has_seasonality=has_seasonality,
                volatility_level=volatility_level,
            )

            # Adatta parametri per statsmodels ARIMA fit
            fit_params = {}

            # Per statsmodels ARIMA, i starting parameters vanno passati come array unico
            # nell'ordine: [AR params, MA params, sigma2]
            p, d, q = self.order
            start_params = []

            # Starting values per parametri AR
            if p > 0 and "ar_start" in smart_params:
                ar_values = smart_params["ar_start"]
                if len(ar_values) < p:
                    ar_values = ar_values + [ar_values[-1]] * (p - len(ar_values))
                else:
                    ar_values = ar_values[:p]
                start_params.extend(ar_values)

            # Starting values per parametri MA
            if q > 0 and "ma_start" in smart_params:
                ma_values = smart_params["ma_start"]
                if len(ma_values) < q:
                    ma_values = ma_values + [ma_values[-1]] * (q - len(ma_values))
                else:
                    ma_values = ma_values[:q]
                start_params.extend(ma_values)

            # Starting value per varianza
            if "sigma2_start" in smart_params:
                start_params.append(smart_params["sigma2_start"])

            # Passa start_params se abbiamo valori
            if start_params:
                fit_params["start_params"] = start_params

            # Parametri optimizer (passati tramite method_kwargs se method specificato)
            # Per ora limitiamo a start_params che è il più importante
            # Gli altri parametri (method, maxiter, disp) potrebbero non avere impact significativo
            # su performance per il tipo di ottimizzazione che facciamo

            # NOTE: statsmodels ARIMA.fit() accetta:
            # - start_params: array di starting values
            # - method: estimator method (non optimizer method come 'lbfgs')
            # - method_kwargs: dict con opzioni per l'estimator
            #
            # Per ora implementiamo solo start_params che ha il maggior impatto

            self.logger.debug(
                f"Smart params computed: volatility={volatility_level}, "
                f"trend={has_trend}, seasonal={has_seasonality}"
            )

            return fit_params

        except Exception as e:
            self.logger.warning(f"Errore nel calcolo smart parameters: {e}")
            # Fallback a parametri sicuri
            return {"method": "lbfgs", "maxiter": 100, "disp": False}

    def _analyze_series_optimized(self, series: pd.Series) -> Dict[str, Any]:
        """
        Analisi ottimizzata delle caratteristiche della serie con operazioni vettorizzate.

        Args:
            series: Serie temporale da analizzare

        Returns:
            Dizionario con caratteristiche della serie
        """
        try:
            # Converti a numpy per operazioni vettorizzate
            values = series.values

            if self.use_vectorized_ops:
                # Usa operazioni vettorizzate ottimizzate

                # Trend detection con regressione lineare vettorizzata
                slope, r_squared, has_trend = VectorizedOps.fast_trend_detection(values)

                # Autocorrelazione ottimizzata con FFT
                autocorr = VectorizedOps.fast_autocorr(values, max_lags=min(20, len(values) // 4))

                # Volatilità e statistiche base (già vettorizzate in NumPy)
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv = std_val / abs(mean_val) if mean_val != 0 else 1.0

                # Rilevamento stagionalità ottimizzato
                seasonality_strength = 0.0
                if len(autocorr) > 12:
                    # Controlla autocorr a lag stagionali comuni (12, 4, 7)
                    seasonal_lags = [
                        min(lag, len(autocorr) - 1) for lag in [4, 7, 12] if lag < len(autocorr)
                    ]
                    if seasonal_lags:
                        seasonality_strength = np.max(np.abs(autocorr[seasonal_lags]))

                analysis = {
                    "length": len(values),
                    "mean": float(mean_val),
                    "std": float(std_val),
                    "cv": float(cv),
                    "trend_slope": float(slope),
                    "trend_r_squared": float(r_squared),
                    "has_trend": has_trend,
                    "seasonality_strength": float(seasonality_strength),
                    "has_seasonality": seasonality_strength > 0.3,
                    "autocorr_lag1": float(autocorr[1]) if len(autocorr) > 1 else 0.0,
                    "method": "vectorized_optimized",
                }

            else:
                # Fallback a operazioni pandas standard
                analysis = {
                    "length": len(series),
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                    "cv": float(series.std() / abs(series.mean())) if series.mean() != 0 else 1.0,
                    "autocorr_lag1": float(series.autocorr(1)) if len(series) > 1 else 0.0,
                    "has_trend": False,  # Analisi semplificata
                    "has_seasonality": False,
                    "method": "pandas_standard",
                }

            return analysis

        except Exception as e:
            self.logger.warning(f"Errore in analisi serie ottimizzata: {e}")
            # Fallback minimale
            return {
                "length": len(series),
                "mean": float(series.mean()) if len(series) > 0 else 0.0,
                "std": float(series.std()) if len(series) > 0 else 1.0,
                "method": "fallback_minimal",
            }

    def _preprocess_with_memory_pool(self, series: pd.Series) -> pd.Series:
        """
        Preprocessing ottimizzato con memory pool per operazioni intermedie.

        Args:
            series: Serie da preprocessare

        Returns:
            Serie preprocessata
        """
        if not self.use_memory_pool or self.memory_pool is None:
            return series  # No preprocessing optimization

        try:
            # Operazioni che beneficiano del memory pooling

            # 1. Rimozione outlier con buffer temporaneo
            if len(series) > 10:
                with ManagedArray(len(series), dtype="float") as temp_buffer:
                    # Z-score per outlier detection
                    z_scores = np.abs((series.values - series.mean()) / series.std())
                    temp_buffer[: len(series)] = z_scores

                    # Identifica outlier (|z| > 3)
                    outlier_mask = temp_buffer[: len(series)] > 3.0

                    if np.any(outlier_mask):
                        # Sostituisci outlier con media mobile
                        clean_values = series.values.copy()
                        outlier_indices = np.where(outlier_mask)[0]

                        for idx in outlier_indices:
                            # Usa finestra simmetrica per la sostituzione
                            window_start = max(0, idx - 2)
                            window_end = min(len(clean_values), idx + 3)
                            window_values = clean_values[window_start:window_end]
                            window_values = window_values[
                                window_values != clean_values[idx]
                            ]  # Escludi il valore outlier

                            if len(window_values) > 0:
                                clean_values[idx] = np.mean(window_values)

                        # Crea serie pulita mantenendo indice originale
                        cleaned_series = pd.Series(
                            clean_values, index=series.index, name=series.name
                        )

                        outlier_count = np.sum(outlier_mask)
                        self.logger.debug(f"Rimossi {outlier_count} outlier usando memory pool")

                        return cleaned_series

            # Se non ci sono outlier o serie troppo piccola, ritorna originale
            return series

        except Exception as e:
            self.logger.warning(f"Errore in preprocessing con memory pool: {e}")
            return series  # Fallback alla serie originale

    def _get_smart_fit_params_from_analysis(
        self, series_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calcola parametri intelligenti usando analisi pre-computata per evitare ricalcoli.

        Args:
            series_analysis: Analisi caratteristiche serie già computata

        Returns:
            Dizionario con parametri ottimizzati per statsmodels ARIMA fit
        """
        try:
            # Estrai caratteristiche dall'analisi pre-computata
            data_length = series_analysis.get("length", 100)
            cv = series_analysis.get("cv", 0.2)
            has_trend = series_analysis.get("has_trend", False)
            has_seasonality = series_analysis.get("has_seasonality", False)

            # Classifica volatilità
            if cv < 0.1:
                volatility_level = "low"
            elif cv < 0.3:
                volatility_level = "medium"
            else:
                volatility_level = "high"

            # Ottieni parametri smart dal modulo optimization
            smart_params = get_smart_starting_params(
                data_length=data_length,
                has_trend=has_trend,
                has_seasonality=has_seasonality,
                volatility_level=volatility_level,
            )

            # Converte a formato statsmodels (stesso logic di _get_smart_fit_params)
            fit_params = {}
            p, d, q = self.order
            start_params = []

            # Starting values per parametri AR
            if p > 0 and "ar_start" in smart_params:
                ar_values = smart_params["ar_start"]
                if len(ar_values) < p:
                    ar_values = ar_values + [ar_values[-1]] * (p - len(ar_values))
                else:
                    ar_values = ar_values[:p]
                start_params.extend(ar_values)

            # Starting values per parametri MA
            if q > 0 and "ma_start" in smart_params:
                ma_values = smart_params["ma_start"]
                if len(ma_values) < q:
                    ma_values = ma_values + [ma_values[-1]] * (q - len(ma_values))
                else:
                    ma_values = ma_values[:q]
                start_params.extend(ma_values)

            # Starting value per varianza
            if "sigma2_start" in smart_params:
                start_params.append(smart_params["sigma2_start"])

            # Passa start_params se abbiamo valori
            if start_params:
                fit_params["start_params"] = start_params

            self.logger.debug(
                f"Smart params from analysis: volatility={volatility_level}, "
                f"trend={has_trend}, seasonal={has_seasonality}"
            )

            return fit_params

        except Exception as e:
            self.logger.warning(f"Errore nel calcolo smart parameters da analisi: {e}")
            return {}  # Fallback vuoto
