"""
Implementazione del modello SARIMA con supporto stagionale.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from ..utils.logger import get_logger
from ..utils.exceptions import ModelTrainingError, ForecastError


class SARIMAForecaster:
    """
    Previsore SARIMA con supporto stagionale.
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12),
        trend: Optional[str] = None,
    ):
        """
        Inizializza il previsore SARIMA.

        Args:
            order: Ordine ARIMA non stagionale (p, d, q)
            seasonal_order: Ordine ARIMA stagionale (P, D, Q, s)
            trend: Parametro di trend ('n', 'c', 't', 'ct')
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.model = None
        self.fitted_model = None
        self.training_data = None
        self.training_metadata = {}
        self.logger = get_logger(__name__)

    def fit(
        self, series: pd.Series, validate_input: bool = True, **fit_kwargs
    ) -> "SARIMAForecaster":
        """
        Addestra il modello SARIMA sui dati delle serie temporali.

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
            self.logger.info(
                f"Fitting SARIMA{self.order}x{self.seasonal_order} model "
                f"to {len(series)} observations"
            )

            if validate_input:
                self._validate_series(series)
                self._validate_seasonal_parameters(series)

            # Memorizza i dati di addestramento e i metadati
            self.training_data = series.copy()
            self.training_metadata = {
                "training_start": series.index.min(),
                "training_end": series.index.max(),
                "training_observations": len(series),
                "order": self.order,
                "seasonal_order": self.seasonal_order,
                "trend": self.trend,
            }

            # Crea e addestra il modello
            self.model = SARIMAX(
                series, order=self.order, seasonal_order=self.seasonal_order, trend=self.trend
            )
            self.fitted_model = self.model.fit(**fit_kwargs)

            # Registra il riepilogo del modello
            self.logger.info("Modello SARIMA addestrato con successo")
            self.logger.info(f"AIC: {self.fitted_model.aic:.2f}")
            self.logger.info(f"BIC: {self.fitted_model.bic:.2f}")

            return self

        except Exception as e:
            self.logger.error(f"Addestramento modello SARIMA fallito: {e}")
            raise ModelTrainingError(f"Impossibile addestrare il modello SARIMA: {e}")

    def forecast(
        self,
        steps: int,
        confidence_intervals: bool = True,
        alpha: float = 0.05,
        return_conf_int: bool = False,
    ) -> Union[pd.Series, Tuple[pd.Series, pd.DataFrame]]:
        """
        Genera previsioni dal modello SARIMA addestrato.

        Args:
            steps: Numero di passaggi da prevedere
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
                raise ForecastError(
                    "Il modello SARIMA deve essere addestrato prima della previsione"
                )

            self.logger.info(f"Generazione previsione SARIMA a {steps} passaggi")

            # Genera previsione
            forecast_result = self.fitted_model.get_forecast(steps=steps, alpha=alpha)
            forecast_values = forecast_result.predicted_mean

            if confidence_intervals:
                conf_int = forecast_result.conf_int()
            else:
                conf_int = None

            # Crea indice previsione
            # Fallback se training_data non è disponibile (modello caricato da disco)
            if self.training_data is None or len(self.training_data) == 0:
                # Usa indice numerico semplice
                forecast_index = range(steps)
            else:
                last_date = self.training_data.index[-1]
                if isinstance(last_date, pd.Timestamp):
                    freq = pd.infer_freq(self.training_data.index)
                    if freq:
                        try:
                            # Converte frequenza stringa in DateOffset e aggiunge al timestamp
                            freq_offset = pd.tseries.frequencies.to_offset(freq)
                            forecast_index = pd.date_range(
                                start=last_date + freq_offset, periods=steps, freq=freq
                            )
                        except Exception:
                            # Fallback: usa frequenza giornaliera
                            forecast_index = pd.date_range(
                                start=last_date + pd.Timedelta(days=1), periods=steps, freq="D"
                            )
                    else:
                        # Fallback: usa frequenza giornaliera se non può essere inferita
                        forecast_index = pd.date_range(
                            start=last_date + pd.Timedelta(days=1), periods=steps, freq="D"
                        )
                else:
                    forecast_index = range(len(self.training_data), len(self.training_data) + steps)

            forecast_series = pd.Series(forecast_values, index=forecast_index, name="forecast")

            self.logger.info(
                f"Previsione SARIMA generata: {forecast_series.iloc[0]:.2f} a {forecast_series.iloc[-1]:.2f}"
            )

            if return_conf_int and conf_int is not None:
                conf_int.index = forecast_index
                return forecast_series, conf_int
            else:
                return forecast_series

        except Exception as e:
            self.logger.error(f"Previsione SARIMA fallita: {e}")
            raise ForecastError(f"Impossibile generare il forecast SARIMA: {e}")

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
            dynamic: Se usare predizione dinamica

        Returns:
            Serie di predizioni
        """
        try:
            if self.fitted_model is None:
                raise ForecastError(
                    "Il modello SARIMA deve essere addestrato prima della predizione"
                )

            predictions = self.fitted_model.predict(start=start, end=end, dynamic=dynamic)

            return predictions

        except Exception as e:
            self.logger.error(f"Predizione SARIMA fallita: {e}")
            raise ForecastError(f"Impossibile generare le predizioni SARIMA: {e}")

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Salva il modello SARIMA addestrato su disco.

        Args:
            filepath: Percorso per salvare il modello
        """
        try:
            if self.fitted_model is None:
                raise ModelTrainingError("Nessun modello SARIMA addestrato da salvare")

            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Salva usando il metodo integrato di statsmodels
            self.fitted_model.save(str(filepath))

            # Salva anche i metadati (inclusi training_data per forecasting)
            metadata_path = filepath.with_suffix(".metadata.pkl")
            with open(metadata_path, "wb") as f:
                pickle.dump(
                    {
                        "order": self.order,
                        "seasonal_order": self.seasonal_order,
                        "trend": self.trend,
                        "training_metadata": self.training_metadata,
                        "training_data": self.training_data,
                    },
                    f,
                )

            self.logger.info(f"SARIMA model saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Impossibile salvare il modello SARIMA: {e}")
            raise ModelTrainingError(f"Impossibile salvare il modello SARIMA: {e}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "SARIMAForecaster":
        """
        Carica modello SARIMA addestrato da disco.

        Args:
            filepath: Percorso del modello salvato

        Returns:
            Istanza SARIMAForecaster caricata
        """
        try:
            filepath = Path(filepath)

            # Carica il modello addestrato
            fitted_model = SARIMAXResults.load(str(filepath))

            # Carica metadati se disponibili
            metadata_path = filepath.with_suffix(".metadata.pkl")
            if metadata_path.exists():
                with open(metadata_path, "rb") as f:
                    metadata = pickle.load(f)
                order = metadata.get("order", (1, 1, 1))
                seasonal_order = metadata.get("seasonal_order", (1, 1, 1, 12))
                trend = metadata.get("trend", None)
                training_metadata = metadata.get("training_metadata", {})
                training_data = metadata.get("training_data", None)
            else:
                order = (1, 1, 1)  # Ordine predefinito
                seasonal_order = (1, 1, 1, 12)  # Ordine stagionale predefinito
                trend = None
                training_metadata = {}
                training_data = None

            # Crea istanza e popola
            instance = cls(order=order, seasonal_order=seasonal_order, trend=trend)
            instance.fitted_model = fitted_model
            instance.training_metadata = training_metadata
            instance.training_data = training_data

            instance.logger.info(f"SARIMA model loaded from {filepath}")

            return instance

        except Exception as e:
            logger = get_logger(__name__)
            logger.error(f"Impossibile caricare il modello SARIMA: {e}")
            raise ModelTrainingError(f"Impossibile caricare il modello SARIMA: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Ottieni informazioni complete del modello SARIMA.

        Returns:
            Dizionario con informazioni del modello
        """
        if self.fitted_model is None:
            return {"status": "not_fitted"}

        info = {
            "status": "fitted",
            "model_type": "SARIMA",
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            "trend": self.trend,
            "aic": self.fitted_model.aic,
            "bic": self.fitted_model.bic,
            "hqic": self.fitted_model.hqic,
            "llf": self.fitted_model.llf,
            "n_observations": self.fitted_model.nobs,
            "params": dict(self.fitted_model.params),
            "training_metadata": self.training_metadata,
        }

        return info

    def get_seasonal_decomposition(self) -> Dict[str, pd.Series]:
        """
        Ottieni decomposizione stagionale del modello addestrato.

        Returns:
            Dizionario con componenti di decomposizione
        """
        if self.fitted_model is None:
            raise ForecastError(
                "Il modello SARIMA deve essere addestrato prima della decomposizione"
            )

        try:
            from statsmodels.tsa.seasonal import seasonal_decompose

            decomposition = seasonal_decompose(
                self.training_data,
                model="additive",
                period=self.seasonal_order[3],  # periodo stagionale
            )

            return {
                "observed": decomposition.observed,
                "trend": decomposition.trend,
                "seasonal": decomposition.seasonal,
                "residual": decomposition.resid,
            }

        except Exception as e:
            self.logger.error(f"Decomposizione stagionale fallita: {e}")
            raise ForecastError(f"Impossibile eseguire la decomposizione stagionale: {e}")

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
            self.logger.warning(
                "La serie ha meno di 10 osservazioni, il modello potrebbe essere inaffidabile"
            )

        if series.isnull().any():
            missing_pct = series.isnull().sum() / len(series) * 100
            self.logger.warning(f"La serie contiene {missing_pct:.1f}% valori mancanti")

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
        if hasattr(series.index, "freq") and series.index.freq is not None:
            freq = series.index.freq
            if "D" in str(freq) and seasonal_period not in [7, 30, 365]:
                self.logger.warning(
                    f"Dati giornalieri con periodo stagionale {seasonal_period} potrebbero non essere appropriati. "
                    "Considera 7 (settimanale), 30 (mensile), o 365 (annuale)."
                )
            elif "M" in str(freq) and seasonal_period != 12:
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
    ) -> Path:
        """
        Generate comprehensive Quarto report for the SARIMA model analysis.

        Args:
            plots_data: Dictionary with plot file paths {'plot_name': 'path/to/plot.png'}
            report_title: Custom title for the report
            output_filename: Custom filename for the report
            format_type: Output format ('html', 'pdf', 'docx')
            include_diagnostics: Whether to include model diagnostics
            include_forecast: Whether to include forecast analysis
            forecast_steps: Number of steps to forecast for the report
            include_seasonal_decomposition: Whether to include seasonal decomposition

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
                raise ModelTrainingError(
                    "Il modello deve essere addestrato prima di generare il report"
                )

            # Set default title
            if report_title is None:
                report_title = f"Analisi Modello SARIMA{self.order}x{self.seasonal_order}"

            # Collect model results
            model_results = {
                "model_type": "SARIMA",
                "order": self.order,
                "seasonal_order": self.seasonal_order,
                "trend": self.trend,
                "model_info": self.get_model_info(),
                "training_data": {
                    "start_date": str(self.training_metadata.get("training_start", "N/A")),
                    "end_date": str(self.training_metadata.get("training_end", "N/A")),
                    "observations": self.training_metadata.get("training_observations", 0),
                },
            }

            # Add seasonal decomposition if requested
            if include_seasonal_decomposition and self.training_data is not None:
                try:
                    decomposition = self.get_seasonal_decomposition()
                    if decomposition is not None:
                        model_results["seasonal_decomposition"] = {
                            "trend_mean": float(np.nanmean(decomposition["trend"].dropna())),
                            "seasonal_amplitude": float(
                                np.nanstd(decomposition["seasonal"].dropna())
                            ),
                            "residual_variance": float(
                                np.nanvar(decomposition["residual"].dropna())
                            ),
                        }
                except Exception as e:
                    self.logger.warning(
                        f"Non è stato possibile includere la decomposizione stagionale: {e}"
                    )

            # Add metrics if training data is available
            if self.training_data is not None and len(self.training_data) > 0:
                evaluator = ModelEvaluator()

                # Get in-sample predictions for evaluation
                predictions = self.predict()

                # Calculate metrics
                if len(predictions) == len(self.training_data):
                    metrics = evaluator.calculate_forecast_metrics(
                        actual=self.training_data, predicted=predictions
                    )
                    model_results["metrics"] = metrics

                # Add diagnostics if requested
                if include_diagnostics:
                    try:
                        diagnostics = evaluator.evaluate_residuals(
                            residuals=self.fitted_model.resid
                        )
                        model_results["diagnostics"] = diagnostics
                    except Exception as e:
                        self.logger.warning(f"Non è stato possibile calcolare i diagnostici: {e}")

            # Add forecast if requested
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

            # Add technical information
            model_results.update(
                {
                    "python_version": f"{np.__version__}",
                    "environment": "ARIMA Forecaster Library",
                    "generation_timestamp": pd.Timestamp.now().isoformat(),
                }
            )

            # Generate report
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
