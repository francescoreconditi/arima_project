"""
Selezione automatica di parametri per modelli Facebook Prophet.

Questo modulo implementa la ricerca dei parametri ottimali per Prophet,
inclusi grid search, random search e ottimizzazione bayesiana.
"""

import itertools
import logging
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error

from arima_forecaster.core.prophet_model import ProphetForecaster
from arima_forecaster.utils.exceptions import ModelTrainingError
from arima_forecaster.utils.logger import get_logger

logger = get_logger(__name__)

# Suppress Prophet warnings during selection
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)


class ProphetModelSelector:
    """
    Selezione automatica parametri per modelli Facebook Prophet.

    Supporta diversi metodi di ricerca:
    - Grid search: Ricerca esaustiva su griglia parametri
    - Random search: Campionamento casuale parametri
    - Cross-validation: Validazione temporale per modelli robusti

    Attributes:
        param_grid: Griglia parametri da testare
        scoring: Metrica per valutazione modelli
        cv_enabled: Se abilitare cross-validation
        n_jobs: Numero processi paralleli
        verbose: Livello verbosity
    """

    def __init__(
        self,
        changepoint_prior_scales: List[float] = [0.001, 0.01, 0.05, 0.1, 0.5],
        seasonality_prior_scales: List[float] = [0.01, 0.1, 1.0, 10.0],
        holidays_prior_scales: List[float] = [0.01, 0.1, 1.0, 10.0],
        seasonality_modes: List[str] = ["additive", "multiplicative"],
        growth_modes: List[str] = ["linear"],
        yearly_seasonalities: List[Union[bool, str]] = ["auto", True, False],
        weekly_seasonalities: List[Union[bool, str]] = ["auto", True, False],
        daily_seasonalities: List[Union[bool, str]] = ["auto", False],
        n_changepoints_range: List[int] = [10, 15, 25, 30],
        scoring: str = "mape",
        cv_enabled: bool = True,
        cv_horizon: str = "30 days",
        cv_initial: str = "365 days",
        cv_period: str = "90 days",
        n_jobs: int = 1,
        max_models: int = 50,
        timeout_minutes: int = 30,
        verbose: bool = True,
    ):
        """
        Inizializza ProphetModelSelector.

        Args:
            changepoint_prior_scales: Scale per changepoint detection
            seasonality_prior_scales: Scale per stagionalità
            holidays_prior_scales: Scale per holidays
            seasonality_modes: Modalità stagionalità ('additive', 'multiplicative')
            growth_modes: Modalità crescita ('linear', 'logistic')
            yearly_seasonalities: Configurazioni stagionalità annuale
            weekly_seasonalities: Configurazioni stagionalità settimanale
            daily_seasonalities: Configurazioni stagionalità giornaliera
            n_changepoints_range: Range numero changepoints
            scoring: Metrica scoring ('mape', 'mae', 'rmse')
            cv_enabled: Se abilitare cross-validation
            cv_horizon: Orizzonte forecast per CV
            cv_initial: Periodo iniziale training per CV
            cv_period: Periodo tra cutoff CV
            n_jobs: Processi paralleli (-1 = tutti)
            max_models: Massimo numero modelli da testare
            timeout_minutes: Timeout totale ricerca
            verbose: Output dettagliato
        """
        self.changepoint_prior_scales = changepoint_prior_scales
        self.seasonality_prior_scales = seasonality_prior_scales
        self.holidays_prior_scales = holidays_prior_scales
        self.seasonality_modes = seasonality_modes
        self.growth_modes = growth_modes
        self.yearly_seasonalities = yearly_seasonalities
        self.weekly_seasonalities = weekly_seasonalities
        self.daily_seasonalities = daily_seasonalities
        self.n_changepoints_range = n_changepoints_range

        self.scoring = scoring
        self.cv_enabled = cv_enabled
        self.cv_horizon = cv_horizon
        self.cv_initial = cv_initial
        self.cv_period = cv_period
        self.n_jobs = n_jobs if n_jobs != -1 else None
        self.max_models = max_models
        self.timeout_minutes = timeout_minutes
        self.verbose = verbose

        # Stato interno
        self.search_results_: List[Dict[str, Any]] = []
        self.best_model_: Optional[ProphetForecaster] = None
        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_score_: Optional[float] = None
        self._search_completed = False

    def search(
        self,
        series: pd.Series,
        exog: Optional[pd.DataFrame] = None,
        country_holidays: Optional[str] = None,
        custom_holidays: Optional[pd.DataFrame] = None,
        method: str = "grid_search",
    ) -> Tuple[ProphetForecaster, List[Dict[str, Any]]]:
        """
        Esegue ricerca parametri ottimali per Prophet.

        Args:
            series: Serie temporale per training
            exog: Variabili esogene (regressori)
            country_holidays: Codice paese per holidays automatici
            custom_holidays: Holiday personalizzati
            method: Metodo ricerca ('grid_search', 'random_search')

        Returns:
            Tupla (miglior_modello, tutti_risultati)
        """
        start_time = time.time()

        if self.verbose:
            logger.info(f"Starting Prophet parameter search using {method}")
            logger.info(f"Series length: {len(series)}, Max models: {self.max_models}")

        try:
            if method == "grid_search":
                results = self._grid_search(series, exog, country_holidays, custom_holidays)
            elif method == "random_search":
                results = self._random_search(series, exog, country_holidays, custom_holidays)
            elif method == "bayesian":
                results = self._bayesian_search(series, exog, country_holidays, custom_holidays)
            else:
                raise ValueError(
                    f"Unknown search method: {method}. Use: 'grid_search', 'random_search', 'bayesian'"
                )

            if not results:
                raise ModelTrainingError("No valid models found during search")

            # Trova miglior modello
            best_result = min(results, key=lambda x: x["score"])

            # Addestra miglior modello finale
            best_model = self._create_model(
                best_result["params"], country_holidays, custom_holidays
            )
            best_model.fit(series, exog)

            # Salva risultati
            self.search_results_ = results
            self.best_model_ = best_model
            self.best_params_ = best_result["params"]
            self.best_score_ = best_result["score"]
            self._search_completed = True

            elapsed = time.time() - start_time
            if self.verbose:
                logger.info(f"Search completed in {elapsed:.1f}s")
                logger.info(f"Best {self.scoring}: {best_result['score']:.4f}")
                logger.info(f"Best params: {best_result['params']}")

            return best_model, results

        except Exception as e:
            raise ModelTrainingError(f"Prophet parameter search failed: {e}")

    def _grid_search(
        self,
        series: pd.Series,
        exog: Optional[pd.DataFrame],
        country_holidays: Optional[str],
        custom_holidays: Optional[pd.DataFrame],
    ) -> List[Dict[str, Any]]:
        """Esegue grid search completa."""

        # Genera tutte le combinazioni parametri
        param_combinations = list(
            itertools.product(
                self.changepoint_prior_scales,
                self.seasonality_prior_scales,
                self.holidays_prior_scales,
                self.seasonality_modes,
                self.growth_modes,
                self.yearly_seasonalities,
                self.weekly_seasonalities,
                self.daily_seasonalities,
                self.n_changepoints_range,
            )
        )

        # Limita numero combinazioni
        if len(param_combinations) > self.max_models:
            import random

            random.shuffle(param_combinations)
            param_combinations = param_combinations[: self.max_models]

        if self.verbose:
            logger.info(f"Testing {len(param_combinations)} parameter combinations")

        # Valuta ogni combinazione
        results = []

        if self.n_jobs == 1:
            # Sequenziale
            for i, params in enumerate(param_combinations):
                result = self._evaluate_params(
                    params, series, exog, country_holidays, custom_holidays
                )
                if result is not None:
                    results.append(result)

                if self.verbose and (i + 1) % 10 == 0:
                    logger.info(f"Evaluated {i + 1}/{len(param_combinations)} models")
        else:
            # Parallelo
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = {
                    executor.submit(
                        self._evaluate_params,
                        params,
                        series,
                        exog,
                        country_holidays,
                        custom_holidays,
                    ): params
                    for params in param_combinations
                }

                for i, future in enumerate(as_completed(futures)):
                    result = future.result()
                    if result is not None:
                        results.append(result)

                    if self.verbose and (i + 1) % 10 == 0:
                        logger.info(f"Completed {i + 1}/{len(param_combinations)} models")

        return results

    def _random_search(
        self,
        series: pd.Series,
        exog: Optional[pd.DataFrame],
        country_holidays: Optional[str],
        custom_holidays: Optional[pd.DataFrame],
    ) -> List[Dict[str, Any]]:
        """Esegue random search."""

        results = []
        n_trials = min(self.max_models, 100)  # Limite ragionevole

        if self.verbose:
            logger.info(f"Random search with {n_trials} trials")

        for i in range(n_trials):
            # Campiona parametri casuali
            params = self._sample_random_params()

            result = self._evaluate_params(params, series, exog, country_holidays, custom_holidays)
            if result is not None:
                results.append(result)

            if self.verbose and (i + 1) % 20 == 0:
                logger.info(f"Random search: {i + 1}/{n_trials} trials")

        return results

    def _bayesian_search(
        self,
        series: pd.Series,
        exog: Optional[pd.DataFrame],
        country_holidays: Optional[str],
        custom_holidays: Optional[pd.DataFrame],
    ) -> List[Dict[str, Any]]:
        """Esegue Bayesian optimization con Optuna."""
        try:
            import optuna
            from optuna.samplers import TPESampler

            # Sopprimi log Optuna se non verbose
            if not self.verbose:
                optuna.logging.set_verbosity(optuna.logging.WARNING)

        except ImportError:
            if self.verbose:
                logger.warning("Optuna non installato, fallback a random search")
            return self._random_search(series, exog, country_holidays, custom_holidays)

        results = []
        best_score = float("inf")

        def objective(trial):
            nonlocal results, best_score

            # Suggerisci parametri da Optuna
            params = (
                trial.suggest_categorical("changepoint_prior_scale", self.changepoint_prior_scales),
                trial.suggest_categorical("seasonality_prior_scale", self.seasonality_prior_scales),
                trial.suggest_categorical("holidays_prior_scale", self.holidays_prior_scales),
                trial.suggest_categorical("seasonality_mode", self.seasonality_modes),
                trial.suggest_categorical("growth", self.growth_modes),
                trial.suggest_categorical("yearly_seasonality", self.yearly_seasonalities),
                trial.suggest_categorical("weekly_seasonality", self.weekly_seasonalities),
                trial.suggest_categorical("daily_seasonality", self.daily_seasonalities),
                trial.suggest_categorical("n_changepoints", self.n_changepoints_range),
            )

            # Valuta parametri
            result = self._evaluate_params(params, series, exog, country_holidays, custom_holidays)

            if result is not None:
                results.append(result)
                score = result["score"]

                # Aggiorna migliore
                if score < best_score:
                    best_score = score

                return score
            else:
                # Penalizza parametri che causano errori
                return float("inf")

        # Crea studio Optuna
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=42, n_startup_trials=min(10, self.max_models // 4)),
        )

        # Ottimizzazione
        if self.verbose:
            logger.info(f"Bayesian optimization with Optuna TPE: {self.max_models} trials")

        study.optimize(
            objective,
            n_trials=self.max_models,
            timeout=self.timeout_minutes * 60 if self.timeout_minutes else None,
            show_progress_bar=self.verbose,
        )

        if self.verbose:
            logger.info(
                f"Bayesian search completed: {len(study.trials)} trials, best score: {best_score:.4f}"
            )

        return results

    def _sample_random_params(self) -> Tuple:
        """Campiona parametri casuali dalle distribuzioni."""
        import random

        return (
            random.choice(self.changepoint_prior_scales),
            random.choice(self.seasonality_prior_scales),
            random.choice(self.holidays_prior_scales),
            random.choice(self.seasonality_modes),
            random.choice(self.growth_modes),
            random.choice(self.yearly_seasonalities),
            random.choice(self.weekly_seasonalities),
            random.choice(self.daily_seasonalities),
            random.choice(self.n_changepoints_range),
        )

    def _evaluate_params(
        self,
        params: Tuple,
        series: pd.Series,
        exog: Optional[pd.DataFrame],
        country_holidays: Optional[str],
        custom_holidays: Optional[pd.DataFrame],
    ) -> Optional[Dict[str, Any]]:
        """Valuta un set di parametri."""

        (
            changepoint_prior_scale,
            seasonality_prior_scale,
            holidays_prior_scale,
            seasonality_mode,
            growth,
            yearly_seasonality,
            weekly_seasonality,
            daily_seasonality,
            n_changepoints,
        ) = params

        param_dict = {
            "changepoint_prior_scale": changepoint_prior_scale,
            "seasonality_prior_scale": seasonality_prior_scale,
            "holidays_prior_scale": holidays_prior_scale,
            "seasonality_mode": seasonality_mode,
            "growth": growth,
            "yearly_seasonality": yearly_seasonality,
            "weekly_seasonality": weekly_seasonality,
            "daily_seasonality": daily_seasonality,
            "n_changepoints": n_changepoints,
        }

        try:
            # Crea e addestra modello
            model = self._create_model(param_dict, country_holidays, custom_holidays)

            if exog is not None:
                for col in exog.columns:
                    model.add_regressor(col)

            model.fit(series, exog)

            # Calcola score
            if self.cv_enabled and len(series) > 365:  # CV solo per serie lunghe
                score = self._cross_validation_score(model, series, exog)
            else:
                score = self._simple_score(model, series)

            return {
                "params": param_dict,
                "score": score,
                "method": "cv" if self.cv_enabled and len(series) > 365 else "simple",
            }

        except Exception as e:
            if self.verbose:
                logger.debug(f"Model failed with params {param_dict}: {e}")
            return None

    def _create_model(
        self,
        params: Dict[str, Any],
        country_holidays: Optional[str],
        custom_holidays: Optional[pd.DataFrame],
    ) -> ProphetForecaster:
        """Crea modello Prophet con parametri specificati."""

        return ProphetForecaster(
            growth=params["growth"],
            n_changepoints=params["n_changepoints"],
            yearly_seasonality=params["yearly_seasonality"],
            weekly_seasonality=params["weekly_seasonality"],
            daily_seasonality=params["daily_seasonality"],
            seasonality_mode=params["seasonality_mode"],
            seasonality_prior_scale=params["seasonality_prior_scale"],
            holidays_prior_scale=params["holidays_prior_scale"],
            changepoint_prior_scale=params["changepoint_prior_scale"],
            holidays=custom_holidays,
            country_holidays=country_holidays,
        )

    def _cross_validation_score(
        self, model: ProphetForecaster, series: pd.Series, exog: Optional[pd.DataFrame]
    ) -> float:
        """Calcola score tramite cross-validation temporale."""

        try:
            # Prepara dati per Prophet CV
            df = pd.DataFrame({"ds": series.index, "y": series.values})
            if exog is not None:
                for col in exog.columns:
                    df[col] = exog[col].values

            # Esegui cross-validation
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cv_results = cross_validation(
                    model.model,
                    initial=self.cv_initial,
                    period=self.cv_period,
                    horizon=self.cv_horizon,
                    parallel="threads" if self.n_jobs and self.n_jobs > 1 else None,
                )

            # Calcola metriche
            metrics = performance_metrics(cv_results)

            if self.scoring == "mape":
                return metrics["mape"].mean()
            elif self.scoring == "mae":
                return metrics["mae"].mean()
            elif self.scoring == "rmse":
                return metrics["rmse"].mean()
            else:
                return metrics["mape"].mean()  # Default

        except Exception as e:
            logger.debug(f"CV failed, using simple score: {e}")
            return self._simple_score(model, series)

    def _simple_score(self, model: ProphetForecaster, series: pd.Series) -> float:
        """Calcola score semplice (in-sample)."""

        try:
            fitted = model.predict()

            if self.scoring == "mae":
                return mean_absolute_error(series, fitted)
            elif self.scoring == "rmse":
                return np.sqrt(mean_squared_error(series, fitted))
            elif self.scoring == "mape":
                return np.mean(np.abs((series - fitted) / series)) * 100
            else:
                return np.mean(np.abs((series - fitted) / series)) * 100

        except Exception as e:
            logger.debug(f"Simple score failed: {e}")
            return float("inf")  # Penalizza modelli che falliscono

    def get_best_model(self) -> ProphetForecaster:
        """
        Restituisce il miglior modello trovato.

        Returns:
            Miglior modello addestrato

        Raises:
            ModelTrainingError: Se ricerca non completata
        """
        if not self._search_completed:
            raise ModelTrainingError("Must run search() before getting best model")

        return self.best_model_

    def get_search_results(self) -> List[Dict[str, Any]]:
        """
        Restituisce tutti i risultati della ricerca.

        Returns:
            Lista con risultati di tutti i modelli testati
        """
        return self.search_results_

    def get_top_models(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Restituisce i top N modelli.

        Args:
            n: Numero di modelli da restituire

        Returns:
            Lista con top N risultati ordinati per score
        """
        if not self.search_results_:
            return []

        sorted_results = sorted(self.search_results_, key=lambda x: x["score"])
        return sorted_results[:n]

    def get_best_params(self) -> Dict[str, Any]:
        """
        Restituisce i migliori parametri trovati.

        Returns:
            Dizionario con migliori parametri

        Raises:
            ModelTrainingError: Se ricerca non completata
        """
        if not self._search_completed:
            raise ModelTrainingError("Must run search() before getting best params")

        return self.best_params_.copy()

    def get_best_score(self) -> float:
        """
        Restituisce il miglior score trovato.

        Returns:
            Miglior score della ricerca

        Raises:
            ModelTrainingError: Se ricerca non completata
        """
        if not self._search_completed:
            raise ModelTrainingError("Must run search() before getting best score")

        return self.best_score_

    def summary(self) -> str:
        """
        Restituisce riassunto della ricerca.

        Returns:
            String con riassunto risultati
        """
        if not self._search_completed:
            return "Search not completed yet"

        summary_lines = [
            f"Prophet Parameter Search Summary",
            f"=" * 40,
            f"Models evaluated: {len(self.search_results_)}",
            f"Scoring metric: {self.scoring}",
            f"Best score: {self.best_score_:.4f}",
            f"",
            f"Best parameters:",
        ]

        for key, value in self.best_params_.items():
            summary_lines.append(f"  {key}: {value}")

        # Top 3 modelli
        top_3 = self.get_top_models(3)
        if len(top_3) > 1:
            summary_lines.extend(
                [
                    f"",
                    f"Top 3 models:",
                ]
            )
            for i, result in enumerate(top_3[:3], 1):
                summary_lines.append(f"  {i}. {self.scoring}={result['score']:.4f}")

        return "\n".join(summary_lines)

    def __repr__(self) -> str:
        """Rappresentazione string del selector."""
        if self._search_completed:
            return f"ProphetModelSelector(completed, best_{self.scoring}={self.best_score_:.4f})"
        else:
            return f"ProphetModelSelector(max_models={self.max_models}, scoring={self.scoring})"
