"""
Selezione automatica del modello SARIMA con ottimizzazione dei parametri stagionali.
"""

import pandas as pd
import numpy as np
import itertools
from typing import List, Tuple, Dict, Any, Optional, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
from tqdm import tqdm

from .sarima_model import SARIMAForecaster
from ..utils.logger import get_logger
from ..utils.exceptions import ModelTrainingError


class SARIMAModelSelector:
    """
    Selezione automatica del modello SARIMA usando grid search con parametri stagionali.
    """

    def __init__(
        self,
        p_range: Tuple[int, int] = (0, 3),
        d_range: Tuple[int, int] = (0, 2),
        q_range: Tuple[int, int] = (0, 3),
        P_range: Tuple[int, int] = (0, 2),
        D_range: Tuple[int, int] = (0, 1),
        Q_range: Tuple[int, int] = (0, 2),
        seasonal_periods: Optional[List[int]] = None,
        information_criterion: str = "aic",
        max_models: Optional[int] = None,
        n_jobs: int = 1,
    ):
        """
        Inizializza il selettore di modelli SARIMA.

        Args:
            p_range: Intervallo valori p (min, max)
            d_range: Intervallo valori d (min, max)
            q_range: Intervallo valori q (min, max)
            P_range: Intervallo valori P stagionali (min, max)
            D_range: Intervallo valori D stagionali (min, max)
            Q_range: Intervallo valori Q stagionali (min, max)
            seasonal_periods: Lista di periodi stagionali da provare (default: [12])
            information_criterion: Criterio per la selezione del modello ('aic', 'bic', 'hqic')
            max_models: Numero massimo di modelli da provare
            n_jobs: Numero di job paralleli per l'addestramento del modello
        """
        self.p_range = p_range
        self.d_range = d_range
        self.q_range = q_range
        self.P_range = P_range
        self.D_range = D_range
        self.Q_range = Q_range
        self.seasonal_periods = seasonal_periods or [12]
        self.information_criterion = information_criterion.lower()
        self.max_models = max_models
        self.n_jobs = n_jobs

        self.results = []
        self.best_model = None
        self.best_order = None
        self.best_seasonal_order = None
        self.logger = get_logger(__name__)

        if self.information_criterion not in ["aic", "bic", "hqic"]:
            raise ValueError("information_criterion deve essere 'aic', 'bic', o 'hqic'")

    def search(
        self, series: pd.Series, verbose: bool = True, suppress_warnings: bool = True
    ) -> "SARIMAModelSelector":
        """
        Esegue grid search per trovare i parametri SARIMA ottimali.

        Args:
            series: Serie temporale da addestrare
            verbose: Se mostrare il progresso
            suppress_warnings: Se sopprimere i warning di statsmodels

        Returns:
            Self per concatenamento dei metodi
        """
        if suppress_warnings:
            warnings.filterwarnings("ignore")

        try:
            self.logger.info(
                f"Avvio selezione modello SARIMA usando {self.information_criterion.upper()}"
            )
            self.logger.info(
                f"Intervalli parametri: p{self.p_range}, d{self.d_range}, q{self.q_range}"
            )
            self.logger.info(
                f"Intervalli stagionali: P{self.P_range}, D{self.D_range}, Q{self.Q_range}"
            )
            self.logger.info(f"Periodi stagionali: {self.seasonal_periods}")

            # Genera tutte le combinazioni di parametri
            param_combinations = self._generate_param_combinations()

            if self.max_models and len(param_combinations) > self.max_models:
                # Campiona casualmente se troppe combinazioni
                np.random.shuffle(param_combinations)
                param_combinations = param_combinations[: self.max_models]
                self.logger.info(f"Limitato a {self.max_models} combinazioni casuali di parametri")

            self.logger.info(f"Test di {len(param_combinations)} combinazioni di parametri")

            # Addestra modelli
            if self.n_jobs == 1:
                # Elaborazione sequenziale
                self.results = self._fit_models_sequential(series, param_combinations, verbose)
            else:
                # Elaborazione parallela
                self.results = self._fit_models_parallel(series, param_combinations, verbose)

            # Trova miglior modello
            if self.results:
                best_result = min(self.results, key=lambda x: x[self.information_criterion])
                self.best_order = best_result["order"]
                self.best_seasonal_order = best_result["seasonal_order"]

                # Addestra il miglior modello
                self.best_model = SARIMAForecaster(
                    order=self.best_order, seasonal_order=self.best_seasonal_order
                )
                self.best_model.fit(series)

                self.logger.info(
                    f"Miglior modello SARIMA: {self.best_order}x{self.best_seasonal_order}"
                )
                self.logger.info(
                    f"Best {self.information_criterion.upper()}: {best_result[self.information_criterion]:.2f}"
                )
            else:
                self.logger.error("Nessun modello addestrato con successo")

            return self

        except Exception as e:
            self.logger.error(f"Selezione modello SARIMA fallita: {e}")
            raise ModelTrainingError(f"Selezione modello SARIMA fallita: {e}")

        finally:
            if suppress_warnings:
                warnings.resetwarnings()

    def _generate_param_combinations(self) -> List[Tuple]:
        """Genera tutte le combinazioni di parametri da testare."""
        combinations = []

        for s in self.seasonal_periods:
            for p in range(self.p_range[0], self.p_range[1] + 1):
                for d in range(self.d_range[0], self.d_range[1] + 1):
                    for q in range(self.q_range[0], self.q_range[1] + 1):
                        for P in range(self.P_range[0], self.P_range[1] + 1):
                            for D in range(self.D_range[0], self.D_range[1] + 1):
                                for Q in range(self.Q_range[0], self.Q_range[1] + 1):
                                    combinations.append(((p, d, q), (P, D, Q, s)))

        return combinations

    def _fit_models_sequential(
        self, series: pd.Series, param_combinations: List[Tuple], verbose: bool
    ) -> List[Dict[str, Any]]:
        """Addestra modelli sequenzialmente."""
        results = []

        iterator = (
            tqdm(param_combinations, desc="Test modelli SARIMA") if verbose else param_combinations
        )

        for order, seasonal_order in iterator:
            try:
                model = SARIMAForecaster(order=order, seasonal_order=seasonal_order)
                model.fit(series, validate_input=False)

                model_info = model.get_model_info()
                results.append(model_info)

                if verbose and hasattr(iterator, "set_postfix"):
                    iterator.set_postfix(
                        {
                            "Best AIC": min(r["aic"] for r in results),
                            "Current": f"{order}x{seasonal_order}",
                        }
                    )

            except Exception as e:
                # Modello fallito nell'addestramento, saltalo
                continue

        return results

    def _fit_models_parallel(
        self, series: pd.Series, param_combinations: List[Tuple], verbose: bool
    ) -> List[Dict[str, Any]]:
        """Addestra modelli in parallelo."""
        results = []

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            # Invia tutti i job
            future_to_params = {
                executor.submit(_fit_sarima_model, series, order, seasonal_order): (
                    order,
                    seasonal_order,
                )
                for order, seasonal_order in param_combinations
            }

            # Raccogli risultati
            iterator = (
                tqdm(
                    as_completed(future_to_params),
                    total=len(param_combinations),
                    desc="Test modelli SARIMA",
                )
                if verbose
                else as_completed(future_to_params)
            )

            for future in iterator:
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)

                        if verbose and hasattr(iterator, "set_postfix"):
                            iterator.set_postfix(
                                {
                                    "Completed": len(results),
                                    "Best AIC": min(r["aic"] for r in results)
                                    if results
                                    else "N/A",
                                }
                            )
                except Exception:
                    # Modello fallito nell'addestramento, saltalo
                    continue

        return results

    def get_best_model(self) -> Optional[SARIMAForecaster]:
        """
        Ottieni il miglior modello SARIMA addestrato.

        Returns:
            Istanza SARIMAForecaster migliore o None se nessun modello addestrato
        """
        return self.best_model

    def get_results_summary(self, top_n: int = 10) -> pd.DataFrame:
        """
        Ottieni riepilogo dei risultati di selezione del modello.

        Args:
            top_n: Numero di migliori modelli da restituire

        Returns:
            DataFrame con risultati dei modelli
        """
        if not self.results:
            return pd.DataFrame()

        # Ordina per criterio di informazione
        sorted_results = sorted(self.results, key=lambda x: x[self.information_criterion])

        # Crea DataFrame di riepilogo
        summary_data = []
        for result in sorted_results[:top_n]:
            summary_data.append(
                {
                    "order": str(result["order"]),
                    "seasonal_order": str(result["seasonal_order"]),
                    "aic": result["aic"],
                    "bic": result["bic"],
                    "hqic": result["hqic"],
                    "n_observations": result["n_observations"],
                }
            )

        return pd.DataFrame(summary_data)

    def plot_selection_results(self, top_n: int = 20) -> None:
        """
        Traccia i risultati della selezione del modello.

        Args:
            top_n: Numero di migliori modelli da tracciare
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            if not self.results:
                self.logger.warning("Nessun risultato da tracciare")
                return

            # Ottieni migliori risultati
            sorted_results = sorted(self.results, key=lambda x: x[self.information_criterion])[
                :top_n
            ]

            # Prepara dati
            model_names = [f"{r['order']}x{r['seasonal_order']}" for r in sorted_results]
            aic_values = [r["aic"] for r in sorted_results]
            bic_values = [r["bic"] for r in sorted_results]
            hqic_values = [r["hqic"] for r in sorted_results]

            # Crea grafico
            fig, ax = plt.subplots(figsize=(12, 8))

            x = np.arange(len(model_names))
            width = 0.25

            ax.bar(x - width, aic_values, width, label="AIC", alpha=0.8)
            ax.bar(x, bic_values, width, label="BIC", alpha=0.8)
            ax.bar(x + width, hqic_values, width, label="HQIC", alpha=0.8)

            ax.set_xlabel("Modelli SARIMA")
            ax.set_ylabel("Valore Criterio di Informazione")
            ax.set_title(f"Confronto Top {top_n} Modelli SARIMA")
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=45, ha="right")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)

            plt.tight_layout()
            plt.show()

        except ImportError:
            self.logger.warning("Matplotlib/Seaborn non disponibili per i grafici")
        except Exception as e:
            self.logger.error(f"Impossibile creare grafico: {e}")


def _fit_sarima_model(
    series: pd.Series, order: Tuple[int, int, int], seasonal_order: Tuple[int, int, int, int]
) -> Optional[Dict[str, Any]]:
    """
    Funzione helper per addestrare un singolo modello SARIMA (per elaborazione parallela).

    Args:
        series: Dati della serie temporale
        order: Ordine ARIMA
        seasonal_order: Ordine ARIMA stagionale

    Returns:
        Dizionario informazioni modello o None se l'addestramento è fallito
    """
    try:
        model = SARIMAForecaster(order=order, seasonal_order=seasonal_order)
        model.fit(series, validate_input=False)
        return model.get_model_info()
    except Exception:
        return None
