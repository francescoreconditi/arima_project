"""
Implementazione dei metodi di riconciliazione per forecast gerarchici.

Include metodi classici (bottom-up, top-down, middle-out) e
metodi ottimali (MinT, OLS, WLS, WLSS).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from enum import Enum
from abc import ABC, abstractmethod
from scipy import linalg
from ..utils.logger import get_logger
from ..utils.exceptions import ForecastError
from .structures import HierarchicalStructure, HierarchyNode


logger = get_logger(__name__)


class ReconciliationMethod(Enum):
    """Metodi di riconciliazione disponibili."""

    BOTTOM_UP = "bottom_up"
    TOP_DOWN = "top_down"
    TOP_DOWN_PROPORTIONS = "top_down_proportions"
    MIDDLE_OUT = "middle_out"
    OLS = "ols"  # Ordinary Least Squares
    WLS = "wls"  # Weighted Least Squares
    MINT_SHRINK = "mint_shrink"  # MinT con shrinkage estimator
    MINT_SAMPLE = "mint_sample"  # MinT con sample covariance
    MINT_DIAGONAL = "mint_diagonal"  # MinT con diagonale
    CUSTOM = "custom"


class BaseReconciler(ABC):
    """
    Classe base astratta per reconciler.
    """

    def __init__(self, hierarchy: HierarchicalStructure, method: ReconciliationMethod):
        """
        Inizializza il reconciler.

        Args:
            hierarchy: Struttura gerarchica
            method: Metodo di riconciliazione
        """
        self.hierarchy = hierarchy
        self.method = method
        self._validate_hierarchy()

    def _validate_hierarchy(self):
        """Valida la gerarchia prima della riconciliazione."""
        is_valid, errors = self.hierarchy.validate_hierarchy()
        if not is_valid:
            raise ValueError(f"Gerarchia non valida: {'; '.join(errors)}")

    @abstractmethod
    def reconcile(
        self, forecasts: Union[np.ndarray, pd.DataFrame], **kwargs
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Riconcilia le previsioni.

        Args:
            forecasts: Previsioni base (tutti i livelli)
            **kwargs: Parametri specifici del metodo

        Returns:
            Previsioni riconciliate
        """
        pass

    def _ensure_numpy(self, forecasts: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Converte input in numpy array."""
        if isinstance(forecasts, pd.DataFrame):
            return forecasts.values
        return np.asarray(forecasts)

    def _restore_dataframe(
        self, reconciled: np.ndarray, original: Union[np.ndarray, pd.DataFrame]
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Ripristina formato DataFrame se necessario."""
        if isinstance(original, pd.DataFrame):
            return pd.DataFrame(reconciled, index=original.index, columns=original.columns)
        return reconciled


class BottomUpReconciler(BaseReconciler):
    """
    Riconciliazione Bottom-Up.

    Usa le previsioni dei livelli più bassi (foglie) e le aggrega
    verso l'alto attraverso la gerarchia.
    """

    def __init__(self, hierarchy: HierarchicalStructure):
        super().__init__(hierarchy, ReconciliationMethod.BOTTOM_UP)

    def reconcile(
        self, forecasts: Union[np.ndarray, pd.DataFrame], **kwargs
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Riconcilia usando approccio bottom-up.

        Args:
            forecasts: Previsioni base (shape: [n_nodes, n_periods])

        Returns:
            Previsioni riconciliate
        """
        forecasts_np = self._ensure_numpy(forecasts)
        S = self.hierarchy.get_summing_matrix()

        # Estrai previsioni bottom level
        leaves = self.hierarchy.get_leaves()
        leaf_ids = {leaf.id for leaf in leaves}
        all_nodes = list(self.hierarchy.nodes.keys())

        # Indici dei nodi foglia
        bottom_indices = [i for i, node_id in enumerate(all_nodes) if node_id in leaf_ids]

        # Previsioni bottom level
        bottom_forecasts = forecasts_np[bottom_indices, :]

        # Riconcilia: moltiplica matrice S per previsioni bottom
        reconciled = S @ bottom_forecasts

        logger.info(f"Bottom-up reconciliation completata: {reconciled.shape}")

        return self._restore_dataframe(reconciled, forecasts)


class TopDownReconciler(BaseReconciler):
    """
    Riconciliazione Top-Down.

    Usa le previsioni del livello più alto e le disaggrega verso il basso
    usando proporzioni storiche o specificate.
    """

    def __init__(self, hierarchy: HierarchicalStructure):
        super().__init__(hierarchy, ReconciliationMethod.TOP_DOWN)

    def reconcile(
        self,
        forecasts: Union[np.ndarray, pd.DataFrame],
        proportions: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        historical_data: Optional[pd.DataFrame] = None,
        proportion_method: str = "average",
        **kwargs,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Riconcilia usando approccio top-down.

        Args:
            forecasts: Previsioni base
            proportions: Proporzioni per disaggregazione (se None, calcola da historical_data)
            historical_data: Dati storici per calcolare proporzioni
            proportion_method: Metodo per calcolare proporzioni ('average', 'last', 'weighted')

        Returns:
            Previsioni riconciliate
        """
        forecasts_np = self._ensure_numpy(forecasts)

        # Se non sono fornite proporzioni, calcolale dai dati storici
        if proportions is None:
            if historical_data is None:
                raise ValueError("Servono proporzioni o dati storici per top-down")
            proportions = self._calculate_proportions(historical_data, proportion_method)
        else:
            proportions = self._ensure_numpy(proportions)

        # Trova indice del nodo root
        roots = [n for n in self.hierarchy.nodes.values() if n.is_root()]
        if not roots:
            raise ValueError("Nessun nodo root trovato")
        root_id = roots[0].id

        all_nodes = list(self.hierarchy.nodes.keys())
        root_idx = all_nodes.index(root_id)

        # Previsione top level
        top_forecast = forecasts_np[root_idx, :]

        # Inizializza risultato
        n_nodes = len(all_nodes)
        n_periods = forecasts_np.shape[1]
        reconciled = np.zeros((n_nodes, n_periods))

        # Imposta previsione root
        reconciled[root_idx, :] = top_forecast

        # Disaggrega ricorsivamente
        self._disaggregate_recursive(root_id, top_forecast, proportions, reconciled, all_nodes)

        logger.info(f"Top-down reconciliation completata")

        return self._restore_dataframe(reconciled, forecasts)

    def _calculate_proportions(
        self, historical_data: pd.DataFrame, method: str = "average"
    ) -> np.ndarray:
        """
        Calcola proporzioni dai dati storici.

        Args:
            historical_data: Dati storici (shape: [n_nodes, n_historical_periods])
            method: Metodo di calcolo ('average', 'last', 'weighted')

        Returns:
            Matrice proporzioni
        """
        if method == "average":
            # Media delle proporzioni storiche
            totals = historical_data.sum(axis=0)
            proportions = historical_data / totals
            proportions = proportions.mean(axis=1).values

        elif method == "last":
            # Proporzioni dell'ultimo periodo
            last_total = historical_data.iloc[:, -1].sum()
            proportions = historical_data.iloc[:, -1] / last_total
            proportions = proportions.values

        elif method == "weighted":
            # Media pesata (pesi maggiori a periodi recenti)
            n_periods = historical_data.shape[1]
            weights = np.exp(np.linspace(-1, 0, n_periods))
            weights /= weights.sum()

            weighted_props = []
            for i in range(n_periods):
                total = historical_data.iloc[:, i].sum()
                props = historical_data.iloc[:, i] / total
                weighted_props.append(props * weights[i])

            proportions = np.sum(weighted_props, axis=0)
        else:
            raise ValueError(f"Metodo proporzioni non supportato: {method}")

        # Normalizza per sicurezza
        proportions = np.nan_to_num(proportions, 0)
        proportions = proportions / proportions.sum() if proportions.sum() > 0 else proportions

        return proportions

    def _disaggregate_recursive(
        self,
        node_id: str,
        node_forecast: np.ndarray,
        proportions: np.ndarray,
        reconciled: np.ndarray,
        all_nodes: List[str],
    ):
        """
        Disaggrega ricorsivamente le previsioni.

        Args:
            node_id: ID del nodo corrente
            node_forecast: Previsione del nodo
            proportions: Proporzioni per disaggregazione
            reconciled: Array risultati da popolare
            all_nodes: Lista tutti i nodi
        """
        node = self.hierarchy.nodes[node_id]

        if not node.children_ids:
            return  # Nodo foglia

        # Calcola quote per i figli
        children_sum = sum(
            proportions[all_nodes.index(child_id)]
            for child_id in node.children_ids
            if child_id in all_nodes
        )

        if children_sum == 0:
            # Distribuisci equamente se non ci sono proporzioni
            children_sum = len(node.children_ids)
            for child_id in node.children_ids:
                if child_id in all_nodes:
                    idx = all_nodes.index(child_id)
                    proportions[idx] = 1.0 / children_sum

        # Disaggrega ai figli
        for child_id in node.children_ids:
            if child_id not in all_nodes:
                continue

            child_idx = all_nodes.index(child_id)
            child_proportion = proportions[child_idx] / children_sum if children_sum > 0 else 0
            child_forecast = node_forecast * child_proportion

            reconciled[child_idx, :] = child_forecast

            # Ricorsione
            self._disaggregate_recursive(
                child_id, child_forecast, proportions, reconciled, all_nodes
            )


class MiddleOutReconciler(BaseReconciler):
    """
    Riconciliazione Middle-Out.

    Parte da un livello intermedio e riconcilia sia verso l'alto
    (aggregazione) che verso il basso (disaggregazione).
    """

    def __init__(self, hierarchy: HierarchicalStructure):
        super().__init__(hierarchy, ReconciliationMethod.MIDDLE_OUT)

    def reconcile(
        self,
        forecasts: Union[np.ndarray, pd.DataFrame],
        middle_level: int,
        proportions: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Riconcilia usando approccio middle-out.

        Args:
            forecasts: Previsioni base
            middle_level: Livello da cui partire
            proportions: Proporzioni per disaggregazione

        Returns:
            Previsioni riconciliate
        """
        forecasts_np = self._ensure_numpy(forecasts)

        # Ottieni nodi del livello intermedio
        middle_nodes = self.hierarchy.get_level_nodes(middle_level)
        if not middle_nodes:
            raise ValueError(f"Nessun nodo trovato al livello {middle_level}")

        all_nodes = list(self.hierarchy.nodes.keys())
        n_nodes = len(all_nodes)
        n_periods = forecasts_np.shape[1]

        reconciled = np.zeros((n_nodes, n_periods))

        # Parti dal middle level
        for node in middle_nodes:
            node_idx = all_nodes.index(node.id)
            reconciled[node_idx, :] = forecasts_np[node_idx, :]

        # Aggrega verso l'alto
        for level in range(middle_level - 1, -1, -1):
            level_nodes = self.hierarchy.get_level_nodes(level)
            for node in level_nodes:
                node_idx = all_nodes.index(node.id)
                # Somma i figli
                child_sum = np.zeros(n_periods)
                for child_id in node.children_ids:
                    if child_id in all_nodes:
                        child_idx = all_nodes.index(child_id)
                        child_sum += reconciled[child_idx, :]
                reconciled[node_idx, :] = child_sum

        # Disaggrega verso il basso
        max_level = max(self.hierarchy.levels.keys())
        for level in range(middle_level + 1, max_level + 1):
            level_nodes = self.hierarchy.get_level_nodes(level)
            for node in level_nodes:
                if node.parent_id and node.parent_id in all_nodes:
                    parent_idx = all_nodes.index(node.parent_id)
                    parent_forecast = reconciled[parent_idx, :]

                    # Calcola proporzione (semplificata)
                    parent_node = self.hierarchy.nodes[node.parent_id]
                    n_siblings = len(parent_node.children_ids)
                    node_idx = all_nodes.index(node.id)

                    # Usa proporzione uniforme o fornita
                    if proportions is not None and node_idx < len(proportions):
                        proportion = proportions[node_idx]
                    else:
                        proportion = 1.0 / n_siblings

                    reconciled[node_idx, :] = parent_forecast * proportion

        logger.info(f"Middle-out reconciliation completata dal livello {middle_level}")

        return self._restore_dataframe(reconciled, forecasts)


class OptimalReconciler(BaseReconciler):
    """
    Riconciliazione ottimale usando metodi statistici.

    Include OLS, WLS e varianti MinT.
    """

    def __init__(self, hierarchy: HierarchicalStructure, method: ReconciliationMethod):
        if method not in [
            ReconciliationMethod.OLS,
            ReconciliationMethod.WLS,
            ReconciliationMethod.MINT_SHRINK,
            ReconciliationMethod.MINT_SAMPLE,
            ReconciliationMethod.MINT_DIAGONAL,
        ]:
            raise ValueError(f"Metodo {method} non supportato per OptimalReconciler")
        super().__init__(hierarchy, method)

    def reconcile(
        self,
        forecasts: Union[np.ndarray, pd.DataFrame],
        errors: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        residuals: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        weights: Optional[np.ndarray] = None,
        lambda_shrink: float = 0.1,
        **kwargs,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Riconcilia usando metodi ottimali.

        Args:
            forecasts: Previsioni base
            errors: Errori di previsione storici (per WLS/MinT)
            residuals: Residui in-sample (per MinT)
            weights: Pesi custom (per WLS)
            lambda_shrink: Parametro shrinkage per MinT

        Returns:
            Previsioni riconciliate
        """
        forecasts_np = self._ensure_numpy(forecasts)

        # Ottieni matrice di coerenza
        S = self.hierarchy.get_summing_matrix()
        n_bottom = S.shape[1]
        n_total = S.shape[0]

        # Calcola matrice dei pesi W
        if self.method == ReconciliationMethod.OLS:
            W = np.eye(n_bottom)  # Identità per OLS

        elif self.method == ReconciliationMethod.WLS:
            if weights is not None:
                W = np.diag(weights[:n_bottom])
            elif errors is not None:
                # Usa varianza degli errori come pesi
                errors_np = self._ensure_numpy(errors)
                variances = np.var(errors_np[:n_bottom, :], axis=1)
                W = np.diag(1.0 / (variances + 1e-10))  # Inverso varianza
            else:
                logger.warning("Nessun peso fornito per WLS, uso OLS")
                W = np.eye(n_bottom)

        elif self.method in [
            ReconciliationMethod.MINT_SHRINK,
            ReconciliationMethod.MINT_SAMPLE,
            ReconciliationMethod.MINT_DIAGONAL,
        ]:
            W = self._estimate_mint_weights(
                residuals if residuals is not None else errors, lambda_shrink
            )
        else:
            W = np.eye(n_bottom)

        # Calcola matrice di riconciliazione G
        # G = (S'WS)^(-1) S'W
        # Assicurati che W abbia la dimensione corretta per S
        if W.shape[0] != S.shape[0]:
            # Se W è per solo bottom level, espandilo per tutte le righe
            if W.shape[0] == n_bottom:
                W_full = np.eye(S.shape[0])
                W_full[-n_bottom:, -n_bottom:] = W
                W = W_full

        SWS = S.T @ W @ S

        # Aggiungi regolarizzazione per stabilità numerica
        SWS_reg = SWS + np.eye(n_bottom) * 1e-10

        try:
            SWS_inv = linalg.inv(SWS_reg)
        except linalg.LinAlgError:
            logger.warning("Matrice singolare, uso pseudo-inversa")
            SWS_inv = linalg.pinv(SWS_reg)

        G = SWS_inv @ S.T @ W

        # Riconcilia: y_tilde = S * G * y_hat
        reconciled = S @ G @ forecasts_np

        # Verifica coerenza
        bottom_reconciled = reconciled[-n_bottom:, :]
        aggregated = S @ bottom_reconciled
        coherence_error = np.mean(np.abs(reconciled - aggregated))

        if coherence_error > 1e-6:
            logger.warning(f"Errore di coerenza: {coherence_error:.2e}")

        logger.info(f"{self.method.value} reconciliation completata")

        return self._restore_dataframe(reconciled, forecasts)

    def _estimate_mint_weights(
        self, residuals: Optional[Union[np.ndarray, pd.DataFrame]], lambda_shrink: float
    ) -> np.ndarray:
        """
        Stima matrice dei pesi per MinT.

        Args:
            residuals: Residui storici
            lambda_shrink: Parametro di shrinkage

        Returns:
            Matrice dei pesi W
        """
        if residuals is None:
            logger.warning("Nessun residuo fornito per MinT, uso identità")
            n_bottom = self.hierarchy.get_summing_matrix().shape[1]
            return np.eye(n_bottom)

        residuals_np = self._ensure_numpy(residuals)
        n_bottom = self.hierarchy.get_summing_matrix().shape[1]

        # Estrai residui bottom level
        bottom_residuals = residuals_np[-n_bottom:, :]

        if self.method == ReconciliationMethod.MINT_SAMPLE:
            # Sample covariance
            cov_matrix = np.cov(bottom_residuals)

        elif self.method == ReconciliationMethod.MINT_DIAGONAL:
            # Solo diagonale (varianze)
            variances = np.var(bottom_residuals, axis=1)
            cov_matrix = np.diag(variances)

        elif self.method == ReconciliationMethod.MINT_SHRINK:
            # Shrinkage estimator (Ledoit-Wolf style)
            sample_cov = np.cov(bottom_residuals)

            # Target: matrice diagonale con varianze medie
            avg_variance = np.mean(np.diag(sample_cov))
            target = np.eye(n_bottom) * avg_variance

            # Shrinkage
            cov_matrix = (1 - lambda_shrink) * sample_cov + lambda_shrink * target
        else:
            cov_matrix = np.eye(n_bottom)

        # Inverti per ottenere pesi (con regolarizzazione)
        cov_reg = cov_matrix + np.eye(n_bottom) * 1e-10

        try:
            W = linalg.inv(cov_reg)
        except linalg.LinAlgError:
            logger.warning("Covarianza singolare, uso pseudo-inversa")
            W = linalg.pinv(cov_reg)

        return W


class HierarchicalReconciler:
    """
    Classe principale per riconciliazione gerarchica.

    Interfaccia unificata per tutti i metodi di riconciliazione.
    """

    def __init__(self, hierarchy: HierarchicalStructure):
        """
        Inizializza il reconciler principale.

        Args:
            hierarchy: Struttura gerarchica
        """
        self.hierarchy = hierarchy
        self.reconcilers = {
            ReconciliationMethod.BOTTOM_UP: BottomUpReconciler(hierarchy),
            ReconciliationMethod.TOP_DOWN: TopDownReconciler(hierarchy),
            ReconciliationMethod.MIDDLE_OUT: MiddleOutReconciler(hierarchy),
            ReconciliationMethod.OLS: OptimalReconciler(hierarchy, ReconciliationMethod.OLS),
            ReconciliationMethod.WLS: OptimalReconciler(hierarchy, ReconciliationMethod.WLS),
            ReconciliationMethod.MINT_SHRINK: OptimalReconciler(
                hierarchy, ReconciliationMethod.MINT_SHRINK
            ),
            ReconciliationMethod.MINT_SAMPLE: OptimalReconciler(
                hierarchy, ReconciliationMethod.MINT_SAMPLE
            ),
            ReconciliationMethod.MINT_DIAGONAL: OptimalReconciler(
                hierarchy, ReconciliationMethod.MINT_DIAGONAL
            ),
        }

    def reconcile(
        self,
        forecasts: Union[np.ndarray, pd.DataFrame],
        method: Union[ReconciliationMethod, str] = ReconciliationMethod.OLS,
        **kwargs,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Riconcilia le previsioni usando il metodo specificato.

        Args:
            forecasts: Previsioni base
            method: Metodo di riconciliazione
            **kwargs: Parametri specifici del metodo

        Returns:
            Previsioni riconciliate
        """
        if isinstance(method, str):
            method = ReconciliationMethod(method)

        if method not in self.reconcilers:
            raise ValueError(f"Metodo {method} non supportato")

        reconciler = self.reconcilers[method]
        return reconciler.reconcile(forecasts, **kwargs)

    def evaluate_methods(
        self,
        forecasts: Union[np.ndarray, pd.DataFrame],
        actuals: Union[np.ndarray, pd.DataFrame],
        methods: Optional[List[ReconciliationMethod]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Valuta diversi metodi di riconciliazione.

        Args:
            forecasts: Previsioni base
            actuals: Valori reali
            methods: Lista metodi da valutare (default: tutti)

        Returns:
            DataFrame con metriche di valutazione
        """
        if methods is None:
            methods = list(self.reconcilers.keys())

        results = []
        actuals_np = self._ensure_numpy(actuals)

        for method in methods:
            try:
                reconciled = self.reconcile(forecasts, method, **kwargs)
                reconciled_np = self._ensure_numpy(reconciled)

                # Calcola metriche
                mae = np.mean(np.abs(reconciled_np - actuals_np))
                rmse = np.sqrt(np.mean((reconciled_np - actuals_np) ** 2))
                mape = np.mean(np.abs((reconciled_np - actuals_np) / (actuals_np + 1e-10))) * 100

                # Verifica coerenza
                S = self.hierarchy.get_summing_matrix()
                n_bottom = S.shape[1]
                bottom_rec = reconciled_np[-n_bottom:, :]
                aggregated = S @ bottom_rec
                coherence_error = np.mean(np.abs(reconciled_np - aggregated))

                results.append(
                    {
                        "method": method.value,
                        "mae": mae,
                        "rmse": rmse,
                        "mape": mape,
                        "coherence_error": coherence_error,
                    }
                )

            except Exception as e:
                logger.error(f"Errore con metodo {method}: {e}")
                results.append(
                    {
                        "method": method.value,
                        "mae": np.nan,
                        "rmse": np.nan,
                        "mape": np.nan,
                        "coherence_error": np.nan,
                    }
                )

        return pd.DataFrame(results).sort_values("rmse")

    def _ensure_numpy(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Converte in numpy array."""
        if isinstance(data, pd.DataFrame):
            return data.values
        return np.asarray(data)


# Alias per retrocompatibilità
MinTReconciler = OptimalReconciler
