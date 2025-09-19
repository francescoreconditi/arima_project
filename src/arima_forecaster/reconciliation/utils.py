"""
Utility functions per Forecast Reconciliation.

Funzioni di supporto per costruire matrici, aggregare/disaggregare
previsioni e calcolare proporzioni.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from scipy import sparse
from ..utils.logger import get_logger


logger = get_logger(__name__)


def build_summing_matrix(
    hierarchy_df: pd.DataFrame, bottom_level_col: str, aggregation_cols: List[str]
) -> np.ndarray:
    """
    Costruisce la matrice di aggregazione S da un DataFrame.

    La matrice S mappa i nodi foglia (bottom level) ai nodi aggregati.
    Ogni riga rappresenta un nodo (bottom o aggregato), ogni colonna
    rappresenta un nodo foglia.

    Args:
        hierarchy_df: DataFrame con la struttura gerarchica
        bottom_level_col: Nome colonna del livello più basso
        aggregation_cols: Lista colonne di aggregazione (dal più alto al più basso)

    Returns:
        Matrice di aggregazione S

    Example:
        >>> df = pd.DataFrame({
        ...     'total': ['Total'] * 4,
        ...     'category': ['A', 'A', 'B', 'B'],
        ...     'product': ['P1', 'P2', 'P3', 'P4']
        ... })
        >>> S = build_summing_matrix(df, 'product', ['total', 'category'])
        >>> S.shape
        (7, 4)  # 7 nodi totali (1 total + 2 categorie + 4 prodotti), 4 foglie
    """
    # Ottieni nodi bottom level unici
    bottom_nodes = hierarchy_df[bottom_level_col].unique()
    n_bottom = len(bottom_nodes)

    # Crea mapping bottom nodes -> indici
    bottom_to_idx = {node: i for i, node in enumerate(bottom_nodes)}

    # Inizializza liste per costruire la matrice
    rows = []
    node_names = []

    # Aggiungi righe per nodi aggregati
    for agg_col in aggregation_cols:
        agg_nodes = hierarchy_df[agg_col].unique()

        for agg_node in agg_nodes:
            # Trova quali bottom nodes appartengono a questo aggregato
            mask = hierarchy_df[agg_col] == agg_node
            contributing_bottoms = hierarchy_df[mask][bottom_level_col].unique()

            # Crea riga per questo nodo aggregato
            row = np.zeros(n_bottom)
            for bottom in contributing_bottoms:
                row[bottom_to_idx[bottom]] = 1

            rows.append(row)
            node_names.append(f"{agg_col}_{agg_node}")

    # Aggiungi righe per bottom level (matrice identità)
    for i, bottom_node in enumerate(bottom_nodes):
        row = np.zeros(n_bottom)
        row[i] = 1
        rows.append(row)
        node_names.append(f"{bottom_level_col}_{bottom_node}")

    S = np.vstack(rows)

    logger.info(
        f"Matrice S creata: shape {S.shape}, {len(node_names)} nodi totali, {n_bottom} foglie"
    )

    return S


def build_summing_matrix_sparse(
    hierarchy_df: pd.DataFrame, bottom_level_col: str, aggregation_cols: List[str]
) -> sparse.csr_matrix:
    """
    Costruisce la matrice di aggregazione S in formato sparse.

    Utile per gerarchie molto grandi dove la matrice S sarebbe
    principalmente composta da zeri.

    Args:
        hierarchy_df: DataFrame con la struttura gerarchica
        bottom_level_col: Nome colonna del livello più basso
        aggregation_cols: Lista colonne di aggregazione

    Returns:
        Matrice di aggregazione S in formato sparse CSR
    """
    bottom_nodes = hierarchy_df[bottom_level_col].unique()
    n_bottom = len(bottom_nodes)
    bottom_to_idx = {node: i for i, node in enumerate(bottom_nodes)}

    # Liste per costruzione sparse
    row_indices = []
    col_indices = []
    data = []
    row_counter = 0

    # Nodi aggregati
    for agg_col in aggregation_cols:
        agg_nodes = hierarchy_df[agg_col].unique()

        for agg_node in agg_nodes:
            mask = hierarchy_df[agg_col] == agg_node
            contributing_bottoms = hierarchy_df[mask][bottom_level_col].unique()

            for bottom in contributing_bottoms:
                row_indices.append(row_counter)
                col_indices.append(bottom_to_idx[bottom])
                data.append(1)

            row_counter += 1

    # Bottom level (identità)
    for i in range(n_bottom):
        row_indices.append(row_counter + i)
        col_indices.append(i)
        data.append(1)

    n_total = row_counter + n_bottom

    S_sparse = sparse.csr_matrix((data, (row_indices, col_indices)), shape=(n_total, n_bottom))

    logger.info(
        f"Matrice S sparse creata: shape {S_sparse.shape}, {S_sparse.nnz} elementi non-zero"
    )

    return S_sparse


def compute_coherency_matrix(S: np.ndarray) -> np.ndarray:
    """
    Calcola la matrice di coerenza C dalla matrice S.

    La matrice C combina l'identità per bottom level con
    la matrice di aggregazione per i livelli superiori.

    Args:
        S: Matrice di aggregazione

    Returns:
        Matrice di coerenza C
    """
    n_total, n_bottom = S.shape

    # Identifica righe bottom (somma = 1 e un solo elemento = 1)
    is_bottom = np.zeros(n_total, dtype=bool)
    for i in range(n_total):
        row = S[i, :]
        if np.sum(row) == 1 and np.max(row) == 1:
            # Verifica se è una riga identità
            nonzero_idx = np.nonzero(row)[0]
            if len(nonzero_idx) == 1:
                is_bottom[i] = True

    # Riordina: prima bottom, poi aggregati
    bottom_indices = np.where(is_bottom)[0]
    agg_indices = np.where(~is_bottom)[0]

    C = np.vstack([S[bottom_indices, :], S[agg_indices, :]])

    return C


def aggregate_forecasts(
    bottom_forecasts: Union[np.ndarray, pd.DataFrame], S: np.ndarray
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Aggrega previsioni bottom-up usando la matrice S.

    Args:
        bottom_forecasts: Previsioni bottom level (shape: [n_bottom, n_periods])
        S: Matrice di aggregazione

    Returns:
        Previsioni aggregate per tutti i livelli
    """
    if isinstance(bottom_forecasts, pd.DataFrame):
        bottom_np = bottom_forecasts.values
    else:
        bottom_np = bottom_forecasts

    # Moltiplica S per bottom forecasts
    aggregated = S @ bottom_np

    if isinstance(bottom_forecasts, pd.DataFrame):
        # Mantieni formato DataFrame
        n_total = S.shape[0]
        columns = bottom_forecasts.columns if hasattr(bottom_forecasts, "columns") else None
        return pd.DataFrame(aggregated, columns=columns)

    return aggregated


def disaggregate_forecasts(
    top_forecast: Union[np.ndarray, pd.Series],
    proportions: Union[np.ndarray, pd.DataFrame],
    hierarchy_levels: List[int],
) -> np.ndarray:
    """
    Disaggrega previsioni top-down usando proporzioni.

    Args:
        top_forecast: Previsione del livello più alto
        proportions: Proporzioni per disaggregazione
        hierarchy_levels: Numero di nodi per ogni livello

    Returns:
        Previsioni disaggregate per tutti i livelli
    """
    if isinstance(top_forecast, pd.Series):
        top_np = top_forecast.values
    else:
        top_np = np.asarray(top_forecast)

    if isinstance(proportions, pd.DataFrame):
        prop_np = proportions.values
    else:
        prop_np = np.asarray(proportions)

    n_periods = len(top_np) if top_np.ndim == 1 else top_np.shape[1]
    n_nodes = len(prop_np)

    # Inizializza risultato
    disaggregated = np.zeros((n_nodes, n_periods))

    # Il primo nodo è il top level
    disaggregated[0, :] = top_np if top_np.ndim == 1 else top_np.flatten()

    # Disaggrega usando le proporzioni
    for i in range(1, n_nodes):
        disaggregated[i, :] = disaggregated[0, :] * prop_np[i]

    return disaggregated


def calculate_proportions(
    historical_data: pd.DataFrame, method: str = "average", weights: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Calcola proporzioni dai dati storici.

    Args:
        historical_data: Dati storici (righe=nodi, colonne=periodi)
        method: Metodo di calcolo
               - 'average': Media delle proporzioni storiche
               - 'last': Proporzioni dell'ultimo periodo
               - 'weighted': Media pesata (pesi esponenziali)
               - 'custom': Usa pesi forniti
        weights: Pesi custom per metodo 'custom'

    Returns:
        DataFrame con proporzioni per ogni nodo
    """
    n_nodes, n_periods = historical_data.shape

    if method == "average":
        # Media delle proporzioni storiche
        proportions = []
        for period in range(n_periods):
            period_data = historical_data.iloc[:, period]
            total = period_data.sum()
            if total > 0:
                props = period_data / total
            else:
                props = pd.Series(1.0 / n_nodes, index=period_data.index)
            proportions.append(props)

        result = pd.concat(proportions, axis=1).mean(axis=1)

    elif method == "last":
        # Proporzioni dell'ultimo periodo
        last_period = historical_data.iloc[:, -1]
        total = last_period.sum()
        if total > 0:
            result = last_period / total
        else:
            result = pd.Series(1.0 / n_nodes, index=last_period.index)

    elif method == "weighted":
        # Media pesata con pesi esponenziali
        exp_weights = np.exp(np.linspace(-1, 0, n_periods))
        exp_weights /= exp_weights.sum()

        weighted_props = pd.DataFrame(0, index=historical_data.index, columns=["prop"])

        for period in range(n_periods):
            period_data = historical_data.iloc[:, period]
            total = period_data.sum()
            if total > 0:
                props = period_data / total * exp_weights[period]
            else:
                props = pd.Series(1.0 / n_nodes * exp_weights[period], index=period_data.index)

            weighted_props["prop"] += props

        result = weighted_props["prop"]

    elif method == "custom":
        if weights is None:
            raise ValueError("Weights necessari per metodo 'custom'")

        if len(weights) != n_periods:
            raise ValueError(f"Lunghezza weights ({len(weights)}) != periodi ({n_periods})")

        # Normalizza weights
        weights = np.asarray(weights)
        weights /= weights.sum()

        weighted_props = pd.DataFrame(0, index=historical_data.index, columns=["prop"])

        for period in range(n_periods):
            period_data = historical_data.iloc[:, period]
            total = period_data.sum()
            if total > 0:
                props = period_data / total * weights[period]
            else:
                props = pd.Series(1.0 / n_nodes * weights[period], index=period_data.index)

            weighted_props["prop"] += props

        result = weighted_props["prop"]

    else:
        raise ValueError(f"Metodo '{method}' non supportato")

    # Normalizza per sicurezza
    total = result.sum()
    if total > 0:
        result = result / total

    return pd.DataFrame({"proportion": result})


def check_aggregation_constraints(
    forecasts: np.ndarray, S: np.ndarray, tolerance: float = 1e-6
) -> Tuple[bool, pd.DataFrame]:
    """
    Verifica che le previsioni rispettino i vincoli di aggregazione.

    Args:
        forecasts: Previsioni per tutti i nodi
        S: Matrice di aggregazione
        tolerance: Tolleranza per errori numerici

    Returns:
        (is_coherent, dataframe con errori per ogni nodo)
    """
    n_total, n_bottom = S.shape

    # Estrai bottom forecasts (assumendo siano gli ultimi n_bottom)
    bottom_forecasts = forecasts[-n_bottom:, :]

    # Calcola aggregati attesi
    expected_aggregates = S @ bottom_forecasts

    # Calcola errori
    errors = np.abs(forecasts - expected_aggregates)

    # Crea DataFrame con risultati
    error_df = pd.DataFrame(
        {
            "node_id": range(n_total),
            "max_error": np.max(errors, axis=1),
            "mean_error": np.mean(errors, axis=1),
            "is_coherent": np.max(errors, axis=1) < tolerance,
        }
    )

    is_coherent = np.all(error_df["is_coherent"])

    if not is_coherent:
        n_incoherent = (~error_df["is_coherent"]).sum()
        logger.warning(f"{n_incoherent} nodi non coerenti (tolerance={tolerance})")

    return is_coherent, error_df


def create_hierarchy_from_dataframe(
    df: pd.DataFrame,
    hierarchy_cols: List[str],
    value_col: str = "value",
    date_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Crea struttura gerarchica da DataFrame lungo.

    Args:
        df: DataFrame in formato lungo
        hierarchy_cols: Colonne che definiscono la gerarchia
        value_col: Colonna con i valori
        date_col: Colonna data (opzionale)

    Returns:
        DataFrame pivotato con gerarchia

    Example:
        >>> df = pd.DataFrame({
        ...     'region': ['North', 'North', 'South', 'South'],
        ...     'store': ['S1', 'S2', 'S3', 'S4'],
        ...     'date': pd.date_range('2024-01-01', periods=4),
        ...     'sales': [100, 150, 200, 120]
        ... })
        >>> hierarchy_df = create_hierarchy_from_dataframe(
        ...     df, ['region', 'store'], 'sales', 'date'
        ... )
    """
    # Aggrega per ogni livello gerarchico
    aggregated_dfs = []

    # Livello più dettagliato
    if date_col:
        detail_df = df.pivot_table(
            index=hierarchy_cols, columns=date_col, values=value_col, aggfunc="sum"
        )
    else:
        detail_df = df.groupby(hierarchy_cols)[value_col].sum().to_frame()

    aggregated_dfs.append(detail_df)

    # Livelli aggregati
    for i in range(len(hierarchy_cols) - 1, 0, -1):
        agg_cols = hierarchy_cols[:i]

        if date_col:
            agg_df = df.pivot_table(
                index=agg_cols, columns=date_col, values=value_col, aggfunc="sum"
            )
        else:
            agg_df = df.groupby(agg_cols)[value_col].sum().to_frame()

        aggregated_dfs.append(agg_df)

    # Totale
    if date_col:
        total_df = df.pivot_table(columns=date_col, values=value_col, aggfunc="sum").to_frame().T
        total_df.index = ["Total"]
    else:
        total_df = pd.DataFrame({"value": [df[value_col].sum()]}, index=["Total"])

    aggregated_dfs.append(total_df)

    # Combina tutti i livelli
    result = pd.concat(aggregated_dfs, axis=0)

    logger.info(f"Creata gerarchia: {result.shape[0]} nodi totali")

    return result


def optimal_reconciliation_weights(residuals: np.ndarray, method: str = "ols") -> np.ndarray:
    """
    Calcola pesi ottimali per riconciliazione da residui.

    Args:
        residuals: Matrice residui (righe=serie, colonne=tempo)
        method: Metodo per calcolare pesi
               - 'ols': Pesi uniformi (identità)
               - 'wls': Inverse varianze
               - 'mint_diagonal': Solo diagonale covarianza
               - 'mint_full': Covarianza completa

    Returns:
        Matrice dei pesi W
    """
    n_series = residuals.shape[0]

    if method == "ols":
        W = np.eye(n_series)

    elif method == "wls":
        # Inverse varianze
        variances = np.var(residuals, axis=1)
        W = np.diag(1.0 / (variances + 1e-10))

    elif method == "mint_diagonal":
        # Solo elementi diagonali della covarianza
        variances = np.var(residuals, axis=1)
        W = np.diag(1.0 / (variances + 1e-10))

    elif method == "mint_full":
        # Inversa della matrice di covarianza completa
        cov_matrix = np.cov(residuals)
        # Aggiungi regolarizzazione
        cov_reg = cov_matrix + np.eye(n_series) * 1e-10

        try:
            W = np.linalg.inv(cov_reg)
        except np.linalg.LinAlgError:
            logger.warning("Covarianza singolare, uso pseudo-inversa")
            W = np.linalg.pinv(cov_reg)

    else:
        raise ValueError(f"Metodo '{method}' non supportato")

    return W
