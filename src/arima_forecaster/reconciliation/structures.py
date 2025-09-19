"""
Strutture gerarchiche per Forecast Reconciliation.

Definisce le classi per rappresentare diverse tipologie di gerarchie
(prodotti, geografiche, temporali) e le loro relazioni.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import networkx as nx
from ..utils.logger import get_logger


logger = get_logger(__name__)


class HierarchyType(Enum):
    """Tipi di gerarchia supportati."""

    PRODUCT = "product"
    GEOGRAPHICAL = "geographical"
    TEMPORAL = "temporal"
    GROUPED = "grouped"
    CUSTOM = "custom"


class AggregationMethod(Enum):
    """Metodi di aggregazione per i nodi."""

    SUM = "sum"
    MEAN = "mean"
    WEIGHTED_MEAN = "weighted_mean"
    CUSTOM = "custom"


@dataclass
class HierarchyNode:
    """
    Rappresenta un nodo nella gerarchia.

    Attributes:
        id: Identificatore univoco del nodo
        name: Nome descrittivo
        level: Livello nella gerarchia (0 = root)
        parent_id: ID del nodo padre
        children_ids: Lista ID dei nodi figli
        weight: Peso per aggregazione weighted
        metadata: Metadati aggiuntivi
    """

    id: str
    name: str
    level: int
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_leaf(self) -> bool:
        """Verifica se il nodo è una foglia (senza figli)."""
        return len(self.children_ids) == 0

    def is_root(self) -> bool:
        """Verifica se il nodo è la radice."""
        return self.parent_id is None


@dataclass
class HierarchyLevel:
    """
    Rappresenta un livello nella gerarchia.

    Attributes:
        level_id: ID del livello
        name: Nome del livello
        nodes: Lista dei nodi a questo livello
        aggregation_method: Metodo di aggregazione
    """

    level_id: int
    name: str
    nodes: List[HierarchyNode] = field(default_factory=list)
    aggregation_method: AggregationMethod = AggregationMethod.SUM


class HierarchicalStructure(ABC):
    """
    Classe base astratta per strutture gerarchiche.
    """

    def __init__(self, name: str, hierarchy_type: HierarchyType = HierarchyType.CUSTOM):
        """
        Inizializza la struttura gerarchica.

        Args:
            name: Nome della gerarchia
            hierarchy_type: Tipo di gerarchia
        """
        self.name = name
        self.hierarchy_type = hierarchy_type
        self.nodes: Dict[str, HierarchyNode] = {}
        self.levels: Dict[int, HierarchyLevel] = {}
        self.graph = nx.DiGraph()
        self._summing_matrix: Optional[np.ndarray] = None
        self._coherency_matrix: Optional[np.ndarray] = None

    @abstractmethod
    def build_hierarchy(self, data: Union[pd.DataFrame, Dict]) -> None:
        """
        Costruisce la gerarchia dai dati.

        Args:
            data: Dati per costruire la gerarchia
        """
        pass

    def add_node(self, node: HierarchyNode) -> None:
        """
        Aggiunge un nodo alla gerarchia.

        Args:
            node: Nodo da aggiungere
        """
        self.nodes[node.id] = node
        self.graph.add_node(node.id, **node.__dict__)

        # Aggiungi al livello appropriato
        if node.level not in self.levels:
            self.levels[node.level] = HierarchyLevel(
                level_id=node.level, name=f"Level_{node.level}"
            )
        self.levels[node.level].nodes.append(node)

        # Aggiungi edge se ha un padre
        if node.parent_id:
            self.graph.add_edge(node.parent_id, node.id)
            # Aggiorna children del padre
            if node.parent_id in self.nodes:
                self.nodes[node.parent_id].children_ids.append(node.id)

    def get_leaves(self) -> List[HierarchyNode]:
        """
        Restituisce tutti i nodi foglia.

        Returns:
            Lista dei nodi foglia
        """
        return [node for node in self.nodes.values() if node.is_leaf()]

    def get_level_nodes(self, level: int) -> List[HierarchyNode]:
        """
        Restituisce tutti i nodi di un livello specifico.

        Args:
            level: Livello da recuperare

        Returns:
            Lista dei nodi al livello specificato
        """
        return self.levels.get(level, HierarchyLevel(level, f"Level_{level}")).nodes

    def get_ancestors(self, node_id: str) -> List[str]:
        """
        Restituisce tutti gli antenati di un nodo.

        Args:
            node_id: ID del nodo

        Returns:
            Lista degli ID degli antenati
        """
        try:
            return list(nx.ancestors(self.graph, node_id))
        except nx.NetworkXError:
            return []

    def get_descendants(self, node_id: str) -> List[str]:
        """
        Restituisce tutti i discendenti di un nodo.

        Args:
            node_id: ID del nodo

        Returns:
            Lista degli ID dei discendenti
        """
        try:
            return list(nx.descendants(self.graph, node_id))
        except nx.NetworkXError:
            return []

    def get_summing_matrix(self) -> np.ndarray:
        """
        Restituisce la matrice di aggregazione S.

        La matrice S mappa i nodi foglia ai nodi aggregati.
        S[i,j] = 1 se il nodo foglia j contribuisce al nodo i.

        Returns:
            Matrice di aggregazione
        """
        if self._summing_matrix is not None:
            return self._summing_matrix

        leaves = self.get_leaves()
        all_nodes = list(self.nodes.keys())

        n_total = len(all_nodes)
        n_bottom = len(leaves)

        S = np.zeros((n_total, n_bottom))

        # Mappa indici
        node_to_idx = {node_id: i for i, node_id in enumerate(all_nodes)}
        leaf_to_idx = {leaf.id: i for i, leaf in enumerate(leaves)}

        # Riempi la matrice
        for i, node_id in enumerate(all_nodes):
            if node_id in leaf_to_idx:
                # Nodo foglia contribuisce a se stesso
                S[i, leaf_to_idx[node_id]] = 1
            else:
                # Nodo aggregato: somma dei discendenti foglia
                descendants = self.get_descendants(node_id)
                for desc_id in descendants:
                    if desc_id in leaf_to_idx:
                        S[i, leaf_to_idx[desc_id]] = 1

        self._summing_matrix = S
        return S

    def get_coherency_matrix(self) -> np.ndarray:
        """
        Restituisce la matrice di coerenza C.

        C combina la matrice identità per i bottom level con
        la matrice di aggregazione per i livelli superiori.

        Returns:
            Matrice di coerenza
        """
        if self._coherency_matrix is not None:
            return self._coherency_matrix

        S = self.get_summing_matrix()
        n_bottom = S.shape[1]

        # Matrice di coerenza: [I | A]^T dove A sono le righe aggregate
        # Separa bottom level da aggregati
        leaves = self.get_leaves()
        leaf_ids = {leaf.id for leaf in leaves}

        all_nodes = list(self.nodes.keys())
        bottom_indices = [i for i, node_id in enumerate(all_nodes) if node_id in leaf_ids]
        agg_indices = [i for i, node_id in enumerate(all_nodes) if node_id not in leaf_ids]

        # Riordina S: prima bottom, poi aggregati
        S_reordered = np.vstack([S[bottom_indices, :], S[agg_indices, :]])

        self._coherency_matrix = S_reordered
        return S_reordered

    def validate_hierarchy(self) -> Tuple[bool, List[str]]:
        """
        Valida la struttura della gerarchia.

        Returns:
            (is_valid, lista_errori)
        """
        errors = []

        # Verifica che ci sia almeno un nodo root
        roots = [n for n in self.nodes.values() if n.is_root()]
        if len(roots) == 0:
            errors.append("Nessun nodo root trovato")
        elif len(roots) > 1:
            errors.append(f"Trovati {len(roots)} nodi root, dovrebbe essercene solo uno")

        # Verifica che non ci siano cicli
        if not nx.is_directed_acyclic_graph(self.graph):
            errors.append("La gerarchia contiene cicli")

        # Verifica connettività
        if not nx.is_weakly_connected(self.graph):
            errors.append("La gerarchia non è connessa")

        # Verifica coerenza parent-children
        for node in self.nodes.values():
            for child_id in node.children_ids:
                if child_id not in self.nodes:
                    errors.append(f"Nodo {node.id} ha figlio {child_id} non esistente")
                elif self.nodes[child_id].parent_id != node.id:
                    errors.append(f"Incoerenza parent-child tra {node.id} e {child_id}")

        is_valid = len(errors) == 0
        return is_valid, errors


class ProductHierarchy(HierarchicalStructure):
    """
    Gerarchia per prodotti (es. SKU -> Sottocategoria -> Categoria -> Totale).
    """

    def __init__(self, name: str = "Product Hierarchy"):
        super().__init__(name, HierarchyType.PRODUCT)

    def build_hierarchy(self, data: pd.DataFrame) -> None:
        """
        Costruisce gerarchia prodotti da DataFrame.

        Args:
            data: DataFrame con colonne rappresentanti i livelli
                  (es. 'total', 'category', 'subcategory', 'sku')
        """
        # Identifica colonne dei livelli (ordinate dal più aggregato al più dettagliato)
        level_columns = []
        for col in ["total", "division", "category", "subcategory", "product", "sku"]:
            if col in data.columns:
                level_columns.append(col)

        if not level_columns:
            raise ValueError("Nessuna colonna di gerarchia trovata nel DataFrame")

        # Crea nodi per ogni livello
        for level_idx, level_col in enumerate(level_columns):
            unique_values = data[level_col].unique()

            for value in unique_values:
                node_id = f"{level_col}_{value}"

                # Trova il parent
                parent_id = None
                if level_idx > 0:
                    parent_col = level_columns[level_idx - 1]
                    parent_values = data[data[level_col] == value][parent_col].unique()
                    if len(parent_values) == 1:
                        parent_id = f"{parent_col}_{parent_values[0]}"

                node = HierarchyNode(
                    id=node_id, name=str(value), level=level_idx, parent_id=parent_id
                )

                self.add_node(node)

        logger.info(
            f"Creata gerarchia prodotti con {len(self.nodes)} nodi su {len(self.levels)} livelli"
        )


class GeographicalHierarchy(HierarchicalStructure):
    """
    Gerarchia geografica (es. Store -> Città -> Regione -> Paese).
    """

    def __init__(self, name: str = "Geographical Hierarchy"):
        super().__init__(name, HierarchyType.GEOGRAPHICAL)

    def build_hierarchy(self, data: pd.DataFrame) -> None:
        """
        Costruisce gerarchia geografica da DataFrame.

        Args:
            data: DataFrame con colonne geografiche
                  (es. 'country', 'region', 'city', 'store')
        """
        # Identifica colonne geografiche
        geo_columns = []
        for col in [
            "world",
            "continent",
            "country",
            "region",
            "province",
            "city",
            "district",
            "store",
            "location",
        ]:
            if col in data.columns:
                geo_columns.append(col)

        if not geo_columns:
            raise ValueError("Nessuna colonna geografica trovata nel DataFrame")

        # Crea gerarchia come per prodotti
        for level_idx, geo_col in enumerate(geo_columns):
            unique_values = data[geo_col].unique()

            for value in unique_values:
                node_id = f"{geo_col}_{value}"

                parent_id = None
                if level_idx > 0:
                    parent_col = geo_columns[level_idx - 1]
                    parent_values = data[data[geo_col] == value][parent_col].unique()
                    if len(parent_values) == 1:
                        parent_id = f"{parent_col}_{parent_values[0]}"

                node = HierarchyNode(
                    id=node_id,
                    name=str(value),
                    level=level_idx,
                    parent_id=parent_id,
                    metadata={"geo_type": geo_col},
                )

                self.add_node(node)

        logger.info(f"Creata gerarchia geografica con {len(self.nodes)} nodi")


class TemporalHierarchy(HierarchicalStructure):
    """
    Gerarchia temporale (es. Giorno -> Settimana -> Mese -> Trimestre -> Anno).
    """

    def __init__(self, name: str = "Temporal Hierarchy"):
        super().__init__(name, HierarchyType.TEMPORAL)

    def build_hierarchy(self, data: Union[pd.DatetimeIndex, pd.DataFrame]) -> None:
        """
        Costruisce gerarchia temporale da DatetimeIndex o DataFrame.

        Args:
            data: DatetimeIndex o DataFrame con colonna datetime
        """
        if isinstance(data, pd.DataFrame):
            if "date" in data.columns:
                dates = pd.to_datetime(data["date"])
            elif "datetime" in data.columns:
                dates = pd.to_datetime(data["datetime"])
            else:
                dates = pd.to_datetime(data.iloc[:, 0])
        else:
            dates = pd.to_datetime(data)

        # Crea livelli temporali
        temporal_data = pd.DataFrame(
            {
                "date": dates,
                "year": dates.year,
                "quarter": dates.quarter,
                "month": dates.month,
                "week": dates.isocalendar().week,
                "day": dates.day,
            }
        )

        # Livello 0: Totale
        total_node = HierarchyNode(id="total", name="Total", level=0)
        self.add_node(total_node)

        # Livello 1: Anni
        for year in temporal_data["year"].unique():
            year_node = HierarchyNode(id=f"year_{year}", name=str(year), level=1, parent_id="total")
            self.add_node(year_node)

            # Livello 2: Trimestri
            year_data = temporal_data[temporal_data["year"] == year]
            for quarter in year_data["quarter"].unique():
                quarter_node = HierarchyNode(
                    id=f"year_{year}_quarter_{quarter}",
                    name=f"{year}-Q{quarter}",
                    level=2,
                    parent_id=f"year_{year}",
                )
                self.add_node(quarter_node)

                # Livello 3: Mesi
                quarter_data = year_data[year_data["quarter"] == quarter]
                for month in quarter_data["month"].unique():
                    month_node = HierarchyNode(
                        id=f"year_{year}_month_{month}",
                        name=f"{year}-{month:02d}",
                        level=3,
                        parent_id=f"year_{year}_quarter_{quarter}",
                    )
                    self.add_node(month_node)

        logger.info(f"Creata gerarchia temporale con {len(self.nodes)} nodi")


class GroupedStructure(HierarchicalStructure):
    """
    Struttura per gerarchie raggruppate (non strettamente gerarchiche).
    Supporta raggruppamenti multipli e cross-funzionali.
    """

    def __init__(self, name: str = "Grouped Structure"):
        super().__init__(name, HierarchyType.GROUPED)
        self.grouping_sets: List[List[str]] = []

    def build_hierarchy(self, data: pd.DataFrame, grouping_sets: List[List[str]]) -> None:
        """
        Costruisce struttura raggruppata.

        Args:
            data: DataFrame con i dati
            grouping_sets: Lista di raggruppamenti
                          es. [['brand'], ['category'], ['brand', 'category']]
        """
        self.grouping_sets = grouping_sets

        # Crea nodo totale
        total_node = HierarchyNode(id="total", name="Total", level=0)
        self.add_node(total_node)

        # Per ogni grouping set, crea i nodi
        for level_idx, group_cols in enumerate(grouping_sets, 1):
            if len(group_cols) == 0:
                continue

            # Trova combinazioni uniche
            unique_groups = data[group_cols].drop_duplicates()

            for _, row in unique_groups.iterrows():
                # Crea ID dal gruppo
                node_id = "_".join([f"{col}_{row[col]}" for col in group_cols])
                node_name = " - ".join([str(row[col]) for col in group_cols])

                node = HierarchyNode(
                    id=node_id,
                    name=node_name,
                    level=level_idx,
                    parent_id="total",  # Tutti collegati al totale
                    metadata={"group_cols": group_cols, "group_values": dict(row)},
                )

                self.add_node(node)

        logger.info(f"Creata struttura raggruppata con {len(self.nodes)} nodi")
