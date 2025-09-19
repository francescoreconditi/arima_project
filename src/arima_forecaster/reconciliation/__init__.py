"""
Modulo Forecast Reconciliation per gerarchie temporali.

Implementa metodi di riconciliazione per garantire coerenza matematica
tra previsioni a diversi livelli di aggregazione gerarchica.
"""

from .structures import (
    HierarchicalStructure,
    GroupedStructure,
    TemporalHierarchy,
    GeographicalHierarchy,
    ProductHierarchy,
    HierarchyNode,
    HierarchyLevel,
)

from .reconcilers import (
    HierarchicalReconciler,
    BottomUpReconciler,
    TopDownReconciler,
    MiddleOutReconciler,
    OptimalReconciler,
    MinTReconciler,
    ReconciliationMethod,
)

from .validators import HierarchyValidator, CoherenceChecker, ReconciliationDiagnostics

from .utils import (
    build_summing_matrix,
    compute_coherency_matrix,
    aggregate_forecasts,
    disaggregate_forecasts,
    calculate_proportions,
)

__all__ = [
    # Structures
    "HierarchicalStructure",
    "GroupedStructure",
    "TemporalHierarchy",
    "GeographicalHierarchy",
    "ProductHierarchy",
    "HierarchyNode",
    "HierarchyLevel",
    # Reconcilers
    "HierarchicalReconciler",
    "BottomUpReconciler",
    "TopDownReconciler",
    "MiddleOutReconciler",
    "OptimalReconciler",
    "MinTReconciler",
    "ReconciliationMethod",
    # Validators
    "HierarchyValidator",
    "CoherenceChecker",
    "ReconciliationDiagnostics",
    # Utils
    "build_summing_matrix",
    "compute_coherency_matrix",
    "aggregate_forecasts",
    "disaggregate_forecasts",
    "calculate_proportions",
]
