"""
Test semplice per Forecast Reconciliation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Aggiungi path per import locale
sys.path.append(str(Path(__file__).parent.parent / "src"))

from arima_forecaster.reconciliation import (
    ProductHierarchy,
    HierarchicalReconciler,
    ReconciliationMethod,
    CoherenceChecker
)


def simple_reconciliation_test():
    """Test semplice di riconciliazione."""

    print("=== TEST SEMPLICE FORECAST RECONCILIATION ===")

    # 1. Crea gerarchia semplice
    hierarchy_df = pd.DataFrame({
        'total': ['Total'] * 2,
        'product': ['A', 'B']
    })

    hierarchy = ProductHierarchy("Simple Hierarchy")
    hierarchy.build_hierarchy(hierarchy_df)

    print(f"Gerarchia creata: {len(hierarchy.nodes)} nodi")

    # 2. Crea previsioni non coerenti
    # Total: 200, Products: A=80, B=60 (somma = 140, non 200!)
    base_forecasts = np.array([
        [200, 210, 220],  # Total
        [80, 85, 90],     # Product A
        [60, 65, 70]      # Product B
    ])

    print("\nPrevisioni base (NON coerenti):")
    print(f"Total: {base_forecasts[0]}")
    print(f"Product A: {base_forecasts[1]}")
    print(f"Product B: {base_forecasts[2]}")
    print(f"Somma A+B: {base_forecasts[1] + base_forecasts[2]}")

    # 3. Test Bottom-Up
    reconciler = HierarchicalReconciler(hierarchy)

    print("\n=== BOTTOM-UP RECONCILIATION ===")
    reconciled_bu = reconciler.reconcile(
        base_forecasts,
        method=ReconciliationMethod.BOTTOM_UP
    )

    print("Previsioni riconciliate (Bottom-Up):")
    print(f"Total: {reconciled_bu[0]}")
    print(f"Product A: {reconciled_bu[1]}")
    print(f"Product B: {reconciled_bu[2]}")
    print(f"Somma A+B: {reconciled_bu[1] + reconciled_bu[2]}")

    # 4. Verifica coerenza
    checker = CoherenceChecker(hierarchy)
    report = checker.check_coherence(reconciled_bu)

    print(f"\nCoerenza check:")
    print(f"- Coerente: {report.is_coherent}")
    print(f"- Errore max: {report.max_error:.2e}")
    print(f"- Errore medio: {report.mean_error:.2e}")

    # 5. Test OLS
    print("\n=== OLS RECONCILIATION ===")
    reconciled_ols = reconciler.reconcile(
        base_forecasts,
        method=ReconciliationMethod.OLS
    )

    print("Previsioni riconciliate (OLS):")
    print(f"Total: {reconciled_ols[0]}")
    print(f"Product A: {reconciled_ols[1]}")
    print(f"Product B: {reconciled_ols[2]}")
    print(f"Somma A+B: {reconciled_ols[1] + reconciled_ols[2]}")

    # Verifica coerenza OLS
    report_ols = checker.check_coherence(reconciled_ols)
    print(f"\nCoerenza check OLS:")
    print(f"- Coerente: {report_ols.is_coherent}")
    print(f"- Errore max: {report_ols.max_error:.2e}")
    print(f"- Errore medio: {report_ols.mean_error:.2e}")

    print("\n=== TEST COMPLETATO ===")


if __name__ == "__main__":
    simple_reconciliation_test()