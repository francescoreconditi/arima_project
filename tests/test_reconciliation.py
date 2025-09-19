"""
Test per il modulo Forecast Reconciliation.
"""

import pytest
import numpy as np
import pandas as pd
from arima_forecaster.reconciliation import (
    HierarchicalStructure,
    ProductHierarchy,
    GeographicalHierarchy,
    TemporalHierarchy,
    HierarchicalReconciler,
    ReconciliationMethod,
    CoherenceChecker,
    HierarchyValidator,
    build_summing_matrix,
    aggregate_forecasts,
    calculate_proportions
)


class TestHierarchicalStructures:
    """Test per le strutture gerarchiche."""

    def test_product_hierarchy_creation(self):
        """Test creazione gerarchia prodotti."""
        # Crea dati gerarchia
        hierarchy_df = pd.DataFrame({
            'total': ['Total'] * 4,
            'category': ['A', 'A', 'B', 'B'],
            'product': ['P1', 'P2', 'P3', 'P4']
        })

        # Crea gerarchia
        hierarchy = ProductHierarchy()
        hierarchy.build_hierarchy(hierarchy_df)

        # Verifica struttura
        assert len(hierarchy.nodes) == 7  # 1 total + 2 categorie + 4 prodotti
        assert len(hierarchy.levels) == 3
        assert len(hierarchy.get_leaves()) == 4

    def test_geographical_hierarchy_creation(self):
        """Test creazione gerarchia geografica."""
        hierarchy_df = pd.DataFrame({
            'country': ['USA'] * 4,
            'region': ['East', 'East', 'West', 'West'],
            'city': ['NYC', 'Boston', 'LA', 'SF']
        })

        hierarchy = GeographicalHierarchy()
        hierarchy.build_hierarchy(hierarchy_df)

        assert len(hierarchy.nodes) == 7
        assert len(hierarchy.get_leaves()) == 4

    def test_temporal_hierarchy_creation(self):
        """Test creazione gerarchia temporale."""
        dates = pd.date_range('2024-01-01', periods=12, freq='ME')

        hierarchy = TemporalHierarchy()
        hierarchy.build_hierarchy(dates)

        # Dovrebbe avere total, anni, trimestri, mesi
        assert len(hierarchy.nodes) > 0
        assert hierarchy.nodes.get('total') is not None

    def test_summing_matrix_generation(self):
        """Test generazione matrice di aggregazione."""
        hierarchy_df = pd.DataFrame({
            'total': ['Total'] * 2,
            'product': ['P1', 'P2']
        })

        hierarchy = ProductHierarchy()
        hierarchy.build_hierarchy(hierarchy_df)

        S = hierarchy.get_summing_matrix()

        # Verifica dimensioni
        assert S.shape == (3, 2)  # 3 nodi totali, 2 foglie

        # Verifica che la somma delle foglie dia il totale
        assert S[0, 0] == 1  # Total include P1
        assert S[0, 1] == 1  # Total include P2

    def test_hierarchy_validation(self):
        """Test validazione gerarchia."""
        hierarchy_df = pd.DataFrame({
            'total': ['Total'] * 2,
            'product': ['P1', 'P2']
        })

        hierarchy = ProductHierarchy()
        hierarchy.build_hierarchy(hierarchy_df)

        is_valid, errors = hierarchy.validate_hierarchy()

        assert is_valid
        assert len(errors) == 0


class TestReconciliationMethods:
    """Test per i metodi di riconciliazione."""

    def setup_method(self):
        """Setup per ogni test."""
        # Crea gerarchia semplice
        hierarchy_df = pd.DataFrame({
            'total': ['Total'] * 3,
            'category': ['A', 'A', 'B'],
            'product': ['P1', 'P2', 'P3']
        })

        self.hierarchy = ProductHierarchy()
        self.hierarchy.build_hierarchy(hierarchy_df)

        # Crea previsioni di esempio
        n_nodes = len(self.hierarchy.nodes)  # Numero reale di nodi creati
        n_periods = 6
        np.random.seed(42)
        self.base_forecasts = np.random.randn(n_nodes, n_periods) * 10 + 100

    def test_bottom_up_reconciliation(self):
        """Test riconciliazione bottom-up."""
        reconciler = HierarchicalReconciler(self.hierarchy)

        reconciled = reconciler.reconcile(
            self.base_forecasts,
            method=ReconciliationMethod.BOTTOM_UP
        )

        # Verifica forma
        assert reconciled.shape == self.base_forecasts.shape

        # Verifica coerenza
        checker = CoherenceChecker(self.hierarchy)
        report = checker.check_coherence(reconciled)
        assert report.is_coherent or report.mean_error < 1e-6

    def test_top_down_reconciliation(self):
        """Test riconciliazione top-down."""
        reconciler = HierarchicalReconciler(self.hierarchy)

        # Crea dati storici per proporzioni
        historical_data = pd.DataFrame(
            np.random.randn(7, 12) * 10 + 100
        )

        reconciled = reconciler.reconcile(
            self.base_forecasts,
            method=ReconciliationMethod.TOP_DOWN,
            historical_data=historical_data
        )

        assert reconciled.shape == self.base_forecasts.shape

    def test_ols_reconciliation(self):
        """Test riconciliazione OLS."""
        reconciler = HierarchicalReconciler(self.hierarchy)

        reconciled = reconciler.reconcile(
            self.base_forecasts,
            method=ReconciliationMethod.OLS
        )

        assert reconciled.shape == self.base_forecasts.shape

        # Verifica coerenza
        checker = CoherenceChecker(self.hierarchy)
        report = checker.check_coherence(reconciled)
        # OLS dovrebbe garantire coerenza
        assert report.mean_error < 1e-3

    def test_mint_reconciliation(self):
        """Test riconciliazione MinT."""
        reconciler = HierarchicalReconciler(self.hierarchy)

        # Simula residui
        residuals = np.random.randn(7, 24)

        reconciled = reconciler.reconcile(
            self.base_forecasts,
            method=ReconciliationMethod.MINT_DIAGONAL,
            residuals=residuals
        )

        assert reconciled.shape == self.base_forecasts.shape

    def test_method_comparison(self):
        """Test confronto tra metodi."""
        reconciler = HierarchicalReconciler(self.hierarchy)

        # Simula valori attuali per valutazione
        actuals = self.base_forecasts + np.random.randn(*self.base_forecasts.shape) * 2

        results = reconciler.evaluate_methods(
            self.base_forecasts,
            actuals,
            methods=[
                ReconciliationMethod.BOTTOM_UP,
                ReconciliationMethod.OLS
            ]
        )

        assert isinstance(results, pd.DataFrame)
        assert 'mae' in results.columns
        assert 'rmse' in results.columns
        assert len(results) == 2


class TestCoherenceValidation:
    """Test per validazione coerenza."""

    def setup_method(self):
        """Setup per test."""
        hierarchy_df = pd.DataFrame({
            'total': ['Total'] * 2,
            'product': ['P1', 'P2']
        })

        self.hierarchy = ProductHierarchy()
        self.hierarchy.build_hierarchy(hierarchy_df)

    def test_coherence_check_coherent(self):
        """Test verifica coerenza per previsioni coerenti."""
        # Crea previsioni perfettamente coerenti
        bottom_forecasts = np.array([[100, 110, 120],
                                     [50, 55, 60]])
        total_forecast = bottom_forecasts.sum(axis=0).reshape(1, -1)

        all_forecasts = np.vstack([total_forecast, bottom_forecasts])

        checker = CoherenceChecker(self.hierarchy)
        report = checker.check_coherence(all_forecasts)

        assert report.is_coherent
        assert report.mean_error < 1e-10

    def test_coherence_check_incoherent(self):
        """Test verifica coerenza per previsioni incoerenti."""
        # Crea previsioni incoerenti
        all_forecasts = np.array([
            [200, 210, 220],  # Total (dovrebbe essere 150, 165, 180)
            [100, 110, 120],  # P1
            [50, 55, 60]      # P2
        ])

        checker = CoherenceChecker(self.hierarchy, tolerance=1e-6)
        report = checker.check_coherence(all_forecasts)

        assert not report.is_coherent
        assert report.n_incoherent_nodes > 0


class TestUtilityFunctions:
    """Test per funzioni utility."""

    def test_build_summing_matrix(self):
        """Test costruzione matrice S."""
        hierarchy_df = pd.DataFrame({
            'total': ['Total'] * 4,
            'category': ['A', 'A', 'B', 'B'],
            'product': ['P1', 'P2', 'P3', 'P4']
        })

        S = build_summing_matrix(
            hierarchy_df,
            'product',
            ['total', 'category']
        )

        # Verifica forma
        assert S.shape[1] == 4  # 4 prodotti (foglie)
        assert S.shape[0] == 7  # 1 total + 2 categorie + 4 prodotti

    def test_aggregate_forecasts(self):
        """Test aggregazione previsioni."""
        # Previsioni bottom level
        bottom_forecasts = np.array([
            [10, 11, 12],
            [20, 21, 22],
            [30, 31, 32],
            [40, 41, 42]
        ])

        # Matrice S semplice (total = somma di tutti)
        S = np.ones((1, 4))

        aggregated = aggregate_forecasts(bottom_forecasts, S)

        expected = np.array([[100, 104, 108]])
        np.testing.assert_array_almost_equal(aggregated, expected)

    def test_calculate_proportions(self):
        """Test calcolo proporzioni."""
        historical_data = pd.DataFrame({
            'P1': [100, 110, 120],
            'P2': [50, 55, 60],
            'P3': [30, 33, 36]
        })

        proportions = calculate_proportions(
            historical_data.T,
            method='average'
        )

        assert isinstance(proportions, pd.DataFrame)
        assert abs(proportions['proportion'].sum() - 1.0) < 1e-10


class TestHierarchyValidator:
    """Test per validatore gerarchie."""

    def test_validation_report_generation(self):
        """Test generazione report validazione."""
        hierarchy_df = pd.DataFrame({
            'total': ['Total'] * 2,
            'product': ['P1', 'P2']
        })

        hierarchy = ProductHierarchy()
        hierarchy.build_hierarchy(hierarchy_df)

        validator = HierarchyValidator(hierarchy)
        report = validator.generate_validation_report()

        assert isinstance(report, pd.DataFrame)
        assert 'Check' in report.columns
        assert 'Status' in report.columns
        assert 'Details' in report.columns


@pytest.mark.integration
class TestIntegration:
    """Test di integrazione end-to-end."""

    def test_full_reconciliation_pipeline(self):
        """Test pipeline completo di riconciliazione."""
        # 1. Crea dati
        dates = pd.date_range('2023-01-01', periods=24, freq='ME')
        np.random.seed(42)

        data = pd.DataFrame({
            'date': dates,
            'P1': np.random.randn(24) * 10 + 100,
            'P2': np.random.randn(24) * 15 + 150,
            'P3': np.random.randn(24) * 12 + 120,
            'P4': np.random.randn(24) * 8 + 80
        })

        # Aggrega per categorie
        data['Cat_A'] = data['P1'] + data['P2']
        data['Cat_B'] = data['P3'] + data['P4']
        data['Total'] = data['Cat_A'] + data['Cat_B']

        # 2. Crea gerarchia
        hierarchy_df = pd.DataFrame({
            'total': ['Total'] * 4,
            'category': ['Cat_A', 'Cat_A', 'Cat_B', 'Cat_B'],
            'product': ['P1', 'P2', 'P3', 'P4']
        })

        hierarchy = ProductHierarchy()
        hierarchy.build_hierarchy(hierarchy_df)

        # 3. Simula previsioni base
        forecast_cols = ['Total', 'Cat_A', 'Cat_B', 'P1', 'P2', 'P3', 'P4']
        base_forecasts = np.random.randn(7, 6) * 10 + 100

        # 4. Riconcilia
        reconciler = HierarchicalReconciler(hierarchy)
        reconciled = reconciler.reconcile(
            base_forecasts,
            method=ReconciliationMethod.OLS
        )

        # 5. Valida
        checker = CoherenceChecker(hierarchy)
        report = checker.check_coherence(reconciled)

        # Verifica risultati
        assert reconciled.shape == base_forecasts.shape
        assert report.mean_error < 0.01  # Tolleranza rilassata per test