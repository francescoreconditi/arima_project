"""
Test per modulo inventory management.

Test completi per balance_optimizer con tutte le funzionalità.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from arima_forecaster.inventory.balance_optimizer import (
    MovementClassifier,
    SlowFastOptimizer,
    PerishableManager,
    MinimumShelfLifeManager,
    MultiEchelonOptimizer,
    CapacityConstrainedOptimizer,
    KittingOptimizer,
)


class TestMovementClassifier:
    """Test per classificazione movimento prodotti."""

    @pytest.fixture
    def sample_data(self):
        """Crea dati di esempio per test."""
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")

        # Prodotti con diversi pattern di movimento
        data = {
            "date": dates,
            "fast_moving": np.random.poisson(100, 100),  # Alto volume
            "medium_moving": np.random.poisson(20, 100),  # Medio volume
            "slow_moving": np.random.poisson(2, 100),  # Basso volume
            "intermittent": np.random.choice([0, 0, 0, 50], 100),  # Intermittente
        }
        return pd.DataFrame(data)

    def test_classify_movement_speed(self, sample_data):
        """Test classificazione velocità movimento."""
        classifier = MovementClassifier()

        # Test fast moving
        fast_result = classifier.classify_movement_speed(sample_data["fast_moving"].values)
        assert fast_result["classification"] == "fast_moving"
        assert fast_result["avg_sales"] > 80

        # Test slow moving
        slow_result = classifier.classify_movement_speed(sample_data["slow_moving"].values)
        assert slow_result["classification"] == "slow_moving"
        assert slow_result["avg_sales"] < 5

    def test_abc_analysis(self, sample_data):
        """Test analisi ABC."""
        classifier = MovementClassifier()

        products = {
            "P1": {"revenue": 100000, "volume": 1000},
            "P2": {"revenue": 50000, "volume": 500},
            "P3": {"revenue": 10000, "volume": 100},
            "P4": {"revenue": 1000, "volume": 10},
        }

        result = classifier.abc_analysis(products)

        assert "A" in result
        assert "B" in result
        assert "C" in result
        assert "P1" in result["A"]  # Prodotto top revenue
        assert "P4" in result["C"]  # Prodotto basso revenue

    def test_xyz_analysis(self, sample_data):
        """Test analisi XYZ variabilità."""
        classifier = MovementClassifier()

        # Crea serie con diversa variabilità
        stable = np.ones(100) * 50 + np.random.normal(0, 1, 100)
        variable = np.ones(100) * 50 + np.random.normal(0, 20, 100)
        erratic = np.random.exponential(50, 100)

        stable_result = classifier.xyz_analysis(stable)
        assert stable_result["category"] == "X"  # Bassa variabilità

        erratic_result = classifier.xyz_analysis(erratic)
        assert erratic_result["category"] in ["Y", "Z"]  # Alta variabilità


class TestSlowFastOptimizer:
    """Test per ottimizzatore slow/fast moving."""

    @pytest.fixture
    def optimizer(self):
        """Crea istanza optimizer."""
        return SlowFastOptimizer()

    def test_optimize_slow_movers(self, optimizer):
        """Test ottimizzazione slow movers."""
        slow_data = np.random.poisson(1, 100)
        result = optimizer.optimize_inventory(sales_data=slow_data, classification="slow_moving")

        assert "reorder_point" in result
        assert "order_quantity" in result
        assert "safety_stock" in result
        assert result["strategy"] == "min_stock"
        assert result["order_quantity"] <= 20  # Ordini piccoli per slow movers

    def test_optimize_fast_movers(self, optimizer):
        """Test ottimizzazione fast movers."""
        fast_data = np.random.poisson(100, 100)
        result = optimizer.optimize_inventory(sales_data=fast_data, classification="fast_moving")

        assert result["strategy"] == "continuous_replenishment"
        assert result["order_quantity"] > 100  # Ordini grandi per fast movers
        assert result["safety_stock"] > 50  # Safety stock alto

    def test_pooling_strategy(self, optimizer):
        """Test strategia pooling per slow movers."""
        locations = ["Milano", "Roma", "Napoli"]
        slow_data = {loc: np.random.poisson(1, 100) for loc in locations}

        result = optimizer.recommend_pooling_strategy(slow_data)

        assert "pooling_recommended" in result
        assert "central_location" in result
        assert "expected_savings" in result


class TestPerishableManager:
    """Test per gestione prodotti deperibili."""

    @pytest.fixture
    def manager(self):
        """Crea istanza manager."""
        return PerishableManager()

    def test_fefo_optimization(self, manager):
        """Test ottimizzazione FEFO."""
        inventory = [
            {"batch": "A", "quantity": 100, "expiry": datetime.now() + timedelta(days=5)},
            {"batch": "B", "quantity": 150, "expiry": datetime.now() + timedelta(days=10)},
            {"batch": "C", "quantity": 80, "expiry": datetime.now() + timedelta(days=3)},
        ]

        result = manager.optimize_fefo_quantity(
            current_inventory=inventory, daily_demand=50, shelf_life_days=14
        )

        assert result["order_quantity"] > 0
        assert result["expected_waste"] < 50
        assert result["rotation_schedule"][0]["batch"] == "C"  # Prima scadenza

    def test_freshness_index(self, manager):
        """Test calcolo indice freschezza."""
        inventory = [
            {"quantity": 100, "days_to_expiry": 10},
            {"quantity": 50, "days_to_expiry": 5},
        ]

        index = manager.calculate_freshness_index(inventory)

        assert 0 <= index <= 1
        assert index > 0.5  # Inventory relativamente fresco


class TestMinimumShelfLifeManager:
    """Test per gestione MSL (Minimum Shelf Life)."""

    @pytest.fixture
    def manager(self):
        """Crea istanza manager MSL."""
        return MinimumShelfLifeManager()

    def test_msl_allocation(self, manager):
        """Test allocazione con vincoli MSL."""
        lotti = [
            {"id": "L1", "quantita": 100, "giorni_residui": 180},
            {"id": "L2", "quantita": 150, "giorni_residui": 90},
            {"id": "L3", "quantita": 200, "giorni_residui": 30},
        ]

        canali = {
            "GDO": {"msl_richiesto": 120, "domanda": 80},
            "Retail": {"msl_richiesto": 60, "domanda": 100},
            "Online": {"msl_richiesto": 30, "domanda": 150},
        }

        result = manager.ottimizza_allocazione_lotti(lotti, canali)

        assert "allocazioni" in result
        assert "lotti_non_allocati" in result
        assert len(result["allocazioni"]) > 0

        # Verifica che GDO riceva solo lotti con MSL sufficiente
        for alloc in result["allocazioni"]:
            if alloc["canale"] == "GDO":
                assert alloc["giorni_residui"] >= 120


class TestMultiEchelonOptimizer:
    """Test per ottimizzazione multi-echelon."""

    @pytest.fixture
    def optimizer(self):
        """Crea istanza optimizer."""
        return MultiEchelonOptimizer()

    def test_optimize_network(self, optimizer):
        """Test ottimizzazione network distribuzione."""
        network = {
            "central_warehouse": {"capacity": 10000, "holding_cost": 2.0, "current_stock": 5000},
            "regional_warehouses": [
                {"id": "RW1", "capacity": 2000, "holding_cost": 3.0, "demand": 100},
                {"id": "RW2", "capacity": 1500, "holding_cost": 3.5, "demand": 80},
            ],
            "stores": [
                {"id": "S1", "capacity": 200, "holding_cost": 5.0, "demand": 20},
                {"id": "S2", "capacity": 150, "holding_cost": 5.5, "demand": 15},
            ],
        }

        result = optimizer.optimize_network(network)

        assert "central_warehouse" in result
        assert "regional_warehouses" in result
        assert "stores" in result
        assert "total_cost" in result
        assert result["total_cost"] > 0

    def test_risk_pooling(self, optimizer):
        """Test strategia risk pooling."""
        locations = {
            "L1": {"demand_mean": 100, "demand_std": 20},
            "L2": {"demand_mean": 80, "demand_std": 15},
            "L3": {"demand_mean": 120, "demand_std": 25},
        }

        result = optimizer.evaluate_risk_pooling(locations)

        assert "pooled_safety_stock" in result
        assert "individual_safety_stock" in result
        assert "savings_percentage" in result
        assert result["pooled_safety_stock"] < result["individual_safety_stock"]


class TestCapacityConstrainedOptimizer:
    """Test per ottimizzazione con vincoli capacità."""

    @pytest.fixture
    def optimizer(self):
        """Crea istanza optimizer."""
        return CapacityConstrainedOptimizer()

    def test_optimize_with_constraints(self, optimizer):
        """Test ottimizzazione con vincoli multipli."""
        products = [
            {"id": "P1", "volume": 2, "weight": 5, "value": 100, "demand": 50},
            {"id": "P2", "volume": 1, "weight": 2, "value": 50, "demand": 100},
            {"id": "P3", "volume": 3, "weight": 8, "value": 150, "demand": 30},
        ]

        constraints = {"max_volume": 200, "max_weight": 500, "max_budget": 5000, "max_pallets": 10}

        result = optimizer.optimize_with_constraints(products, constraints)

        assert "optimal_quantities" in result
        assert "total_volume" in result
        assert "total_weight" in result
        assert result["total_volume"] <= constraints["max_volume"]
        assert result["total_weight"] <= constraints["max_weight"]

    def test_pallet_optimization(self, optimizer):
        """Test ottimizzazione pallet."""
        products = [
            {"id": "P1", "units_per_pallet": 100, "demand": 250},
            {"id": "P2", "units_per_pallet": 80, "demand": 160},
        ]

        result = optimizer.optimize_pallet_configuration(products, max_pallets=5)

        assert "pallet_allocation" in result
        assert "utilization_rate" in result
        assert sum(result["pallet_allocation"].values()) <= 5


class TestKittingOptimizer:
    """Test per ottimizzazione kitting/bundling."""

    @pytest.fixture
    def optimizer(self):
        """Crea istanza optimizer."""
        return KittingOptimizer()

    def test_analyze_kit_strategy(self, optimizer):
        """Test analisi strategia kit."""
        kit_definition = {
            "kit_id": "KIT001",
            "components": [
                {"id": "C1", "quantity": 2, "cost": 10},
                {"id": "C2", "quantity": 1, "cost": 20},
                {"id": "C3", "quantity": 3, "cost": 5},
            ],
        }

        demand_data = {
            "kit_demand": np.random.poisson(20, 100),
            "component_demands": {
                "C1": np.random.poisson(10, 100),
                "C2": np.random.poisson(5, 100),
                "C3": np.random.poisson(15, 100),
            },
        }

        result = optimizer.analyze_kit_strategy(kit_definition, demand_data)

        assert "strategy" in result
        assert result["strategy"] in ["make_to_stock", "assemble_to_order"]
        assert "cost_comparison" in result
        assert "service_level_impact" in result

    def test_component_availability(self, optimizer):
        """Test calcolo disponibilità componenti."""
        inventory = {
            "C1": 100,
            "C2": 50,
            "C3": 150,
        }

        kit_bom = [
            {"component": "C1", "quantity": 2},
            {"component": "C2", "quantity": 1},
            {"component": "C3", "quantity": 3},
        ]

        max_kits = optimizer.calculate_max_kits(inventory, kit_bom)

        assert max_kits == 50  # Limitato da C2 (50/1)


class TestIntegration:
    """Test di integrazione tra moduli."""

    def test_complete_inventory_optimization(self):
        """Test ottimizzazione completa inventory."""
        # Crea dati esempio
        sales_data = np.random.poisson(50, 365)

        # 1. Classifica movimento
        classifier = MovementClassifier()
        movement = classifier.classify_movement_speed(sales_data)

        # 2. Ottimizza in base a classificazione
        optimizer = SlowFastOptimizer()
        inventory_params = optimizer.optimize_inventory(sales_data, movement["classification"])

        # 3. Se deperibile, applica FEFO
        if movement["classification"] == "fast_moving":
            perishable_mgr = PerishableManager()
            inventory = [
                {"batch": "A", "quantity": 100, "expiry": datetime.now() + timedelta(days=7)}
            ]
            fefo_result = perishable_mgr.optimize_fefo_quantity(
                inventory, movement["avg_sales"], shelf_life_days=14
            )
            inventory_params.update(fefo_result)

        # Verifica risultati
        assert "reorder_point" in inventory_params
        assert "order_quantity" in inventory_params
        assert inventory_params["order_quantity"] > 0

    def test_multi_location_optimization(self):
        """Test ottimizzazione multi-location con vincoli."""
        # Setup network
        network = {
            "central_warehouse": {"capacity": 10000, "holding_cost": 2.0, "current_stock": 5000},
            "regional_warehouses": [
                {"id": "RW1", "capacity": 2000, "holding_cost": 3.0, "demand": 100},
            ],
        }

        # Ottimizza network
        multi_optimizer = MultiEchelonOptimizer()
        network_result = multi_optimizer.optimize_network(network)

        # Applica vincoli capacità
        capacity_optimizer = CapacityConstrainedOptimizer()
        products = [{"id": "P1", "volume": 2, "weight": 5, "value": 100, "demand": 50}]
        constraints = {"max_volume": 1000, "max_weight": 2500}

        final_result = capacity_optimizer.optimize_with_constraints(products, constraints)

        assert final_result["total_volume"] <= constraints["max_volume"]
        assert "optimal_quantities" in final_result
