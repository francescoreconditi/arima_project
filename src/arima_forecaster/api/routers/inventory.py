"""
Router per endpoint di Inventory Management e ottimizzazione scorte.

Gestisce classificazione ABC/XYZ, ottimizzazione slow/fast moving, safety stock,
multi-echelon optimization e gestione vincoli di capacità.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from pydantic import BaseModel, Field, validator
import pandas as pd
import numpy as np

from arima_forecaster.utils.logger import get_logger

logger = get_logger(__name__)

# Crea router con prefix e tags
router = APIRouter(
    prefix="/inventory",
    tags=["Inventory Management"],
    responses={404: {"description": "Not found"}},
)

"""
INVENTORY MANAGEMENT ROUTER

Gestisce l'ottimizzazione completa dell'inventory management:

• POST /inventory/classify-movement      - Classificazione ABC/XYZ movement analysis
• POST /inventory/optimize-slow-fast     - Ottimizzazione slow/fast moving items  
• POST /inventory/safety-stock          - Calcolo safety stock dinamico
• POST /inventory/reorder-points        - Calcolo punti di riordino ottimali
• POST /inventory/eoq-calculation       - Economic Order Quantity optimization
• POST /inventory/multi-echelon         - Ottimizzazione multi-warehouse network
• POST /inventory/capacity-constraints  - Gestione vincoli capacità (volume/peso/budget)
• POST /inventory/kitting-analysis      - Analisi bundle/kit strategies

Caratteristiche:
- Classificazione automatica ABC/XYZ per movement pattern
- Ottimizzazione safety stock con demand uncertainty
- Multi-echelon optimization con risk pooling
- Capacity constraints con algoritmi di ottimizzazione
- Kitting analysis per Make-to-Stock vs Assemble-to-Order
"""

# =============================================================================
# MODELLI RICHIESTA E RISPOSTA
# =============================================================================


class ProductSalesData(BaseModel):
    """Dati vendite prodotto per classificazione."""

    product_id: str = Field(..., description="ID univoco prodotto")
    product_name: str = Field(..., description="Nome prodotto")
    category: Optional[str] = Field(None, description="Categoria prodotto")
    monthly_sales: List[float] = Field(..., description="Vendite mensili ultimi 12 mesi")
    unit_cost: float = Field(..., description="Costo unitario prodotto")
    sale_price: float = Field(..., description="Prezzo vendita unitario")
    current_stock: int = Field(0, description="Stock attuale disponibile")
    lead_time_days: int = Field(7, description="Lead time medio fornitori (giorni)")


class MovementClassificationRequest(BaseModel):
    """Richiesta classificazione movement ABC/XYZ."""

    products: List[ProductSalesData] = Field(..., description="Lista prodotti da classificare")
    abc_thresholds: Optional[Dict[str, float]] = Field(
        default={"A": 0.8, "B": 0.95, "C": 1.0},
        description="Soglie percentuali cumulative ABC analysis",
    )
    xyz_cv_thresholds: Optional[Dict[str, float]] = Field(
        default={"X": 0.5, "Y": 1.0, "Z": float("inf")},
        description="Soglie coefficiente variazione XYZ analysis",
    )


class ProductClassification(BaseModel):
    """Risultato classificazione singolo prodotto."""

    product_id: str
    product_name: str
    category: Optional[str]
    abc_class: str = Field(..., description="Classificazione ABC (A/B/C)")
    xyz_class: str = Field(..., description="Classificazione XYZ (X/Y/Z)")
    combined_class: str = Field(..., description="Classe combinata (AX, AY, AZ, etc.)")
    annual_revenue: float = Field(..., description="Ricavi annuali prodotto")
    revenue_percentage: float = Field(..., description="% ricavi sul totale")
    revenue_cumulative: float = Field(..., description="% ricavi cumulativa")
    coefficient_variation: float = Field(..., description="CV domanda (volatilità)")
    demand_predictability: str = Field(..., description="Prevedibilità domanda (High/Medium/Low)")


class MovementClassificationResponse(BaseModel):
    """Risposta classificazione movement completa."""

    classification_id: str = Field(..., description="ID analisi classificazione")
    created_at: datetime = Field(..., description="Timestamp creazione")
    total_products: int = Field(..., description="Numero totale prodotti analizzati")
    total_annual_revenue: float = Field(..., description="Ricavi annuali totali")
    products: List[ProductClassification] = Field(..., description="Classificazioni prodotti")
    summary_by_class: Dict[str, Dict[str, Union[int, float]]] = Field(
        ..., description="Riassunto per classe"
    )
    recommendations: List[str] = Field(..., description="Raccomandazioni strategiche")


# Slow/Fast Moving Optimization
class SlowFastOptimizationRequest(BaseModel):
    """Richiesta ottimizzazione slow/fast moving items."""

    classification_results: List[ProductClassification] = Field(
        ..., description="Risultati classificazione ABC/XYZ"
    )
    inventory_costs: Optional[Dict[str, float]] = Field(
        default={"holding_cost_rate": 0.25, "stockout_cost_multiplier": 3.0},
        description="Costi inventory (holding cost %, stockout multiplier)",
    )
    service_level_targets: Optional[Dict[str, float]] = Field(
        default={"A": 0.98, "B": 0.95, "C": 0.90}, description="Service level target per classe ABC"
    )


class SlowFastOptimizationResponse(BaseModel):
    """Risposta ottimizzazione slow/fast moving."""

    optimization_id: str = Field(..., description="ID ottimizzazione")
    created_at: datetime = Field(..., description="Timestamp creazione")
    total_products: int = Field(..., description="Prodotti ottimizzati")
    total_investment_reduction: float = Field(..., description="Riduzione investimento totale")
    recommendations: List[Dict[str, Any]] = Field(..., description="Raccomandazioni per prodotto")
    summary_metrics: Dict[str, float] = Field(..., description="Metriche aggregate ottimizzazione")


# Safety Stock Calculation
class SafetyStockRequest(BaseModel):
    """Richiesta calcolo safety stock dinamico."""

    product_id: str = Field(..., description="ID prodotto")
    historical_demand: List[float] = Field(..., description="Domanda storica mensile")
    lead_time_days: int = Field(..., description="Lead time medio (giorni)")
    lead_time_variability: Optional[float] = Field(0.2, description="Variabilità lead time (CV)")
    service_level: float = Field(0.95, description="Service level target (0-1)")
    seasonal_adjustment: Optional[bool] = Field(
        True, description="Applica aggiustamento stagionale"
    )


class SafetyStockResponse(BaseModel):
    """Risposta calcolo safety stock."""

    product_id: str
    recommended_safety_stock: float = Field(..., description="Safety stock raccomandato")
    average_demand: float = Field(..., description="Domanda media periodo")
    demand_std_dev: float = Field(..., description="Deviazione standard domanda")
    service_level_achieved: float = Field(..., description="Service level stimato")
    reorder_point: float = Field(..., description="Punto riordino raccomandato")
    maximum_stock: float = Field(..., description="Stock massimo raccomandato")
    turnover_ratio: float = Field(..., description="Turnover ratio stimato")


# EOQ Calculation
class EOQRequest(BaseModel):
    """Richiesta calcolo Economic Order Quantity."""

    product_id: str = Field(..., description="ID prodotto")
    annual_demand: float = Field(..., description="Domanda annuale unità")
    ordering_cost: float = Field(..., description="Costo per ordine")
    holding_cost_per_unit: float = Field(..., description="Costo holding per unità/anno")
    unit_cost: float = Field(..., description="Costo unitario prodotto")
    quantity_discounts: Optional[List[Dict[str, float]]] = Field(
        None, description="Sconti quantità [{min_qty, discount_rate}]"
    )


class EOQResponse(BaseModel):
    """Risposta calcolo EOQ."""

    product_id: str
    optimal_order_quantity: float = Field(..., description="Quantità ordine ottimale")
    annual_ordering_cost: float = Field(..., description="Costo ordinazione annuale")
    annual_holding_cost: float = Field(..., description="Costo holding annuale")
    total_annual_cost: float = Field(..., description="Costo totale annuale")
    order_frequency: float = Field(..., description="Frequenza ordinazione (volte/anno)")
    time_between_orders: float = Field(..., description="Tempo tra ordini (giorni)")
    discount_analysis: Optional[Dict[str, Any]] = Field(None, description="Analisi sconti quantità")


# Multi-Echelon Optimization
class LocationData(BaseModel):
    """Dati singola location per multi-echelon."""

    location_id: str = Field(..., description="ID location/warehouse")
    location_name: str = Field(..., description="Nome location")
    demand_forecast: List[float] = Field(..., description="Forecast domanda mensile")
    current_inventory: Dict[str, float] = Field(..., description="Inventory attuale per prodotto")
    capacity_constraints: Optional[Dict[str, float]] = Field(
        None, description="Vincoli capacità {volume, weight, value}"
    )
    transportation_costs: Dict[str, float] = Field(
        ..., description="Costi trasporto verso altre location"
    )


class MultiEchelonRequest(BaseModel):
    """Richiesta ottimizzazione multi-echelon."""

    locations: List[LocationData] = Field(..., description="Dati locations network")
    products: List[str] = Field(..., description="Lista prodotti da ottimizzare")
    optimization_horizon: int = Field(6, description="Orizzonte ottimizzazione (mesi)")
    risk_pooling_enabled: bool = Field(True, description="Abilita risk pooling")
    centralization_options: Optional[Dict[str, Any]] = Field(
        None, description="Opzioni centralizzazione"
    )


class MultiEchelonResponse(BaseModel):
    """Risposta ottimizzazione multi-echelon."""

    optimization_id: str
    total_locations: int
    optimization_horizon: int
    recommended_allocation: Dict[str, Dict[str, float]] = Field(
        ..., description="Allocazione per location/prodotto"
    )
    total_cost_reduction: float = Field(..., description="Riduzione costi totali")
    risk_pooling_benefits: Dict[str, float] = Field(..., description="Benefici risk pooling")
    transportation_optimization: Dict[str, Any] = Field(..., description="Ottimizzazione trasporti")
    recommendations: List[str] = Field(..., description="Raccomandazioni strategiche")


# Capacity Constraints
class CapacityConstraintsRequest(BaseModel):
    """Richiesta ottimizzazione con vincoli capacità."""

    products: List[Dict[str, Any]] = Field(
        ..., description="Prodotti con dati dimensioni/peso/valore"
    )
    total_capacity: Dict[str, float] = Field(
        ..., description="Capacità totali {volume, weight, value}"
    )
    priority_weights: Optional[Dict[str, float]] = Field(
        default={"revenue": 0.4, "margin": 0.3, "turnover": 0.3},
        description="Pesi priorità ottimizzazione",
    )
    constraints_type: str = Field("mixed", description="Tipo vincoli (volume/weight/value/mixed)")


class CapacityConstraintsResponse(BaseModel):
    """Risposta ottimizzazione vincoli capacità."""

    optimization_id: str
    optimized_portfolio: List[Dict[str, Any]] = Field(
        ..., description="Portfolio prodotti ottimizzato"
    )
    capacity_utilization: Dict[str, float] = Field(..., description="Utilizzo capacità per tipo")
    total_revenue: float = Field(..., description="Ricavi totali portfolio")
    total_profit: float = Field(..., description="Profitto totale portfolio")
    efficiency_score: float = Field(..., description="Score efficienza utilizzo spazio")
    excluded_products: List[Dict[str, str]] = Field(..., description="Prodotti esclusi e motivo")


# Kitting Analysis
class KittingAnalysisRequest(BaseModel):
    """Richiesta analisi bundle/kitting strategies."""

    main_products: List[Dict[str, Any]] = Field(..., description="Prodotti principali per bundle")
    component_products: List[Dict[str, Any]] = Field(..., description="Componenti disponibili")
    demand_correlation: Optional[Dict[str, float]] = Field(
        None, description="Correlazione domanda tra prodotti"
    )
    assembly_costs: Dict[str, float] = Field(..., description="Costi assemblaggio per tipo bundle")
    storage_cost_difference: float = Field(
        ..., description="Differenza costo storage bundle vs componenti"
    )


class KittingAnalysisResponse(BaseModel):
    """Risposta analisi kitting."""

    analysis_id: str
    recommended_bundles: List[Dict[str, Any]] = Field(..., description="Bundle raccomandati")
    make_to_stock_vs_assemble: Dict[str, str] = Field(
        ..., description="Raccomandazione per prodotto"
    )
    cost_benefit_analysis: Dict[str, float] = Field(..., description="Analisi costi/benefici")
    inventory_reduction: float = Field(..., description="Riduzione inventory stimata")
    service_level_impact: Dict[str, float] = Field(..., description="Impatto service level")
    implementation_priority: List[str] = Field(..., description="Priorità implementazione")


# =============================================================================
# DEPENDENCY INJECTION
# =============================================================================


def get_inventory_services():
    """Dependency per ottenere i servizi inventory management."""
    try:
        from arima_forecaster.inventory.balance_optimizer import (
            MovementClassifier,
            SlowFastOptimizer,
            MultiEchelonOptimizer,
            CapacityConstrainedOptimizer,
            KittingOptimizer,
        )

        return {
            "classifier": MovementClassifier(),
            "slow_fast": SlowFastOptimizer(),
            "multi_echelon": MultiEchelonOptimizer(),
            "capacity": CapacityConstrainedOptimizer(),
            "kitting": KittingOptimizer(),
        }
    except ImportError:
        # Fallback se moduli non disponibili
        return None


# =============================================================================
# ENDPOINT IMPLEMENTATIONS
# =============================================================================


@router.post("/classify-movement", response_model=MovementClassificationResponse)
async def classify_product_movement(
    request: MovementClassificationRequest,
    services: Optional[Dict] = Depends(get_inventory_services),
):
    """
    Classifica i prodotti usando analisi ABC/XYZ per movement pattern.

    <h4>Analisi ABC/XYZ Movement Classification:</h4>
    <table >
        <tr><th>Tipo</th><th>Descrizione</th><th>Caratteristiche</th></tr>
        <tr><td>ABC Analysis</td><td>Classificazione per valore ricavi</td><td>A: 80% ricavi, B: 15% ricavi, C: 5% ricavi</td></tr>
        <tr><td>XYZ Analysis</td><td>Classificazione per prevedibilità domanda</td><td>X: CV<0.5, Y: CV<1.0, Z: CV>1.0</td></tr>
        <tr><td>Combined</td><td>Matrice 9 classi (AX, AY, AZ, BX, BY, BZ, CX, CY, CZ)</td><td>Strategie inventory differentiate</td></tr>
    </table>

    <h4>Esempio Richiesta:</h4>
    <pre><code>
    {
        "products": [
            {
                "product_id": "SKU001",
                "product_name": "Laptop Gaming XYZ",
                "category": "Electronics",
                "monthly_sales": [120, 135, 110, 145, 160, 140, 125, 155, 170, 135, 145, 150],
                "unit_cost": 800.0,
                "sale_price": 1200.0,
                "current_stock": 50,
                "lead_time_days": 14
            }
        ],
        "abc_thresholds": {"A": 0.8, "B": 0.95, "C": 1.0},
        "xyz_cv_thresholds": {"X": 0.5, "Y": 1.0, "Z": 999.0}
    }
    </code></pre>

    <h4>Risposta di Esempio:</h4>
    <pre><code>
    {
        "classification_id": "class-abc123",
        "created_at": "2024-08-23T14:30:00",
        "total_products": 1,
        "total_annual_revenue": 1620000.0,
        "products": [
            {
                "product_id": "SKU001",
                "product_name": "Laptop Gaming XYZ",
                "category": "Electronics",
                "abc_class": "A",
                "xyz_class": "Y",
                "combined_class": "AY",
                "annual_revenue": 1620000.0,
                "revenue_percentage": 100.0,
                "revenue_cumulative": 100.0,
                "coefficient_variation": 0.15,
                "demand_predictability": "Medium"
            }
        ],
        "summary_by_class": {
            "AY": {"count": 1, "revenue_share": 100.0, "avg_cv": 0.15}
        },
        "recommendations": [
            "Classe AY: High-value medium-variability - Implement dynamic safety stock",
            "Focus on demand sensing for classe A products",
            "Consider EOQ optimization for stable demand patterns"
        ]
    }
    </code></pre>

    <h4>Strategie per Classe Combinata:</h4>
    - **AX**: High value, predictable - EOQ, low safety stock
    - **AY**: High value, medium variability - Dynamic safety stock
    - **AZ**: High value, volatile - High safety stock, frequent review
    - **BX**: Medium value, predictable - Standard EOQ
    - **BY**: Medium value, medium variability - Balanced approach
    - **BZ**: Medium value, volatile - Safety stock focus
    - **CX**: Low value, predictable - Bulk ordering, low holding
    - **CY**: Low value, medium variability - Simple reorder rules
    - **CZ**: Low value, volatile - Basic inventory rules, accept stockouts
    """
    try:
        # Genera ID classificazione
        classification_id = f"class-{uuid.uuid4().hex[:8]}"

        # Calcola ricavi annuali per ogni prodotto
        products_data = []
        total_annual_revenue = 0.0

        for product in request.products:
            # Calcola ricavi annuali
            annual_demand = sum(product.monthly_sales)
            annual_revenue = annual_demand * product.sale_price
            total_annual_revenue += annual_revenue

            # Calcola coefficient of variation per XYZ
            mean_demand = np.mean(product.monthly_sales)
            std_demand = np.std(product.monthly_sales)
            cv = std_demand / mean_demand if mean_demand > 0 else float("inf")

            products_data.append({"product": product, "annual_revenue": annual_revenue, "cv": cv})

        # Ordina per ricavi per ABC analysis
        products_data.sort(key=lambda x: x["annual_revenue"], reverse=True)

        # Classificazione ABC
        cumulative_revenue = 0.0
        classified_products = []

        for data in products_data:
            cumulative_revenue += data["annual_revenue"]
            revenue_percentage = (data["annual_revenue"] / total_annual_revenue) * 100
            revenue_cumulative = (cumulative_revenue / total_annual_revenue) * 100

            # Classifica ABC
            if revenue_cumulative <= request.abc_thresholds["A"] * 100:
                abc_class = "A"
            elif revenue_cumulative <= request.abc_thresholds["B"] * 100:
                abc_class = "B"
            else:
                abc_class = "C"

            # Classifica XYZ
            if data["cv"] <= request.xyz_cv_thresholds["X"]:
                xyz_class = "X"
                predictability = "High"
            elif data["cv"] <= request.xyz_cv_thresholds["Y"]:
                xyz_class = "Y"
                predictability = "Medium"
            else:
                xyz_class = "Z"
                predictability = "Low"

            combined_class = f"{abc_class}{xyz_class}"

            classified_products.append(
                ProductClassification(
                    product_id=data["product"].product_id,
                    product_name=data["product"].product_name,
                    category=data["product"].category,
                    abc_class=abc_class,
                    xyz_class=xyz_class,
                    combined_class=combined_class,
                    annual_revenue=data["annual_revenue"],
                    revenue_percentage=revenue_percentage,
                    revenue_cumulative=revenue_cumulative,
                    coefficient_variation=data["cv"],
                    demand_predictability=predictability,
                )
            )

        # Crea summary per classe
        summary_by_class = {}
        for product in classified_products:
            class_key = product.combined_class
            if class_key not in summary_by_class:
                summary_by_class[class_key] = {"count": 0, "revenue_share": 0.0, "avg_cv": 0.0}
            summary_by_class[class_key]["count"] += 1
            summary_by_class[class_key]["revenue_share"] += product.revenue_percentage
            summary_by_class[class_key]["avg_cv"] += product.coefficient_variation

        # Calcola media CV per classe
        for class_key in summary_by_class:
            if summary_by_class[class_key]["count"] > 0:
                summary_by_class[class_key]["avg_cv"] /= summary_by_class[class_key]["count"]

        # Genera raccomandazioni
        recommendations = []
        a_products = len([p for p in classified_products if p.abc_class == "A"])
        high_variability = len([p for p in classified_products if p.xyz_class == "Z"])

        if a_products > 0:
            recommendations.append(
                "Focus on demand sensing for classe A products - High impact on revenue"
            )
        if high_variability > 0:
            recommendations.append(
                f"{high_variability} products with high variability - Implement dynamic safety stock"
            )

        # Raccomandazioni per classi principali
        for class_key in summary_by_class:
            abc, xyz = class_key[0], class_key[1]
            count = summary_by_class[class_key]["count"]

            if abc == "A" and xyz == "X":
                recommendations.append(
                    f"Classe {class_key} ({count} products): High-value predictable - EOQ optimization, low safety stock"
                )
            elif abc == "A" and xyz == "Z":
                recommendations.append(
                    f"Classe {class_key} ({count} products): High-value volatile - High safety stock, frequent review"
                )
            elif abc == "C" and xyz == "Z":
                recommendations.append(
                    f"Classe {class_key} ({count} products): Low-value volatile - Basic rules, accept stockouts"
                )

        return MovementClassificationResponse(
            classification_id=classification_id,
            created_at=datetime.now(),
            total_products=len(classified_products),
            total_annual_revenue=total_annual_revenue,
            products=classified_products,
            summary_by_class=summary_by_class,
            recommendations=recommendations,
        )

    except Exception as e:
        logger.error(f"Errore classificazione movement: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Errore classificazione: {str(e)}")


@router.post("/optimize-slow-fast", response_model=SlowFastOptimizationResponse)
async def optimize_slow_fast_moving(
    request: SlowFastOptimizationRequest, services: Optional[Dict] = Depends(get_inventory_services)
):
    """
    Ottimizza l'inventory management per slow/fast moving items basato su classificazione ABC/XYZ.

    <h4>Strategie Ottimizzazione per Categoria:</h4>
    <table >
        <tr><th>Classe</th><th>Strategia</th><th>Safety Stock</th><th>Review Frequency</th></tr>
        <tr><td>AX (Fast-Predictable)</td><td>EOQ Optimization</td><td>Low (1-2 weeks)</td><td>Monthly</td></tr>
        <tr><td>AY (Fast-Medium)</td><td>Dynamic Safety Stock</td><td>Medium (2-4 weeks)</td><td>Bi-weekly</td></tr>
        <tr><td>AZ (Fast-Volatile)</td><td>High Service Level</td><td>High (4-8 weeks)</td><td>Weekly</td></tr>
        <tr><td>CZ (Slow-Volatile)</td><td>Minimize Investment</td><td>Very Low</td><td>Quarterly</td></tr>
    </table>

    <h4>Esempio Richiesta:</h4>
    <pre><code>
    {
        "classification_results": [
            {
                "product_id": "SKU001",
                "product_name": "Laptop Gaming",
                "abc_class": "A",
                "xyz_class": "Y",
                "combined_class": "AY",
                "annual_revenue": 1620000.0,
                "coefficient_variation": 0.15
            }
        ],
        "inventory_costs": {
            "holding_cost_rate": 0.25,
            "stockout_cost_multiplier": 3.0
        },
        "service_level_targets": {
            "A": 0.98,
            "B": 0.95,
            "C": 0.90
        }
    }
    </code></pre>

    <h4>Risposta di Esempio:</h4>
    <pre><code>
    {
        "optimization_id": "opt-def456",
        "created_at": "2024-08-23T14:45:00",
        "total_products": 1,
        "total_investment_reduction": 125000.0,
        "recommendations": [
            {
                "product_id": "SKU001",
                "current_strategy": "Standard EOQ",
                "recommended_strategy": "Dynamic Safety Stock with Demand Sensing",
                "safety_stock_adjustment": "+15%",
                "review_frequency": "Bi-weekly",
                "investment_change": -25000.0,
                "service_level_improvement": "+2.5%"
            }
        ],
        "summary_metrics": {
            "total_products_optimized": 1,
            "inventory_reduction_pct": 18.5,
            "service_level_improvement": 2.5,
            "cost_savings_annual": 45000.0
        }
    }
    </code></pre>
    """
    try:
        optimization_id = f"opt-{uuid.uuid4().hex[:8]}"

        recommendations = []
        total_investment_reduction = 0.0
        optimized_products = 0

        for product in request.classification_results:
            # Strategia basata su classe combinata
            abc_class = product.abc_class
            xyz_class = product.xyz_class
            combined_class = product.combined_class

            current_strategy = "Standard EOQ"
            investment_change = 0.0
            service_improvement = 0.0

            # Logica ottimizzazione basata su classe
            if combined_class == "AX":  # High value, predictable
                recommended_strategy = "EOQ Optimization with Low Safety Stock"
                safety_stock_adjustment = "-20%"
                review_frequency = "Monthly"
                investment_change = -product.annual_revenue * 0.15  # Riduzione 15%
                service_improvement = 1.0

            elif combined_class == "AY":  # High value, medium variability
                recommended_strategy = "Dynamic Safety Stock with Demand Sensing"
                safety_stock_adjustment = "+15%"
                review_frequency = "Bi-weekly"
                investment_change = -product.annual_revenue * 0.08  # Riduzione 8%
                service_improvement = 2.5

            elif combined_class == "AZ":  # High value, volatile
                recommended_strategy = "High Service Level with Buffer Stock"
                safety_stock_adjustment = "+40%"
                review_frequency = "Weekly"
                investment_change = product.annual_revenue * 0.10  # Aumento 10%
                service_improvement = 5.0

            elif combined_class in ["BX", "BY"]:  # Medium value
                recommended_strategy = "Balanced EOQ with Standard Safety Stock"
                safety_stock_adjustment = "0%"
                review_frequency = "Monthly"
                investment_change = -product.annual_revenue * 0.05  # Riduzione 5%
                service_improvement = 1.5

            elif combined_class == "BZ":  # Medium value, volatile
                recommended_strategy = "Safety Stock Focus with Frequent Review"
                safety_stock_adjustment = "+25%"
                review_frequency = "Bi-weekly"
                investment_change = product.annual_revenue * 0.05  # Aumento 5%
                service_improvement = 3.0

            elif combined_class in ["CX", "CY"]:  # Low value
                recommended_strategy = "Simple Reorder Rules - Low Investment"
                safety_stock_adjustment = "-30%"
                review_frequency = "Quarterly"
                investment_change = -product.annual_revenue * 0.25  # Riduzione 25%
                service_improvement = -1.0

            elif combined_class == "CZ":  # Low value, volatile
                recommended_strategy = "Minimize Investment - Accept Stockouts"
                safety_stock_adjustment = "-50%"
                review_frequency = "Quarterly"
                investment_change = -product.annual_revenue * 0.40  # Riduzione 40%
                service_improvement = -5.0

            else:
                recommended_strategy = "Standard Approach"
                safety_stock_adjustment = "0%"
                review_frequency = "Monthly"

            recommendations.append(
                {
                    "product_id": product.product_id,
                    "product_name": product.product_name,
                    "combined_class": combined_class,
                    "current_strategy": current_strategy,
                    "recommended_strategy": recommended_strategy,
                    "safety_stock_adjustment": safety_stock_adjustment,
                    "review_frequency": review_frequency,
                    "investment_change": investment_change,
                    "service_level_improvement": f"{service_improvement:+.1f}%",
                    "priority": "High"
                    if abc_class == "A"
                    else "Medium"
                    if abc_class == "B"
                    else "Low",
                }
            )

            total_investment_reduction += abs(investment_change) if investment_change < 0 else 0
            optimized_products += 1

        # Calcola metriche aggregate
        total_annual_revenue = sum([p.annual_revenue for p in request.classification_results])
        inventory_reduction_pct = (
            (total_investment_reduction / total_annual_revenue) * 100
            if total_annual_revenue > 0
            else 0
        )
        avg_service_improvement = np.mean(
            [
                float(r["service_level_improvement"].replace("+", "").replace("%", ""))
                for r in recommendations
            ]
        )
        estimated_cost_savings = (
            total_investment_reduction * request.inventory_costs["holding_cost_rate"]
        )

        summary_metrics = {
            "total_products_optimized": optimized_products,
            "inventory_reduction_pct": round(inventory_reduction_pct, 1),
            "service_level_improvement": round(avg_service_improvement, 1),
            "cost_savings_annual": round(estimated_cost_savings, 0),
            "high_priority_products": len([r for r in recommendations if r["priority"] == "High"]),
            "investment_increase_products": len(
                [r for r in recommendations if r["investment_change"] > 0]
            ),
            "investment_decrease_products": len(
                [r for r in recommendations if r["investment_change"] < 0]
            ),
        }

        return SlowFastOptimizationResponse(
            optimization_id=optimization_id,
            created_at=datetime.now(),
            total_products=optimized_products,
            total_investment_reduction=total_investment_reduction,
            recommendations=recommendations,
            summary_metrics=summary_metrics,
        )

    except Exception as e:
        logger.error(f"Errore ottimizzazione slow/fast: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Errore ottimizzazione: {str(e)}")


@router.post("/safety-stock", response_model=SafetyStockResponse)
async def calculate_safety_stock(request: SafetyStockRequest):
    """
    Calcola safety stock dinamico basato su domanda storica e lead time variability.

    <h4>Calcolo Safety Stock Formula:</h4>
    <table >
        <tr><th>Componente</th><th>Formula</th><th>Descrizione</th></tr>
        <tr><td>Demand Variability</td><td>σ_demand × √(Lead_Time)</td><td>Variabilità domanda nel periodo</td></tr>
        <tr><td>Lead Time Variability</td><td>Avg_Demand × σ_lead_time</td><td>Impatto variabilità lead time</td></tr>
        <tr><td>Safety Factor</td><td>Z-score(Service_Level)</td><td>Fattore sicurezza per service level</td></tr>
        <tr><td>Safety Stock</td><td>Z × √(σ²_demand×LT + μ²_demand×σ²_LT)</td><td>Formula combinata</td></tr>
    </table>

    <h4>Esempio Richiesta:</h4>
    <pre><code>
    {
        "product_id": "SKU001",
        "historical_demand": [120, 135, 110, 145, 160, 140, 125, 155, 170, 135, 145, 150],
        "lead_time_days": 14,
        "lead_time_variability": 0.2,
        "service_level": 0.95,
        "seasonal_adjustment": true
    }
    </code></pre>

    <h4>Risposta di Esempio:</h4>
    <pre><code>
    {
        "product_id": "SKU001",
        "recommended_safety_stock": 45.8,
        "average_demand": 140.8,
        "demand_std_dev": 18.2,
        "service_level_achieved": 0.951,
        "reorder_point": 101.2,
        "maximum_stock": 187.0,
        "turnover_ratio": 8.5
    }
    </code></pre>
    """
    try:
        from scipy.stats import norm

        # Calcoli di base
        historical_demand = np.array(request.historical_demand)
        avg_demand = np.mean(historical_demand)
        demand_std = np.std(historical_demand)

        # Converti lead time da giorni a mesi per coerenza
        lead_time_months = request.lead_time_days / 30.0

        # Z-score per service level
        z_score = norm.ppf(request.service_level)

        # Calcola variabilità combinata
        # Varianza domanda nel lead time
        demand_variance_lt = (demand_std**2) * lead_time_months

        # Varianza lead time
        lead_time_variance = (request.lead_time_variability * request.lead_time_days) ** 2
        avg_demand_daily = avg_demand / 30.0  # Converti in domanda giornaliera
        lead_time_impact = (avg_demand_daily**2) * lead_time_variance / (30.0**2)

        # Safety stock combinato
        total_variance = demand_variance_lt + lead_time_impact
        safety_stock = z_score * np.sqrt(total_variance)

        # Aggiustamento stagionale se richiesto
        if request.seasonal_adjustment:
            # Calcola seasonality index semplice
            monthly_avg = avg_demand
            seasonal_factors = historical_demand / monthly_avg
            seasonal_cv = np.std(seasonal_factors) / np.mean(seasonal_factors)
            seasonal_multiplier = 1 + (seasonal_cv * 0.5)  # Aggiustamento moderato
            safety_stock *= seasonal_multiplier

        # Calcoli derivati
        lead_time_demand = avg_demand_daily * request.lead_time_days
        reorder_point = lead_time_demand + safety_stock

        # EOQ approssimativo per maximum stock
        annual_demand = avg_demand * 12
        # Assumiamo costi standard se non forniti
        eoq_approx = np.sqrt((2 * annual_demand * 50) / (avg_demand * 0.25))  # Costi stimati
        maximum_stock = safety_stock + eoq_approx

        # Turnover ratio
        turnover_ratio = annual_demand / (safety_stock + eoq_approx / 2)

        # Service level reale raggiunto
        service_level_achieved = norm.cdf(z_score)

        return SafetyStockResponse(
            product_id=request.product_id,
            recommended_safety_stock=round(safety_stock, 1),
            average_demand=round(avg_demand, 1),
            demand_std_dev=round(demand_std, 1),
            service_level_achieved=round(service_level_achieved, 3),
            reorder_point=round(reorder_point, 1),
            maximum_stock=round(maximum_stock, 1),
            turnover_ratio=round(turnover_ratio, 1),
        )

    except Exception as e:
        logger.error(f"Errore calcolo safety stock: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Errore calcolo safety stock: {str(e)}")


@router.post("/eoq-calculation", response_model=EOQResponse)
async def calculate_economic_order_quantity(request: EOQRequest):
    """
    Calcola Economic Order Quantity (EOQ) ottimale con analisi sconti quantità.

    <h4>Formula EOQ Classica:</h4>
    <table >
        <tr><th>Parametro</th><th>Formula</th><th>Descrizione</th></tr>
        <tr><td>EOQ Base</td><td>√(2×D×S/H)</td><td>D=Domanda annuale, S=Costo ordine, H=Costo holding</td></tr>
        <tr><td>Costo Totale</td><td>(D×S/Q) + (Q×H/2)</td><td>Costo ordinazione + Costo holding</td></tr>
        <tr><td>Freq. Ordini</td><td>D/Q</td><td>Numero ordini per anno</td></tr>
        <tr><td>Tempo tra Ordini</td><td>365/Freq</td><td>Giorni tra ordini consecutivi</td></tr>
    </table>

    <h4>Esempio Richiesta:</h4>
    <pre><code>
    {
        "product_id": "SKU001",
        "annual_demand": 1200,
        "ordering_cost": 50.0,
        "holding_cost_per_unit": 8.0,
        "unit_cost": 25.0,
        "quantity_discounts": [
            {"min_qty": 500, "discount_rate": 0.05},
            {"min_qty": 1000, "discount_rate": 0.10}
        ]
    }
    </code></pre>

    <h4>Risposta di Esempio:</h4>
    <pre><code>
    {
        "product_id": "SKU001",
        "optimal_order_quantity": 387.3,
        "annual_ordering_cost": 154.9,
        "annual_holding_cost": 154.9,
        "total_annual_cost": 309.8,
        "order_frequency": 3.1,
        "time_between_orders": 117.8,
        "discount_analysis": {
            "eoq_without_discount": 387.3,
            "best_discount_qty": 1000,
            "discount_savings": 2250.0,
            "total_cost_with_discount": 28809.8
        }
    }
    </code></pre>
    """
    try:
        # Calcolo EOQ base
        D = request.annual_demand
        S = request.ordering_cost
        H = request.holding_cost_per_unit

        # EOQ formula classica
        eoq_base = np.sqrt((2 * D * S) / H)

        # Costi con EOQ base
        annual_ordering_cost = (D * S) / eoq_base
        annual_holding_cost = (eoq_base * H) / 2
        total_annual_cost = annual_ordering_cost + annual_holding_cost

        # Frequenza ordini e tempo
        order_frequency = D / eoq_base
        time_between_orders = 365 / order_frequency

        # Analisi sconti quantità se presenti
        discount_analysis = None
        optimal_quantity = eoq_base
        final_total_cost = total_annual_cost

        if request.quantity_discounts:
            best_option = {
                "quantity": eoq_base,
                "total_cost": total_annual_cost + (D * request.unit_cost),
                "discount_rate": 0.0,
                "description": "EOQ without discount",
            }

            # Analizza ogni sconto
            for discount in request.quantity_discounts:
                qty = discount["min_qty"]
                discount_rate = discount["discount_rate"]

                # Costo prodotto con sconto
                discounted_unit_cost = request.unit_cost * (1 - discount_rate)

                # Costi inventory con quantità scontata
                ordering_cost_discount = (D * S) / qty
                holding_cost_discount = (qty * H) / 2
                product_cost = D * discounted_unit_cost

                total_cost_discount = ordering_cost_discount + holding_cost_discount + product_cost

                if total_cost_discount < best_option["total_cost"]:
                    best_option = {
                        "quantity": qty,
                        "total_cost": total_cost_discount,
                        "discount_rate": discount_rate,
                        "description": f"Discount {discount_rate * 100}% at qty {qty}",
                    }

            discount_analysis = {
                "eoq_without_discount": round(eoq_base, 1),
                "best_discount_qty": best_option["quantity"],
                "best_discount_rate": best_option["discount_rate"],
                "discount_savings": round(
                    (total_annual_cost + D * request.unit_cost) - best_option["total_cost"], 2
                ),
                "total_cost_with_discount": round(best_option["total_cost"], 2),
                "recommendation": best_option["description"],
            }

            # Aggiorna quantità ottimale se sconto conviene
            if best_option["discount_rate"] > 0:
                optimal_quantity = best_option["quantity"]
                # Ricalcola costi per quantità ottimale con sconto
                final_ordering_cost = (D * S) / optimal_quantity
                final_holding_cost = (optimal_quantity * H) / 2
                final_total_cost = final_ordering_cost + final_holding_cost
                order_frequency = D / optimal_quantity
                time_between_orders = 365 / order_frequency

        return EOQResponse(
            product_id=request.product_id,
            optimal_order_quantity=round(optimal_quantity, 1),
            annual_ordering_cost=round(
                final_ordering_cost if "final_ordering_cost" in locals() else annual_ordering_cost,
                2,
            ),
            annual_holding_cost=round(
                final_holding_cost if "final_holding_cost" in locals() else annual_holding_cost, 2
            ),
            total_annual_cost=round(final_total_cost, 2),
            order_frequency=round(order_frequency, 1),
            time_between_orders=round(time_between_orders, 1),
            discount_analysis=discount_analysis,
        )

    except Exception as e:
        logger.error(f"Errore calcolo EOQ: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Errore calcolo EOQ: {str(e)}")


@router.post("/multi-echelon", response_model=MultiEchelonResponse)
async def optimize_multi_echelon_inventory(
    request: MultiEchelonRequest, services: Optional[Dict] = Depends(get_inventory_services)
):
    """
    Ottimizza inventory allocation across multiple locations con risk pooling benefits.

    <h4>Multi-Echelon Optimization Benefits:</h4>
    <table >
        <tr><th>Strategia</th><th>Benefici</th><th>Applicabilità</th></tr>
        <tr><td>Risk Pooling</td><td>Riduzione safety stock 20-40%</td><td>Domanda indipendente tra location</td></tr>
        <tr><td>Centralization</td><td>Economies of scale, minor handling</td><td>Alti volumi, costi trasporto bassi</td></tr>
        <tr><td>Postponement</td><td>Flessibilità allocazione</td><td>Prodotti configurabili</td></tr>
        <tr><td>Transshipment</td><td>Bilanciamento dinamico</td><td>Location vicine, urgenza bassa</td></tr>
    </table>

    <h4>Esempio Richiesta:</h4>
    <pre><code>
    {
        "locations": [
            {
                "location_id": "WH_NORTH",
                "location_name": "Warehouse North",
                "demand_forecast": [150, 160, 145, 170, 155, 165],
                "current_inventory": {"SKU001": 200, "SKU002": 150},
                "capacity_constraints": {"volume": 10000, "weight": 5000},
                "transportation_costs": {"WH_SOUTH": 2.5, "WH_CENTRAL": 1.8}
            }
        ],
        "products": ["SKU001", "SKU002"],
        "optimization_horizon": 6,
        "risk_pooling_enabled": true
    }
    </code></pre>

    <h4>Risposta di Esempio:</h4>
    <pre><code>
    {
        "optimization_id": "multi-ghi789",
        "total_locations": 3,
        "optimization_horizon": 6,
        "recommended_allocation": {
            "WH_NORTH": {"SKU001": 450, "SKU002": 320},
            "WH_SOUTH": {"SKU001": 380, "SKU002": 290}
        },
        "total_cost_reduction": 125000.0,
        "risk_pooling_benefits": {
            "safety_stock_reduction": 85000.0,
            "inventory_reduction_pct": 28.5
        },
        "transportation_optimization": {
            "total_transport_cost": 15000.0,
            "cost_reduction": 5000.0
        },
        "recommendations": [
            "Implement risk pooling for SKU001 - 35% safety stock reduction",
            "Consider centralization for SKU002 - low transport costs",
            "Setup transshipment between WH_NORTH and WH_SOUTH"
        ]
    }
    </code></pre>
    """
    try:
        optimization_id = f"multi-{uuid.uuid4().hex[:8]}"

        # Analizza domanda totale e variabilità per location
        total_demand = {}
        location_stats = {}

        for location in request.locations:
            location_stats[location.location_id] = {
                "total_demand": sum(location.demand_forecast),
                "avg_demand": np.mean(location.demand_forecast),
                "demand_std": np.std(location.demand_forecast),
                "cv": np.std(location.demand_forecast) / np.mean(location.demand_forecast)
                if np.mean(location.demand_forecast) > 0
                else 0,
            }

            for product in request.products:
                if product not in total_demand:
                    total_demand[product] = {"total": 0, "by_location": {}, "pooled_std": 0}

                # Assumiamo domanda uniforme per prodotto per semplicità
                product_demand = sum(location.demand_forecast) / len(request.products)
                total_demand[product]["by_location"][location.location_id] = product_demand
                total_demand[product]["total"] += product_demand

        # Calcolo risk pooling benefits
        risk_pooling_benefits = {}
        if request.risk_pooling_enabled:
            total_safety_stock_reduction = 0

            for product in request.products:
                # Safety stock individuale per location
                individual_safety_stock = 0
                for location_id in total_demand[product]["by_location"]:
                    location_demand = total_demand[product]["by_location"][location_id]
                    location_std = location_stats[location_id]["demand_std"] / len(request.products)
                    # Safety stock approssimativo (1.65 per 95% service level)
                    individual_safety_stock += 1.65 * location_std * np.sqrt(14 / 30)  # 2 settimane

                # Safety stock pooled (beneficio risk pooling)
                pooled_demand = total_demand[product]["total"]
                pooled_std = np.sqrt(
                    sum(
                        [
                            (location_stats[loc]["demand_std"] / len(request.products)) ** 2
                            for loc in location_stats
                        ]
                    )
                )
                pooled_safety_stock = 1.65 * pooled_std * np.sqrt(14 / 30)

                # Beneficio risk pooling
                safety_stock_reduction = individual_safety_stock - pooled_safety_stock
                total_safety_stock_reduction += safety_stock_reduction

            # Stima valore monetario (assumiamo costo medio €50/unit)
            avg_unit_value = 50
            risk_pooling_benefits = {
                "safety_stock_reduction": round(total_safety_stock_reduction * avg_unit_value, 0),
                "inventory_reduction_pct": round(
                    (
                        total_safety_stock_reduction
                        / sum([sum(location.demand_forecast) for location in request.locations])
                    )
                    * 100,
                    1,
                ),
            }

        # Ottimizzazione allocazione semplificata
        recommended_allocation = {}
        for location in request.locations:
            recommended_allocation[location.location_id] = {}

            location_capacity = location.capacity_constraints or {}
            available_volume = location_capacity.get("volume", float("inf"))

            for product in request.products:
                base_allocation = total_demand[product]["by_location"][location.location_id]

                # Aggiustamento per risk pooling (riduzione safety stock)
                if request.risk_pooling_enabled:
                    # Riduzione del 25% safety stock per risk pooling
                    pooling_adjustment = base_allocation * 0.75
                else:
                    pooling_adjustment = base_allocation

                # Considera vincoli capacità
                final_allocation = min(pooling_adjustment, available_volume / len(request.products))
                recommended_allocation[location.location_id][product] = round(final_allocation, 0)

        # Calcolo costi trasporto (semplificato)
        total_transport_cost = 0
        transport_savings = 0

        for location in request.locations:
            for destination, cost_per_unit in location.transportation_costs.items():
                # Stima trasferimenti basato su sbilanciamenti
                if destination in recommended_allocation:
                    estimated_transfers = 50  # Unità stimate
                    total_transport_cost += estimated_transfers * cost_per_unit

        # Assume 25% riduzione costi trasporto per ottimizzazione
        transport_savings = total_transport_cost * 0.25

        # Costo totale e risparmi
        total_cost_reduction = (
            risk_pooling_benefits.get("safety_stock_reduction", 0) + transport_savings
        )

        # Genera raccomandazioni
        recommendations = []

        if (
            request.risk_pooling_enabled
            and risk_pooling_benefits.get("inventory_reduction_pct", 0) > 15
        ):
            recommendations.append(
                f"Implement risk pooling - {risk_pooling_benefits['inventory_reduction_pct']}% inventory reduction"
            )

        # Analizza se centralizzazione conviene
        high_transport_locations = [
            loc
            for loc in request.locations
            if np.mean(list(loc.transportation_costs.values())) < 2.0
        ]
        if high_transport_locations:
            recommendations.append(
                f"Consider centralization for {len(high_transport_locations)} locations with low transport costs"
            )

        # Raccomanda transshipment per location vicine
        close_pairs = []
        for location in request.locations:
            for dest, cost in location.transportation_costs.items():
                if cost < 3.0:  # Soglia per "vicine"
                    close_pairs.append((location.location_id, dest))

        if close_pairs:
            recommendations.append(
                f"Setup transshipment between {len(close_pairs)} location pairs with low transport costs"
            )

        return MultiEchelonResponse(
            optimization_id=optimization_id,
            total_locations=len(request.locations),
            optimization_horizon=request.optimization_horizon,
            recommended_allocation=recommended_allocation,
            total_cost_reduction=round(total_cost_reduction, 0),
            risk_pooling_benefits=risk_pooling_benefits,
            transportation_optimization={
                "total_transport_cost": round(total_transport_cost, 0),
                "cost_reduction": round(transport_savings, 0),
                "optimization_opportunities": len(close_pairs),
            },
            recommendations=recommendations,
        )

    except Exception as e:
        logger.error(f"Errore ottimizzazione multi-echelon: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Errore multi-echelon: {str(e)}")


@router.post("/capacity-constraints", response_model=CapacityConstraintsResponse)
async def optimize_with_capacity_constraints(
    request: CapacityConstraintsRequest, services: Optional[Dict] = Depends(get_inventory_services)
):
    """
    Ottimizza portfolio prodotti considerando vincoli di capacità (volume/peso/valore).

    <h4>Algoritmo Knapsack Multi-Constraint:</h4>
    <table >
        <tr><th>Vincolo</th><th>Metrica</th><th>Ottimizzazione</th></tr>
        <tr><td>Volume</td><td>m³ occupati</td><td>Massimizza revenue/m³</td></tr>
        <tr><td>Peso</td><td>kg totali</td><td>Massimizza margin/kg</td></tr>
        <tr><td>Valore</td><td>€ investiti</td><td>Massimizza ROI</td></tr>
        <tr><td>Misto</td><td>Score pesato</td><td>Ottimizza funzione obiettivo combinata</td></tr>
    </table>

    <h4>Esempio Richiesta:</h4>
    <pre><code>
    {
        "products": [
            {
                "product_id": "SKU001",
                "name": "Laptop Premium",
                "volume": 0.05,
                "weight": 2.5,
                "value": 1200,
                "cost": 800,
                "margin": 400,
                "annual_demand": 500,
                "turnover": 12
            }
        ],
        "total_capacity": {
            "volume": 1000.0,
            "weight": 5000.0,
            "value": 500000.0
        },
        "priority_weights": {
            "revenue": 0.4,
            "margin": 0.3,
            "turnover": 0.3
        },
        "constraints_type": "mixed"
    }
    </code></pre>

    <h4>Risposta di Esempio:</h4>
    <pre><code>
    {
        "optimization_id": "cap-jkl012",
        "optimized_portfolio": [
            {
                "product_id": "SKU001",
                "name": "Laptop Premium",
                "recommended_quantity": 350,
                "capacity_utilization": {"volume": 17.5, "weight": 875, "value": 280000},
                "score": 8.5,
                "priority_rank": 1
            }
        ],
        "capacity_utilization": {
            "volume": 87.5,
            "weight": 65.2,
            "value": 95.0
        },
        "total_revenue": 420000.0,
        "total_profit": 140000.0,
        "efficiency_score": 92.1,
        "excluded_products": [
            {"product_id": "SKU005", "reason": "Low margin efficiency"}
        ]
    }
    </code></pre>
    """
    try:
        optimization_id = f"cap-{uuid.uuid4().hex[:8]}"

        # Calcola score di efficienza per ogni prodotto
        products_with_scores = []

        for product in request.products:
            # Calcola metriche per scoring
            revenue_per_unit = product.get("value", 0) - product.get("cost", 0)
            margin_per_unit = product.get("margin", revenue_per_unit)
            annual_revenue = revenue_per_unit * product.get("annual_demand", 0)

            # Efficienza per dimensione
            volume_efficiency = (
                annual_revenue / product.get("volume", 1) if product.get("volume", 0) > 0 else 0
            )
            weight_efficiency = (
                annual_revenue / product.get("weight", 1) if product.get("weight", 0) > 0 else 0
            )
            value_efficiency = (
                annual_revenue / product.get("value", 1) if product.get("value", 0) > 0 else 0
            )

            # Score combinato basato sui pesi
            if request.constraints_type == "volume":
                efficiency_score = volume_efficiency
            elif request.constraints_type == "weight":
                efficiency_score = weight_efficiency
            elif request.constraints_type == "value":
                efficiency_score = value_efficiency
            else:  # mixed
                # Normalizza le efficienze e applica pesi
                max_vol_eff = max(
                    [
                        p.get("annual_demand", 0)
                        * (p.get("value", 0) - p.get("cost", 0))
                        / p.get("volume", 1)
                        for p in request.products
                    ]
                )
                max_weight_eff = max(
                    [
                        p.get("annual_demand", 0)
                        * (p.get("value", 0) - p.get("cost", 0))
                        / p.get("weight", 1)
                        for p in request.products
                    ]
                )
                max_val_eff = max(
                    [
                        p.get("annual_demand", 0)
                        * (p.get("value", 0) - p.get("cost", 0))
                        / p.get("value", 1)
                        for p in request.products
                    ]
                )

                norm_vol = volume_efficiency / max_vol_eff if max_vol_eff > 0 else 0
                norm_weight = weight_efficiency / max_weight_eff if max_weight_eff > 0 else 0
                norm_val = value_efficiency / max_val_eff if max_val_eff > 0 else 0

                # Score pesato
                efficiency_score = (
                    norm_vol * request.priority_weights.get("revenue", 0.4)
                    + norm_weight * request.priority_weights.get("margin", 0.3)
                    + norm_val * request.priority_weights.get("turnover", 0.3)
                ) * 10  # Scala a 0-10

            products_with_scores.append(
                {
                    "product": product,
                    "efficiency_score": efficiency_score,
                    "annual_revenue": annual_revenue,
                    "margin_per_unit": margin_per_unit,
                }
            )

        # Ordina per efficienza (greedy algorithm per knapsack)
        products_with_scores.sort(key=lambda x: x["efficiency_score"], reverse=True)

        # Algoritmo greedy per selezione ottimale
        selected_products = []
        remaining_capacity = request.total_capacity.copy()
        excluded_products = []

        for item in products_with_scores:
            product = item["product"]
            demand = product.get("annual_demand", 0)

            # Verifica vincoli capacità
            volume_needed = product.get("volume", 0) * demand
            weight_needed = product.get("weight", 0) * demand
            value_needed = product.get("value", 0) * demand

            # Calcola quantità massima possibile con vincoli
            max_qty_volume = (
                remaining_capacity["volume"] / product.get("volume", float("inf"))
                if product.get("volume", 0) > 0
                else float("inf")
            )
            max_qty_weight = (
                remaining_capacity["weight"] / product.get("weight", float("inf"))
                if product.get("weight", 0) > 0
                else float("inf")
            )
            max_qty_value = (
                remaining_capacity["value"] / product.get("value", float("inf"))
                if product.get("value", 0) > 0
                else float("inf")
            )

            max_feasible_qty = min(demand, max_qty_volume, max_qty_weight, max_qty_value)

            if max_feasible_qty >= 1:  # Almeno 1 unità
                # Calcola utilizzo capacità
                actual_volume = product.get("volume", 0) * max_feasible_qty
                actual_weight = product.get("weight", 0) * max_feasible_qty
                actual_value = product.get("value", 0) * max_feasible_qty

                selected_products.append(
                    {
                        "product_id": product.get("product_id", ""),
                        "name": product.get("name", ""),
                        "recommended_quantity": int(max_feasible_qty),
                        "capacity_utilization": {
                            "volume": round(actual_volume, 2),
                            "weight": round(actual_weight, 2),
                            "value": round(actual_value, 2),
                        },
                        "score": round(item["efficiency_score"], 1),
                        "priority_rank": len(selected_products) + 1,
                        "annual_revenue": round(
                            item["annual_revenue"] * (max_feasible_qty / demand), 0
                        ),
                        "annual_margin": round(item["margin_per_unit"] * max_feasible_qty, 0),
                    }
                )

                # Aggiorna capacità rimanente
                remaining_capacity["volume"] -= actual_volume
                remaining_capacity["weight"] -= actual_weight
                remaining_capacity["value"] -= actual_value

            else:
                # Prodotto escluso
                reason = "Exceeds capacity constraints"
                if max_qty_volume < 1:
                    reason = "Volume constraint violation"
                elif max_qty_weight < 1:
                    reason = "Weight constraint violation"
                elif max_qty_value < 1:
                    reason = "Value constraint violation"

                excluded_products.append(
                    {"product_id": product.get("product_id", ""), "reason": reason}
                )

        # Calcola metriche aggregate
        total_revenue = sum([p["annual_revenue"] for p in selected_products])
        total_profit = sum([p["annual_margin"] for p in selected_products])

        # Calcola utilizzo capacità percentuale
        capacity_utilization = {
            "volume": round(
                (
                    (request.total_capacity["volume"] - remaining_capacity["volume"])
                    / request.total_capacity["volume"]
                )
                * 100,
                1,
            ),
            "weight": round(
                (
                    (request.total_capacity["weight"] - remaining_capacity["weight"])
                    / request.total_capacity["weight"]
                )
                * 100,
                1,
            ),
            "value": round(
                (
                    (request.total_capacity["value"] - remaining_capacity["value"])
                    / request.total_capacity["value"]
                )
                * 100,
                1,
            ),
        }

        # Score di efficienza (media pesata utilizzo)
        efficiency_score = np.mean(list(capacity_utilization.values()))

        return CapacityConstraintsResponse(
            optimization_id=optimization_id,
            optimized_portfolio=selected_products,
            capacity_utilization=capacity_utilization,
            total_revenue=round(total_revenue, 0),
            total_profit=round(total_profit, 0),
            efficiency_score=round(efficiency_score, 1),
            excluded_products=excluded_products,
        )

    except Exception as e:
        logger.error(f"Errore ottimizzazione capacity constraints: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Errore capacity constraints: {str(e)}")


@router.post("/kitting-analysis", response_model=KittingAnalysisResponse)
async def analyze_kitting_strategies(
    request: KittingAnalysisRequest, services: Optional[Dict] = Depends(get_inventory_services)
):
    """
    Analizza strategie bundle/kitting per ottimizzare Make-to-Stock vs Assemble-to-Order.

    <h4>Kitting Strategy Decision Matrix:</h4>
    <table >
        <tr><th>Scenario</th><th>Demand Pattern</th><th>Strategy</th><th>Benefits</th></tr>
        <tr><td>High Volume + Predictable</td><td>Stabile, alta correlazione</td><td>Make-to-Stock Kits</td><td>Economies of scale</td></tr>
        <tr><td>High Volume + Variable</td><td>Volatile, bassa correlazione</td><td>Assemble-to-Order</td><td>Flessibilità</td></tr>
        <tr><td>Low Volume + Predictable</td><td>Stabile, bassa frequenza</td><td>Component Stock</td><td>Riduzione inventory</td></tr>
        <tr><td>Customizable Products</td><td>Varianti multiple</td><td>Postponement</td><td>Mass customization</td></tr>
    </table>

    <h4>Esempio Richiesta:</h4>
    <pre><code>
    {
        "main_products": [
            {
                "product_id": "KIT001",
                "name": "Starter Kit Gaming",
                "components": ["MOUSE001", "PAD001", "HEADSET001"],
                "demand_forecast": [50, 45, 60, 55, 48, 52],
                "assembly_time_minutes": 15,
                "selling_price": 299.0
            }
        ],
        "component_products": [
            {
                "product_id": "MOUSE001",
                "name": "Gaming Mouse",
                "cost": 45.0,
                "demand_volatility": 0.25,
                "standalone_sales": true
            }
        ],
        "demand_correlation": {
            "MOUSE001_PAD001": 0.75
        },
        "assembly_costs": {
            "labor_per_minute": 0.5,
            "packaging_cost": 2.0
        },
        "storage_cost_difference": 1.2
    }
    </code></pre>

    <h4>Risposta di Esempio:</h4>
    <pre><code>
    {
        "analysis_id": "kit-mno345",
        "recommended_bundles": [
            {
                "bundle_id": "KIT001",
                "name": "Starter Kit Gaming",
                "strategy": "Make-to-Stock",
                "confidence": 0.82,
                "expected_margin_improvement": 18.5,
                "risk_score": 0.25
            }
        ],
        "make_to_stock_vs_assemble": {
            "KIT001": "Make-to-Stock - High correlation, stable demand"
        },
        "cost_benefit_analysis": {
            "assembly_cost_savings": 2400.0,
            "inventory_cost_increase": 800.0,
            "net_benefit": 1600.0,
            "payback_months": 3.2
        },
        "inventory_reduction": 12500.0,
        "service_level_impact": {
            "current": 0.92,
            "projected": 0.95
        },
        "implementation_priority": [
            "Phase 1: Implement KIT001 - High ROI, low risk",
            "Phase 2: Monitor component correlation changes"
        ]
    }
    </code></pre>
    """
    try:
        analysis_id = f"kit-{uuid.uuid4().hex[:8]}"

        recommended_bundles = []
        make_to_stock_decisions = {}
        total_assembly_savings = 0
        total_inventory_cost_change = 0

        for main_product in request.main_products:
            # Analisi demand pattern
            demand_data = main_product.get("demand_forecast", [])
            avg_demand = np.mean(demand_data)
            demand_cv = np.std(demand_data) / avg_demand if avg_demand > 0 else 0

            # Calcola correlazione media dei componenti
            components = main_product.get("components", [])
            avg_correlation = 0
            correlation_count = 0

            for i, comp1 in enumerate(components):
                for j, comp2 in enumerate(components[i + 1 :], i + 1):
                    corr_key = f"{comp1}_{comp2}"
                    reverse_key = f"{comp2}_{comp1}"
                    correlation = request.demand_correlation.get(
                        corr_key, request.demand_correlation.get(reverse_key, 0.5)
                    )
                    avg_correlation += correlation
                    correlation_count += 1

            if correlation_count > 0:
                avg_correlation /= correlation_count
            else:
                avg_correlation = 0.5  # Default neutrale

            # Decision matrix per kitting strategy
            volume_score = min(avg_demand / 100, 1.0)  # Normalizza a 0-1
            predictability_score = max(0, 1 - demand_cv)  # Più basso CV = più prevedibile
            correlation_score = avg_correlation

            # Score combinato per Make-to-Stock vs Assemble-to-Order
            mts_score = volume_score * 0.4 + predictability_score * 0.35 + correlation_score * 0.25

            # Decisione strategica
            if mts_score >= 0.7:
                strategy = "Make-to-Stock"
                reason = "High volume, predictable demand, strong component correlation"
                confidence = mts_score
                risk_score = 1 - mts_score
            elif mts_score >= 0.4:
                strategy = "Hybrid (Partial Pre-Assembly)"
                reason = "Medium volume/predictability - assemble popular components"
                confidence = mts_score
                risk_score = 1 - mts_score
            else:
                strategy = "Assemble-to-Order"
                reason = "Low volume/unpredictable - maintain component flexibility"
                confidence = 1 - mts_score
                risk_score = mts_score

            # Calcola benefici economici
            assembly_time = main_product.get("assembly_time_minutes", 10)
            labor_cost_per_unit = assembly_time * request.assembly_costs.get(
                "labor_per_minute", 0.5
            )
            packaging_cost = request.assembly_costs.get("packaging_cost", 2.0)

            # Stima risparmi per economies of scale
            annual_demand = sum(demand_data) * 2  # Stima annuale

            if strategy == "Make-to-Stock":
                # Risparmi assemblaggio batch vs singolo
                batch_savings_per_unit = labor_cost_per_unit * 0.3  # 30% efficienza batch
                total_batch_savings = batch_savings_per_unit * annual_demand

                # Costo inventory aggiuntivo per kit vs componenti
                additional_inventory_cost = annual_demand * request.storage_cost_difference
            else:
                total_batch_savings = 0
                additional_inventory_cost = 0

            # Margin improvement stimato
            selling_price = main_product.get("selling_price", 100)
            margin_improvement_pct = (total_batch_savings / (selling_price * annual_demand)) * 100

            total_assembly_savings += total_batch_savings
            total_inventory_cost_change += additional_inventory_cost

            recommended_bundles.append(
                {
                    "bundle_id": main_product.get("product_id", ""),
                    "name": main_product.get("name", ""),
                    "strategy": strategy,
                    "confidence": round(confidence, 2),
                    "expected_margin_improvement": round(margin_improvement_pct, 1),
                    "risk_score": round(risk_score, 2),
                    "annual_volume": int(annual_demand),
                    "demand_cv": round(demand_cv, 2),
                    "component_correlation": round(avg_correlation, 2),
                }
            )

            make_to_stock_decisions[main_product.get("product_id", "")] = f"{strategy} - {reason}"

        # Cost-benefit analysis aggregate
        net_benefit = total_assembly_savings - total_inventory_cost_change
        payback_months = (
            abs(total_inventory_cost_change) / (total_assembly_savings / 12)
            if total_assembly_savings > 0
            else float("inf")
        )

        cost_benefit_analysis = {
            "assembly_cost_savings": round(total_assembly_savings, 0),
            "inventory_cost_increase": round(total_inventory_cost_change, 0),
            "net_benefit": round(net_benefit, 0),
            "payback_months": round(min(payback_months, 99), 1),
            "roi_percentage": round((net_benefit / max(total_inventory_cost_change, 1)) * 100, 1),
        }

        # Inventory reduction estimate
        # Assumiamo che kitting riduca safety stock per economies of scale
        total_component_inventory = sum(
            [
                comp.get("cost", 50) * 100  # Stima inventory per componente
                for comp in request.component_products
            ]
        )
        inventory_reduction = total_component_inventory * 0.15  # 15% riduzione media

        # Service level impact
        service_level_impact = {
            "current": 0.92,  # Baseline stimato
            "projected": min(
                0.98, 0.92 + (avg_correlation * 0.06)
            ),  # Miglioramento per correlazione
        }

        # Implementation priority
        implementation_priority = []

        # Ordina bundle per ROI
        high_roi_bundles = [
            b
            for b in recommended_bundles
            if b["expected_margin_improvement"] > 10 and b["confidence"] > 0.6
        ]
        if high_roi_bundles:
            best_bundle = max(high_roi_bundles, key=lambda x: x["expected_margin_improvement"])
            implementation_priority.append(
                f"Phase 1: Implement {best_bundle['bundle_id']} - High ROI ({best_bundle['expected_margin_improvement']:.1f}%), low risk"
            )

        medium_bundles = [
            b for b in recommended_bundles if b["strategy"] == "Hybrid (Partial Pre-Assembly)"
        ]
        if medium_bundles:
            implementation_priority.append(
                "Phase 2: Test hybrid approach for medium-confidence bundles"
            )

        implementation_priority.append("Phase 3: Monitor component correlation changes quarterly")
        implementation_priority.append("Phase 4: Evaluate customer satisfaction with kit options")

        return KittingAnalysisResponse(
            analysis_id=analysis_id,
            recommended_bundles=recommended_bundles,
            make_to_stock_vs_assemble=make_to_stock_decisions,
            cost_benefit_analysis=cost_benefit_analysis,
            inventory_reduction=round(inventory_reduction, 0),
            service_level_impact=service_level_impact,
            implementation_priority=implementation_priority,
        )

    except Exception as e:
        logger.error(f"Errore analisi kitting: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Errore kitting analysis: {str(e)}")
