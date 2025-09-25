"""
Modulo Inventory Management per ARIMA Forecaster
Ottimizzazione scorte e bilanciamento overstock/stockout
"""

from .balance_optimizer import (
    # Classi principali
    SafetyStockCalculator,
    TotalCostAnalyzer,
    InventoryAlertSystem,
    InventoryKPIDashboard,
    AdaptiveForecastEngine,
    MinimumShelfLifeManager,
    # NEW v0.5.0: Dynamic Pricing + VMI
    SafeDynamicPricingManager,
    ControlledVMIPilot,
    # Modelli dati esistenti
    CostiGiacenza,
    AnalisiRischio,
    AlertLevel,
    LivelloServizio,
    TipoCanale,
    RequisitoMSL,
    AllocationResult,
    # NEW v0.5.0: Config e Results
    DynamicPricingConfig,
    PricingSuggestion,
    VMIConfig,
    VMIEvaluationResult,
)

__all__ = [
    # Core classes
    "SafetyStockCalculator",
    "TotalCostAnalyzer",
    "InventoryAlertSystem",
    "InventoryKPIDashboard",
    "AdaptiveForecastEngine",
    "MinimumShelfLifeManager",
    # NEW v0.5.0: Advanced features
    "SafeDynamicPricingManager",
    "ControlledVMIPilot",
    # Data models
    "CostiGiacenza",
    "AnalisiRischio",
    "AlertLevel",
    "LivelloServizio",
    "TipoCanale",
    "RequisitoMSL",
    "AllocationResult",
    # NEW v0.5.0: Config models
    "DynamicPricingConfig",
    "PricingSuggestion",
    "VMIConfig",
    "VMIEvaluationResult",
]
