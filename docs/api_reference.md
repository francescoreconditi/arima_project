# API Reference - Inventory Management Functions

## 1. MovementClassifier

### Class: MovementClassifier

Classifica i prodotti per velocità di movimento usando analisi ABC/XYZ.

#### Metodi Principali

```python
def classify_movement_speed(product: InventoryProduct) -> MovementClassification
```

**Parametri:**
- `product`: Dati prodotto con domanda, giacenze e storico

**Ritorna:** 
- `MovementClassification` con speed, abc_class, xyz_class

**Esempio:**
```python
classifier = MovementClassifier()
classification = classifier.classify_movement_speed(product)
print(f"Speed: {classification.movement_speed}")  # FAST/MEDIUM/SLOW
```

---

## 2. SlowFastOptimizer

### Class: SlowFastOptimizer

Ottimizza strategie inventory per prodotti slow e fast moving.

#### Metodi Principali

```python
def optimize_inventory(
    product: InventoryProduct,
    params: Optional[SlowFastOptimizationParams] = None
) -> SlowFastOptimizationResult
```

**Parametri:**
- `product`: Dati prodotto da ottimizzare
- `params`: Parametri ottimizzazione (opzionale)

**Ritorna:** 
- `SlowFastOptimizationResult` con reorder point, quantity, max stock

#### Parametri di Configurazione

```python
@dataclass
class SlowFastOptimizationParams:
    service_level_fast: float = 0.98      # Service level prodotti fast
    service_level_slow: float = 0.90      # Service level prodotti slow  
    review_frequency_fast: int = 7        # Giorni revisione fast
    review_frequency_slow: int = 30       # Giorni revisione slow
    holding_cost_rate: float = 0.25       # Tasso costo mantenimento
    ordering_cost: float = 50.0           # Costo fisso ordinazione
    max_slow_months: int = 6              # Max mesi stock slow
```

**Esempio:**
```python
optimizer = SlowFastOptimizer()
params = SlowFastOptimizationParams(
    service_level_fast=0.99,
    review_frequency_fast=5
)
result = optimizer.optimize_inventory(product, params)
```

---

## 3. PerishableManager

### Class: PerishableManager

Gestisce prodotti deperibili con logica FEFO (First Expired First Out).

#### Metodi Principali

```python
def optimize_fefo_quantity(
    product: PerishableProduct,
    demand_forecast: List[float],
    current_inventory_ages: List[int],
    params: Optional[FEFOOptimizationParams] = None
) -> FEFOOptimizationResult
```

**Parametri:**
- `product`: Prodotto deperibile con shelf life
- `demand_forecast`: Previsioni domanda prossimi periodi  
- `current_inventory_ages`: Età lotti esistenti (giorni)
- `params`: Parametri ottimizzazione FEFO

**Ritorna:**
- `FEFOOptimizationResult` con quantità ottimale, spreco previsto, costi

#### Modello Dati Prodotto Deperibile

```python
@dataclass  
class PerishableProduct:
    sku: str
    shelf_life_days: int                    # Giorni shelf life
    perishability: PerishabilityType        # SHORT/MEDIUM/LONG_LIFE
    expiry_buffer_days: int = 30            # Buffer giorni scadenza
    waste_cost_per_unit: float = 0.0       # Costo smaltimento
    holding_cost_rate: float = 0.25         # Tasso mantenimento
    cold_chain_required: bool = False       # Catena del freddo
```

**Esempio:**
```python
perishable = PerishableProduct(
    sku="FARMACO001",
    shelf_life_days=180,
    perishability=PerishabilityType.MEDIUM_LIFE,
    waste_cost_per_unit=15.0
)

manager = PerishableManager()
result = manager.optimize_fefo_quantity(
    product=perishable,
    demand_forecast=[25, 30, 28],
    current_inventory_ages=[10, 25, 45]
)
```

---

## 4. MultiEchelonOptimizer

### Class: MultiEchelonOptimizer

Ottimizza inventory in reti multi-livello con risk pooling.

#### Metodi Principali

```python
def optimize_network(
    network: List[EchelonNode],
    connections: List[EchelonConnection],
    products: List[InventoryProduct],
    config: Optional[MultiEchelonConfig] = None
) -> MultiEchelonOptimizationResult
```

**Parametri:**
- `network`: Lista nodi rete (DC, warehouse, store)
- `connections`: Collegamenti tra nodi con costi trasporto
- `products`: Prodotti da ottimizzare  
- `config`: Configurazione ottimizzazione

#### Configurazione Risk Pooling

```python
def analyze_risk_pooling(
    demand_data: Dict[str, List[float]],
    pooling_config: RiskPoolingConfig
) -> RiskPoolingAnalysis
```

**Esempio:**
```python
# Definire nodi rete
nodes = [
    EchelonNode("DC_CENTRALE", EchelonType.DISTRIBUTION_CENTER, 10000, 0.20),
    EchelonNode("WAREHOUSE_NORD", EchelonType.REGIONAL_WAREHOUSE, 2000, 0.25)
]

# Analizzare risk pooling
optimizer = MultiEchelonOptimizer()
analysis = optimizer.analyze_risk_pooling(
    demand_data={"NORD": [100, 120, 95], "SUD": [80, 90, 110]},
    pooling_config=RiskPoolingConfig(pooling_factor=0.7)
)
```

---

## 5. CapacityConstrainedOptimizer

### Class: CapacityConstrainedOptimizer

Ottimizza allocazione inventory con vincoli di capacità.

#### Metodi Principali

```python
def optimize_with_constraints(
    products: List[ProductConstraintData],
    constraints: Dict[CapacityType, CapacityConstraint],
    optimization_objective: str = "profit_maximization"
) -> CapacityOptimizationResult
```

**Parametri:**
- `products`: Lista prodotti con dati vincoli
- `constraints`: Vincoli capacità (volume, peso, budget, pallet)
- `optimization_objective`: Obiettivo ottimizzazione

#### Tipi Vincoli Supportati

```python
class CapacityType(Enum):
    VOLUME = "volume"                    # m³
    WEIGHT = "weight"                    # kg  
    BUDGET = "budget"                    # €
    PALLET_POSITIONS = "pallet_positions"  # posizioni
```

#### Analisi Sensitivity

```python
def capacity_sensitivity_analysis(
    base_solution: CapacityOptimizationResult,
    constraints: Dict[CapacityType, CapacityConstraint]
) -> Dict[CapacityType, SensitivityResult]
```

**Esempio:**
```python
constraints = {
    CapacityType.VOLUME: CapacityConstraint(
        constraint_type=CapacityType.VOLUME,
        total_capacity=5000.0,
        current_usage=3200.0
    ),
    CapacityType.BUDGET: CapacityConstraint(
        constraint_type=CapacityType.BUDGET, 
        total_capacity=500000.0,
        current_usage=380000.0
    )
}

optimizer = CapacityConstrainedOptimizer()
result = optimizer.optimize_with_constraints(products, constraints)
```

---

## 6. KittingOptimizer

### Class: KittingOptimizer

Ottimizza strategie bundle vs componenti singoli.

#### Metodi Principali

```python
def analyze_kit_strategy(kit_product: KitProduct) -> KitStrategyAnalysis
```

**Parametri:**
- `kit_product`: Dati kit con componenti e domanda

**Ritorna:**
- `KitStrategyAnalysis` con confronto Make-to-Stock vs Assemble-to-Order

#### Ottimizzazione Stock Componenti

```python
def optimize_component_inventory(
    kit_product: KitProduct,
    params: KitOptimizationParams,
    optimization_horizon_days: int = 90
) -> ComponentOptimizationResult
```

#### Modello Dati Kit

```python
@dataclass
class KitProduct:
    kit_sku: str
    components: List[KitComponent]           # Lista componenti
    kit_demand: float                       # Domanda kit/periodo
    kit_price: float                        # Prezzo vendita kit
    assembly_cost: float                    # Costo assemblaggio
    assembly_time_days: int = 1             # Tempo assemblaggio
    seasonal_pattern: bool = False          # Seasonalità
    customer_customization: bool = False    # Personalizzazione
```

**Esempio:**
```python
components = [
    KitComponent("COMP_A", quantity_per_kit=2, unit_cost=15.0),
    KitComponent("COMP_B", quantity_per_kit=1, unit_cost=25.0)
]

kit = KitProduct(
    kit_sku="KIT_001",
    components=components,
    kit_demand=80,
    kit_price=75.0,
    assembly_cost=5.0
)

optimizer = KittingOptimizer()
analysis = optimizer.analyze_kit_strategy(kit)
```

---

## 7. Modelli Dati Base

### InventoryProduct

```python
@dataclass
class InventoryProduct:
    sku: str                                # Codice prodotto
    annual_demand: float                    # Domanda annuale
    average_inventory: float                # Giacenza media
    demand_history: List[float]             # Storico domanda
    unit_cost: float                        # Costo unitario
    lead_time_days: int = 14                # Lead time giorni
    service_level: float = 0.95             # Service level target
    holding_cost_rate: float = 0.25         # Tasso mantenimento
    ordering_cost: float = 50.0             # Costo ordinazione
    supplier_id: Optional[str] = None       # ID fornitore
```

### MovementClassification

```python
@dataclass
class MovementClassification:
    movement_speed: MovementSpeed           # SLOW/MEDIUM/FAST
    abc_class: ABCClassification           # A/B/C (valore)  
    xyz_class: XYZClassification           # X/Y/Z (variabilità)
    turnover_ratio: float                  # Rapporto rotazione
    coefficient_of_variation: float         # CV domanda
    annual_value: float                    # Valore annuale
    recommended_strategy: str              # Strategia consigliata
```

---

## 8. Risultati e Metriche

### SlowFastOptimizationResult

```python
@dataclass
class SlowFastOptimizationResult:
    reorder_point: float                   # Punto riordino
    order_quantity: float                  # Quantità riordino  
    max_stock: float                       # Stock massimo
    safety_stock: float                    # Stock sicurezza
    review_frequency: int                  # Giorni revisione
    expected_stockout_rate: float          # Tasso stockout previsto
    total_annual_cost: float               # Costo totale annuale
    holding_cost: float                    # Costo mantenimento
    ordering_cost: float                   # Costo ordinazioni
```

### FEFOOptimizationResult

```python
@dataclass  
class FEFOOptimizationResult:
    optimal_quantity: float                # Quantità ottimale
    expected_waste: float                  # Spreco previsto
    total_cost: float                      # Costo totale
    holding_cost: float                    # Costo mantenimento
    waste_cost: float                      # Costo spreco
    service_level_achieved: float          # Service level raggiunto
    days_coverage: float                   # Giorni copertura
    expiry_risk_score: float              # Rischio scadenza
```

---

## 9. Esempi Completi per Sviluppatori

### Esempio 1: Pipeline Completa Classificazione

```python
from arima_forecaster.inventory.balance_optimizer import *

def classify_inventory_portfolio(products: List[InventoryProduct]):
    """Classifica portfolio completo prodotti"""
    classifier = MovementClassifier()
    results = []
    
    for product in products:
        # Classificazione
        classification = classifier.classify_movement_speed(product)
        
        # Ottimizzazione basata su classificazione
        if classification.movement_speed == MovementSpeed.SLOW:
            optimizer = SlowFastOptimizer()
            optimization = optimizer.optimize_inventory(
                product, 
                SlowFastOptimizationParams(review_frequency_slow=30)
            )
        else:
            optimizer = SlowFastOptimizer()  
            optimization = optimizer.optimize_inventory(
                product,
                SlowFastOptimizationParams(review_frequency_fast=7)
            )
        
        results.append({
            'sku': product.sku,
            'classification': classification,
            'optimization': optimization
        })
    
    return results
```

### Esempio 2: Gestione Multi-Constraint

```python
def optimize_warehouse_allocation(
    products: List[InventoryProduct],
    volume_limit: float,
    budget_limit: float
) -> Dict:
    """Ottimizza allocazione con vincoli multipli"""
    
    # Converti a formato constraint
    constraint_products = [
        ProductConstraintData(
            sku=p.sku,
            volume_per_unit=estimate_volume(p),
            cost_per_unit=p.unit_cost,
            priority_score=calculate_priority(p)
        ) for p in products
    ]
    
    constraints = {
        CapacityType.VOLUME: CapacityConstraint(
            constraint_type=CapacityType.VOLUME,
            total_capacity=volume_limit,
            current_usage=0.0
        ),
        CapacityType.BUDGET: CapacityConstraint(
            constraint_type=CapacityType.BUDGET, 
            total_capacity=budget_limit,
            current_usage=0.0
        )
    }
    
    # Ottimizza
    optimizer = CapacityConstrainedOptimizer()
    result = optimizer.optimize_with_constraints(
        constraint_products, 
        constraints,
        "profit_maximization"
    )
    
    # Analisi sensitivity
    sensitivity = optimizer.capacity_sensitivity_analysis(result, constraints)
    
    return {
        'selected_products': result.selected_products,
        'capacity_utilization': result.capacity_utilization,
        'total_profit': result.total_profit,
        'shadow_prices': {k: v.shadow_price for k, v in sensitivity.items()}
    }
```

### Esempio 3: Integrazione FEFO + Multi-Echelon

```python
def optimize_pharma_supply_chain(
    perishable_products: List[PerishableProduct],
    network_nodes: List[EchelonNode]
) -> Dict:
    """Ottimizza supply chain farmaceutica con deperibilità"""
    
    results = {}
    
    for product in perishable_products:
        # FEFO optimization per ogni nodo
        node_optimizations = {}
        
        for node in network_nodes:
            # Adatta parametri per tipo nodo
            if node.node_type == EchelonType.RETAIL_STORE:
                buffer_days = 7   # Negozi buffer ridotto
            else:
                buffer_days = 30  # Warehouse buffer maggiore
            
            # Ottimizzazione FEFO specifica nodo
            perishable_manager = PerishableManager()
            fefo_result = perishable_manager.optimize_fefo_quantity(
                product=product,
                demand_forecast=get_node_demand_forecast(node, product),
                current_inventory_ages=get_node_inventory_ages(node, product),
                params=FEFOOptimizationParams(expiry_buffer_days=buffer_days)
            )
            
            node_optimizations[node.node_id] = fefo_result
        
        # Multi-echelon coordination
        echelon_optimizer = MultiEchelonOptimizer()
        network_result = echelon_optimizer.optimize_network(
            network_nodes,
            get_network_connections(),
            [convert_perishable_to_inventory(product)]
        )
        
        results[product.sku] = {
            'node_fefo_results': node_optimizations,
            'network_optimization': network_result,
            'total_waste_minimized': sum(r.expected_waste for r in node_optimizations.values())
        }
    
    return results
```

---

## 10. Error Handling e Best Practices

### Gestione Errori

```python
from arima_forecaster.utils.exceptions import (
    InventoryOptimizationError,
    CapacityConstraintViolationError,
    InsufficientDataError
)

try:
    result = optimizer.optimize_inventory(product)
except InsufficientDataError as e:
    logger.warning(f"Dati insufficienti per {product.sku}: {e}")
    # Usa parametri default
    result = fallback_optimization(product)
except CapacityConstraintViolationError as e:
    logger.error(f"Vincoli violati: {e}")
    # Rilassa vincoli o cambia obiettivo
    result = optimize_with_relaxed_constraints(product)
```

### Performance Optimization

```python
# Batch processing per grandi dataset
def optimize_large_inventory(products: List[InventoryProduct], batch_size: int = 1000):
    results = []
    for i in range(0, len(products), batch_size):
        batch = products[i:i+batch_size]
        batch_results = process_batch_parallel(batch)
        results.extend(batch_results)
    return results

# Caching per evitare ricalcoli
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_classification(sku: str, annual_demand: float, avg_inventory: float):
    # Implementazione cached
    pass
```

### Validazione Dati

```python
def validate_inventory_product(product: InventoryProduct) -> bool:
    """Valida consistenza dati prodotto"""
    if product.annual_demand <= 0:
        raise ValueError(f"Domanda annuale non valida per {product.sku}")
    
    if product.average_inventory < 0:
        raise ValueError(f"Giacenza media negativa per {product.sku}")
    
    if len(product.demand_history) < 3:
        logger.warning(f"Storico insufficiente per {product.sku}")
    
    return True
```

---

**Versione API**: 2.0.0  
**Compatibilità**: Python 3.8+  
**Aggiornato**: Settembre 2024