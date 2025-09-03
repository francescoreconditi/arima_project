# Documentazione Tecnica - Funzionalità Avanzate di Gestione Magazzino

## 1. Introduzione

Questo documento fornisce una guida tecnica completa per le sei nuove funzionalità avanzate implementate nel sistema di gestione magazzino ARIMA:

1. **Slow/Fast Moving** - Classificazione e ottimizzazione prodotti per velocità di rotazione
2. **Perishable/FEFO** - Gestione prodotti deperibili con logica First Expired First Out
3. **🆕 Minimum Shelf Life (MSL) Management** - Allocazione multi-canale GDO con requisiti MSL
4. **Multi-Echelon** - Ottimizzazione inventario multi-livello con risk pooling
5. **Capacity Constraints** - Gestione vincoli di capacità (volume, peso, budget, posti pallet)
6. **Kitting/Bundle** - Ottimizzazione strategie bundle vs componenti singoli

## 2. Architettura del Sistema

### 2.1 Gerarchia delle Classi

```python
# Classi principali implementate
MovementClassifier     # Classificazione ABC/XYZ e velocità movimento
SlowFastOptimizer     # Strategie ottimizzazione slow/fast moving
PerishableManager     # Gestione prodotti deperibili con FEFO
MinimumShelfLifeManager # 🆕 Allocazione MSL multi-canale GDO
MultiEchelonOptimizer # Ottimizzazione multi-livello magazzini
CapacityConstrainedOptimizer # Gestione vincoli capacità
KittingOptimizer      # Ottimizzazione bundle e kit
```

### 2.2 Enumerazioni Supportate

```python
MovementSpeed        # SLOW, MEDIUM, FAST
ABCClassification   # A, B, C (per valore)
XYZClassification   # X, Y, Z (per variabilità)
PerishabilityType   # NON_PERISHABLE, SHORT_LIFE, MEDIUM_LIFE, LONG_LIFE
CapacityType        # VOLUME, WEIGHT, BUDGET, PALLET_POSITIONS
```

## 3. Slow/Fast Moving - Gestione Velocità Rotazione

### 3.1 Classificazione Automatica

La classificazione si basa su turnover ratio e coefficiente di variazione:

```python
from arima_forecaster.inventory.balance_optimizer import MovementClassifier

# Inizializzazione classificatore
classifier = MovementClassifier()

# Dati prodotto richiesti
product_data = InventoryProduct(
    sku="PROD001",
    annual_demand=1200,
    average_inventory=100,
    demand_history=[98, 105, 87, 120, 95],  # Ultimi 5 periodi
    unit_cost=25.0
)

# Classificazione automatica
classification = classifier.classify_movement_speed(product_data)
print(f"Velocità: {classification.movement_speed}")  # FAST
print(f"ABC Class: {classification.abc_class}")      # A
print(f"XYZ Class: {classification.xyz_class}")      # X
```

### 3.2 Metriche di Classificazione

#### Turnover Ratio
```
Turnover = Domanda Annuale / Giacenza Media
- FAST: turnover >= 6 (rotazione ogni 2 mesi)
- MEDIUM: 2 <= turnover < 6 (rotazione 2-6 mesi)  
- SLOW: turnover < 2 (rotazione oltre 6 mesi)
```

#### Classificazione ABC (per Valore)
```
Valore Annuale = Domanda × Costo Unitario
- Classe A: Top 80% del valore cumulativo
- Classe B: 15% successivo del valore
- Classe C: Ultimo 5% del valore
```

#### Classificazione XYZ (per Variabilità)
```
CV = Deviazione Standard / Media della Domanda
- Classe X: CV <= 0.5 (variabilità bassa)
- Classe Y: 0.5 < CV <= 1.0 (variabilità media)
- Classe Z: CV > 1.0 (variabilità alta)
```

### 3.3 Strategie di Ottimizzazione

```python
from arima_forecaster.inventory.balance_optimizer import SlowFastOptimizer

optimizer = SlowFastOptimizer()

# Parametri ottimizzazione
params = SlowFastOptimizationParams(
    service_level_fast=0.98,    # 98% per prodotti fast
    service_level_slow=0.90,    # 90% per prodotti slow
    review_frequency_fast=7,    # Revisione settimanale fast
    review_frequency_slow=30,   # Revisione mensile slow
    max_slow_months=6          # Max 6 mesi stock slow
)

# Ottimizzazione
result = optimizer.optimize_inventory(product_data, params)

print(f"Punto riordino: {result.reorder_point}")
print(f"Quantità riordino: {result.order_quantity}")
print(f"Stock massimo: {result.max_stock}")
print(f"Frequenza controllo: {result.review_frequency} giorni")
```

### 3.4 Formule Matematiche Implementate

#### Economic Order Quantity (EOQ) Modificata
```
EOQ_slow = sqrt(2 × D × K / h) × adjustment_factor
- D = domanda annuale
- K = costo fisso ordinazione
- h = costo mantenimento per unità/anno
- adjustment_factor = 0.8 per slow, 1.2 per fast
```

#### Safety Stock Dinamico
```
SS = z × σ × sqrt(L + R)
- z = z-score per service level
- σ = deviazione standard domanda
- L = lead time
- R = review period (più lungo per slow)
```

#### Reorder Point Adattivo
```
ROP = (D × L) + SS + buffer_slow
- buffer_slow = extra buffer per prodotti slow moving
```

## 4. Perishable/FEFO - Gestione Prodotti Deperibili

### 4.1 Configurazione Base

```python
from arima_forecaster.inventory.balance_optimizer import PerishableManager

# Dati prodotto deperibile
perishable_product = PerishableProduct(
    sku="MED001",
    shelf_life_days=365,        # 1 anno shelf life
    perishability=PerishabilityType.MEDIUM_LIFE,
    expiry_buffer_days=30,      # Buffer sicurezza 30 giorni
    waste_cost_per_unit=15.0,   # Costo smaltimento
    holding_cost_rate=0.25      # 25% annuo
)

manager = PerishableManager()
```

### 4.2 Calcolo Quantità Ottimale FEFO

```python
# Parametri FEFO
fefo_params = FEFOOptimizationParams(
    waste_penalty_factor=2.0,    # Penalità spreco 2x
    expiry_buffer_days=30,       # Buffer scadenza
    max_age_ratio=0.8           # Max 80% shelf life utilizzabile
)

# Ottimizzazione FEFO
fefo_result = manager.optimize_fefo_quantity(
    product=perishable_product,
    demand_forecast=[25, 30, 28, 32, 27],  # Prossimi 5 periodi
    current_inventory_ages=[10, 25, 45, 60],  # Età lotti esistenti
    params=fefo_params
)

print(f"Quantità ottimale: {fefo_result.optimal_quantity}")
print(f"Spreco previsto: {fefo_result.expected_waste}")
print(f"Costo totale: {fefo_result.total_cost}")
```

### 4.3 Algoritmo FEFO Implementato

```python
def fefo_allocation_algorithm(demand, inventory_lots):
    """
    Algoritmo First Expired First Out
    1. Ordina lotti per data scadenza crescente
    2. Alloca domanda ai lotti più vecchi non scaduti
    3. Calcola spreco per lotti scaduti
    """
    allocated = []
    waste = 0
    
    for lot in sorted(inventory_lots, key=lambda x: x.expiry_date):
        if lot.is_expired():
            waste += lot.quantity
        else:
            allocation = min(demand, lot.quantity)
            allocated.append((lot, allocation))
            demand -= allocation
            if demand <= 0:
                break
    
    return allocated, waste
```

### 4.4 Classificazione Prodotti per Deperibilità

```python
def classify_perishability(shelf_life_days):
    """Classificazione automatica deperibilità"""
    if shelf_life_days <= 7:
        return PerishabilityType.SHORT_LIFE    # Settimanale
    elif shelf_life_days <= 90:
        return PerishabilityType.MEDIUM_LIFE   # Trimestrale  
    elif shelf_life_days <= 365:
        return PerishabilityType.LONG_LIFE     # Annuale
    else:
        return PerishabilityType.NON_PERISHABLE
```

## 5. 🆕 Minimum Shelf Life (MSL) Management

### 5.1 Panoramica MSL

Il sistema MSL (Minimum Shelf Life) gestisce l'allocazione ottimale di prodotti deperibili ai diversi canali di vendita in base ai requisiti di vita residua minima.

**Problema risolto**: "GDO A richiede prodotti con minimo 60 giorni residui, GDO B richiede minimo 90 giorni"

### 5.2 Tipi di Canale Supportati

```python
from arima_forecaster.inventory.balance_optimizer import TipoCanale

# Canali predefiniti con MSL
TipoCanale.GDO_PREMIUM      # MSL: 90 giorni
TipoCanale.GDO_STANDARD     # MSL: 60 giorni  
TipoCanale.RETAIL_TRADIZIONALE # MSL: 45 giorni
TipoCanale.ONLINE_DIRETTO   # MSL: 30 giorni
TipoCanale.OUTLET_SCONTI    # MSL: 15 giorni
TipoCanale.B2B_WHOLESALE    # MSL: 120 giorni
```

### 5.3 Configurazione Requisiti MSL Personalizzati

```python
from arima_forecaster.inventory.balance_optimizer import (
    MinimumShelfLifeManager, 
    RequisitoMSL, 
    TipoCanale
)

# Inizializza manager MSL
msl_manager = MinimumShelfLifeManager()

# Configura requisiti personalizzati per prodotto specifico
requisito_custom = RequisitoMSL(
    canale=TipoCanale.GDO_PREMIUM,
    prodotto_codice="YOGURT_BIO",
    msl_giorni=100,  # Override default 90 → 100 giorni
    priorita=1,
    attivo=True,
    note="Esselunga richiede 100gg per biologici"
)

msl_manager.aggiungi_requisito_msl(requisito_custom)
```

### 5.4 Algoritmo di Ottimizzazione FEFO + MSL

```python
# Setup lotti e domanda
lotti_disponibili = [...]  # Lista LottoPerishable
domanda_canali = {
    "gdo_premium": 500,
    "gdo_standard": 800,
    "retail": 300,
    "online": 200
}
prezzo_canali = {
    "gdo_premium": 4.50,   # Prezzo più alto
    "gdo_standard": 3.80,
    "retail": 3.50,
    "online": 3.90
}

# Esegui ottimizzazione
risultati = msl_manager.ottimizza_allocazione_lotti(
    lotti_disponibili=lotti_disponibili,
    domanda_canali=domanda_canali,
    prezzo_canali=prezzo_canali
)

# Analisi risultati
for canale_id, allocazioni in risultati.items():
    for alloc in allocazioni:
        print(f"Lotto {alloc.lotto_id} → {canale_id}")
        print(f"  Quantità: {alloc.quantita_allocata}")
        print(f"  Shelf life residua: {alloc.giorni_shelf_life_residui}")
        print(f"  Margine MSL: {alloc.margine_msl} giorni")
        print(f"  Urgenza: {alloc.urgenza}")
        print(f"  Valore: €{alloc.valore_allocato}")
```

### 5.5 Report e Metriche MSL

```python
# Genera report dettagliato
report = msl_manager.genera_report_allocazioni(risultati)

print("=== REPORT MSL ===")
print(f"Valore totale: €{report['summary']['valore_totale_allocato']:,.2f}")
print(f"Canali serviti: {report['summary']['numero_canali_serviti']}")
print(f"Efficienza: {report['efficienza_allocazione']}%")

# Distribuzione urgenze
for urgenza, count in report['distribuzione_urgenze'].items():
    print(f"{urgenza.capitalize()}: {count} allocazioni")

# Suggerimenti azioni
azioni = msl_manager.suggerisci_azioni_msl(risultati)
for azione in azioni:
    print(f"[{azione['priorita']}] {azione['descrizione']}")
    print(f"→ {azione['azione']}")
```

### 5.6 Integrazione con PerishableManager

```python
# Integrazione FEFO + MSL
perishable_mgr = PerishableManager()
perishable_mgr.abilita_msl_integration(msl_manager)

# Ottimizzazione combinata
risultato_integrato = perishable_mgr.ottimizza_fefo_con_msl(
    lotti_esistenti=lotti,
    domanda_canali=domanda_canali,
    prezzo_canali=prezzo_canali
)

print("=== FEFO + MSL INTEGRATO ===")
print(f"Tipo: {risultato_integrato['tipo_ottimizzazione']}")
print(f"Valore totale: €{risultato_integrato['summary']['valore_totale_allocato']}")
print(f"Aderenza FEFO: {risultato_integrato['metriche_fefo']['aderenza_fefo_percentuale']}%")
print(f"Canali premium serviti: {risultato_integrato['vantaggi_msl']['canali_premium_serviti']}")
```

### 5.7 Analisi What-If Scenari MSL

```python
# Test diversi scenari MSL
scenari = {
    "conservativo": lambda msl: msl + 20,  # +20 giorni MSL
    "standard": lambda msl: msl,           # MSL default
    "aggressivo": lambda msl: max(7, msl - 15)  # -15 giorni MSL
}

for nome, modifica_msl in scenari.items():
    msl_temp = MinimumShelfLifeManager()
    
    # Applica modifica MSL
    for canale in TipoCanale:
        requisito = RequisitoMSL(
            canale=canale,
            prodotto_codice="TEST",
            msl_giorni=modifica_msl(canale.value[2]),
            priorita=1
        )
        msl_temp.aggiungi_requisito_msl(requisito)
    
    # Ottimizza con nuovo scenario
    risultati_scenario = msl_temp.ottimizza_allocazione_lotti(lotti, domanda_canali, prezzo_canali)
    valore_scenario = sum(sum(a.valore_allocato for a in alloc) for alloc in risultati_scenario.values())
    
    print(f"Scenario {nome}: €{valore_scenario:,.2f}")
```

## 6. Multi-Echelon - Ottimizzazione Multi-Livello

### 5.1 Configurazione Rete Multi-Livello

```python
from arima_forecaster.inventory.balance_optimizer import MultiEchelonOptimizer

# Definizione nodi rete
nodes = [
    EchelonNode(
        node_id="DC_CENTRALE",
        node_type=EchelonType.DISTRIBUTION_CENTER,
        capacity=10000,
        holding_cost_rate=0.20,
        service_time_days=1
    ),
    EchelonNode(
        node_id="MAGAZZINO_NORD", 
        node_type=EchelonType.REGIONAL_WAREHOUSE,
        capacity=2000,
        holding_cost_rate=0.25,
        service_time_days=2
    ),
    EchelonNode(
        node_id="PUNTO_VENDITA_001",
        node_type=EchelonType.RETAIL_STORE,
        capacity=500,
        holding_cost_rate=0.30,
        service_time_days=0
    )
]

# Definizione collegamenti
connections = [
    EchelonConnection("DC_CENTRALE", "MAGAZZINO_NORD", transport_time=1, transport_cost=5.0),
    EchelonConnection("MAGAZZINO_NORD", "PUNTO_VENDITA_001", transport_time=1, transport_cost=2.0)
]

optimizer = MultiEchelonOptimizer()
```

### 5.2 Risk Pooling e Centralizzazione

```python
# Configurazione risk pooling
pooling_config = RiskPoolingConfig(
    pooling_factor=0.7,         # 30% riduzione variabilità
    economies_of_scale=0.9,     # 10% riduzione costi
    transport_cost_factor=1.2,  # 20% aumento costi trasporto
    service_level_target=0.95
)

# Analisi risk pooling
pooling_analysis = optimizer.analyze_risk_pooling(
    demand_data=multi_location_demand,
    current_config=decentralized_config,
    pooling_config=pooling_config
)

print(f"Riduzione stock totale: {pooling_analysis.total_inventory_reduction}%")
print(f"Risparmi annuali: €{pooling_analysis.annual_savings:,.0f}")
```

### 5.3 Ottimizzazione Stock Allocation

```python
def optimize_echelon_allocation(network, total_demand):
    """
    Ottimizzazione allocazione stock multi-livello
    Minimizza: Holding Cost + Stockout Cost + Transport Cost
    """
    
    # Modello ottimizzazione lineare
    allocation = {}
    
    for product in products:
        # Calcola domanda aggregata per livello
        level_demand = aggregate_demand_by_level(network, product)
        
        # Ottimizza allocazione per minimizzare costi totali
        for node in network.nodes:
            safety_stock = calculate_echelon_safety_stock(
                node, product, network.service_level
            )
            cycle_stock = calculate_echelon_cycle_stock(
                node, product, network.review_frequency  
            )
            
            allocation[node.id] = safety_stock + cycle_stock
    
    return allocation
```

### 5.4 Metriche Multi-Echelon

#### Total Landed Cost
```
TLC = Product Cost + Transport Cost + Holding Cost + Stockout Cost
```

#### Inventory Investment per Echelon
```
Investment_Level_i = Σ(Stock_Node_j × Unit_Cost_j) per j ∈ Level_i
```

#### Service Level Aggregato
```
Service_Level_Network = Π(Service_Level_Node_i) per i ∈ Network
```

## 6. Capacity Constraints - Gestione Vincoli Capacità

### 6.1 Tipi di Vincoli Supportati

```python
from arima_forecaster.inventory.balance_optimizer import CapacityConstrainedOptimizer

# Definizione vincoli
capacity_constraints = {
    CapacityType.VOLUME: CapacityConstraint(
        constraint_type=CapacityType.VOLUME,
        total_capacity=5000.0,    # m³ totali
        current_usage=3200.0,     # m³ utilizzati
        unit_of_measure="m3"
    ),
    CapacityType.WEIGHT: CapacityConstraint(
        constraint_type=CapacityType.WEIGHT,
        total_capacity=15000.0,   # kg totali  
        current_usage=12000.0,    # kg utilizzati
        unit_of_measure="kg"
    ),
    CapacityType.BUDGET: CapacityConstraint(
        constraint_type=CapacityType.BUDGET,
        total_capacity=500000.0,  # € budget
        current_usage=380000.0,   # € utilizzati
        unit_of_measure="EUR"
    )
}
```

### 6.2 Ottimizzazione con Vincoli

```python
optimizer = CapacityConstrainedOptimizer()

# Prodotti da ottimizzare
products_with_constraints = [
    ProductConstraintData(
        sku="PROD001",
        volume_per_unit=0.5,      # m³
        weight_per_unit=2.0,      # kg
        cost_per_unit=25.0,       # €
        priority_score=0.9        # Alta priorità
    ),
    # ... altri prodotti
]

# Ottimizzazione
result = optimizer.optimize_with_constraints(
    products=products_with_constraints,
    constraints=capacity_constraints,
    optimization_objective="profit_maximization"
)

print(f"Prodotti selezionati: {len(result.selected_products)}")
print(f"Utilizzo capacità volume: {result.capacity_utilization['volume']:.1%}")
print(f"Profitto totale: €{result.total_profit:,.0f}")
```

### 6.3 Algoritmi di Ottimizzazione

#### Knapsack Problem per Budget Constraint
```python
def solve_budget_knapsack(products, budget_limit):
    """
    Risolve problema zaino per vincolo budget
    Massimizza profitto soggetto a vincolo budget
    """
    n = len(products)
    dp = [[0 for _ in range(int(budget_limit) + 1)] for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(1, int(budget_limit) + 1):
            cost = int(products[i-1].cost_per_unit)
            profit = products[i-1].profit_per_unit
            
            if cost <= w:
                dp[i][w] = max(
                    dp[i-1][w],  # Non include prodotto
                    dp[i-1][w-cost] + profit  # Include prodotto
                )
            else:
                dp[i][w] = dp[i-1][w]
    
    return backtrack_solution(dp, products, budget_limit)
```

#### Multi-Constraint Optimization
```python
def optimize_multi_constraint(products, constraints):
    """
    Ottimizzazione multi-vincolo con programmazione lineare
    """
    from scipy.optimize import linprog
    
    # Coefficienti funzione obiettivo (profitti)
    c = [-p.profit_per_unit for p in products]  # Minimizza -profitto
    
    # Matrice vincoli
    A_ub = []
    b_ub = []
    
    for constraint_type, constraint in constraints.items():
        if constraint_type == CapacityType.VOLUME:
            A_ub.append([p.volume_per_unit for p in products])
        elif constraint_type == CapacityType.WEIGHT:
            A_ub.append([p.weight_per_unit for p in products])
        elif constraint_type == CapacityType.BUDGET:
            A_ub.append([p.cost_per_unit for p in products])
        
        available_capacity = constraint.total_capacity - constraint.current_usage
        b_ub.append(available_capacity)
    
    # Risoluzione
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * len(products))
    return result
```

### 6.4 Analisi Sensitivity

```python
def capacity_sensitivity_analysis(base_solution, constraints):
    """
    Analisi sensibilità vincoli di capacità
    """
    sensitivity_results = {}
    
    for constraint_type, constraint in constraints.items():
        # Test aumento capacità +10%
        increased_capacity = constraint.total_capacity * 1.1
        new_constraint = constraint.copy()
        new_constraint.total_capacity = increased_capacity
        
        new_solution = optimize_with_constraints(products, {constraint_type: new_constraint})
        
        profit_increase = new_solution.total_profit - base_solution.total_profit
        capacity_increase = increased_capacity - constraint.total_capacity
        
        sensitivity_results[constraint_type] = {
            'shadow_price': profit_increase / capacity_increase,
            'profit_increase': profit_increase,
            'capacity_value': capacity_increase
        }
    
    return sensitivity_results
```

## 7. Kitting/Bundle - Ottimizzazione Bundle

### 7.1 Configurazione Bundle

```python
from arima_forecaster.inventory.balance_optimizer import KittingOptimizer

# Definizione componenti kit
components = [
    KitComponent(
        component_sku="COMP_A",
        quantity_per_kit=2,
        unit_cost=15.0,
        standalone_demand=100,
        lead_time_days=7
    ),
    KitComponent(
        component_sku="COMP_B", 
        quantity_per_kit=1,
        unit_cost=25.0,
        standalone_demand=50,
        lead_time_days=10
    )
]

# Definizione kit
kit_product = KitProduct(
    kit_sku="KIT_001",
    components=components,
    kit_demand=80,          # Domanda kit/mese
    kit_price=75.0,         # Prezzo vendita kit
    assembly_cost=5.0,      # Costo assemblaggio
    assembly_time_days=2    # Tempo assemblaggio
)
```

### 7.2 Analisi Make-to-Stock vs Assemble-to-Order

```python
optimizer = KittingOptimizer()

# Analisi strategie
strategy_analysis = optimizer.analyze_kit_strategy(kit_product)

print("=== CONFRONTO STRATEGIE ===")
print(f"Make-to-Stock Kit:")
print(f"  - Costo holding: €{strategy_analysis.mts_holding_cost:.0f}")
print(f"  - Service level: {strategy_analysis.mts_service_level:.1%}")
print(f"  - Lead time: {strategy_analysis.mts_lead_time} giorni")

print(f"Assemble-to-Order:")  
print(f"  - Costo holding: €{strategy_analysis.ato_holding_cost:.0f}")
print(f"  - Service level: {strategy_analysis.ato_service_level:.1%}")
print(f"  - Lead time: {strategy_analysis.ato_lead_time} giorni")

print(f"Strategia consigliata: {strategy_analysis.recommended_strategy}")
```

### 7.3 Ottimizzazione Stock Componenti

```python
# Parametri ottimizzazione
kit_params = KitOptimizationParams(
    service_level_target=0.95,
    holding_cost_rate=0.25,
    shortage_cost_multiplier=3.0,
    assembly_capacity_per_day=50
)

# Ottimizzazione
optimization_result = optimizer.optimize_component_inventory(
    kit_product=kit_product,
    params=kit_params,
    optimization_horizon_days=90
)

for component in optimization_result.component_plans:
    print(f"Componente {component.sku}:")
    print(f"  - Stock sicurezza: {component.safety_stock}")  
    print(f"  - Punto riordino: {component.reorder_point}")
    print(f"  - Q.tà riordino: {component.order_quantity}")
```

### 7.4 Formule Matematiche Kit

#### Component Safety Stock per Kit
```
SS_component = z × σ_total × sqrt(LT + ATO)
dove σ_total = sqrt(σ²_standalone + (qty_per_kit × σ_kit)²)
```

#### Kit Service Level Aggregato
```
SL_kit = Π(SL_component_i) per tutti i componenti i del kit
```

#### Make vs Buy Decision
```
Cost_MTS = H × I_avg + K × (D/Q)
Cost_ATO = H × Σ(I_component_i) + A × D_kit
dove A = assembly cost per kit
```

## 8. Integrazione e Best Practices

### 8.1 Pipeline di Integrazione

```python
def integrated_inventory_optimization():
    """Pipeline completa ottimizzazione inventario"""
    
    # Step 1: Classificazione movimento
    classifier = MovementClassifier()
    movement_class = classifier.classify_movement_speed(product)
    
    # Step 2: Verifica deperibilità  
    if product.shelf_life_days:
        perishable_manager = PerishableManager()
        fefo_strategy = perishable_manager.optimize_fefo_quantity(product)
    
    # Step 3: Analisi vincoli capacità
    if warehouse_constraints:
        capacity_optimizer = CapacityConstrainedOptimizer()
        capacity_result = capacity_optimizer.optimize_with_constraints(products, constraints)
    
    # Step 4: Ottimizzazione multi-echelon se applicabile
    if multi_location_network:
        echelon_optimizer = MultiEchelonOptimizer()
        echelon_allocation = echelon_optimizer.optimize_network(network)
    
    # Step 5: Analisi kitting se componenti
    if product.is_kit or product.is_component:
        kit_optimizer = KittingOptimizer()
        kit_strategy = kit_optimizer.analyze_kit_strategy(product)
    
    return integrated_recommendation
```

### 8.2 Performance Monitoring

```python
# KPI da monitorare per ogni funzionalità
monitoring_kpis = {
    'slow_fast': ['turnover_ratio', 'stock_coverage_days', 'carrying_cost'],
    'perishable': ['waste_percentage', 'days_to_expiry', 'fefo_compliance'],
    'multi_echelon': ['total_network_stock', 'service_level_by_node', 'transport_cost'],
    'capacity': ['capacity_utilization', 'constraint_violations', 'opportunity_cost'],
    'kitting': ['kit_availability', 'component_stockout_freq', 'assembly_efficiency']
}
```

### 8.3 Configurazione Parametri Produzione

```python
# File configurazione production-ready
production_config = {
    "slow_fast": {
        "turnover_thresholds": {"fast": 6.0, "slow": 2.0},
        "service_levels": {"fast": 0.98, "slow": 0.90},
        "review_frequencies": {"fast": 7, "slow": 30}
    },
    "perishable": {
        "waste_penalty_factor": 2.0,
        "expiry_buffer_days": 30,  
        "max_age_ratio": 0.8
    },
    "multi_echelon": {
        "risk_pooling_factor": 0.7,
        "transport_cost_weight": 0.3,
        "service_level_network": 0.95
    },
    "capacity": {
        "utilization_target": 0.85,
        "safety_margin": 0.10,
        "optimization_objective": "profit_maximization"
    },
    "kitting": {
        "ato_threshold_ratio": 0.6,
        "assembly_capacity_buffer": 0.20,
        "component_safety_multiplier": 1.2
    }
}
```

## 9. Casi d'Uso Avanzati

### 9.1 Scenario Farmaceutico
```python
# Prodotto farmaceutico con vincoli speciali
pharma_product = PerishableProduct(
    sku="FARMACO_A",
    shelf_life_days=180,       # 6 mesi
    cold_chain_required=True,  # Catena del freddo
    regulatory_buffer_days=14, # Buffer normativo
    batch_tracking=True        # Tracciabilità lotto
)

# Ottimizzazione specifica farmaceutica
pharma_strategy = integrated_pharma_optimization(pharma_product)
```

### 9.2 Scenario Automotive 
```python
# Componente automotive multi-echelon
auto_component = MultiEchelonProduct(
    sku="COMP_AUTO_001",
    demand_variability="high",     # Alta variabilità
    critical_for_production=True,  # Componente critico
    supplier_lead_time=45,         # 45 giorni lead time
    transport_mode="sea_freight"   # Trasporto marittimo
)

# Strategia supply chain automotive
auto_strategy = automotive_supply_chain_optimization(auto_component)
```

### 9.3 Scenario E-commerce
```python
# Bundle e-commerce stagionale
ecommerce_bundle = KitProduct(
    kit_sku="BUNDLE_NATALE",
    seasonal_pattern=True,      # Stagionalità forte
    promotional_periods=["Q4"], # Promozioni Q4  
    customer_customization=True, # Personalizzazione
    dropshipping_components=["COMP_C"]  # Alcuni componenti dropship
)

# Ottimizzazione e-commerce
ecommerce_strategy = ecommerce_inventory_optimization(ecommerce_bundle)
```

## 10. Troubleshooting e FAQ

### 10.1 Problemi Comuni

**Q: Le previsioni SARIMA restituiscono NaN**
```python
# Soluzione: Implementare fallback robusto
try:
    sarima_forecast = sarima_model.predict(30)
    if pd.isna(sarima_forecast).any():
        raise ValueError("NaN predictions")
except Exception:
    # Fallback ad ARIMA semplice
    arima_model = ARIMAForecaster(order=(1,1,1))
    forecast = arima_model.predict(30)
```

**Q: Errori memoria con dataset grandi**
```python
# Soluzione: Processamento batch
def process_large_inventory(products, batch_size=1000):
    results = []
    for i in range(0, len(products), batch_size):
        batch = products[i:i+batch_size]
        batch_results = optimize_batch(batch)
        results.extend(batch_results)
    return results
```

**Q: Vincoli capacità non rispettati**
```python
# Soluzione: Verifica e debug vincoli
def debug_capacity_constraints(result, constraints):
    for constraint_type, constraint in constraints.items():
        usage = calculate_actual_usage(result.selected_products, constraint_type)
        if usage > constraint.total_capacity:
            logger.warning(f"Constraint {constraint_type} violated: {usage} > {constraint.total_capacity}")
```

### 10.2 Performance Tuning

```python
# Configurazione per performance
performance_config = {
    "enable_parallel_processing": True,
    "max_workers": 4,
    "cache_intermediate_results": True,
    "batch_size": 1000,
    "memory_limit_gb": 8
}
```

---

**Versione**: 2.0.0  
**Data**: Settembre 2024  
**Maintainer**: Claude Code AI Assistant