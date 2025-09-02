# Quick Start Guide - Funzionalità Avanzate Inventory Management

## 1. Setup Veloce

```bash
# Installa dipendenze se non già fatto
uv sync --all-extras

# Verifica installazione
cd examples/
uv run python advanced_features_demo.py
```

## 2. Slow/Fast Moving - 5 Minuti

### Scenario: Classificare prodotti per velocità rotazione

```python
from arima_forecaster.inventory.balance_optimizer import (
    MovementClassifier, SlowFastOptimizer, InventoryProduct
)

# 1. Crea i tuoi dati prodotto
product = InventoryProduct(
    sku="PROD001",
    annual_demand=1200,          # Domanda anno
    average_inventory=100,       # Giacenza media
    demand_history=[95, 105, 87, 120, 98],  # Ultimi 5 periodi
    unit_cost=25.0              # Costo unitario
)

# 2. Classifica automaticamente
classifier = MovementClassifier()
classification = classifier.classify_movement_speed(product)

print(f"Velocità: {classification.movement_speed}")  # FAST/MEDIUM/SLOW
print(f"ABC Class: {classification.abc_class}")      # A/B/C (per valore)  
print(f"XYZ Class: {classification.xyz_class}")      # X/Y/Z (per variabilità)

# 3. Ottimizza strategia
optimizer = SlowFastOptimizer()
result = optimizer.optimize_inventory(product)

print(f"Punto riordino: {result.reorder_point}")
print(f"Quantità riordino: {result.order_quantity}")
```

**Risultato**: Strategia ottimale per Fast = revisione settimanale, Slow = revisione mensile

---

## 3. Perishable/FEFO - 5 Minuti

### Scenario: Gestire prodotti con scadenza

```python
from arima_forecaster.inventory.balance_optimizer import (
    PerishableManager, PerishableProduct, PerishabilityType
)

# 1. Definisci prodotto deperibile
perishable = PerishableProduct(
    sku="FARMACO001",
    shelf_life_days=180,         # 6 mesi shelf life
    perishability=PerishabilityType.MEDIUM_LIFE,
    expiry_buffer_days=30,       # Buffer sicurezza 30 giorni
    waste_cost_per_unit=15.0     # Costo smaltimento €15
)

# 2. Ottimizza con logica FEFO
manager = PerishableManager()
result = manager.optimize_fefo_quantity(
    product=perishable,
    demand_forecast=[25, 30, 28, 32, 27],  # Prossimi 5 mesi
    current_inventory_ages=[10, 25, 45, 60]  # Età lotti esistenti (giorni)
)

print(f"Quantità ottimale: {result.optimal_quantity}")
print(f"Spreco previsto: {result.expected_waste}")
print(f"Costo totale: {result.total_cost}")
```

**Risultato**: Minimizza sprechi usando First Expired First Out con buffer sicurezza

---

## 4. Multi-Echelon - 5 Minuti  

### Scenario: Ottimizzare rete multi-magazzino

```python
from arima_forecaster.inventory.balance_optimizer import (
    MultiEchelonOptimizer, EchelonNode, EchelonType
)

# 1. Definisci rete magazzini
nodes = [
    EchelonNode(
        node_id="DC_CENTRALE",
        node_type=EchelonType.DISTRIBUTION_CENTER,
        capacity=10000,
        holding_cost_rate=0.20     # 20% costo mantenimento
    ),
    EchelonNode(
        node_id="MAGAZZINO_NORD",
        node_type=EchelonType.REGIONAL_WAREHOUSE, 
        capacity=2000,
        holding_cost_rate=0.25     # 25% costo mantenimento
    )
]

# 2. Analizza risk pooling
optimizer = MultiEchelonOptimizer()
analysis = optimizer.analyze_risk_pooling(
    demand_data={"NORD": [100, 120, 95], "SUD": [80, 90, 110]},
    pooling_factor=0.7           # 30% riduzione variabilità
)

print(f"Riduzione stock totale: {analysis.total_inventory_reduction}%")
print(f"Risparmi annuali: €{analysis.annual_savings:,.0f}")
```

**Risultato**: Risk pooling riduce stock totale 15-30% mantenendo service level

---

## 5. Capacity Constraints - 5 Minuti

### Scenario: Gestire vincoli di capacità

```python
from arima_forecaster.inventory.balance_optimizer import (
    CapacityConstrainedOptimizer, CapacityConstraint, CapacityType
)

# 1. Definisci vincoli magazzino
constraints = {
    CapacityType.VOLUME: CapacityConstraint(
        constraint_type=CapacityType.VOLUME,
        total_capacity=5000.0,    # m³ totali
        current_usage=3200.0      # m³ utilizzati
    ),
    CapacityType.BUDGET: CapacityConstraint(
        constraint_type=CapacityType.BUDGET,
        total_capacity=500000.0,  # € budget
        current_usage=380000.0    # € utilizzati
    )
}

# 2. Ottimizza allocazione
optimizer = CapacityConstrainedOptimizer()
result = optimizer.optimize_with_constraints(
    products=products_list,      # Lista prodotti
    constraints=constraints,
    optimization_objective="profit_maximization"
)

print(f"Prodotti selezionati: {len(result.selected_products)}")
print(f"Utilizzo volume: {result.capacity_utilization['volume']:.1%}")
print(f"Profitto totale: €{result.total_profit:,.0f}")
```

**Risultato**: Massimizza profitto rispettando vincoli volume e budget

---

## 6. Kitting/Bundle - 5 Minuti

### Scenario: Ottimizzare strategia kit vs componenti

```python
from arima_forecaster.inventory.balance_optimizer import (
    KittingOptimizer, KitProduct, KitComponent
)

# 1. Definisci kit e componenti
components = [
    KitComponent("COMP_A", quantity_per_kit=2, unit_cost=15.0),
    KitComponent("COMP_B", quantity_per_kit=1, unit_cost=25.0)
]

kit = KitProduct(
    kit_sku="KIT_001",
    components=components,
    kit_demand=80,              # Domanda kit/mese
    kit_price=75.0,             # Prezzo vendita
    assembly_cost=5.0           # Costo assemblaggio
)

# 2. Analizza strategia ottimale
optimizer = KittingOptimizer()
analysis = optimizer.analyze_kit_strategy(kit)

print("=== CONFRONTO STRATEGIE ===")
print(f"Make-to-Stock: Costo €{analysis.mts_holding_cost:.0f}")
print(f"Assemble-to-Order: Costo €{analysis.ato_holding_cost:.0f}")
print(f"Strategia consigliata: {analysis.recommended_strategy}")
```

**Risultato**: Sceglie Make-to-Stock per alta domanda, Assemble-to-Order per personalizzazione

---

## 7. Demo Completo Integrato - 2 Minuti

```python
# Script demo completo con tutti i 5 tipi
uv run python examples/advanced_features_demo.py
```

Esegue automaticamente:
- Classificazione 6 prodotti diversi
- Ottimizzazione slow/fast moving
- Gestione 2 prodotti deperibili 
- Analisi multi-echelon 3 nodi
- Vincoli capacità volume/budget
- Strategia bundle per 1 kit

**Output**: Report completo con raccomandazioni per ogni categoria

---

## 8. Casi d'Uso Comuni

### 🏥 Farmaceutico
```python
# Prodotti con scadenza rigida + controllo batch
perishable_pharma = PerishableProduct(
    shelf_life_days=90,          # 3 mesi
    regulatory_buffer_days=14,   # Buffer normativo
    cold_chain_required=True     # Catena del freddo
)
```

### 🚗 Automotive  
```python
# Componenti critici multi-location
auto_network = MultiEchelonOptimizer()
# Analisi per evitare fermo produzione
```

### 🛒 E-commerce
```python  
# Bundle stagionali con personalizzazione
seasonal_kit = KitProduct(
    seasonal_pattern=True,
    promotional_periods=["Q4"],
    customer_customization=True
)
```

### 🏭 Manifatturiero
```python
# Vincoli capacità warehouse + budget
constraints = {
    CapacityType.WEIGHT: 15000,    # kg max
    CapacityType.PALLET_POSITIONS: 500,  # posti pallet
    CapacityType.BUDGET: 200000    # € budget
}
```

---

## 9. Best Practices

### ⚡ Performance
- Usa batch processing per >1000 prodotti
- Cache classificazioni per evitare ricalcolo
- Parallel processing: `n_jobs=-1`

### 📊 Monitoraggio KPI
```python
# KPI essenziali da tracciare
kpis = {
    'turnover_ratio': 'target > 4x',
    'waste_percentage': 'target < 5%', 
    'capacity_utilization': 'target 85%',
    'service_level': 'target > 95%'
}
```

### 🔧 Troubleshooting
- **NaN predictions**: Usa fallback ARIMA(1,1,1)
- **Memoria insufficiente**: Riduci batch_size
- **Vincoli violati**: Verifica capacità disponibili

---

## 10. Prossimi Passi

1. **Test sul tuo dataset**: Sostituisci dati demo con dati reali
2. **Personalizza parametri**: Adatta soglie alle tue esigenze business  
3. **Integra con ERP**: Usa CSV output per import automatico
4. **Monitora performance**: Traccia KPI per ottimizzazione continua
5. **Scala gradualmente**: Inizia con pochi prodotti, espandi

### Documentazione Completa
- `docs/advanced_inventory_features.md`: Guida tecnica dettagliata
- `docs/slow_fast_moving_theory.md`: Teoria e casi d'uso
- `examples/`: Script demo per ogni funzionalità

### Support
- Issues: Utilizza GitHub Issues per segnalazioni
- Performance: Vedi sezione troubleshooting documentazione tecnica

---

**Tempo totale**: 30 minuti per implementare tutte e 5 le funzionalità avanzate!

**Versione**: 2.0.0  
**Aggiornato**: Settembre 2024